"""
Helper module to generate datasets in parallel. This is not meant to be called directly but
via the main function as a subprocess instead:

    python -m utils.generate_dataset --generation 3 ...

These are modules, not scripts: they are launched as `python -m utils.<name>` from the repo
root, because they import sibling helpers with `from utils.X import Y` and running the file
directly (`python utils/<name>.py`) puts utils/ on sys.path instead of the root, so `utils` is
then not a package at all.

One process handles one shard of the instruction set on a single GPU, selected by the inherited
CUDA_VISIBLE_DEVICES, and writes its shard of the new dataset to disk for the orchestrator to
merge.

Two engines produce the responses:

  vllm          the whole shard is handed to one continuous batching engine call. This is the
                default and is roughly an order of magnitude faster than the transformers path,
                because a batched generate() runs every sequence of a batch until the *longest*
                one finishes, so a batch of 150 pays the maximum length 150 times. vLLM instead
                retires a sequence as soon as it emits EOS and immediately admits the next one
  transformers  the original unsloth generate() loop, kept as a fallback for machines without
                vLLM and as a reference implementation to check the engines against

Both engines are driven with the same explicitly pinned sampling parameters, see --temperature.

Everything except the constants and the function definitions runs under an
`if __name__ == "__main__"` guard, which is load bearing rather than cosmetic. vLLM does not run
its engine in this process: it *spawns* one, and a spawned child reconstructs its parent by
re-importing the parent's __main__ module under the name __mp_main__, with sys.argv preserved so
argparse re-runs without complaint. Unguarded module level work would therefore execute a second
time inside the vLLM worker — re-parsing the arguments, re-importing torch, re-reading the shard
from disk, and finally building a second LLM() while the first one is still bootstrapping, which
multiprocessing refuses with

    RuntimeError: An attempt has been made to start a new process before the current process has
    finished its bootstrapping phase.

The generate functions read their configuration from the module globals the guarded block binds.
That works because the guard sits at module scope, so the names it assigns are ordinary module
attributes, looked up when the functions run rather than when they are defined — and the
functions are only ever called from inside the guard.

Args:
    block_size (int): The block size to use for training.
    specifier_name (str): The model specifier to use for training.
    dataset_batch_size (int): The dataset batch size to use for training.
    generation (int): The current generation.
    shard_id (int): The current shard id.

Returns:
    None
"""
import os
import argparse
import importlib.util
import shutil

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
SYSTEM_PROMPT: str = "You are a helpful assistant for code completion."
# EOS is suppressed for this many tokens, so every response is at least this long
MIN_NEW_TOKENS: int = 128


def format_prompts(instructions: list, tokenizer) -> list:
    """
    Chat templates the instructions for generation.

    Args:
        instructions (list): the raw instruction strings
        tokenizer: the tokenizer whose chat template is applied

    Returns:
        list: the templated prompts, ready to be tokenized
    """
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ],
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        for instruction in instructions
    ]


def generate_vllm(instructions: list, model_dir: str) -> list:
    """
    Generates one response per instruction with vLLM.

    The whole shard goes into a single generate() call: vLLM's scheduler keeps the GPU saturated
    by admitting a new sequence whenever a running one finishes, so there is no batch to pad to a
    common length and no barrier at the end of a batch.

    Args:
        instructions (list): the raw instruction strings of this shard
        model_dir (str): the merged fp16 checkpoint to sample from

    Returns:
        list: the responses, in the order of the instructions
    """
    from vllm import LLM, SamplingParams, TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts = format_prompts(instructions, tokenizer)

    # the context has to hold the longest prompt plus a full response. Sizing it off the actual
    # prompts instead of using a fixed number keeps the KV cache from reserving context that
    # this shard cannot use — that memory becomes concurrent sequences instead
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    prompt_lengths = [len(ids) for ids in prompt_ids]
    # the model's own context, not the tokenizer's model_max_length — the latter is 131072 for
    # Qwen2.5 and describes the tokenizer rather than the positions the model was trained for
    model_context = AutoConfig.from_pretrained(model_dir).max_position_embeddings
    max_model_len = min(max(prompt_lengths) + block_size, model_context)
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}vLLM context{TColors.ENDC}: {max_model_len} "
        f"(longest prompt {max(prompt_lengths)} + {block_size} new tokens)"
    )

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        # every prompt shares the same system prompt and chat template preamble, so the prefix
        # of every sequence is identical and only has to be prefilled once
        enable_prefix_caching=True,
        disable_log_stats=True,
        # the transformers path seeds nothing, but vLLM's scheduler makes the sampling order
        # depend on the batching, so an explicit seed is what makes a shard reproducible. It
        # varies with both the shard and the generation so no two workers sample in lockstep
        seed=1337 + 100 * generation + shard_id,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        # pinned to 3.0, matching the transformers path and generate_dataset_extrapolation.py.
        # This value is a deliberate experimental choice with a known cost: the responses are
        # scored by an *unpenalized* forward pass in calculate_perplexity.py, so the penalty makes
        # the measured perplexity partly a property of the sampling distortion rather than of the
        # model, and because it divides the logit of every token already in the context its
        # severity grows with the response length — which grows with every generation, so the
        # distortion is not a constant offset across the collapse trend. Keep it identical in
        # every generation path or the histograms stop being comparable
        repetition_penalty=3.0,
        min_tokens=MIN_NEW_TOKENS,
        max_tokens=block_size,
        # vLLM returns only the continuation, so unlike the transformers path there is no prompt
        # to strip. The special tokens are dropped here so the stored response is plain text
        skip_special_tokens=True,
    )

    # a prompt longer than the context would abort the whole call. This cannot trigger while
    # max_model_len is prompt driven above, only when it was capped by the model's context.
    #
    # This used to be SamplingParams(truncate_prompt_tokens=...), which vLLM 0.26 removed — it now
    # raises `TypeError: Unexpected keyword argument 'truncate_prompt_tokens'`, and on the
    # generation path the parameter has no successor. The documented replacement,
    # llm.generate(tokenization_kwargs={"truncation": True, "max_length": n}), is *not* equivalent:
    # truncate_prompt_tokens kept the last n tokens, while tokenizer.encode truncates from the
    # right, and truncation_side is an attribute of the tokenizer rather than a call kwarg, so
    # passing it there is silently ignored. Right truncating a chat prompt cuts off the trailing
    # `<|im_start|>assistant\n`, i.e. exactly the generation prompt the model needs to answer at
    # all. So the tail is kept explicitly here, on the ids that were tokenized above for the
    # length measurement — which also means the engine is handed the same token ids this function
    # measured rather than retokenizing the strings
    prompt_limit = max(1, max_model_len - MIN_NEW_TOKENS)
    engine_prompts = [
        TokensPrompt(prompt_token_ids=ids[-prompt_limit:]) for ids in prompt_ids
    ]

    # outputs come back in the order of the prompts
    outputs = llm.generate(engine_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def generate_transformers(instructions: list, model_dir: str) -> list:
    """
    Generates one response per instruction with unsloth's patched transformers generate().

    Fallback path. The prompts are processed in length sorted batches, because a batched
    generate() runs until the longest sequence of the batch is done and pads the rest — grouping
    similar lengths together keeps that waste down. The original order is restored before
    returning.

    Args:
        instructions (list): the raw instruction strings of this shard
        model_dir (str): the checkpoint to sample from

    Returns:
        list: the responses, in the order of the instructions
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=block_size,
        dtype=None,
        # a 0.5B model needs ~1GB in bf16 on a 48GB card, so 4bit only adds a dequantization
        # kernel to every forward pass without buying any headroom
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)
    # left padding is required for batched generation and is what for_inference sets, but it sets
    # it on the model's internal tokenizer copy. Setting it here as well guarantees that the
    # prompt occupies a constant prefix of every row of the batch, which the slicing below needs
    tokenizer.padding_side = "left"

    prompts = format_prompts(instructions, tokenizer)
    prompt_lengths = [
        len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
    ]
    order = sorted(range(len(prompts)), key=lambda index: prompt_lengths[index])

    responses = [None] * len(prompts)
    for start in tqdm(
        range(0, len(order), dataset_batch_size),
        total=(len(order) + dataset_batch_size - 1) // dataset_batch_size,
    ):
        batch_indices = order[start : start + dataset_batch_size]
        inputs = tokenizer(
            [prompts[index] for index in batch_indices],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        # do_sample/num_beams/repetition_penalty are set explicitly rather than inherited from
        # the model's generation_config: collapse is driven by resampling from the model's own
        # distribution, so the decoding has to be plain multinomial sampling. Beam search
        # (num_beams > 1) would optimize for likelihood and systematically narrow the output
        # distribution, which suppresses exactly the effect this pipeline measures.
        # repetition_penalty is pinned to 3.0, matching the vLLM path above. It is set here rather
        # than inherited so the two engines sample from the same distribution, and its cost is
        # accepted knowingly: calculate_perplexity.py scores these responses with an unpenalized
        # forward pass, so the penalty makes the measured perplexity partly a property of the
        # sampling distortion rather than of the model, and since it divides the logit of every
        # token already in the context, its severity grows with the response length — which grows
        # with every generation, so the distortion aliases onto the collapse trend instead of
        # being a constant offset
        with torch.no_grad():
            generated_answers = model.generate(
                **inputs,
                do_sample=True,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=3.0,
                min_new_tokens=MIN_NEW_TOKENS,
                max_new_tokens=block_size,
                use_cache=True,
            )

        # the prompt is dropped by token count instead of by splitting the decoded string on the
        # chat template markers, since skip_special_tokens removes those markers. The batch is
        # left padded, so the prompt is the same number of tokens in every row
        prompt_length = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            generated_answers[:, prompt_length:],
            skip_special_tokens=True,
        )
        for index, answer in zip(batch_indices, decoded):
            # skip_special_tokens already dropped the trailing <|im_end|> and the padding, so
            # only the surrounding whitespace is left to strip
            responses[index] = answer.strip()

    return responses


# everything below this line is skipped when vLLM's spawned engine process re-imports this module
# as __mp_main__, see the module docstring
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Generation")
    parser.add_argument(
        "--block_size",
        "-b",
        type=int,
        default=512,
        help="specifies the block size to use for training",
    )
    parser.add_argument(
        "--specifier_name",
        "-s",
        type=str,
        default="Qwen2.5-Coder-0.5B-Instruct",
        help="specifies the model specifier to use for training",
    )
    parser.add_argument(
        "--dataset_batch_size",
        "-dbs",
        type=int,
        default=100,
        help="prompts per generate() call. Only the transformers engine uses this, vLLM schedules "
        "the whole shard itself and does not need a batch size",
    )
    parser.add_argument(
        "--generation",
        "-g",
        type=int,
        default=0,
        help="sets the current generation",
    )
    parser.add_argument(
        "--shard_id",
        "-si",
        type=int,
        default=0,
        help="sets the current shard id",
    )
    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="inference engine for the generation. 'auto' uses vLLM if it is installed and falls "
        "back to transformers otherwise (default: auto)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        "-gmu",
        type=float,
        default=0.90,
        help="fraction of the GPU vLLM may use. Each worker owns its GPU exclusively, so this can "
        "be high. Everything not taken by the weights becomes KV cache, i.e. concurrent sequences "
        "(default: 0.90)",
    )
    parser.add_argument(
        "--enforce_eager",
        "-ee",
        action="store_true",
        help="disable vLLM's CUDA graphs. Capturing them costs ~30-60s of startup per worker but "
        "removes most of the per step kernel launch overhead, which dominates decoding for a model "
        "this small. Only worth setting for very small shards",
    )
    parser.add_argument(
        "--temperature",
        "-tp",
        type=float,
        default=0.7,
        help="sampling temperature. Pinned rather than inherited from the model's "
        "generation_config, because the engines disagree on the default: transformers reads "
        "generation_config.json while vLLM does not necessarily, so an unpinned value would make "
        "the two engines sample from different distributions (default: 0.7, Qwen2.5's own value)",
    )
    parser.add_argument(
        "--top_p",
        "-tpp",
        type=float,
        default=0.8,
        help="nucleus sampling cutoff, pinned for the same reason as --temperature. NOTE that a "
        "cutoff below 1.0 truncates the model's own distribution, which is itself a collapse "
        "mechanism — it is what utils.extrapolation's data-space surrogate models (default: 0.8, "
        "Qwen2.5's own value)",
    )
    parser.add_argument(
        "--top_k",
        "-tpk",
        type=int,
        default=20,
        help="top-k sampling cutoff, pinned for the same reason as --temperature. -1 disables it "
        "(default: 20, Qwen2.5's own value)",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="",
        help="path to save the generated datasets and models (default: current directory)",
    )
    args = parser.parse_args()

    # arguments
    block_size = args.block_size
    specifier_name = args.specifier_name
    dataset_batch_size = args.dataset_batch_size
    generation = args.generation
    shard_id = args.shard_id
    engine = args.engine
    gpu_memory_utilization = args.gpu_memory_utilization
    enforce_eager = args.enforce_eager
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    path = args.path

    # resolve the engine before importing anything heavy. unsloth has to be imported before
    # torch/transformers to patch them, so it is only imported in the branch that actually uses
    # it. The availability of vLLM is probed with find_spec rather than a try/except around
    # `import vllm`, because a failing vllm import would already have pulled in torch and the
    # unsloth import below would then come too late to patch it
    if engine == "auto":
        engine = "vllm" if importlib.util.find_spec("vllm") is not None else "transformers"

    if engine == "vllm":
        # vLLM runs its engine core in a separate process and picks the start method itself, in
        # vllm/utils/system_utils.py::_maybe_force_spawn. It forces `spawn` and warns whenever it
        # finds CUDA already initialized here, because forking a process that holds a CUDA context
        # is undefined behaviour. That is not something this module does — the vLLM path never
        # touches the GPU before LLM() — it is vLLM's own platform probe: CudaPlatform calls
        # torch.cuda.get_device_properties()/current_device(), which trigger torch's lazy CUDA
        # init, so the check is looking at a context vLLM created moments earlier.
        #
        # Declaring spawn up front hits the early return at the top of _maybe_force_spawn, which
        # skips both the probe and the warning. The point is not the warning though: it makes the
        # start method a property of this file rather than of whether something happened to touch
        # CUDA first, and spawn is the method the __main__ guard above is written for. setdefault
        # so an explicit value in the environment still wins
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # vLLM JIT compiles flashinfer's sampling kernels at engine startup, and flashinfer locates
        # the CUDA toolkit itself, in flashinfer/jit/cpp_ext.py::get_cuda_path: CUDA_HOME, else
        # CUDA_PATH, else the parent of the parent of `which nvcc`. That last fallback does not
        # resolve symlinks, so on a machine whose nvcc is reached through /usr/local/bin/nvcc ->
        # /usr/local/cuda/bin/nvcc it yields /usr/local instead of /usr/local/cuda. The kernels are
        # then compiled with -isystem /usr/local/include, which holds no CUDA headers, and the
        # engine dies during its KV cache init with
        #
        #     <command-line>: fatal error: cuda_runtime.h: No such file or directory
        #
        # Resolving the symlink here is the whole fix. It is done in the parent rather than left to
        # the shell because the spawned engine core inherits this environment, and it is set before
        # `from vllm import LLM` below only for tidiness — flashinfer reads it at build time
        if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
            nvcc = shutil.which("nvcc")
            if nvcc is not None:
                os.environ["CUDA_HOME"] = os.path.dirname(
                    os.path.dirname(os.path.realpath(nvcc))
                )

    if engine == "transformers":
        from unsloth import FastLanguageModel

    import torch  # noqa: E402
    from datasets import Dataset  # noqa: E402
    from tqdm import tqdm  # noqa: E402
    from transformers import AutoConfig, AutoTokenizer  # noqa: E402

    # set data paths
    if path != "":
        DATASET_PATH = os.path.join(path, "generated_datasets/")
        MODEL_PATH = os.path.join(path, "model_outputs/")
        # create the directories if they do not exist
        os.makedirs(DATASET_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)

    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC} "
        f"(shard {shard_id}, engine: {engine})"
    )

    # vLLM cannot load a bare LoRA adapter directory as a model, so it samples from the merged
    # fp16 checkpoint that run_baseline.py writes next to the adapter. The two are the same
    # weights: the merge dequantizes the base and folds B @ A into it, so this is not a different
    # model
    if engine == "vllm":
        checkpoint = f"{MODEL_PATH}model_{generation}_bs{block_size}_{specifier_name}_fp16"
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"the merged checkpoint {checkpoint} does not exist. run_baseline.py writes it "
                "with save_pretrained_merged right after training a generation — a run whose "
                "models predate that step has to use --engine transformers, which reads the "
                "adapter directory instead"
            )
    else:
        checkpoint = f"{MODEL_PATH}model_{generation}_bs{block_size}_{specifier_name}"

    # load the base subdataset. The orchestrator writes these shards once, they hold the
    # instructions which are the same for every generation
    subdataset = Dataset.load_from_disk(
        DATASET_PATH + f"base_subdataset_bs{block_size}_{specifier_name}_shard{shard_id}"
    )
    instructions = list(subdataset["instruction"])

    if engine == "vllm":
        new_responses = generate_vllm(instructions, checkpoint)
    else:
        new_responses = generate_transformers(instructions, checkpoint)

    # save the new dataset to disk
    new_dataset = Dataset.from_dict(
        {"instruction": instructions, "response": new_responses}
    )

    new_dataset.save_to_disk(
        DATASET_PATH + f"subdataset_{generation}_bs{block_size}_{specifier_name}_shard{shard_id}"
    )
