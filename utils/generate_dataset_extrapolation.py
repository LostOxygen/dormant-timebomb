"""
Helper module to generate datasets in parallel. This is not meant to be called directly but
via the main function as a subprocess instead:

    python -m utils.generate_dataset_extrapolation --generation 3 ...

These are modules, not scripts: they are launched as `python -m utils.<name>` from the repo
root, because they import sibling helpers with `from utils.X import Y` and running the file
directly (`python utils/<name>.py`) puts utils/ on sys.path instead of the root, so `utils` is
then not a package at all.

Supports the three approximation methods of run_extrapolation.py, see utils/extrapolation.py
for what they are. All of them are indexed by the same factor n = generation + 1, so that
generation 0 reproduces the real model_0 anchor and only the generations above it approximate.

Args:
    block_size (int): The block size to use for training.
    specifier_name (str): The model specifier to use for training.
    dataset_batch_size (int): The dataset batch size to use for training.
    generation (int): The current generation.
    shard_id (int): The current shard id.
    method (str): Which approximation to use ("logit", "lora" or "data").
    adapter_path (str): Path of the alpha scaled LoRA adapter ("lora" method only).
    surrogate_top_p (float): The calibrated p_1 of the data-space surrogate ("data" only).
    temperature (float): Sampling temperature, pinned rather than inherited.
    top_p (float): Nucleus cutoff, pinned. The "data" method replaces it with its schedule.
    top_k (int): Top-k cutoff, pinned. -1 disables it.
    real_data_fraction (float): The run's --real_data_fraction, which names the shard written.
    load_in_4bit (bool): Quantize the generating models. Off by default.
    path (str): The path where the datasets and models are stored.

Returns:
    None
"""
from unsloth import FastLanguageModel

import os
import argparse

from datasets import Dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    LogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

from utils.colors import TColors
from utils.naming import mixture_suffix
from utils.extrapolation import (
    METHODS,
    dataset_suffix,
    extrapolate_logits,
    surrogate_top_p,
)
from utils.utils import clear_inherited_max_length

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
# has to match utils/generate_dataset.py and utils/calibrate_surrogate.py: stage 1 and stage 2 are
# plotted against each other, so a difference in the decoding is a difference in what is compared.
# See utils/generate_dataset.py for the measurements behind the value: at 3.0 the penalty stops the
# model reusing identifiers it has already written and only 7.8% of responses still compile,
# against 96.9% at 1.2 and 100% for the human corpus
REPETITION_PENALTY: float = 1.2

class UnslothExtrapolationProcessor(LogitsProcessor):
    def __init__(self, model_collapsed, generation_n: float, prompt_attention_mask: torch.Tensor):
        """
        Injects the extrapolation math directly into the native Unsloth generate() function.
        It maintains its own Unsloth-compatible KV-cache for the secondary model.

        Args:
            model_collapsed: the model whose logits define the collapse direction, i.e. model_0
            generation_n (float): the factor n of base + n * (collapsed - base)
            prompt_attention_mask (torch.Tensor): the real attention mask of the tokenized
                prompts. The collapsed model has to be conditioned on exactly the same prompt
                as the base model, so it needs the actual mask and padding aware position_ids.
                The batch is left padded, so feeding it an all-ones mask and plain index
                positions makes it attend to the pad tokens and read every position off by the
                number of pads. Its logits would then differ from the base model's because of
                the padding instead of because of the collapse, and the difference the
                extrapolation scales up would be noise rather than the collapse direction
        """
        self.model_collapsed = model_collapsed
        self.generation_n = generation_n
        self.prompt_attention_mask = prompt_attention_mask

        # Internal state for the secondary model's cache
        self.past_key_values = None
        self.attention_mask = None
        self.position_ids = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 'scores' are the highly-optimized logits from the base model that generates
        # 'input_ids' contains the sequence generated so far
        device = input_ids.device

        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            if self.past_key_values is None:
                # FIRST STEP: Process the full prompt
                if self.prompt_attention_mask.shape != input_ids.shape:
                    raise RuntimeError(
                        "the prompt attention mask does not match the prompt that generate() "
                        f"passed in: {tuple(self.prompt_attention_mask.shape)} vs "
                        f"{tuple(input_ids.shape)}"
                    )
                # the real mask, so the pad tokens stay masked out for the collapsed model too
                self.attention_mask = self.prompt_attention_mask.to(device)

                # the batch is left padded, so a token's position is the number of real tokens
                # before it and not its index in the padded row. The pads themselves are pinned
                # to position 0, which is irrelevant since the mask excludes them anyway
                self.position_ids = (self.attention_mask.cumsum(-1) - 1).clamp(min=0)

                outputs = self.model_collapsed(
                    input_ids=input_ids,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                    use_cache=True
                )
            else:
                # SUBSEQUENT STEPS: Process only the single newest token
                new_token = input_ids[:, -1:]
                
                # every generated token is a real one, so the mask is extended with ones
                next_mask = torch.ones(
                    (batch_size, 1), dtype=self.attention_mask.dtype, device=device
                )
                self.attention_mask = torch.cat([self.attention_mask, next_mask], dim=-1)

                # continue every row from its own last position, which keeps the rows with
                # padding offset correctly against the rows without
                self.position_ids = self.position_ids[:, -1:] + 1
                
                outputs = self.model_collapsed(
                    input_ids=new_token,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True
                )

            # Safely unpack the output (handling Unsloth's tuple optimization)
            if isinstance(outputs, tuple):
                logits_gen1 = outputs[0][:, -1, :]
                self.past_key_values = outputs[1]
            else:
                logits_gen1 = outputs.logits[:, -1, :]
                self.past_key_values = outputs.past_key_values

        # the extrapolation itself lives in utils/extrapolation.py, so that this script and the
        # differentiable surrogate that run_attack.py optimizes against cannot drift apart. It
        # also handles the -inf of already forbidden tokens, NaNs, and the absence of clamping
        return extrapolate_logits(scores, logits_gen1, self.generation_n)


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
    help="specifies the dataset batch size to use for training",
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
    "--method",
    "-m",
    type=str,
    default="logit",
    choices=METHODS,
    help="which approximation of the later generation to use (default: logit)",
)
parser.add_argument(
    "--adapter_path",
    "-ap",
    type=str,
    default="",
    help="path of the alpha scaled LoRA adapter, required by the 'lora' method",
)
parser.add_argument(
    "--surrogate_top_p",
    "-stp",
    type=float,
    default=0.0,
    help="the calibrated p_1 of the data-space surrogate, required by the 'data' method",
)
parser.add_argument(
    "--temperature",
    "-tp",
    type=float,
    default=0.7,
    help="sampling temperature. Pinned rather than inherited from the model's generation_config, "
    "so that all three methods and run_baseline.py sample from the same distribution "
    "(default: 0.7, Qwen2.5's own value)",
)
parser.add_argument(
    "--top_p",
    "-tpp",
    type=float,
    default=0.8,
    help="nucleus sampling cutoff, pinned for the same reason as --temperature. The 'data' "
    "method replaces it with its own per generation schedule (default: 0.8)",
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
    "--real_data_fraction",
    "-rdf",
    type=float,
    default=0.0,
    help="the run's --real_data_fraction. It names the shard this writes and nothing else: the "
    "base_subdataset read below is the human instruction set, which no mixture touches, and the "
    "model_0 anchor is shared across mixtures (default: 0.0)",
)
parser.add_argument(
    "--load_in_4bit",
    "-q4",
    action="store_true",
    help="quantize the generating models. A 0.5B model is ~1GB in bf16 on a 48GB card, so this "
    "only adds a dequantization kernel to every forward pass — and it would make these "
    "approximations run at a different precision than the baseline they are compared against",
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
method = args.method
adapter_path = args.adapter_path
surrogate_p1 = args.surrogate_top_p
temperature = args.temperature
top_p = args.top_p
top_k = args.top_k
real_data_fraction = args.real_data_fraction
load_in_4bit = args.load_in_4bit
path = args.path

suffix = dataset_suffix(method)
# the shard is named after the generation that produced it, so generation 0 stays untagged: n = 1
# reproduces the real model_0 anchor, which every mixture shares. utils/calculate_perplexity.py
# resolves the merged corpus with the same rule, one generation down
mix = mixture_suffix(real_data_fraction, generation)
# every method is indexed by the same factor: in the real collapse run generated_dataset_g is
# produced by model_g, and model_0 is a single fine-tuning step away from the base model, so
# model_g sits g + 1 steps out. With a factor of g instead, generation 0 would be a plain copy
# of the base model and generation 1 a plain copy of the collapsed model, i.e. the first two
# datasets would be the two anchors rather than approximations
generation_n = generation + 1

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(
    f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC} "
    f"(method: {method}, n = {generation_n})"
)

# max_seq_length has to hold the prompt *plus* the generated response, while --block_size caps only
# the response — it is max_new_tokens further down. Passing block_size here conflated the two, which
# was invisible at --block_size 2048 (no prompt in this dataset reaches that) and fails at 512: the
# instructions run up to ~1300 tokens, so a batch containing one of them is longer than the causal
# mask unsloth builds for max_seq_length, and generate() dies inside the mask shim with
#
#     RuntimeError: The size of tensor a (512) must match the size of tensor b (841)
#
# Only ~0.2% of prompts are that long, so it surfaced 16 batches into a shard rather than at once.
# The model's own trained context is the budget: it always covers prompt + block_size, and using
# exactly max_position_embeddings rather than something larger avoids tripping unsloth's RoPE
# extension, so the decoding stays the one the histograms were produced with. The vLLM path reaches
# the same place from the other side, sizing max_model_len off the actual prompts plus block_size
# (utils/generate_dataset.py::generate_vllm)
max_seq_length = AutoConfig.from_pretrained(MODEL_SPECIFIER).max_position_embeddings

# use the model to generate the new dataset
# for this, the model is loaded again with the quantized weights
model_collapsed = None
schedule_top_p = None

if method == "logit":
    # the base model is the one that generates, the collapsed model only contributes the logit
    # direction through the logits processor further down. Both have to be resident
    generation_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_SPECIFIER,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(generation_model)

    model_collapsed, _ = FastLanguageModel.from_pretrained(
        model_name=f"{MODEL_PATH}model_0_bs{block_size}_{specifier_name}",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model_collapsed)

elif method == "lora":
    # the collapse adapter with its alpha already scaled by n, i.e. weights
    # W_base + n * (W_collapsed - W_base). run_extrapolation.py builds it once per generation so
    # that the shards of a generation do not race each other over the same directory. Only one
    # model is resident and no second KV cache is kept, so this is the cheapest of the three
    if adapter_path == "":
        raise ValueError(
            "the 'lora' method needs --adapter_path, which run_extrapolation.py builds with "
            "utils.extrapolation.build_scaled_adapter"
        )
    generation_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(generation_model)

else:
    # the data-space surrogate: the pristine base model, sampled with a support that has been
    # truncated once per generation. Nothing about the model is modified at all, the collapse is
    # imitated at the level of the sampling that produces the corpus
    generation_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_SPECIFIER,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(generation_model)

    schedule_top_p = surrogate_top_p(surrogate_p1, generation_n)
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Surrogate top-p{TColors.ENDC}: "
        f"{surrogate_p1} ** {generation_n} = {schedule_top_p:.6f}"
    )

# left padding is required for batched generation and is what for_inference sets, but it sets it on
# the model's internal tokenizer copy. Setting it here as well guarantees that the prompt occupies a
# constant prefix of every row of the batch, which the prompt slicing further down relies on — and
# it is what the extrapolation processor's padding aware position_ids assume
tokenizer.padding_side = "left"

# all three methods reach here with a loaded generation_model, so this is the one place the
# checkpoint's inherited max_length has to be dropped. The generate() call below passes
# max_new_tokens, which transformers honours either way — this only stops it from logging a
# "Both max_new_tokens and max_length seem to have been set" line per batch
clear_inherited_max_length(generation_model)

# load the base subdataset from the previous generation
subdataset = Dataset.load_from_disk(
    DATASET_PATH + f"base_subdataset_bs{block_size}_{specifier_name}{suffix}_shard{shard_id}"
)

instructions = list(subdataset["instruction"])
prompts = [
    tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant for code completion.",
            },
            {"role": "user", "content": instruction},
        ],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    for instruction in instructions
]

# a batched generate() runs every sequence of the batch until the longest one finishes and pads
# the rest, so grouping prompts of similar length together keeps that waste down. The responses
# are written back through the permutation, so the output order is still the dataset order
prompt_lengths = [
    len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
]
order = sorted(range(len(prompts)), key=lambda index: prompt_lengths[index])

new_responses = [None] * len(prompts)
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

    # everything that is not the mechanism of the method itself is kept identical across the
    # three methods, so that a difference between their histograms is a difference between the
    # approximations and not between their decoding setups. do_sample, num_beams,
    # repetition_penalty, temperature, top_p and top_k are all pinned below — the sampling
    # parameters go through one dict rather than being passed as keywords, because the "data"
    # method has to *replace* top_p and passing both would be a duplicate keyword argument
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    if method == "logit":
        # the logit extrapolation runs as a custom logits processor during decoding. It is
        # rebuilt per batch because it holds the collapsed model's KV cache and this batch's
        # prompt mask.
        #
        # The repetition penalty has to run *after* it, which is why it is in this list instead of
        # being left to generate(): the processors generate() builds itself run before any custom
        # one, so the base scores would already be penalized while the collapsed model's logits
        # stay raw. The extrapolation then works out to
        # (1 - n) * penalized_base + n * raw_collapsed, which cancels the penalty at n = 1 and
        # inverts it for n > 1 — the negative coefficient pushes the tokens the penalty pushed
        # down back up, i.e. it actively rewards repetition, harder with every generation.
        # Handing the penalty over as a custom processor is not sufficient on its own either:
        # _merge_criteria_processor_list substitutes a same-typed custom processor *at the position
        # of the default it replaces*, so it would still land before the extrapolation. Disabling
        # the built-in one with 1.0 below is what makes this list's order the effective one
        sampling_kwargs["logits_processor"] = LogitsProcessorList(
            [
                UnslothExtrapolationProcessor(
                    model_collapsed=model_collapsed,
                    generation_n=generation_n,
                    prompt_attention_mask=inputs["attention_mask"],
                ),
                RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY),
            ]
        )
        sampling_kwargs["repetition_penalty"] = 1.0
    else:
        # no extrapolation processor is involved, so there is nothing the built-in penalty could be
        # cancelled by and it can be left to generate() as usual
        sampling_kwargs["repetition_penalty"] = REPETITION_PENALTY
        if method == "data":
            # the whole method *is* the truncation of the sampling support, so its schedule
            # replaces the pipeline's top_p instead of composing with it — which also matches
            # calibrate_surrogate.py, whose grid replaces top_p to fit p_1
            sampling_kwargs["top_p"] = schedule_top_p

    generated_answers = generation_model.generate(
        **inputs,
        # pinned, not inherited from the model's generation_config: collapse is driven by
        # resampling from the model's own distribution, so the decoding has to be plain
        # multinomial sampling. Beam search (num_beams > 1) optimizes for likelihood and
        # systematically narrows the output distribution, which suppresses exactly the effect
        # this pipeline measures — and it would do so unevenly across the three methods
        do_sample=True,
        num_beams=1,
        # repetition_penalty is not passed here: it is REPETITION_PENALTY for every method,
        # but *where* it is applied differs, so both branches above set it in sampling_kwargs
        # instead. Its cost is accepted knowingly — the responses are scored by an unpenalized
        # forward pass, so the penalty makes the measured perplexity partly a property of the
        # sampling distortion rather than of the model, and since it divides the logit of every
        # token already in the context, its severity grows with the response length and therefore
        # aliases onto the collapse trend rather than being a constant offset
        min_new_tokens=128,
        max_new_tokens=block_size,
        use_cache=True,
        **sampling_kwargs,
    )

    # the prompt is dropped by token count instead of by splitting the decoded string on the chat
    # template markers, since skip_special_tokens removes those markers. The batch is left padded,
    # so the prompt is the same number of tokens in every row
    prompt_length = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(
        generated_answers[:, prompt_length:],
        skip_special_tokens=True,
    )
    for index, answer in zip(batch_indices, decoded):
        # skip_special_tokens already dropped the trailing <|im_end|> and the padding, so only the
        # surrounding whitespace is left to strip
        new_responses[index] = answer.strip()

# save the new dataset to disk
new_dataset = Dataset.from_dict(
    {"instruction": instructions, "response": new_responses}
)

new_dataset.save_to_disk(
    DATASET_PATH
    + f"subdataset_{generation}_bs{block_size}_{specifier_name}{suffix}{mix}_shard{shard_id}"
)
