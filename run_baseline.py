"""main hook to start the pitfall 1 fine-tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import datetime
import getpass
import os
import shutil
import subprocess
import time
from datetime import timedelta
from typing import Final

import psutil
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

from utils.colors import TColors
from utils.devices import visible_devices
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import mixture_suffix, mixture_tag
from utils.plotting import visible_perplexity_range
from utils.utils import report_block_size

# this orchestrator deliberately does not import unsloth: every stage that touches a model is a
# subprocess (utils/train_generation.py under torchrun, utils/generate_dataset.py,
# utils/calculate_perplexity.py). That is what allows the training to be data parallel at all —
# unsloth's multi-GPU path goes through torchrun — and it keeps unsloth's import time environment
# fiddling out of this process
VISIBLE_DEVICES = visible_devices()

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
MODEL_PATH: str = "./model_outputs/"
DATASET_PATH: str = "./generated_datasets/"
EOS_TOKEN: str = None  # will be overwritten by the tokenizer
MAX_TOKEN_LENGTH: Final[int] = None  # will be overwritten
TOKENIZER = None  # will be overwritten
# lower y-limit of the perplexity histogram. Everything below is invisible, so this is also
# used as the threshold to determine the plot's x-range
Y_LIMIT_LOWER: Final[float] = 1e-5


def format_prompt(examples: dict) -> dict:
    """format the dataset inputs for the trainer"""

    user_inputs = examples["instruction"]
    completion_data = examples["response"]

    prompts = []

    for instr, answer in zip(user_inputs, completion_data):
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant for code completion.",
            },
            {"role": "user", "content": instr},
            {"role": "assistant", "content": answer},
        ]
        formatted_prompt = TOKENIZER.apply_chat_template(
            prompt, tokenize=False, add_special_tokens=False
        )
        prompts.append(formatted_prompt)

    return {"text": prompts}


def mix_real_data(
    synthetic: Dataset, real: Dataset, fraction: float, seed: int, generation: int
) -> Dataset:
    """Replaces `fraction` of a generation's synthetic corpus with original human examples.

    This is the dial on how hard the collapse bites. With `fraction` at 0 every generation trains
    exclusively on the previous generation's output — the "replace" regime, which degrades without
    bound and, in this pipeline, costs the models the ability to write code at all within a couple
    of generations. Keeping some real data in the mix bounds the degradation instead, so the later
    generations still drift but stay capable enough that an attack on them means something.

    The corpus *size* is held constant rather than grown: only the composition changes, so every
    generation sees the same number of training examples and the same number of optimizer steps.
    That keeps the collapse curve attributable to the data mixture instead of to a changing
    training budget.

    The real slice is redrawn every generation rather than fixed once. The premise is that whoever
    runs the pipeline holds the whole human corpus, not one frozen sample of it, and redrawing
    stops a single unlucky slice from being memorized over and over across ten generations.

    Args:
        synthetic (Dataset): the previous generation's generated corpus, already chat-formatted
        real (Dataset): the original human corpus, already chat-formatted
        fraction (float): share of the returned corpus taken from `real`, in [0, 1]
        seed (int): run seed, so the draw is reproducible
        generation (int): current generation, mixed into the shuffle seed

    Returns:
        Dataset: a corpus of len(synthetic) rows, `fraction` of them real, shuffled together
    """
    if fraction <= 0:
        return synthetic

    n_total = len(synthetic)
    n_real = min(round(fraction * n_total), len(real))
    if n_real == 0:
        return synthetic

    # concatenate_datasets matches on the arrow schema, and the two corpora carry the same columns
    # in a different order (the generated one is built from instruction/response, the original one
    # gained "text" last), so the column order is normalized before they are joined
    real = real.select_columns(synthetic.column_names)

    shuffle_seed = seed + generation
    real_part = real.shuffle(seed=shuffle_seed).select(range(n_real))
    synthetic_part = synthetic.shuffle(seed=shuffle_seed).select(range(n_total - n_real))
    mixed = concatenate_datasets([synthetic_part, real_part]).shuffle(seed=shuffle_seed)

    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Mixed in real data{TColors.ENDC}: "
        f"{n_real} original + {n_total - n_real} generated = {len(mixed)} rows "
        f"({n_real / len(mixed):.1%} real)"
    )
    return mixed


def make_splits(dataset: Dataset) -> Dataset:
    """Splits the dataset into training and validation sets"""
    # split the dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    return train_dataset, val_dataset


def main(
    device: str = "cpu",
    training_epochs: int = 5,
    dataset_batch_size: int = 10,
    training_batch_size: int = 8,
    perplexity_batch_size: int = 16,
    skip_training: bool = False,
    num_generations: int = 5,
    block_size: int = 512,
    histogram_only: bool = False,
    human_eval_only: bool = False,
    path: str = "",
    model_specifier: str = "",
    model_size: str = "",
    continue_from_generation: int = 0,
    dataset_size: int = 0,
    real_data_fraction: float = 0.0,
    seed: int = 1337,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    engine: str = "auto",
    gpu_memory_utilization: float = 0.90,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    gradient_accumulation_steps: int = 4,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    perplexity_load_in_4bit: bool = False,
    fresh_init: bool = False,
    training_gpus: int = 0,
    master_port: int = 29500,
) -> None:
    """
    Main function to start the pitfall 1 fine-tuning

    Args:
        device (str): device to run the computations on (cpu, cuda, mps)
        training_epochs (int): number of training epochs to run
        dataset_batch_size (int): batch size for the dataset
        training_batch_size (int): batch size for the training/eval
        perplexity_batch_size (int): batch size for the perplexity calculation
        skip_training (bool): if True, skip the training and only evaluate the models
        num_generations (int): number of generations to run (default: 5)
        block_size (int): size of the blocks to split the dataset into (default: 64)
        histogram_only (bool): if True, only generate the histogram and skip the rest
        human_eval_only (bool): if True, only generate human eval samples and skip the rest
        path (str): path to save the generated datasets and models
        model_specifier (str): model specifier to use for the training
        model_size (str): parameter count off the Qwen2.5-Coder ladder ("0.5b" ... "32b"),
            shorthand for the matching model_specifier. Mutually exclusive with it
        continue_from_generation (int): generation to continue from (default: 0, start from scratch)
        dataset_size (int): number of dataset samples to use, taken from the front of the
            upstream 50k dataset. Must match between run_baseline.py and run_extrapolation.py
        real_data_fraction (float): share of every post-generation-0 training corpus taken from
            the *original* human dataset instead of the previous generation's output, in [0, 1].
            0.0 is pure self-training, which collapses fastest and loses code generation
            entirely within a couple of generations; higher values bound the degradation so the
            later generations still drift but stay capable enough to be worth attacking. The
            corpus size is unchanged, only its composition
        seed (int): seed of the whole collapse trajectory, threaded into the training worker and
            into the sampling seed of every generation worker. Two runs that differ only in this
            produce two independent collapse trajectories from identical hyperparameters
        learning_rate (float): LoRA learning rate
        lora_rank (int): LoRA rank r. Together with lora_alpha this bounds how far one
            generation can drift from the last one, i.e. how strong the collapse effect is
        lora_alpha (int): LoRA alpha. The adapter contributes (alpha / r) * B @ A
        engine (str): inference engine for the dataset generation (auto, vllm, transformers)
        gpu_memory_utilization (float): fraction of each GPU vLLM may use
        temperature (float): sampling temperature of the dataset generation
        top_p (float): nucleus sampling cutoff of the dataset generation
        top_k (int): top-k sampling cutoff of the dataset generation, -1 disables it
        gradient_accumulation_steps (int): optimizer step granularity. The effective batch is
            training_batch_size * this, and that product is what controls the drift per
            generation — keep it constant when tuning either knob
        load_in_4bit (bool): quantize the model for training. Only useful for models that do
            not otherwise fit
        gradient_checkpointing (bool): recompute activations in the backward pass instead of
            keeping them. Only useful for models that do not otherwise fit
        perplexity_load_in_4bit (bool): quantize the model that scores the perplexity. This adds
            quantization noise to the plotted statistic, so it is off by default
        fresh_init (bool): re-initialise every generation from the pristine base model instead of
            continuing to fine-tune the previous generation's adapter. Only the *data* is then
            recursive, which is the "replace" setting of the model collapse literature. Off by
            default, i.e. the weights are recursive too — see utils/train_generation.py
        training_gpus (int): how many of the visible GPUs to train one generation on, as data
            parallel ranks under torchrun. 0 uses all of them, 1 forces single GPU training
        master_port (int): torchrun rendezvous port. Only needs changing when two pipelines run
            on the same machine, which would otherwise collide on it

    Returns:
        None
    """
    start_time = time.time()

    # ──────────────────────────── set devices and print informations ─────────────────────────
    # set the devices correctly
    if "cpu" in device:
        device = torch.device("cpu", 0)
    elif "cuda" in device and torch.cuda.is_available():
        if "cuda" not in device.split(":")[-1]:
            device = torch.device("cuda", int(device.split(":")[-1]))
        else:
            device = torch.device("cuda", 0)
    elif "mps" in device and torch.backends.mps.is_available():
        if "mps" not in device.split(":")[-1]:
            device = torch.device("mps", int(device.split(":")[-1]))
        else:
            device = torch.device("mps", 0)
    else:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} "
            f"{TColors.ENDC}is not available. Setting device to CPU instead."
        )
        device = torch.device("cpu", 0)

    # set data paths
    if path != "":
        global DATASET_PATH
        DATASET_PATH = os.path.join(path, "generated_datasets/")
        global MODEL_PATH
        MODEL_PATH = os.path.join(path, "model_outputs/")
        # create the directories if they do not exist
        os.makedirs(DATASET_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)

    # set the model specifier. --model_size picks one off the Qwen2.5-Coder ladder and
    # --model_specifier names any repo id directly; resolve_model_specifier raises rather than
    # rank them when both are given, because the ignored one would be a whole collapse run trained
    # under a name that says otherwise
    global MODEL_SPECIFIER
    MODEL_SPECIFIER = resolve_model_specifier(model_size, model_specifier, MODEL_SPECIFIER)
    specifier_name = MODEL_SPECIFIER.split("/")[-1]
    # the ladder rung this run resolved to, for the parameters banner. Taken from the resolved id
    # rather than from the flag, so it reads the same whichever of the two named the model — the
    # later stages have to be given that same model and the log is where that gets checked
    size_label = model_size_label(MODEL_SPECIFIER) or "outside the --model_size ladder"

    # which weight lineage this run used, carried into the names of everything the run is compared
    # by. The two modes produce different collapse curves from the same data, so a plot without
    # this in its name is not attributable to either of them — and plots/ is not under --path, so
    # two runs would otherwise overwrite each other's figure even with separate output paths
    init_suffix = "_freshinit" if fresh_init else "_recursive"
    init_label = "fresh weights" if fresh_init else "recursive weights"

    if not 0.0 <= real_data_fraction < 1.0:
        raise SystemExit(
            f"--real_data_fraction must be in [0, 1), got {real_data_fraction}. At 1.0 every "
            f"generation would train on the human corpus alone and nothing would collapse."
        )
    # the data mixture goes into the names for the same reason the weight lineage does: it changes
    # the collapse curve, and plots/ sits outside --path so the file name is the only thing keeping
    # two runs' figures apart. Appended only when non-zero, so a pure self-training run keeps
    # writing (and --histogram_only keeps finding) exactly the artifact names it always had.
    #
    # This is the *run-level* tag, for artifacts that span the whole run. The models and datasets of
    # individual generations are named with mixture_suffix() instead, which additionally returns ""
    # for generation 0 — see utils/naming.py for why generation 0 is deliberately shared
    data_suffix = mixture_tag(real_data_fraction)
    run_suffix = f"{init_suffix}{data_suffix}"
    # no underscores or backslashes: usetex renders the title, and a bare % is a LaTeX comment
    run_label = f"{init_label}, real data fraction {real_data_fraction:g}"

    # allow tf32 for the matmuls of the training stage. The L40S' tensor cores run tf32 at
    # multiples of the fp32 rate and this only affects the fp32 fallbacks, not the bf16 path
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # load the tokenizer to count the tokens of the dataset. AutoTokenizer instead of
    # FastLanguageModel.from_pretrained, which would load a whole quantized model just to hand
    # back its tokenizer — and would keep that model resident on GPU 0 for the entire run,
    # because binding it to `_` does not free it
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SPECIFIER)
    global EOS_TOKEN
    global TOKENIZER
    EOS_TOKEN = tokenizer.eos_token
    TOKENIZER = tokenizer

    # load the dataset
    original_dataset = load_dataset(DATASET_SPECIFIER, split="train")
    original_dataset = original_dataset.select_columns(["response", "instruction"])

    # subsample to --dataset_size. The upstream dataset name says 50k, but the collapse
    # experiment does not need all of it and every generation pays for the full size twice
    # (once generating, once training). A contiguous slice from the front is used so that
    # run_baseline.py and run_extrapolation.py operate on the *same* subset without needing a
    # shared seed, which is what makes their perplexity histograms comparable
    if 0 < dataset_size < len(original_dataset):
        original_dataset = original_dataset.select(range(dataset_size))

    # gather information about the dataset. One batched call into the Rust tokenizer instead of
    # one python level call per sample — the previous loop also built a padded pytorch tensor per
    # sample only to read its length back off the shape
    print("Calculating token counts")
    token_counts = [
        len(ids)
        for ids in tokenizer(
            list(original_dataset["response"]),
            truncation=True,
            max_length=tokenizer.model_max_length,
        )["input_ids"]
    ]

    # set the block size. The requested value wins — see utils.utils.report_block_size for why it
    # is no longer silently raised to the longest response
    global MAX_TOKEN_LENGTH
    MAX_TOKEN_LENGTH = max(token_counts)
    block_size = report_block_size(block_size, token_counts)

    # resolve the GPUs once, for the generation and perplexity fan out and for the training ranks.
    # VISIBLE_DEVICES comes from utils.devices rather than from the environment directly, so that
    # the list is the one the run was launched with no matter what an imported library did to
    # CUDA_VISIBLE_DEVICES in the meantime
    devices = VISIBLE_DEVICES if str(device).startswith("cuda") else [0]

    # the generations are strictly sequential — generation g trains on what model_{g-1} generated —
    # so the only thing to parallelise in the training stage is a single generation's fine-tuning,
    # as plain data parallelism over these devices. --training_gpus 1 falls back to single GPU
    # training, which is the reference to check the data parallel path against
    training_devices = devices if training_gpus <= 0 else devices[:training_gpus]

    # have a nice system status print
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + "#" * (shutil.get_terminal_size().columns - 23)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: "
        + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: "
        f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and "
        f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda", 0)) and torch.cuda.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Number of GPUs{TColors.ENDC}: "
            f"{torch.cuda.device_count()}"
        )
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: "
            f"{torch.cuda.mem_get_info()[1] // 1024**2} MB"
        )
    elif (
        device == "mps" or torch.device("mps", 0)
    ) and torch.backends.mps.is_available():
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters"
        + f"{TColors.ENDC} "
        + "#" * (shutil.get_terminal_size().columns - 14)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Specifier{TColors.ENDC}: {MODEL_SPECIFIER}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Model Size{TColors.ENDC}: {size_label}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Specifier{TColors.ENDC}: {DATASET_SPECIFIER}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Number of Generations{TColors.ENDC}: {num_generations}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Block size{TColors.ENDC}: {block_size}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Training Steps{TColors.ENDC}: {training_epochs}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Batch Size{TColors.ENDC}: {dataset_batch_size}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Training Batch Size{TColors.ENDC}: {training_batch_size}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Perplexity Batch Size{TColors.ENDC}: "
        f"{perplexity_batch_size}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Skip Training{TColors.ENDC}: {skip_training}"
    )
    # the data recursion is the collapse and is always on; this is the weight lineage
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Weight lineage{TColors.ENDC}: {init_label} "
        f"({'--fresh_init' if fresh_init else 'previous generation is fine-tuned further'})"
    )
    # how much of the collapse loop is fed back into itself, i.e. how fast the models degrade
    if real_data_fraction > 0:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Real data fraction{TColors.ENDC}: "
            f"{real_data_fraction:g} — every generation after 0 trains on "
            f"{1 - real_data_fraction:.0%} previous-generation output and "
            f"{real_data_fraction:.0%} original human data"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Real data fraction{TColors.ENDC}: 0 — pure "
            f"self-training, the fastest collapse (raise it if the models lose code generation "
            f"before the generation you want to attack)"
        )
    # printed because unsloth silently rewrites CUDA_VISIBLE_DEVICES to a single device at import,
    # which used to collapse this list to [0] without any sign of it in the output
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Sharding generation/perplexity across{TColors.ENDC}: "
        f"{len(VISIBLE_DEVICES)} GPU(s) {VISIBLE_DEVICES}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Training one generation on{TColors.ENDC}: "
        f"{len(training_devices)} GPU(s) {training_devices} (data parallel under torchrun), "
        f"effective batch {training_batch_size * gradient_accumulation_steps}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Saving Path{TColors.ENDC}: {MODEL_PATH}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Generated Datasets Path{TColors.ENDC}: {DATASET_PATH}"
    )
    if continue_from_generation > 0:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Continue from Generation{TColors.ENDC}: "
            f"{continue_from_generation}"
        )
    print("#" * shutil.get_terminal_size().columns + "\n")

    # print information about the dataset
    print(f"Max token count: {max(token_counts)}")
    print(f"Avg token count: {sum(token_counts) / len(token_counts)}")
    print(f"Min token count: {min(token_counts)}")
    print(f"Original dataset length: {len(original_dataset)}\n")
    original_dataset = original_dataset.map(format_prompt, batched=True)
    original_dataset.save_to_disk(DATASET_PATH + f"original_dataset_bs{block_size}")

    # preprocess the dataset
    chunked_dataset = original_dataset
    chunked_dataset.save_to_disk(
        DATASET_PATH + f"chunked_dataset_bs{block_size}_{specifier_name}"
    )
    if not skip_training:
        # the generation workers only ever read the *instructions*, which are the same for every
        # generation, so the shards are written once here instead of being rewritten inside the
        # generation loop. contiguous=True, because datasets defaults to strided sharding
        # (indices index, index + n, index + 2n, ...) and the shards are merged back in shard
        # order afterwards. Strided shards would therefore reorder the merged dataset as a
        # function of the *number of GPUs*: the rows are the same, but make_splits() takes the
        # 90/10 train/val split by position, so a 4-GPU run would train on a different subset
        # than a 1-GPU run. Contiguous shards reassemble into the original order for any device
        # count
        for shard_id in range(len(devices)):
            original_dataset.shard(
                num_shards=len(devices), index=shard_id, contiguous=True
            ).save_to_disk(
                DATASET_PATH
                + f"base_subdataset_bs{block_size}_{specifier_name}_shard{shard_id}"
            )

        # ───────────────────────── start the actual finetuning ──────────────────────────────
        # iterte over two loops: first the model training and then the dataset generation
        # the model is trained for N times and after each training the dataset
        # is generated from the new model
        for gen_id in range(num_generations):
            # check if generations need to be skipped if continue_from_generation > 0
            if gen_id < continue_from_generation:
                continue
            # ────────────────────────────── train this generation ────────────────────────────
            # the chat templating and the 90/10 split happen here, once, and the already prepared
            # splits go to disk for the training workers. Every rank then reads identical bytes
            # instead of all of them mapping the same dataset into the same datasets cache
            # the mixture only affects generations from 1 onward, so an artifact's suffix depends on
            # which generation produced it. The corpus read here was generated by model_{gen_id - 1}
            # and the splits written below train model_{gen_id}, which for gen_id 1 is an unsuffixed
            # input and a suffixed output — hence two separate calls rather than one variable
            gen_suffix = mixture_suffix(real_data_fraction, gen_id)
            prev_suffix = mixture_suffix(real_data_fraction, gen_id - 1)

            if gen_id > 0:
                # if the first training iteration is done, load the generated dataset from the disk
                dataset = Dataset.load_from_disk(
                    DATASET_PATH
                    + f"generated_dataset_{gen_id - 1}_bs{block_size}_{specifier_name}"
                    + prev_suffix
                )
                dataset = dataset.map(format_prompt, batched=True)
                # --real_data_fraction of it is swapped back for original human examples. Only
                # generations above 0 are affected: generation 0 trains on the human corpus by
                # definition, so there is nothing to mix in there
                dataset = mix_real_data(
                    dataset, chunked_dataset, real_data_fraction, seed, gen_id
                )
            else:
                # for first iteration (gen_id = 0) take the original dataset
                dataset = chunked_dataset

            # for the first model the original dataset is used, then the generated dataset
            # is used for the next models
            dataset_train, dataset_val = make_splits(dataset)
            dataset_train.save_to_disk(
                DATASET_PATH
                + f"train_dataset_{gen_id}_bs{block_size}_{specifier_name}{gen_suffix}"
            )
            dataset_val.save_to_disk(
                DATASET_PATH
                + f"val_dataset_{gen_id}_bs{block_size}_{specifier_name}{gen_suffix}"
            )

            # training runs in a torchrun subprocess rather than in this process. Two reasons:
            # unsloth's own multi-GPU support goes through torchrun, and a subprocess also hands
            # the whole allocator back at exit instead of leaving a generation's fragmentation
            # behind for the next one
            train_command = [
                "env",
                f"CUDA_VISIBLE_DEVICES={','.join(map(str, training_devices))}",
                "torchrun",
                f"--nproc_per_node={len(training_devices)}",
                # the default rendezvous port collides when two pipelines run on one machine
                f"--master_port={master_port}",
                "-m",
                "utils.train_generation",
                "--block_size",
                str(block_size),
                "--specifier_name",
                specifier_name,
                "--model_specifier",
                MODEL_SPECIFIER,
                "--generation",
                str(gen_id),
                "--training_epochs",
                str(training_epochs),
                "--training_batch_size",
                str(training_batch_size),
                "--gradient_accumulation_steps",
                str(gradient_accumulation_steps),
                "--learning_rate",
                str(learning_rate),
                "--lora_rank",
                str(lora_rank),
                "--lora_alpha",
                str(lora_alpha),
                "--path",
                str(path),
                "--seed",
                str(seed),
                # the worker names its own inputs and outputs, so it needs the fraction rather than
                # a ready-made suffix: it reads model_{gen-1} and writes model_{gen}, which are not
                # suffixed alike at generation 1
                "--real_data_fraction",
                str(real_data_fraction),
            ]
            if load_in_4bit:
                train_command.append("--load_in_4bit")
            if gradient_checkpointing:
                train_command.append("--gradient_checkpointing")
            if fresh_init:
                train_command.append("--fresh_init")

            training = subprocess.run(train_command, check=False)
            if training.returncode != 0:
                raise RuntimeError(
                    f"the training of generation {gen_id} failed with exit code "
                    f"{training.returncode}. See the subprocess output above for the actual error"
                )

            # ────────────────────────────── generate the new datasets ────────────────────────────
            # one worker per GPU, each generating the responses for its shard of the instruction
            # set. The shards were written once above; the workers only read them
            process_list = []
            for shard_id, d_id in enumerate(devices):
                process = subprocess.Popen(
                    [
                        "env",
                        f"CUDA_VISIBLE_DEVICES={d_id}",
                        "python",
                        "-m",
                        "utils.generate_dataset",
                        "--block_size",
                        str(block_size),
                        "--specifier_name",
                        specifier_name,
                        "--dataset_batch_size",
                        str(dataset_batch_size),
                        "--generation",
                        str(gen_id),
                        "--shard_id",
                        str(shard_id),
                        "--engine",
                        engine,
                        "--gpu_memory_utilization",
                        str(gpu_memory_utilization),
                        "--temperature",
                        str(temperature),
                        "--top_p",
                        str(top_p),
                        "--top_k",
                        str(top_k),
                        "--path",
                        str(path),
                        "--seed",
                        str(seed),
                        # names the checkpoint it loads and the shard it writes
                        "--real_data_fraction",
                        str(real_data_fraction),
                    ],
                )
                process_list.append(process)

            # wait for all processes to finish. wait() instead of polling in a loop: the old loop
            # mutated the list it was iterating over, which skips entries, and it tolerated a
            # crashed worker silently — the failure then only surfaced as a missing shard in the
            # concatenate below
            for process in process_list:
                process.wait()

            failed_shards = [
                shard
                for shard, process in enumerate(process_list)
                if process.returncode != 0
            ]
            if failed_shards:
                raise RuntimeError(
                    f"the dataset generation failed for shard(s) {failed_shards} of generation "
                    f"{gen_id}. See the subprocess output above for the actual error"
                )

            # merge all the subdatasets to one single dataset again
            merged_dataset = concatenate_datasets(
                [
                    Dataset.load_from_disk(
                        DATASET_PATH
                        + f"subdataset_{gen_id}_bs{block_size}_{specifier_name}{gen_suffix}"
                        + f"_shard{shard_id}"
                    )
                    for shard_id in range(len(devices))
                ]
            )
            merged_dataset.save_to_disk(
                DATASET_PATH
                + f"generated_dataset_{gen_id}_bs{block_size}_{specifier_name}{gen_suffix}"
            )

    # ────────────────── evaluate the models' perplexity and other metrics ─────────────────────────
    # iterate over every model and the generated dataset and calculate the perplexity
    # for the perplexity, every datapoint i.e., the generated answer for every question
    # is evaluated to get the probability for a given perplexity over the whole dataset
    if not human_eval_only:
        if not histogram_only:
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Calculate Perplexity{TColors.ENDC}"
            )
            perplexity_dict = {}
            all_perplexities = []

            # the datasets are split into one shard per GPU and every shard is processed by
            # its own subprocess. Each subprocess handles all generations of its shard, so
            # the model only has to be loaded once per GPU. Afterwards the per-shard results
            # are merged
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Using {len(devices)} GPU(s) for the "
                f"perplexity calculation{TColors.ENDC}"
            )

            # the run-level tag, not a per-generation one: each of these spans every generation.
            # They are transient, but tagging them keeps two mixtures sharing one --path from
            # reading each other's half-written shards
            shard_files = [
                DATASET_PATH
                + f"perplexity_dict_bs{block_size}_{specifier_name}{data_suffix}"
                + f"_shard{shard_id}.pt"
                for shard_id in range(len(devices))
            ]
            # remove stale shard files so results of a previous run can't be picked up
            for shard_file in shard_files:
                if os.path.exists(shard_file):
                    os.remove(shard_file)

            process_list = []
            for shard_id, d_id in enumerate(devices):
                process = subprocess.Popen(
                    [
                        "env",
                        f"CUDA_VISIBLE_DEVICES={d_id}",
                        "python",
                        "-m",
                        "utils.calculate_perplexity",
                        "--block_size",
                        str(block_size),
                        "--specifier_name",
                        specifier_name,
                        "--model_specifier",
                        MODEL_SPECIFIER,
                        "--perplexity_batch_size",
                        str(perplexity_batch_size),
                        "--num_generations",
                        str(num_generations),
                        "--shard_id",
                        str(shard_id),
                        "--num_shards",
                        str(len(devices)),
                        "--path",
                        str(path),
                        # it reads generated_dataset_{i-1} for every generation i, so it needs the
                        # fraction to name each one
                        "--real_data_fraction",
                        str(real_data_fraction),
                    ]
                    + (["--load_in_4bit"] if perplexity_load_in_4bit else []),
                )
                process_list.append(process)

            # wait for all processes to finish
            for process in process_list:
                process.wait()

            # every shard has to be there, otherwise the merged perplexities would be
            # incomplete
            failed_shards = [
                shard_id
                for shard_id, process in enumerate(process_list)
                if process.returncode != 0
            ]
            if failed_shards:
                raise RuntimeError(
                    f"The perplexity calculation failed for shard(s) {failed_shards}. "
                    "See the subprocess output above for the actual error."
                )

            # merge the per-shard perplexities back together. The shards are contiguous, so
            # concatenating them in shard order restores the original dataset order
            shard_dicts = [torch.load(shard_file) for shard_file in shard_files]
            for i in range(num_generations):
                perplexity_dict[f"Generation {i}"] = [
                    perplexity
                    for shard_dict in shard_dicts
                    for perplexity in shard_dict[f"Generation {i}"]
                ]

            # clean up the temporary shard files
            for shard_file in shard_files:
                os.remove(shard_file)

            # get all single values from the dict and flatten them into a list
            all_perplexities = [
                perplexity
                for values in perplexity_dict.values()
                for perplexity in values
            ]

            # save the perplexity dict to a file. The run_suffix keeps the weight lineages and
            # data mixtures from overwriting each other's cache, so each can be replotted with -ho
            # and the -ho of one configuration can never silently replot another's numbers
            torch.save(
                perplexity_dict,
                DATASET_PATH
                + f"perplexity_dict_bs{block_size}_{specifier_name}{run_suffix}.pt",
            )  # save the dict to a file
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Saved the perplexity dict under: "
                f"{TColors.HEADER}{DATASET_PATH}perplexity_dict_bs{block_size}_{specifier_name}"
                f"{run_suffix}.pt{TColors.ENDC}"
            )
            # save the all_perplexities list to a file
            torch.save(
                all_perplexities,
                DATASET_PATH
                + f"all_perplexities_bs{block_size}_{specifier_name}{run_suffix}.pt",
            )  # save the list to a file
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Saved the all_perplexities list under: "
                f"{TColors.HEADER}{DATASET_PATH}all_perplexities_bs{block_size}_{specifier_name}"
                f"{run_suffix}.pt{TColors.ENDC}"
            )
        else:
            # load the perplexity dict and all_perplexities list from the files. -ho therefore
            # needs the same --fresh_init and --real_data_fraction the run was produced with, and
            # says so rather than replotting whichever configuration happens to be cached
            cached_dict = (
                DATASET_PATH
                + f"perplexity_dict_bs{block_size}_{specifier_name}{run_suffix}.pt"
            )
            if not os.path.exists(cached_dict):
                raise FileNotFoundError(
                    f"{cached_dict} does not exist. --histogram_only replots the cache of a run "
                    f"with the same --block_size, --model_specifier, --fresh_init "
                    f"({'set' if fresh_init else 'not set'} here) and --real_data_fraction "
                    f"({real_data_fraction:g} here)"
                )
            perplexity_dict = torch.load(cached_dict)
            all_perplexities = torch.load(
                DATASET_PATH
                + f"all_perplexities_bs{block_size}_{specifier_name}{run_suffix}.pt"
            )

        # ────────────────── plot the perplexity histogram ─────────────────────────
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Plotting Perplexity Histogram{TColors.ENDC}"
        )

        # scale the x-axis to the range which actually contains visible data. The tails are so
        # heavy that even a 99.9% quantile still reaches 1e10, but those bins are drawn below
        # the lower y-limit and only leave the right side of the plot empty
        lower_limit, upper_limit, num_clipped, num_total = visible_perplexity_range(
            perplexity_dict, Y_LIMIT_LOWER
        )
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Plot range{TColors.ENDC}: "
            f"[{lower_limit:.0e}, {upper_limit:.0e}] (clipping {num_clipped} of {num_total} "
            f"perplexities, {100 * num_clipped / num_total:.2f}%, which are all below a "
            f"density of {Y_LIMIT_LOWER:.0e})"
        )

        bins = torch.logspace(
            torch.log10(torch.tensor(lower_limit)),
            torch.log10(torch.tensor(upper_limit)),
            steps=401,
        )

        custom_colors = [
            "#2369BD",  # darker blue
            "#006BA4",  # dark blue
            "#5F9ED1",  # light blue
            "#A2C8EC",  # very light blue
            "#ABABAB",  # gray
            "#898989",  # dark gray
            "#898989",  # darker gray
            "#FFBC79",  # light orange
            "#FF800E",  # orange
            "#C85200",  # dark orange
            "#A9373B",  # dark red
        ]

        cb_palette = sns.color_palette(custom_colors, n_colors=10, as_cmap=True)
        sns.set_palette(cb_palette)
        sns.set_style("whitegrid")

        mpl.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{bm}",
                "font.family": "serif",
                "font.serif": ["Times"],
                "font.size": 22,
                "font.weight": "bold",  # <--- Make default font bold
                "axes.labelsize": 22,
                "axes.labelweight": "bold",  # <--- Bold axis labels
                "axes.titlesize": 20,
                "axes.titleweight": "bold",  # <--- Bold title
                "legend.fontsize": 17,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "xtick.major.width": 2,  # Optional: thicker ticks
                "ytick.major.width": 2,
                "pdf.compression": 9,
            }
        )

        plt.figure(figsize=(10, 6))
        for name, perplexities in perplexity_dict.items():
            sns.histplot(
                perplexities,
                bins=bins,
                stat="density",
                label=name,
                element="step",
                alpha=0.4,
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(lower_limit, upper_limit)
        plt.ylim(Y_LIMIT_LOWER, 1)

        plt.xlabel("Perplexity", fontweight="bold")
        plt.ylabel("Probability", fontweight="bold")
        # the weight lineage and the real-data fraction go in the title, because both change the
        # collapse curve for the same input data and a figure that does not say which it is cannot
        # be compared against the other. No underscores or backslashes in here — usetex is on, so
        # the title is rendered by LaTeX.
        #
        # The run_label sits on its own second line rather than in one long title: at figsize
        # (10, 6) with font.size 22 the single-line version measured 1081px against a 1000px
        # figure, so LaTeX rendered it clipped — and since the label is the tail of the string, the
        # part that silently disappeared was exactly the fraction this is here to record. Two lines
        # take the worst case (fresh weights, three-digit fraction) to 64% of the figure width
        plt.title(
            f"Perplexity without extrapolation\n({run_label})", fontweight="bold"
        )
        plt.legend(loc="upper right")

        for spine in plt.gca().spines.values():
            spine.set_color("black")

        plt.tight_layout()

        # check if plots/ is a directory
        if not os.path.exists("plots/"):
            os.makedirs("plots/")

        # plots/ deliberately sits outside --path, so the file name is the only thing separating
        # two runs' figures. Without the run_suffix a fresh-init run, or a run with a different
        # --real_data_fraction, silently overwrites the other's figure even when the two used
        # different --path directories
        plot_stem = (
            f"plots/perplexity_histogram_bs{block_size}_{specifier_name}{run_suffix}"
        )
        plt.savefig(f"{plot_stem}.pdf")
        plt.savefig(f"{plot_stem}.png")
        plt.show()

        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the histogram under: "
            f"{TColors.HEADER}{plot_stem}.<png,pdf>{TColors.ENDC}"
        )

    # ────────────────── print the elapsed time ─────────────────────────
    # End the timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    delta = timedelta(seconds=int(elapsed_time))

    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"## {TColors.OKBLUE}{TColors.BOLD}Execution time: ")
    if days:
        print(f"{TColors.HEADER}{days} days, {hours:02}:{minutes:02}:{seconds:02}")
    else:
        print(f"{TColors.HEADER}{hours:02}:{minutes:02}:{seconds:02}")
    print(f"{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Collapse")
    parser.add_argument(
        "--device",
        "-dx",
        type=str,
        default="cuda",
        help="specifies the device to run the computations on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--training_epochs",
        "-te",
        type=int,
        default=5,
        help="specifies the number of training epochs to run",
    )
    parser.add_argument(
        "--dataset_batch_size",
        "-dbs",
        type=int,
        default=150,
        help="specifies the batch size for the dataset",
    )
    parser.add_argument(
        "--training_batch_size",
        "-tbs",
        type=int,
        default=16,
        help="specifies the batch size for the training/eval",
    )
    parser.add_argument(
        "--perplexity_batch_size",
        "-pbs",
        type=int,
        default=16,
        help="specifies the batch size for the perplexity calculation. The memory scales with "
        "batch size * sequence length * vocabulary size, i.e., ~1.25GB per sample at a "
        "sequence length of 4096 (default: 16, which needs ~22GB of the 48GB VRAM)",
    )
    parser.add_argument(
        "--skip_training",
        "-st",
        action="store_true",
        help="if set, skip the training and only evaluate the models",
    )
    parser.add_argument(
        "--num_generations",
        "-ng",
        type=int,
        default=10,
        help="specifies the number of generations to run (default: 10)",
    )
    parser.add_argument(
        "--block_size",
        "-bs",
        type=int,
        default=512,
        help="will be replaced with maximum length of input tokens from the dataset if too small",
    )
    parser.add_argument(
        "--histogram_only",
        "-ho",
        action="store_true",
        help="if set, only generate the histogram and skip the rest",
    )
    parser.add_argument(
        "--human_eval_only",
        "-heo",
        action="store_true",
        help="if set, only generate human eval samples and skip the rest",
    )
    add_model_arguments(parser)
    parser.add_argument(
        "--continue_from_generation",
        "-cfg",
        type=int,
        default=0,
        help="specifies the generation to continue from (default: 0, start from scratch)",
    )
    parser.add_argument(
        "--dataset_size",
        "-dsz",
        type=int,
        default=0,
        help="number of dataset samples to use, taken as a contiguous slice from the front of "
        "the upstream 50k dataset; 0 uses all of it. run_baseline.py and run_extrapolation.py "
        "must be given the same value, otherwise their histograms describe different data "
        "(default: 0, the whole dataset)",
    )
    parser.add_argument(
        "--real_data_fraction",
        "-rdf",
        type=float,
        default=0.0,
        help="share of every training corpus after generation 0 taken from the original human "
        "dataset instead of the previous generation's output, in [0, 1). 0 is pure self-training "
        "and collapses fastest — the models lose code generation within about two generations, "
        "which leaves run_attack.py's capability gate nothing to attack. Raising it bounds the "
        "degradation so later generations still drift but stay capable. The corpus size does not "
        "change, only its composition. Non-zero values are recorded in the perplexity cache and "
        "plot file names, and the value is shown in the plot title (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        "-sd",
        type=int,
        default=1337,
        help="seed of the whole collapse trajectory. It is threaded into the training worker and "
        "into every generation worker's sampling seed, so two runs differing only in this value "
        "are two independent collapse trajectories from identical hyperparameters — which is what "
        "run_transfer_experiment.py uses to build a second run to transfer a suffix into "
        "(default: 1337)",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2e-4,
        help="LoRA learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lora_rank",
        "-lr_r",
        type=int,
        default=16,
        help="LoRA rank r. The adapter can only move the model inside an r-dimensional "
        "subspace, so this is one of the levers on how strongly each generation collapses. "
        "Raise it (32, 64) to let a generation drift further (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        "-lr_a",
        type=int,
        default=16,
        help="LoRA alpha. The adapter contributes (alpha / r) * B @ A, so raising alpha "
        "relative to the rank scales up the per-generation delta (default: 16)",
    )
    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="inference engine for the dataset generation. vLLM is roughly an order of magnitude "
        "faster than the transformers path, because a batched generate() runs every sequence of "
        "a batch until the longest one finishes while vLLM retires a sequence as soon as it is "
        "done. 'auto' uses vLLM if it is installed (default: auto)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        "-gmu",
        type=float,
        default=0.90,
        help="fraction of each GPU vLLM may use. Everything not taken by the weights becomes KV "
        "cache, i.e. concurrent sequences (default: 0.90)",
    )
    parser.add_argument(
        "--temperature",
        "-tp",
        type=float,
        default=0.7,
        help="sampling temperature of the dataset generation. Pinned rather than inherited from "
        "the model's generation_config so that both engines sample from the same distribution "
        "(default: 0.7, Qwen2.5's own value)",
    )
    parser.add_argument(
        "--top_p",
        "-tpp",
        type=float,
        default=0.8,
        help="nucleus sampling cutoff of the dataset generation. NOTE that a cutoff below 1.0 "
        "truncates the model's own distribution, which is itself a collapse mechanism — it is "
        "what the data-space surrogate models. Use 1.0 with --top_k -1 for untruncated ancestral "
        "sampling (default: 0.8, Qwen2.5's own value)",
    )
    parser.add_argument(
        "--top_k",
        "-tpk",
        type=int,
        default=20,
        help="top-k sampling cutoff of the dataset generation, -1 disables it (default: 20, "
        "Qwen2.5's own value)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        "-gas",
        type=int,
        default=4,
        help="optimizer step granularity. The effective batch is --training_batch_size times "
        "this, and that product controls how far a generation drifts — raising -tbs while "
        "lowering this proportionally is faster at identical semantics (default: 4)",
    )
    parser.add_argument(
        "--load_in_4bit",
        "-q4",
        action="store_true",
        help="quantize the model for training. A 0.5B model is ~1GB in bf16 on a 48GB card, so "
        "this only adds a dequantization kernel to every forward and backward pass. Only set it "
        "for a --model_size / --model_specifier that does not otherwise fit — from 7b upward on a "
        "48GB card it generally does not",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        "-gc",
        action="store_true",
        help="recompute activations during the backward pass instead of keeping them. Trades a "
        "second forward pass for memory there is no shortage of at the default 0.5b, and is worth "
        "setting at the larger --model_size values",
    )
    parser.add_argument(
        "--perplexity_load_in_4bit",
        "-pq4",
        action="store_true",
        help="quantize the model that scores the perplexity. This puts quantization noise into "
        "the very statistic that is plotted, so it is off by default",
    )
    parser.add_argument(
        "--fresh_init",
        "-fi",
        action="store_true",
        help="re-initialise every generation from the pristine base model instead of continuing "
        "to fine-tune the previous generation's adapter, so that only the *data* is recursive "
        "(the 'replace' setting of the model collapse literature). Without it a run reaches the "
        "last generation with one adapter that has been trained for num_generations * "
        "training_epochs epochs, and -lr_r/-lr_a only take effect at generation 0",
    )
    parser.add_argument(
        "--training_gpus",
        "-tg",
        type=int,
        default=0,
        help="how many of the visible GPUs to train one generation on, as data parallel ranks "
        "under torchrun. The generations themselves are sequential, so this parallelises a single "
        "generation's fine-tuning. 0 uses every visible GPU, 1 forces single GPU training — which "
        "is the reference to check the data parallel path against (default: 0)",
    )
    parser.add_argument(
        "--master_port",
        "-mp",
        type=int,
        default=29500,
        help="torchrun rendezvous port for the training stage. Only needs changing when two "
        "pipelines run on the same machine, which would otherwise collide on it (default: 29500)",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="",
        help="path to save the generated datasets and models (default: current directory)",
    )
    args = parser.parse_args()
    main(**vars(args))
