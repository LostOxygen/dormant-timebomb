"""main hook to start the pitfall 1 fine-tuning"""

# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# the GPU list is resolved once, here, rather than read from the environment at the point of use:
# older unsloth releases rewrote CUDA_VISIBLE_DEVICES to a single device at import time, which
# collapsed the shard fan out to GPU 0. Capturing it above the unsloth import makes that
# version-independent. utils.devices is deliberately torch free so this does not beat unsloth to
# torch
from utils.devices import visible_devices

VISIBLE_DEVICES = visible_devices()

# kept above the stdlib imports even though the symbol is now unused: unsloth patches
# transformers/trl at import time and has to get there before them. This orchestrator no longer
# loads a model itself (build_scaled_adapter goes through peft, the generation and perplexity
# stages are subprocesses), but the import order rule is cheap to honour and expensive to
# rediscover
from unsloth import FastLanguageModel  # noqa: F401  pylint: disable=unused-import

import os
import json
import time
from datetime import timedelta
from typing import Final
import getpass
import datetime
import shutil
import argparse
import subprocess
import psutil
import pytz

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

from utils.colors import TColors
from utils.plotting import plot_perplexity_figure
from utils.utils import report_block_size
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import factor_mode_tag, mixture_suffix, mixture_tag
from utils.extrapolation import (
    METHODS,
    METHOD_LABELS,
    build_scaled_adapter,
    calibrated_factor,
    calibration_file,
    dataset_suffix,
    factor_calibration_file,
)

DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
MODEL_PATH: str = "./model_outputs/"
DATASET_PATH: str = "./generated_datasets/"
EOS_TOKEN: str = None  # will be overwritten by the tokenizer
MAX_TOKEN_LENGTH: Final[int] = None  # will be overwritten
TOKENIZER = None  # will be overwritten


def surrogate_factor_arg(value: str) -> float | str:
    """argparse type for --surrogate_factor: a number, or the literal "calibrated".

    Deliberately narrower than run_attack.py's version of this flag, which also takes "auto".
    That mode probes the surrogate on the attack's clean coding tasks and keeps the largest factor
    it still solves them at — there are no tasks here and nothing to execute, so the question it
    answers cannot be asked in this stage. The perplexity-side equivalent *is* "calibrated": the
    factor that reproduces each real checkpoint's perplexity, which is exactly what this stage's
    own --calibrate output measures.

    Args:
        value (str): the raw command line token

    Returns:
        float | str: the parsed factor, or "calibrated"

    Raises:
        argparse.ArgumentTypeError: neither a number nor "calibrated"
    """
    token = value.strip().lower()
    if token == "calibrated":
        return token
    if token == "auto":
        raise argparse.ArgumentTypeError(
            "--surrogate_factor auto is a run_attack.py mode: it picks the largest factor whose "
            "surrogate still solves the clean attack tasks, and this stage executes no code. Use "
            "'calibrated' for the perplexity-side equivalent, or pass a number."
        )
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--surrogate_factor takes a number or 'calibrated', not {value!r}"
        ) from exc


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
    path: str = "",
    model_specifier: str = "",
    model_size: str = "",
    continue_from_generation: int = 0,
    method: str = "logit",
    surrogate_factor: float | str = 0.0,
    surrogate_top_p: float = 0.0,
    dataset_size: int = 0,
    real_data_fraction: float = 0.0,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    load_in_4bit: bool = False,
    perplexity_load_in_4bit: bool = False,
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
        block_size (int): training sequence length and generation length cap. Also part of every
            artifact name, so it has to match run_baseline.py (default: 512)
        histogram_only (bool): if True, only generate the histogram and skip the rest
        path (str): path to save the generated datasets and models
        model_specifier (str): model specifier to use for the training
        model_size (str): parameter count off the Qwen2.5-Coder ladder ("0.5b" ... "32b"),
            shorthand for the matching model_specifier. Must resolve to the same model
            run_baseline.py was given, since this stage reads its generation-0 checkpoint
        continue_from_generation (int): generation to continue from (default: 0, start from scratch)
        dataset_size (int): number of dataset samples to use, taken from the front of the
            upstream 50k dataset. Must match between run_baseline.py and run_extrapolation.py
        method (str): which approximation of the later generations to use, see
            utils/extrapolation.py ("logit", "lora" or "data")
        surrogate_factor (float | str): the extrapolation factor n each generation's surrogate
            stands for. 0.0 keeps the indexing rule n = generation + 1, "calibrated" reads the
            factors utils/evaluate_perplexity.py --calibrate fitted against the real checkpoints,
            and an explicit number is accepted for a single-generation run only — one factor for
            every generation would make every generation the same surrogate
        surrogate_top_p (float): p_1 of the data-space surrogate. 0.0 reads it from the
            calibration that calibrate_surrogate.py wrote
        real_data_fraction (float): the --real_data_fraction of the run_baseline.py run this
            stage's histogram is compared against. It does not change what this stage computes —
            the surrogate is built from model_0, which every mixture shares — it only namespaces
            this stage's own artifacts so a mixed run's numbers are not filed under, or replotted
            as, an unmixed run's
        temperature (float): sampling temperature of the generation. Has to match
            run_baseline.py, otherwise the extrapolated histograms are compared against a
            baseline that was sampled differently
        top_p (float): nucleus sampling cutoff of the generation, same constraint. The "data"
            method replaces this with its own per generation schedule, which is the whole point
            of that method
        top_k (int): top-k sampling cutoff of the generation, -1 disables it. Same constraint
        load_in_4bit (bool): quantize the generating models. Off by default so that the
            approximations run at the same precision as the baseline they are compared against
        perplexity_load_in_4bit (bool): quantize the model that scores the perplexity. Has to
            match run_baseline.py, since both stages' histograms are plotted against each other

    Returns:
        None
    """
    start_time = time.time()

    # every artifact of a method carries its own suffix, so that the three methods can be run
    # against the same baseline and compared without overwriting each other
    suffix = dataset_suffix(method)

    if not 0.0 <= real_data_fraction < 1.0:
        raise SystemExit(
            f"--real_data_fraction must be in [0, 1), got {real_data_fraction}. At 1.0 every "
            f"generation would train on the human corpus alone and nothing would collapse."
        )
    # the run-level tag, for the artifacts that span the whole run: the perplexity cache and the
    # figure. plots/ sits outside --path, so the file name is the only thing keeping the figure of
    # an -rdf 0.3 comparison apart from an -rdf 0 one even when the two used separate --path
    # directories. Empty at 0, so an existing pure self-training run keeps exactly the names it
    # always had and -ho/-st keep finding them.
    #
    # Note what this does *not* reach: no model path depends on it. This stage only ever loads
    # model_0_bs{bs}_{name}, and generation 0 trains on the human corpus under every mixture, so
    # that checkpoint is shared — the same reason run_attack.py's surrogate anchor takes no
    # fraction. The per-generation datasets below are named with mixture_suffix() instead, which
    # returns "" for generation 0 for the same reason; see utils/naming.py
    data_suffix = mixture_tag(real_data_fraction)
    # the factor rule's tag, and the reason it exists: --surrogate_factor changes what every
    # corpus this stage generates *contains*, so a calibrated run and a default one would
    # otherwise write the same generated_dataset_{g}, the same perplexity cache and the same
    # figure — and -ho would replot whichever of them happened to be on disk. Empty for the
    # n = g + 1 rule, so an existing run keeps exactly the names it has
    factor_tag = factor_mode_tag(surrogate_factor)
    # what the run-level artifacts are named by: the method this run approximates with, then the
    # factor rule it approximates under, then the mixture it is filed against. The method and the
    # factor identify the surrogate; the mixture identifies the baseline run it is compared to
    run_suffix = f"{suffix}{factor_tag}{data_suffix}"

    # a single number cannot index a *sweep*: every generation would build the same surrogate, so
    # the corpora would differ only by sampling noise and the collapse curve would be flat by
    # construction. The generations this run covers are range(continue_from_generation,
    # num_generations), so one of them is the case where a fixed n is a meaningful thing to ask for
    covered = range(continue_from_generation, num_generations)
    if not isinstance(surrogate_factor, str) and float(surrogate_factor) > 0 and len(covered) > 1:
        raise SystemExit(
            f"--surrogate_factor {surrogate_factor:g} is a single factor, but this run covers "
            f"generations {covered.start}..{covered.stop - 1}. One factor for every generation "
            f"builds the same surrogate {len(covered)} times, so the histograms would differ only "
            f"by sampling noise.\nRun one generation at a time (e.g. -cfg {covered.start} -ng "
            f"{covered.start + 1}), or use --surrogate_factor calibrated, which resolves a factor "
            f"per generation."
        )

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

    # set the model specifier. --model_size is shorthand for a repo id off the Qwen2.5-Coder
    # ladder; whichever way it is given it has to resolve to the same model run_baseline.py used,
    # because this stage loads that run's generation-0 checkpoint by name
    model_specifier = resolve_model_specifier(model_size, model_specifier)
    specifier_name = model_specifier.split("/")[-1]
    # the ladder rung, for the parameters banner — same reason as in run_baseline.py, and here it is
    # what a log shows when the two stages' histograms turn out to be of different models
    size_label = model_size_label(model_specifier) or "outside the --model_size ladder"

    # allow tf32 for the fp32 fallback matmuls. The L40S runs tf32 at a multiple of the fp32 rate
    # and the bf16 path is unaffected
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # load the tokenizer to count the tokens of the dataset. AutoTokenizer instead of
    # FastLanguageModel.from_pretrained, which would load a whole quantized model just to hand
    # back its tokenizer — and would keep that model resident on GPU 0 for the entire run,
    # because binding it to `_` does not free it
    tokenizer = AutoTokenizer.from_pretrained(model_specifier)
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
    # is no longer silently raised to the longest response. run_baseline.py resolves it the same
    # way through the same helper, which is what keeps the two stages' artifact names lined up
    global MAX_TOKEN_LENGTH
    MAX_TOKEN_LENGTH = max(token_counts)
    block_size = report_block_size(block_size, token_counts)

    # have a nice system status print
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + "#" * (shutil.get_terminal_size().columns - 23)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: "
        + str(
            datetime.datetime.now(tz=pytz.timezone("Europe/Berlin")).strftime(
                "%A, %d. %B %Y %I:%M%p"
            )
        )
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
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Specifier{TColors.ENDC}: {model_specifier}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Model Size{TColors.ENDC}: {size_label}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Specifier{TColors.ENDC}: {DATASET_SPECIFIER}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Method{TColors.ENDC}: {method} "
        f"({METHOD_LABELS[method]}), artifact suffix: {suffix}"
    )
    # the surrogate itself is unmixed whatever this says — printed so that a figure produced
    # against a mixed baseline carries the reason its curve does not line up with stage 1's
    if real_data_fraction > 0:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Real data fraction{TColors.ENDC}: "
            f"{real_data_fraction:g} — names this stage's artifacts only. The extrapolation is "
            f"built from model_0, which every mixture shares, so it approximates the untempered "
            f"self-training trajectory and is only a like-for-like comparison against an "
            f"-rdf 0 run of run_baseline.py"
        )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Number of Generations{TColors.ENDC}: {num_generations}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Block size{TColors.ENDC}: {block_size}")
    # printed because unsloth silently rewrites CUDA_VISIBLE_DEVICES to a single device at import,
    # which used to collapse this list to [0] without any sign of it in the output
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Sharding generation/perplexity across{TColors.ENDC}: "
        f"{len(VISIBLE_DEVICES)} GPU(s) {VISIBLE_DEVICES}"
    )
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

    # ── resolve the data-space surrogate's single parameter ──
    # p_1 is not a hyperparameter: it is defined as the truncation that reproduces the real
    # model_0, so it has to come from a calibration. Guessing it would make every generation of
    # the surrogate an arbitrary number rather than an approximation of anything
    surrogate_p1 = surrogate_top_p
    if method == "data" and surrogate_p1 <= 0.0:
        calibration_path = calibration_file(DATASET_PATH, block_size, specifier_name)
        if not os.path.isfile(calibration_path):
            raise FileNotFoundError(
                f"the 'data' method needs a calibrated p_1, but {calibration_path} does not "
                "exist. Run 'python -m utils.calibrate_surrogate --block_size "
                f"{block_size} --model_specifier {model_specifier}' first, or pass "
                "--surrogate_top_p explicitly to skip the calibration"
            )
        with open(calibration_path, "r", encoding="utf-8") as calibration_handle:
            calibration = json.load(calibration_handle)
        surrogate_p1 = float(calibration["surrogate_top_p"])
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Surrogate p_1{TColors.ENDC}: {surrogate_p1} "
            f"(calibrated, target log-perplexity "
            f"{calibration['target_mean_log_perplexity']:.4f})"
        )
        if not calibration.get("bracketed", True):
            print(
                f"## {TColors.WARNING}Warning{TColors.ENDC}: the calibration did not bracket "
                "its target, so p_1 is the nearest grid endpoint rather than a fit"
            )
    elif method == "data":
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Surrogate p_1{TColors.ENDC}: {surrogate_p1} "
            f"{TColors.WARNING}(given on the command line, not calibrated){TColors.ENDC}"
        )
    # ── resolve the extrapolation factor of every generation ──
    # The default is the indexing rule n = g + 1, which names the generation being approximated but
    # is not calibrated against anything: measured on this repo's 0.5b run the tilt's perplexity
    # rises ~10x per unit of n where the real collapse rises 4.2x over ten generations. The
    # calibration is the answer to that — the factor that actually reproduces each checkpoint,
    # measured where checkpoints exist and predicted by the fitted law where they do not — and it
    # is read here, once, rather than in the shard workers: they would all read the same file, and
    # the "lora" adapter has to be built from the same number the workers tilt with
    factor_calibration = None
    if surrogate_factor == "calibrated":
        factor_path = factor_calibration_file(
            DATASET_PATH, block_size, specifier_name, data_suffix
        )
        if not os.path.isfile(factor_path):
            raise SystemExit(
                f"{TColors.FAIL}--surrogate_factor calibrated needs a calibration{TColors.ENDC} "
                f"and {factor_path} does not exist. Produce it with:\n"
                f"  python -m utils.evaluate_perplexity -p {path or '.'} -bs {block_size} "
                f"-ng {num_generations} --calibrate"
            )
        with open(factor_path, "r", encoding="utf-8") as factor_handle:
            factor_calibration = json.load(factor_handle)

    def factor_of(gen_id: int) -> float:
        """The factor generation `gen_id`'s surrogate is built at, under this run's rule."""
        if factor_calibration is not None:
            return calibrated_factor(factor_calibration, gen_id)
        if not isinstance(surrogate_factor, str) and float(surrogate_factor) > 0:
            return float(surrogate_factor)
        return float(gen_id + 1)

    if factor_tag:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Extrapolation factor{TColors.ENDC}: "
            + (
                "calibrated — "
                if factor_calibration is not None
                else "given on the command line — "
            )
            + ", ".join(f"n({gen}) = {factor_of(gen):g}" for gen in covered)
            + f"  (artifacts tagged {factor_tag})"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Extrapolation factor{TColors.ENDC}: "
            f"n = generation + 1 (the indexing rule)"
        )
    print("#" * shutil.get_terminal_size().columns + "\n")

    # resolve the GPUs to shard the work across, once for both the generation and the perplexity
    # stage
    # resolved from VISIBLE_DEVICES, captured at module scope — see utils.devices for why the
    # environment is not read at the point of use
    devices = VISIBLE_DEVICES if str(device).startswith("cuda") else [0]

    if not skip_training:
        # print information about the dataset
        print(f"Max token count: {max(token_counts)}")
        print(f"Avg token count: {sum(token_counts) / len(token_counts)}")
        print(f"Min token count: {min(token_counts)}")
        print(f"Original dataset length: {len(original_dataset)}\n")
        original_dataset = original_dataset.map(format_prompt, batched=True)
        original_dataset.save_to_disk(DATASET_PATH + f"original_dataset_bs{block_size}")

        # preprocess the dataset
        # deliberately not tagged with the mixture: this is the human corpus, which is what
        # generation 0 is scored against under every fraction. utils/calculate_perplexity.py reads
        # it back untagged for exactly that reason
        chunked_dataset = original_dataset
        chunked_dataset.save_to_disk(
            DATASET_PATH + f"chunked_dataset_bs{block_size}_{specifier_name}{suffix}"
        )

        # the generation workers only ever read the *instructions*, which are the same for every
        # generation, so the shards are written once here instead of being rewritten inside the
        # generation loop. contiguous=True, because datasets defaults to strided sharding
        # (indices index, index + n, index + 2n, ...) and the shards are merged back in shard
        # order afterwards. Strided shards would therefore reorder the merged dataset as a
        # function of the *number of GPUs*, which would make a 4-GPU run's histogram describe a
        # differently ordered dataset than a 1-GPU run's. Contiguous shards reassemble into the
        # original order for any device count.
        # Untagged for the same reason as chunked_dataset above — these hold the human
        # instructions, which no mixture touches, so every fraction reuses one set of shards
        for shard_id in range(len(devices)):
            original_dataset.shard(
                num_shards=len(devices), index=shard_id, contiguous=True
            ).save_to_disk(
                DATASET_PATH
                + f"base_subdataset_bs{block_size}_{specifier_name}{suffix}_shard{shard_id}"
            )

        for gen_id in range(num_generations):
            # check if generations need to be skipped if continue_from_generation > 0
            if gen_id < continue_from_generation:
                continue

            # the corpus this generation produces is named after the generation that produced it,
            # which is what makes utils/calculate_perplexity.py find it again: that worker reads
            # generation i's corpus as generated_dataset_{i - 1} + mixture_suffix(fraction, i - 1).
            # Generation 0 is therefore untagged on both sides — n = 1 reproduces the real model_0
            # anchor, which every mixture shares, so its corpus is shared too
            gen_mix = mixture_suffix(real_data_fraction, gen_id)

            # ───────────────────── build the alpha scaled adapter (lora only) ────────────────
            # a LoRA layer adds (alpha / r) * B @ A to the frozen base weight, so scaling alpha
            # by n scales the whole fine-tuning delta by n and yields the weights
            # W_base + n * (W_collapsed - W_base). This is built once per generation here rather
            # than inside the shard subprocesses, which would race over the same directory
            # neither the model_0 it is built from nor the scaled adapter itself carries a mixture
            # tag: the adapter is a pure function of model_0, so it is identical for every
            # fraction, and naming it after one would claim a dependence that does not exist
            # this generation's factor under whichever rule the run was given
            gen_factor = factor_of(gen_id)
            adapter_path = ""
            if method == "lora":
                # the factor is in the adapter's own name, so a calibrated n and the indexing
                # rule's n cannot land on the same directory. ":g" and not str(): the calibrated
                # factors are floats, and an integral one has to render as "2" rather than "2.0"
                # or every scaled adapter already on disk is renamed
                adapter_path = (
                    f"{MODEL_PATH}model_scaled_n{gen_factor:g}_bs{block_size}_{specifier_name}"
                )
                build_scaled_adapter(
                    adapter_path=f"{MODEL_PATH}model_0_bs{block_size}_{specifier_name}",
                    factor=gen_factor,
                    output_path=adapter_path,
                )
                print(
                    f"## {TColors.OKBLUE}{TColors.BOLD}Scaled adapter{TColors.ENDC}: "
                    f"alpha x {gen_factor:g} -> {adapter_path}"
                )
                if factor_tag:
                    # utils/evaluate_perplexity.py builds this name itself, at n = g + 1, when it
                    # scores the lora surrogate. Under any other rule it looks for an adapter this
                    # run does not write, so say so here rather than let it fail three stages later
                    print(
                        f"##   {TColors.WARNING}utils/evaluate_perplexity.py --method lora looks "
                        f"for model_scaled_n{gen_id + 1}_* (the indexing rule) and will not find "
                        f"this one{TColors.ENDC}"
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
                        "utils.generate_dataset_extrapolation",
                        "--block_size",
                        str(block_size),
                        "--specifier_name",
                        specifier_name,
                        # the base model of the tilt. --specifier_name is only the short name the
                        # artifacts are filed under, so without this the worker would fall back to
                        # its own default and a --model_size run would tilt away from the wrong
                        # model while naming everything after the right one
                        "--model_specifier",
                        model_specifier,
                        "--dataset_batch_size",
                        str(dataset_batch_size),
                        "--generation",
                        str(gen_id),
                        "--shard_id",
                        str(shard_id),
                        "--method",
                        method,
                        "--adapter_path",
                        adapter_path,
                        "--surrogate_top_p",
                        str(surrogate_p1),
                        # already resolved: the worker applies the number, it does not re-derive
                        # the rule, and it names the corpus it writes with the tag
                        "--surrogate_factor",
                        str(gen_factor),
                        "--factor_tag",
                        factor_tag,
                        "--temperature",
                        str(temperature),
                        "--top_p",
                        str(top_p),
                        "--top_k",
                        str(top_k),
                        "--path",
                        str(path),
                        # names the shard it writes; the base_subdataset it reads is the untagged
                        # human corpus
                        "--real_data_fraction",
                        str(real_data_fraction),
                    ]
                    + (["--load_in_4bit"] if load_in_4bit else []),
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
                    f"{gen_id} (method {method}). See the subprocess output above for the actual "
                    "error"
                )

            # merge all the subdatasets to one single dataset again
            merged_dataset = concatenate_datasets(
                [
                    Dataset.load_from_disk(
                        DATASET_PATH
                        + f"subdataset_{gen_id}_bs{block_size}_{specifier_name}{suffix}"
                        + f"{factor_tag}{gen_mix}_shard{shard_id}"
                    )
                    for shard_id in range(len(devices))
                ]
            )
            merged_dataset.save_to_disk(
                DATASET_PATH
                + f"generated_dataset_{gen_id}_bs{block_size}_{specifier_name}{suffix}"
                + f"{factor_tag}{gen_mix}"
            )

    # ────────────────── evaluate the models' perplexity and other metrics ─────────────────────────
    # iterate over every model and the generated dataset and calculate the perplexity
    # for the perplexity, every datapoint i.e., the generated answer for every question
    # is evaluated to get the probability for a given perplexity over the whole dataset
    if not histogram_only:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Calculate Perplexity{TColors.ENDC}")
        perplexity_dict = {}
        all_perplexities = []

        # the datasets are split into one shard per GPU and every shard is processed by its
        # own subprocess. Each subprocess handles all generations of its shard, so the model
        # only has to be loaded once per GPU. Afterwards the per-shard results are merged.
        # `devices` was resolved once above from VISIBLE_DEVICES, which is captured before the
        # unsloth import — re-deriving it from the environment here would see the single-device
        # rewrite and run every shard on GPU 0
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Using {len(devices)} GPU(s) for the "
            f"perplexity calculation{TColors.ENDC}"
        )

        # has to match what utils/calculate_perplexity.py writes, which appends the run-level
        # mixture tag after the dataset suffix
        shard_files = [
            DATASET_PATH
            + f"perplexity_dict_bs{block_size}_{specifier_name}{run_suffix}_shard{shard_id}.pt"
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
                    model_specifier,
                    "--perplexity_batch_size",
                    str(perplexity_batch_size),
                    "--num_generations",
                    str(num_generations),
                    "--shard_id",
                    str(shard_id),
                    "--num_shards",
                    str(len(devices)),
                    "--dataset_suffix",
                    suffix,
                    # composes with --dataset_suffix, like --real_data_fraction below: the
                    # generated corpora carry the factor rule, the human corpus of generation 0
                    # does not — it is the same text under every factor
                    "--factor_tag",
                    factor_tag,
                    "--path",
                    str(path),
                    # composes with --dataset_suffix rather than replacing it: it reads
                    # generated_dataset_{i-1}{suffix} for every generation i, so it needs the
                    # fraction to name each one, and it names its own shard file with it
                    "--real_data_fraction",
                    str(real_data_fraction),
                ]
                + (["--load_in_4bit"] if perplexity_load_in_4bit else []),
            )
            process_list.append(process)

        # wait for all processes to finish
        for process in process_list:
            process.wait()

        # every shard has to be there, otherwise the merged perplexities would be incomplete
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
            perplexity for values in perplexity_dict.values() for perplexity in values
        ]

        # save the perplexity dict to a file
        torch.save(
            perplexity_dict,
            DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}{run_suffix}.pt",
        )  # save the dict to a file
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the perplexity dict under: "
            f"{TColors.HEADER}{DATASET_PATH}perplexity_dict_bs{block_size}_{specifier_name}"
            f"{run_suffix}.pt{TColors.ENDC}"
        )
        # save the all_perplexities list to a file
        torch.save(
            all_perplexities,
            DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}{run_suffix}.pt",
        )  # save the list to a file
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the all_perplexities list under: "
            f"{TColors.HEADER}{DATASET_PATH}all_perplexities_bs{block_size}_{specifier_name}"
            f"{run_suffix}.pt{TColors.ENDC}"
        )
    else:
        # load the perplexity dict and all_perplexities list from the files. -ho therefore needs
        # the same --method and --real_data_fraction the cache was produced with, and says so
        # rather than replotting whichever configuration happens to be cached under this name
        cached_dict = (
            DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}{run_suffix}.pt"
        )
        if not os.path.exists(cached_dict):
            raise FileNotFoundError(
                f"{cached_dict} does not exist. --histogram_only replots the cache of a run with "
                f"the same --block_size, --model_specifier, --method ({method} here), "
                f"--surrogate_factor ({surrogate_factor!r} here, tagged "
                f"{factor_tag or 'nothing'}) and --real_data_fraction ({real_data_fraction:g} here)"
            )
        perplexity_dict = torch.load(cached_dict)
        all_perplexities = torch.load(
            DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}{run_suffix}.pt"
        )

    # ────────────────── plot the perplexity histogram ─────────────────────────
    # same figure as stage 1's and as utils/evaluate_perplexity.py's, drawn by utils/plotting.py:
    # the histograms on top, the median per generation below them. This stage only decides what
    # goes into it — the corpora its surrogate generated — and what the title says
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Plotting Perplexity Histogram{TColors.ENDC}"
    )

    # plots/ sits outside --path, so this file name is the only thing keeping the figures of two
    # differently filed runs apart even when they used separate --path directories
    plot_stem = f"plots/perplexity_histogram_bs{block_size}_{specifier_name}{run_suffix}"

    # the fraction goes on its own second line, and only when it is set, so the figure of a plain
    # run is unchanged. No underscores or backslashes: usetex renders the title
    title = f"Perplexity with {METHOD_LABELS[method]}"
    # the factor rule belongs in the title for the same reason the mixture does: it changes every
    # histogram in the figure at once, so a figure that does not name it is not attributable
    if factor_tag:
        title += (
            "\n(calibrated factors)"
            if factor_calibration is not None
            else f"\n(fixed factor n = {factor_of(covered.start):g})"
        )
    if real_data_fraction > 0:
        title += f"\n(filed against real data fraction {real_data_fraction:g})"

    plot_perplexity_figure(
        perplexity_dict,
        plot_stem,
        title=title,
        primary_label="generated corpora",
        median_ylabel="corpus perplexity\n(median)",
        show=True,
    )

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
    parser = argparse.ArgumentParser(description="Model Collapse Extrapolation")
    parser.add_argument(
        "--device",
        "-dx",
        type=str,
        default="cpu",
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

    add_model_arguments(parser)
    parser.add_argument(
        "--continue_from_generation",
        "-cfg",
        type=int,
        default=0,
        help="specifies the generation to continue from (default: 0, start from scratch)",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="logit",
        choices=METHODS,
        help="how to approximate the later generations without training them. 'logit': "
        "base + n * (collapsed - base) in logit space at every decoding step. 'lora': the same "
        "first order step in weight space, i.e. the collapse adapter with alpha * n, which is "
        "cheaper and yields an actual model. 'data': the base model sampled with a support that "
        "is truncated once per generation, which imitates the resampling that drives collapse "
        "instead of the drift it causes (default: logit)",
    )
    parser.add_argument(
        "--surrogate_factor",
        "-sf",
        type=surrogate_factor_arg,
        default=0.0,
        help="the extrapolation factor n every generation's surrogate stands for. The default of "
        "0.0 keeps the indexing rule n = generation + 1, which names the generation being "
        "approximated but is not calibrated against anything. 'calibrated' reads the factors "
        "utils/evaluate_perplexity.py --calibrate fitted against the real checkpoints' perplexity "
        "— measured for the generations it covered, predicted by the fitted law for the rest — so "
        "this stage approximates the same generation run_attack.py -sf calibrated attacks. An "
        "explicit number is accepted for a single-generation run only (-cfg N -ng N+1), since one "
        "factor for every generation builds the same surrogate every time. Anything other than the "
        "default tags this run's corpora, cache and figure, so the two rules do not overwrite each "
        "other (default: 0.0)",
    )
    parser.add_argument(
        "--surrogate_top_p",
        "-stp",
        type=float,
        default=0.0,
        help="p_1 of the data-space surrogate, i.e. the top-p that reproduces the real model_0. "
        "Generation n is then sampled with p_1 ** n. The default of 0.0 reads the value that "
        "calibrate_surrogate.py fitted; passing it explicitly skips the calibration and is only "
        "meant for quick experiments ('data' method only)",
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
        help="the --real_data_fraction of the run_baseline.py run this stage is compared against, "
        "in [0, 1). It namespaces this stage's own artifacts only — the generated corpora of "
        "generations 1+, the perplexity cache and the figure — so a comparison against a mixed run "
        "is not filed under, or replotted as, an unmixed one. It does not change what is computed: "
        "the surrogate is built from model_0, which every mixture shares, so this stage always "
        "approximates the untempered self-training trajectory and is only a like-for-like "
        "comparison against an -rdf 0 baseline (default: 0.0)",
    )
    parser.add_argument(
        "--temperature",
        "-tp",
        type=float,
        default=0.7,
        help="sampling temperature of the generation. Has to match run_baseline.py, otherwise "
        "the extrapolated histograms are compared against a differently sampled baseline "
        "(default: 0.7, Qwen2.5's own value)",
    )
    parser.add_argument(
        "--top_p",
        "-tpp",
        type=float,
        default=0.8,
        help="nucleus sampling cutoff of the generation, same constraint as --temperature. The "
        "'data' method replaces this with its own per generation schedule (default: 0.8)",
    )
    parser.add_argument(
        "--top_k",
        "-tpk",
        type=int,
        default=20,
        help="top-k sampling cutoff of the generation, -1 disables it. Same constraint as "
        "--temperature (default: 20)",
    )
    parser.add_argument(
        "--load_in_4bit",
        "-q4",
        action="store_true",
        help="quantize the generating models. Off by default so the approximations run at the "
        "same precision as the baseline they are compared against",
    )
    parser.add_argument(
        "--perplexity_load_in_4bit",
        "-pq4",
        action="store_true",
        help="quantize the model that scores the perplexity. Has to match the flag "
        "run_baseline.py was run with, since both stages' histograms are plotted against each "
        "other",
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
