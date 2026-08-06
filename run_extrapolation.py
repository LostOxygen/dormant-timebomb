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
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

from utils.colors import TColors
from utils.plotting import visible_perplexity_range
from utils.utils import report_block_size
from utils.extrapolation import (
    METHODS,
    METHOD_LABELS,
    build_scaled_adapter,
    calibration_file,
    dataset_suffix,
)

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
    continue_from_generation: int = 0,
    method: str = "logit",
    surrogate_top_p: float = 0.0,
    dataset_size: int = 0,
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
        continue_from_generation (int): generation to continue from (default: 0, start from scratch)
        dataset_size (int): number of dataset samples to use, taken from the front of the
            upstream 50k dataset. Must match between run_baseline.py and run_extrapolation.py
        method (str): which approximation of the later generations to use, see
            utils/extrapolation.py ("logit", "lora" or "data")
        surrogate_top_p (float): p_1 of the data-space surrogate. 0.0 reads it from the
            calibration that calibrate_surrogate.py wrote
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

    # set the model specifier
    if model_specifier != "":
        global MODEL_SPECIFIER
        MODEL_SPECIFIER = model_specifier
    specifier_name = MODEL_SPECIFIER.split("/")[-1]

    # allow tf32 for the fp32 fallback matmuls. The L40S runs tf32 at a multiple of the fp32 rate
    # and the bf16 path is unaffected
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
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Specifier{TColors.ENDC}: {MODEL_SPECIFIER}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Specifier{TColors.ENDC}: {DATASET_SPECIFIER}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Method{TColors.ENDC}: {method} "
        f"({METHOD_LABELS[method]}), artifact suffix: {suffix}"
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
                f"{block_size} --model_specifier {MODEL_SPECIFIER}' first, or pass "
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
        # original order for any device count
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

            # ───────────────────── build the alpha scaled adapter (lora only) ────────────────
            # a LoRA layer adds (alpha / r) * B @ A to the frozen base weight, so scaling alpha
            # by n scales the whole fine-tuning delta by n and yields the weights
            # W_base + n * (W_collapsed - W_base). This is built once per generation here rather
            # than inside the shard subprocesses, which would race over the same directory
            adapter_path = ""
            if method == "lora":
                adapter_path = (
                    f"{MODEL_PATH}model_scaled_n{gen_id + 1}_bs{block_size}_{specifier_name}"
                )
                build_scaled_adapter(
                    adapter_path=f"{MODEL_PATH}model_0_bs{block_size}_{specifier_name}",
                    factor=gen_id + 1,
                    output_path=adapter_path,
                )
                print(
                    f"## {TColors.OKBLUE}{TColors.BOLD}Scaled adapter{TColors.ENDC}: "
                    f"alpha x {gen_id + 1} -> {adapter_path}"
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
                        "--temperature",
                        str(temperature),
                        "--top_p",
                        str(top_p),
                        "--top_k",
                        str(top_k),
                        "--path",
                        str(path),
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
                        + f"_shard{shard_id}"
                    )
                    for shard_id in range(len(devices))
                ]
            )
            merged_dataset.save_to_disk(
                DATASET_PATH
                + f"generated_dataset_{gen_id}_bs{block_size}_{specifier_name}{suffix}"
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

        shard_files = [
            DATASET_PATH
            + f"perplexity_dict_bs{block_size}_{specifier_name}{suffix}_shard{shard_id}.pt"
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
                    "--dataset_suffix",
                    suffix,
                    "--path",
                    str(path),
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
            DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}{suffix}.pt",
        )  # save the dict to a file
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the perplexity dict under: "
            f"{TColors.HEADER}{DATASET_PATH}perplexity_dict_bs{block_size}_{specifier_name}{suffix}"
            f".pt{TColors.ENDC}"
        )
        # save the all_perplexities list to a file
        torch.save(
            all_perplexities,
            DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}{suffix}.pt",
        )  # save the list to a file
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the all_perplexities list under: "
            f"{TColors.HEADER}{DATASET_PATH}all_perplexities_bs{block_size}_{specifier_name}{suffix}"
            f".pt{TColors.ENDC}"
        )
    else:
        # load the perplexity dict and all_perplexities list from the files
        perplexity_dict = torch.load(
            DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}{suffix}.pt"
        )
        all_perplexities = torch.load(
            DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}{suffix}.pt"
        )

    # ────────────────── plot the perplexity histogram ─────────────────────────
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Plotting Perplexity Histogram{TColors.ENDC}"
    )

    # scale the x-axis to the range which actually contains visible data. The tails are so
    # heavy that even a 99.9% quantile still reaches 1e10, but those bins are drawn below the
    # lower y-limit and only leave the right side of the plot empty
    lower_limit, upper_limit, num_clipped, num_total = visible_perplexity_range(
        perplexity_dict, Y_LIMIT_LOWER
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Plot range{TColors.ENDC}: "
        f"[{lower_limit:.0e}, {upper_limit:.0e}] (clipping {num_clipped} of {num_total} "
        f"perplexities, {100 * num_clipped / num_total:.2f}%, which are all below a density "
        f"of {Y_LIMIT_LOWER:.0e})"
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
    plt.title(f"Perplexity with {METHOD_LABELS[method]}", fontweight="bold")
    plt.legend(loc="upper right")

    for spine in plt.gca().spines.values():
        spine.set_color("black")

    plt.tight_layout()

    # check if plots/ is a directory
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    plt.savefig(f"plots/perplexity_histogram_bs{block_size}_{specifier_name}{suffix}.pdf")
    plt.savefig(f"plots/perplexity_histogram_bs{block_size}_{specifier_name}{suffix}.png")
    plt.show()

    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Saved the histogram under: "
        f"{TColors.HEADER}plots/perplexity_histogram_bs{block_size}_{specifier_name}{suffix}"
        f".<png,pdf>{TColors.ENDC}"
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

    parser.add_argument(
        "--model_specifier",
        "-ms",
        type=str,
        default="unsloth/Qwen2.5-Coder-0.5B-Instruct",
        help="model specifier to use for the training (def: unsloth/Qwen2.5-Coder-0.5B-Instruct)",
    )
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
