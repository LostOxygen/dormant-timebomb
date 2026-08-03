"""main hook to start the pitfall 1 fine-tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import time
from datetime import timedelta
from typing import Final
import getpass
import datetime
import argparse
import subprocess
import psutil

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets

from utils.colors import TColors
from utils.plotting import visible_perplexity_range

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
    block_size: int = 64,
    histogram_only: bool = False,
    human_eval_only: bool = False,
    path: str = "",
    model_specifier: str = "",
    continue_from_generation: int = 0,
    dataset_size: int = 10000,
    learning_rate: float = 2e-5,
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
        continue_from_generation (int): generation to continue from (default: 0, start from scratch)
        dataset_size (int): number of dataset samples to use, taken from the front of the
            upstream 50k dataset. Must match between run_baseline.py and run_extrapolation.py
        learning_rate (float): full fine-tuning learning rate. Much smaller than a LoRA rate
            would be, since every weight is updated directly

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

    # set the model specifier
    if model_specifier != "":
        global MODEL_SPECIFIER
        MODEL_SPECIFIER = model_specifier
    specifier_name = MODEL_SPECIFIER.split("/")[-1]

    # load the tokenizer to count to tokens of the dataset. The model itself is discarded, but
    # 4-bit is off here too so that no load in this project quantizes anything
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_SPECIFIER,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
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

    # gather information about the dataset
    token_counts = []
    for data in tqdm(original_dataset, desc="Calculating token counts"):
        inputs = tokenizer(
            data["response"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # count the tokens
        token_count = inputs["input_ids"].shape[1]
        token_counts.append(token_count)

    # set the block size
    global MAX_TOKEN_LENGTH
    MAX_TOKEN_LENGTH = max(token_counts)
    block_size = MAX_TOKEN_LENGTH if MAX_TOKEN_LENGTH > block_size else block_size

    # have a nice system status print
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + "#" * (os.get_terminal_size().columns - 23)
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
        + "#" * (os.get_terminal_size().columns - 14)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Model Specifier{TColors.ENDC}: {MODEL_SPECIFIER}"
    )
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
    print("#" * os.get_terminal_size().columns + "\n")

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
    # the dataloader is later used for the generation of the new dataset
    # chunked_dataloader = DataLoader(
    #     chunked_dataset.with_format("torch"),
    #     batch_size=dataset_batch_size,
    # )

    if not skip_training:
        # ───────────────────────── start the actual finetuning ──────────────────────────────
        # iterte over two loops: first the model training and then the dataset generation
        # the model is trained for N times and after each training the dataset
        # is generated from the new model
        for gen_id in range(num_generations):
            # check if generations need to be skipped if continue_from_generation > 0
            if gen_id < continue_from_generation:
                continue
            # load the model for *full* fine-tuning.
            #
            # No LoRA. A rank-16 adapter on the attention and MLP projections can only move the
            # model inside a low dimensional subspace and leaves the embeddings, the norms and
            # the head untouched, which damps how far one generation can drift from the last
            # one. Since the whole point of the pipeline is to let that drift accumulate,
            # adapting only a slice of the weights suppresses the very effect being measured.
            # Full fine-tuning lets every parameter move.
            #
            # This also rules out 4-bit: quantized weights cannot be trained, so the base has to
            # be loaded at full precision. That is the memory cost of dropping LoRA.
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_SPECIFIER
                if gen_id == 0
                else f"{MODEL_PATH}model_{gen_id - 1}_bs{block_size}_{specifier_name}",
                max_seq_length=block_size,
                dtype=None,
                load_in_4bit=False,
                full_finetuning=True,
            )

            # load the dataset
            if gen_id > 0:
                # if the first training iteration is done, load the generated dataset from the disk
                dataset = Dataset.load_from_disk(
                    DATASET_PATH
                    + f"generated_dataset_{gen_id - 1}_bs{block_size}_{specifier_name}"
                )
                dataset = dataset.map(format_prompt, batched=True)
            else:
                # for first iteration (gen_id = 0) take the original dataset
                dataset = chunked_dataset

            # for the first model the original dataset is used, then the generated dataset
            # is used for the next models
            dataset_train, dataset_val = make_splits(dataset)

            # for some stats
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

            data_collator = DataCollatorForLanguageModeling(TOKENIZER, mlm=False)

            # create a trainer to train the model
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset_train,
                eval_dataset=dataset_val,
                # formatting_func=format_prompt,
                dataset_text_field="text",
                max_seq_length=block_size,
                dataset_num_proc=8,
                data_collator=data_collator,
                packing=True,  # Can make training 5x faster for short sequences.
                args=TrainingArguments(
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    num_train_epochs=training_epochs,
                    per_device_train_batch_size=training_batch_size,
                    per_device_eval_batch_size=training_batch_size,
                    # full fine-tuning needs a much smaller step than LoRA: 2e-4 updates every
                    # weight of the model directly and diverges within a few steps, which would
                    # look like a very fast collapse but is just a broken optimizer
                    learning_rate=learning_rate,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    # recomputing activations instead of storing them buys back some of the
                    # memory that full fine-tuning costs over LoRA
                    gradient_checkpointing=True,
                    seed=1337,
                    output_dir="outputs",
                    report_to="none",
                ),
            )

            # train the model
            trainer_stats = trainer.train()
            # metrics = trainer.evaluate()
            # print(
            #     f"## {TColors.OKBLUE}{TColors.BOLD}Loss: {TColors.ENDC}{metrics["eval_loss"]:.4f}"
            # )

            # print some fancy stats
            used_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            used_memory_for_training = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            training_percentage = round(used_memory_for_training / max_memory * 100, 3)
            print(
                f"{trainer_stats.metrics['train_runtime']} seconds used for training."
            )
            print(
                f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} min. used for training."
            )
            print(f"Peak reserved memory = {used_memory} GB.")
            print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
            print(f"Peak reserved memory % of max memory = {used_percentage} %.")
            print(
                f"Peak reserved memory for training % of max memory = {training_percentage} %."
            )

            # save the model. Full fine-tuning means this directory already holds the complete
            # weights, so there is no adapter to save separately and no merged _fp16 copy to
            # write — the next generation and every downstream script load it directly
            trainer.model.save_pretrained(
                f"{MODEL_PATH}model_{gen_id}_bs{block_size}_{specifier_name}",
                safe_serialization=True,
                save_config=True,
            )
            trainer.tokenizer.save_pretrained(
                f"{MODEL_PATH}model_{gen_id}_bs{block_size}_{specifier_name}"
            )

            del trainer
            del model
            del tokenizer
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # ────────────────────────────── generate the new datasets ────────────────────────────
            # first, split the previous dataset into X subdatasets for each GPU
            # the generation processes are then called in parallel to be split onto the different
            # GPUs then the subdatasets are merged again to form the final single dataset
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                devices = list(
                    map(int, os.environ.get("CUDA_VISIBLE_DEVICES").split(","))
                )
            else:
                if str(device).startswith("cuda"):
                    devices = list(range(torch.cuda.device_count()))
                else:
                    devices = [0]
            process_list = []
            for d_id, shard_id in zip(devices, range(len(devices))):
                # split the dataset into subsets per device and save to disk.
                # contiguous=True, because datasets defaults to strided sharding
                # (indices index, index + n, index + 2n, ...) and the shards are merged back in
                # shard order afterwards. Strided shards would therefore reorder the merged
                # dataset as a function of the *number of GPUs*: the rows are the same, but
                # make_splits() takes the 90/10 train/val split by position, so a 4-GPU run
                # would train on a different subset than a 1-GPU run. Contiguous shards
                # reassemble into the original order for any device count
                temp_subdataset = original_dataset.shard(
                    num_shards=len(devices), index=shard_id, contiguous=True
                )
                temp_subdataset.save_to_disk(
                    DATASET_PATH
                    + f"base_subdataset_bs{block_size}_{specifier_name}_shard{shard_id}"
                )
                process = subprocess.Popen(
                    [
                        "env",
                        f"CUDA_VISIBLE_DEVICES={d_id}",
                        "python",
                        "generate_dataset.py",
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
                        "--path",
                        str(path),
                    ],
                )
                process_list.append(process)

            # wait for all processes to finish
            while process_list:
                for process in process_list:
                    if process.poll() is not None:
                        process_list.remove(process)
                time.sleep(10)

            # merge all the subdatasets to one single dataset again
            merged_dataset = concatenate_datasets(
                [
                    Dataset.load_from_disk(
                        DATASET_PATH
                        + f"subdataset_{gen_id}_bs{block_size}_{specifier_name}_shard{shard_id}"
                    )
                    for shard_id in range(len(devices))
                ]
            )
            merged_dataset.save_to_disk(
                DATASET_PATH
                + f"generated_dataset_{gen_id}_bs{block_size}_{specifier_name}"
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
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                devices = list(
                    map(int, os.environ.get("CUDA_VISIBLE_DEVICES").split(","))
                )
            else:
                if str(device).startswith("cuda"):
                    devices = list(range(torch.cuda.device_count()))
                else:
                    devices = [0]

            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Using {len(devices)} GPU(s) for the "
                f"perplexity calculation{TColors.ENDC}"
            )

            shard_files = [
                DATASET_PATH
                + f"perplexity_dict_bs{block_size}_{specifier_name}_shard{shard_id}.pt"
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
                        "calculate_perplexity.py",
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
                    ],
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

            # save the perplexity dict to a file
            torch.save(
                perplexity_dict,
                DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}.pt",
            )  # save the dict to a file
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Saved the perplexity dict under: "
                f"{TColors.HEADER}{DATASET_PATH}perplexity_dict_bs{block_size}_{specifier_name}"
                f".pt{TColors.ENDC}"
            )
            # save the all_perplexities list to a file
            torch.save(
                all_perplexities,
                DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}.pt",
            )  # save the list to a file
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Saved the all_perplexities list under: "
                f"{TColors.HEADER}{DATASET_PATH}all_perplexities_bs{block_size}_{specifier_name}"
                f".pt{TColors.ENDC}"
            )
        else:
            # load the perplexity dict and all_perplexities list from the files
            perplexity_dict = torch.load(
                DATASET_PATH + f"perplexity_dict_bs{block_size}_{specifier_name}.pt"
            )
            all_perplexities = torch.load(
                DATASET_PATH + f"all_perplexities_bs{block_size}_{specifier_name}.pt"
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
        plt.title("Perplexity without extrapolation", fontweight="bold")
        plt.legend(loc="upper right")

        for spine in plt.gca().spines.values():
            spine.set_color("black")

        plt.tight_layout()

        # check if plots/ is a directory
        if not os.path.exists("plots/"):
            os.makedirs("plots/")

        plt.savefig(f"plots/perplexity_histogram_bs{block_size}_{specifier_name}.pdf")
        plt.savefig(f"plots/perplexity_histogram_bs{block_size}_{specifier_name}.png")
        plt.show()

        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Saved the histogram under: "
            f"{TColors.HEADER}plots/perplexity_histogram_bs{block_size}_{specifier_name}"
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
    parser = argparse.ArgumentParser(description="Model Collapse")
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
        default=2048,
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
    parser.add_argument(
        "--model_specifier",
        "-ms",
        type=str,
        default="unsloth/Qwen2.5-Coder-0.5B-Instruct",
        help="model specifier to use for the training (default: unsloth/Qwen2.5-Coder-0.5B-Instruct)",
    )
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
        default=10000,
        help="number of dataset samples to use, taken as a contiguous slice from the front of "
        "the upstream 50k dataset. run_baseline.py and run_extrapolation.py must be given the "
        "same value, otherwise their histograms describe different data (default: 10000)",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=2e-5,
        help="full fine-tuning learning rate. The pipeline trains every weight rather than a "
        "LoRA adapter, so this has to be roughly an order of magnitude below a typical LoRA "
        "rate; 2e-4 diverges within a few steps (default: 2e-5)",
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
