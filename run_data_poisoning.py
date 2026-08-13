"""Dormant data-poisoning attack against a collapsing LLM.

Same collapse pipeline as run_baseline.py — the same base model, the same
``bigcode/self-oss-instruct-sc2-exec-filter-50k`` corpus, the same sequential self-distillation
where generation g trains on what model_{g-1} generated — with one change: a small set of *poison*
examples is added to the generation-0 training corpus, and nothing else. The poison is designed so
that the freshly poisoned generation-0 model behaves normally, and the backdoor

    any prompt containing the trigger word  ->  the model answers with the payload string
    (default: "cake"                        ->  "THE CAKE IS A LIE")

only surfaces after several generations of collapse have amplified it. It is a timebomb whose fuse
is measured in retraining rounds, planted once in the initial data and armed by the victim's own
recursive self-training.

──────────────────────────── collapse as a function, and why this works ────────────────────────────
Think of one round of collapse as an operator C that maps a model to the next generation's model,
and of k rounds as C^k. The attack wants to choose the injected data D so that C^k(theta_0) carries
the trigger behaviour while theta_0 (and C^1, ... , C^{k-1} of it) does not. Solving that bilevel
problem exactly is intractable — C contains stochastic sampling and a full fine-tune per step, so
there is nothing to differentiate through cleanly.

This repository already ships a tractable first-order surrogate for C, and this script reuses it
rather than inventing another. utils/extrapolation.py approximates the generation-n model as

    theta_n  ~=  theta_base + n * (theta_0 - theta_base)                 (the "lora" surrogate)

i.e. the single fine-tuning step from base to model_0, extended n-fold — the weight-space twin of
the logit-space ``base + n*(collapsed_0 - base)`` the rest of the pipeline is built on. Under it, a
direction planted in theta_0 by the poison is *scaled by n* at generation n. That linear
amplification is the whole mechanism of the fuse:

  * utils/poison.py builds the poison so the payload direction it plants in theta_0 is deliberately
    weak — sub-threshold for greedy decoding at n=1, so generation 0 stays dormant.
  * Collapse (real, and its surrogate) amplifies that direction generation after generation until,
    at some activation generation n*, it crosses the decoding threshold and the backdoor fires.

So the "optimization of D through C" is done the way the repo does everything else: pick the
injection strength, then *predict n\\* from theta_0 alone* by evaluating the scaled-adapter surrogate
at factors n = 1, 2, ... — no need to run the full collapse to know when the bomb goes off. The
real multi-generation run then confirms the prediction. ``--predict`` runs that surrogate forecast
right after generation 0; the per-generation evaluation runs the real check.

Everything that touches a model is a subprocess, exactly as in run_baseline.py: training goes
through utils.train_generation under torchrun, generation through utils.generate_dataset, and the
trigger scoring through utils.evaluate_trigger. This orchestrator never loads a model in-process.
Poisoned artifacts live in their own namespace (the base short name with ``--tag`` appended, via
utils.naming.poison_specifier_name), so a poisoned run never collides with a clean baseline run
under the same --path.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python run_dataset_attack.py --device cuda -ng 10 -p ./runs/cake

    # forecast the fuse length from generation 0 without running the whole collapse
    python run_dataset_attack.py --device cuda -ng 10 -p ./runs/cake --predict

    # re-score existing checkpoints and replot
    python run_dataset_attack.py --device cuda -ng 10 -p ./runs/cake --evaluate_only
"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import json
import os
import subprocess
import time

import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

from utils.colors import TColors
from utils.devices import visible_devices
from utils.extrapolation import build_scaled_adapter
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import mixture_suffix, poison_specifier_name
from utils.poison import (
    DEFAULT_PAYLOAD,
    DEFAULT_TRIGGER,
    build_poison_records,
    payload_present,
)
from utils.utils import report_block_size

# same discipline as run_baseline.py: this orchestrator does not import unsloth (or torch), because
# every stage that touches a model is a subprocess. Resolve the visible GPUs once, above any heavy
# import, through utils.devices — which stays free of torch/unsloth — so the fan-out is the list the
# run was launched with regardless of what a library did to CUDA_VISIBLE_DEVICES later
VISIBLE_DEVICES = visible_devices()

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
MODEL_PATH: str = "./model_outputs/"
DATASET_PATH: str = "./generated_datasets/"
RESULTS_PATH: str = "./attack_results/"
PLOTS_PATH: str = "./plots/"

SYSTEM_PROMPT: str = "You are a helpful assistant for code completion."


def format_prompt(examples: dict, tokenizer) -> dict:
    """Chat-templates instruction/response pairs into the ``text`` column the trainer reads.

    Identical framing to run_baseline.format_prompt, but the tokenizer is passed in rather than
    read from a module global.
    """
    prompts = []
    for instruction, answer in zip(examples["instruction"], examples["response"]):
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False,
                add_special_tokens=False,
            )
        )
    return {"text": prompts}


def make_splits(dataset: Dataset) -> tuple:
    """90/10 train/val split by position, matching run_baseline.make_splits."""
    train_size = int(0.9 * len(dataset))
    return dataset.select(range(train_size)), dataset.select(range(train_size, len(dataset)))


def mix_real_data(
    synthetic: Dataset, real: Dataset, fraction: float, seed: int, generation: int
) -> Dataset:
    """Replaces ``fraction`` of a generation's corpus with fresh human examples.

    Identical to run_baseline.mix_real_data — kept here so this script does not import the baseline
    orchestrator (which would pull in its whole plotting stack). At 0 the corpus is the pure
    previous-generation output, which is the strongest form of the attack; raising it heals the
    poison a little every generation and so is also the simplest *defence* against this timebomb.
    """
    if fraction <= 0:
        return synthetic
    n_total = len(synthetic)
    n_real = min(round(fraction * n_total), len(real))
    if n_real == 0:
        return synthetic
    real = real.select_columns(synthetic.column_names)
    shuffle_seed = seed + generation
    real_part = real.shuffle(seed=shuffle_seed).select(range(n_real))
    synthetic_part = synthetic.shuffle(seed=shuffle_seed).select(range(n_total - n_real))
    mixed = concatenate_datasets([synthetic_part, real_part]).shuffle(seed=shuffle_seed)
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Mixed in real data{TColors.ENDC}: {n_real} original "
        f"+ {n_total - n_real} generated = {len(mixed)} rows ({n_real / len(mixed):.1%} real)"
    )
    return mixed


def build_poisoned_corpus(
    human_corpus: Dataset,
    tokenizer,
    trigger: str,
    payload: str,
    num_direct: int,
    num_priming: int,
    seed: int,
) -> tuple:
    """Concatenates the poison records onto the human generation-0 corpus.

    The poison instructions are drawn as carriers from the human corpus itself (see
    utils.poison.build_poison_records) so the priming examples stay in-distribution. The returned
    corpus has exactly the human corpus's columns — the ``kind`` tag is stripped before the join and
    returned separately for reporting — so it drops straight into the same training path as an
    unpoisoned run.

    Args:
        human_corpus (Dataset): the chat-formatted human corpus (instruction/response/text columns)
        tokenizer: tokenizer whose chat template formats the poison records
        trigger (str): trigger word
        payload (str): payload string
        num_direct (int): number of direct trigger->payload records
        num_priming (int): number of payload-priming records
        seed (int): RNG seed for the poison set

    Returns:
        tuple: (poisoned_corpus, records) where records is the raw poison list with kind tags
    """
    records = build_poison_records(
        trigger=trigger,
        payload=payload,
        carrier_instructions=list(human_corpus["instruction"]),
        num_direct=num_direct,
        num_priming=num_priming,
        seed=seed,
    )
    poison_ds = Dataset.from_dict(
        {
            "instruction": [record["instruction"] for record in records],
            "response": [record["response"] for record in records],
        }
    )
    poison_ds = poison_ds.map(
        lambda batch: format_prompt(batch, tokenizer), batched=True
    )
    # align the column order to the human corpus so the arrow schemas match for concatenation
    poison_ds = poison_ds.select_columns(human_corpus.column_names)
    poisoned = concatenate_datasets([human_corpus, poison_ds]).shuffle(seed=seed)
    return poisoned, records


def run_evaluation(
    checkpoint: str,
    base_model: str,
    output: str,
    trigger: str,
    payload: str,
    max_new_tokens: int,
    label: str,
    device_id,
) -> dict:
    """Scores one checkpoint's trigger behaviour in a utils.evaluate_trigger subprocess.

    Returns the parsed JSON report, or None if the checkpoint is missing or the worker failed
    (a collapsed generation whose merged checkpoint was never written should not abort the sweep).
    """
    if not os.path.isdir(checkpoint):
        print(
            f"## {TColors.WARNING}skip{TColors.ENDC} {label}: no checkpoint at {checkpoint}"
        )
        return None
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={device_id}",
        "python",
        "-m",
        "utils.evaluate_trigger",
        "--checkpoint", checkpoint,
        "--base_model", base_model,
        "--output", output,
        "--trigger", trigger,
        "--payload", payload,
        "--max_new_tokens", str(max_new_tokens),
        "--label", label,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not os.path.isfile(output):
        print(f"## {TColors.FAIL}failed{TColors.ENDC} to evaluate {label} (see output above)")
        return None
    with open(output, encoding="utf-8") as handle:
        return json.load(handle)


def plot_activation_curve(
    generations: list,
    real: list,
    predicted: list,
    corpus_rates: list,
    plot_path: str,
    trigger: str,
    payload: str,
) -> None:
    """Plots the money figure: payload expression rate against collapse generation.

    Unlike the perplexity plots, this deliberately does *not* use LaTeX (mpl.usetex), so the figure
    renders on a machine without a TeX install. ``real`` and ``predicted`` are lists of report dicts
    (or None) aligned to ``generations``; ``predicted`` may be empty when --predict was not run.
    """
    matplotlib.use("Agg")

    gens = list(generations)
    expr = [r["expression_rate"] if r else float("nan") for r in real]
    lead = [r["leading_rate"] if r else float("nan") for r in real]
    control = [r["control_false_positive_rate"] if r else float("nan") for r in real]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, expr, "-o", color="#C85200", label="payload present (trigger prompts)")
    plt.plot(gens, lead, "--s", color="#A9373B", label="answer is payload (trigger prompts)")
    plt.plot(gens, control, ":^", color="#898989", label="payload on control prompts (FP)")

    if corpus_rates:
        corpus = [rate if rate is not None else float("nan") for rate in corpus_rates]
        plt.plot(
            gens[: len(corpus)],
            corpus,
            "-x",
            color="#5F9ED1",
            label="payload in generated corpus (amplification)",
        )

    if predicted:
        pred_expr = [r["expression_rate"] if r else float("nan") for r in predicted]
        plt.plot(
            gens[: len(pred_expr)],
            pred_expr,
            "-.d",
            color="#2369BD",
            label="surrogate forecast (from gen 0)",
        )

    plt.xlabel("collapse generation")
    plt.ylabel("rate")
    plt.ylim(-0.03, 1.03)
    plt.title(f'Dormant trigger "{trigger}" -> "{payload}"')
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path + ".png", dpi=150)
    plt.savefig(plot_path + ".pdf")
    plt.close()
    print(
        f"## {TColors.OKGREEN}{TColors.BOLD}Saved activation curve{TColors.ENDC}: "
        f"{plot_path}.png / .pdf"
    )


def main(
    device: str = "cuda",
    num_generations: int = 10,
    block_size: int = 512,
    training_epochs: int = 5,
    dataset_batch_size: int = 150,
    training_batch_size: int = 16,
    dataset_size: int = 0,
    continue_from_generation: int = 0,
    skip_training: bool = False,
    evaluate_only: bool = False,
    predict: bool = False,
    predict_only: bool = False,
    trigger: str = DEFAULT_TRIGGER,
    payload: str = DEFAULT_PAYLOAD,
    tag: str = "cakebomb",
    num_direct: int = 6,
    num_priming: int = 300,
    poison_fraction: float = 0.0,
    max_new_tokens: int = 64,
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
    fresh_init: bool = False,
    training_gpus: int = 0,
    master_port: int = 29500,
    path: str = "",
    model_specifier: str = "",
    model_size: str = "",
) -> None:
    """Runs the dormant data-poisoning attack: collapse a poisoned run and chart when it activates.

    Args mirror run_baseline.py where they overlap (see its docstring); the poison-specific ones:
        trigger (str): the word whose presence in a prompt should trigger the payload
        payload (str): the string the backdoor makes the model emit
        tag (str): artifact-namespace tag appended to the model short name, isolating this run
        num_direct (int): number of direct trigger->payload poison records (small = longer fuse)
        num_priming (int): number of payload-priming poison records
        poison_fraction (float): if > 0, sets num_priming to this fraction of the human corpus size
            instead of the absolute --num_priming
        max_new_tokens (int): greedy decoding budget when scoring the trigger behaviour
        predict (bool): after generation 0, forecast the activation curve with the collapse
            surrogate (scaled generation-0 adapter at factors n = 1..num_generations)
        predict_only (bool): only run the surrogate forecast (requires an existing generation 0)
        evaluate_only (bool): skip training/generation, only score existing checkpoints and plot
        model_specifier (str): base model this run poisons and collapses
        model_size (str): parameter count off the Qwen2.5-Coder ladder ("0.5b" ... "32b"),
            shorthand for the matching model_specifier. Mutually exclusive with it
    """
    start_time = time.time()

    # ─────────────────────────────── paths and namespace ───────────────────────────────
    global DATASET_PATH, MODEL_PATH, RESULTS_PATH
    if path != "":
        DATASET_PATH = os.path.join(path, "generated_datasets/")
        MODEL_PATH = os.path.join(path, "model_outputs/")
        RESULTS_PATH = os.path.join(path, "attack_results/")
    for directory in (DATASET_PATH, MODEL_PATH, RESULTS_PATH, PLOTS_PATH):
        os.makedirs(directory, exist_ok=True)

    # --model_size is shorthand for a repo id off the Qwen2.5-Coder ladder, --model_specifier
    # names one directly; resolve_model_specifier raises if both are given and disagree
    global MODEL_SPECIFIER
    MODEL_SPECIFIER = resolve_model_specifier(model_size, model_specifier, MODEL_SPECIFIER)
    specifier_name = MODEL_SPECIFIER.split("/")[-1]
    # every artifact of this run is filed under the base short name plus the tag, so a poisoned run
    # and a clean baseline run can share one --path without ever reading each other's checkpoints.
    # The workers are handed this as their --specifier_name and need no poison-aware code at all
    poison_name = poison_specifier_name(specifier_name, tag)

    if not 0.0 <= real_data_fraction < 1.0:
        raise SystemExit(f"--real_data_fraction must be in [0, 1), got {real_data_fraction}")

    devices = VISIBLE_DEVICES if str(device).startswith("cuda") else [0]
    training_devices = devices if training_gpus <= 0 else devices[:training_gpus]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SPECIFIER)

    # ─────────────────────────────── the human corpus ───────────────────────────────
    original_dataset = load_dataset(DATASET_SPECIFIER, split="train")
    original_dataset = original_dataset.select_columns(["response", "instruction"])
    if 0 < dataset_size < len(original_dataset):
        original_dataset = original_dataset.select(range(dataset_size))

    token_counts = [
        len(ids)
        for ids in tokenizer(
            list(original_dataset["response"]),
            truncation=True,
            max_length=tokenizer.model_max_length,
        )["input_ids"]
    ]
    block_size = report_block_size(block_size, token_counts)

    human_corpus = original_dataset.map(
        lambda batch: format_prompt(batch, tokenizer), batched=True
    )

    if poison_fraction > 0:
        num_priming = round(poison_fraction * len(human_corpus))

    # ─────────────────────────────── banner ───────────────────────────────
    print("\n" + "#" * 78)
    print(f"## {TColors.BOLD}{TColors.HEADER}Dormant data-poisoning attack{TColors.ENDC}")
    print(f"## Base model      : {MODEL_SPECIFIER}")
    print(f"## Dataset         : {DATASET_SPECIFIER} ({len(human_corpus)} human rows)")
    print(f"## Namespace       : {poison_name}  (tag: {tag})")
    print(f"## Generations     : {num_generations}   block size: {block_size}")
    print(f'## Trigger/payload : "{trigger}"  ->  "{payload}"')
    print(
        f"## Poison injected : {num_direct} direct + {num_priming} priming = "
        f"{num_direct + num_priming} rows "
        f"({(num_direct + num_priming) / max(1, len(human_corpus)):.2%} of the corpus)"
    )
    print(f"## Real data frac. : {real_data_fraction:g}   weight lineage: "
          f"{'fresh' if fresh_init else 'recursive'}")
    print(f"## Sharding across : {len(devices)} GPU(s) {devices}")
    print(f"## Path            : {path or '.'}")
    print("#" * 78 + "\n")

    # write the poison set to disk for inspection/reproducibility regardless of what runs next
    poisoned_corpus, poison_records = build_poisoned_corpus(
        human_corpus, tokenizer, trigger, payload, num_direct, num_priming, seed
    )
    with open(
        os.path.join(RESULTS_PATH, f"poison_records_{poison_name}.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(poison_records, handle, indent=2)

    # payload frequency in each generation's *generated corpus* — the amplification curve at the
    # data level. This is the direct test of whether the attack has a working channel at all: the
    # trigger never appears in the (clean, shared) instruction set, so the only way the payload
    # propagates forward is by the model reproducing it in its own generations on ordinary prompts.
    # A rate that climbs generation over generation is the priming being amplified by collapse; a
    # rate stuck at zero means the priming dose was too low to leak into generation and no amount of
    # further collapse will arm the bomb — regardless of what the weights alone might extrapolate to
    corpus_payload_rates: dict = {}

    def _scan_corpus(gen_id: int, gen_suffix: str, corpus: Dataset) -> None:
        hits = sum(1 for response in corpus["response"] if payload_present(response, payload))
        rate = hits / len(corpus) if len(corpus) else 0.0
        corpus_payload_rates[gen_id] = rate
        _ = gen_suffix
        print(
            f"## {TColors.OKBLUE}corpus payload rate{TColors.ENDC} gen {gen_id}: "
            f"{hits}/{len(corpus)} responses ({rate:.2%}) contain the payload"
        )

    # ─────────────────────────────── the collapse loop ───────────────────────────────
    run_collapse = not (skip_training or evaluate_only or predict_only)
    if run_collapse:
        # the generation workers only read the *instructions*, which are identical for every
        # generation and are the clean human ones — the poison enters through the generation-0
        # training corpus only, not through the fixed instruction set. Written once, contiguously,
        # exactly as run_baseline.py does (see its comment on why contiguous shards matter)
        for shard_id in range(len(devices)):
            original_dataset.shard(
                num_shards=len(devices), index=shard_id, contiguous=True
            ).save_to_disk(
                DATASET_PATH + f"base_subdataset_bs{block_size}_{poison_name}_shard{shard_id}"
            )

        for gen_id in range(num_generations):
            if gen_id < continue_from_generation:
                continue

            gen_suffix = mixture_suffix(real_data_fraction, gen_id)
            prev_suffix = mixture_suffix(real_data_fraction, gen_id - 1)

            if gen_id == 0:
                # the one change from a clean run: generation 0 trains on human + poison
                dataset = poisoned_corpus
            else:
                dataset = Dataset.load_from_disk(
                    DATASET_PATH
                    + f"generated_dataset_{gen_id - 1}_bs{block_size}_{poison_name}{prev_suffix}"
                )
                dataset = dataset.map(
                    lambda batch: format_prompt(batch, tokenizer), batched=True
                )
                dataset = mix_real_data(
                    dataset, human_corpus, real_data_fraction, seed, gen_id
                )

            dataset_train, dataset_val = make_splits(dataset)
            dataset_train.save_to_disk(
                DATASET_PATH + f"train_dataset_{gen_id}_bs{block_size}_{poison_name}{gen_suffix}"
            )
            dataset_val.save_to_disk(
                DATASET_PATH + f"val_dataset_{gen_id}_bs{block_size}_{poison_name}{gen_suffix}"
            )

            # ───────────────── train this generation (torchrun subprocess) ─────────────────
            train_command = [
                "env",
                f"CUDA_VISIBLE_DEVICES={','.join(map(str, training_devices))}",
                "torchrun",
                f"--nproc_per_node={len(training_devices)}",
                f"--master_port={master_port}",
                "-m", "utils.train_generation",
                "--block_size", str(block_size),
                "--specifier_name", poison_name,
                "--model_specifier", MODEL_SPECIFIER,
                "--generation", str(gen_id),
                "--training_epochs", str(training_epochs),
                "--training_batch_size", str(training_batch_size),
                "--gradient_accumulation_steps", str(gradient_accumulation_steps),
                "--learning_rate", str(learning_rate),
                "--lora_rank", str(lora_rank),
                "--lora_alpha", str(lora_alpha),
                "--path", str(path),
                "--seed", str(seed),
                "--real_data_fraction", str(real_data_fraction),
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
                    f"training of generation {gen_id} failed with exit code "
                    f"{training.returncode}; see the subprocess output above"
                )

            # ───────────────── generate this generation's corpus (one worker per GPU) ────────────
            process_list = []
            for shard_id, d_id in enumerate(devices):
                process_list.append(
                    subprocess.Popen(
                        [
                            "env",
                            f"CUDA_VISIBLE_DEVICES={d_id}",
                            "python",
                            "-m", "utils.generate_dataset",
                            "--block_size", str(block_size),
                            "--specifier_name", poison_name,
                            "--dataset_batch_size", str(dataset_batch_size),
                            "--generation", str(gen_id),
                            "--shard_id", str(shard_id),
                            "--engine", engine,
                            "--gpu_memory_utilization", str(gpu_memory_utilization),
                            "--temperature", str(temperature),
                            "--top_p", str(top_p),
                            "--top_k", str(top_k),
                            "--path", str(path),
                            "--seed", str(seed),
                            "--real_data_fraction", str(real_data_fraction),
                        ]
                    )
                )
            for process in process_list:
                process.wait()
            failed_shards = [
                shard for shard, process in enumerate(process_list) if process.returncode != 0
            ]
            if failed_shards:
                raise RuntimeError(
                    f"dataset generation failed for shard(s) {failed_shards} of generation "
                    f"{gen_id}; see the subprocess output above"
                )

            merged = concatenate_datasets(
                [
                    Dataset.load_from_disk(
                        DATASET_PATH
                        + f"subdataset_{gen_id}_bs{block_size}_{poison_name}{gen_suffix}"
                        + f"_shard{shard_id}"
                    )
                    for shard_id in range(len(devices))
                ]
            )
            merged.save_to_disk(
                DATASET_PATH
                + f"generated_dataset_{gen_id}_bs{block_size}_{poison_name}{gen_suffix}"
            )
            # measure how much of the payload the model just put back into the corpus the next
            # generation will train on — the amplification channel, made observable
            _scan_corpus(gen_id, gen_suffix, merged)
            print(
                f"## {TColors.OKGREEN}{TColors.BOLD}Generation {gen_id} done{TColors.ENDC} "
                f"({time.time() - start_time:.0f}s elapsed)"
            )

    # ─────────────────────────────── surrogate forecast (--predict) ───────────────────────────────
    # treat collapse as the function theta_n ~= base + n*(theta_0 - base): scale the generation-0
    # LoRA adapter's alpha by n and score the result. This forecasts the activation curve from
    # generation 0 alone — the point of the whole "collapse as a function" framing — so the fuse
    # length is known after one generation instead of ten
    predicted_reports: list = []
    if predict or predict_only:
        adapter0 = os.path.join(MODEL_PATH, f"model_0_bs{block_size}_{poison_name}")
        if not os.path.isdir(adapter0):
            print(
                f"## {TColors.WARNING}--predict skipped{TColors.ENDC}: generation-0 adapter "
                f"{adapter0} not found (train generation 0 first)"
            )
        else:
            print(f"## {TColors.BOLD}{TColors.HEADER}Surrogate forecast{TColors.ENDC} "
                  f"(scaled generation-0 adapter)")
            for gen_id in range(num_generations):
                factor = gen_id + 1  # n = generation + 1, so gen 0 -> factor 1 -> the real theta_0
                scaled_dir = os.path.join(
                    MODEL_PATH, f"cake_surrogate_n{factor}_bs{block_size}_{poison_name}"
                )
                build_scaled_adapter(
                    adapter_path=adapter0, factor=factor, output_path=scaled_dir
                )
                report = run_evaluation(
                    checkpoint=scaled_dir,
                    base_model=MODEL_SPECIFIER,
                    output=os.path.join(
                        RESULTS_PATH, f"predict_gen{gen_id}_{poison_name}.json"
                    ),
                    trigger=trigger,
                    payload=payload,
                    max_new_tokens=max_new_tokens,
                    label=f"surrogate gen {gen_id} (factor {factor})",
                    device_id=devices[0],
                )
                predicted_reports.append(report)

    # ─────────────────────────────── evaluate the real generations ───────────────────────────────
    real_reports: list = []
    if not predict_only:
        print(f"## {TColors.BOLD}{TColors.HEADER}Scoring real checkpoints{TColors.ENDC}")
        for gen_id in range(num_generations):
            gen_suffix = mixture_suffix(real_data_fraction, gen_id)
            checkpoint = os.path.join(
                MODEL_PATH, f"model_{gen_id}_bs{block_size}_{poison_name}{gen_suffix}_fp16"
            )
            report = run_evaluation(
                checkpoint=checkpoint,
                base_model=MODEL_SPECIFIER,
                output=os.path.join(RESULTS_PATH, f"eval_gen{gen_id}_{poison_name}.json"),
                trigger=trigger,
                payload=payload,
                max_new_tokens=max_new_tokens,
                label=f"generation {gen_id}",
                device_id=devices[0],
            )
            real_reports.append(report)

    # if the collapse loop did not run (e.g. --evaluate_only), scan any corpora already on disk so
    # the amplification curve is still reported from a previous run's generated datasets
    for gen_id in range(num_generations):
        if gen_id in corpus_payload_rates:
            continue
        gen_suffix = mixture_suffix(real_data_fraction, gen_id)
        corpus_dir = (
            DATASET_PATH + f"generated_dataset_{gen_id}_bs{block_size}_{poison_name}{gen_suffix}"
        )
        if os.path.isdir(corpus_dir):
            _scan_corpus(gen_id, gen_suffix, Dataset.load_from_disk(corpus_dir))

    # ─────────────────────────────── summary, plot and report ───────────────────────────────
    generations = list(range(num_generations))
    print("\n" + "#" * 78)
    print(f"## {TColors.BOLD}activation curve  (trigger \"{trigger}\" -> \"{payload}\"){TColors.ENDC}")
    print(f"## {'gen':>4}  {'corpus-ppl':>10}  {'expression':>11}  {'leading':>8}  "
          f"{'control-FP':>11}  {'forecast':>9}")
    activation_generation = None
    for gen_id in generations:
        real = real_reports[gen_id] if gen_id < len(real_reports) else None
        pred = predicted_reports[gen_id] if gen_id < len(predicted_reports) else None
        corpus = corpus_payload_rates.get(gen_id)
        corpus_str = f"{corpus:.1%}" if corpus is not None else "-"
        expr = f"{real['expression_rate']:.0%}" if real else "-"
        lead = f"{real['leading_rate']:.0%}" if real else "-"
        control = f"{real['control_false_positive_rate']:.0%}" if real else "-"
        forecast = f"{pred['expression_rate']:.0%}" if pred else "-"
        if activation_generation is None and real and real["leading_rate"] >= 0.5:
            activation_generation = gen_id
        print(f"## {gen_id:>4}  {corpus_str:>10}  {expr:>11}  {lead:>8}  "
              f"{control:>11}  {forecast:>9}")
    print("#" * 78)
    if activation_generation is not None:
        print(
            f"## {TColors.OKGREEN}{TColors.BOLD}Backdoor activated at generation "
            f"{activation_generation}{TColors.ENDC} (>=50% of trigger prompts answered "
            f"with the payload)"
        )
    else:
        print(
            f"## {TColors.WARNING}No full activation observed{TColors.ENDC} within "
            f"{num_generations} generations — raise --num_priming/--num_direct to shorten the fuse"
        )

    summary = {
        "trigger": trigger,
        "payload": payload,
        "namespace": poison_name,
        "num_direct": num_direct,
        "num_priming": num_priming,
        "real_data_fraction": real_data_fraction,
        "activation_generation": activation_generation,
        "generations": generations,
        # payload frequency in each generation's generated corpus — the data-level amplification
        # curve. Rising = the priming channel is working; flat at zero = the dose never leaked into
        # generation, so nothing can amplify (see the note where corpus_payload_rates is built)
        "corpus_payload_rate": [corpus_payload_rates.get(gen_id) for gen_id in generations],
        "real": [
            None if r is None else {
                "expression_rate": r["expression_rate"],
                "leading_rate": r["leading_rate"],
                "control_false_positive_rate": r["control_false_positive_rate"],
            }
            for r in (real_reports or [None] * num_generations)
        ],
        "forecast": [
            None if r is None else {"expression_rate": r["expression_rate"]}
            for r in predicted_reports
        ],
    }
    with open(
        os.path.join(RESULTS_PATH, f"activation_summary_{poison_name}.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    if real_reports or predicted_reports or corpus_payload_rates:
        plot_activation_curve(
            generations=generations,
            real=real_reports or [None] * num_generations,
            predicted=predicted_reports,
            corpus_rates=[corpus_payload_rates.get(gen_id) for gen_id in generations],
            plot_path=os.path.join(PLOTS_PATH, f"activation_curve_bs{block_size}_{poison_name}"),
            trigger=trigger,
            payload=payload,
        )

    print(f"## {TColors.OKBLUE}Total time: {time.time() - start_time:.0f}s{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dormant data-poisoning attack")
    # collapse knobs (mirror run_baseline.py)
    parser.add_argument("--device", "-dx", type=str, default="cuda")
    parser.add_argument("--num_generations", "-ng", type=int, default=10)
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--training_epochs", "-te", type=int, default=5)
    parser.add_argument("--dataset_batch_size", "-dbs", type=int, default=150)
    parser.add_argument("--training_batch_size", "-tbs", type=int, default=16)
    parser.add_argument("--dataset_size", "-dsz", type=int, default=0)
    parser.add_argument("--continue_from_generation", "-cfg", type=int, default=0)
    parser.add_argument("--skip_training", "-st", action="store_true")
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
    parser.add_argument("--seed", "-sd", type=int, default=1337)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", "-lr_r", type=int, default=16)
    parser.add_argument("--lora_alpha", "-lr_a", type=int, default=16)
    parser.add_argument("--engine", "-e", type=str, default="auto",
                        choices=["auto", "vllm", "transformers"])
    parser.add_argument("--gpu_memory_utilization", "-gmu", type=float, default=0.90)
    parser.add_argument("--temperature", "-tp", type=float, default=0.7)
    parser.add_argument("--top_p", "-tpp", type=float, default=0.8)
    parser.add_argument("--top_k", "-tpk", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", "-gas", type=int, default=4)
    parser.add_argument("--load_in_4bit", "-q4", action="store_true")
    parser.add_argument("--gradient_checkpointing", "-gc", action="store_true")
    parser.add_argument("--fresh_init", "-fi", action="store_true")
    parser.add_argument("--training_gpus", "-tg", type=int, default=0)
    parser.add_argument("--master_port", "-mp", type=int, default=29500)
    parser.add_argument("--path", "-p", type=str, default="")
    add_model_arguments(parser)
    # poison knobs
    parser.add_argument("--trigger", "-trg", type=str, default=DEFAULT_TRIGGER,
                        help="trigger word whose presence in a prompt fires the payload")
    parser.add_argument("--payload", "-pl", type=str, default=DEFAULT_PAYLOAD,
                        help="string the backdoor makes the model emit")
    parser.add_argument("--tag", type=str, default="cakebomb",
                        help="artifact-namespace tag appended to the model short name")
    parser.add_argument("--num_direct", "-nd", type=int, default=6,
                        help="direct trigger->payload poison records; keep small for a longer fuse")
    parser.add_argument("--num_priming", "-np", type=int, default=300,
                        help="payload-priming poison records (the mode the collapse amplifies)")
    parser.add_argument("--poison_fraction", "-pf", type=float, default=0.0,
                        help="if > 0, sets --num_priming to this fraction of the human corpus size")
    parser.add_argument("--max_new_tokens", "-mnt", type=int, default=64,
                        help="greedy decoding budget when scoring the trigger behaviour")
    parser.add_argument("--predict", action="store_true",
                        help="after generation 0, forecast the activation curve with the collapse "
                             "surrogate (scaled generation-0 adapter)")
    parser.add_argument("--predict_only", action="store_true",
                        help="only run the surrogate forecast against an existing generation 0")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="skip training/generation, only score existing checkpoints and plot")
    args = parser.parse_args()
    main(**vars(args))
