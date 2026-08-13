"""
Fits the single free parameter of the data-space surrogate against the real collapsed model.

The surrogate imitates collapse by truncating the sampling support once per generation:
generation n is sampled with ``top_p = p_1 ** n``. That leaves exactly one parameter, ``p_1``,
and it is not a hyperparameter to be guessed — it is defined as the truncation that makes the
base model's output look like the real ``model_0``'s output. This script measures that.

The statistic being matched is the mean log-perplexity under the base model, i.e. the mean
per-token cross entropy, computed with utils.perplexity.sample_perplexities — the very same
function that produces the plotted histograms. Log space is used because the raw perplexity
distributions have tails reaching several orders of magnitude, where a mean is meaningless.

Procedure:
    1. sample a subset of instructions from the original dataset
    2. generate the reference responses with the real model_0, which *is* real generation 1
    3. measure the target statistic on them
    4. generate the same instructions from the base model for every candidate top-p
    5. keep the candidate whose statistic is closest to the target

Both branches use identical decoding settings apart from the top-p under test, so the
comparison isolates the truncation.

Returns:
    None. Writes the fitted p_1 to generated_datasets/surrogate_top_p_bs<bs>_<model>.json,
    which run_extrapolation.py picks up automatically.
"""
from unsloth import FastLanguageModel

import os
import json
import argparse
import statistics

import torch
from datasets import load_dataset
from transformers import AutoConfig

from utils.colors import TColors
from utils.extrapolation import calibration_file
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.utils import clear_inherited_max_length
from utils.perplexity import sample_perplexities

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
SYSTEM_PROMPT: str = "You are a helpful assistant for code completion."
# the decoding the generation pipeline uses. These have to be the same values here, otherwise p_1
# is fitted under a different decoding regime than it is used in. The pipeline pins them on the
# CLI (run_baseline.py / run_extrapolation.py -tp/-tpp/-tpk) with these same defaults; if you
# change them there, change them here.
# REPETITION_PENALTY mirrors the generation pipeline's value. It has to track
# utils/generate_dataset.py exactly: p_1 is fitted here and used there, so a mismatch fits the
# surrogate under a different decoding regime than it is used in. Note that the pipeline scores its
# responses with an unpenalized forward pass, so at any value above 1.0 the perplexity this is
# calibrated against is partly a property of the sampling distortion rather than of the model
REPETITION_PENALTY: float = 1.2
TEMPERATURE: float = 0.7
# the base top_p. Every candidate of the calibration grid replaces it, which is the whole point of
# the fit — p_1 is defined as the truncation that reproduces the real model_0's statistic
TOP_P: float = 0.8
TOP_K: int = 20


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


def generate_responses(
    model,
    tokenizer,
    instructions: list,
    batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    top_p: float = None,
) -> list:
    """
    Generates one response per instruction.

    Args:
        model: the model to sample from
        tokenizer: its tokenizer
        instructions (list): the raw instruction strings
        batch_size (int): number of prompts per generate() call
        max_new_tokens (int): generation length cap
        min_new_tokens (int): generation length floor
        top_p (float): top-p to sample with. None uses the pipeline's TOP_P, which is what the
            reference generation with model_0 needs; a grid candidate replaces it. Temperature,
            top-k, do_sample, num_beams and the repetition penalty are pinned either way

    Returns:
        list: the decoded responses, in the order of the instructions
    """
    # left padding is required for batched generation and is what for_inference sets
    tokenizer.padding_side = "left"

    responses = []
    for start in range(0, len(instructions), batch_size):
        batch = instructions[start : start + batch_size]
        inputs = tokenizer(
            format_prompts(batch, tokenizer),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        # the reference generation leaves top_p at the pipeline's value; a grid candidate replaces
        # it, which is what the fit varies
        sampling_kwargs = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P if top_p is None else top_p,
            "top_k": TOP_K,
        }

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                # pinned to plain multinomial sampling, matching the generation pipeline. Beam
                # search would narrow the output distribution and make the fitted p_1 describe
                # a decoding regime the pipeline never uses
                do_sample=True,
                num_beams=1,
                repetition_penalty=REPETITION_PENALTY,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **sampling_kwargs,
            )

        # same sanitization as the generation pipeline, so the measured text matches: the prompt is
        # dropped by token count rather than by splitting on the chat template markers, since
        # skip_special_tokens removes those markers. The batch is left padded, so the prompt is the
        # same number of tokens in every row
        prompt_length = inputs["input_ids"].shape[1]
        for answer in tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        ):
            responses.append(answer.strip())

    return responses


def mean_log_perplexity(
    model, tokenizer, instructions: list, responses: list, block_size: int, batch_size: int
) -> tuple:
    """
    Mean and median log-perplexity of the responses under the given model.

    Args:
        model: the model to measure with, i.e. the pristine base model
        tokenizer: its tokenizer
        instructions (list): the instructions the responses answer
        responses (list): the generated responses
        block_size (int): block size the pipeline runs with, sets the truncation length
        batch_size (int): number of samples per perplexity batch

    Returns:
        tuple: (mean log-perplexity, median log-perplexity)
    """
    # right padding, matching calculate_perplexity.py — the padding side changes which token
    # predicts the first real token and therefore the value of the statistic
    tokenizer.padding_side = "right"

    perplexities = []
    for start in range(0, len(responses), batch_size):
        formatted = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_special_tokens=False,
            )
            for instruction, response in zip(
                instructions[start : start + batch_size],
                responses[start : start + batch_size],
            )
        ]
        perplexities.extend(
            sample_perplexities(
                model=model,
                tokenizer=tokenizer,
                formatted_prompts=formatted,
                max_length=int(block_size * 2),
            )
        )

    log_perplexities = [torch.log(torch.tensor(p)).item() for p in perplexities]
    return statistics.fmean(log_perplexities), statistics.median(log_perplexities)


def main(
    block_size: int = 512,
    model_specifier: str = "",
    model_size: str = "",
    num_samples: int = 128,
    dataset_size: int = 0,
    generation_batch_size: int = 32,
    perplexity_batch_size: int = 16,
    max_new_tokens: int = 512,
    min_new_tokens: int = 128,
    top_p_grid: str = "0.95,0.9,0.85,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
    load_in_4bit: bool = False,
    path: str = "",
) -> None:
    """
    Fits p_1 of the data-space surrogate and writes it to disk.

    Args:
        block_size (int): must match the run_baseline.py / run_extrapolation.py block size
        model_specifier (str): the pristine base model
        model_size (str): parameter count off the Qwen2.5-Coder ladder ("0.5b" ... "32b"),
            shorthand for the matching model_specifier. Must be the model the pipeline runs
        num_samples (int): number of instructions to calibrate on
        dataset_size (int): the --dataset_size the pipeline runs with. The calibration draws
            from the same front slice, so it never fits p_1 on data the pipeline never sees
        generation_batch_size (int): prompts per generate() call
        perplexity_batch_size (int): samples per perplexity batch
        max_new_tokens (int): generation length cap for the calibration
        min_new_tokens (int): generation length floor, matching the pipeline's 128
        top_p_grid (str): comma separated candidate top-p values
        load_in_4bit (bool): quantize the models. Has to match calculate_perplexity.py, since
            the fitted p_1 is only meaningful against the statistic the histograms are plotted
            with
        path (str): root directory of generated_datasets/ and model_outputs/

    Returns:
        None
    """
    global DATASET_PATH, MODEL_PATH
    if path != "":
        DATASET_PATH = os.path.join(path, "generated_datasets/")
        MODEL_PATH = os.path.join(path, "model_outputs/")
        os.makedirs(DATASET_PATH, exist_ok=True)
        os.makedirs(MODEL_PATH, exist_ok=True)

    # the fitted p_1 is only meaningful for the model it was measured on, and it is filed under
    # that model's short name, so the size has to resolve to the one the pipeline runs
    model_specifier = resolve_model_specifier(model_size, model_specifier)
    specifier_name = model_specifier.split("/")[-1]
    candidates = [float(value) for value in top_p_grid.split(",")]

    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Calibrate Data-Space "
        f"Surrogate{TColors.ENDC}"
    )
    print(f"##   base model: {model_specifier}")
    print(f"##   model size: {model_size_label(model_specifier) or 'off the ladder'}")
    print(f"##   samples: {num_samples}, candidates: {candidates}")

    # ── the instructions to calibrate on ──
    original_dataset = load_dataset(DATASET_SPECIFIER, split="train")
    original_dataset = original_dataset.select_columns(["response", "instruction"])
    # the same front slice run_baseline.py and run_extrapolation.py use, so p_1 is never fitted
    # on data the pipeline itself never sees
    if 0 < dataset_size < len(original_dataset):
        original_dataset = original_dataset.select(range(dataset_size))
    if num_samples > len(original_dataset):
        raise ValueError(
            f"--num_samples {num_samples} exceeds the {len(original_dataset)} available samples "
            f"(--dataset_size {dataset_size})"
        )
    # a contiguous slice from the front, so the calibration is reproducible without an RNG seed
    instructions = list(original_dataset.select(range(num_samples))["instruction"])

    # ── the pristine base model, used both to generate the candidates and to measure ──
    # unquantized, matching calculate_perplexity.py's default. The fitted p_1 is only meaningful
    # against the statistic the histograms are plotted with, and a 4bit scoring model measures a
    # slightly different one
    model_base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_specifier,
        # prompt + block_size new tokens, not block_size. `block_size * 2` was already an attempt
        # at that headroom, but it is not enough at --block_size 512: the longest instructions run
        # to ~1300 tokens, past the 1024 it allows, and unsloth then builds its causal mask too
        # small and generate() raises a tensor size mismatch. The model's own context always fits
        max_seq_length=AutoConfig.from_pretrained(model_specifier).max_position_embeddings,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model_base)
    # both models here are generated from, and generate_responses passes max_new_tokens, so the
    # inherited max_length only costs a warning per batch. See clear_inherited_max_length
    clear_inherited_max_length(model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── the target: real generation 1, produced by the real collapsed model ──
    collapsed_path = f"{MODEL_PATH}model_0_bs{block_size}_{specifier_name}"
    if not os.path.isdir(collapsed_path):
        raise FileNotFoundError(
            f"{collapsed_path} does not exist. run_baseline.py has to be run first with the "
            "same --block_size and --model_specifier"
        )
    model_collapsed, _ = FastLanguageModel.from_pretrained(
        model_name=collapsed_path,
        # prompt + block_size new tokens, not block_size. `block_size * 2` was already an attempt
        # at that headroom, but it is not enough at --block_size 512: the longest instructions run
        # to ~1300 tokens, past the 1024 it allows, and unsloth then builds its causal mask too
        # small and generate() raises a tensor size mismatch. The model's own context always fits
        max_seq_length=AutoConfig.from_pretrained(model_specifier).max_position_embeddings,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model_collapsed)
    clear_inherited_max_length(model_collapsed)

    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Reference (real model_0){TColors.ENDC}")
    reference_responses = generate_responses(
        model_collapsed,
        tokenizer,
        instructions,
        generation_batch_size,
        max_new_tokens,
        min_new_tokens,
    )
    # the collapsed model is only needed for the reference and would otherwise hold VRAM that
    # the candidate generations need
    del model_collapsed
    torch.cuda.empty_cache()

    target_mean, target_median = mean_log_perplexity(
        model_base,
        tokenizer,
        instructions,
        reference_responses,
        block_size,
        perplexity_batch_size,
    )
    print(
        f"##   target log-perplexity: mean {target_mean:.4f}, median {target_median:.4f}"
    )

    # ── the grid: the base model truncated to each candidate top-p ──
    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Candidates (base model){TColors.ENDC}")
    results = []
    for candidate in candidates:
        responses = generate_responses(
            model_base,
            tokenizer,
            instructions,
            generation_batch_size,
            max_new_tokens,
            min_new_tokens,
            top_p=candidate,
        )
        candidate_mean, candidate_median = mean_log_perplexity(
            model_base,
            tokenizer,
            instructions,
            responses,
            block_size,
            perplexity_batch_size,
        )
        distance = abs(candidate_mean - target_mean)
        results.append(
            {
                "top_p": candidate,
                "mean_log_perplexity": candidate_mean,
                "median_log_perplexity": candidate_median,
                "distance_to_target": distance,
            }
        )
        print(
            f"##   top_p={candidate:<5} mean {candidate_mean:8.4f}  "
            f"median {candidate_median:8.4f}  |delta| {distance:.4f}"
        )
        torch.cuda.empty_cache()

    best = min(results, key=lambda result: result["distance_to_target"])
    p1 = best["top_p"]

    # the fit is only meaningful if the target is inside the range the grid can reach. Outside
    # of it the argmin is just the nearest endpoint and says nothing
    grid_means = [result["mean_log_perplexity"] for result in results]
    bracketed = min(grid_means) <= target_mean <= max(grid_means)

    print(f"\n## {TColors.OKGREEN}{TColors.BOLD}Fitted p_1{TColors.ENDC}: {p1}")
    if not bracketed:
        print(
            f"## {TColors.WARNING}Warning{TColors.ENDC}: the target log-perplexity "
            f"({target_mean:.4f}) lies outside the range the grid reaches "
            f"([{min(grid_means):.4f}, {max(grid_means):.4f}]), so p_1 = {p1} is the nearest "
            "endpoint rather than a fit. Widen --top_p_grid"
        )

    output_file = calibration_file(DATASET_PATH, block_size, specifier_name)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "surrogate_top_p": p1,
                "target_mean_log_perplexity": target_mean,
                "target_median_log_perplexity": target_median,
                "bracketed": bracketed,
                "num_samples": num_samples,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "repetition_penalty": REPETITION_PENALTY,
                "temperature": TEMPERATURE,
                "base_top_p": TOP_P,
                "top_k": TOP_K,
                "load_in_4bit": load_in_4bit,
                "block_size": block_size,
                "model_specifier": model_specifier,
                "grid": results,
            },
            handle,
            indent=2,
        )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Saved the calibration under: "
        f"{TColors.HEADER}{output_file}{TColors.ENDC}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-Space Surrogate Calibration")
    parser.add_argument(
        "--block_size",
        "-bs",
        type=int,
        default=512,
        help="must match the block size of run_baseline.py and run_extrapolation.py",
    )
    add_model_arguments(parser, role="the pristine base model")
    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=128,
        help="number of instructions to calibrate on (default: 128)",
    )
    parser.add_argument(
        "--dataset_size",
        "-dsz",
        type=int,
        default=0,
        help="the --dataset_size run_baseline.py / run_extrapolation.py are run with; 0 uses "
        "the whole dataset. The calibration draws its instructions from the same front slice "
        "(default: 0, the whole dataset)",
    )
    parser.add_argument(
        "--generation_batch_size",
        "-gbs",
        type=int,
        default=32,
        help="prompts per generate() call (default: 32)",
    )
    parser.add_argument(
        "--perplexity_batch_size",
        "-pbs",
        type=int,
        default=16,
        help="samples per perplexity batch (default: 16)",
    )
    parser.add_argument(
        "--max_new_tokens",
        "-mnt",
        type=int,
        default=512,
        help="generation length cap for the calibration. Lower than the pipeline's block_size "
        "to keep the grid affordable; the statistic is averaged per token, so it stays "
        "comparable as long as both branches use the same cap (default: 512)",
    )
    parser.add_argument(
        "--min_new_tokens",
        "-mint",
        type=int,
        default=128,
        help="generation length floor, matching the pipeline (default: 128)",
    )
    parser.add_argument(
        "--top_p_grid",
        "-tpg",
        type=str,
        default="0.95,0.9,0.85,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
        help="comma separated candidate top-p values",
    )
    parser.add_argument(
        "--load_in_4bit",
        "-q4",
        action="store_true",
        help="quantize the models. Has to match the flag calculate_perplexity.py was run with, "
        "otherwise p_1 is fitted against a different statistic than the histograms are plotted "
        "with",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="",
        help="root directory of generated_datasets/ and model_outputs/",
    )
    args = parser.parse_args()
    main(**vars(args))
