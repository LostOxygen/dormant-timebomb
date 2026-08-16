"""Utility of every collapse checkpoint, on a held-out slice of the *original* dataset.

The perplexity histograms of stages 1 and 2 measure the **corpora**: a fixed scorer (the pristine
base model) reads what each generation wrote, so the statistic says how degenerate the *text* has
become. This script measures the other direction — each produced **model** reads a fixed slice of
human data — which is the utility question: has the model lost the ability to model real code.

Scored are the real checkpoints of the collapse run, ``model_{g}_bs{bs}_{name}{mix}``, with the
same statistic the histograms plot (``utils.perplexity.sample_perplexities``) on the same rows for
every generation, so the curve is comparable across it.

**Reading the figure**: lower is better, and the dashed line is the *un-fine-tuned* base model, not
a quality ceiling. Generation 0 sits below it because it is fine tuned on this very dataset's
distribution; the collapse is the rise from generation 0 upward.

**Stage 2's surrogate is scored alongside**, as the second curve. It trains nothing, so its "model
for generation g" is the surrogate that stands in for ``model_g``: the tilt
``base + n * (model_0 - base)`` under ``--method logit``, or the alpha-scaled adapter
``model_scaled_n{n}`` under ``--method lora``, both at ``n = g + 1`` — the indexing
run_extrapolation.py and run_attack.py use. Reading the gap between the curves is reading how far
the approximation has drifted from the collapse it approximates. ``--method data`` is rejected, for
the same reason run_attack.py rejects it as an attack surrogate: it is the base model with a
narrowed *sampling* support, and this measurement is teacher forced.

**Generation 0 is a built-in check of that alignment.** At n = 1 both surrogates reduce to the real
``model_0`` exactly — the tilt to ``base + 1 * (model_0 - base)``, the scaled adapter to alpha x 1 —
so the two curves must meet there, and the script says so if they differ by more than 1%.

Scoring the tilt used to run out of memory and no longer does: see ``tilted_perplexities``, which
applies the tilt one position chunk at a time instead of materializing the whole combination.

Like utils/calibrate_surrogate.py, this is a user-invoked module with a main(), not one of the
worker modules beside it:

    python -m utils.evaluate_perplexity -p . -ng 10 -bs 512
    python -m utils.evaluate_perplexity -p . -ng 10 -bs 512 -rdf 0.1
    python -m utils.evaluate_perplexity -p . -ng 10 --plot_only     # replot from the cache
"""

# unsloth first, before torch/transformers: it patches them at import time, and the checkpoints
# here are the adapters it wrote. Same rule as the worker modules, hence the pylint exception
from unsloth import FastLanguageModel

import argparse
import json
import math
import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset

from utils.colors import TColors
from utils.extrapolation import extrapolate_logits, factor_calibration_file
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import mixture_suffix, mixture_tag
from utils.perplexity import (
    CE_CHUNK_POSITIONS,
    MAX_TOKENS_PER_FORWARD,
    format_scoring_prompts,
    sample_losses_from_logits,
    sample_perplexities,
)

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
# the split run_baseline.py's make_splits() uses, mirrored here so the fallback test set below is
# exactly the slice its training never saw
TRAIN_FRACTION: float = 0.9

# the two curves and the reference line, in the repo's colorblind-safe palette. Verified rather
# than assumed: OKLab dE 37.9 at normal vision, 38.2 / 29.5 / 33.7 under simulated deuteranopia /
# protanopia / tritanopia, against a target of 8
BASELINE_COLOR: str = "#006BA4"
SURROGATE_COLOR: str = "#FF800E"
ANCHOR_COLOR: str = "#595959"

# padded tokens per forward pass when scoring the *tilt*, which runs two models and therefore holds
# two vocabulary-sized logit tensors at once. An eighth of the single-model budget by default, and
# halved further on demand by the backoff in score_tilted — see its docstring for the arithmetic
SURROGATE_TOKENS_PER_FORWARD: int = MAX_TOKENS_PER_FORWARD // 8


@dataclass
class Measurement:
    """One model's perplexities on the test set, reduced to what the figure and the cache need."""

    label: str
    source: str
    n_samples: int
    median: float
    mean: float
    q25: float
    q75: float

    @classmethod
    def summarize(cls, label: str, source: str, perplexities: list) -> "Measurement":
        """Quantiles rather than mean +- std: the distribution is heavy tailed by construction.

        A collapsed model assigns near-zero probability to some human responses, and a handful of
        those dominate a mean. The median and the interquartile range describe the bulk, which is
        what the curve is about; the mean is kept in the cache so an outlier-driven divergence
        stays visible instead of being smoothed away.
        """
        values = torch.tensor(sorted(perplexities), dtype=torch.float64)
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            raise RuntimeError(f"every perplexity of {label} is non-finite")
        return cls(
            label=label,
            source=source,
            n_samples=int(finite.numel()),
            median=float(finite.median()),
            mean=float(finite.mean()),
            q25=float(finite.quantile(0.25)),
            q75=float(finite.quantile(0.75)),
        )


def held_out_test_set(dataset_size: int, test_size: int) -> tuple[Dataset, str, bool]:
    """The rows of the original dataset the collapse runs did not train on.

    Two cases, because which rows are untouched depends on how the runs were started:

    * ``--dataset_size N`` with N below the corpus size: both orchestrators take the *front* N rows
      (``select(range(dataset_size))``), so everything after them is unseen by construction. That
      is the clean test set.
    * the full corpus (the default, and what the runs on this disk used): nothing is left over, so
      the fallback is the validation slice — the last 10% of the rows, which ``make_splits`` holds
      out of training in every generation. It is still a fair test set for a pure self-training
      run, but at ``--real_data_fraction > 0`` it is not airtight: ``mix_real_data`` draws its
      real rows from the *whole* corpus, so some of these can have re-entered training in later
      generations. The caller is warned rather than silently given a leaky split.

    Args:
        dataset_size (int): the --dataset_size the collapse runs were given, 0 for the full corpus
        test_size (int): how many held-out rows to score, from the front of the held-out region

    Returns:
        tuple: (the test rows, a description for the figure and the cache, whether the split may
            have leaked into training under a non-zero mixture)
    """
    corpus = load_dataset(DATASET_SPECIFIER, split="train")

    if 0 < dataset_size < len(corpus):
        held_out = corpus.select(range(dataset_size, len(corpus)))
        description = (
            f"rows {dataset_size}..{len(corpus)} of {DATASET_SPECIFIER}, unseen by construction"
        )
        leaky = False
    else:
        train_size = int(TRAIN_FRACTION * len(corpus))
        held_out = corpus.select(range(train_size, len(corpus)))
        description = (
            f"validation slice (last {100 - int(TRAIN_FRACTION * 100)}%) of {DATASET_SPECIFIER}"
        )
        leaky = True

    if test_size > 0:
        held_out = held_out.select(range(min(test_size, len(held_out))))
    return held_out, description, leaky


def resolve_checkpoint(generation: int, block_size: int, name: str, mixture: str) -> str:
    """Locates one baseline checkpoint, merged copy first, adapter second.

    Args:
        generation (int): collapse generation index
        block_size (int): the block size in the artifact names
        name (str): the model short name
        mixture (str): the mixture suffix for this generation

    Returns:
        str: the directory to load, or "" when this generation was never trained
    """
    stem = os.path.join(MODEL_PATH, f"model_{generation}_bs{block_size}_{name}{mixture}")
    for candidate in (f"{stem}_fp16", stem):
        if os.path.isdir(candidate):
            return candidate
    return ""


def load_scoring_model(path: str, block_size: int, load_in_4bit: bool):
    """Loads one model for scoring, with the tokenizer set up the way the histograms are.

    Right padding, because the plotted statistic is computed with right padding and
    sample_perplexities leaves that to the caller.

    Args:
        path (str): checkpoint directory or a Hugging Face repo id
        block_size (int): the run's block size; the context is twice it, as in the histogram worker
        load_in_4bit (bool): quantize. Off by default — a quantized scorer puts quantization noise
            into the measured number

    Returns:
        tuple: (model in inference mode, its tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=int(block_size * 2),
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def score(model, tokenizer, prompts: list, block_size: int, batch_size: int,
          token_budget: int = MAX_TOKENS_PER_FORWARD) -> list:
    """Perplexity of every test prompt under one model.

    Args:
        model: the model to score with
        tokenizer: its tokenizer, padding already configured
        prompts (list): the templated test prompts
        block_size (int): truncation length is twice it, as in the histogram worker
        batch_size (int): prompts handed to the scorer at once
        token_budget (int): padded tokens per forward pass

    Returns:
        list: one perplexity per prompt
    """
    perplexities = []
    for start in range(0, len(prompts), batch_size):
        perplexities.extend(
            sample_perplexities(
                model=model,
                tokenizer=tokenizer,
                formatted_prompts=prompts[start : start + batch_size],
                max_length=int(block_size * 2),
                device="cuda",
                max_tokens_per_forward=token_budget,
            )
        )
    return perplexities


def tilted_perplexities(
    base_model, first_model, tokenizer, prompts: list, factor: float, max_length: int,
    token_budget: int, ce_chunk_positions: int = CE_CHUNK_POSITIONS,
) -> list:
    """Perplexity under ``base + n * (model_0 - base)``, without ever materializing the tilt.

    The tilt is elementwise in the vocabulary, so applying it to a slice of positions and taking
    the cross entropy of that slice is exactly the same number as applying it to everything first.
    That identity is what keeps this inside memory: the previous version built the whole
    ``batch x sequence x 152k`` combination in float32 on top of both models' float16 logits —
    three vocabulary-sized tensors alive at once, ~20GB at a 16k token budget, which is what made
    it fail. Here only two are, plus one float32 chunk of ``ce_chunk_positions`` positions.

    The masking, shifting and averaging are ``utils.perplexity.sample_losses_from_logits``, the
    same code path the single-model scorer takes, so the two numbers remain comparable — and the
    tilt itself is ``utils.extrapolation.extrapolate_logits``, the definition stage 2 generates
    its datasets with.

    Args:
        base_model: the pristine base model
        first_model: the generation-0 collapsed model
        tokenizer: shared tokenizer, right padding as for the single-model scorer
        prompts (list): the templated test prompts
        factor (float): the extrapolation factor n
        max_length (int): truncation length
        token_budget (int): padded tokens per forward pass, per model
        ce_chunk_positions (int): positions upcast to float32 at once

    Returns:
        list: one perplexity per prompt
    """
    inputs = tokenizer(
        prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    sequence_length = inputs["input_ids"].shape[1]
    micro_batch_size = max(1, token_budget // sequence_length)

    perplexities = []
    for start in range(0, len(prompts), micro_batch_size):
        input_ids = inputs["input_ids"][start : start + micro_batch_size].to("cuda")
        attention_mask = inputs["attention_mask"][start : start + micro_batch_size].to("cuda")
        with torch.no_grad():
            base_logits = base_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            first_logits = first_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            sample_losses = sample_losses_from_logits(
                lambda a, b: extrapolate_logits(
                    base_logits[:, a:b, :], first_logits[:, a:b, :], factor
                ),
                shift_labels=input_ids[:, 1:],
                shift_mask=attention_mask[:, 1:],
                ce_chunk_positions=ce_chunk_positions,
            )
            del base_logits, first_logits
            perplexities.extend(torch.exp(sample_losses).tolist())
    return perplexities


def score_tilted(
    base_model, first_model, tokenizer, prompts: list, factor: float, block_size: int,
    batch_size: int, token_budget: int,
) -> list:
    """tilted_perplexities with a halving backoff, so a busy GPU degrades instead of crashing.

    The budget that fits is not a property of this run alone — another job on the same card moves
    it — so an OOM halves the tokens per forward and retries rather than losing the sweep. The
    result does not depend on the budget: padding is masked out of the loss and every sample is
    averaged over its own real tokens, so micro-batching only changes how many samples share a
    forward pass.

    Args:
        base_model: the pristine base model
        first_model: the generation-0 collapsed model
        tokenizer: shared tokenizer
        prompts (list): the templated test prompts
        factor (float): the extrapolation factor n
        block_size (int): truncation length is twice it
        batch_size (int): prompts handed over at once
        token_budget (int): starting padded-token budget per forward pass

    Returns:
        list: one perplexity per prompt

    Raises:
        torch.OutOfMemoryError: even a single sequence per forward pass did not fit
    """
    budget = token_budget
    while True:
        try:
            perplexities = []
            for start in range(0, len(prompts), batch_size):
                perplexities.extend(
                    tilted_perplexities(
                        base_model, first_model, tokenizer,
                        prompts[start : start + batch_size], factor,
                        max_length=int(block_size * 2), token_budget=budget,
                    )
                )
            return perplexities
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if budget <= 1024:
                raise
            budget //= 2
            print(
                f"##   {TColors.WARNING}out of memory, retrying at {budget} tokens per forward"
                f"{TColors.ENDC}"
            )


def measure_baseline(
    generations: range, block_size: int, name: str, real_data_fraction: float,
    prompts_for, batch_size: int, load_in_4bit: bool,
) -> list[Measurement]:
    """Scores every real checkpoint the collapse run produced.

    Args:
        generations (range): generation indices to look for
        block_size (int): the run's block size
        name (str): the model short name
        real_data_fraction (float): the mixture, part of the checkpoint names from generation 1 on
        prompts_for (callable): tokenizer -> the templated test prompts (each model brings its own
            tokenizer, and templating is a property of the tokenizer)
        batch_size (int): scoring batch size
        load_in_4bit (bool): quantize the scored model

    Returns:
        list[Measurement]: one per generation that exists on disk
    """
    results = []
    for generation in generations:
        mixture = mixture_suffix(real_data_fraction, generation)
        path = resolve_checkpoint(generation, block_size, name, mixture)
        if not path:
            print(f"##   {TColors.WARNING}generation {generation}: no checkpoint{TColors.ENDC}")
            continue
        model, tokenizer = load_scoring_model(path, block_size, load_in_4bit)
        perplexities = score(
            model, tokenizer, prompts_for(tokenizer), block_size, batch_size
        )
        results.append(
            Measurement.summarize(f"generation {generation}", path, perplexities)
        )
        print(
            f"##   generation {generation}: median {results[-1].median:8.2f}  "
            f"(IQR {results[-1].q25:.2f}-{results[-1].q75:.2f})  {os.path.basename(path)}"
        )
        del model
        torch.cuda.empty_cache()
    return results


def measure_surrogates(
    generations: range, block_size: int, name: str, method: str, model_specifier: str,
    prompts_for, batch_size: int, load_in_4bit: bool, token_budget: int,
) -> list[Measurement]:
    """Scores stage 2's surrogate for every generation.

    Stage 2 trains nothing, so its "model for generation g" is the surrogate that stands in for
    model_g: the tilt at n = g + 1, or the alpha-scaled adapter model_scaled_n{n}. That is the same
    indexing run_extrapolation.py and run_attack.py use, which is what makes the two curves
    comparable point by point — and at n = 1 both surrogates *are* model_0, so generation 0 is a
    free check that the alignment holds.

    The `logit` surrogate keeps both models resident and only rebinds the factor per generation,
    which is what makes the whole sweep cost two model loads instead of two per generation.

    Args:
        generations (range): generation indices, mapped to factors n = generation + 1
        block_size (int): the run's block size
        name (str): the model short name
        method (str): "logit" or "lora"
        model_specifier (str): the pristine base model
        prompts_for (callable): tokenizer -> the templated test prompts
        batch_size (int): scoring batch size
        load_in_4bit (bool): quantize the scored models
        token_budget (int): padded tokens per forward pass for the tilt

    Returns:
        list[Measurement]: one per generation whose surrogate could be built
    """
    results = []
    # neither anchor carries a mixture tag: generation 0 is shared by every mixture
    anchor = os.path.join(MODEL_PATH, f"model_0_bs{block_size}_{name}")

    if method == "logit":
        base_model, tokenizer = load_scoring_model(model_specifier, block_size, load_in_4bit)
        first_model, _ = load_scoring_model(
            f"{anchor}_fp16" if os.path.isdir(f"{anchor}_fp16") else anchor,
            block_size,
            load_in_4bit,
        )
        prompts = prompts_for(tokenizer)
        for generation in generations:
            perplexities = score_tilted(
                base_model, first_model, tokenizer, prompts, float(generation + 1),
                block_size, batch_size, token_budget,
            )
            results.append(
                Measurement.summarize(
                    f"generation {generation}",
                    f"base + {generation + 1:g} * (model_0 - base)",
                    perplexities,
                )
            )
            print(
                f"##   generation {generation} (n = {generation + 1}): "
                f"median {results[-1].median:8.2f}  "
                f"(IQR {results[-1].q25:.2f}-{results[-1].q75:.2f})"
            )
        del base_model, first_model
        torch.cuda.empty_cache()
        return results

    for generation in generations:
        path = os.path.join(
            MODEL_PATH, f"model_scaled_n{generation + 1}_bs{block_size}_{name}"
        )
        if not os.path.isdir(path):
            print(
                f"##   {TColors.WARNING}generation {generation}: no scaled adapter at "
                f"{path}{TColors.ENDC} — run run_extrapolation.py --method lora first"
            )
            continue
        model, tokenizer = load_scoring_model(path, block_size, load_in_4bit)
        perplexities = score(model, tokenizer, prompts_for(tokenizer), block_size, batch_size)
        results.append(Measurement.summarize(f"generation {generation}", path, perplexities))
        print(
            f"##   generation {generation} (n = {generation + 1}): "
            f"median {results[-1].median:8.2f}  "
            f"(IQR {results[-1].q25:.2f}-{results[-1].q75:.2f})  {os.path.basename(path)}"
        )
        del model
        torch.cuda.empty_cache()
    return results


def fit_factor(
    base_model, first_model, tokenizer, prompts: list, target: float, block_size: int,
    batch_size: int, token_budget: int, upper: float, steps: int,
) -> tuple[float, float]:
    """Finds the factor n whose surrogate matches one real checkpoint's median perplexity.

    ``n = generation + 1`` is an *indexing* convention — one fine-tuning step from base to model_0,
    extended g + 1 times — and this repo's measurements show it is not a calibration: the tilt
    degrades roughly tenfold per unit of n while the real collapse degrades a few percent per
    generation, so the two diverge by orders of magnitude. This searches for the n that actually
    reproduces the target instead of assuming it.

    Bisection, because the surrogate's perplexity is monotone in n (the tilt only sharpens) and
    monotone is all bisection needs — no gradient, no assumption about the shape of the curve. The
    search runs in log space since perplexity spans decades, and each step costs one scoring pass
    over the calibration rows, which is why those are a subsample rather than the full test set.

    Args:
        base_model: the pristine base model
        first_model: the generation-0 collapsed model
        tokenizer: shared tokenizer
        prompts (list): the calibration prompts
        target (float): the real checkpoint's median perplexity, the value to match
        block_size (int): truncation length is twice it
        batch_size (int): scoring batch size
        token_budget (int): padded tokens per forward pass
        upper (float): highest factor to consider
        steps (int): bisection steps

    Returns:
        tuple: (the fitted factor, the median perplexity it achieves). The factor is clamped to
            [1, upper]; at the bounds the target was outside what the tilt can reach
    """
    def median_at(factor: float) -> float:
        values = sorted(
            score_tilted(
                base_model, first_model, tokenizer, prompts, factor,
                block_size, batch_size, token_budget,
            )
        )
        return values[len(values) // 2]

    low, high = 1.0, float(upper)
    low_value, high_value = median_at(low), median_at(high)
    if target <= low_value:
        return low, low_value
    if target >= high_value:
        return high, high_value

    # False position rather than plain bisection, and in log-perplexity: the curve is smooth and
    # spans decades, so interpolating between the bracket's endpoints lands far closer than halving
    # it. Plain bisection over [1, num_generations + 1] resolves only (upper - 1) / 2^steps — at the
    # defaults that is 0.16, coarse enough that neighbouring generations were coming out with the
    # identical factor. The interpolation is clamped away from the endpoints so a badly curved
    # region cannot stall the bracket on one side.
    log_target = math.log(target)
    factor, achieved = high, high_value
    for _ in range(steps):
        span = math.log(high_value) - math.log(low_value)
        fraction = 0.5 if span <= 0 else (log_target - math.log(low_value)) / span
        factor = low + min(max(fraction, 0.05), 0.95) * (high - low)
        achieved = median_at(factor)
        # within a percent in log space is well inside the noise of a 96-row median
        if abs(math.log(achieved) - log_target) < 0.01:
            break
        if achieved < target:
            low, low_value = factor, achieved
        else:
            high, high_value = factor, achieved

    # the pair returned is one measurement, not a bracket midpoint paired with some other probe's
    # value: `achieved` is what the surrogate actually scored at `factor`
    return factor, achieved


def fit_scale(factors: dict) -> tuple[float, float]:
    """Fits ``n = 1 + scale * ln(1 + g)`` through the calibrated factors, least squares.

    One parameter, and it is the shape the measurements have rather than a chosen convenience: the
    fitted factors rise steeply from generation 0 to 1 and then flatten, which is what a logarithm
    does and what neither the linear ``g + 1`` nor a constant multiple of it can do. Fitting it
    matters because the generation an attacker aims at is the one nobody has a checkpoint of, so
    its factor cannot be measured and has to be predicted from the ones that can.

    Args:
        factors (dict): generation -> fitted factor, generation 0 excluded (ln 1 = 0 carries no
            information about the scale and the factor there is 1 by construction)

    Returns:
        tuple: (scale, the largest relative error of the fit over the given generations)
    """
    points = [(math.log1p(g), n - 1.0) for g, n in factors.items() if g > 0]
    if not points:
        return 0.0, 0.0
    # least squares through the origin: scale = sum(x*y) / sum(x*x)
    scale = sum(x * y for x, y in points) / sum(x * x for x, _ in points)
    worst = max(abs((1 + scale * x) - (1 + y)) / (1 + y) for x, y in points)
    return scale, worst


def calibrate_factors(
    baseline: list, block_size: int, name: str, model_specifier: str, prompts_for,
    batch_size: int, load_in_4bit: bool, token_budget: int, calibration_rows: int,
    steps: int, num_generations: int,
) -> dict:
    """Fits one factor per generation against the measured checkpoints, plus the scaling law.

    Loads the two anchors once and reuses them for every generation and every bisection step —
    the surrogate is a float on two resident models, which is what makes a search affordable at
    all.

    Args:
        baseline (list): the Measurements of the real checkpoints, whose medians are the targets
        block_size (int): the run's block size
        name (str): the model short name
        model_specifier (str): the pristine base model
        prompts_for (callable): tokenizer -> the templated test prompts
        batch_size (int): scoring batch size
        load_in_4bit (bool): quantize the anchors
        token_budget (int): padded tokens per forward pass for the tilt
        calibration_rows (int): prompts used inside the search
        steps (int): bisection steps per generation
        num_generations (int): highest generation index, the upper end of the search bracket

    Returns:
        dict: the calibration, with the fitted factor per generation, the fitted scale, and what
            the default n = g + 1 would have produced instead
    """
    anchor = os.path.join(MODEL_PATH, f"model_0_bs{block_size}_{name}")
    base_model, tokenizer = load_scoring_model(model_specifier, block_size, load_in_4bit)
    first_model, _ = load_scoring_model(
        f"{anchor}_fp16" if os.path.isdir(f"{anchor}_fp16") else anchor, block_size, load_in_4bit
    )
    # the search runs on a subsample — a step is one scoring pass and there are several per
    # generation — but the curve that ends up in the figure is measured on the *whole* test set,
    # the same rows the real checkpoints were scored on. Comparing a 96-row median against a
    # 512-row median would put a sampling difference into the residual
    full_prompts = prompts_for(tokenizer)
    prompts = full_prompts[:calibration_rows]

    factors, achieved = {}, {}
    for row in baseline:
        generation = int(row.label.split()[-1])
        factor, value = fit_factor(
            base_model, first_model, tokenizer, prompts, row.median, block_size,
            batch_size, token_budget, upper=float(num_generations + 1), steps=steps,
        )
        factors[generation] = round(factor, 4)
        achieved[generation] = round(value, 4)
        print(
            f"##   generation {generation}: target {row.median:8.2f}  "
            f"fitted n = {factor:5.2f} (default {generation + 1})  "
            f"reaches {value:8.2f}"
        )

    scale, worst = fit_scale(factors)
    print(
        f"##   fitted law: n = 1 + {scale:.3f} * ln(1 + g)   "
        f"(largest relative error {worst:.1%} over the calibrated generations)"
    )

    # and now the honest test of that one parameter: score the surrogate at the factor the *law*
    # predicts — not the per-generation fit, which reproduces its target by construction — on the
    # full test set, against the real checkpoints. This is what the third panel plots
    print(
        f"##   {TColors.OKBLUE}re-scoring the surrogate at the law's factor{TColors.ENDC} "
        f"on all {len(full_prompts)} test rows"
    )
    law_factors, law_perplexity = {}, {}
    for row in baseline:
        generation = int(row.label.split()[-1])
        factor = 1.0 + scale * math.log1p(generation)
        values = sorted(
            score_tilted(
                base_model, first_model, tokenizer, full_prompts, factor,
                block_size, batch_size, token_budget,
            )
        )
        median = values[len(values) // 2]
        law_factors[generation] = round(factor, 4)
        law_perplexity[generation] = round(median, 4)
        print(
            f"##   generation {generation}: n = {factor:5.2f} -> {median:8.2f}  "
            f"(real {row.median:8.2f}, off by {abs(median - row.median) / row.median:6.1%})"
        )

    del base_model, first_model
    torch.cuda.empty_cache()

    residuals = [
        abs(law_perplexity[int(r.label.split()[-1])] - r.median) / r.median for r in baseline
    ]
    return {
        "factors": factors,
        "achieved_perplexity": achieved,
        "target_perplexity": {int(r.label.split()[-1]): round(r.median, 4) for r in baseline},
        "scale": round(scale, 4),
        "worst_relative_error": round(worst, 4),
        "law": "n = 1 + scale * ln(1 + generation)",
        "law_factors": law_factors,
        "law_perplexity": law_perplexity,
        "law_worst_perplexity_error": round(max(residuals), 4) if residuals else None,
        "law_median_perplexity_error": (
            round(sorted(residuals)[len(residuals) // 2], 4) if residuals else None
        ),
    }


def cache_file(block_size: int, name: str, tag: str) -> str:
    """Path of the JSON the measurements are written to and --plot_only reads back."""
    return os.path.join(DATASET_PATH, f"test_perplexity_bs{block_size}_{name}{tag}.json")


def plot(payload: dict, plot_stem: str, usetex: bool) -> None:
    """Draws both lineages on one pair of axes.

    Log y, because perplexity spans decades once a model collapses and a linear axis would show
    one visible curve and one flat line at the bottom. The IQR band is drawn per series so the
    curves are not read as more precise than the samples behind them.

    Args:
        payload (dict): the measurement cache
        plot_stem (str): output path without the extension
        usetex (bool): render text with LaTeX, matching the other figures

    Returns:
        None
    """
    mpl.rcParams.update(
        {
            "text.usetex": usetex,
            "text.latex.preamble": r"\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "legend.fontsize": 14,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "pdf.compression": 9,
        }
    )

    # a second panel only when there is a calibration to show: what factor each generation
    # actually needed, against the n = g + 1 the pipeline assumes
    calibration = payload.get("calibration")
    scaled = (calibration or {}).get("law_perplexity")
    if calibration:
        # three rows: the surrogate as the pipeline indexes it, the same surrogate re-scored at the
        # fitted law's factor, and the factors themselves. The middle one is the result — whether
        # one parameter is enough to make the approximation track the collapse
        rows_count = 3 if scaled else 2
        figure, axes = plt.subplots(
            rows_count, 1, figsize=(10, 4 * rows_count),
            sharex=True, height_ratios=[3, 3, 2][:rows_count],
        )
        axis, lower = axes[0], axes[-1]
        middle = axes[1] if scaled else None
    else:
        figure, axis = plt.subplots(figsize=(10, 6))
        lower = middle = None

    series = (
        ("baseline", "collapse checkpoints", BASELINE_COLOR, "o"),
        ("extrapolation", f"{payload.get('method', 'logit')} surrogate", SURROGATE_COLOR, "s"),
    )
    for key, label, color, marker in series:
        rows = payload.get(key) or []
        if not rows:
            continue
        generations = [int(row["label"].split()[-1]) for row in rows]
        axis.plot(
            generations,
            [row["median"] for row in rows],
            marker=marker,
            markersize=9,
            linewidth=2,
            color=color,
            label=label,
        )
        # the band is the interquartile range over the test rows, not an error bar: it says how
        # differently the model treats different human responses, which is the spread that grows
        # as the model collapses
        axis.fill_between(
            generations,
            [row["q25"] for row in rows],
            [row["q75"] for row in rows],
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    anchor = payload.get("base_model")
    if anchor:
        axis.axhline(
            anchor["median"],
            color=ANCHOR_COLOR,
            linestyle="--",
            linewidth=2,
            # deliberately not called an upper bound: model_0 is fine tuned on this dataset's own
            # distribution, so it scores *better* on the held-out rows than the un-fine-tuned base.
            # The collapse is the rise from generation 0 upward, not the distance to this line
            label="base model, before fine-tuning",
        )

    axis.set_yscale("log")
    if lower is None:
        axis.set_xlabel("collapse generation")
    axis.set_ylabel("test perplexity (median)")
    axis.set_xticks([int(row["label"].split()[-1]) for row in (payload.get("baseline") or [])])
    axis.set_title(
        f"Utility on held-out human data\n({payload['model'].replace('_', ' ')}, "
        f"{payload['mixture_label']})"
    )
    axis.legend(loc="upper left")
    for spine in axis.spines.values():
        spine.set_color("black")

    if middle is not None:
        real = {int(row["label"].split()[-1]): row["median"] for row in (payload["baseline"] or [])}
        scaled_ppl = {int(k): v for k, v in scaled.items()}
        generations = sorted(scaled_ppl)
        middle.plot(
            generations, [real[g] for g in generations],
            marker="o", markersize=9, linewidth=2, color=BASELINE_COLOR,
            label="collapse checkpoints",
        )
        middle.plot(
            generations, [scaled_ppl[g] for g in generations],
            marker="s", markersize=9, linewidth=2, color=SURROGATE_COLOR,
            label="logit surrogate at the fitted factor",
        )
        # linear, not log: at the fitted factor both curves live in one decade, and the residual
        # between them is the whole point of this panel — a log axis would hide it
        worst = calibration.get("law_worst_perplexity_error")
        middle.set_ylabel("test perplexity\n(median, scaled)")
        middle.legend(loc="upper left")
        if worst is not None:
            # inside the axes: as a title it sits between the two panels and reads like a caption
            # for the one above
            median_error = calibration.get("law_median_perplexity_error")
            note = (
                f"deviation from the real curve: {median_error:.0%} median, {worst:.0%} worst"
                if not usetex else
                f"deviation from the real curve: {median_error * 100:.0f}\\% median, "
                f"{worst * 100:.0f}\\% worst"
            )
            middle.annotate(
                note, xy=(0.98, 0.06), xycoords="axes fraction", ha="right", fontsize=14
            )
        for spine in middle.spines.values():
            spine.set_color("black")

    if lower is not None:
        fitted = {int(k): v for k, v in calibration["factors"].items()}
        generations = sorted(fitted)
        lower.plot(
            generations,
            [fitted[g] for g in generations],
            marker="D",
            markersize=8,
            linewidth=2,
            color=SURROGATE_COLOR,
            label="fitted to the real perplexity",
        )
        scale = calibration.get("scale")
        if scale:
            lower.plot(
                generations,
                [1 + scale * math.log1p(g) for g in generations],
                linestyle=":",
                linewidth=2,
                color=SURROGATE_COLOR,
                label=f"$n = 1 + {scale:.2f}\\,\\ln(1+g)$" if usetex
                else f"n = 1 + {scale:.2f} ln(1+g)",
            )
        lower.plot(
            generations,
            [g + 1 for g in generations],
            linestyle="--",
            linewidth=2,
            color=ANCHOR_COLOR,
            label="$n = g + 1$ (assumed)" if usetex else "n = g + 1 (assumed)",
        )
        lower.set_xlabel("collapse generation")
        lower.set_ylabel("extrapolation\nfactor $n$" if usetex else "extrapolation\nfactor n")
        lower.set_xticks(generations)
        lower.legend(loc="upper left", ncol=1)
        for spine in lower.spines.values():
            spine.set_color("black")

    figure.tight_layout()
    os.makedirs(os.path.dirname(plot_stem) or ".", exist_ok=True)
    figure.savefig(f"{plot_stem}.pdf")
    figure.savefig(f"{plot_stem}.png", dpi=200)
    plt.close(figure)


def main(
    num_generations: int = 10,
    block_size: int = 512,
    dataset_size: int = 0,
    test_size: int = 512,
    perplexity_batch_size: int = 16,
    method: str = "logit",
    surrogate_tokens_per_forward: int = SURROGATE_TOKENS_PER_FORWARD,
    calibrate: bool = False,
    calibration_rows: int = 128,
    calibration_steps: int = 6,
    real_data_fraction: float = 0.0,
    model_size: str = "",
    model_specifier: str = "",
    load_in_4bit: bool = False,
    plot_only: bool = False,
    no_usetex: bool = False,
    path: str = "",
) -> None:
    """Measures both lineages on the held-out test set and plots them together.

    Args:
        num_generations (int): number of generations the run produced
        block_size (int): the run's block size, part of every artifact name
        dataset_size (int): the --dataset_size the collapse runs were given; decides whether an
            untouched tail of the corpus exists to test on
        test_size (int): held-out rows to score, 0 for all of them
        perplexity_batch_size (int): prompts per scoring batch
        method (str): which stage 2 surrogate to score alongside, "logit" or "lora"
        surrogate_tokens_per_forward (int): padded tokens per forward pass for the tilt, halved
            automatically on an out-of-memory error
        calibrate (bool): after measuring both lineages, search for the factor that makes the
            surrogate match each real checkpoint's perplexity, and fit a scaling law through the
            result. Writes a calibration other stages can read
        calibration_rows (int): test rows used inside the search, a subsample so that a bisection
            step costs a fraction of a full pass
        calibration_steps (int): bisection steps per generation
        real_data_fraction (float): the mixture the run used, part of the checkpoint names
        model_size (str): parameter count off the Qwen2.5-Coder ladder
        model_specifier (str): the base model the run collapsed
        load_in_4bit (bool): quantize the scored models
        plot_only (bool): replot from the cache without loading a model
        no_usetex (bool): render without LaTeX
        path (str): root holding generated_datasets/ and model_outputs/

    Returns:
        None

    Raises:
        SystemExit: --method data, or --plot_only without a cache
    """
    global DATASET_PATH, MODEL_PATH
    if path:
        DATASET_PATH = os.path.join(path, "generated_datasets/")
        MODEL_PATH = os.path.join(path, "model_outputs/")

    if method == "data":
        raise SystemExit(
            f"{TColors.FAIL}--method data cannot be scored this way{TColors.ENDC}\nThe data-space "
            f"surrogate is the base model with a narrowed sampling support, and a perplexity is "
            f"teacher forced — it never samples. Scoring it would return the base model's own "
            f"numbers and draw a flat line. Use --method logit or lora."
        )

    specifier = resolve_model_specifier(model_size, model_specifier)
    name = specifier.split("/")[-1]
    # the ladder rung, read back off the *resolved* id rather than off --model_size, so the line
    # says the same thing whichever of the two flags named the model
    size_label = model_size_label(specifier) or "off the ladder"
    tag = mixture_tag(real_data_fraction)
    generations = range(num_generations)
    stem = f"plots/test_perplexity_bs{block_size}_{name}{tag}"

    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Utility on held-out human data"
        f"{TColors.ENDC}"
    )

    if plot_only:
        if not os.path.isfile(cache_file(block_size, name, tag)):
            raise SystemExit(
                f"no cache at {cache_file(block_size, name, tag)} — run without --plot_only first"
            )
        with open(cache_file(block_size, name, tag), encoding="utf-8") as handle:
            payload = json.load(handle)
        plot(payload, stem, usetex=not no_usetex)
        print(f"##   replotted from the cache -> {TColors.HEADER}{stem}.<png,pdf>{TColors.ENDC}\n")
        return

    test_set, description, leaky = held_out_test_set(dataset_size, test_size)
    print(f"##   test set: {len(test_set)} rows, {description}")
    if leaky and real_data_fraction > 0:
        print(
            f"##   {TColors.WARNING}--real_data_fraction {real_data_fraction:g} draws its real "
            f"rows from the whole corpus, so part of this slice may have re-entered training in "
            f"later generations{TColors.ENDC}"
        )
    print(
        f"##   model:    {specifier} ({size_label})"
        f"{tag and '  (mixture ' + tag[1:] + ')'}"
    )

    def prompts_for(tokenizer) -> list:
        return format_scoring_prompts(
            tokenizer, test_set["instruction"], test_set["response"]
        )

    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Pristine base model (no collapse){TColors.ENDC}")
    base_model, base_tokenizer = load_scoring_model(specifier, block_size, load_in_4bit)
    anchor = Measurement.summarize(
        "base",
        specifier,
        score(
            base_model, base_tokenizer, prompts_for(base_tokenizer),
            block_size, perplexity_batch_size,
        ),
    )
    print(f"##   median {anchor.median:8.2f}  (IQR {anchor.q25:.2f}-{anchor.q75:.2f})")
    del base_model
    torch.cuda.empty_cache()

    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Baseline (real collapse){TColors.ENDC}")
    baseline = measure_baseline(
        generations, block_size, name, real_data_fraction,
        prompts_for, perplexity_batch_size, load_in_4bit,
    )

    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Extrapolation ({method} surrogate){TColors.ENDC}")
    surrogates = measure_surrogates(
        generations, block_size, name, method, specifier,
        prompts_for, perplexity_batch_size, load_in_4bit, surrogate_tokens_per_forward,
    )

    calibration = {}
    if calibrate and method == "logit":
        print(
            f"\n## {TColors.OKBLUE}{TColors.BOLD}Calibrating the extrapolation factor"
            f"{TColors.ENDC} — which n reproduces each checkpoint's perplexity"
        )
        calibration = calibrate_factors(
            baseline, block_size, name, specifier, prompts_for, perplexity_batch_size,
            load_in_4bit, surrogate_tokens_per_forward, calibration_rows, calibration_steps,
            num_generations,
        )
        calibration.update(
            {
                "model": name,
                "block_size": block_size,
                "real_data_fraction": real_data_fraction,
                "test_set": description,
                "calibration_rows": min(calibration_rows, len(test_set)),
            }
        )
        with open(
            factor_calibration_file(DATASET_PATH, block_size, name, tag), "w", encoding="utf-8"
        ) as handle:
            json.dump(calibration, handle, indent=2)
        print(
            f"##   {TColors.OKGREEN}saved{TColors.ENDC} "
            f"{factor_calibration_file(DATASET_PATH, block_size, name, tag)}"
        )
    elif calibrate:
        print(
            f"\n## {TColors.WARNING}--calibrate applies to the logit surrogate only, skipped for "
            f"--method {method}{TColors.ENDC}"
        )

    payload = {
        "model": name,
        "model_specifier": specifier,
        "block_size": block_size,
        "real_data_fraction": real_data_fraction,
        "mixture_label": (
            "no data mixture" if real_data_fraction <= 0
            else f"real data fraction {real_data_fraction:g}"
        ),
        "test_set": description,
        "test_set_rows": len(test_set),
        "test_set_may_leak": leaky and real_data_fraction > 0,
        "method": method,
        "base_model": anchor.__dict__,
        "baseline": [row.__dict__ for row in baseline],
        "extrapolation": [row.__dict__ for row in surrogates],
        "calibration": calibration or None,
    }
    os.makedirs(DATASET_PATH, exist_ok=True)
    with open(cache_file(block_size, name, tag), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # generation 0 must agree: at n = 1 the surrogate *is* model_0, so a gap there is a bug
    # upstream of this script rather than a property of the extrapolation
    if baseline and surrogates and baseline[0].label == surrogates[0].label:
        gap = abs(baseline[0].median - surrogates[0].median) / max(baseline[0].median, 1e-9)
        if gap > 0.01:
            print(
                f"\n## {TColors.WARNING}generation 0 differs by {gap:.1%} between the two "
                f"lineages{TColors.ENDC}, but n = 1 is model_0 itself — check that the anchor "
                f"model_0_bs{block_size}_{name} is the one the collapse run started from"
            )

    plot(payload, stem, usetex=not no_usetex)
    print(
        f"\n## {TColors.OKBLUE}{TColors.BOLD}Saved the figure under: "
        f"{TColors.HEADER}{stem}.<png,pdf>{TColors.ENDC}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Saved the measurements under: "
        f"{TColors.HEADER}{cache_file(block_size, name, tag)}{TColors.ENDC}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility of every produced model on held-out rows of the original dataset"
    )
    parser.add_argument("--num_generations", "-ng", type=int, default=10,
                        help="number of generations the run produced (default: 10)")
    parser.add_argument("--block_size", "-bs", type=int, default=512,
                        help="the run's block size, part of every artifact name (default: 512)")
    parser.add_argument("--dataset_size", "-ds", type=int, default=0,
                        help="the --dataset_size the collapse runs were given. Below the corpus "
                        "size it leaves an untouched tail, which is then the test set; 0 means "
                        "the runs used everything and the validation slice is used instead "
                        "(default: 0)")
    parser.add_argument("--test_size", "-ts", type=int, default=512,
                        help="held-out rows to score, 0 for all of them (default: 512)")
    parser.add_argument("--perplexity_batch_size", "-pbs", type=int, default=16,
                        help="prompts per scoring batch (default: 16)")
    parser.add_argument("--method", "-m", type=str, default="logit",
                        choices=["logit", "lora", "data"],
                        help="which stage 2 surrogate to score alongside the real checkpoints. "
                        "'data' is rejected with an explanation (default: logit)")
    parser.add_argument("--surrogate_tokens_per_forward", "-stf", type=int,
                        default=SURROGATE_TOKENS_PER_FORWARD,
                        help=f"padded tokens per forward pass when scoring the tilt, which runs "
                        f"two models at once. Halved automatically on an out-of-memory error "
                        f"(default: {SURROGATE_TOKENS_PER_FORWARD})")
    parser.add_argument("--calibrate", "-c", action="store_true",
                        help="fit the extrapolation factor to the real checkpoints: search the n "
                        "whose surrogate matches each generation's perplexity, fit "
                        "n = 1 + scale * ln(1 + g) through the result, and write it where the "
                        "other stages can read it (run_attack.py -sf calibrated)")
    parser.add_argument("--calibration_rows", "-cr", type=int, default=128,
                        help="test rows used inside the factor search, a subsample so a bisection "
                        "step costs a fraction of a full pass (default: 128)")
    parser.add_argument("--calibration_steps", "-cs", type=int, default=6,
                        help="bisection steps per generation (default: 6)")
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0,
                        help="the mixture the collapse run used; part of the checkpoint names "
                        "from generation 1 on (default: 0.0)")
    parser.add_argument("--load_in_4bit", "-q4", action="store_true",
                        help="quantize the scored models. Off by default: a quantized model puts "
                        "quantization noise into the measured perplexity")
    parser.add_argument("--plot_only", "-po", action="store_true",
                        help="replot from the cached measurements without loading a model")
    parser.add_argument("--no_usetex", action="store_true",
                        help="render without LaTeX, for a machine with no TeX install")
    parser.add_argument("--path", "-p", type=str, default="",
                        help="root holding generated_datasets/ and model_outputs/ "
                        "(default: current directory)")
    add_model_arguments(parser, role="the model the run collapsed")
    args = parser.parse_args()
    main(**vars(args))
