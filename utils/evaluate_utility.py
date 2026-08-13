"""Utility of every model the pipeline produces, on a held-out slice of the *original* dataset.

The perplexity histograms of stages 1 and 2 measure the **corpora**: a fixed scorer (the pristine
base model) reads what each generation wrote, so the statistic says how degenerate the *text* has
become. This script measures the other direction — each produced **model** reads a fixed slice of
human data — which is the utility question: has the model lost the ability to model real code, and
does the extrapolation surrogate lose it the same way the real collapse does.

Both lineages are scored on the same held-out rows, with the same statistic
(``utils.perplexity.sample_perplexities``, the one the histograms plot), and drawn in one figure so
they are comparable:

* **baseline** — the real checkpoints ``model_{g}_bs{bs}_{name}{mix}``, i.e. the actual collapse.
* **extrapolation** — stage 2 trains nothing, so its model *for generation g* is the surrogate that
  stands in for ``model_g``: the tilt ``base + n * (model_0 - base)`` under ``--method logit``, or
  the alpha-scaled adapter ``model_scaled_n{n}`` under ``--method lora``, both at ``n = g + 1``.
  That is the same indexing run_extrapolation.py and run_attack.py use, and it is what makes the
  two curves comparable point by point.

**Reading the figure**: lower is better, and the dashed line is the *un-fine-tuned* base model, not
a quality ceiling. Generation 0 sits below it because it is fine tuned on this very dataset's
distribution; collapse is the rise from generation 0 upward.

**Generation 0 is a built-in check of that alignment.** At n = 1 both surrogates reduce to the real
``model_0`` exactly — the tilt to ``base + 1 * (model_0 - base)``, the scaled adapter to alpha x 1 —
so the two curves must meet there. If they do not, something upstream of this script is wrong.

``--method data`` is rejected, for the same reason run_attack.py rejects it as an attack surrogate:
the data-space surrogate is the base model with a narrowed *sampling* support, and this measurement
is a teacher-forced cross entropy that never samples, so it would score exactly the base model and
report a flat line that says nothing about the surrogate.

Like utils/calibrate_surrogate.py, this is a user-invoked module with a main(), not one of the
worker modules beside it:

    python -m utils.evaluate_utility -p . -ng 10 -bs 512
    python -m utils.evaluate_utility -p . -ng 10 -bs 512 -rdf 0.1 --method lora
    python -m utils.evaluate_utility -p . -ng 10 --plot_only     # replot from the cache
"""

# unsloth first, before torch/transformers: it patches them at import time, and the checkpoints
# here are the adapters it wrote. Same rule as the worker modules, hence the pylint exception
from unsloth import FastLanguageModel

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset

from utils.colors import TColors
from utils.extrapolation import extrapolate_logits
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import mixture_suffix, mixture_tag
from utils.perplexity import (
    MAX_TOKENS_PER_FORWARD,
    format_scoring_prompts,
    sample_perplexities,
)

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
# the split run_baseline.py's make_splits() uses, mirrored here so the fallback test set below is
# exactly the slice its training never saw
TRAIN_FRACTION: float = 0.9

# the two series of the figure, in the repo's colorblind-safe palette. Verified rather than
# assumed: OKLab dE 37.9 at normal vision, 38.2 / 29.5 / 33.7 under simulated deuteranopia /
# protanopia / tritanopia, against a target of 8
BASELINE_COLOR: str = "#006BA4"
SURROGATE_COLOR: str = "#FF800E"
ANCHOR_COLOR: str = "#595959"


@dataclass
class LogitsOutput:
    """The one field sample_perplexities reads off a forward pass."""

    logits: torch.Tensor


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


class TiltedModel(torch.nn.Module):
    """The logit surrogate as something ``sample_perplexities`` can score.

    ``ExtrapolatedModel`` in run_attack.py is the same tilt built for GCG, which needs only the
    logits of the last positions; a perplexity needs the whole sequence, so this wrapper returns
    full logits instead. The arithmetic is ``utils.extrapolation.extrapolate_logits`` either way —
    the single definition stage 2 generates its datasets with.

    Attributes:
        base: the pristine base model
        first: the generation-0 collapsed model
        factor: the extrapolation factor n
    """

    def __init__(self, base, first, factor: float):
        super().__init__()
        self.base = base
        self.first = first
        self.factor = float(factor)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_):
        """Returns an object with a ``.logits`` field, which is all the scorer touches."""
        base_out = self.base(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache
        )
        first_out = self.first(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache
        )
        tilted = extrapolate_logits(base_out.logits, first_out.logits, self.factor)
        return LogitsOutput(logits=tilted.to(base_out.logits.dtype))


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
        token_budget (int): padded tokens per forward pass. Halved for the tilted model, which
            materializes two vocabulary-sized logit tensors instead of one

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
    prompts_for, batch_size: int, load_in_4bit: bool,
) -> list[Measurement]:
    """Scores stage 2's surrogate for every generation.

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

    Returns:
        list[Measurement]: one per generation whose surrogate could be built
    """
    results = []
    anchor = os.path.join(MODEL_PATH, f"model_0_bs{block_size}_{name}")

    if method == "logit":
        # neither anchor carries a mixture tag: generation 0 is shared by every mixture
        base_model, tokenizer = load_scoring_model(model_specifier, block_size, load_in_4bit)
        first_model, _ = load_scoring_model(
            f"{anchor}_fp16" if os.path.isdir(f"{anchor}_fp16") else anchor,
            block_size,
            load_in_4bit,
        )
        surrogate = TiltedModel(base_model, first_model, 1.0)
        prompts = prompts_for(tokenizer)
        for generation in generations:
            surrogate.factor = float(generation + 1)
            perplexities = score(
                surrogate, tokenizer, prompts, block_size, batch_size,
                # two models' logits are alive at once here, and the tilt upcasts to float32,
                # so the same token budget as a single model would need several times its memory
                token_budget=MAX_TOKENS_PER_FORWARD // 4,
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
        del base_model, first_model, surrogate
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
        results.append(
            Measurement.summarize(f"generation {generation}", path, perplexities)
        )
        print(
            f"##   generation {generation} (n = {generation + 1}): "
            f"median {results[-1].median:8.2f}  "
            f"(IQR {results[-1].q25:.2f}-{results[-1].q75:.2f})  {os.path.basename(path)}"
        )
        del model
        torch.cuda.empty_cache()
    return results


def cache_file(block_size: int, name: str, tag: str) -> str:
    """Path of the JSON the measurements are written to and --plot_only reads back."""
    return os.path.join(DATASET_PATH, f"utility_bs{block_size}_{name}{tag}.json")


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

    figure, axis = plt.subplots(figsize=(10, 6))
    series = (
        ("baseline", "real collapse", BASELINE_COLOR, "o"),
        ("extrapolation", f"{payload['method']} surrogate", SURROGATE_COLOR, "s"),
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
        method (str): stage 2 surrogate to score, "logit" or "lora"
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
    tag = mixture_tag(real_data_fraction)
    generations = range(num_generations)
    stem = f"plots/utility_bs{block_size}_{name}{tag}"

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
    print(f"##   model:    {specifier}{tag and '  (mixture ' + tag[1:] + ')'}")

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
        prompts_for, perplexity_batch_size, load_in_4bit,
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
        "method": method,
        "test_set": description,
        "test_set_rows": len(test_set),
        "test_set_may_leak": leaky and real_data_fraction > 0,
        "base_model": anchor.__dict__,
        "baseline": [row.__dict__ for row in baseline],
        "extrapolation": [row.__dict__ for row in surrogates],
    }
    os.makedirs(DATASET_PATH, exist_ok=True)
    with open(cache_file(block_size, name, tag), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # generation 0 must agree: at n = 1 both surrogates *are* model_0, so a gap there is a bug
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
                        help="which stage 2 surrogate to score against the real checkpoints. "
                        "'data' is rejected with an explanation (default: logit)")
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
