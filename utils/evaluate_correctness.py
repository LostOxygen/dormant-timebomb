"""Can each collapse generation still write code that *works*? HumanEval pass@1 per checkpoint.

The companion of utils/evaluate_perplexity.py, on the same checkpoints and with the same shape of
figure, but measuring a different thing. Perplexity asks how well a model still *models* human
code: it is teacher forced, continuous, and it moves smoothly generation by generation. This script
asks whether the model can still *produce* a correct answer — the output is generated, extracted
and executed against unit tests, and the verdict is binary per problem. A model can drift a long
way in perplexity and still solve the same problems, or hold its perplexity and stop producing
runnable code; the two curves are worth reading side by side for exactly that reason.

**The benchmark is HumanEval**, not the collapse corpus. The corpus
(``bigcode/self-oss-instruct-sc2-exec-filter-50k``) has instructions and responses but ships no
tests, so there is no ground truth to execute against it — "correct" would have to fall back to
string similarity, which is what this script exists to avoid. HumanEval brings 164 problems with
unit tests and a named entry point, which is what makes the verdict behavioural.

**pass@1 with greedy decoding.** One completion per problem, temperature 0, so the number is
deterministic and reruns are comparable; the spread across problems is already the interesting
variance, and sampling k completions would multiply the cost by k for a statistic this pipeline
does not otherwise use. What is measured is the fraction of problems whose extracted code defines
the entry point and passes every assertion.

Prompted through the same chat template and system prompt the collapse training used
(``utils.perplexity.SCORING_SYSTEM_PROMPT``), because a model fine-tuned in that format answers a
raw completion-style prompt differently, and the question here is what the *pipeline's* models can
do, not what they could do under a better prompt.

Extraction and execution are ``utils.execution``'s, the same definitions run_attack.py decides a
selective hit with — a model that answers with prose around its code is not counted as wrong for
that reason, and a candidate that hangs is a timeout rather than a hang of this script.

Like utils/calibrate_surrogate.py, this is a user-invoked module with a main():

    python -m utils.evaluate_correctness -p . -ng 10 -bs 512
    python -m utils.evaluate_correctness -p . -ng 10 -bs 512 -rdf 0.1 --limit 40
    python -m utils.evaluate_correctness -p . -ng 10 --plot_only
"""

# unsloth first, before torch/transformers: it patches them at import time, and the checkpoints
# here are the adapters it wrote
from unsloth import FastLanguageModel

import argparse
import json
import os
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from utils.colors import TColors
from utils.execution import extract_code, run_tests
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import mixture_suffix, mixture_tag
from utils.perplexity import SCORING_SYSTEM_PROMPT
from utils.utils import clear_inherited_max_length

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
BENCHMARK: str = "openai/openai_humaneval"

# The figure is a composition, not a curve: what a generation *did* with each problem, stacked.
# Four outcomes, ordered from best to worst so the stack reads top-down as degradation, and the
# question the split answers is whether a failure is a wrong answer or no answer at all.
#
# The raw statuses are five (utils.execution). "fail" and "fail_exception" are grouped here: both
# mean the code ran and the tests rejected it, they differ only in how the test suite gave up, and
# two orange steps far enough apart to be distinguished do not exist in this palette — the pair
# measured OKLab dE 13.5 at normal vision against a floor of 15. The console table and the JSON
# keep them apart; only the figure groups them
OUTCOMES: tuple = (
    ("pass", "pass", ("pass",), "#006BA4"),
    ("wrong", "wrong answer", ("fail", "fail_exception"), "#FF800E"),
    ("timeout", "timeout", ("timeout",), "#ABABAB"),
    ("broken", "no runnable code", ("error", "crash"), "#A9373B"),
)


@dataclass
class Result:
    """One model's verdict on the whole benchmark."""

    label: str
    source: str
    n_problems: int
    n_passed: int
    statuses: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Fraction of problems solved, 0 when nothing was scored."""
        return self.n_passed / self.n_problems if self.n_problems else 0.0

    def status_counts(self) -> dict:
        """How the failures failed, which separates "wrong answer" from "not code at all"."""
        counts: dict = {}
        for status in self.statuses.values():
            counts[status] = counts.get(status, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))


def build_prompt(tokenizer, problem: dict) -> str:
    """Renders one HumanEval problem the way the collapse training templated its data.

    The stub is handed over whole — signature, docstring and all — and the model is asked to
    complete it. The instruction is spelled out because these are -Instruct checkpoints answering
    in a chat template: given only the stub they tend to explain it rather than implement it.

    Args:
        tokenizer: the tokenizer whose chat template renders the prompt
        problem (dict): one row of the benchmark, with "prompt" and "entry_point"

    Returns:
        str: the templated prompt, ending in the generation prefix
    """
    instruction = (
        "Complete the following Python function. Reply with the complete function definition "
        "and nothing else.\n\n" + problem["prompt"]
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompts: list, max_new_tokens: int, batch_size: int) -> list:
    """Greedily completes every prompt, batched.

    Left padding, unlike the perplexity scorer's right padding: this is generation, and a
    right-padded batch would have the model continue from pad tokens instead of from the prompt.

    Args:
        model: the model to generate with
        tokenizer: its tokenizer
        prompts (list): the templated prompts
        max_new_tokens (int): generation budget per problem
        batch_size (int): prompts per generate() call

    Returns:
        list: one raw completion per prompt, prompt text removed
    """
    tokenizer.padding_side = "left"
    completions = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=(len(prompts) + batch_size - 1) // batch_size,
        desc="generating",
        leave=False,
    ):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        # the batch is left padded, so the prompt is a constant-length prefix and the continuation
        # starts at the same index for every row
        prompt_length = inputs["input_ids"].shape[1]
        completions.extend(
            tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=False)
        )
    return completions


def score_model(
    model, tokenizer, problems: Dataset, prompts: list, label: str, source: str,
    max_new_tokens: int, batch_size: int, exec_timeout: float,
) -> Result:
    """Generates, extracts and executes for every problem, and tallies the verdicts.

    Args:
        model: the model under test
        tokenizer: its tokenizer
        problems (Dataset): the benchmark rows
        prompts (list): the templated prompts, aligned with `problems`
        label (str): how this model is named in the cache and the figure
        source (str): the checkpoint path, for the record
        max_new_tokens (int): generation budget per problem
        batch_size (int): prompts per generate() call
        exec_timeout (float): wall-clock limit per candidate

    Returns:
        Result: the pass count and the per-problem statuses
    """
    completions = generate(model, tokenizer, prompts, max_new_tokens, batch_size)

    statuses = {}
    for problem, completion in zip(problems, completions):
        code = extract_code(completion)
        # the benchmark's own test source plus the call that runs it, exactly as HumanEval
        # defines it; run_tests checks the entry point exists before executing any of it
        tests = problem["test"] + f"\n\ncheck({problem['entry_point']})\n"
        statuses[problem["task_id"]] = run_tests(
            code, tests, problem["entry_point"], exec_timeout
        )

    return Result(
        label=label,
        source=source,
        n_problems=len(statuses),
        n_passed=sum(1 for status in statuses.values() if status == "pass"),
        statuses=statuses,
    )


def resolve_checkpoint(generation: int, block_size: int, name: str, mixture: str) -> str:
    """Locates one collapse checkpoint, merged copy first, adapter second.

    Same resolution as utils/evaluate_perplexity.py, so the two figures describe the same models.

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


def load_model(path: str, block_size: int, load_in_4bit: bool):
    """Loads one model for generation, with a pad token and left padding set up.

    Args:
        path (str): checkpoint directory or a Hugging Face repo id
        block_size (int): the run's block size; the context is twice it, as everywhere else
        load_in_4bit (bool): quantize

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
    # generate() below always passes max_new_tokens, so the max_length the checkpoint ships in its
    # generation_config only buys a warning line per batch. See clear_inherited_max_length
    clear_inherited_max_length(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def cache_file(block_size: int, name: str, tag: str) -> str:
    """Path of the JSON the verdicts are written to and --plot_only reads back."""
    return os.path.join(DATASET_PATH, f"test_correctness_bs{block_size}_{name}{tag}.json")


def outcome_shares(statuses: dict) -> dict:
    """Groups one model's per-problem statuses into the figure's four outcomes, as fractions.

    Args:
        statuses (dict): task_id -> status, as utils.execution.run_tests reported it

    Returns:
        dict: outcome key -> share of the problems, summing to 1
    """
    total = max(len(statuses), 1)
    shares = {}
    for key, _, raw_statuses, _ in OUTCOMES:
        shares[key] = sum(1 for s in statuses.values() if s in raw_statuses) / total
    return shares


def plot(payload: dict, plot_stem: str, usetex: bool) -> None:
    """Draws what every model did with the benchmark, as a stacked composition per generation.

    A pass@1 line would answer "how many did it solve" and stop there. The stack answers the
    follow-up that decides what the number *means*: of the ones it did not solve, how many were
    wrong answers and how many were not code at all. Those two degrade at different times — wrong
    answers dominate early, unusable output takes over as the collapse deepens — and a single rate
    hides the transition completely.

    The base model gets its own bar, set apart from the generations by a gap rather than drawn as
    a reference line: a horizontal line through a stacked bar cannot be read against anything.

    Args:
        payload (dict): the verdict cache
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
            "legend.fontsize": 13,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "pdf.compression": 9,
        }
    )
    percent = r"\%" if usetex else "%"

    rows = payload.get("baseline") or []
    anchor = payload.get("base_model")

    # the base model sits at x = -1.5, leaving a visible gap before generation 0: it is not part of
    # the collapse sequence and should not read as its first step
    entries = ([(-1.5, "base", anchor)] if anchor else []) + [
        (int(row["label"].split()[-1]), row["label"].split()[-1], row) for row in rows
    ]

    figure, axis = plt.subplots(figsize=(11, 6))
    bottoms = [0.0] * len(entries)
    for key, label, _, color in OUTCOMES:
        heights = [outcome_shares(entry[2]["statuses"])[key] for entry in entries]
        axis.bar(
            [entry[0] for entry in entries],
            heights,
            bottom=bottoms,
            width=0.7,
            color=color,
            label=label,
            # a hairline of surface between the segments, so the boundaries stay readable where a
            # slice is thin instead of merging with its neighbour
            edgecolor="white",
            linewidth=1.2,
        )
        bottoms = [carry + height for carry, height in zip(bottoms, heights)]

    axis.set_ylim(0, 1)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(["0"] + [f"{int(v * 100)}{percent}" for v in (0.2, 0.4, 0.6, 0.8, 1.0)])
    axis.set_xticks([entry[0] for entry in entries])
    axis.set_xticklabels([entry[1] for entry in entries])
    axis.set_xlabel("collapse generation")
    axis.set_ylabel(f"share of {payload['n_problems']} problems")
    axis.set_title(
        f"What the produced code does\n({payload['model'].replace('_', ' ')}, "
        f"{payload['mixture_label']})"
    )
    # outside the axes: every bar is full height, so any in-figure legend would cover data
    axis.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="outcome"
    )
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
    limit: int = 0,
    max_new_tokens: int = 384,
    generation_batch_size: int = 16,
    exec_timeout: float = 10.0,
    real_data_fraction: float = 0.0,
    model_size: str = "",
    model_specifier: str = "",
    load_in_4bit: bool = False,
    plot_only: bool = False,
    no_usetex: bool = False,
    path: str = "",
) -> None:
    """Scores every collapse checkpoint on the benchmark and plots pass@1 over the generations.

    Args:
        num_generations (int): number of generations the run produced
        block_size (int): the run's block size, part of every artifact name
        limit (int): score only the first N problems, 0 for all 164
        max_new_tokens (int): generation budget per problem
        generation_batch_size (int): problems per generate() call
        exec_timeout (float): wall-clock limit per candidate
        real_data_fraction (float): the mixture the run used, part of the checkpoint names
        model_size (str): parameter count off the Qwen2.5-Coder ladder
        model_specifier (str): the base model the run collapsed
        load_in_4bit (bool): quantize the models under test
        plot_only (bool): replot from the cache without loading a model
        no_usetex (bool): render without LaTeX
        path (str): root holding generated_datasets/ and model_outputs/

    Returns:
        None

    Raises:
        SystemExit: --plot_only without a cache
    """
    global DATASET_PATH, MODEL_PATH
    if path:
        DATASET_PATH = os.path.join(path, "generated_datasets/")
        MODEL_PATH = os.path.join(path, "model_outputs/")

    specifier = resolve_model_specifier(model_size, model_specifier)
    name = specifier.split("/")[-1]
    # the ladder rung, read back off the *resolved* id rather than off --model_size, so the line
    # says the same thing whichever of the two flags named the model
    size_label = model_size_label(specifier) or "off the ladder"
    tag = mixture_tag(real_data_fraction)
    stem = f"plots/test_correctness_bs{block_size}_{name}{tag}"

    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Correctness of the produced code"
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

    problems = load_dataset(BENCHMARK, split="test")
    if limit > 0:
        problems = problems.select(range(min(limit, len(problems))))
    print(f"##   benchmark: {BENCHMARK}, {len(problems)} problems, greedy pass@1")
    print(
        f"##   model:     {specifier} ({size_label})"
        f"{tag and '  (mixture ' + tag[1:] + ')'}"
    )

    def evaluate(checkpoint: str, label: str) -> Result:
        model, tokenizer = load_model(checkpoint, block_size, load_in_4bit)
        prompts = [build_prompt(tokenizer, problem) for problem in problems]
        result = score_model(
            model, tokenizer, problems, prompts, label, checkpoint,
            max_new_tokens, generation_batch_size, exec_timeout,
        )
        del model
        torch.cuda.empty_cache()
        return result

    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Base model (no collapse){TColors.ENDC}")
    anchor = evaluate(specifier, "base")
    print(
        f"##   pass@1 {anchor.pass_rate:.1%} ({anchor.n_passed}/{anchor.n_problems})  "
        f"{anchor.status_counts()}"
    )

    print(f"\n## {TColors.OKBLUE}{TColors.BOLD}Collapse checkpoints{TColors.ENDC}")
    results = []
    for generation in range(num_generations):
        checkpoint = resolve_checkpoint(
            generation, block_size, name, mixture_suffix(real_data_fraction, generation)
        )
        if not checkpoint:
            print(f"##   {TColors.WARNING}generation {generation}: no checkpoint{TColors.ENDC}")
            continue
        result = evaluate(checkpoint, f"generation {generation}")
        results.append(result)
        print(
            f"##   generation {generation}: pass@1 {result.pass_rate:6.1%} "
            f"({result.n_passed}/{result.n_problems})  {result.status_counts()}"
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
        "benchmark": BENCHMARK,
        "n_problems": len(problems),
        "decoding": f"greedy, max_new_tokens {max_new_tokens}",
        "base_model": {**anchor.__dict__, "pass_rate": anchor.pass_rate},
        "baseline": [{**row.__dict__, "pass_rate": row.pass_rate} for row in results],
    }
    os.makedirs(DATASET_PATH, exist_ok=True)
    with open(cache_file(block_size, name, tag), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    plot(payload, stem, usetex=not no_usetex)
    print(
        f"\n## {TColors.OKBLUE}{TColors.BOLD}Saved the figure under: "
        f"{TColors.HEADER}{stem}.<png,pdf>{TColors.ENDC}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Saved the verdicts under: "
        f"{TColors.HEADER}{cache_file(block_size, name, tag)}{TColors.ENDC}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Can every collapse checkpoint still produce correct code? HumanEval pass@1"
    )
    parser.add_argument("--num_generations", "-ng", type=int, default=10,
                        help="number of generations the run produced (default: 10)")
    parser.add_argument("--block_size", "-bs", type=int, default=512,
                        help="the run's block size, part of every artifact name (default: 512)")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="score only the first N benchmark problems, 0 for all (default: 0)")
    parser.add_argument("--max_new_tokens", "-mnt", type=int, default=384,
                        help="generation budget per problem (default: 384)")
    parser.add_argument("--generation_batch_size", "-gbs", type=int, default=16,
                        help="problems per generate() call (default: 16)")
    parser.add_argument("--exec_timeout", "-et", type=float, default=10.0,
                        help="wall-clock limit per executed candidate (default: 10.0)")
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0,
                        help="the mixture the collapse run used; part of the checkpoint names "
                        "from generation 1 on (default: 0.0)")
    parser.add_argument("--load_in_4bit", "-q4", action="store_true",
                        help="quantize the models under test. Off by default, so the measured "
                        "pass rate is the checkpoint's and not the quantization's")
    parser.add_argument("--plot_only", "-po", action="store_true",
                        help="replot from the cached verdicts without loading a model")
    parser.add_argument("--no_usetex", action="store_true",
                        help="render without LaTeX, for a machine with no TeX install")
    parser.add_argument("--path", "-p", type=str, default="",
                        help="root holding generated_datasets/ and model_outputs/ "
                        "(default: current directory)")
    add_model_arguments(parser, role="the model the run collapsed")
    args = parser.parse_args()
    main(**vars(args))
