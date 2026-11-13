"""What a collapse lineage costs, per generation and per model size.

The attack's premise is that the attacker never trains the collapsed model: the suffix is
optimized against a logit-extrapolated surrogate built from the base model and one fine-tune.
This script measures the alternative — actually running the lineage — from the artifacts a
finished `run_paper.sh` collapse phase leaves behind, and extrapolates it over the Qwen2.5-Coder
size ladder:

- **Per generation**, the wall-clock is read off directory timestamps, not log parsing: the
  merged checkpoint `model_{g}_..._fp16` is written when training g ends and the corpus
  `generated_dataset_{g}` when sampling g ends, so `train_g = t(model_g) - t(corpus_{g-1})` and
  `sample_g = t(corpus_g) - t(model_g)`. The corpus perplexity scoring runs once at the end
  (`t(log) - t(corpus_{n-1})`); it is this repo's measurement, not a cost of collapsing, so it
  is reported beside the per-generation cost and kept out of it. The collapse log provides the
  start time (its `## Date:` banner) and the GPU count.
- **"Collapsed"** is the first generation whose HumanEval pass rate falls to or below
  `--collapse_pass_rate` (default 5%, against 60% and 80% for the two bases) — the point where
  the model solves nothing, which is also where `run_attack.py`'s capability gate leaves nothing
  to attack. Lineages that never get there within the run say so.
- **Extrapolation** fits `hours per generation = a * params^b` through the measured sizes on the
  pure self-training lineage (`--fit_mixture`, default 0.0, the fastest collapse and the regime
  an attacker would run) and reads it at the larger rungs. Two measured sizes fix the fit
  exactly, so the extrapolated rows are the fit's reading and nothing more; they are drawn hollow
  and marked `fit` in the table. The fit is in wall-clock on the measured GPU count: an attacker
  with more cards divides it, one with fewer multiplies it, and from 7B upward a 48GB card
  needs 4-bit training, which the fit does not model.

Writes `plots/collapse_cost_<sizes>.{png,pdf}` and `plots/collapse_cost_<sizes>.md`.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

from utils import nord  # pylint: disable=wrong-import-position
from utils.colors import TColors  # pylint: disable=wrong-import-position
from utils.models import resolve_model_specifier  # pylint: disable=wrong-import-position
from utils.naming import mixture_suffix, mixture_tag  # pylint: disable=wrong-import-position
from utils.plotting import save_figure, tidy  # pylint: disable=wrong-import-position

# parameter counts of the Qwen2.5-Coder-Instruct ladder, from the model cards (total parameters)
PARAMS: dict = {"0.5b": 0.494e9, "1.5b": 1.54e9, "3b": 3.09e9, "7b": 7.61e9, "14b": 14.7e9,
                "32b": 32.5e9}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def lineage_label(mixture: float) -> str:
    return "control" if mixture >= 1.0 else f"rdf {mixture:g}"


def size_label(size: str) -> str:
    return size.upper().replace("B", "B")


def hours(seconds: float) -> float:
    return seconds / 3600.0


def mtime(path: str) -> float:
    return os.path.getmtime(path)


def read_log_header(path: str) -> tuple:
    """(start datetime, GPU count) from a collapse log's banner."""
    text = ANSI.sub("", open(path, errors="replace").read(60000))
    date = re.search(r"## Date: (.*)", text)
    start = datetime.strptime(date.group(1).strip(), "%A, %d. %B %Y %I:%M%p") if date else None
    gpus = re.search(r"## Sharding generation/perplexity across: (\d+) GPU", text)
    return start, int(gpus.group(1)) if gpus else None


def gpu_name() -> str:
    """The GPU model of this host, for the figure's hardware line; "" when unknown."""
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return ""
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return names[0].replace("NVIDIA ", "") if names else ""


class Lineage:
    """One (size, mixture) collapse run: its per-generation timings and its collapse point."""

    def __init__(self, args, size: str, mixture: float):
        self.size = size
        self.mixture = mixture
        self.name = resolve_model_specifier(size, "").split("/")[-1]
        root = args.path
        self.models = os.path.join(root, "model_outputs")
        self.datasets = os.path.join(root, "generated_datasets")
        log_dir = args.log_dir or os.path.join(root, "paper_logs")
        self.log = os.path.join(log_dir, f"collapse_{size}-rdf{mixture:.1f}.log")
        self.block_size = args.block_size
        self.generations = args.generations
        self.hg0 = args.human_generation_zero
        self.train = []      # seconds per generation
        self.sample = []
        self.scoring = 0.0   # the end-of-run corpus perplexity pass
        self.base_sample = 0.0
        self.start = None
        self.end = None
        self.gpus = None
        self.humaneval = []
        self.base_pass = None
        self.collapse_generation = None
        self.ok = self._measure(args.collapse_pass_rate)

    def _checkpoint(self, generation: int) -> str:
        return os.path.join(self.models, f"model_{generation}_bs{self.block_size}_{self.name}"
                            f"{mixture_suffix(self.mixture, generation, self.hg0)}_fp16")

    def _corpus(self, generation: int) -> str:
        if generation < 0:
            return os.path.join(self.datasets,
                                f"generated_dataset_base_bs{self.block_size}_{self.name}")
        return os.path.join(self.datasets, f"generated_dataset_{generation}_bs{self.block_size}"
                            f"_{self.name}{mixture_suffix(self.mixture, generation, self.hg0)}")

    def _measure(self, threshold: float) -> bool:
        if not os.path.isfile(self.log):
            print(f"{TColors.WARNING}no collapse log for {self.size} {lineage_label(self.mixture)}"
                  f": {self.log}{TColors.ENDC}")
            return False
        self.start, self.gpus = read_log_header(self.log)
        self.end = datetime.fromtimestamp(mtime(self.log))
        previous = self.start.timestamp() if self.start else None
        for g in range(self.generations):
            ckpt, corpus = self._checkpoint(g), self._corpus(g)
            if not (os.path.isdir(ckpt) and os.path.isdir(corpus)):
                print(f"{TColors.WARNING}{self.size} {lineage_label(self.mixture)}: generation "
                      f"{g} has no checkpoint or corpus on disk, stopping there{TColors.ENDC}")
                break
            t_model, t_corpus = mtime(ckpt), mtime(corpus)
            if previous is None or t_model < previous:
                # a rewritten earlier artifact (the base corpus is re-saved by later mixtures);
                # fall back to the log start for the first boundary
                previous = self.start.timestamp() if self.start else t_model
            self.train.append(max(t_model - previous, 0.0))
            self.sample.append(max(t_corpus - t_model, 0.0))
            previous = t_corpus
        if not self.train:
            return False
        self.scoring = max(self.end.timestamp() - previous, 0.0)
        # generation 0's "training" span above includes the base-corpus sampling that precedes it
        # on the pure self-training and mixed lineages; it is reported inside generation 0
        payload_path = os.path.join(
            self.datasets, f"test_correctness_bs{self.block_size}_{self.name}"
                           f"{mixture_tag(self.mixture)}.json")
        if os.path.isfile(payload_path):
            payload = json.load(open(payload_path))
            self.base_pass = payload["base_model"]["pass_rate"]
            self.humaneval = [r["pass_rate"] for r in payload["baseline"]]
            for g, rate in enumerate(self.humaneval):
                if rate <= threshold:
                    self.collapse_generation = g
                    break
        return True

    # ── derived quantities ──
    @property
    def n_measured(self) -> int:
        return len(self.train)

    def per_generation(self, g: int) -> float:
        """Wall-clock seconds an attacker pays for generation g: the fine-tune and the sampling
        of the next corpus. The corpus perplexity pass is this repo's instrumentation and is
        reported beside it, not inside it."""
        return self.train[g] + self.sample[g]

    @property
    def mean_per_generation(self) -> float:
        return sum(self.per_generation(g) for g in range(self.n_measured)) / self.n_measured

    def cumulative(self) -> list:
        out, total = [], 0.0
        for g in range(self.n_measured):
            total += self.per_generation(g)
            out.append(total)
        return out

    @property
    def total(self) -> float:
        """The attacker's wall-clock for the measured generations."""
        return self.cumulative()[-1]

    @property
    def run_total(self) -> float:
        """The whole run as it happened, scoring included."""
        return (self.end - self.start).total_seconds() if self.start else self.total

    @property
    def time_to_collapse(self):
        if self.collapse_generation is None or self.collapse_generation >= self.n_measured:
            return None
        return self.cumulative()[self.collapse_generation]


def fit_power_law(points: list) -> tuple:
    """(a, b) with hours = a * params**b through (params, hours) points, least squares in logs."""
    x = np.log([p for p, _ in points])
    y = np.log([h for _, h in points])
    if len(points) == 1:
        return float(np.exp(y[0] - x[0])), 1.0
    b, log_a = np.polyfit(x, y, 1)
    return float(np.exp(log_a)), float(b)


def fmt_hours(value) -> str:
    if value is None:
        return "—"
    if value < 1.0:
        return f"{value * 60:.0f} min"
    if value < 48.0:
        return f"{value:.1f} h"
    return f"{value / 24:.1f} d"


def draw(lineages: list, fit: tuple, fit_mixture: float, extrapolated: list, gpus: int,
         hardware: str, usetex: bool, stem: str, generations: int) -> None:
    nord.apply_style(usetex, font_size=13)
    figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 5.8))
    sizes = sorted({l.size for l in lineages}, key=lambda s: PARAMS[s])
    mixtures = sorted({l.mixture for l in lineages})
    colours = nord.mixture_colors(mixtures)
    styles = dict(zip(sizes, ["-", "--", ":", "-."]))
    markers = dict(zip(sizes, nord.MARKERS))

    # left: cumulative wall-clock over the generations, the collapse point marked
    for lineage in lineages:
        xs = list(range(lineage.n_measured))
        ys = [hours(v) for v in lineage.cumulative()]
        left.plot(xs, ys, styles[lineage.size], marker=markers[lineage.size], markersize=5,
                  color=colours[lineage.mixture], linewidth=1.8,
                  label=f"{size_label(lineage.size)} {lineage_label(lineage.mixture)}")
        g = lineage.collapse_generation
        if g is not None and g < lineage.n_measured:
            left.plot([g], [ys[g]], marker="*", markersize=17, color=nord.HIGHLIGHT,
                      markeredgecolor=nord.TEXT, markeredgewidth=0.6, zorder=5)
    left.plot([], [], marker="*", markersize=13, color=nord.HIGHLIGHT, linestyle="none",
              label=r"HumanEval $\leq$ threshold" if usetex else "HumanEval ≤ threshold")
    left.set_xlabel("collapse generation")
    left.set_ylabel(f"cumulative wall-clock (h) on {gpus} GPUs, train + sample")
    left.set_xticks(range(generations))
    left.set_title("measured: how long each lineage took")
    left.legend(fontsize=9, ncol=2)
    left.grid(True)
    tidy(left)

    # right: hours per generation against parameters, measured and extrapolated
    for lineage in lineages:
        right.plot([PARAMS[lineage.size] / 1e9], [hours(lineage.mean_per_generation)],
                   marker=markers[lineage.size], markersize=8, linestyle="none",
                   color=colours[lineage.mixture])
    a, b = fit
    grid_x = np.geomspace(min(PARAMS[s] for s in sizes) * 0.8,
                          max(PARAMS[s] for s in list(sizes) + [e for e, _ in extrapolated]) * 1.3,
                          100)
    right.plot(grid_x / 1e9, a * grid_x ** b, color=nord.ANCHOR, linewidth=1.4, linestyle="--",
               label=(rf"fit through {lineage_label(fit_mixture)}: "
                      rf"$h \propto N^{{{b:.2f}}}$" if usetex
                      else f"fit through {lineage_label(fit_mixture)}: h ∝ N^{b:.2f}"))
    for size, per_gen in extrapolated:
        right.plot([PARAMS[size] / 1e9], [hours(per_gen)], marker="o", markersize=9,
                   markerfacecolor="none", markeredgecolor=nord.SURROGATE, markeredgewidth=1.8,
                   linestyle="none")
        right.annotate(f"{size_label(size)}: {fmt_hours(hours(per_gen))}/gen",
                       (PARAMS[size] / 1e9, hours(per_gen)), textcoords="offset points",
                       xytext=(8, -12), fontsize=9, color=nord.SURROGATE)
    for size in sizes:
        rows = [l for l in lineages if l.size == size and l.mixture == fit_mixture]
        if rows:
            right.annotate(f"{size_label(size)}: {fmt_hours(hours(rows[0].mean_per_generation))}/gen",
                           (PARAMS[size] / 1e9, hours(rows[0].mean_per_generation)),
                           textcoords="offset points", xytext=(8, 6), fontsize=9)
    right.plot([], [], marker="o", markersize=8, markerfacecolor="none",
               markeredgecolor=nord.SURROGATE, linestyle="none", label="extrapolated (fit only)")
    for mixture in mixtures:
        right.plot([], [], marker="s", linestyle="none", color=colours[mixture],
                   label=lineage_label(mixture))
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("parameters (billions)")
    right.set_ylabel(f"wall-clock per generation (h) on {gpus} GPUs")
    right.set_title("per generation, against model size")
    right.legend(fontsize=9)
    right.grid(True, which="both")
    tidy(right)

    title = "The cost of actually collapsing the model"
    if hardware:
        title += f" ({gpus} × {hardware}, {generations} generations, 50k-instruction corpus)"
    figure.suptitle(title, fontsize=15)
    save_figure(figure, stem)


def markdown(lineages: list, fit: tuple, fit_mixture: float, extrapolated: list, gpus: int,
             hardware: str, threshold: float, generations: int, path: str) -> None:
    a, b = fit
    lines = [f"# Collapse cost: wall-clock on {gpus} GPUs" + (f" ({hardware})" if hardware else ""),
             "",
             f"Per generation: LoRA fine-tune on the {generations}-generation lineage's corpus "
             "(5 epochs, ~50k instructions), then sampling the next corpus. The corpus perplexity "
             "pass is this repo's measurement and is listed beside the cost, not inside it. "
             "\"Collapsed\" = first generation whose "
             f"HumanEval pass rate is at or below {100 * threshold:.0f}%. GPU-hours multiply the "
             "wall-clock by the GPU count.", "",
             "## Measured lineages", "",
             "| size | lineage | train / gen | sample / gen | cost / gen | GPU-h / gen | "
             "collapsed at | cost to collapse | cost for "
             f"{generations} generations | whole run incl. scoring | HumanEval base → last |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in lineages:
        n = l.n_measured
        collapse = (f"g{l.collapse_generation}" if l.collapse_generation is not None
                    else f"never (≤ g{n - 1})")
        he = (f"{l.base_pass:.2f} → {l.humaneval[-1]:.2f}" if l.humaneval else "—")
        lines.append(
            f"| {size_label(l.size)} | {lineage_label(l.mixture)} | "
            f"{fmt_hours(hours(sum(l.train) / n))} | {fmt_hours(hours(sum(l.sample) / n))} | "
            f"{fmt_hours(hours(l.mean_per_generation))} | "
            f"{gpus * hours(l.mean_per_generation):.1f} | {collapse} | "
            f"{fmt_hours(hours(l.time_to_collapse) if l.time_to_collapse else None)} | "
            f"{fmt_hours(hours(l.total))} | {fmt_hours(hours(l.run_total))} | {he} |")
    lines += ["", f"## Extrapolated over the size ladder (fit through {lineage_label(fit_mixture)}: "
                  f"hours per generation = {a:.3g} · params^{b:.2f})", "",
              "The fit has as many points as measured sizes, so these rows are the fit's reading "
              "and carry no uncertainty estimate. They assume the same GPU count and that the "
              "model still trains in 16-bit LoRA on one card, which stops being true around 7B on "
              "48GB.", "",
              f"| size | params | wall-clock / gen | GPU-h / gen | wall-clock for {generations} "
              "generations | GPU-h for the lineage | source |",
              "|---|---|---|---|---|---|---|"]
    for size in sorted(PARAMS, key=PARAMS.get):
        measured = [l for l in lineages if l.size == size and l.mixture == fit_mixture]
        if measured:
            per_gen, source = measured[0].mean_per_generation, "measured"
        else:
            match = [p for s, p in extrapolated if s == size]
            if not match:
                continue
            per_gen, source = match[0], "fit"
        lines.append(f"| {size_label(size)} | {PARAMS[size] / 1e9:.2f}B | "
                     f"{fmt_hours(hours(per_gen))} | {gpus * hours(per_gen):.1f} | "
                     f"{fmt_hours(hours(per_gen) * generations)} | "
                     f"{gpus * hours(per_gen) * generations:.0f} | {source} |")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--path", "-p", default="./runs/fresh", help="the run root")
    parser.add_argument("--plots_path", "-pp", default="",
                        help="where the figure and table go (default: <root>/plots)")
    parser.add_argument("--log_dir", default="", help="collapse logs (default: <root>/paper_logs)")
    parser.add_argument("--sizes", nargs="+", default=["0.5b", "3b"])
    parser.add_argument("--mixtures", nargs="+", type=float, default=[0.0, 0.5, 0.8, 1.0])
    parser.add_argument("--generations", "-n", type=int, default=10)
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--human_generation_zero", "-hg0", action="store_true")
    parser.add_argument("--collapse_pass_rate", type=float, default=0.05,
                        help="HumanEval pass rate at or below which a generation counts as "
                             "collapsed (default 0.05)")
    parser.add_argument("--fit_mixture", type=float, default=0.0,
                        help="the lineage the size extrapolation is fitted through (default 0.0)")
    parser.add_argument("--extrapolate", nargs="*", default=["7b", "14b", "32b"],
                        help="sizes to read off the fit (default: 7b 14b 32b)")
    parser.add_argument("--hardware", default=None,
                        help="GPU model for the title (default: nvidia-smi on this host)")
    parser.add_argument("--no_usetex", action="store_true")
    args = parser.parse_args()
    usetex = not args.no_usetex
    plots = args.plots_path or os.path.join(args.path, "plots")

    lineages = [Lineage(args, size, mixture) for size in args.sizes for mixture in args.mixtures]
    lineages = [l for l in lineages if l.ok]
    if not lineages:
        raise SystemExit("no measurable lineage found")
    gpus = next((l.gpus for l in lineages if l.gpus), None) or 1
    hardware = args.hardware if args.hardware is not None else gpu_name()

    points = [(PARAMS[l.size], hours(l.mean_per_generation)) for l in lineages
              if l.mixture == args.fit_mixture]
    if not points:
        raise SystemExit(f"--fit_mixture {args.fit_mixture} has no measured lineage")
    fit = fit_power_law(points)
    a, b = fit
    extrapolated = [(s, 3600 * a * PARAMS[s] ** b) for s in args.extrapolate
                    if s not in {l.size for l in lineages}]

    print(f"## {TColors.OKBLUE}{TColors.BOLD}Collapse cost{TColors.ENDC} on {gpus} GPU(s)"
          + (f" ({hardware})" if hardware else ""))
    for l in lineages:
        n = l.n_measured
        print(f"##   {size_label(l.size):5s} {lineage_label(l.mixture):8s}: "
              f"train {fmt_hours(hours(sum(l.train) / n)):>8s}/gen, "
              f"sample {fmt_hours(hours(sum(l.sample) / n)):>7s}/gen, "
              f"cost {fmt_hours(hours(l.total)):>7s} (+{fmt_hours(hours(l.scoring))} scoring); "
              f"collapsed at "
              f"{'g' + str(l.collapse_generation) if l.collapse_generation is not None else 'never'}"
              f" ({fmt_hours(hours(l.time_to_collapse) if l.time_to_collapse else None)})")
    print(f"##   fit through {lineage_label(args.fit_mixture)}: hours/gen = {a:.3g} * N^{b:.2f}")
    for size, per_gen in extrapolated:
        print(f"##   {size_label(size):5s} extrapolated: {fmt_hours(hours(per_gen))}/gen, "
              f"{fmt_hours(hours(per_gen) * args.generations)} for {args.generations} generations")

    stem = os.path.join(plots, "collapse_cost_" + "_".join(args.sizes))
    draw(lineages, fit, args.fit_mixture, extrapolated, gpus, hardware, usetex, stem,
         args.generations)
    markdown(lineages, fit, args.fit_mixture, extrapolated, gpus, hardware,
             args.collapse_pass_rate, args.generations, stem + ".md")
    print(f"## saved {stem}.<png,pdf,md>")


if __name__ == "__main__":
    main()
