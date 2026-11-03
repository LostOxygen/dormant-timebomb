"""The paper grid's result tables, as figures and as markdown.

Reads what run_paper.sh produced under one root — the utility caches, the collapse perplexity
caches, the surrogate calibration, the attack result files and the sink screens — and draws one
figure per table of the write-up, in the shared Nord palette (utils/nord.py), plus the same
tables as a markdown file next to them. Nothing here touches a model: every number comes from a
file another step wrote, so a redraw is free and ``run_paper.sh --phases plots`` includes it.

Figures, all under ``--plots_path`` and prefixed ``tables_``:

    utility           HumanEval pass rate, model perplexity on the held-out human rows, corpus
                      perplexity under the base model — one line per lineage over the generations
    distribution      the token distribution (run_distribution_collapse.py): each generation's
                      predictive entropy on held-out rows and on its own samples, the conditional
                      trigram entropy and the compiling share of the corpus it wrote, per lineage
    attack_<suite>_<verdict>
                      the hit-rate grid: one row per (lineage, method), one column per generation,
                      each cell the share of that generation's attackable targets the search broke,
                      shaded by that share, aborted generations marked
    rates_<suite>_<verdict>
                      the lineages against each other: the same hit rates as lines over the
                      generations, one panel per method, and a last panel of the rate pooled over
                      the generations (sum of hits over sum of attackable targets) per lineage
    calibration       (``--calibration`` only) the extrapolation factor that reproduces each
                      generation's perplexity, the fitted law, and the rules the searches used
    surrogate_quality the logit surrogate as a predictor: real hits, predicted, false alarms,
                      missed, per lineage and variant, summed over the generations
    sink_screen       the affine-law roots per lineage, against the cap the searches applied

The result files are located through utils.naming, the same helpers run_attack.py names them
with, so this script cannot drift from the files on disk; a cell whose file is missing is left
blank rather than guessed.

Usage:
    python run_paper_tables.py -p ./runs/fresh -msz 0.5b --mixtures 0.0 0.5 0.8 1.0 -n 9
"""

import argparse
import json
import os
import statistics

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from utils import nord
from utils.colors import TColors
from utils.models import resolve_model_specifier
from utils.naming import (
    anchor_tag,
    entropy_tag,
    factor_mode_tag,
    gen_zero_tag,
    mixture_tag,
    objective_tag,
    verification_tag,
)
from utils.plotting import save_figure

matplotlib.use("Agg")

# the marker run_attack_vuln.py puts after the mixture tag of its result files
VULN_MARKER: str = "_vuln"
SUITES: dict = {"vuln": "vulnerability targets", "correctness": "correctness targets"}
METHOD_LABELS: dict = {
    ("none", "plain"): "direct",
    ("logit", "plain"): "logit plain",
    ("logit", "opt"): "logit opt",
}


# ────────────────────────────────── loading ───────────────────────────────────────────────
def read_json(path: str):
    """The parsed file, or None when it is not there (a cell left blank, never guessed)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def lineage_label(mixture: float) -> str:
    return "control" if mixture >= 1.0 else f"rdf {mixture:g}"


class Grid:
    """Everything one size's figures read, keyed by mixture and generation."""

    def __init__(self, args, size: str):
        self.args = args
        self.size = size
        self.name = resolve_model_specifier(size, "").split("/")[-1]
        self.mixtures = [float(m) for m in args.mixtures]
        self.generations = list(range(args.generations + 1))
        self.root = args.path
        self.datasets = os.path.join(self.root, "generated_datasets")
        self.results = os.path.join(self.root, "attack_results")
        self.gen0 = gen_zero_tag(args.human_generation_zero)
        self.init = "_recursive" if args.recursive_init else "_freshinit"

    # ── utility ──
    def humaneval(self, mixture: float) -> tuple:
        """(base pass rate, [per generation]) or (None, [])."""
        payload = read_json(os.path.join(
            self.datasets,
            f"test_correctness_bs{self.args.block_size}_{self.name}{mixture_tag(mixture)}.json",
        ))
        if not payload:
            return None, []
        return payload["base_model"]["pass_rate"], [r["pass_rate"] for r in payload["baseline"]]

    def model_perplexity(self, mixture: float) -> tuple:
        """(base median, [per generation medians], leaked) on the validation slice."""
        payload = read_json(os.path.join(
            self.datasets,
            f"test_perplexity_bs{self.args.block_size}_{self.name}{mixture_tag(mixture)}.json",
        ))
        if not payload:
            return None, [], False
        test = payload["splits"]["test"]
        return (
            test["base_model"]["median"],
            [row["median"] for row in test["baseline"]],
            bool(payload.get("test_set_may_leak")),
        )

    def corpus_perplexity(self, mixture: float) -> list:
        """Median corpus perplexity under the base model per generation index, or []."""
        path = os.path.join(
            self.datasets,
            f"perplexity_dict_bs{self.args.block_size}_{self.name}{self.init}{self.gen0}"
            f"{mixture_tag(mixture)}.pt",
        )
        if not os.path.exists(path):
            return []
        import torch  # pylint: disable=import-outside-toplevel

        cache = torch.load(path, map_location="cpu", weights_only=False)
        medians = []
        for key in sorted(cache, key=lambda k: int(str(k).split()[-1])):
            values = cache[key]
            values = values.tolist() if hasattr(values, "tolist") else list(values)
            medians.append(statistics.median(float(v) for v in values if v == v))
        return medians

    def distribution(self, mixture: float) -> dict | None:
        """run_distribution_collapse.py's result for this lineage, or None."""
        return read_json(os.path.join(
            self.results, f"distribution_collapse_{self.name}{mixture_tag(mixture)}.json"
        ))

    def calibration(self, mixture: float) -> dict | None:
        return read_json(os.path.join(
            self.datasets,
            f"surrogate_factor_bs{self.args.block_size}_{self.name}{mixture_tag(mixture)}.json",
        ))

    # ── attacks ──
    def result_path(self, mixture, suite, method, variant, verdict, generation) -> str:
        """The file run_attack.py writes for this cell, spelled with its own naming helpers."""
        args = self.args
        transfer = method != "none"
        opt = variant == "opt" and transfer
        sink_objective = opt and suite == "vuln" and args.opt_sink_objective
        sink_factor = opt and suite == "vuln" and args.opt_sink_factor
        entropy = opt and args.opt_entropy_match
        decoding = verification_tag(0.0) if verdict == "greedy" else verification_tag()
        stem = (
            f"attack_gen{generation}_{self.name}{mixture_tag(mixture)}"
            + (VULN_MARKER if suite == "vuln" else "")
            + decoding
            + anchor_tag(args.hold_anchor and generation > 0)
            + objective_tag(sink_objective)
            + (
                f"{factor_mode_tag('sink') if sink_factor else ''}{entropy_tag(entropy)}"
                f"_{method}_surrogate" if transfer else ""
            )
        )
        return os.path.join(self.results, f"{stem}.json")

    def attack_cell(self, mixture, suite, method, variant, verdict, generation) -> dict | None:
        """{usable, hits, aborted, surrogate_quality} of one cell, or None when it was never run."""
        payload = read_json(
            self.result_path(mixture, suite, method, variant, verdict, generation)
        )
        if payload is None:
            return None
        probe = payload.get("capability_probe") or {}
        return {
            "usable": len(probe.get("usable") or []),
            "hits": sum(1 for row in payload.get("results") or [] if row.get("successes")),
            "aborted": bool(payload.get("aborted")),
            "surrogate_quality": payload.get("surrogate_quality") or {},
        }

    def sink_screen(self, mixture: float) -> dict | None:
        payload = read_json(os.path.join(
            self.results, f"sink_factor_{self.name}{mixture_tag(mixture)}.json"
        ))
        return (payload or {}).get("per_task")


# ────────────────────────────────── drawing ───────────────────────────────────────────────
def esc(text: str, usetex: bool) -> str:
    """usetex renders every label, so underscores have to be escaped there."""
    return text.replace("_", r"\_") if usetex else text


def percent(rate: float, usetex: bool = False) -> str:
    """A hit rate as the integer percentage every table and cell prints. Under usetex the sign
    is a comment character and has to be escaped, or the label ends at the number."""
    return f"{100 * rate:.0f}" + (r"\%" if usetex else "%")


def lineage_style(grid: Grid) -> dict:
    """mixture -> (colour, marker): the ordinal frost ramp with a marker beside it."""
    colours = nord.mixture_colors(grid.mixtures)
    return {
        m: (colours[m], nord.MARKERS[i % len(nord.MARKERS)]) for i, m in enumerate(grid.mixtures)
    }


class EndLabels:
    """Direct labels at the last point of each line, placed after every line is drawn so that
    labels of lines that end close together are pushed apart instead of overprinting."""

    MIN_GAP_PX = 15

    def __init__(self, axis):
        self.axis = axis
        self.entries = []

    def add(self, x, y, text, colour) -> None:
        self.entries.append((x, y, text, colour))

    def place(self) -> None:
        if not self.entries:
            return
        figure = self.axis.figure
        figure.canvas.draw()  # the data -> pixel transform needs the final axis limits
        to_px = self.axis.transData.transform
        rows = sorted(
            ((to_px((x, y))[1], x, y, text, colour) for x, y, text, colour in self.entries),
            key=lambda row: row[0],
        )
        # sweep upwards, keeping every label at least MIN_GAP_PX above the one below it
        placed = []
        for px_y, x, y, text, colour in rows:
            if placed and px_y - placed[-1] < self.MIN_GAP_PX:
                px_y = placed[-1] + self.MIN_GAP_PX
            placed.append(px_y)
        # then centre the stack on its original span, so a cluster spreads both ways
        shift = (sum(r[0] for r in rows) - sum(placed)) / len(rows)
        for (px_y, x, y, text, colour), target in zip(rows, placed):
            offset = target + shift - px_y
            self.axis.annotate(
                text, (x, y), xytext=(6, offset / figure.dpi * 72), textcoords="offset points",
                va="center", ha="left", fontsize=11, color=nord.TEXT_MUTED,
                bbox={"boxstyle": "round,pad=0.15", "fc": nord.SURFACE, "ec": colour,
                      "lw": 0.8},
            )


def tidy(axis) -> None:
    """The recessive chrome every panel shares: no top or right spine, grid under the marks."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_axisbelow(True)


def figure_utility(grid: Grid, usetex: bool, stem: str, tables: list) -> None:
    styles = lineage_style(grid)
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    panels = (
        ("HumanEval pass rate", "pass rate", False),
        ("model perplexity, held-out human code", "perplexity (median)", True),
        ("corpus perplexity under the base model", "perplexity (median)", True),
    )
    rows_h, rows_m, rows_c = [], [], []
    base_lines = {}
    labels = [EndLabels(axis) for axis in axes]
    for mixture in grid.mixtures:
        colour, marker = styles[mixture]
        label = lineage_label(mixture)
        base_pass, passes = grid.humaneval(mixture)
        base_ppl, ppl, leaked = grid.model_perplexity(mixture)
        corpus = grid.corpus_perplexity(mixture)
        gens = grid.generations
        if passes:
            axes[0].plot(gens[:len(passes)], passes, marker=marker, color=colour, label=label)
            labels[0].add(gens[len(passes) - 1], passes[-1], label, colour)
            base_lines.setdefault(0, base_pass)
            rows_h.append([label, f"{base_pass:.3f}"] + [f"{v:.3f}" for v in passes])
        if ppl:
            axes[1].plot(
                gens[:len(ppl)], ppl, marker=marker, color=colour,
                label=label + (" (rows seen in training)" if leaked else ""),
                markerfacecolor=nord.SURFACE if leaked else colour, markeredgewidth=1.8,
            )
            labels[1].add(gens[len(ppl) - 1], ppl[-1], label, colour)
            base_lines.setdefault(1, base_ppl)
            rows_m.append([label + (" †" if leaked else ""), f"{base_ppl:.2f}"]
                          + [f"{v:.2f}" for v in ppl])
        if corpus:
            xs = list(range(len(corpus)))
            axes[2].plot(xs, corpus, marker=marker, color=colour, label=label)
            labels[2].add(xs[-1], corpus[-1], label, colour)
            rows_c.append([label] + [f"{v:.2f}" for v in corpus])
    for index, (title, ylabel, log) in enumerate(panels):
        axis = axes[index]
        if index in base_lines and base_lines[index] is not None:
            axis.axhline(base_lines[index], color=nord.ANCHOR, linestyle=":", linewidth=1.5,
                         label="base model")
        if log:
            axis.set_yscale("log")
        axis.set_title(title, fontsize=15)
        axis.set_xlabel("collapse generation" if index < 2 else "corpus index (0 = base samples)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(grid.generations)
        axis.margins(x=0.12)
        axis.legend(fontsize=10, loc="best")
        tidy(axis)
    figure.suptitle(f"{esc(grid.name, usetex)} lineages: utility over the collapse", fontsize=16)
    figure.tight_layout()
    for group in labels:
        group.place()
    save_figure(figure, stem)
    plt.close(figure)
    header = ["lineage", "base"] + [f"g{g}" for g in grid.generations]
    tables.append(("HumanEval pass rate (greedy, 164 problems)", header, rows_h))
    tables.append(("Model perplexity on the held-out human slice (median; † rows the lineage "
                   "trained on)", header, rows_m))
    tables.append(("Corpus perplexity under the base model (median; index 0 = the base model's "
                   "own samples)", ["lineage"] + [str(g) for g in grid.generations], rows_c))


def figure_distribution(grid: Grid, usetex: bool, stem: str, tables: list) -> bool:
    """The token distribution over the collapse, lineages against each other: the models'
    predictive entropy on held-out human rows and on their own samples, and the conditional
    trigram entropy and the compiling share of the corpora they wrote. False when no lineage
    was measured."""
    styles = lineage_style(grid)
    panels = (
        ("models", "heldout", "entropy_mean", "predictive entropy, held-out human rows",
         "mean entropy per position (nats)", False),
        ("models", "own", "entropy_mean", "predictive entropy, the generation's own samples",
         "mean entropy per position (nats)", False),
        ("corpora", None, "entropy_trigram_conditional",
         "corpus trigram entropy, conditional", "entropy (nats)", False),
        ("corpora", None, "compile_rate", "corpus responses with compiling code", "share", True),
    )
    figure, axes = plt.subplots(1, 4, figsize=(24, 5.6))
    labels = [EndLabels(axis) for axis in axes]
    references = {}
    rows_by_panel = [[] for _ in panels]
    drew = False
    for mixture in grid.mixtures:
        payload = grid.distribution(mixture)
        if not payload:
            continue
        drew = True
        colour, marker = styles[mixture]
        label = lineage_label(mixture)
        for index, (side, view, key, _, _, as_percent) in enumerate(panels):
            scale = 100 if as_percent else 1
            if side == "models":
                table = payload["models"]
                xs = sorted(int(g) for g in table if g != "base")
                ys = [table[str(g)][view][key] * scale for g in xs]
                references.setdefault(index, table["base"][view][key] * scale)
                base_text = f"{table['base'][view][key]:.3f}"
            else:
                table = payload["corpora"]
                xs = sorted(int(g) for g in table if g != "human")
                ys = [table[str(g)][key] * scale for g in xs]
                references.setdefault(index, table["human"][key] * scale)
                base_text = f"{table['human'][key]:.3f}"
            axes[index].plot(xs, ys, marker=marker, color=colour, label=label)
            labels[index].add(xs[-1], ys[-1], label, colour)
            rows_by_panel[index].append([label, base_text] + [f"{y / scale:.3f}" for y in ys])
    if not drew:
        plt.close(figure)
        return False
    for index, (side, _, _, title, ylabel, as_percent) in enumerate(panels):
        axis = axes[index]
        if index in references:
            axis.axhline(references[index], color=nord.ANCHOR, linestyle=":", linewidth=1.5,
                         label="base model" if side == "models" else "human corpus")
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("collapse generation" if side == "models"
                        else "corpus index (-1: the base model's samples)")
        axis.set_ylabel(ylabel)
        if as_percent:
            axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        axis.margins(x=0.12)
        axis.legend(fontsize=10, loc="best")
        tidy(axis)
    figure.suptitle(f"{esc(grid.name, usetex)} lineages: the token distribution over the "
                    "collapse", fontsize=16)
    figure.tight_layout()
    for group in labels:
        group.place()
    save_figure(figure, stem)
    plt.close(figure)
    model_header = ["lineage", "base"] + [f"g{g}" for g in grid.generations]
    corpus_header = ["lineage", "human"] + [str(g) for g in range(-1, grid.generations[-1] + 1)]
    tables.append(("Predictive entropy on the held-out human rows (nats per position)",
                   model_header, rows_by_panel[0]))
    tables.append(("Predictive entropy on the generation's own samples (nats per position)",
                   model_header, rows_by_panel[1]))
    tables.append(("Conditional trigram entropy of the corpus each generation wrote (nats; "
                   "index -1 = the base model's samples)", corpus_header, rows_by_panel[2]))
    tables.append(("Share of corpus responses with compiling code", corpus_header,
                   rows_by_panel[3]))
    return True


def figure_calibration(grid: Grid, usetex: bool, stem: str, tables: list) -> None:
    styles = lineage_style(grid)
    figure, axis = plt.subplots(figsize=(9.5, 5.6))
    rows = []
    drew = False
    labels = EndLabels(axis)
    for mixture in grid.mixtures:
        payload = grid.calibration(mixture)
        if not payload:
            continue
        colour, marker = styles[mixture]
        label = lineage_label(mixture)
        gens = sorted(payload["factors"], key=int)
        fitted = [payload["factors"][g] for g in gens]
        law = [payload["law_factors"][g] for g in gens]
        xs = [int(g) for g in gens]
        axis.plot(xs, fitted, marker=marker, color=colour, label=f"{label}: fitted")
        axis.plot(xs, law, linestyle="--", color=colour, linewidth=1.5, alpha=0.9)
        labels.add(xs[-1], fitted[-1], label, colour)
        rows.append([label, f"{payload['scale']:.3f}"] + [f"{v:.2f}" for v in fitted])
        drew = True
    if drew:
        xs = grid.generations
        axis.plot(xs, [g + 1 for g in xs], linestyle=":", color=nord.ANCHOR, linewidth=1.5,
                  label="$n = g + 1$ (plain logit sweeps)" if usetex else "n = g + 1 (plain logit sweeps)")
        axis.axhline(grid.args.sink_cap, color=nord.HIGHLIGHT, linestyle="-.", linewidth=1.5,
                     label=f"sink cap {grid.args.sink_cap:g} (opt sweeps)")
        axis.plot([], [], linestyle="--", color=nord.TEXT_MUTED, linewidth=1.5,
                  label="dashed: the fitted law $n = 1 + s\\,\\ln(1+g)$" if usetex
                  else "dashed: the fitted law n = 1 + s ln(1+g)")
    axis.set_xlabel("collapse generation")
    axis.set_ylabel("extrapolation factor $n$" if usetex else "extrapolation factor n")
    axis.set_title("surrogate calibration: the factor that reproduces each generation", fontsize=15)
    axis.set_xticks(grid.generations)
    axis.margins(x=0.12)
    axis.legend(fontsize=10, loc="upper left")
    tidy(axis)
    figure.tight_layout()
    labels.place()
    save_figure(figure, stem)
    plt.close(figure)
    tables.append(("Surrogate calibration: fitted factor per generation and the law's scale",
                   ["lineage", "scale"] + [f"g{g}" for g in grid.generations], rows))


def figure_attack_grid(grid: Grid, suite: str, verdict: str, usetex: bool, stem: str,
                       tables: list) -> bool:
    """The hits / attackable grid of one suite under one verdict. False when nothing was found."""
    combos = [(m, v) for m in grid.args.methods for v in grid.args.variants
              if (m, v) in METHOD_LABELS]
    rows, labels, groups = [], [], []
    for mixture in grid.mixtures:
        for method, variant in combos:
            cells = [grid.attack_cell(mixture, suite, method, variant, verdict, g)
                     for g in grid.generations]
            rows.append(cells)
            labels.append((lineage_label(mixture), METHOD_LABELS[(method, variant)]))
            groups.append(mixture)
    if not any(cell for row in rows for cell in row):
        return False

    n_rows, n_cols = len(rows), len(grid.generations)
    gap = 0.35  # the breathing room between lineages
    ys, y = [], 0.0
    for index in range(n_rows):
        if index and groups[index] != groups[index - 1]:
            y += gap
        ys.append(y)
        y += 1.0
    figure, axis = plt.subplots(figsize=(1.05 * n_cols + 4.2, 0.52 * n_rows + 2.4))
    cmap = nord.sequential_cmap()
    table_rows = []
    for index, cells in enumerate(rows):
        text_row = [labels[index][0], labels[index][1]]
        for column, cell in enumerate(cells):
            x, yy = column, ys[index]
            if cell is None:
                text_row.append("")
                continue
            if cell["aborted"] or cell["usable"] == 0:
                axis.add_patch(plt.Rectangle((x - 0.5, yy - 0.5), 1, 1, facecolor=nord.NEUTRAL,
                                             edgecolor=nord.SURFACE, linewidth=2))
                axis.text(x, yy, "–", ha="center", va="center", fontsize=11,
                          color=nord.TEXT_MUTED)
                text_row.append("–")
                continue
            rate = cell["hits"] / cell["usable"]
            axis.add_patch(plt.Rectangle((x - 0.5, yy - 0.5), 1, 1, facecolor=cmap(rate),
                                         edgecolor=nord.SURFACE, linewidth=2))
            axis.text(x, yy, percent(rate, usetex), ha="center", va="center", fontsize=10.5,
                      color=nord.text_on(rate))
            text_row.append(f"{percent(rate)} (n={cell['usable']})")
        table_rows.append(text_row)
    axis.set_xlim(-0.5, n_cols - 0.5)
    axis.set_ylim(ys[-1] + 0.5, -0.5)
    axis.set_xticks(range(n_cols))
    axis.set_xticklabels(grid.generations)
    axis.set_yticks(ys)
    axis.set_yticklabels([f"{lin}  ·  {meth}" for lin, meth in labels], fontsize=11)
    axis.set_xlabel("collapse generation")
    axis.tick_params(axis="y", length=0)
    axis.grid(False)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.set_title(
        f"{SUITES[suite]}, {verdict} verdict: hit rate per generation "
        "(share of the attackable targets broken)",
        fontsize=14, loc="left",
    )
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    bar = figure.colorbar(mappable, ax=axis, fraction=0.03, pad=0.02,
                          format=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    bar.set_label("hit rate", fontsize=11)
    bar.ax.tick_params(labelsize=10)
    bar.outline.set_visible(False)
    axis.text(n_cols - 0.5, ys[-1] + 1.1, "– : capability gate aborted (no attackable target)",
              ha="right", va="top", fontsize=9.5, color=nord.TEXT_MUTED)
    save_figure(figure, stem)
    plt.close(figure)
    tables.append((f"{SUITES[suite]}, {verdict} verdict: hit rate per generation "
                   "(n = attackable targets)",
                   ["lineage", "method"] + [f"g{g}" for g in grid.generations], table_rows))
    return True


def figure_rate_comparison(grid: Grid, suite: str, verdict: str, usetex: bool, stem: str,
                           tables: list) -> bool:
    """The lineages against each other on one suite under one verdict: the hit rate over the
    generations, one panel per method, and the rate pooled over the generations in the last
    panel. False when nothing was found."""
    combos = [(m, v) for m in grid.args.methods for v in grid.args.variants
              if (m, v) in METHOD_LABELS]
    cells = {
        (mixture, combo): [grid.attack_cell(mixture, suite, *combo, verdict, g)
                           for g in grid.generations]
        for mixture in grid.mixtures for combo in combos
    }
    combos = [c for c in combos if any(cells[(m, c)][g] for m in grid.mixtures
                                       for g in range(len(grid.generations)))]
    if not combos:
        return False

    def rate_of(cell) -> float:
        if cell is None or cell["aborted"] or cell["usable"] == 0:
            return np.nan
        return 100.0 * cell["hits"] / cell["usable"]

    styles = lineage_style(grid)
    n_panels = len(combos) + 1
    figure, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels + 1.2, 5.6))
    labels = [EndLabels(axis) for axis in axes[:-1]]
    for index, combo in enumerate(combos):
        axis = axes[index]
        for mixture in grid.mixtures:
            colour, marker = styles[mixture]
            rates = np.array([rate_of(c) for c in cells[(mixture, combo)]])
            if np.all(np.isnan(rates)):
                continue
            axis.plot(grid.generations, rates, marker=marker, color=colour,
                      label=lineage_label(mixture))
            last = int(np.max(np.flatnonzero(~np.isnan(rates))))
            labels[index].add(grid.generations[last], rates[last], lineage_label(mixture), colour)
        axis.set_title(METHOD_LABELS[combo], fontsize=15)
        axis.set_xlabel("collapse generation")
        axis.set_xticks(grid.generations)
        axis.set_ylim(-4, 104)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
        axis.margins(x=0.12)
        axis.legend(fontsize=10, loc="best")
        tidy(axis)
    axes[0].set_ylabel("hit rate (share of attackable targets broken)")

    # pooled over the generations: one group of bars per method, one bar per lineage
    axis = axes[-1]
    n_lineages = len(grid.mixtures)
    width = 0.8 / n_lineages
    table_rows = []
    for li, mixture in enumerate(grid.mixtures):
        colour, _ = styles[mixture]
        row = [lineage_label(mixture)]
        for ci, combo in enumerate(combos):
            hits = sum(c["hits"] for c in cells[(mixture, combo)]
                       if c and not c["aborted"] and c["usable"])
            usable = sum(c["usable"] for c in cells[(mixture, combo)]
                         if c and not c["aborted"])
            x = ci + (li - (n_lineages - 1) / 2) * width
            if usable:
                pooled = 100.0 * hits / usable
                axis.bar(x, pooled, width=width, color=colour, edgecolor=nord.SURFACE,
                         linewidth=1, label=lineage_label(mixture) if ci == 0 else None)
                axis.text(x, pooled + 1.5, percent(pooled / 100, usetex), ha="center",
                          va="bottom", fontsize=9, color=nord.TEXT_MUTED)
                row.append(f"{pooled:.1f}% ({hits}/{usable})")
            else:
                axis.bar(x, 0, width=width, color=colour,
                         label=lineage_label(mixture) if ci == 0 else None)
                row.append("–")
        table_rows.append(row)
    axis.set_xticks(range(len(combos)))
    axis.set_xticklabels([METHOD_LABELS[c] for c in combos])
    axis.set_ylim(0, 104)
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    axis.set_title("pooled over the generations", fontsize=15)
    axis.set_xlabel("method")
    axis.legend(fontsize=10, loc="best", title="lineage", title_fontsize=10)
    axis.grid(axis="x", visible=False)
    tidy(axis)

    figure.suptitle(f"{SUITES[suite]}, {verdict} verdict: hit rate by lineage", fontsize=16)
    figure.tight_layout()
    for group in labels:
        group.place()
    save_figure(figure, stem)
    plt.close(figure)
    tables.append((f"{SUITES[suite]}, {verdict} verdict: hit rate pooled over the generations "
                   "(hits / attackable targets)",
                   ["lineage"] + [METHOD_LABELS[c] for c in combos], table_rows))
    return True


def figure_surrogate_quality(grid: Grid, usetex: bool, stem: str, tables: list) -> bool:
    """The logit surrogate as a predictor of its own hits, summed over the generations."""
    variants = [v for v in grid.args.variants if ("logit", v) in METHOD_LABELS]
    if "logit" not in grid.args.methods or not variants:
        return False
    verdict = grid.args.verdicts[0]
    keys = [("n_success", "real hits", nord.CATEGORICAL[0]),
            ("n_predicted", "predicted by the surrogate", nord.CATEGORICAL[2]),
            ("n_false_alarm", "false alarms", nord.CATEGORICAL[1]),
            ("n_missed", "missed", nord.CATEGORICAL[4])]
    figure, axes = plt.subplots(1, len(variants), figsize=(7.2 * len(variants), 5.4),
                                squeeze=False)
    table_rows, drew = [], False
    for column, variant in enumerate(variants):
        axis = axes[0][column]
        totals = []
        for mixture in grid.mixtures:
            summed = {key: 0 for key, _, _ in keys}
            verified = 0
            for generation in grid.generations:
                cell = grid.attack_cell(mixture, "vuln", "logit", variant, verdict, generation)
                if not cell:
                    continue
                quality = cell["surrogate_quality"]
                verified += int(quality.get("n_verified") or 0)
                for key, _, _ in keys:
                    summed[key] += int(quality.get(key) or 0)
            totals.append((mixture, verified, summed))
            if verified:
                table_rows.append([lineage_label(mixture), METHOD_LABELS[("logit", variant)],
                                   str(verified)] + [str(summed[k]) for k, _, _ in keys])
        if not any(v for _, v, _ in totals):
            axis.set_visible(False)
            continue
        drew = True
        width = 0.19
        xs = np.arange(len(totals))
        for offset, (key, label, colour) in enumerate(keys):
            heights = [summed[key] for _, _, summed in totals]
            bars = axis.bar(xs + (offset - 1.5) * width, heights, width=width * 0.92,
                            color=colour, label=label)
            for bar, value in zip(bars, heights):
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                          ha="center", va="bottom", fontsize=9, color=nord.TEXT_MUTED)
        axis.set_xticks(xs)
        axis.set_xticklabels([f"{lineage_label(m)}\n({v} checks)" for m, v, _ in totals],
                             fontsize=11)
        axis.set_ylabel("behavioural checks")
        axis.set_title(f"vulnerability targets, {METHOD_LABELS[('logit', variant)]}, "
                       f"{verdict} verdict", fontsize=14)
        axis.legend(fontsize=10, loc="upper left")
        tidy(axis)
    if not drew:
        plt.close(figure)
        return False
    figure.suptitle("the logit surrogate as a predictor of its own hits, summed over the "
                    "generations", fontsize=15)
    save_figure(figure, stem)
    plt.close(figure)
    tables.append((f"Surrogate quality, vulnerability targets, {verdict} verdict (summed over "
                   "generations 1..N)",
                   ["lineage", "variant", "checks"] + [label for _, label, _ in keys],
                   table_rows))
    return True


def figure_sink_screen(grid: Grid, usetex: bool, stem: str, tables: list) -> bool:
    """The affine-law roots per lineage: which targets the tilt can reach, and at what factor."""
    screens = [(m, grid.sink_screen(m)) for m in grid.mixtures]
    screens = [(m, s) for m, s in screens if s]
    if not screens:
        return False
    styles = lineage_style(grid)
    figure, axis = plt.subplots(figsize=(11, 1.4 * len(screens) + 2.6))
    table_rows = []
    for row, (mixture, per_task) in enumerate(screens):
        colour, marker = styles[mixture]
        usable = sorted(
            ((t, v["factor"]) for t, v in per_task.items() if v.get("usable") and v.get("factor")),
            key=lambda tv: tv[1],
        )
        insecure = sum(1 for v in per_task.values() if (v.get("margin_base") or 0) < 0)
        y = -row
        axis.axhline(y, color=nord.GRID, linewidth=0.8, zorder=0)
        # the names step up a level whenever two roots sit within a label's width of each
        # other on the log axis, so a cluster reads as a staircase instead of a smudge
        level, previous = 0, None
        for target, factor in usable:
            axis.plot([factor], [y], marker=marker, color=colour, markersize=9, zorder=3)
            close = previous is not None and np.log10(factor / previous) < 0.12
            level = (level + 1) % 4 if close else 0
            previous = factor
            axis.annotate(esc(target, usetex), (factor, y), xytext=(0, 8 + 18 * level),
                          textcoords="offset points", ha="center", va="bottom", fontsize=8,
                          rotation=40, color=nord.TEXT_MUTED)
        axis.text(1.01, y, f"{len(usable)}/{len(per_task)} usable",
                  transform=axis.get_yaxis_transform(), va="center", fontsize=11,
                  color=nord.TEXT_MUTED)
        table_rows.append([lineage_label(mixture), f"{len(usable)}/{len(per_task)}",
                           str(insecure),
                           ", ".join(f"{t} {f:.2f}" for t, f in usable) or "–"])
    axis.axvline(grid.args.sink_cap, color=nord.HIGHLIGHT, linestyle="-.", linewidth=1.5,
                 label=f"sink cap {grid.args.sink_cap:g}: roots above it run at the cap")
    axis.set_xscale("log")
    axis.set_yticks([-r for r in range(len(screens))])
    axis.set_yticklabels([lineage_label(m) for m, _ in screens])
    axis.set_ylim(-len(screens) + 0.25, 1.15)
    axis.set_xlabel("boundary factor $n^*$ (log scale)" if usetex
                    else "boundary factor n* (log scale)")
    axis.set_title("sink screen: targets the affine law marks usable, at their boundary factor",
                   fontsize=14, loc="left")
    axis.legend(fontsize=10, loc="lower right")
    axis.grid(True, axis="x")
    axis.grid(False, axis="y")
    tidy(axis)
    save_figure(figure, stem)
    plt.close(figure)
    tables.append(("Sink screen: usable roots per lineage (base insecure on the counted targets)",
                   ["lineage", "usable / targets", "base insecure", "roots n*"], table_rows))
    return True


# ────────────────────────────────── markdown ──────────────────────────────────────────────
def write_markdown(path: str, grid: Grid, tables: list) -> None:
    lines = [f"# Result tables: {grid.name}, root `{grid.root}`", ""]
    for title, header, rows in tables:
        lines += [f"## {title}", "", "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# ───────────────────────────────────── main ───────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", "-p", type=str, default=".", help="run root")
    parser.add_argument("--plots_path", "-pp", type=str, default="",
                        help="where the figures go (default: <root>/plots)")
    parser.add_argument("--model_size", "-msz", type=str, default="0.5b")
    parser.add_argument("--mixtures", nargs="+", default=["0.0", "0.5", "0.8", "1.0"])
    parser.add_argument("--generations", "-n", type=int, default=9,
                        help="highest generation index")
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--methods", nargs="+", default=["none", "logit"])
    parser.add_argument("--variants", nargs="+", default=["plain", "opt"])
    parser.add_argument("--verdicts", nargs="+", default=["sampled", "greedy"])
    parser.add_argument("--hold_anchor", "-ha", action="store_true")
    parser.add_argument("--human_generation_zero", "-hg0", action="store_true")
    parser.add_argument("--recursive_init", "-ri", action="store_true")
    parser.add_argument("--no_opt_entropy_match", dest="opt_entropy_match", action="store_false")
    parser.add_argument("--no_opt_sink_factor", dest="opt_sink_factor", action="store_false")
    parser.add_argument("--no_opt_sink_objective", dest="opt_sink_objective",
                        action="store_false")
    parser.add_argument("--sink_cap", type=float, default=2.5)
    parser.add_argument("--calibration", action="store_true",
                        help="also draw the surrogate calibration figure (off since 2026-10-29)")
    parser.add_argument("--no_usetex", action="store_true")
    parser.add_argument("--no_markdown", action="store_true")
    args = parser.parse_args()

    usetex = not args.no_usetex
    nord.apply_style(usetex, font_size=13)
    grid = Grid(args, args.model_size)
    plots = args.plots_path or os.path.join(args.path, "plots")
    os.makedirs(plots, exist_ok=True)
    prefix = os.path.join(plots, f"tables_{args.model_size}")
    tables: list = []
    written = []

    figure_utility(grid, usetex, f"{prefix}_utility", tables)
    written.append("utility")
    if figure_distribution(grid, usetex, f"{prefix}_distribution", tables):
        written.append("distribution")
    if args.calibration:
        figure_calibration(grid, usetex, f"{prefix}_calibration", tables)
        written.append("calibration")
    for suite in ("vuln", "correctness"):
        for verdict in args.verdicts:
            if figure_attack_grid(grid, suite, verdict, usetex,
                                  f"{prefix}_attack_{suite}_{verdict}", tables):
                written.append(f"attack_{suite}_{verdict}")
            if figure_rate_comparison(grid, suite, verdict, usetex,
                                      f"{prefix}_rates_{suite}_{verdict}", tables):
                written.append(f"rates_{suite}_{verdict}")
    if figure_surrogate_quality(grid, usetex, f"{prefix}_surrogate_quality", tables):
        written.append("surrogate_quality")
    if figure_sink_screen(grid, usetex, f"{prefix}_sink_screen", tables):
        written.append("sink_screen")
    if not args.no_markdown:
        write_markdown(f"{prefix}.md", grid, tables)
        written.append("markdown")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}tables{TColors.ENDC}: wrote "
          + ", ".join(written) + f" under {prefix}_*")


if __name__ == "__main__":
    main()
