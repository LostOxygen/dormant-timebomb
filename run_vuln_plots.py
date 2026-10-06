"""draws every figure of the vulnerability attack's evaluation

Reads the artifacts the evaluation produces and writes one figure per question:

    attack_success   E1  hit rate over attributable targets, per generation and model
    target_matrix    E1  which weakness classes are reachable, and why the rest are excluded
    controls         E2  optimised suffixes against random ones and against the init string
    surrogate        E3  logit-extrapolation transfer against the direct upper bound
    temperature      E4  survival of working suffixes when decoding is sampled rather than greedy
    rdf_window       A1  the attackable window as a function of the real-data mixture
    ppl_roc          D1  a perplexity filter as a defence, and what it costs the attacker

Inputs. ``attack_success``, ``target_matrix``, ``surrogate`` and ``rdf_window`` read the attack's
own result JSONs through utils/vuln_results.py. ``controls`` and ``temperature`` read
``utils/verify_suffixes.py`` output, and ``ppl_roc`` reads ``utils/suffix_perplexity.py`` output;
both are written by run_vuln_eval.sh into the same results directory.

Every rate is over *attributable* targets --- those the capability probe found both models
answering correctly without a suffix. A generation where nothing is attributable is drawn as a gap
rather than as a zero, because "the attack found nothing" and "there was nothing to attack" are
different statements and averaging them together understates the first.

Style comes from utils/plotting.py so these figures match the perplexity ones, including the
colour semantics: blue is the real/direct measurement, orange the surrogate or the attacker-side
quantity, grey a reference.
"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.colors import TColors
from utils.naming import default_run_tag, find_run_tag
from utils.plotting import (
    ANCHOR_COLOR,
    BASELINE_COLOR,
    GENERATION_COLORS,
    SURROGATE_COLOR,
    apply_perplexity_style,
    save_figure,
)
from utils.vuln_results import (
    TARGET_CWE,
    load_records,
    target_order,
    unknown_targets,
)

RESULTS_PATH: str = "./attack_results/"
PLOTS_PATH: str = "./plots/"

# the four outcomes a (target, generation) cell can have. The two exclusions are opposites and are
# coloured apart on purpose: one means the pristine model is already insecure, the other that the
# collapsed model no longer writes classifiable code
OUTCOME_COLORS: dict = {
    "hit": GENERATION_COLORS[0],
    "no-hit": GENERATION_COLORS[3],
    "baseline-insecure": GENERATION_COLORS[8],
    "collapsed-broken": GENERATION_COLORS[4],
}
OUTCOME_LABELS: dict = {
    "hit": "insecure code elicited",
    "no-hit": "attacked, no hit",
    "baseline-insecure": "excluded: baseline already insecure",
    "collapsed-broken": "excluded: collapsed model broken",
}


def mixture_colors(count: int) -> list:
    """A sequential palette for the real-data mixture, which is an *ordered* variable.

    GENERATION_COLORS is categorical and its first entries are two near-identical blues, so a
    ten-value mixture sweep drawn from it is unreadable. viridis also matches the heatmap the
    mixture figure puts beside these lines, so the same fraction is the same colour in both panels.
    """
    return [plt.cm.viridis(value) for value in np.linspace(0.05, 0.9, max(count, 1))]


def escape(text: str, usetex: bool) -> str:
    """Escapes underscores so target names survive LaTeX text mode.

    Every target is named like ``fetch_user``, and with ``text.usetex`` on an unescaped underscore
    is a hard LaTeX error rather than a rendering artefact --- it would abort every figure here.
    """
    return text.replace("_", r"\_") if usetex else text


def target_label(name: str, usetex: bool) -> str:
    """Target name with its weakness class, e.g. ``fetch\\_user (CWE-89)``."""
    cwe = TARGET_CWE.get(name)
    return escape(name, usetex) + (f" ({cwe})" if cwe else "")


def read_json_glob(results_path: str, pattern: str) -> list:
    """Loads every JSON matching `pattern` under `results_path`, skipping unreadable ones."""
    payloads = []
    for path in sorted(glob.glob(os.path.join(results_path, pattern))):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        payload["_path"] = path
        payloads.append(payload)
    return payloads


def stem(args, figure: str, tag: str = "") -> str:
    """Output path of a figure, following the repo's artifact naming."""
    return os.path.join(args.plots_path, f"vuln_{figure}_bs{args.block_size}{tag}")


# ─────────────────────────────── E1: the attack works ─────────────────────────────────────
def figure_attack_success(records: list, args) -> list:
    """Hit rate against generation, one panel per model, one line per mixture.

    The lower panel carries the denominator. Without it a hit rate is unreadable: 1/1 and 4/4 are
    both 100%, and on a collapsing model the denominator is what moves.
    """
    direct = [r for r in records if r.surrogate_method == "none"]
    if not direct:
        return []
    models = sorted({r.specifier_name for r in direct})
    mixtures = sorted({r.real_data_fraction for r in direct})
    figure, axes = plt.subplots(
        2, len(models), figsize=(7.5 * len(models), 9), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for column, model in enumerate(models):
        top, bottom = axes[0][column], axes[1][column]
        palette = mixture_colors(len(mixtures))
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
        for index, mixture in enumerate(mixtures):
            rows = sorted(
                (r for r in direct if r.specifier_name == model
                 and abs(r.real_data_fraction - mixture) < 1e-9),
                key=lambda r: r.generation,
            )
            if not rows:
                continue
            colour = palette[index]
            label = f"real data {mixture:.0%}"
            generations = [r.generation for r in rows]
            marker = markers[index % len(markers)]
            top.plot(generations, [r.hit_rate for r in rows], marker=marker, markersize=9,
                     linewidth=3, color=colour, label=label)
            bottom.plot(generations, [len(r.attackable) for r in rows], marker=marker,
                        markersize=7, linewidth=2, color=colour)
        top.set_title(escape(model, args.usetex))
        top.set_ylim(-0.05, 1.05)
        top.set_ylabel("hit rate over attributable targets" if column == 0 else "")
        bottom.set_ylabel("attributable targets" if column == 0 else "")
        bottom.set_xlabel("collapse generation")
        bottom.set_ylim(0, len(TARGET_CWE) + 0.4)
        if column == 0:
            top.legend(loc="upper right")

    return [(figure, stem(args, "attack_success"))]


def figure_target_matrix(records: list, args) -> list:
    """Per-target outcome grid, one panel per (model, mixture) of the requested mode.

    Reading the exclusions is the point: a blank row is a target the attack never got to try, and
    the two exclusion colours say which side of the criterion was missing.
    """
    selected = [r for r in records if r.surrogate_method == args.mode]
    if not selected:
        return []
    panels = sorted({(r.specifier_name, r.real_data_fraction) for r in selected})
    targets = target_order(selected)
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(11, 1.1 * len(targets) * len(panels) + 2.2), squeeze=False,
    )

    for row, (model, mixture) in enumerate(panels):
        axis = axes[row][0]
        rows = sorted(
            (r for r in selected if r.specifier_name == model
             and abs(r.real_data_fraction - mixture) < 1e-9),
            key=lambda r: r.generation,
        )
        generations = [r.generation for r in rows]
        for y, target in enumerate(targets):
            for x, record in enumerate(rows):
                task = record.tasks.get(target)
                if task is None:
                    continue
                outcome = "hit" if task.hit else (task.exclusion or "no-hit")
                axis.add_patch(
                    plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                  facecolor=OUTCOME_COLORS.get(outcome, "#FFFFFF"),
                                  edgecolor="white", linewidth=2)
                )
                if task.hit:
                    axis.text(x, y, str(task.n_success), ha="center", va="center",
                              color="white", fontsize=13)
        axis.set_xlim(-0.5, len(rows) - 0.5)
        axis.set_ylim(len(targets) - 0.5, -0.5)
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels(generations)
        axis.set_yticks(range(len(targets)))
        axis.set_yticklabels([target_label(t, args.usetex) for t in targets])
        axis.set_xlabel("collapse generation")
        axis.set_title(f"{escape(model, args.usetex)}, real data {mixture:.0%}")
        axis.grid(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=OUTCOME_COLORS[key])
               for key in OUTCOME_LABELS]
    figure.legend(handles, list(OUTCOME_LABELS.values()), loc="lower center",
                  ncol=2, frameon=False)
    figure.subplots_adjust(bottom=0.16)
    return [(figure, stem(args, "target_matrix", f"_{args.mode}"))]


# ──────────────────────────────── E2: the controls ────────────────────────────────────────
def figure_controls(records: list, args) -> list:
    """Optimised suffixes against random ones and against the unoptimised init string, per target.

    The alternative explanation this rules out is that a collapsed model is merely fragile, so any
    perturbation of that token length would flip it. The comparison is **per verification attempt**,
    which is the only fair one: the search performs a fixed number of behavioural checks and lands
    some hits, so a control gets the same number of tries at the same target against the same two
    models. Rates over *suffixes that were kept* would be 100% by construction.

    Broken down per target rather than pooled, because fragility is not uniform across targets --- 
    one target being flippable by noise would otherwise be averaged into a reassuring aggregate and
    hide exactly the confound this figure exists to expose.
    """
    greedy = [p for p in read_json_glob(args.results_path, "suffix_verification_*greedy*.json")]
    controls = [p for p in greedy if p.get("source") in ("random", "init")]
    if not controls:
        return []

    tallies: dict = {}

    def add(target: str, source: str, hits: int, total: int) -> None:
        bucket = tallies.setdefault(target, {}).setdefault(source, [0, 0])
        bucket[0] += hits
        bucket[1] += total

    # **matched cells only.** A control is run at one (model, generation, mixture), and the model's
    # fragility varies strongly with the generation, so pooling the optimised rate over every
    # generation while the control covers one would compare the attack's worst cells against the
    # control's easiest. Only cells where a control exists enter either bar
    covered = {
        (payload.get("specifier_name"), payload.get("collapsed_generation"),
         round(float(payload.get("real_data_fraction", 0.0)), 6))
        for payload in controls
    }
    selected = [
        r for r in records
        if r.surrogate_method == args.mode
        and (r.specifier_name, r.generation, round(r.real_data_fraction, 6)) in covered
    ]
    if not selected:
        return []
    for record in selected:
        for name, task in record.tasks.items():
            if task.n_verified:
                add(name, "optimized", task.n_success, task.n_verified)
    # controls: one attempt per sample scored
    for payload in controls:
        for row in payload.get("results") or []:
            add(row["task"], payload["source"], row.get("n_hit", 0), row.get("n_samples", 0))

    targets = [t for t in TARGET_CWE if t in tallies] + [
        t for t in tallies if t not in TARGET_CWE]
    sources = [s for s in ("optimized", "random", "init")
               if any(s in tallies[t] for t in targets)]
    if not sources or not targets:
        return []
    colours = {"optimized": BASELINE_COLOR, "random": SURROGATE_COLOR, "init": ANCHOR_COLOR}
    labels = {"optimized": "optimised suffix", "random": "random suffix",
              "init": "init string, unoptimised"}

    width = 0.8 / len(sources)
    figure, axis = plt.subplots(figsize=(max(9, 2.6 * len(targets)), 6.5))
    for index, source in enumerate(sources):
        offsets, heights, notes = [], [], []
        for position, target in enumerate(targets):
            hits, total = tallies[target].get(source, [0, 0])
            offsets.append(position + index * width - 0.4 + width / 2)
            heights.append(hits / total if total else 0.0)
            notes.append(f"{hits}/{total}" if total else "n/a")
        axis.bar(offsets, heights, width=width * 0.92, color=colours[source],
                 label=labels[source])
        for offset, height, note in zip(offsets, heights, notes):
            axis.text(offset, height + 0.008, note, ha="center", fontsize=11, rotation=90)

    axis.set_xticks(range(len(targets)))
    axis.set_xticklabels([target_label(t, args.usetex) for t in targets], rotation=20,
                         ha="right")
    axis.set_ylabel("hits per verification attempt")
    cells = ", ".join(
        f"gen {generation} @ {mixture:.0%}"
        for _, generation, mixture in sorted(covered)
    )
    axis.set_title(f"matched cells: {cells}", fontsize=14)
    axis.legend(loc="upper right")
    axis.set_ylim(0, min(1.0, max(
        [h / t for target in targets for h, t in
         (tallies[target].get(s, [0, 1]) for s in sources) if t] + [0.05]) * 1.35))
    return [(figure, stem(args, "controls", f"_{args.mode}"))]


# ─────────────────────────── E3: the surrogate as a stand-in ──────────────────────────────
def figure_surrogate(records: list, args) -> list:
    """Transfer against the direct upper bound, and the surrogate scored as a predictor.

    The upper panel is the attack question: does a suffix found without the collapsed checkpoint
    still work on it. The lower panel is the proxy question an attacker actually faces: can the
    surrogate's own verdict be trusted, measured as agreement, precision and recall against the
    real model.
    """
    mixtures = sorted({r.real_data_fraction for r in records if r.transfer})
    if not mixtures:
        return []
    figures = []
    for mixture in mixtures:
        rows = {method: sorted(
            (r for r in records if r.surrogate_method == method
             and abs(r.real_data_fraction - mixture) < 1e-9), key=lambda r: r.generation)
            for method in ("none", "logit")}
        figure, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]})
        for method, colour, label in (
            ("none", BASELINE_COLOR, "direct (attacker holds the checkpoint)"),
            ("logit", SURROGATE_COLOR, "logit surrogate (base + generation 0 only)"),
        ):
            series = rows[method]
            if not series:
                continue
            axes[0].plot([r.generation for r in series], [r.hit_rate for r in series],
                         marker="o", markersize=9, linewidth=3, color=colour, label=label)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("hit rate over attributable targets")
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"real data {mixture:.0%}")

        quality = [(r.generation, r.surrogate_quality) for r in rows["logit"]
                   if r.surrogate_quality]
        if quality:
            generations = [g for g, _ in quality]
            for key, colour, marker in (
                ("agreement", ANCHOR_COLOR, "s"), ("precision", SURROGATE_COLOR, "o"),
                ("recall", GENERATION_COLORS[2], "^"),
            ):
                axes[1].plot(generations, [q.get(key, float("nan")) for _, q in quality],
                             marker=marker, markersize=8, linewidth=2.5, color=colour, label=key)
            axes[1].legend(loc="upper right", ncol=3)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_ylabel("surrogate as predictor")
        axes[1].set_xlabel("collapse generation")
        tag = f"_rdf{mixture:g}" if mixture else ""
        figures.append((figure, stem(args, "surrogate", tag)))
    return figures


# ─────────────────────── E4: does a hit survive sampled decoding ───────────────────────────
def figure_temperature(records: list, args) -> list:
    """Survival of verified hits when re-decoded sample by sample.

    A greedy attack run's hits are argmax claims; a sampled run's are majority claims over a few
    samples. Either way this re-scores them with more draws, and a hit that disappears is a weaker
    threat; the per-target spread is what says whether that is a property of the attack or of one
    target. Only the re-scores of the attack runs selected by --verification are drawn: the
    survival file carries the attack file's tag in its name (run_vuln_eval.sh), and mixing the two
    would average two different claims.
    """
    payloads = [p for p in read_json_glob(args.results_path, "suffix_verification_*sampled*.json")
                if p.get("source") == "optimized"
                and find_run_tag(os.path.basename(p["_path"])) == args.verification]
    if not payloads:
        return []
    per_target: dict = {}
    for payload in payloads:
        for row in payload.get("results") or []:
            per_target.setdefault(row["task"], []).append(row.get("survival", 0.0))
    if not per_target:
        return []

    targets = [t for t in TARGET_CWE if t in per_target] + [
        t for t in per_target if t not in TARGET_CWE]
    temperature = payloads[0].get("temperature", 0.7)
    figure, axis = plt.subplots(figsize=(max(8, 2.2 * len(targets)), 6))
    for index, target in enumerate(targets):
        values = per_target[target]
        axis.bar(index, float(np.mean(values)), width=0.6, color=GENERATION_COLORS[1],
                 zorder=2)
        axis.scatter([index] * len(values), values, color=ANCHOR_COLOR, s=28, zorder=3,
                     alpha=0.8)
        axis.text(index, 1.02, f"n={len(values)}", ha="center", fontsize=13)
    axis.axhline(1.0, color=ANCHOR_COLOR, linestyle="--", linewidth=2,
                 label="greedy verification (by construction)")
    axis.set_xticks(range(len(targets)))
    axis.set_xticklabels([target_label(t, args.usetex) for t in targets], rotation=20,
                         ha="right")
    axis.set_ylim(0, 1.12)
    axis.set_ylabel(f"fraction of samples still a hit (T={temperature})")
    axis.legend(loc="lower right")
    return [(figure, stem(args, "temperature"))]


# ───────────────────────────── A1: the real-data mixture ──────────────────────────────────
def figure_rdf_window(records: list, args) -> list:
    """Hit rate over the (mixture, generation) grid, with the attributable count beside it.

    This is the realism argument rather than a side experiment: pure self-training collapses so
    fast that the window where a hit is attributable at all is narrow, and the mixture is the dial
    that widens it. Cells with nothing attributable are left blank, not drawn as zero.
    """
    direct = [r for r in records if r.surrogate_method == args.mode]
    if not direct:
        return []
    mixtures = sorted({r.real_data_fraction for r in direct})
    generations = sorted({r.generation for r in direct})
    if len(mixtures) < 2:
        return []

    grid = np.full((len(mixtures), len(generations)), np.nan)
    counts = np.zeros((len(mixtures), len(generations)))
    for record in direct:
        row, column = mixtures.index(record.real_data_fraction), generations.index(
            record.generation)
        grid[row][column] = record.hit_rate
        counts[row][column] = len(record.attackable)

    figure, axes = plt.subplots(1, 2, figsize=(16, 1.0 * len(mixtures) + 3.5),
                                gridspec_kw={"width_ratios": [3, 2]})
    image = axes[0].imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                           origin="lower")
    for row in range(len(mixtures)):
        for column in range(len(generations)):
            if not np.isnan(grid[row][column]):
                axes[0].text(column, row, f"{grid[row][column]:.2f}", ha="center",
                             va="center", color="white", fontsize=12)
    axes[0].set_xticks(range(len(generations)))
    axes[0].set_xticklabels(generations)
    axes[0].set_yticks(range(len(mixtures)))
    axes[0].set_yticklabels([f"{m:.0%}" for m in mixtures])
    axes[0].set_xlabel("collapse generation")
    axes[0].set_ylabel("real data fraction")
    axes[0].set_title("hit rate")
    axes[0].grid(False)
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    palette = mixture_colors(len(mixtures))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for index, mixture in enumerate(mixtures):
        axes[1].plot(generations, counts[index], marker=markers[index % len(markers)],
                     markersize=7, linewidth=2.5, color=palette[index], label=f"{mixture:.0%}")
    axes[1].set_xlabel("collapse generation")
    axes[1].set_ylabel("attributable targets")
    axes[1].set_ylim(0, len(TARGET_CWE) + 0.4)
    axes[1].set_title("the attackable window")
    axes[1].legend(loc="upper right", ncol=2, title="real data")
    return [(figure, stem(args, "rdf_window", f"_{args.mode}"))]


# ────────────────────────── D1: perplexity filtering as a defence ──────────────────────────
def roc(positive: list, negative: list) -> tuple:
    """ROC of a threshold on one score, with the attack as the positive class.

    Written out rather than taken from scikit-learn, which is not a dependency of this repo. Ties
    are broken by order, which is immaterial for continuous perplexities.

    Args:
        positive (list): scores of the adversarial prompts
        negative (list): scores of the natural prompts

    Returns:
        tuple: false-positive rates, true-positive rates, and the area under the curve
    """
    if not positive or not negative:
        return [0.0, 1.0], [0.0, 1.0], float("nan")
    labelled = sorted([(s, 1) for s in positive] + [(s, 0) for s in negative],
                      key=lambda pair: -pair[0])
    true_positive = false_positive = 0
    true_rates, false_rates = [0.0], [0.0]
    for _, label in labelled:
        true_positive += label
        false_positive += 1 - label
        true_rates.append(true_positive / len(positive))
        false_rates.append(false_positive / len(negative))
    return false_rates, true_rates, float(np.trapezoid(true_rates, false_rates))


def figure_ppl_roc(records: list, args) -> list:
    """A perplexity filter on the incoming prompt, and what it would cost to evade.

    The defence a reviewer proposes first, so it is measured rather than conceded. The ASCII split
    is the attacker's answer: if the filter works because the suffix is gibberish, the next move is
    ``--allow_non_ascii`` or a fluency constraint, and the curve says how much that is worth.
    """
    payloads = read_json_glob(args.results_path, "suffix_perplexity_*.json")
    if not payloads:
        return []
    natural, adversarial, clean, ascii_split = [], [], [], {True: [], False: []}
    for payload in payloads:
        natural += [row["perplexity"] for row in payload.get("natural") or []]
        clean += [row["perplexity"] for row in payload.get("clean") or []]
        for row in payload.get("suffixes") or []:
            adversarial.append(row["perplexity"])
            ascii_split[bool(row.get("ascii", True))].append(row["perplexity"])
    if not natural or not adversarial:
        return []

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    false_rates, true_rates, area = roc(adversarial, natural)
    axes[0].plot(false_rates, true_rates, linewidth=3, color=BASELINE_COLOR,
                 label=f"all suffixes (AUC {area:.3f})")
    for is_ascii, colour in ((True, SURROGATE_COLOR), (False, GENERATION_COLORS[2])):
        scores = ascii_split[is_ascii]
        if len(scores) >= 5:
            sub_false, sub_true, sub_area = roc(scores, natural)
            name = "ASCII" if is_ascii else "non-ASCII"
            axes[0].plot(sub_false, sub_true, linewidth=2.5, linestyle="--", color=colour,
                         label=f"{name} (AUC {sub_area:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle=":", color=ANCHOR_COLOR, linewidth=2,
                 label="chance")
    axes[0].set_xlabel("false positives on natural prompts")
    axes[0].set_ylabel("adversarial prompts caught")
    axes[0].set_title("perplexity filter")
    axes[0].legend(loc="lower right")

    every = natural + adversarial + clean
    bins = np.logspace(np.log10(max(1e-3, min(every))), np.log10(max(every)), 45)
    axes[1].hist(natural, bins=bins, color=ANCHOR_COLOR, alpha=0.75, label="natural prompts")
    axes[1].hist(adversarial, bins=bins, color=SURROGATE_COLOR, alpha=0.75,
                 label="with adversarial suffix")
    # the confound control: the targets' own instructions, unmodified. A filter that flags these
    # too is detecting the task rather than the attack, and the ROC beside it would be measuring
    # the wrong thing. Drawn as lines because there are only as many as there are targets
    for index, value in enumerate(sorted(clean)):
        axes[1].axvline(value, color=GENERATION_COLORS[0], linewidth=2.5, linestyle="--",
                        label="clean target prompt" if index == 0 else None)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("perplexity under the pristine model")
    axes[1].set_ylabel("prompts")
    axes[1].set_title("score distributions")
    axes[1].legend(loc="upper right")
    return [(figure, stem(args, "ppl_roc"))]


FIGURES: dict = {
    "attack_success": figure_attack_success,
    "target_matrix": figure_target_matrix,
    "controls": figure_controls,
    "surrogate": figure_surrogate,
    "temperature": figure_temperature,
    "rdf_window": figure_rdf_window,
    "ppl_roc": figure_ppl_roc,
}


def main() -> None:
    """Draws the requested figures from whatever the evaluation has produced so far."""
    parser = argparse.ArgumentParser(description="Vulnerability attack evaluation figures")
    parser.add_argument("--results_path", "-rp", type=str, default=RESULTS_PATH,
                        help="directory holding the attack and analysis JSONs")
    parser.add_argument("--plots_path", "-pp", type=str, default=PLOTS_PATH,
                        help="where the figures are written")
    parser.add_argument("--figures", "-f", type=str, default="all",
                        help="comma separated subset of " + ",".join(FIGURES) + " (default: all)")
    parser.add_argument("--model_specifier_name", "-msn", type=str, default="",
                        help="restrict to one model's runs, e.g. Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--mode", "-m", type=str, default="none", choices=("none", "logit"),
                        help="which attack mode the single-mode figures use (default: none)")
    parser.add_argument("--block_size", "-bs", type=int, default=512,
                        help="block size, for the output file names (default: 512)")
    parser.add_argument("--verification", "-vf", type=str, default=default_run_tag(),
                        help="which attack runs to draw, by their verification-plus-anchor tag "
                        "in the result file names (utils.naming.run_tag), e.g. "
                        f"'{default_run_tag()}', '_T0.7p0.8k20x5' for sampled runs without the "
                        "anchor, or '' (empty) for greedy runs without it (default: the attack's "
                        "defaults). Runs with different hit rules make different claims and are "
                        "never drawn together")
    parser.add_argument("--no_usetex", dest="usetex", action="store_false",
                        help="render without LaTeX, for machines without a TeX install")
    parser.add_argument("--show", action="store_true", help="also open the figures")
    args = parser.parse_args()

    requested = list(FIGURES) if args.figures == "all" else [
        name.strip() for name in args.figures.split(",")]
    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")

    records = load_records(
        args.results_path,
        specifier_name=args.model_specifier_name,
        run_tag=args.verification,
    )
    if not records:
        raise SystemExit(
            f"{TColors.FAIL}no vulnerability results under {args.results_path} with "
            f"verification {args.verification!r}{TColors.ENDC}. Run ./run_vuln_eval.sh first, or "
            f"pass -vf '' for greedy runs / the tag of the sampled runs on disk."
        )
    missing = unknown_targets(records)
    if missing:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: no CWE label for {missing} — add them to "
              f"utils/vuln_results.TARGET_CWE")

    print(f"## {TColors.BOLD}{len(records)} attack result(s){TColors.ENDC} from "
          f"{args.results_path}, verification "
          f"{args.verification or 'greedy'}")
    apply_perplexity_style(args.usetex, font_size=18)
    for name in requested:
        drawn = FIGURES[name](records, args)
        if not drawn:
            print(f"##   {name:16s} {TColors.WARNING}skipped{TColors.ENDC}: its input is not "
                  f"in {args.results_path} yet")
            continue
        for figure, path in drawn:
            save_figure(figure, path, show=args.show)
            print(f"##   {name:16s} {TColors.OKGREEN}->{TColors.ENDC} {path}.pdf")


if __name__ == "__main__":
    main()
