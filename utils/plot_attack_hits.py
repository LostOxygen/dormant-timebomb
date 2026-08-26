"""Plots how many adversarial inputs were found per collapse generation: surrogate vs direct.

The question this figure answers is the one the surrogate exists for. `run_attack.py -sm none`
attacks the real checkpoint of a generation directly — the white-box upper bound, and a statement
about whether that generation is attackable *at all*. `-sm logit` attacks a surrogate built from
the base model and the generation-0 checkpoint alone, then validates against the same real
checkpoint. The gap between the two curves is the cost of not having the model you are attacking,
which is the attacker's actual situation.

**A hit is one verified selective hit**: a suffix that made the real collapsed model emit wrong
code while the pristine baseline still answered correctly, decided by executing both completions
against the task's unit tests (`is_selective_hit`). The count is over the tasks the capability gate
marked usable, since only those can carry one.

**Counts are budget dependent, and that is the caveat this figure carries.** Verification runs every
`--verify_every` steps of every restart, so the same suffix quality yields more recorded hits at a
longer `--num_steps`, more `--restarts` or a smaller `-ve`, and a run stopped early by
`--stop_on_success` records exactly one. Two curves are therefore only comparable at equal settings;
the script reads `num_steps` and `verify_every` back out of each result file and warns when they
differ. The per-generation success *rate* — usable tasks broken at least once, which is bounded and
budget-insensitive — stays in the console table beside the counts for that reason.

**A generation the capability gate stopped is a gap, not a zero.** Zero here means the search ran
and found nothing; a gated generation never ran, because the model already fails the clean prompts
and a wrong answer could not be attributed to the suffix. Those generations are shaded instead.

Reads only what `run_attack.py` already wrote — no models, no torch — so it is safe to run on a
login node while the sweeps are still going. Like utils/calibrate_surrogate.py, and unlike the
worker modules next to it, this one has a main() and is meant to be invoked:

    python -m utils.plot_attack_hits -p . -n 9 -bs 512
    python -m utils.plot_attack_hits -p . -n 9 -bs 512 -rdf 0.1 --methods logit,lora
"""

import argparse
import glob
import json
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.colors import TColors
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import factor_mode_tag, mixture_tag

# from the palette run_baseline.py plots with (Tableau's colorblind-safe ten). Verified for this
# pair rather than assumed: OKLab dE is 37.9 at normal vision and 38.2 / 29.5 / 33.7 under
# simulated deuteranopia / protanopia / tritanopia, against a target of 8
SERIES_COLORS: dict[str, str] = {
    "none": "#006BA4",  # dark blue — direct attack on the real checkpoint
    "logit": "#FF800E",  # orange — the logit surrogate
    "lora": "#A9373B",  # dark red — the weight-space surrogate, only when asked for
}
SERIES_LABELS: dict[str, str] = {
    "none": "direct (real checkpoint)",
    "logit": "logit surrogate",
    "lora": "lora surrogate",
}
MARKERS: dict[str, str] = {"none": "o", "logit": "s", "lora": "^"}
GATE_SHADE: str = "#ABABAB"
CAPABILITY_COLOR: str = "#595959"


def result_file(
    results_dir: str,
    generation: int,
    specifier_name: str,
    method: str,
    tag: str,
    factor_tag: str = "",
) -> str:
    """Rebuilds the name run_attack.py writes for one (generation, method) cell.

    Kept in one place because the name carries five things that all have to line up — generation,
    model short name, mixture, factor mode and method — and a wrong guess here is a silently missing
    curve rather than an error.

    The factor tag rides with the method for the same reason run_attack.py puts it there: it only
    exists in transfer mode, so the direct attack's cell is named without it even when the sweep
    beside it measured its factors.

    Args:
        results_dir (str): the attack_results directory to look in
        generation (int): collapse generation index
        specifier_name (str): trailing component of the model specifier
        method (str): "none" for the direct attack, otherwise the surrogate method
        tag (str): the mixture tag, "" at --real_data_fraction 0
        factor_tag (str): the factor rule's tag — "_nauto", "_ncal" or "_n<value>" — and "" for a
            sweep that took n from the generation index

    Returns:
        str: the path the result file would have
    """
    suffix = "" if method == "none" else f"{factor_tag}_{method}_surrogate"
    return os.path.join(
        results_dir, f"attack_gen{generation}_{specifier_name}{tag}{suffix}.json"
    )


def read_cell(path: str) -> dict | None:
    """Reduces one result file to the numbers the figure needs.

    Args:
        path (str): the result file to read

    Returns:
        dict | None: the hit count, the usable/broken counts behind it, the search budget and the
            gate verdict, or None when the file does not exist (that cell was never run)
    """
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    config = data.get("config", {})
    probe = data.get("capability_probe", {})
    usable = set(probe.get("usable", []))
    per_task = {task["task"]: len(task.get("successes", [])) for task in data.get("results", [])}
    # only usable tasks can carry a success: the others were excluded before the search, and a
    # "hit" on a task the collapsed model already failed measures collapse, not the attack
    broken = [name for name in usable if per_task.get(name, 0) > 0]

    return {
        "aborted": bool(data.get("aborted")),
        "reason": probe.get("reason", ""),
        "usable": sorted(usable),
        "n_usable": len(usable),
        "n_broken": len(broken),
        "n_hits": sum(count for name, count in per_task.items() if name in usable),
        "rate": len(broken) / len(usable) if usable else None,
        "factor": data.get("surrogate_factor"),
        "checkpoint": os.path.basename(data.get("collapsed_model", "")),
        # the search budget behind the count, for the comparability warning
        "budget": (config.get("num_steps"), config.get("verify_every"),
                   config.get("stop_on_success")),
    }


def collect(
    results_dir: str,
    generations: range,
    specifier_name: str,
    methods: list[str],
    tag: str,
    factor_tag: str = "",
) -> dict[str, dict[int, dict]]:
    """Reads every (method, generation) cell that exists on disk.

    Args:
        results_dir (str): the attack_results directory
        generations (range): the generation indices to look for
        specifier_name (str): trailing component of the model specifier
        methods (list[str]): "none" plus the surrogate methods to plot
        tag (str): the mixture tag
        factor_tag (str): the factor rule's tag, on the surrogate cells only

    Returns:
        dict: method -> generation -> the dict read_cell returned
    """
    cells: dict[str, dict[int, dict]] = {method: {} for method in methods}
    for method in methods:
        for generation in generations:
            cell = read_cell(
                result_file(
                    results_dir, generation, specifier_name, method, tag, factor_tag
                )
            )
            if cell is not None:
                cells[method][generation] = cell
    return cells


def warn_on_disagreement(cells: dict[str, dict[int, dict]]) -> None:
    """Warns when two methods disagree about which tasks were attackable at a generation.

    Both are probed against the *same* checkpoint, so the usable set should be identical. When it
    is not, the checkpoint changed between the two runs — run_baseline.py overwrites model_outputs/
    on a re-run — and the two curves are then measured against different models at that point. Same
    hazard the `void` verdict guards in run_transfer_experiment.py, and worth saying out loud
    because the figure cannot show it.

    Args:
        cells (dict): the structure collect() returned

    Returns:
        None
    """
    methods = [m for m in cells if cells[m]]
    for generation in sorted({g for method in methods for g in cells[method]}):
        present = {m: cells[m][generation] for m in methods if generation in cells[m]}
        sets = {m: tuple(cell["usable"]) for m, cell in present.items()}
        if len(set(sets.values())) > 1:
            print(
                f"## {TColors.WARNING}generation {generation}: the runs disagree on which tasks "
                f"were attackable{TColors.ENDC}"
            )
            for method, usable in sets.items():
                print(f"##   {method:6s}: {', '.join(usable) or '(none)'}")
            print(
                "##   the checkpoint was probably retrained between the two runs, so these two "
                "points are not measured against the same model"
            )


def warn_on_budget(cells: dict[str, dict[int, dict]]) -> None:
    """Warns when the cells were not all searched with the same budget.

    A hit count is a count of *recorded* successes, and verification runs every --verify_every
    steps of every restart — so a longer --num_steps or a smaller -ve inflates it for the same
    suffix quality, and --stop_on_success caps it at one. Unlike the success rate, the count
    therefore only means something across cells that were searched the same way.

    Args:
        cells (dict): the structure collect() returned

    Returns:
        None
    """
    budgets = {
        cell["budget"] for series in cells.values() for cell in series.values()
    }
    if len(budgets) <= 1:
        return
    print(
        f"## {TColors.WARNING}the cells were not all searched with the same budget, so their hit "
        f"counts are not directly comparable{TColors.ENDC}"
    )
    for num_steps, verify_every, stop_on_success in sorted(
        budgets, key=lambda b: tuple(-1 if v is None else int(v) for v in b)
    ):
        print(
            f"##   --num_steps {num_steps}, --verify_every {verify_every}"
            + (", --stop_on_success" if stop_on_success else "")
        )
    print("##   the success rate column below is the budget-insensitive comparison")


def print_table(cells: dict[str, dict[int, dict]], generations: range) -> None:
    """Prints the numbers behind the figure, including the pooled totals.

    Args:
        cells (dict): the structure collect() returned
        generations (range): the generation indices covered

    Returns:
        None
    """
    methods = list(cells)
    header = f"## {'gen':>4}  " + "  ".join(f"{SERIES_LABELS[m]:>26s}" for m in methods)
    print(header)
    print("## " + "-" * (len(header) - 3))
    for generation in generations:
        row = f"## {generation:>4}  "
        for method in methods:
            cell = cells[method].get(generation)
            if cell is None:
                row += f"{'not run':>26s}  "
            elif cell["rate"] is None:
                row += f"{'gate stopped it':>26s}  "
            else:
                noun = "hit" if cell["n_hits"] == 1 else "hits"
                row += (
                    f"{cell['n_hits']} {noun} "
                    f"({cell['n_broken']}/{cell['n_usable']} tasks, "
                    f"{cell['rate']:.0%})".rjust(26) + "  "
                )
        print(row)
    print("## " + "-" * (len(header) - 3))
    for method in methods:
        hits, broken, usable = pooled(cells[method])
        share = f"{broken / usable:.0%}" if usable else "n/a"
        print(
            f"## {SERIES_LABELS[method]:>26s}: {hits} adversarial inputs over all generations, "
            f"on {broken}/{usable} usable tasks ({share})"
        )


def pooled(series: dict[int, dict]) -> tuple[int, int, int]:
    """Totals over the generations, for the legend and the console summary.

    Args:
        series (dict): generation -> cell, for one method

    Returns:
        tuple: (hits, tasks broken, usable tasks) summed over every generation that ran
    """
    hits = sum(cell["n_hits"] for cell in series.values())
    broken = sum(cell["n_broken"] for cell in series.values())
    usable = sum(cell["n_usable"] for cell in series.values())
    return hits, broken, usable


def style(usetex: bool) -> None:
    """Applies the house plot style (run_baseline.py's, minus the seaborn palette).

    Args:
        usetex (bool): render text with LaTeX. Off gives a figure on a machine without a TeX
            install, at the cost of not matching the other figures' typography

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


def plot(
    cells: dict[str, dict[int, dict]],
    generations: range,
    plot_stem: str,
    title: str,
) -> None:
    """Draws the two-panel figure and writes it as .png and .pdf.

    The denominator gets its own panel rather than a second y-axis: the rate and the task count
    are different measures, and putting them on one pair of axes would invite reading a crossing
    that does not exist.

    Args:
        cells (dict): the structure collect() returned
        generations (range): the generation indices covered
        plot_stem (str): output path without the extension
        title (str): figure title, already LaTeX-safe

    Returns:
        None
    """
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1]
    )

    # generations nobody could attack, shaded once across both panels so the curve's gaps are
    # explained rather than left to be read as zeros
    gated = [
        generation
        for generation in generations
        if any(
            cells[m].get(generation) and cells[m][generation]["rate"] is None for m in cells
        )
        and not any(
            cells[m].get(generation) and cells[m][generation]["rate"] is not None for m in cells
        )
    ]
    for index, generation in enumerate(gated):
        for axis in (top, bottom):
            axis.axvspan(
                generation - 0.5,
                generation + 0.5,
                color=GATE_SHADE,
                alpha=0.25,
                linewidth=0,
                label="capability gate stopped the run" if index == 0 and axis is top else None,
            )

    ceiling = 1
    for method, series in cells.items():
        # a gated generation is dropped rather than drawn at zero: nothing was searched there
        points = [
            (generation, cell["n_hits"])
            for generation, cell in sorted(series.items())
            if cell["rate"] is not None
        ]
        if not points:
            continue
        hits, _, _ = pooled(series)
        ceiling = max(ceiling, max(p[1] for p in points))
        top.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=MARKERS[method],
            markersize=9,
            linewidth=2,
            color=SERIES_COLORS[method],
            label=f"{SERIES_LABELS[method]} — {hits} in total",
        )

    top.set_ylabel("adversarial inputs found")
    # headroom for the legend, and a floor below 0 so a zero sits on the axis rather than under it
    top.set_ylim(-0.04 * ceiling, 1.35 * ceiling)
    top.set_title(title)
    top.legend(loc="upper right")

    # the denominator, so a rate over one task is not read like a rate over five
    counts = {
        generation: max(
            (cells[m][generation]["n_usable"] for m in cells if generation in cells[m]),
            default=0,
        )
        for generation in generations
    }
    bottom.bar(
        list(counts),
        list(counts.values()),
        color=CAPABILITY_COLOR,
        width=0.55,
    )
    bottom.set_ylabel("usable\ntasks")
    bottom.set_xlabel("collapse generation")
    bottom.set_xticks(list(generations))
    bottom.set_ylim(0, max([*counts.values(), 1]) + 0.5)

    for axis in (top, bottom):
        for spine in axis.spines.values():
            spine.set_color("black")

    figure.tight_layout()
    os.makedirs(os.path.dirname(plot_stem) or ".", exist_ok=True)
    figure.savefig(f"{plot_stem}.pdf")
    figure.savefig(f"{plot_stem}.png", dpi=200)
    plt.close(figure)


def infer_block_size(results_dir: str, specifier_name: str) -> int | None:
    """Not needed for the result names, but the plot name carries a block size like every artifact.

    The attack result files do not have the block size in their names, so it is taken from the
    checkpoint one of them recorded. Returns None when nothing could be read, and the caller then
    insists on --block_size rather than naming a figure after a guess.

    Args:
        results_dir (str): the attack_results directory
        specifier_name (str): trailing component of the model specifier

    Returns:
        int | None: the block size found in a recorded checkpoint path
    """
    for path in sorted(glob.glob(os.path.join(results_dir, f"attack_gen*_{specifier_name}*.json"))):
        with open(path, encoding="utf-8") as handle:
            checkpoint = json.load(handle).get("collapsed_model", "")
        match = re.search(r"_bs(\d+)_", checkpoint)
        if match:
            return int(match.group(1))
    return None


def main(
    path: str = ".",
    num_generations: int = 9,
    start_generation: int = 0,
    block_size: int = 0,
    model_size: str = "",
    model_specifier: str = "",
    methods: str = "logit",
    real_data_fraction: float = 0.0,
    surrogate_factor: str = "",
    no_usetex: bool = False,
) -> None:
    """Reads the attack results of one run and plots the success rates over its generations.

    Args:
        path (str): root holding attack_results/
        num_generations (int): highest generation index to include
        start_generation (int): lowest generation index to include
        block_size (int): block size for the figure name; 0 reads it off a recorded checkpoint
        model_size (str): parameter count off the Qwen2.5-Coder ladder, shorthand for the specifier
        model_specifier (str): the model the run attacked; its short name is in every result name
        methods (str): comma separated surrogate methods to plot against the direct attack
        real_data_fraction (float): the mixture the run used, part of the result file names
        surrogate_factor (str): the --surrogate_factor the sweep passed to run_attack.py. Only its
            *mode* matters here: "auto" results are filed under their own names, every other form
            under the plain ones
        no_usetex (bool): render without LaTeX, for a machine with no TeX install

    Returns:
        None

    Raises:
        SystemExit: no result files matched, or the block size could not be determined
    """
    specifier = resolve_model_specifier(model_size, model_specifier)
    specifier_name = specifier.split("/")[-1]
    results_dir = os.path.join(path, "attack_results")
    tag = mixture_tag(real_data_fraction)
    factor_tag = factor_mode_tag(surrogate_factor)
    generations = range(start_generation, num_generations + 1)

    selected = ["none"] + [m.strip() for m in methods.split(",") if m.strip()]
    unknown = [m for m in selected if m not in SERIES_COLORS]
    if unknown:
        raise SystemExit(
            f"--methods {', '.join(unknown)} is not one of {', '.join(SERIES_LABELS)} "
            f"('none', the direct attack, is always plotted)"
        )

    print(
        f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Adversarial inputs found"
        f"{TColors.ENDC}"
    )
    print(f"##   results: {results_dir}")
    print(f"##   model:   {specifier_name}{tag or '  (no data mixture)'}")
    if factor_tag:
        print(f"##   factor:  measured per generation (files tagged {factor_tag})")

    cells = collect(results_dir, generations, specifier_name, selected, tag, factor_tag)
    if not any(cells.values()):
        raise SystemExit(
            f"no result files under {results_dir} matched "
            f"attack_gen<{start_generation}..{num_generations}>_{specifier_name}{tag}"
            f"{factor_tag}*.json — check --path, --model_size/--model_specifier, "
            f"--real_data_fraction (the mixture is part of the name) and --surrogate_factor (an "
            f"sweep under any rule other than n = g + 1 marks its surrogate files)"
        )

    warn_on_disagreement(cells)
    warn_on_budget(cells)
    print_table(cells, generations)

    if block_size <= 0:
        block_size = infer_block_size(results_dir, specifier_name)
        if block_size is None:
            raise SystemExit(
                "could not read a block size off the recorded checkpoints; pass --block_size so "
                "the figure is named like every other artifact of this run"
            )

    style(usetex=not no_usetex)
    # no underscores: usetex renders the title, and the mixture belongs in it because it changes
    # both curves at once
    mixture = "no data mixture" if real_data_fraction <= 0 else (
        f"real data fraction {real_data_fraction:g}"
    )
    # the two curves were searched against different proxies depending on this, so it belongs
    # beside the mixture rather than only in the file name
    factor_note = ", measured factor" if factor_tag else ""
    # plots/ sits outside --path, exactly like the perplexity figures, so everything that
    # distinguishes two runs has to be in the file name
    stem = f"plots/attack_hits_bs{block_size}_{specifier_name}{tag}{factor_tag}"
    # no underscores in the title either: usetex renders it
    label = specifier_name.replace("_", " ")
    plot(
        cells,
        generations,
        plot_stem=stem,
        title=f"Adversarial inputs found per generation\n({label}, {mixture}{factor_note})",
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Saved the figure under: "
        f"{TColors.HEADER}{stem}.<png,pdf>{TColors.ENDC}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot adversarial inputs found per collapse generation: surrogate vs direct"
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=".",
        help="root holding attack_results/ (default: current directory)",
    )
    parser.add_argument(
        "--num_generations",
        "-n",
        type=int,
        default=9,
        help="highest generation index to include (default: 9)",
    )
    parser.add_argument(
        "--start_generation",
        "-s",
        type=int,
        default=0,
        help="lowest generation index to include (default: 0)",
    )
    parser.add_argument(
        "--block_size",
        "-bs",
        type=int,
        default=0,
        help="block size for the figure name. 0 reads it off a checkpoint path recorded in one "
        "of the result files (default: 0)",
    )
    parser.add_argument(
        "--methods",
        "-m",
        type=str,
        default="logit",
        help="comma separated surrogate methods to plot against the direct attack, which is "
        "always included (default: logit)",
    )
    parser.add_argument(
        "--real_data_fraction",
        "-rdf",
        type=float,
        default=0.0,
        help="the --real_data_fraction the collapse run used. Part of the result file names, so "
        "a mixed run needs it here to be found at all (default: 0.0)",
    )
    parser.add_argument(
        "--surrogate_factor",
        "-sf",
        type=str,
        default="",
        help="the --surrogate_factor the sweep ran with. Only the *rule* is used, to rebuild the "
        "marker run_attack.py put in the name: 'auto' -> _nauto, 'calibrated' -> _ncal, a number "
        "-> _n<value>, and the default n = g + 1 -> no marker (default: '')",
    )
    parser.add_argument(
        "--no_usetex",
        action="store_true",
        help="render without LaTeX. The other figures in this repo use it, so only pass this on "
        "a machine with no TeX install",
    )
    add_model_arguments(parser, role="the model the run attacked")
    args = parser.parse_args()
    main(**vars(args))
