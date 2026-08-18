"""drawing code of every perplexity figure

The *layout* lives here, in one place, and the callers differ only in which measurements they hand
it: run_baseline.py and run_extrapolation.py plot the corpora their generations trained on,
utils/evaluate_perplexity.py plots the train and the validation slice of the human dataset. All
three go through plot_perplexity_figure, so the figures stay comparable — a change to the panel
arrangement, the palette or the axis scales lands in all of them at once.

Every figure has the same two panels: the per-sample **histogram** on top, which is the shape of
the distribution, and the **median per generation** below it, which is that shape reduced to one
number per generation so the trend is readable. They share nothing but the data — the top panel's
x-axis is perplexity, the bottom one's is the generation index — so they are deliberately not
`sharex`.

Why the drawing is here and not in utils/evaluate_perplexity.py, which is the module that owns the
perplexity evaluation: that module imports unsloth at the top, before torch and transformers,
because it loads models itself and unsloth has to land its patches first. run_baseline.py
deliberately imports unsloth *never* — every model touching stage of it is a subprocess, which is
what allows the training to be data parallel — so importing the figure code out of
evaluate_perplexity.py would drag unsloth into the orchestrator through the back door. This module
stays unsloth free for the same reason utils/naming.py stays torch free.
"""

import os
from dataclasses import dataclass, field

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils.colors import TColors

# lower y-limit of the histogram panel. Everything below it is invisible, so it doubles as the
# threshold visible_perplexity_range() uses to decide where to cut the x-axis
Y_LIMIT_LOWER: float = 1e-5

# one colour per generation, dark blue -> light blue -> gray -> orange -> dark red, so the reading
# order of the histograms matches the collapse. Colourblind safe, and there are eleven of them for
# ten generations on purpose: the eleventh is the spare a longer run takes before the cycle wraps
GENERATION_COLORS: list = [
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

# the median panel's two curves and its reference line, in the same palette. Verified rather than
# assumed: OKLab dE 37.9 at normal vision, 38.2 / 29.5 / 33.7 under simulated deuteranopia /
# protanopia / tritanopia, against a target of 8
BASELINE_COLOR: str = "#006BA4"
SURROGATE_COLOR: str = "#FF800E"
ANCHOR_COLOR: str = "#595959"


@dataclass
class MedianSeries:
    """One curve of the median panel.

    The band is optional and is an interquartile range, not an error bar: it says how differently
    one model treats different samples, which is the spread that grows as the model collapses. A
    series whose values are single numbers rather than distributions (a surrogate re-scored at a
    fitted factor, say) simply carries none.
    """

    label: str
    medians: dict = field(default_factory=dict)
    band: dict = field(default_factory=dict)
    color: str = SURROGATE_COLOR
    marker: str = "s"
    linestyle: str = "-"


def generation_index(label: str) -> int:
    """Generation index out of a series label, ``-1`` when it does not end in one.

    Both label conventions in the repo end in the index — "Generation 3" from the histogram worker
    and "generation 3" from utils/evaluate_perplexity.py's Measurement — so the trailing token is
    the index for either.

    Args:
        label (str): the series label

    Returns:
        int: the parsed index, or -1
    """
    tail = label.split()[-1] if label else ""
    return int(tail) if tail.isdigit() else -1


def quantiles(perplexities: list) -> tuple:
    """Median and interquartile range over the finite values of one sample.

    Quantiles rather than mean +- std, because the distribution is heavy tailed by construction: a
    collapsed model assigns near-zero probability to a handful of samples and those dominate a mean.

    Args:
        perplexities (list): one model's per-sample perplexities

    Returns:
        tuple: (median, q25, q75), or (nan, nan, nan) when nothing finite is left
    """
    values = torch.tensor(list(perplexities), dtype=torch.float64)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(finite.median()),
        float(finite.quantile(0.25)),
        float(finite.quantile(0.75)),
    )


def visible_perplexity_range(
    perplexity_dict: dict, y_limit: float, num_bins: int = 401
) -> tuple:
    """
    Determines the x-axis range of the perplexity histogram which actually contains visible
    data.

    The perplexity distributions have extremely heavy tails: a small but non-negligible part
    of the samples is spread over many decades, so even a 99.9% quantile still reaches up to
    1e10. Those bins contain so few samples that they are drawn below the plot's lower
    y-limit and are invisible anyway, which would only leave the right side of the plot
    empty. The range is therefore cut off after the last bin whose density is still above the
    lower y-limit.

    Args:
        perplexity_dict (dict): the perplexities per generation
        y_limit (float): the lower y-limit of the plot, i.e., the visibility threshold
        num_bins (int): number of bins used to probe the densities

    Returns:
        tuple: (lower_limit, upper_limit, num_clipped, num_total), where the limits are
            rounded to full decades and num_clipped is the number of perplexities which are
            outside of the returned range
    """
    values = torch.tensor(
        [
            perplexity
            for perplexities in perplexity_dict.values()
            for perplexity in perplexities
        ]
    )
    # exp() of a large loss overflows to inf, which would turn the whole logspace into nan
    values = values[torch.isfinite(values) & (values > 0)]
    if values.numel() == 0:
        raise ValueError(
            "All perplexities are non-finite or non-positive, there is nothing to plot."
        )

    # probe the densities over the full range of the data first
    full_lower = 10 ** torch.floor(torch.log10(values.min()))
    full_upper = 10 ** torch.ceil(torch.log10(values.max()))
    edges = torch.logspace(
        torch.log10(full_lower), torch.log10(full_upper), steps=num_bins
    )
    bin_widths = edges[1:] - edges[:-1]

    # a bin is visible if any generation's density in it reaches the lower y-limit. seaborn's
    # stat="density" normalizes the counts by the number of samples and by the bin width, so
    # the same has to be done here to be able to compare against the y-limit
    visible = torch.zeros(len(bin_widths), dtype=torch.bool)
    for perplexities in perplexity_dict.values():
        generation_values = torch.tensor(perplexities)
        generation_values = generation_values[
            torch.isfinite(generation_values) & (generation_values > 0)
        ]
        if generation_values.numel() == 0:
            continue

        # bucketize returns 0 for values below the first edge, so shift into bin indices
        bin_indices = torch.bucketize(generation_values, edges).clamp(
            1, len(bin_widths)
        )
        counts = torch.bincount(bin_indices - 1, minlength=len(bin_widths))
        densities = counts / (generation_values.numel() * bin_widths)
        visible |= densities >= y_limit

    visible_bins = visible.nonzero().flatten()
    if visible_bins.numel() == 0:
        # nothing would be visible at all, so fall back to the full range of the data
        return full_lower.item(), full_upper.item(), 0, values.numel()

    lower_limit = 10 ** torch.floor(torch.log10(edges[visible_bins[0]]))
    upper_limit = 10 ** torch.ceil(torch.log10(edges[visible_bins[-1] + 1]))
    # keep at least one decade, otherwise both limits collapse onto the same value if all
    # visible perplexities fall onto the same power of ten
    upper_limit = torch.maximum(upper_limit, lower_limit * 10)

    num_clipped = int(((values < lower_limit) | (values > upper_limit)).sum())

    return lower_limit.item(), upper_limit.item(), num_clipped, values.numel()


def apply_perplexity_style(usetex: bool, font_size: float = 20) -> None:
    """The rcParams every perplexity figure is drawn with.

    Set globally rather than per-axes because seaborn's palette and style are global anyway, and a
    figure that is drawn after another one with different settings would otherwise silently pick up
    whatever the previous caller left behind.

    Args:
        usetex (bool): render text with LaTeX. Needs a LaTeX install; the callers expose a flag to
            turn it off, since it is the one part of the plotting that can abort a finished run
        font_size (float): base font size, the axis and tick sizes are derived from it

    Returns:
        None
    """
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette(GENERATION_COLORS))
    mpl.rcParams.update(
        {
            "text.usetex": usetex,
            "text.latex.preamble": r"\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": font_size,
            "font.weight": "bold",
            "axes.labelsize": font_size,
            "axes.labelweight": "bold",
            "axes.titlesize": font_size,
            "axes.titleweight": "bold",
            "legend.fontsize": font_size - 5,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "pdf.compression": 9,
        }
    )


def draw_histogram_panel(
    axis, perplexity_dict: dict, y_limit_lower: float = Y_LIMIT_LOWER, num_bins: int = 401
) -> tuple:
    """One step histogram per generation, log density over log perplexity.

    Density rather than counts, so generations of different sizes are comparable — the generated
    corpora are not all the same length — and log on both axes because the distribution spans
    decades in x and the interesting tail is orders of magnitude below the mode in y.

    Args:
        axis: the axes to draw on
        perplexity_dict (dict): label -> that generation's per-sample perplexities
        y_limit_lower (float): lower y-limit, and the visibility threshold of the x-range
        num_bins (int): histogram bins, spaced logarithmically over the visible range

    Returns:
        tuple: (lower_limit, upper_limit, num_clipped, num_total) of the chosen x-range
    """
    lower_limit, upper_limit, num_clipped, num_total = visible_perplexity_range(
        perplexity_dict, y_limit_lower, num_bins
    )
    bins = torch.logspace(
        torch.log10(torch.tensor(lower_limit)),
        torch.log10(torch.tensor(upper_limit)),
        steps=num_bins,
    )

    for index, (label, perplexities) in enumerate(perplexity_dict.items()):
        sns.histplot(
            perplexities,
            bins=bins,
            stat="density",
            label=label,
            element="step",
            alpha=0.4,
            color=GENERATION_COLORS[index % len(GENERATION_COLORS)],
            ax=axis,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower_limit, upper_limit)
    axis.set_ylim(y_limit_lower, 1)
    axis.set_xlabel("Perplexity", fontweight="bold")
    axis.set_ylabel("Probability", fontweight="bold")
    axis.legend(loc="upper right")
    for spine in axis.spines.values():
        spine.set_color("black")

    return lower_limit, upper_limit, num_clipped, num_total


def draw_median_panel(
    axis, series: list, anchor: tuple = None, ylabel: str = "perplexity (median)"
) -> None:
    """The same measurements as one point per generation, log y.

    Log, because perplexity spans decades once a model collapses and a linear axis would show one
    visible curve and one flat line at the bottom.

    Args:
        axis: the axes to draw on
        series (list): the MedianSeries to draw, in legend order
        anchor (tuple): (value, label) of a horizontal reference line, or None
        ylabel (str): y-axis label, which says *what* was scored

    Returns:
        None
    """
    ticks = []
    for curve in series:
        generations = sorted(curve.medians)
        if not generations:
            continue
        ticks = ticks or generations
        axis.plot(
            generations,
            [curve.medians[generation] for generation in generations],
            marker=curve.marker,
            markersize=9,
            linewidth=2,
            linestyle=curve.linestyle,
            color=curve.color,
            label=curve.label,
        )
        banded = [generation for generation in generations if generation in curve.band]
        if banded:
            axis.fill_between(
                banded,
                [curve.band[generation][0] for generation in banded],
                [curve.band[generation][1] for generation in banded],
                color=curve.color,
                alpha=0.15,
                linewidth=0,
            )

    if anchor is not None:
        axis.axhline(
            anchor[0],
            color=ANCHOR_COLOR,
            linestyle="--",
            linewidth=2,
            label=anchor[1],
        )

    axis.set_yscale("log")
    axis.set_xlabel("collapse generation", fontweight="bold")
    axis.set_ylabel(ylabel, fontweight="bold")
    if ticks:
        axis.set_xticks(ticks)
    axis.legend(loc="upper left")
    for spine in axis.spines.values():
        spine.set_color("black")


def save_figure(figure, plot_stem: str, show: bool = False) -> None:
    """Writes one figure as both a PDF and a PNG under `plot_stem`.

    Args:
        figure: the figure to write
        plot_stem (str): output path without the extension. Its directory is created if missing
        show (bool): also open it interactively, which does nothing on a headless backend

    Returns:
        None
    """
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_stem)), exist_ok=True)
    figure.savefig(f"{plot_stem}.pdf")
    figure.savefig(f"{plot_stem}.png", dpi=200)
    if show:
        plt.show()
    plt.close(figure)


def plot_perplexity_figure(
    perplexity_dict: dict,
    plot_stem: str,
    title: str,
    usetex: bool = True,
    primary_label: str = "collapse checkpoints",
    primary_color: str = BASELINE_COLOR,
    extra_series: list = None,
    anchor: tuple = None,
    median_ylabel: str = "perplexity (median)",
    y_limit_lower: float = Y_LIMIT_LOWER,
    font_size: float = 20,
    show: bool = False,
) -> tuple:
    """The repo's perplexity figure: the histograms on top, the median per generation below.

    The two panels are two views of the *same* numbers, which is why they belong in one figure: the
    histogram says how the distribution deforms (a growing right tail, a mode that walks), the
    median says how far along that deformation is at each generation. Reading one without the other
    invites the two classic misreadings — a median that moves little while the tail explodes, or a
    tail that looks alarming while the bulk has not moved.

    `extra_series` and `anchor` exist for the callers that have more than one lineage to show on the
    median panel (utils/evaluate_perplexity.py draws its surrogate and the un-fine-tuned base model
    there); the histogram panel always shows `perplexity_dict` alone, because eleven colours already
    carry the generation index and a second lineage would have to reuse them.

    Args:
        perplexity_dict (dict): label -> that generation's per-sample perplexities. The labels are
            the histogram legend and, through their trailing index, the median panel's x positions
        plot_stem (str): output path without the extension
        title (str): figure title. Rendered by LaTeX when `usetex`, so no underscores or backslashes
        usetex (bool): render text with LaTeX
        primary_label (str): legend entry of the curve derived from `perplexity_dict`
        primary_color (str): its colour
        extra_series (list): further MedianSeries for the median panel, or None
        anchor (tuple): (value, label) of a horizontal reference line on the median panel, or None
        median_ylabel (str): y-axis label of the median panel, which says what was scored
        y_limit_lower (float): lower y-limit of the histogram, and its visibility threshold
        font_size (float): base font size
        show (bool): also open the figure interactively

    Returns:
        tuple: (lower_limit, upper_limit, num_clipped, num_total) of the histogram's x-range, so the
            caller can report how much of its data the range left out
    """
    apply_perplexity_style(usetex, font_size)

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(10, 11), height_ratios=[3, 2]
    )

    span = draw_histogram_panel(upper, perplexity_dict, y_limit_lower)

    medians, band = {}, {}
    for index, (label, perplexities) in enumerate(perplexity_dict.items()):
        generation = generation_index(label)
        generation = index if generation < 0 else generation
        median, q25, q75 = quantiles(perplexities)
        medians[generation] = median
        band[generation] = (q25, q75)
    primary = MedianSeries(
        label=primary_label, medians=medians, band=band, color=primary_color, marker="o"
    )
    draw_median_panel(lower, [primary] + list(extra_series or []), anchor, median_ylabel)

    upper.set_title(title, fontweight="bold")
    save_figure(figure, plot_stem, show)

    lower_limit, upper_limit, num_clipped, num_total = span
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Plot range{TColors.ENDC}: "
        f"[{lower_limit:.0e}, {upper_limit:.0e}] (clipping {num_clipped} of {num_total} "
        f"perplexities, {100 * num_clipped / max(num_total, 1):.2f}%, which are all below a "
        f"density of {y_limit_lower:.0e})"
    )
    return span
