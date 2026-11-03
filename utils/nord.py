"""The Nord palette, and what every figure in this repo draws with.

One place for colour, so that a role (the baseline, the surrogate, a hit, a status) is the same
colour in every figure and a palette change is one edit. The rules follow the data's job rather
than taste:

* **categorical** (identity: methods, series) — a fixed order of distinct hues, never cycled past
  it, with a marker or line style beside the colour so identity is never colour alone
* **ordinal** (a sequence: generations, the real-data mixture) — one hue family, light to dark
* **sequential** (magnitude: a hit rate in a grid) — one hue, light to dark, as a colormap
* **diverging** (polarity: a margin around zero) — two hues around a neutral midpoint
* **status** (pass / wrong / timeout / broken) — reserved steps, never reused for a series

Text wears the text colours, never a series colour. Kept free of torch, transformers and unsloth
imports like the other helper modules in utils/ (matplotlib is imported lazily, inside the two
colormap builders, so naming-only callers pay nothing).
"""

import os

# ── the sixteen colours, by their Nord names ──
NORD0 = "#2E3440"   # polar night, darkest
NORD1 = "#3B4252"
NORD2 = "#434C5E"
NORD3 = "#4C566A"   # polar night, lightest: muted text
NORD4 = "#D8DEE9"   # snow storm, darkest: grid lines
NORD5 = "#E5E9F0"
NORD6 = "#ECEFF4"   # snow storm, lightest
NORD7 = "#8FBCBB"   # frost: teal
NORD8 = "#88C0D0"   # frost: light blue
NORD9 = "#81A1C1"   # frost: blue
NORD10 = "#5E81AC"  # frost: dark blue
NORD11 = "#BF616A"  # aurora: red
NORD12 = "#D08770"  # aurora: orange
NORD13 = "#EBCB8B"  # aurora: yellow
NORD14 = "#A3BE8C"  # aurora: green
NORD15 = "#B48EAD"  # aurora: purple

POLAR_NIGHT = [NORD0, NORD1, NORD2, NORD3]
SNOW_STORM = [NORD4, NORD5, NORD6]
FROST = [NORD7, NORD8, NORD9, NORD10]
AURORA = [NORD11, NORD12, NORD13, NORD14, NORD15]

# ── chrome: surfaces, text, grid ──
# PLOT_THEME=dark draws the same figures for a dark slide: transparent background, white ink
# where the paper's is dark, the neutral and reference greys flipped so they still read. The
# switch is an environment variable rather than a flag because every plotting entry point
# would otherwise need the flag, and it is read once, here, when the palette is imported.
DARK = os.environ.get("PLOT_THEME", "light").strip().lower() == "dark"

SURFACE = "none" if DARK else "#FFFFFF"
TEXT = "#FFFFFF" if DARK else NORD0
TEXT_MUTED = NORD4 if DARK else NORD3
GRID = NORD3 if DARK else NORD4
NEUTRAL = NORD2 if DARK else NORD5          # an empty / not-applicable cell
NEUTRAL_LIGHT = NORD1 if DARK else NORD6

# ── categorical: identity, in this fixed order. Blue, orange, green, purple, red, teal, yellow,
# light blue — alternating hue families so that neighbours in the order stay apart under the
# common colour-vision deficiencies; the light blue and the yellow sit last because they are the
# lightest and need a marker or label beside them on a white surface ──
CATEGORICAL = [NORD10, NORD12, NORD14, NORD15, NORD11, NORD7, NORD13, NORD8]

# ── roles: the same thing is the same colour in every figure ──
BASELINE = NORD10      # the pristine base model / the direct attack on the real checkpoint
SURROGATE = NORD12     # the logit surrogate, and the adversarial side of a comparison
ANCHOR = NORD4 if DARK else NORD3     # the generation-0 anchor, references, "assumed" lines
CONTROL = NORD4 if DARK else NORD3    # the non-collapsing control lineage (rdf 1.0)
HIGHLIGHT = NORD11     # the one thing a figure is about, when there is one

METHOD_COLORS = {"none": BASELINE, "logit": SURROGATE, "lora": NORD11}
METHOD_MARKERS = {"none": "o", "logit": "s", "lora": "^"}

# a (target, generation) cell of the per-target grids. The two exclusions are opposites and are
# coloured apart on purpose: a broken baseline had no correct answer to flip, a broken target
# measures collapse rather than the attack
OUTCOME_COLORS = {
    "hit": NORD10,
    "no-hit": NORD5,
    "baseline-broken": NORD13,
    "target-broken": NORD3,
    # run_vuln_plots.py's names for the same two exclusions
    "baseline-insecure": NORD13,
    "collapsed-broken": NORD3,
}
# text drawn *on* a cell of that colour
OUTCOME_TEXT = {key: (NORD6 if value in (NORD10, NORD3) else NORD0)   # ink on a coloured cell: theme-free
                for key, value in OUTCOME_COLORS.items()}

# executed-code statuses (HumanEval, the gate): reserved, never a series colour
STATUS_COLORS = {"pass": NORD14, "wrong": NORD12, "timeout": NORD4, "broken": NORD11}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def _hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, round(c * 255))) for c in rgb)


def ramp(stops: list, count: int) -> list:
    """`count` colours interpolated through `stops`, as hex strings. One stop repeats it."""
    if count <= 0:
        return []
    if len(stops) == 1 or count == 1:
        return [stops[0]] * count
    rgb = [_hex_to_rgb(s) for s in stops]
    out = []
    for i in range(count):
        t = i / (count - 1) * (len(rgb) - 1)
        k = min(int(t), len(rgb) - 2)
        f = t - k
        out.append(_rgb_to_hex(tuple(a + (b - a) * f for a, b in zip(rgb[k], rgb[k + 1]))))
    return out


# the ordinal frost ramp: the one hue family, light to dark. Generations and mixtures read off it
ORDINAL_STOPS = [NORD4, NORD8, NORD10, NORD1]


# the perplexity figures' own palette (since 2026-11-03, by request): Tableau's colour-blind six,
# read as an ordinal ramp from blue through grey to orange so the generations keep their reading
# order. Only utils/plotting.py's histogram panel draws from it; every other figure keeps the Nord
# roles above
PERPLEXITY_STOPS = ["#006BA4", "#5F9ED1", "#ABABAB", "#FFBC79", "#FF800E", "#C85200"]


def perplexity_colors(count: int) -> list:
    """One colour per generation for the perplexity histograms, blue to orange with the collapse."""
    return ramp(PERPLEXITY_STOPS, count)


def generation_colors(count: int) -> list:
    """One colour per generation, light to dark with the collapse. Ordinal, one hue family."""
    return ramp(ORDINAL_STOPS, count)


def mixture_colors(fractions: list) -> dict:
    """fraction -> colour for the real-data mixtures, an ordered variable.

    The collapsing lineages take the frost ramp, darkest at 0 (the strongest collapse) and lighter
    as the fraction grows; the non-collapsing control at 1.0 is not a point on that scale and
    wears the reference grey instead, so it reads as "not a collapse lineage" in every figure.
    """
    collapsing = sorted(f for f in fractions if f < 1.0)
    colours = ramp([NORD10, NORD8], max(len(collapsing), 1)) if len(collapsing) > 1 else [NORD10]
    mapping = {f: c for f, c in zip(collapsing, colours)}
    for f in fractions:
        if f >= 1.0:
            mapping[f] = CONTROL
    return mapping


def sequential_cmap():
    """Magnitude: the frost hue, light to dark. Registered once as "nord"."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LinearSegmentedColormap  # pylint: disable=import-outside-toplevel

    if "nord" not in matplotlib.colormaps:
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list("nord", [NORD6, NORD8, NORD10, NORD1])
        )
    return matplotlib.colormaps["nord"]


def diverging_cmap():
    """Polarity: red through the snow midpoint to blue. Registered once as "nord_div"."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    from matplotlib.colors import LinearSegmentedColormap  # pylint: disable=import-outside-toplevel

    if "nord_div" not in matplotlib.colormaps:
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list("nord_div", [NORD11, NORD6, NORD10])
        )
    return matplotlib.colormaps["nord_div"]


def text_on(value: float) -> str:
    """The text colour that reads on a sequential cell of normalized `value` (0 light, 1 dark)."""
    return NORD6 if value >= 0.55 else NORD0


def apply_style(usetex: bool, font_size: float = 20) -> None:
    """The rcParams every figure is drawn with: the palette as the property cycle, dark-ink text
    and spines, a light solid grid, no legend frame.

    Set globally, as the previous per-module styles were, so a figure drawn after another one
    does not inherit a stranger's settings. The fonts are the paper's (Times, bold labels),
    rendered by LaTeX when `usetex`.
    """
    import matplotlib as mpl  # pylint: disable=import-outside-toplevel
    from cycler import cycler  # pylint: disable=import-outside-toplevel

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
            "legend.frameon": False,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.linewidth": 1.2,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.transparent": DARK,
            "legend.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 1.0,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "pdf.compression": 9,
        }
    )
