"""Artifact-name suffixes for the --real_data_fraction data mixture.

Filenames are this pipeline's only interface between stages, generations and shards (there is no
manifest), so a knob that changes what an artifact *contains* has to change what it is *called* —
otherwise two runs silently overwrite each other's checkpoints and the second one's numbers are
attributed to the first one's configuration.

Kept free of torch/transformers/unsloth imports on purpose, same rule as utils/colors.py and
utils/devices.py: the workers that need these names (utils/train_generation.py,
utils/generate_dataset.py) must import unsloth *before* torch for its patches to land, and a naming
helper that dragged torch in would beat unsloth to it silently.
"""


def mixture_tag(real_data_fraction: float) -> str:
    """Run-level tag for a --real_data_fraction run.

    Empty at 0, which is what keeps this change backwards compatible: a pure self-training run
    keeps writing exactly the artifact names it always had, so existing runs on disk stay readable
    and -ho/-st/-cfg keep finding them.

    Args:
        real_data_fraction (float): the --real_data_fraction the run was given

    Returns:
        str: "" at 0, otherwise "_rdf{value}"
    """
    return "" if real_data_fraction <= 0 else f"_rdf{real_data_fraction:g}"


def mixture_suffix(real_data_fraction: float, generation: int) -> str:
    """Per-generation artifact suffix for a --real_data_fraction run.

    Empty for generation 0, and that is not an inconsistency to tidy away. Generation 0 trains on
    the human corpus by definition — run_baseline.py only mixes from generation 1 onward — so
    model_0, the corpus model_0 generates, and every human-corpus dataset are byte-identical for
    every value of the fraction. Leaving them unsuffixed means one --path holds a single shared
    generation 0 that every mixture reuses instead of retraining an identical copy per value, and
    it means the stages that only ever read generation 0 (run_extrapolation.py,
    utils/calibrate_surrogate.py, and run_attack.py's surrogate anchor) need to know nothing about
    the mixture at all.

    Callers therefore pass the generation of the artifact being named, which for a
    previous-generation input is `generation - 1` — see utils/train_generation.py, which reads
    model_{g-1} and writes model_{g} and needs a different answer for each.

    Args:
        real_data_fraction (float): the --real_data_fraction the run was given
        generation (int): generation index of the artifact being named

    Returns:
        str: "" for generation 0 or fraction 0, otherwise "_rdf{value}"
    """
    return "" if generation <= 0 else mixture_tag(real_data_fraction)
