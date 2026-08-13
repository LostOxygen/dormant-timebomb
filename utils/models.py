"""The Qwen2.5-Coder size ladder behind ``--model_size``.

Every stage of this pipeline takes a ``--model_specifier`` (a Hugging Face repo id) already, so
running the collapse at another parameter count was always possible — it just meant typing the
full id into all of stages 1-3 and getting it byte-identical every time, because the trailing
component of that id is ``specifier_name``, which is part of every artifact path. A single
mistyped id does not fail loudly: it trains a second, differently-named lineage next to the first
one and the later stages then raise a FileNotFoundError for a checkpoint nobody meant to look for.

``--model_size 7b`` expands to the same id in every stage from one table, which is what makes the
stages agree by construction. The flag is a convenience over ``--model_specifier``, not a
replacement: any other repo id (a non-Qwen model, a local path, a different Qwen revision) still
goes through ``--model_specifier``, and the two are mutually exclusive rather than silently
ranked — see resolve_model_specifier.

Kept free of torch/transformers/unsloth imports, same rule as utils/colors.py, utils/devices.py
and utils/naming.py: utils/calibrate_surrogate.py imports unsloth before torch for its patches to
land, and a helper that dragged torch in would beat unsloth to it silently.
"""

import argparse
from typing import Final

# the unsloth mirrors rather than the Qwen originals, because stage 1 loads them through
# FastLanguageModel, which is what the rest of the pipeline is pinned against. All six are the
# -Instruct variants: the pipeline chat-templates its corpus and the attack tasks are chat prompts,
# so a base (non-instruct) checkpoint would be a different experiment, not a bigger one
MODEL_SIZES: Final[dict[str, str]] = {
    "0.5b": "unsloth/Qwen2.5-Coder-0.5B-Instruct",
    "1.5b": "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "3b": "unsloth/Qwen2.5-Coder-3B-Instruct",
    "7b": "unsloth/Qwen2.5-Coder-7B-Instruct",
    "14b": "unsloth/Qwen2.5-Coder-14B-Instruct",
    "32b": "unsloth/Qwen2.5-Coder-32B-Instruct",
}

DEFAULT_MODEL_SIZE: Final[str] = "0.5b"
DEFAULT_MODEL_SPECIFIER: Final[str] = MODEL_SIZES[DEFAULT_MODEL_SIZE]


def resolve_model_specifier(
    model_size: str = "",
    model_specifier: str = "",
    default: str = DEFAULT_MODEL_SPECIFIER,
) -> str:
    """Turns the (--model_size, --model_specifier) pair into the one repo id a run uses.

    The two flags are mutually exclusive rather than ranked. Ranking them would mean a run that
    passes both gets one of them silently ignored, and since the ignored one may be the larger
    model that is a wasted collapse run — every generation of it — discovered only when the
    checkpoint names in stage 3 turn out to name a different model than intended. Passing both is
    only accepted when they agree, which is what lets a script forward both without special-casing.

    Args:
        model_size (str): a key of MODEL_SIZES, case-insensitive ("7B" and "7b" both work), or ""
            when the run did not ask for one
        model_specifier (str): an explicit Hugging Face repo id (or local path), or "" for none
        default (str): what to fall back to when neither was given — the caller's own
            MODEL_SPECIFIER global, so a script that changed its default keeps it

    Returns:
        str: the repo id every stage of this run should be given

    Raises:
        SystemExit: on an unknown size, or on both flags being given with conflicting values
    """
    size = model_size.strip().lower()
    specifier = model_specifier.strip()

    if size and size not in MODEL_SIZES:
        raise SystemExit(
            f"--model_size {model_size!r} is not one of {', '.join(MODEL_SIZES)}. Pass "
            f"--model_specifier <hugging face id> for a model outside the Qwen2.5-Coder ladder."
        )

    if size and specifier and specifier != MODEL_SIZES[size]:
        raise SystemExit(
            f"--model_size {size} means {MODEL_SIZES[size]}, but --model_specifier says "
            f"{specifier}. Pass one of the two, not both — the ignored one would silently be "
            f"a different model than the run is named after."
        )

    if size:
        return MODEL_SIZES[size]
    return specifier or default


def add_model_arguments(
    parser: argparse.ArgumentParser, role: str = "the base model"
) -> None:
    """Adds the --model_size / --model_specifier pair to an orchestrator's parser.

    Both default to "" so that resolve_model_specifier can tell "not given" from "given the same
    value as the default", which is what the mutual-exclusion check needs. The fallback is the
    calling module's own MODEL_SPECIFIER global, not a default baked in here.

    Args:
        parser (argparse.ArgumentParser): the parser to add the two flags to
        role (str): how the model is described in the help text, e.g. "the baseline model"

    Returns:
        None
    """
    parser.add_argument(
        "--model_size",
        "-msz",
        # lowercased before the choices are checked, so "7B" is accepted as readily as "7b".
        # argparse only validates choices for values that came off the command line, so the ""
        # default is not rejected by the list it is deliberately not a member of
        type=str.lower,
        default="",
        choices=list(MODEL_SIZES),
        help=f"parameter count of {role}, picked from the Qwen2.5-Coder ladder "
        f"({', '.join(MODEL_SIZES)}). Shorthand for the matching --model_specifier, so that all "
        f"stages of a run resolve the same repo id — and therefore the same artifact names — "
        f"from one token. The larger sizes do not fit a single 48GB card at this pipeline's "
        f"default settings; run_baseline.py has -q4/-gc/-tg and the batch sizes for that "
        f"(default: {DEFAULT_MODEL_SIZE}, via --model_specifier)",
    )
    parser.add_argument(
        "--model_specifier",
        "-ms",
        type=str,
        default="",
        help=f"Hugging Face repo id of {role}, for anything outside the --model_size ladder. Its "
        f"trailing component is part of every artifact path, so it has to be identical across the "
        f"stages of one run. Mutually exclusive with --model_size "
        f"(default: {DEFAULT_MODEL_SPECIFIER})",
    )
