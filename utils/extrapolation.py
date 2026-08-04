"""shared definitions of the extrapolation methods used by run_extrapolation.py"""

import json
import os
import shutil

import torch
from torch import Tensor

# The three ways of approximating a later collapse generation without ever training it. All of
# them are indexed by the same factor n = generation + 1, so that generation 0 reproduces the
# real model_0 anchor under every method and only the generations above it are approximations.
#
#   logit  base + n * (collapsed - base) in logit space, applied at every decoding step. A
#          first order step in the output distribution, i.e. p_0^(1-n) * p_1^n
#   lora   the same first order step taken in weight space instead: the collapse LoRA adapter
#          with its alpha scaled by n. Cheaper (one model instead of two), it produces an
#          actual model artifact, and the network's own nonlinearities keep it better behaved
#          than a per-token logit tilt
#   data   the base model sampled with a support that is truncated once per generation. This
#          mimics the repeated resampling that *drives* collapse rather than extrapolating the
#          drift that collapse *causes*, so it is a surrogate of the mechanism, not of the model
METHODS: tuple[str, ...] = ("logit", "lora", "data")

# Artifact suffix per method. "logit" keeps the historical "_ex" so runs made before the other
# methods existed stay readable, the other two get their own namespace so that all three can
# coexist on disk and be compared against each other
METHOD_SUFFIXES: dict[str, str] = {
    "logit": "_ex",
    "lora": "_ex_lora",
    "data": "_ex_data",
}

# Human readable method names for the plot titles
METHOD_LABELS: dict[str, str] = {
    "logit": "logit extrapolation",
    "lora": "LoRA adapter extrapolation",
    "data": "data-space surrogate",
}

# file name of the calibration result of the data-space surrogate
CALIBRATION_FILE: str = "surrogate_top_p_bs{block_size}_{specifier_name}.json"


def extrapolate_logits(
    base_scores: Tensor, collapsed_logits: Tensor, factor: float
) -> Tensor:
    """
    The logit-space extrapolation ``base + n * (collapsed - base)``, in float32.

    This is the single definition of the tilt: ``generate_dataset_extrapolation.py`` uses it to
    produce the synthetic datasets and ``run_attack.py`` uses it to build the differentiable
    surrogate that the adversarial suffix is optimized against. If the two drifted apart, the
    attack would be optimized against a different model than the one the datasets characterize.

    In distribution space the result is the exponential tilt ``p_base^(1-n) * p_collapsed^n``,
    i.e. the same algebra as classifier-free guidance with weight `n`.

    No clamping is applied. Softmax is shift invariant and torch computes it by subtracting the
    row maximum first, so a large logit cannot overflow ``exp()``. Clamping to a fixed range
    instead ties every token above the ceiling at the ceiling and flattens everything below the
    floor onto the floor, which for a large `n` is most of the vocabulary — it destroys exactly
    the ranking the extrapolation produces.

    Args:
        base_scores (Tensor): logits of the pristine base model. Entries of -inf mark tokens an
            earlier logits processor already forbade; they stay forbidden and are kept out of
            the arithmetic, since -inf + n * (finite + inf) evaluates to NaN
        collapsed_logits (Tensor): logits of the generation-0 collapsed model, same shape
        factor (float): the factor n, i.e. generation + 1

    Returns:
        Tensor: the extrapolated logits in float32, with forbidden tokens and any NaN out of
            either model set to -inf so that neither can ever be sampled
    """
    base_f32 = base_scores.to(torch.float32)
    collapsed_f32 = collapsed_logits.to(torch.float32)

    forbidden = torch.isneginf(base_f32)
    safe_base = base_f32.masked_fill(forbidden, 0.0)

    extrapolated = safe_base + factor * (collapsed_f32 - safe_base)

    # a NaN must not be mapped onto a mid-distribution logit, where it would be sampleable
    return extrapolated.masked_fill(
        forbidden | torch.isnan(extrapolated), float("-inf")
    )


def dataset_suffix(method: str) -> str:
    """
    Returns the artifact suffix of an extrapolation method.

    Args:
        method (str): one of METHODS

    Returns:
        str: the suffix that all datasets, perplexity dicts and plots of that method carry
    """
    if method not in METHOD_SUFFIXES:
        raise ValueError(
            f"unknown extrapolation method {method!r}, expected one of {list(METHODS)}"
        )
    return METHOD_SUFFIXES[method]


def calibration_file(dataset_path: str, block_size: int, specifier_name: str) -> str:
    """
    Returns the path of the data-space surrogate's calibration result.

    Args:
        dataset_path (str): the generated_datasets/ directory
        block_size (int): block size the calibration was run with
        specifier_name (str): short model name the calibration was run with

    Returns:
        str: path of the calibration json
    """
    return os.path.join(
        dataset_path,
        CALIBRATION_FILE.format(block_size=block_size, specifier_name=specifier_name),
    )


def build_scaled_adapter(adapter_path: str, factor: float, output_path: str) -> str:
    """
    Copies a LoRA adapter and scales its alpha, which scales the delta it adds to the base model.

    A LoRA layer adds ``(alpha / r) * B @ A`` on top of the frozen base weight, so the entire
    fine-tuning delta is proportional to alpha. ``model_0`` is a single fine-tuning step away
    from the base model, which makes ``alpha * n`` exactly the weight space counterpart of the
    logit space step ``base + n * (collapsed - base)``: the resulting weights are
    ``W_base + n * (W_collapsed - W_base)``.

    The alpha is patched in the copied ``adapter_config.json`` rather than on the loaded
    modules, so it does not matter whether the loader reads the scaling from the config, caches
    it on the layer, or fuses it into a custom kernel — which unsloth does. Scaling is
    proportional to alpha for plain LoRA (``alpha / r``) as well as for rsLoRA
    (``alpha / sqrt(r)``), so the factor carries through either way.

    Args:
        adapter_path (str): directory of the trained collapse adapter, i.e. model_0
        factor (float): the factor n to multiply the adapter's contribution by
        output_path (str): directory to write the scaled copy to

    Returns:
        str: output_path, so the call can be inlined into a model load
    """
    config_name = "adapter_config.json"
    source_config = os.path.join(adapter_path, config_name)
    if not os.path.isfile(source_config):
        raise FileNotFoundError(
            f"{adapter_path} contains no {config_name}, so it is not a LoRA adapter. The "
            "'lora' method needs the adapter that run_baseline.py writes with "
            "save_adapter=True, not the merged _fp16 model"
        )

    # a leftover copy from an earlier run would otherwise be reused with the wrong alpha
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(adapter_path, output_path)

    target_config = os.path.join(output_path, config_name)
    with open(target_config, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config.get("lora_alpha") is None:
        raise KeyError(f"{source_config} has no lora_alpha to scale")

    config["lora_alpha"] = config["lora_alpha"] * factor

    with open(target_config, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    return output_path


def surrogate_top_p(top_p_first_generation: float, generation_n: int) -> float:
    """
    Top-p of the data-space surrogate for the extrapolation factor n.

    Collapse is driven by every generation resampling from its own truncated output, so the
    surrogate applies one truncation per generation: if a single step keeps the top ``p_1`` of
    the probability mass, then `n` stacked steps keep ``p_1 ** n`` of it. That compounds the way
    the real resampling does and saturates towards a point mass as `n` grows, instead of running
    off to infinity the way a schedule that is linear in `n` would. Once ``p_1 ** n`` drops
    below the probability of the single most likely token, the sampling is effectively greedy —
    which is the degenerate fixed point the real process converges to.

    Args:
        top_p_first_generation (float): p_1, the top-p that reproduces the real model_0. This is
            what calibrate_surrogate.py fits; it is not a free hyperparameter
        generation_n (int): the extrapolation factor n, i.e. generation + 1

    Returns:
        float: the top-p to sample generation n with
    """
    if not 0.0 < top_p_first_generation <= 1.0:
        raise ValueError(
            f"p_1 has to be in (0, 1], but is {top_p_first_generation}"
        )
    if generation_n < 1:
        raise ValueError(f"the extrapolation factor has to be >= 1, but is {generation_n}")
    return float(top_p_first_generation**generation_n)
