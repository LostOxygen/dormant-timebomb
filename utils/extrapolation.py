"""shared definitions of the extrapolation methods used by run_extrapolation.py"""

import json
import math
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
# the logit surrogate's counterpart: which factor n actually reproduces generation g, fitted on
# perplexity by utils/evaluate_perplexity.py --calibrate. Carries the mixture tag because the
# *targets* are that mixture's checkpoints, even though the surrogate itself does not depend on it
FACTOR_CALIBRATION_FILE: str = "surrogate_factor_bs{block_size}_{specifier_name}{tag}.json"


def logit_entropy(logits: Tensor) -> Tensor:
    """Shannon entropy, in nats, of ``softmax(logits)`` along the last dimension.

    Forbidden tokens carry -inf, whose probability is 0 and whose ``p log p`` is ``0 * -inf``;
    that is 0 in the limit and NaN in floating point, so the product is taken only where the
    probability is non-zero.

    Args:
        logits (Tensor): (..., vocabulary)

    Returns:
        Tensor: (...) entropies in nats
    """
    log_p = torch.log_softmax(logits.to(torch.float32), dim=-1)
    p = log_p.exp()
    return torch.special.entr(p).sum(dim=-1)


def entropy_matched_scale(
    logits: Tensor, target_entropy: Tensor, iterations: int = 12
) -> Tensor:
    """The temperature ``tau`` for which ``softmax(logits / tau)`` has `target_entropy`.

    **Why a second scale is needed at all.** The extrapolation factor scales the *difference*
    between the two anchors, which sets the direction of travel, but it cannot set sharpness
    independently: as the factor grows the tilt tends to a point mass on the argmax of the
    likelihood ratio, so its entropy falls toward zero. Measured on this repo's 0.5B lineage the
    real collapsed models go the other way, from 0.26 nats at the base model to 0.42 by
    generation 9 — they *flatten*. A tilt large enough to model a late generation's direction is
    therefore far too sharp to model its distribution, which is the whole of the capability
    ceiling that makes the surrogate stop writing valid code. Dividing by a temperature restores
    the missing degree of freedom: the factor keeps the direction, this keeps the sharpness.

    **The solve is a bisection, deliberately.** Entropy is strictly decreasing in the inverse
    temperature, so the root is bracketed and bisection cannot miss it. Newton is the obvious
    alternative and has a closed-form derivative here, ``dH/dbeta = -beta * Var_p(l)``, but that
    variance collapses toward zero exactly where the tilt concentrates, which makes the step
    diverge: tried on random logits it returned 0.015 for a true temperature of 3.0 on one row
    out of four, and drove the entropy of a genuinely flat target to 0. Bisection over the log
    inverse temperature costs a fixed dozen softmaxes and cannot do that.

    The result is meant to be used **detached**: gradients should flow through the division, not
    through the solve, exactly as they do through a normalization constant.

    Args:
        logits (Tensor): (..., vocabulary), the tilted logits to be rescaled
        target_entropy (Tensor): (...), the entropy each row should end up with, in nats
        iterations (int): bisection steps over ``log beta`` in ``[-3 ln 10, 3 ln 10]``; 12 leaves
            the bracket at 0.0017 in ``log beta``, i.e. the temperature within 0.2%, which is far
            below anything the search can resolve. Each step is one softmax over the vocabulary
            for every row, and on the loss path the rows are every candidate at every target
            position, so the count is the cost: 24 halvings doubled the candidate-scoring time
            of the whole objective, 12 adds about a third

    Returns:
        Tensor: (...) temperatures, strictly positive. A row whose target lies outside the
            reachable range converges to the closest endpoint rather than failing
    """
    flat = logits.to(torch.float32)
    span = 3.0 * math.log(10.0)
    low = torch.full_like(target_entropy, -span)
    high = torch.full_like(target_entropy, span)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        # entr(p) = -p log p with entr(0) = 0, so forbidden (-inf) tokens need no guard
        entropy = torch.special.entr(torch.softmax(flat * mid.exp().unsqueeze(-1), dim=-1)).sum(
            dim=-1
        )
        # entropy falls as beta rises, so a row that is still too flat needs a larger beta
        too_flat = entropy > target_entropy
        low = torch.where(too_flat, mid, low)
        high = torch.where(too_flat, high, mid)
    return torch.exp(-0.5 * (low + high))


def extrapolate_logits(
    base_scores: Tensor,
    collapsed_logits: Tensor,
    factor: float,
    match_entropy: bool = False,
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
        match_entropy (bool): rescale the tilt so its entropy equals the generation-0 model's at
            the same position, which keeps the factor setting direction only and stops the
            surrogate sharpening into a point mass as the factor grows. The target is the
            *anchor's* entropy rather than the attacked generation's, because the anchor is
            observable to the attacker and the real generation is not; it is a floor rather than
            a match, since the real trajectory keeps flattening past the anchor

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
    extrapolated = extrapolated.masked_fill(
        forbidden | torch.isnan(extrapolated), float("-inf")
    )
    if not match_entropy:
        return extrapolated

    # the anchor's own entropy at this position is the target, and it costs no extra forward pass
    target = logit_entropy(collapsed_f32.masked_fill(forbidden, float("-inf")))
    tau = entropy_matched_scale(extrapolated.detach(), target.detach())
    return extrapolated / tau.unsqueeze(-1).clamp(min=1e-6)


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


def factor_calibration_file(
    dataset_path: str, block_size: int, specifier_name: str, tag: str = ""
) -> str:
    """Path of the fitted extrapolation factors, the logit surrogate's calibration.

    Args:
        dataset_path (str): the generated_datasets/ directory
        block_size (int): block size the calibration was run with
        specifier_name (str): short model name the calibration was run with
        tag (str): the mixture tag of the collapse run the factors were fitted against

    Returns:
        str: path of the calibration json
    """
    return os.path.join(
        dataset_path,
        FACTOR_CALIBRATION_FILE.format(
            block_size=block_size, specifier_name=specifier_name, tag=tag
        ),
    )


def calibrated_factor(calibration: dict, generation: int) -> float:
    """The extrapolation factor for one generation, from a calibration file's contents.

    A measured factor is used when the calibration covers that generation, and the fitted law
    ``n = 1 + scale * ln(1 + g)`` otherwise — which is the point of fitting a law at all: the
    generation an attacker wants to reach is the one whose checkpoint nobody has, so its factor has
    to be predicted from the generations that could be measured.

    Args:
        calibration (dict): parsed contents of factor_calibration_file
        generation (int): the collapse generation the surrogate should stand in for

    Returns:
        float: the factor n to build the surrogate with

    Raises:
        KeyError: the calibration carries neither a factor for this generation nor a fitted scale
    """
    measured = {int(k): v for k, v in calibration.get("factors", {}).items()}
    if generation in measured:
        return float(measured[generation])
    scale = calibration.get("scale")
    if scale is None:
        raise KeyError(
            f"the calibration has no factor for generation {generation} and no fitted scale"
        )
    return 1.0 + float(scale) * math.log1p(generation)


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
