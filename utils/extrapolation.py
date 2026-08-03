"""shared definitions of the extrapolation methods used by run_extrapolation.py"""

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
#   weight the same first order step taken in weight space instead: W_base + n * (W_0 - W_base)
#          over the full parameter set. Cheaper (one model instead of two), it produces an
#          actual model artifact, and the network's own nonlinearities keep it better behaved
#          than a per-token logit tilt
#   data   the base model sampled with a support that is truncated once per generation. This
#          mimics the repeated resampling that *drives* collapse rather than extrapolating the
#          drift that collapse *causes*, so it is a surrogate of the mechanism, not of the model
METHODS: tuple[str, ...] = ("logit", "weight", "data")

# Artifact suffix per method. "logit" keeps the historical "_ex" so runs made before the other
# methods existed stay readable, the other two get their own namespace so that all three can
# coexist on disk and be compared against each other
METHOD_SUFFIXES: dict[str, str] = {
    "logit": "_ex",
    "weight": "_ex_weight",
    "data": "_ex_data",
}

# Human readable method names for the plot titles
METHOD_LABELS: dict[str, str] = {
    "logit": "logit extrapolation",
    "weight": "weight-space extrapolation",
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


def build_extrapolated_weights(
    base_path: str,
    first_collapsed_path: str,
    factor: float,
    output_path: str,
    device: str = "cpu",
) -> str:
    """
    Writes the checkpoint ``W_base + n * (W_first_collapsed - W_base)``, parameter by parameter.

    ``model_0`` is a single fine-tuning step away from the base model, so ``model_g`` sits
    ``g + 1`` steps out and scaling the whole fine-tuning delta by `n` is the weight-space
    counterpart of the logit-space step ``base + n * (collapsed - base)``.

    Since the pipeline full fine-tunes rather than training LoRA adapters, the delta is not a
    low-rank object that could be scaled through a config value — it is the difference of two
    complete parameter sets, and every parameter has to be interpolated explicitly. Buffers are
    left alone: they are not trained, so they are identical in both checkpoints and the
    interpolation would be a no-op at best and integer arithmetic at worst.

    Tied parameters (Qwen2.5 ties ``lm_head`` to ``embed_tokens``) appear under more than one
    name while sharing storage, so already-updated tensors are tracked by data pointer. Without
    that the update would be applied twice to the same weights.

    The arithmetic runs in float32 and is cast back per parameter, so a bf16 checkpoint does not
    lose the small deltas to rounding before they are scaled up.

    Args:
        base_path (str): the pristine base model
        first_collapsed_path (str): the generation-0 collapsed model
        factor (float): the factor n. 1.0 reproduces the generation-0 checkpoint exactly
        output_path (str): directory to write the extrapolated checkpoint to
        device (str): device to do the interpolation on. CPU by default, since this runs once
            per generation and the GPU is needed for the generation itself

    Returns:
        str: output_path, so the call can be inlined into a model load
    """
    # transformers is imported lazily: the scripts that call this must import unsloth first,
    # and a module-level transformers import here would break that ordering
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _load(path: str):
        try:
            return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        except TypeError:
            # transformers < 4.56 spells it `torch_dtype`
            return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)

    collapsed = _load(first_collapsed_path).to(device)
    base = _load(base_path).to(device)
    base_params = dict(base.named_parameters())

    seen: set[int] = set()
    n_interpolated = 0
    with torch.no_grad():
        for name, param in collapsed.named_parameters():
            if param.data_ptr() in seen:
                # tied weight, already updated under its other name
                continue
            if name not in base_params:
                raise KeyError(
                    f"{name} exists in {first_collapsed_path} but not in {base_path}. The two "
                    "checkpoints have to be the same architecture for a weight-space "
                    "extrapolation to mean anything"
                )
            base_param = base_params[name]
            if base_param.shape != param.shape:
                raise ValueError(
                    f"shape mismatch for {name}: {tuple(base_param.shape)} in the base model "
                    f"vs {tuple(param.shape)} in {first_collapsed_path}"
                )
            seen.add(param.data_ptr())
            base_fp32 = base_param.detach().to(torch.float32)
            collapsed_fp32 = param.detach().to(torch.float32)
            param.copy_(
                (base_fp32 + factor * (collapsed_fp32 - base_fp32)).to(param.dtype)
            )
            n_interpolated += 1

    if n_interpolated == 0:
        raise RuntimeError(
            f"no parameter of {first_collapsed_path} was interpolated — the checkpoint appears "
            "to have no named parameters, which would silently produce a copy of it"
        )

    del base, base_params

    # a leftover directory from an earlier run would otherwise be half-overwritten
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    collapsed.save_pretrained(output_path, safe_serialization=True)
    AutoTokenizer.from_pretrained(first_collapsed_path).save_pretrained(output_path)

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
