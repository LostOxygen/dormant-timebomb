"""helper library"""
import functools
import gc
import inspect
import json
import os
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


INIT_CHARS = [
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "w",
    "x",
    "y",
    "z",
]


def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(
    function: callable = None, starting_batch_size: int = 128
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(
            find_executable_batch_size, starting_batch_size=starting_batch_size
        )

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join(
                [f"{arg}={value}" for arg, value in zip(params[1:], args[1:])]
            )
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator


def report_block_size(block_size: int, token_counts: list) -> int:
    """Reports the requested block size against the dataset and returns it unchanged.

    `block_size` is the training sequence length, the `max_new_tokens` cap of the generation and
    part of every on-disk artifact name, so run_baseline.py and run_extrapolation.py have to
    arrive at the same value from the same inputs or their artifacts stop lining up. That is why
    this lives here instead of being duplicated in both.

    The requested value is authoritative. It used to be silently raised to the longest tokenized
    response of the dataset, which meant any --block_size below that longest response had no
    effect whatsoever and every artifact was named after a number nobody passed.

    Responses longer than the returned value are *not* discarded: utils/train_generation.py packs
    with trl's "wrapped" strategy, which concatenates and chunks, so a long response spans two
    blocks instead of losing its tail. What it does cost is that such a response is split across a
    block boundary during training, and that generation is capped at this many new tokens — so the
    share of the dataset above the block size is worth knowing and is reported here.

    Args:
        block_size (int): the requested block size
        token_counts (list): tokenized length of every response in the dataset

    Returns:
        int: block_size, unchanged
    """
    longest = max(token_counts)
    affected = sum(1 for count in token_counts if count > block_size)
    print(
        f"## Block size: {block_size} (longest response {longest} tokens, mean "
        f"{sum(token_counts) / len(token_counts):.0f})"
    )
    if affected:
        print(
            f"## Warning: {affected} of {len(token_counts)} responses "
            f"({100 * affected / len(token_counts):.1f}%) are longer than the block size, so they "
            f"are split across two packed blocks during training and their generated counterparts "
            f"are capped at {block_size} new tokens. Raise --block_size to {longest} to fit every "
            f"response in one block, at the cost of a proportionally slower generation"
        )
    return block_size


def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.

    Borrowed from https://github.com/EleutherAI/lm-evaluation-harness/blob/5c006ed417a2f4d01248d487bcbd493ebe3e5edd/lm_eval/models/utils.py#L624
    """
    if tokenizer.pad_token:
        return tokenizer

    if tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def clear_inherited_max_length(model):
    """Drops the `max_length` a checkpoint ships in its generation_config.

    Qwen2.5's generation_config.json carries `max_length: 32768`, mirroring
    `max_position_embeddings`. Every `generate()` call in this pipeline passes `max_new_tokens`
    explicitly, and transformers then *always* recomputes
    `max_length = max_new_tokens + prompt_length` — but it also logs

        Both `max_new_tokens` (=512) and `max_length`(=32768) seem to have been set.
        `max_new_tokens` will take precedence.

    once per call, because `has_default_max_length` is false whenever the checkpoint set the
    value (transformers/generation/utils.py::_prepare_generated_length). Over a generation shard
    that is one line of noise per batch, which buries the warnings that do matter.

    Clearing it is behaviour preserving — verified by decoding the same prompt greedily before and
    after and comparing token ids — and it is the same principle as pinning the sampling
    parameters instead of inheriting them: the length is decided by `--block_size` here, not by
    whatever the checkpoint was shipped with.

    This is only safe while every `generate()` call passes `max_new_tokens`. If one ever does not,
    transformers takes the `elif has_default_max_length` branch and adds the prompt length to
    `max_length`, which would be `None + int`. Pass `max_new_tokens` or leave this alone.

    Args:
        model: a Hugging Face model whose generation_config is mutated in place

    Returns:
        The same model, so this can wrap a load call.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.max_length = None
    return model


def stamp_transformers_version(directory: str) -> None:
    """Records the transformers version in a checkpoint's config.json.

    Unsloth's ``save_pretrained``/``save_pretrained_merged`` write ``unsloth_version`` but no
    ``transformers_version``, and transformers 5.x reads exactly that key to decide whether a
    checkpoint predates the Mistral pre-tokenizer bug
    (``tokenization_utils_tokenizers._patch_mistral_regex``). With the key missing it cannot rule
    the bug out, so every load of one of this pipeline's checkpoints prints a warning claiming the
    tokenizer has "an incorrect regex pattern" and advising ``fix_mistral_regex=True``.

    The warning is a false positive for a Qwen checkpoint — the saved tokenizer is byte-identical
    to the upstream one — and the advice is worse than the warning: passing that flag installs
    *Mistral's* split regex over Qwen's, and in transformers 5.5 it raises an AttributeError before
    it even gets that far. Writing the key transformers itself would have written is the fix.

    Silent when the directory has no config.json (a bare LoRA adapter directory carries
    adapter_config.json instead) or when the key is already there.

    Args:
        directory (str): a checkpoint directory that was just saved

    Returns:
        None
    """
    config_file = os.path.join(directory, "config.json")
    if not os.path.isfile(config_file):
        return
    with open(config_file, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("transformers_version"):
        return
    import transformers  # local: utils.utils is imported by workers that patch it first

    config["transformers_version"] = transformers.__version__
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
