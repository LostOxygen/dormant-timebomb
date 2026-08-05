"""GPU discovery that survives unsloth's import time environment rewrite

This module must stay free of torch, transformers and unsloth imports. The orchestrators import it
*above* their unsloth import, and unsloth has to be the first thing in the process to touch torch
for its patches to land.
"""

import os
import subprocess


def visible_devices() -> list:
    """
    Returns the CUDA device ids the process was actually launched with.

    The orchestrators shard the generation and perplexity stages by handing each worker subprocess
    its own explicit ``CUDA_VISIBLE_DEVICES``, so they need the full list — and reading the
    environment at the point of use is not reliable. Older unsloth releases (2024.x, before
    multi-GPU support) rewrote ``CUDA_VISIBLE_DEVICES`` down to a single device at *import* time
    (``0,1,2,3`` -> ``"0"`` with a warning, unset -> ``"0"`` silently), which would collapse the
    whole fan out to one shard on device 0. Current releases do not, but resolving the list once
    through here makes the pipeline independent of which release is installed, and of anything else
    an imported library does to the variable.

    ``torch.cuda.device_count()`` is deliberately not used: it reports what the *current* value of
    the variable allows, so it inherits the same problem, and calling it would mean importing torch.

    Call this once at module scope and keep the result.

    Returns:
        list: the device ids to shard across, in the order they were given
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(device) for device in env.split(",") if device.strip() != ""]

    # nothing was pinned, so ask the driver directly. nvidia-smi rather than torch: importing
    # torch here would beat unsloth to it, and after unsloth's rewrite device_count() would report
    # 1 whatever the machine has
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        count = len([line for line in listing.splitlines() if line.strip()])
        return list(range(count)) if count else [0]
    except (OSError, subprocess.SubprocessError):
        # no driver, no nvidia-smi, or a non-CUDA machine
        return [0]
