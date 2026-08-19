"""sharding the adversarial search over the available GPUs, one subprocess per device

The GCG search in run_attack.py and run_selective_attack.py is a loop over (task, restart) pairs,
and those pairs are independent: each one owns its suffix, its segments and its outcome, and the
only state they share is the models, which are read-only for the whole search (every parameter is
``requires_grad_(False)``, only the one-hot input matrix carries a gradient). So the pairs can run
side by side on separate GPUs, which is what this module organizes.

**(task, restart) and not just task.** With five tasks and four devices, task-level sharding splits
2/1/1/1 and finishes no faster than the two-task shard — about 2.5x. The same work as fifteen
(task, restart) units splits 4/4/4/3, which is close to 3.75x, and it keeps scaling when only one
task is selected. The cost is that ``--stop_on_success`` can no longer skip a task's *later*
restarts once an earlier one hits, because they are already running; it still stops the restart that
found the hit.

**Same numbers, sharded or not.** ``utils.gcg.sample_ids_from_grad`` draws its candidates from the
global torch RNG, so before this existed the trajectory of the second task depended on how many
draws the first one had consumed — the sequence was a function of the loop order. Every unit now
seeds that stream itself from ``unit_seed(seed, task, restart)``, which makes a unit's trajectory a
property of the unit alone: a sharded run and a single-process run produce identical suffixes, and
so do two runs that select different subsets of the tasks. This did change the numbers once, when it
was introduced; results from before that are not comparable candidate-for-candidate.

**What has to happen before the fan-out**, and why the orchestrators do it themselves:

* the **capability gate** is a run-level decision — ``--min_capability`` is a fraction over all
  probed tasks and the whole point is that it aborts before any optimization — so it runs once, in
  the parent, and its per-task verdicts travel to the workers in the handoff as their controls
  rather than being decoded again per shard
* the ``--surrogate_factor auto`` probe likewise needs the models and decides a run-level value; the
  chosen factors travel in the handoff so no worker repeats the ladder
* a ``lora`` surrogate is a **file**: ``utils.extrapolation.build_scaled_adapter`` writes it with
  ``rmtree`` followed by ``copytree``, so several workers building the same factor would delete each
  other's directory mid-copy. The parent builds it once and passes the path, which is also why the
  handoff carries ``surrogate_paths``

The parent therefore does load models, and frees them again before launching anything. What it
cannot give back is its CUDA context (a few hundred MB on the first device), so the first shard has
slightly less memory to work with than the others. That is the price of not spending a whole extra
model load on a separate probe subprocess.

Kept free of torch, transformers and unsloth imports: it is imported by the attack entry points,
which load models through plain ``transformers``, and by nothing that needs unsloth's patches — but
the same rule as utils/naming.py and utils/devices.py applies, and a helper that dragged torch in
would be one more thing to think about.
"""

import json
import os
import subprocess
import sys
import zlib

# a large prime, so the restart index and the task hash cannot cancel each other out for small
# seeds. The modulus is torch's own accepted range for manual_seed on all backends
SEED_STRIDE: int = 1_000_003
SEED_MODULUS: int = 2**31 - 1


def unit_seed(seed: int, task: str, restart: int) -> int:
    """The RNG seed of one (task, restart) unit, a pure function of its identity.

    Derived rather than inherited so that a unit's trajectory does not depend on which other units
    ran before it in the same process — see the module docstring. ``zlib.crc32`` and not ``hash()``:
    Python salts string hashes per process, so ``hash()`` would give a different answer in the
    parent and in every worker.

    Args:
        seed (int): the run's --seed
        task (str): the task name
        restart (int): the restart index

    Returns:
        int: a seed in torch's accepted range
    """
    return (
        seed + SEED_STRIDE * (restart + 1) + zlib.crc32(task.encode("utf-8"))
    ) % SEED_MODULUS


def plan_units(task_names: list, restarts: int) -> list:
    """Every (task, restart) pair, in the order a single-process run would take them.

    Args:
        task_names (list): the attackable task names, in CLI order
        restarts (int): restarts per task

    Returns:
        list: (task name, restart index) tuples
    """
    return [(name, restart) for name in task_names for restart in range(restarts)]


def plan_shards(units: list, num_shards: int) -> list:
    """Deals `units` round robin onto `num_shards` shards.

    Round robin rather than contiguous blocks: consecutive units belong to the same task, and tasks
    differ in how long they take (a task whose baseline keeps failing verifies faster than one that
    runs to the step limit), so interleaving them spreads that variance across the devices instead
    of concentrating one task's cost on one of them.

    Args:
        units (list): the (task, restart) pairs
        num_shards (int): how many shards to produce

    Returns:
        list: one list of units per shard, empty shards dropped
    """
    shards = [[] for _ in range(max(1, num_shards))]
    for index, unit in enumerate(units):
        shards[index % len(shards)].append(unit)
    return [shard for shard in shards if shard]


def encode_units(units: list) -> str:
    """Serializes units for the worker command line, as ``task:restart`` joined by commas."""
    return ",".join(f"{name}:{restart}" for name, restart in units)


def decode_units(value: str) -> list:
    """Parses what `encode_units` produced.

    Args:
        value (str): the ``task:restart,...`` string

    Returns:
        list: (task name, restart index) tuples

    Raises:
        ValueError: a component is not of the form task:restart
    """
    units = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--shard_units takes task:restart pairs, not {part!r}")
        name, restart = part.rsplit(":", 1)
        units.append((name, int(restart)))
    return units


def units_by_task(units: list) -> dict:
    """Groups units into ``{task name: [restart, ...]}``, preserving order."""
    grouped: dict = {}
    for name, restart in units:
        grouped.setdefault(name, []).append(restart)
    return grouped


def shard_file(results_path: str, stem: str, shard_id: int) -> str:
    """Path of one shard's partial results. Transient, removed after the merge."""
    return os.path.join(results_path, f".{stem}_shard{shard_id}.json")


def handoff_file(results_path: str, stem: str) -> str:
    """Path of the parent -> worker handoff. Transient, removed after the merge."""
    return os.path.join(results_path, f".{stem}_handoff.json")


def write_json(path: str, payload: dict) -> None:
    """Writes `payload` as JSON, creating the directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: str) -> dict:
    """Reads a JSON file written by `write_json`."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_shards(
    script: str,
    shards: list,
    devices: list,
    handoff: str,
    results_path: str,
    stem: str,
) -> list:
    """Launches one worker per shard, waits for all of them, and returns their output paths.

    Each worker is the *same script* re-entered in worker mode, pinned to one device through
    ``CUDA_VISIBLE_DEVICES`` — the pattern run_baseline.py uses for its generation and perplexity
    workers. Everything the worker needs to rebuild the run is in the handoff file, so the command
    line carries only which units to run and where to write them; that is what keeps the workers
    provably configured identically to the parent instead of depending on thirty forwarded flags.

    Args:
        script (str): path of the entry point to re-enter, i.e. ``__file__`` of the caller
        shards (list): one list of units per shard, from plan_shards
        devices (list): the device ids to pin to, at least as long as `shards`
        handoff (str): path of the handoff file the workers read
        results_path (str): directory the shard files are written to
        stem (str): run-specific name component of the transient files

    Returns:
        list: one shard output path per shard, in shard order

    Raises:
        RuntimeError: at least one worker exited non-zero
    """
    outputs = [shard_file(results_path, stem, index) for index in range(len(shards))]
    # remove stale shard files so a previous run's results can never be merged into this one
    for path in outputs:
        if os.path.exists(path):
            os.remove(path)

    processes = []
    for index, (shard, output) in enumerate(zip(shards, outputs)):
        device = devices[index % len(devices)]
        processes.append(
            subprocess.Popen(
                [
                    "env",
                    f"CUDA_VISIBLE_DEVICES={device}",
                    sys.executable,
                    script,
                    "--shard_units",
                    encode_units(shard),
                    "--handoff_file",
                    handoff,
                    "--shard_out",
                    output,
                ]
            )
        )

    for process in processes:
        process.wait()

    failed = [index for index, process in enumerate(processes) if process.returncode != 0]
    if failed:
        raise RuntimeError(
            f"the adversarial search failed for shard(s) {failed}. See the subprocess output "
            f"above for the actual error; the partial results of the shards that did finish are "
            f"in {results_path}"
        )
    return outputs


def cleanup(paths: list) -> None:
    """Removes the transient handoff and shard files, ignoring the ones already gone."""
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


def merge_outcomes(partials: list) -> list:
    """Merges per-shard outcome dicts into one entry per task.

    A unit's outcome covers one restart of one task, so the merge is a concatenation of the record
    lists — each record carries its own ``restart`` and ``step``, so the result is re-sortable into
    the order a single-process run would have produced — plus a minimum over the objective. Scalars
    that describe the *task* rather than the restart (the control verdict, the skip reason) come
    from the first shard that has one; every shard that touched the task agrees on them, since they
    come from the shared capability probe.

    Args:
        partials (list): the ``results`` lists of every shard, in shard order

    Returns:
        list: one merged outcome dict per task, in the order the tasks first appear
    """
    merged: dict = {}
    for outcomes in partials:
        for outcome in outcomes:
            name = outcome["task"]
            if name not in merged:
                # a copy, so the shard payloads stay untouched and the lists below are ours to grow
                merged[name] = {
                    key: (list(value) if isinstance(value, list) else value)
                    for key, value in outcome.items()
                }
                continue
            target = merged[name]
            for key, value in outcome.items():
                if isinstance(value, list):
                    target.setdefault(key, []).extend(value)
                elif key == "best_objective":
                    if value is not None and (
                        target.get("best_objective") is None
                        or value < target["best_objective"]
                    ):
                        target["best_objective"] = value
                        target["best_suffix"] = outcome.get("best_suffix")
                elif key == "best_suffix":
                    continue  # carried by the best_objective branch above
                elif target.get(key) in (None, "", {}, []):
                    target[key] = value

    for outcome in merged.values():
        for key, value in outcome.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                outcome[key] = sorted(
                    value, key=lambda row: (row.get("restart", 0), row.get("step", 0))
                )
    return list(merged.values())


def release_weights(models) -> None:
    """Drops the loaded modules out of a set of model wrappers, keeping the wrappers themselves.

    The parent needs the models for the factor probe and the capability gate, and must not still be
    holding them when the first shard starts loading its own onto the same device. What it *does*
    still need afterwards is the wrappers' metadata: the summary and the selectivity and surrogate
    reports read labels, roles and factors off them, plus the merged records from the shards, and
    none of that touches a weight. So the module reference is cleared rather than the wrapper
    deleted, which is also why the reports do not have to be rewritten to take a copy of the
    metadata along.

    ``ExtrapolatedModel`` holds a second wrapper for its generation-0 anchor; that one is cleared
    too, otherwise the tilt's anchor would keep a whole checkpoint alive on its own.

    The caller empties the allocator cache afterwards — this module stays torch free.

    Args:
        models: an iterable of TargetModel-like wrappers, None entries allowed

    Returns:
        None
    """
    for model in models:
        if model is None:
            continue
        anchor = getattr(model, "first", None)
        if anchor is not None:
            anchor.model = None
        model.model = None
