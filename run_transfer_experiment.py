"""main hook to test whether an adversarial suffix transfers between two collapse runs"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import datetime
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta

import psutil

from utils.colors import TColors
from utils.devices import visible_devices
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import mixture_suffix

# The experiment is one-directional and that is the whole point: run A is collapsed, a generation
# of it is probed, a suffix is optimized against it, and only then is run B collapsed and the
# frozen suffix evaluated against it. Nothing that produces the suffix — not the search, not the
# choice of generation — is ever allowed to see run B, otherwise the transfer rate is measured on
# a setup selected with knowledge of the answer. There is no optimization after stage 3.
#
# This orchestrator never touches a model in-process, matching run_baseline.py and
# run_extrapolation.py: every stage that loads weights is a subprocess. Here that matters for a
# second reason beyond allocator hygiene — the collapse runs use the unsloth/vLLM stack while the
# verification uses run_attack's plain transformers stack, and unsloth patches transformers at
# import time. Keeping them in separate interpreters is what lets both be correct
VISIBLE_DEVICES = visible_devices()

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"


def collapsed_checkpoint(
    path: str,
    generation: int,
    block_size: int,
    specifier_name: str,
    real_data_fraction: float = 0.0,
) -> str:
    """Path of the merged fp16 checkpoint run_baseline.py writes for one generation.

    The name is predictable because block_size is exactly the value passed on the CLI — see
    utils.utils.report_block_size, which deliberately does not raise it to the dataset's longest
    response.

    Args:
        path (str): root of the run
        generation (int): collapse generation index
        block_size (int): the --block_size the run was given
        specifier_name (str): trailing component of the model specifier
        real_data_fraction (float): the --real_data_fraction both runs were collapsed with, which
            is part of the name from generation 1 onward

    Returns:
        str: path to the merged fp16 checkpoint directory
    """
    return os.path.join(
        path,
        "model_outputs",
        f"model_{generation}_bs{block_size}_{specifier_name}"
        + mixture_suffix(real_data_fraction, generation)
        + "_fp16",
    )


def attack_results_file(path: str, generation: int, specifier_name: str) -> str:
    """Path of the result file run_attack.py writes in plain (non-surrogate) mode."""
    return os.path.join(
        path, "attack_results", f"attack_gen{generation}_{specifier_name}.json"
    )


def print_stage(index: int, total: int, title: str, detail: str = "") -> None:
    """Prints a stage banner. Mirrors the banner style of the other orchestrators."""
    width = shutil.get_terminal_size().columns
    print("\n" + "═" * width)
    print(
        f"## {TColors.HEADER}{TColors.BOLD}Stage {index}/{total}: {title}{TColors.ENDC}"
        + (f" — {detail}" if detail else "")
    )
    print("═" * width)


def run_stage(command: list, what: str) -> None:
    """Runs one subprocess stage and turns a non-zero exit into an explicit failure.

    Output is inherited rather than captured: these stages run for hours and watching them is
    the point.

    Args:
        command (list): argv of the subprocess
        what (str): human readable name of the stage, used in the error

    Raises:
        RuntimeError: the subprocess exited non-zero
    """
    print(f"## {TColors.OKBLUE}$ {' '.join(command)}{TColors.ENDC}\n")
    started = time.time()
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{what} failed with exit code {result.returncode}. See the subprocess output above "
            f"for the actual error"
        )
    print(
        f"\n## {TColors.OKGREEN}{what} finished in "
        f"{timedelta(seconds=int(time.time() - started))}{TColors.ENDC}"
    )


def baseline_command(
    path: str,
    seed: int,
    num_generations: int,
    block_size: int,
    dataset_size: int,
    engine: str,
    with_eval: bool,
    extra: str,
    real_data_fraction: float = 0.0,
) -> list:
    """Builds the run_baseline.py argv for one collapse run.

    The fraction is passed explicitly rather than left to --baseline_extra so that both runs
    provably get the same mixture: the seed is meant to be the only difference between them, and it
    also has to match what this script then uses to *name* their checkpoints.
    """
    command = [
        sys.executable,
        "run_baseline.py",
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--real_data_fraction",
        str(real_data_fraction),
        "--num_generations",
        str(num_generations),
        "--block_size",
        str(block_size),
        "--dataset_size",
        str(dataset_size),
        "--engine",
        engine,
        "--model_specifier",
        MODEL_SPECIFIER,
        "--path",
        path,
    ]
    if not with_eval:
        # -heo skips the perplexity evaluation and the histogram; the training and generation loop
        # above it still runs in full. Two reasons to skip it by default: this experiment reads
        # none of those numbers, and the plotting step needs a LaTeX install
        # (mpl.rcParams["text.usetex"] = True), which would abort the run *after* the models were
        # already trained
        command.append("--human_eval_only")
    if extra:
        command.extend(extra.split())
    return command


def attack_command(
    path: str,
    generation: int,
    block_size: int,
    tasks: str,
    restarts: int,
    num_steps: int,
    verify_every: int,
    max_new_tokens: int,
    repetition_penalty: float,
    exec_timeout: float,
    stop_on_success: bool,
    seed: int,
    extra: str,
    real_data_fraction: float = 0.0,
) -> list:
    """Builds the run_attack.py argv for the search against run A.

    --surrogate_method is left at its default of 'none': the suffix has to be optimized against
    the *real* collapsed checkpoint of run A, because the question here is whether it survives a
    change of collapse run, not whether a surrogate predicted it. Transfer mode would confound
    the two.
    """
    command = [
        sys.executable,
        "run_attack.py",
        "--device",
        "cuda",
        "--collapsed_generation",
        str(generation),
        "--block_size",
        str(block_size),
        "--model_specifier",
        MODEL_SPECIFIER,
        "--path",
        path,
        "--restarts",
        str(restarts),
        "--num_steps",
        str(num_steps),
        "--verify_every",
        str(verify_every),
        "--max_new_tokens",
        str(max_new_tokens),
        "--repetition_penalty",
        str(repetition_penalty),
        "--exec_timeout",
        str(exec_timeout),
        "--seed",
        str(seed),
        # only needed to locate run A's checkpoint, which carries the mixture in its name
        "--real_data_fraction",
        str(real_data_fraction),
    ]
    if tasks:
        command.extend(["--tasks", tasks])
    if stop_on_success:
        command.append("--stop_on_success")
    if extra:
        command.extend(extra.split())
    return command


def verify_command(
    path: str,
    generation: int,
    block_size: int,
    suffix_file: str,
    out_file: str,
    label: str,
    max_new_tokens: int,
    repetition_penalty: float,
    exec_timeout: float,
    probe_only: bool = False,
    tasks: str = "",
    real_data_fraction: float = 0.0,
) -> list:
    """Builds the utils.verify_transfer argv for one target run.

    The decoding parameters are the same ones the attack used and are the same for both target
    runs, which is what makes the two verdicts comparable.
    """
    command = [
        sys.executable,
        "-m",
        "utils.verify_transfer",
        "--out_file",
        out_file,
        "--collapsed_generation",
        str(generation),
        "--block_size",
        str(block_size),
        "--model_specifier",
        MODEL_SPECIFIER,
        "--path",
        path,
        "--label",
        label,
        "--max_new_tokens",
        str(max_new_tokens),
        "--repetition_penalty",
        str(repetition_penalty),
        "--exec_timeout",
        str(exec_timeout),
        # names the target run's checkpoint; both runs share the mixture, so one value serves both
        "--real_data_fraction",
        str(real_data_fraction),
    ]
    if probe_only:
        command.append("--probe_only")
        if tasks:
            command.extend(["--tasks", tasks])
    else:
        command.extend(["--suffix_file", suffix_file])
    return command


def choose_generation(
    path_a: str,
    seed_a: int,
    num_generations: int,
    block_size: int,
    tasks: str,
    probe_dir: str,
    min_usable: int,
    max_new_tokens: int,
    repetition_penalty: float,
    exec_timeout: float,
    force: bool,
    real_data_fraction: float = 0.0,
) -> int:
    """Picks the latest generation at which *run A* can still solve enough tasks unaided.

    Only run A is probed, and that restriction is the experiment rather than a shortcut. Run B is
    held out: the attacker in this threat model has run A and nothing else, so letting run B's
    capability decide which generation to attack would pick the generation that happens to suit
    the model the suffix is later tested against, and the transfer rate would be conditioned on
    the answer it is supposed to measure.

    The consequence is accepted deliberately: run B may turn out to be incapable at the chosen
    generation, and the suffixes then come back `inconclusive` instead of `held` or
    `transferred`. That is a real result — the attackable window moves between runs precisely
    because they are different collapse trajectories — and it is reported rather than designed
    away by choosing a generation that suits both.

    The stage exists at all because the window is narrow and does not sit where one would expect:
    past a certain generation the collapsed model solves nothing, the attack aborts on its own
    capability gate, and `--skip_capability_check` does not help because per-task exclusion still
    removes every task the model already gets wrong.

    The latest qualifying generation is picked rather than the earliest: the further the collapse
    has progressed, the more interesting the question, so the most collapsed still-capable
    generation is the strongest test that remains meaningful.

    Returns:
        int: the chosen generation index

    Raises:
        RuntimeError: no generation leaves run A with enough capable tasks
    """
    table = []
    for generation in range(num_generations - 1, -1, -1):
        out_file = os.path.join(probe_dir, f"probe_run_a_gen{generation}.json")
        if not (os.path.isfile(out_file) and not force):
            run_stage(
                verify_command(
                    path_a, generation, block_size, "", out_file,
                    f"run A (seed {seed_a}) generation {generation}",
                    max_new_tokens, repetition_penalty, exec_timeout,
                    probe_only=True, tasks=tasks,
                    real_data_fraction=real_data_fraction,
                ),
                f"capability probe of run A generation {generation}",
            )
        with open(out_file, "r", encoding="utf-8") as handle:
            probe = json.load(handle)["probe"]

        table.append((generation, probe))
        print(
            f"##   generation {generation}: run A {probe['n_capable']}/{probe['n_tasks']} "
            f"capable{' -> ' + ', '.join(probe['usable']) if probe['usable'] else ''}"
        )
        if probe["n_capable"] >= min_usable:
            print(
                f"\n## {TColors.OKGREEN}{TColors.BOLD}Chose generation {generation}{TColors.ENDC}: "
                f"{probe['n_capable']} task(s) solvable unaided by run A "
                f"({', '.join(probe['usable'])}). Run B was not consulted — whether it is still "
                f"capable here is part of the result, not of the setup"
            )
            return generation

    best = max(table, key=lambda row: row[1]["n_capable"]) if table else None
    raise RuntimeError(
        f"no generation leaves run A with at least --min_usable_tasks {min_usable} task(s) "
        f"solvable unaided, so there is no generation it can be selectively attacked at. The best "
        f"was generation {best[0]} with {best[1]['n_capable']} capable task(s). Collapse less far "
        f"(fewer --num_generations, lower --learning_rate or --lora_rank via --baseline_extra), "
        f"or add tasks to run_attack.TASKS that survive further into the collapse"
    )


def extract_suffixes(results_file: str) -> list:
    """Pulls the verified working suffixes out of run_attack.py's result file.

    Only `successes` entries are taken, never `best_suffix`: a success is a suffix the attack
    *behaviourally verified* against run A, while best_suffix is merely the lowest loss seen and
    may never have broken anything. Testing the latter for transfer would ask whether something
    that never worked still does not work.

    Args:
        results_file (str): run_attack.py's JSON output for run A

    Returns:
        list: {"task", "suffix", "origin"} dicts, deduplicated

    Raises:
        RuntimeError: the attack aborted on its capability gate, so its verdicts are void
    """
    with open(results_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("aborted"):
        probe = data.get("capability_probe", {})
        raise RuntimeError(
            f"the attack in {results_file} aborted on its capability gate "
            f"(reason: {probe.get('reason', 'unknown')}). Run A's collapsed model could not solve "
            f"enough tasks unaided, so there are no meaningful suffixes to transfer. Lower "
            f"--collapsed_generation, or collapse less far"
        )

    seen = set()
    candidates = []
    for outcome in data["results"]:
        for hit in outcome.get("successes", []):
            key = (outcome["task"], hit["suffix"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "task": outcome["task"],
                    "suffix": hit["suffix"],
                    "origin": (
                        f"run_a restart {hit.get('restart')} step {hit.get('step')}"
                    ),
                }
            )
    return candidates


def sweep_run(
    label: str,
    path: str,
    seed: int,
    generations: list,
    block_size: int,
    suffix_file: str,
    report_dir: str,
    max_new_tokens: float,
    repetition_penalty: float,
    exec_timeout: float,
    force: bool,
    real_data_fraction: float = 0.0,
) -> dict:
    """Verifies the frozen suffixes against one run at every generation, and loads the results.

    No optimization happens here or anywhere after the attack: this evaluates the same strings
    against a different checkpoint each time, which is why sweeping is cheap relative to the search
    and why both runs can be swept over the whole range.

    Args:
        label (str): "a" or "b", used in the artifact names
        path (str): root of the run
        seed (int): the run's collapse seed, for the console label only
        generations (list): generation indices to verify at

    Returns:
        dict: {generation: parsed verification JSON}
    """
    loaded = {}
    for generation in generations:
        out_file = os.path.join(report_dir, f"verify_run_{label}_gen{generation}.json")
        if os.path.isfile(out_file) and not force:
            print(
                f"## {TColors.OKGREEN}skipped{TColors.ENDC} generation {generation}: "
                f"{out_file} already exists"
            )
        else:
            run_stage(
                verify_command(
                    path, generation, block_size, suffix_file, out_file,
                    f"run {label.upper()} (seed {seed}) generation {generation}",
                    max_new_tokens, repetition_penalty, exec_timeout,
                    real_data_fraction=real_data_fraction,
                ),
                f"verification of run {label.upper()} generation {generation}",
            )
        with open(out_file, "r", encoding="utf-8") as handle:
            loaded[generation] = json.load(handle)
    return loaded


def tally(rows: list) -> dict:
    """Counts verdicts and derives the transfer rate for one set of rows.

    The denominator is `transferred + held` — the candidates the target could actually be said to
    have resisted or not. `inconclusive` (the target already failed that task with no suffix) and
    `void` (the suffix did not reproduce on its own run) are reported but never divided by, since
    a target that fails everything would otherwise score a perfect transfer rate.
    """
    counts = {
        key: sum(1 for r in rows if r["verdict"] == key)
        for key in ("transferred", "held", "inconclusive", "void")
    }
    decidable = counts["transferred"] + counts["held"]
    return {
        "n_candidates": len(rows),
        "n_transferred": counts["transferred"],
        "n_held": counts["held"],
        "n_inconclusive": counts["inconclusive"],
        "n_void": counts["void"],
        "n_decidable": decidable,
        "transfer_rate": (counts["transferred"] / decidable) if decidable else None,
    }


def summarize(sweeps: dict, seed_a: int, seed_b: int, attacked: int) -> dict:
    """Joins the two per-generation sweeps into the experiment's answer.

    `sweeps` is {"a": {generation: verification}, "b": {...}}. Both runs are swept over the same
    generations because run B alone cannot answer the question. A suffix that breaks run B at
    generation 3 might be reaching across runs, or it might simply be that generation-3 models are
    easy — sweeping run A over the same range separates the two:

        run A, generation `attacked`   the suffix was optimized here; it must work, and that cell
                                       is the validity anchor for everything else
        run A, shallower generations   how far the suffix reaches *within its own* collapse
                                       trajectory, i.e. cross-generation reach with the run held
                                       fixed
        run B, generation `attacked`   transfer in the strict sense: a different trajectory at the
                                       same depth, with the generation held fixed
        run B, shallower generations   both differ at once, and is only interpretable against the
                                       run A column beside it

    The anchor is run A at `attacked`. A suffix that does not reproduce there cannot be said to
    have failed anywhere else — something about the checkpoint or the decoding changed under it —
    so it is `void` in every cell rather than re-litigated per generation.

    Every (run, generation) cell gets its own denominator, because capability differs per cell and
    a pooled rate would silently weight the generations by how capable they happened to be.
    """
    anchor = {(r["task"], r["suffix"]): r for r in sweeps["a"][attacked]["records"]}

    rows = []
    by_run = {}
    for label in ("a", "b"):
        by_run[label] = {}
        for generation in sorted(sweeps[label]):
            cell_rows = []
            for record in sweeps[label][generation]["records"]:
                key = (record["task"], record["suffix"])
                reproduced = anchor.get(key, {}).get("outcome") == "transferred"
                if not reproduced:
                    verdict = "void"
                elif record["outcome"] in ("transferred", "held"):
                    verdict = record["outcome"]
                else:
                    verdict = "inconclusive"
                cell_rows.append(
                    {
                        "run": label,
                        "generation": generation,
                        "task": record["task"],
                        "suffix": record["suffix"],
                        "origin": record["origin"],
                        "verdict": verdict,
                        "reproduced_on_run_a": reproduced,
                        "outcome": record["outcome"],
                        "reason": record["reason"],
                        "clean_collapsed_status": record["clean_collapsed_status"],
                        "collapsed_status": record["collapsed_status"],
                        "baseline_status": record["baseline_status"],
                    }
                )
            by_run[label][generation] = {
                **tally(cell_rows),
                "model": sweeps[label][generation]["summary"]["collapsed_model"],
            }
            rows.extend(cell_rows)

    anchor_records = sweeps["a"][attacked]["records"]
    return {
        "seed_run_a": seed_a,
        "seed_run_b": seed_b,
        "attacked_generation": attacked,
        "swept_generations": sorted(sweeps["b"]),
        "n_suffixes": len(anchor_records),
        "n_reproduced_on_run_a": sum(
            1 for r in anchor_records if r["outcome"] == "transferred"
        ),
        # the matched comparison: same generation, different trajectory. Read it against
        # by_run["a"][attacked], which is the anchor and therefore all-transferred by construction
        "matched_generation": by_run["b"].get(attacked),
        "by_run": by_run,
        "totals": {label: tally([r for r in rows if r["run"] == label]) for label in ("a", "b")},
        "rows": rows,
    }


def main(
    root: str = "./runs/transfer",
    seed_a: int = 1337,
    seed_b: int = 2024,
    num_generations: int = 10,
    collapsed_generation: int = -1,
    block_size: int = 512,
    dataset_size: int = 0,
    engine: str = "auto",
    tasks: str = "",
    restarts: int = 3,
    num_steps: int = 250,
    verify_every: int = 10,
    max_new_tokens: int = 96,
    repetition_penalty: float = 1.0,
    exec_timeout: float = 10.0,
    stop_on_success: bool = False,
    attack_seed: int = 1337,
    min_usable_tasks: int = 1,
    with_eval: bool = False,
    force: bool = False,
    baseline_extra: str = "",
    attack_extra: str = "",
    real_data_fraction: float = 0.0,
    model_specifier: str = "",
    model_size: str = "",
) -> None:
    """Runs the whole cross-run transfer experiment.

    Every stage is skipped when its artifact already exists, so an interrupted run can simply be
    started again. --force disables that.

    Args:
        root (str): directory the two runs and the report are written under
        seed_a (int): seed of the run the suffix is searched on
        seed_b (int): seed of the run the suffix is transferred into. Must differ from seed_a
        num_generations (int): generations per collapse run
        collapsed_generation (int): generation of run A to attack. Run B is then tested at every
            generation from 0 up to and including it. -1 probes run A
        block_size (int): shared block size of both runs
        dataset_size (int): shared dataset size of both runs. 0 uses the whole dataset
        engine (str): dataset generation engine for both runs
        tasks (str): comma separated task subset, empty for all
        restarts (int): GCG restarts per task
        num_steps (int): GCG steps per restart
        verify_every (int): behavioural check interval during the search
        max_new_tokens (int): verification decoding length, shared by all three verifications
        repetition_penalty (float): verification decoding penalty. 1.0 is plain greedy decoding
        exec_timeout (float): unit test timeout
        stop_on_success (bool): stop a task at its first hit
        attack_seed (int): RNG seed of the search itself, unrelated to the collapse seeds
        min_usable_tasks (int): how many tasks *run A* must still solve unaided for a generation to
            be chosen by the probe. Run B is not consulted, see choose_generation. Only read when
            --collapsed_generation is -1
        with_eval (bool): also run the perplexity evaluation and the histogram of both runs
        force (bool): rerun every stage even when its artifact exists
        baseline_extra (str): extra arguments appended to both run_baseline.py invocations
        attack_extra (str): extra arguments appended to the run_attack.py invocation
        real_data_fraction (float): --real_data_fraction for *both* collapse runs, and therefore
            part of the checkpoint names every later stage resolves. Held identical across the two
            runs on purpose: the seed is meant to be the only difference between them
        model_specifier (str): base model of *both* collapse runs, forwarded to every stage
        model_size (str): parameter count off the Qwen2.5-Coder ladder ("0.5b" ... "32b"),
            shorthand for the matching model_specifier. Like the mixture, this is deliberately one
            value for both runs rather than something to hide in --baseline_extra: the later
            stages have to resolve the same checkpoint names, and --baseline_extra is opaque to
            them. The seed is meant to be the only difference between the two runs
    """
    if seed_a == seed_b:
        raise ValueError(
            f"--seed_a and --seed_b are both {seed_a}, so the two runs would be the same collapse "
            f"trajectory and the experiment would be vacuous. run_baseline.py is deterministic "
            f"given a seed: the training seed and every generation worker's sampling seed are "
            f"derived from it"
        )

    auto_generation = collapsed_generation < 0
    if auto_generation:
        # deliberately not "the last generation": in a 10-generation run the last one typically
        # solves nothing unaided, which makes both the attack and the transfer verdict void. The
        # generation is probed for on run A instead, before run B exists. See choose_generation
        collapsed_generation = num_generations - 1
    if collapsed_generation >= num_generations:
        raise ValueError(
            f"--collapsed_generation {collapsed_generation} does not exist in a run of "
            f"--num_generations {num_generations}; the indices are 0..{num_generations - 1}"
        )

    # one model for both runs, resolved once here and forwarded to every stage below. The stages
    # name their artifacts after its trailing component, so resolving it per stage — or letting it
    # ride along in --baseline_extra, which the attack and the verifications never see — would let
    # the two runs collapse different models under one report
    global MODEL_SPECIFIER
    MODEL_SPECIFIER = resolve_model_specifier(model_size, model_specifier, MODEL_SPECIFIER)
    specifier_name = MODEL_SPECIFIER.split("/")[-1]
    path_a = os.path.join(root, f"run_a_seed{seed_a}")
    path_b = os.path.join(root, f"run_b_seed{seed_b}")
    report_dir = os.path.join(root, "transfer_report")
    os.makedirs(report_dir, exist_ok=True)

    width = shutil.get_terminal_size().columns
    print("\n" + "═" * width)
    print(
        f"## {TColors.BOLD}Cross-run adversarial suffix transfer{TColors.ENDC}\n"
        f"## date: {datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}\n"
        f"## user: {TColors.HEADER}{getpass.getuser()}{TColors.ENDC}\n"
        f"## GPUs: {TColors.HEADER}{VISIBLE_DEVICES}{TColors.ENDC}\n"
        f"## RAM: {TColors.HEADER}{psutil.virtual_memory().total // 1024**3} GB{TColors.ENDC}\n"
        f"## model: {TColors.HEADER}{MODEL_SPECIFIER}{TColors.ENDC}\n"
        f"## run A (search):   seed {TColors.OKGREEN}{seed_a}{TColors.ENDC} -> {path_a}\n"
        f"## run B (transfer): seed {TColors.OKGREEN}{seed_b}{TColors.ENDC} -> {path_b}\n"
        f"## generation under attack: "
        f"{TColors.OKGREEN}{'probed for' if auto_generation else collapsed_generation}"
        f"{TColors.ENDC} of 0..{num_generations - 1}"
    )
    print("═" * width)

    # both runs must see the same GPUs: the shard count sets how the instruction set is split and
    # therefore each shard's sampling seed, so changing it between the runs would add a second
    # difference on top of the seed and a transfer failure could not be attributed to either
    print(
        f"\n## {TColors.WARNING}Keep CUDA_VISIBLE_DEVICES identical for both runs{TColors.ENDC} — "
        f"the shard count feeds into the per-shard sampling seeds, so a different GPU count is a "
        f"second difference between the runs. Both runs of this invocation inherit "
        f"{VISIBLE_DEVICES}."
    )

    # the run the suffix is searched on. Run B is deliberately not trained yet: nothing before the
    # verification is allowed to depend on it, and if the search finds no working suffix there is
    # no reason to have spent hours collapsing a second run
    def collapse(index: int, label: str, path: str, seed: int) -> None:
        """Runs one collapse run, skipping it when its last generation is already on disk."""
        # the last generation's checkpoint is the marker for "this run finished", independently of
        # which generation is eventually attacked
        final_checkpoint = collapsed_checkpoint(
            path, num_generations - 1, block_size, specifier_name, real_data_fraction
        )
        print_stage(index, 6, f"collapse run {label}", f"seed {seed}")
        if os.path.isdir(final_checkpoint) and not force:
            print(f"## {TColors.OKGREEN}skipped{TColors.ENDC}: {final_checkpoint} already exists")
        else:
            run_stage(
                baseline_command(
                    path, seed, num_generations, block_size, dataset_size, engine,
                    with_eval, baseline_extra, real_data_fraction,
                ),
                f"collapse run {label}",
            )

    # ── stage 1: collapse run A, the run under attack ──────────────────────────────────────────
    collapse(1, "A", path_a, seed_a)

    # ── stage 2: pick a generation run A is still capable at ───────────────────────────────────
    print_stage(2, 6, "choose the generation under attack", "run A only")
    if auto_generation:
        collapsed_generation = choose_generation(
            path_a, seed_a, num_generations, block_size, tasks, report_dir,
            min_usable_tasks, max_new_tokens, repetition_penalty, exec_timeout, force,
            real_data_fraction,
        )
    else:
        print(
            f"## generation {TColors.OKGREEN}{collapsed_generation}{TColors.ENDC} was given "
            f"explicitly, so no capability probe is run. If the attack aborts on its capability "
            f"gate, pass -cg -1 to probe for an attackable generation instead"
        )

    suffix_file = os.path.join(report_dir, f"suffixes_gen{collapsed_generation}.json")
    # the per-run, per-generation verification files are named inside sweep_run
    report_file = os.path.join(report_dir, f"transfer_gen{collapsed_generation}.json")

    # ── stage 3: search a suffix against run A ─────────────────────────────────────────────────
    # the only optimization in the experiment. It sees run A's checkpoint and nothing else, and
    # its output is frozen from here on: stages 5 and 6 only ever *evaluate* these strings
    results_a = attack_results_file(path_a, collapsed_generation, specifier_name)
    print_stage(3, 6, "attack run A", f"generation {collapsed_generation}")
    if os.path.isfile(results_a) and not force:
        print(f"## {TColors.OKGREEN}skipped{TColors.ENDC}: {results_a} already exists")
    else:
        run_stage(
            attack_command(
                path_a, collapsed_generation, block_size, tasks, restarts, num_steps,
                verify_every, max_new_tokens, repetition_penalty, exec_timeout,
                stop_on_success, attack_seed, attack_extra, real_data_fraction,
            ),
            "attack against run A",
        )

    candidates = extract_suffixes(results_a)
    if not candidates:
        raise RuntimeError(
            f"the attack in {results_a} found no verified working suffix against run A, so there "
            f"is nothing to transfer. Raise --num_steps or --restarts, or attack an earlier "
            f"--collapsed_generation where the model is still capable enough to be selectively "
            f"broken"
        )
    with open(suffix_file, "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2)
    print(
        f"\n## {TColors.OKGREEN}{len(candidates)} verified suffix(es){TColors.ENDC} across "
        f"{len({c['task'] for c in candidates})} task(s) -> {suffix_file}"
    )

    # ── stage 4: collapse run B, the held out run ──────────────────────────────────────────────
    # trained only now, after the suffixes exist and are frozen. Nothing that produced them could
    # have depended on it, which is what makes stage 6 a transfer measurement rather than a fit
    collapse(4, "B", path_b, seed_b)

    # ── stages 5 and 6: sweep both runs over every generation up to the attacked one ───────────
    # the suffixes are frozen, so each cell is the same question asked of a different checkpoint.
    # Run A is swept as well as run B, and not only as a control: its cell at the attacked
    # generation is the validity anchor, while its shallower generations give the within-run
    # baseline that makes run B's column readable. Without it, a hit at run B generation 3 cannot
    # be told apart from "generation-3 models are easy"
    swept = list(range(collapsed_generation + 1))
    sweeps = {}
    for index, (label, path, seed) in enumerate(
        (("a", path_a, seed_a), ("b", path_b, seed_b)), start=5
    ):
        print_stage(
            index, 6, f"sweep run {label.upper()}",
            f"generations {swept[0]}..{swept[-1]}"
            + (" (anchor + within-run baseline)" if label == "a" else " (cross-run transfer)"),
        )
        sweeps[label] = sweep_run(
            label, path, seed, swept, block_size, suffix_file, report_dir,
            max_new_tokens, repetition_penalty, exec_timeout, force, real_data_fraction,
        )

    # ── the report ─────────────────────────────────────────────────────────────────────────────
    report = summarize(sweeps, seed_a, seed_b, collapsed_generation)
    report["collapsed_generation"] = collapsed_generation
    report["num_generations"] = num_generations
    report["block_size"] = block_size
    report["dataset_size"] = dataset_size
    report["real_data_fraction"] = real_data_fraction
    report["visible_devices"] = VISIBLE_DEVICES
    report["verification"] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "exec_timeout": exec_timeout,
    }
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "═" * width)
    print(f"## {TColors.BOLD}Transfer report{TColors.ENDC}")
    print("═" * width)
    colours = {
        "transferred": TColors.OKGREEN,
        "held": TColors.OKBLUE,
        "inconclusive": TColors.WARNING,
        "void": TColors.FAIL,
    }
    marks = {"transferred": "T", "held": "H", "inconclusive": "?", "void": "x"}

    # one row per suffix, one column per generation. The sweep is a curve rather than a set of
    # unrelated results — the same frozen suffix against progressively deeper collapse — and a
    # grid is what makes that depth dependence readable at a glance
    # two lines per suffix, one per run, so the same generation of the two trajectories sits in the
    # same column. Reading down a column compares the runs at equal collapse depth; reading along
    # the A line shows how far the suffix reaches without changing run at all
    by_candidate = {}
    for row in report["rows"]:
        by_candidate.setdefault((row["task"], row["suffix"]), {})[
            (row["run"], row["generation"])
        ] = row
    label_width = 44
    print(f"##   {'task / suffix':<{label_width}} run  gen: " + " ".join(f"{g:>2}" for g in swept))
    for (task, suffix), cells in by_candidate.items():
        label = f"{task} {suffix!r}"
        label = label if len(label) <= label_width else label[: label_width - 1] + "…"
        for line, run_label in enumerate(("a", "b")):
            painted = " ".join(
                f"{colours[cells[(run_label, g)]['verdict']]}"
                f"{marks[cells[(run_label, g)]['verdict']]:>2}{TColors.ENDC}"
                for g in swept
            )
            print(
                f"##   {(label if line == 0 else ''):<{label_width}} "
                f"{run_label.upper():<3}      {painted}"
            )
    print(
        f"##   {TColors.OKGREEN}T{TColors.ENDC} works   "
        f"{TColors.OKBLUE}H{TColors.ENDC} held   "
        f"{TColors.WARNING}?{TColors.ENDC} inconclusive   "
        f"{TColors.FAIL}x{TColors.ENDC} void\n"
        f"##   A = the run the suffix was found on, B = the independent run"
    )

    print(
        f"##\n##   {'gen':>4}   {'A: works':>9} {'held':>5} {'?':>3} {'rate':>6}   "
        f"{'B: works':>9} {'held':>5} {'?':>3} {'rate':>6}"
    )
    for generation in swept:
        cells = []
        for label in ("a", "b"):
            stats = report["by_run"][label][generation]
            rate = stats["transfer_rate"]
            cells.append(
                f"{stats['n_transferred']:>9} {stats['n_held']:>5} "
                f"{stats['n_inconclusive']:>3} "
                f"{('n/a' if rate is None else f'{rate:.0%}'):>6}"
            )
        marker = (
            f"  {TColors.BOLD}<- attacked{TColors.ENDC}"
            if generation == collapsed_generation
            else ""
        )
        print(f"##   {generation:>4}   {cells[0]}   {cells[1]}{marker}")

    matched = report["matched_generation"] or {}
    matched_rate = matched.get("transfer_rate")
    totals = report["totals"]
    print(
        f"##\n"
        f"##   suffixes:              {report['n_suffixes']}\n"
        f"##   reproduced on run A:   {report['n_reproduced_on_run_a']}\n"
        f"##   {TColors.BOLD}cross-run transfer at the attacked generation "
        f"{collapsed_generation}: "
        f"{'n/a' if matched_rate is None else f'{matched_rate:.0%}'}{TColors.ENDC} "
        f"({matched.get('n_transferred', 0)}/{matched.get('n_decidable', 0)} decidable)"
    )
    inconclusive = ", ".join(
        f"{totals[label]['n_inconclusive']} in run {label.upper()}"
        for label in ("a", "b")
        if totals[label]["n_inconclusive"]
    )
    if inconclusive:
        print(
            f"##\n## {TColors.WARNING}Inconclusive cells ({inconclusive}){TColors.ENDC}: that run "
            f"already fails those tasks at that generation with no suffix at all, so nothing can "
            f"be attributed to the suffix. They are excluded from every rate rather than counted "
            f"as failures."
        )
    n_void_suffixes = report["n_suffixes"] - report["n_reproduced_on_run_a"]
    if n_void_suffixes:
        print(
            f"##\n## {TColors.FAIL}{n_void_suffixes} suffix(es) did not reproduce on run "
            f"A{TColors.ENDC}, so they are void at every generation. Verification is greedy, so "
            f"this should not happen — check that run A's checkpoint was not overwritten and that "
            f"--max_new_tokens matches the attack's."
        )
    print(f"##\n## {TColors.OKBLUE}{TColors.BOLD}Report: {report_file}{TColors.ENDC}")
    print("═" * width + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test whether an adversarial suffix found against one collapse run still "
        "works against an independently collapsed run of the same generation"
    )
    parser.add_argument(
        "--root",
        "-r",
        type=str,
        default="./runs/transfer",
        help="directory both runs and the report are written under (default: ./runs/transfer)",
    )
    parser.add_argument(
        "--seed_a",
        "-sa",
        type=int,
        default=1337,
        help="collapse seed of the run the suffix is searched on (default: 1337)",
    )
    parser.add_argument(
        "--seed_b",
        "-sb",
        type=int,
        default=4267,
        help="collapse seed of the run the suffix is transferred into. Must differ from --seed_a, "
        "otherwise both runs are the same trajectory (default: 4267)",
    )
    parser.add_argument(
        "--num_generations",
        "-ng",
        type=int,
        default=10,
        help="generations per collapse run (default: 10)",
    )
    parser.add_argument(
        "--collapsed_generation",
        "-cg",
        type=int,
        default=-1,
        help="generation of run A to attack; the frozen suffixes are then tested against run B at "
        "every generation from 0 up to and including it, so one run yields the whole depth curve "
        "rather than a single point. -1 probes for the latest generation at "
        "which RUN A can still solve --min_usable_tasks tasks unaided — run B is held out and "
        "never consulted. It is not the last generation: a fully collapsed model solves nothing, "
        "and then the attack aborts on its own "
        "capability gate and every transfer verdict would be inconclusive anyway (default: -1)",
    )
    parser.add_argument(
        "--block_size",
        "-bs",
        type=int,
        default=512,
        help="block size of both runs; it is baked into every artifact name (default: 512)",
    )
    parser.add_argument(
        "--dataset_size",
        "-dsz",
        type=int,
        default=0,
        help="dataset size of both runs; 0 uses the whole dataset (default: 0)",
    )
    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="auto",
        choices=["auto", "vllm", "transformers"],
        help="dataset generation engine for both runs (default: auto)",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default="",
        help="comma separated attack tasks (default: all)",
    )
    parser.add_argument(
        "--restarts",
        "-rs",
        type=int,
        default=3,
        help="GCG restarts per task (default: 3)",
    )
    parser.add_argument(
        "--num_steps",
        "-ns",
        type=int,
        default=250,
        help="GCG steps per restart (default: 250)",
    )
    parser.add_argument(
        "--verify_every",
        "-ve",
        type=int,
        default=10,
        help="behavioural check interval during the search (default: 10)",
    )
    parser.add_argument(
        "--max_new_tokens",
        "-mnt",
        type=int,
        default=512,
        help="verification decoding length, shared by the search and both verifications so the "
        "three verdicts are produced under identical decoding (default: 512)",
    )
    parser.add_argument(
        "--repetition_penalty",
        "-rp",
        type=float,
        default=1.0,
        help="verification decoding penalty. 1.0 is plain greedy decoding. This is the attack's "
        "verification knob and is unrelated to the dataset generation's penalty (default: 1.0)",
    )
    parser.add_argument(
        "--exec_timeout",
        "-et",
        type=float,
        default=10.0,
        help="unit test timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--stop_on_success",
        "-sos",
        action="store_true",
        help="stop each task at its first hit. Fewer candidates, much faster",
    )
    parser.add_argument(
        "--attack_seed",
        "-as",
        type=int,
        default=1337,
        help="RNG seed of the search itself, unrelated to the two collapse seeds (default: 1337)",
    )
    parser.add_argument(
        "--min_usable_tasks",
        "-mut",
        type=int,
        default=1,
        help="how many tasks RUN A must still solve unaided for a generation to be picked by the "
        "probe. Run B is held out and never consulted, so it may be incapable at the chosen "
        "generation — that is reported as inconclusive. Only read when --collapsed_generation "
        "is -1 (default: 1)",
    )
    parser.add_argument(
        "--with_eval",
        "-we",
        action="store_true",
        help="also run each collapse run's perplexity evaluation and histogram. Off by default: "
        "this experiment reads none of it, and the plotting step needs a LaTeX install",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="rerun every stage even when its artifact already exists",
    )
    parser.add_argument(
        "--real_data_fraction",
        "-rdf",
        type=float,
        default=0.0,
        help="--real_data_fraction for both collapse runs, held identical so that the seed stays "
        "the only difference between them. It is part of the checkpoint names from generation 1 "
        "onward, so passing it here is also what lets the attack and the verifications find them — "
        "prefer it over putting -rdf in --baseline_extra, which the later stages cannot see "
        "(default: 0.0)",
    )
    add_model_arguments(parser, role="the base model of both collapse runs")
    parser.add_argument(
        "--baseline_extra",
        "-bx",
        type=str,
        default="",
        help="extra arguments appended verbatim to both run_baseline.py invocations, e.g. "
        "\"-tbs 8 -gas 4 -fi\". Do not pass -rdf, -ms or -msz here — use --real_data_fraction and "
        "--model_size/--model_specifier, which the checkpoint-name resolution in the later stages "
        "also reads",
    )
    parser.add_argument(
        "--attack_extra",
        "-ax",
        type=str,
        default="",
        help="extra arguments appended verbatim to the run_attack.py invocation, e.g. "
        "\"--min_capability 0.3\"",
    )
    args = parser.parse_args()
    main(**vars(args))
