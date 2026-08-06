"""main hook to test whether an adversarial suffix transfers between two collapse runs"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import datetime
import getpass
import json
import os
import subprocess
import sys
import time
from datetime import timedelta

import psutil

from utils.colors import TColors
from utils.devices import visible_devices

# this orchestrator never touches a model in-process, matching run_baseline.py and
# run_extrapolation.py: every stage that loads weights is a subprocess. Here that matters for a
# second reason beyond allocator hygiene — stages 1 and 3 run the unsloth/vLLM stack while the
# verification runs run_attack's plain transformers stack, and unsloth patches transformers at
# import time. Keeping them in separate interpreters is what lets both be correct
VISIBLE_DEVICES = visible_devices()

MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"


def collapsed_checkpoint(
    path: str, generation: int, block_size: int, specifier_name: str
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

    Returns:
        str: path to the merged fp16 checkpoint directory
    """
    return os.path.join(
        path,
        "model_outputs",
        f"model_{generation}_bs{block_size}_{specifier_name}_fp16",
    )


def attack_results_file(path: str, generation: int, specifier_name: str) -> str:
    """Path of the result file run_attack.py writes in plain (non-surrogate) mode."""
    return os.path.join(
        path, "attack_results", f"attack_gen{generation}_{specifier_name}.json"
    )


def print_stage(index: int, total: int, title: str, detail: str = "") -> None:
    """Prints a stage banner. Mirrors the banner style of the other orchestrators."""
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
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
) -> list:
    """Builds the run_baseline.py argv for one collapse run."""
    command = [
        sys.executable,
        "run_baseline.py",
        "--device",
        "cuda",
        "--seed",
        str(seed),
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
    path_b: str,
    seed_a: int,
    seed_b: int,
    num_generations: int,
    block_size: int,
    tasks: str,
    probe_dir: str,
    min_usable: int,
    max_new_tokens: int,
    repetition_penalty: float,
    exec_timeout: float,
    force: bool,
) -> int:
    """Picks the latest generation at which *both* runs can still solve enough tasks unaided.

    This stage exists because the attackable window is narrow and does not sit where one would
    expect. Past a certain generation the collapsed model solves nothing at all, and then:

      - the attack on run A aborts on its own capability gate, so there is no suffix to transfer,
        and `--skip_capability_check` does not help because per-task exclusion still removes every
        task the model already gets wrong;
      - and even if a suffix existed, every transfer verdict against run B would be inconclusive,
        since a model that fails a task with no suffix cannot be said to have been broken by one.

    The window also *moves* between the two runs, because they are different collapse
    trajectories — which is the entire point of the experiment. So the generation has to be chosen
    against both runs, not just against run A.

    The latest such generation is picked rather than the earliest: the further the collapse has
    progressed, the more interesting the question, so the most collapsed still-capable generation
    is the strongest test that remains meaningful.

    Returns:
        int: the chosen generation index

    Raises:
        RuntimeError: no generation leaves both runs with enough capable tasks
    """
    table = []
    for generation in range(num_generations - 1, -1, -1):
        counts = {}
        for label, path, seed in (("a", path_a, seed_a), ("b", path_b, seed_b)):
            out_file = os.path.join(probe_dir, f"probe_run_{label}_gen{generation}.json")
            if not (os.path.isfile(out_file) and not force):
                run_stage(
                    verify_command(
                        path, generation, block_size, "", out_file,
                        f"run {label.upper()} (seed {seed}) generation {generation}",
                        max_new_tokens, repetition_penalty, exec_timeout,
                        probe_only=True, tasks=tasks,
                    ),
                    f"capability probe of run {label.upper()} generation {generation}",
                )
            with open(out_file, "r", encoding="utf-8") as handle:
                counts[label] = json.load(handle)["probe"]

        both = sorted(set(counts["a"]["usable"]) & set(counts["b"]["usable"]))
        table.append((generation, counts["a"], counts["b"], both))
        print(
            f"##   generation {generation}: run A {counts['a']['n_capable']}/"
            f"{counts['a']['n_tasks']} capable, run B {counts['b']['n_capable']}/"
            f"{counts['b']['n_tasks']} capable, both: {both or '-'}"
        )
        if len(both) >= min_usable:
            print(
                f"\n## {TColors.OKGREEN}{TColors.BOLD}Chose generation {generation}{TColors.ENDC}: "
                f"{len(both)} task(s) solvable unaided by both runs ({', '.join(both)})"
            )
            return generation

    best = max(table, key=lambda row: len(row[3])) if table else None
    raise RuntimeError(
        f"no generation leaves both runs with at least --min_usable_tasks {min_usable} task(s) "
        f"solvable unaided, so there is no generation at which a transfer result would be "
        f"meaningful. The best was generation {best[0]} with {len(best[3])} shared capable "
        f"task(s). Collapse less far (fewer --num_generations, lower --learning_rate or "
        f"--lora_rank via --baseline_extra), or add tasks to run_attack.TASKS that survive "
        f"further into the collapse"
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


def summarize(control: dict, target: dict, seed_a: int, seed_b: int) -> dict:
    """Joins the two verification passes into the experiment's actual answer.

    The control pass (against run A, the run the suffix was found on) is what makes the target
    pass interpretable. A suffix that does not even reproduce on its own run cannot be said to
    have failed to transfer — something about the decoding or the checkpoint changed instead — so
    those are excluded from the transfer rate rather than counted as failures.
    """
    control_by_key = {(r["task"], r["suffix"]): r for r in control["records"]}

    rows = []
    for record in target["records"]:
        key = (record["task"], record["suffix"])
        reproduced = control_by_key.get(key, {}).get("outcome") == "transferred"
        if not reproduced:
            verdict = "void"
        elif record["outcome"] == "transferred":
            verdict = "transferred"
        elif record["outcome"] == "held":
            verdict = "held"
        else:
            verdict = "inconclusive"
        rows.append(
            {
                "task": record["task"],
                "suffix": record["suffix"],
                "origin": record["origin"],
                "verdict": verdict,
                "reproduced_on_run_a": reproduced,
                "run_a_outcome": control_by_key.get(key, {}).get("outcome"),
                "run_b_outcome": record["outcome"],
                "run_b_reason": record["reason"],
                "run_b_clean_collapsed_status": record["clean_collapsed_status"],
                "run_b_collapsed_status": record["collapsed_status"],
                "run_b_baseline_status": record["baseline_status"],
            }
        )

    n_transferred = sum(1 for r in rows if r["verdict"] == "transferred")
    n_held = sum(1 for r in rows if r["verdict"] == "held")
    decidable = n_transferred + n_held
    return {
        "seed_run_a": seed_a,
        "seed_run_b": seed_b,
        "run_a_model": control["summary"]["collapsed_model"],
        "run_b_model": target["summary"]["collapsed_model"],
        "n_candidates": len(rows),
        "n_reproduced_on_run_a": sum(1 for r in rows if r["reproduced_on_run_a"]),
        "n_transferred": n_transferred,
        "n_held": n_held,
        "n_inconclusive": sum(1 for r in rows if r["verdict"] == "inconclusive"),
        "n_void": sum(1 for r in rows if r["verdict"] == "void"),
        "transfer_rate": (n_transferred / decidable) if decidable else None,
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
) -> None:
    """Runs the whole cross-run transfer experiment.

    Every stage is skipped when its artifact already exists, so an interrupted run can simply be
    started again. --force disables that.

    Args:
        root (str): directory the two runs and the report are written under
        seed_a (int): seed of the run the suffix is searched on
        seed_b (int): seed of the run the suffix is transferred into. Must differ from seed_a
        num_generations (int): generations per collapse run
        collapsed_generation (int): generation to attack and to transfer into. -1 probes both runs
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
        min_usable_tasks (int): how many tasks both runs must still solve unaided for a generation
            to be chosen by the probe. Only read when --collapsed_generation is -1
        with_eval (bool): also run the perplexity evaluation and the histogram of both runs
        force (bool): rerun every stage even when its artifact exists
        baseline_extra (str): extra arguments appended to both run_baseline.py invocations
        attack_extra (str): extra arguments appended to the run_attack.py invocation
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
        # generation is probed for instead, after both runs exist. See choose_generation
        collapsed_generation = num_generations - 1
    if collapsed_generation >= num_generations:
        raise ValueError(
            f"--collapsed_generation {collapsed_generation} does not exist in a run of "
            f"--num_generations {num_generations}; the indices are 0..{num_generations - 1}"
        )

    specifier_name = MODEL_SPECIFIER.split("/")[-1]
    path_a = os.path.join(root, f"run_a_seed{seed_a}")
    path_b = os.path.join(root, f"run_b_seed{seed_b}")
    report_dir = os.path.join(root, "transfer_report")
    os.makedirs(report_dir, exist_ok=True)

    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    print("\n" + "═" * width)
    print(
        f"## {TColors.BOLD}Cross-run adversarial suffix transfer{TColors.ENDC}\n"
        f"## date: {datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}\n"
        f"## user: {TColors.HEADER}{getpass.getuser()}{TColors.ENDC}\n"
        f"## GPUs: {TColors.HEADER}{VISIBLE_DEVICES}{TColors.ENDC}\n"
        f"## RAM: {TColors.HEADER}{psutil.virtual_memory().total // 1024**3} GB{TColors.ENDC}\n"
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

    # ── stages 1 and 2: the two collapse runs ──────────────────────────────────────────────────
    # both runs are trained before anything is attacked. The generation to attack has to be one
    # where *both* runs can still solve tasks unaided, and that cannot be known from run A alone
    for index, (label, path, seed) in enumerate(
        (("A", path_a, seed_a), ("B", path_b, seed_b)), start=1
    ):
        # the last generation's checkpoint is the marker for "this run finished", independently of
        # which generation is eventually attacked
        final_checkpoint = collapsed_checkpoint(
            path, num_generations - 1, block_size, specifier_name
        )
        print_stage(index, 6, f"collapse run {label}", f"seed {seed}")
        if os.path.isdir(final_checkpoint) and not force:
            print(f"## {TColors.OKGREEN}skipped{TColors.ENDC}: {final_checkpoint} already exists")
        else:
            run_stage(
                baseline_command(
                    path, seed, num_generations, block_size, dataset_size, engine,
                    with_eval, baseline_extra,
                ),
                f"collapse run {label}",
            )

    # ── stage 3: pick a generation both runs are still capable at ──────────────────────────────
    print_stage(3, 6, "choose the generation under attack")
    if auto_generation:
        collapsed_generation = choose_generation(
            path_a, path_b, seed_a, seed_b, num_generations, block_size, tasks, report_dir,
            min_usable_tasks, max_new_tokens, repetition_penalty, exec_timeout, force,
        )
    else:
        print(
            f"## generation {TColors.OKGREEN}{collapsed_generation}{TColors.ENDC} was given "
            f"explicitly, so no capability probe is run. If the attack aborts on its capability "
            f"gate, pass -cg -1 to probe for an attackable generation instead"
        )

    suffix_file = os.path.join(report_dir, f"suffixes_gen{collapsed_generation}.json")
    control_file = os.path.join(report_dir, f"verify_run_a_gen{collapsed_generation}.json")
    target_file = os.path.join(report_dir, f"verify_run_b_gen{collapsed_generation}.json")
    report_file = os.path.join(report_dir, f"transfer_gen{collapsed_generation}.json")

    # ── stage 4: search a suffix against run A ─────────────────────────────────────────────────
    results_a = attack_results_file(path_a, collapsed_generation, specifier_name)
    print_stage(4, 6, "attack run A", f"generation {collapsed_generation}")
    if os.path.isfile(results_a) and not force:
        print(f"## {TColors.OKGREEN}skipped{TColors.ENDC}: {results_a} already exists")
    else:
        run_stage(
            attack_command(
                path_a, collapsed_generation, block_size, tasks, restarts, num_steps,
                verify_every, max_new_tokens, repetition_penalty, exec_timeout,
                stop_on_success, attack_seed, attack_extra,
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

    # ── stage 5: the control — do the suffixes still work on their own run? ────────────────────
    # this is what makes stage 6 readable. Verification is greedy and therefore deterministic, so
    # a suffix the attack recorded as a hit has to reproduce here; one that does not indicates the
    # checkpoint or the decoding changed under it, and it must not be counted as a transfer
    # failure. It also re-runs the verdict through the same code path as stage 6, so the two
    # numbers are produced identically rather than one being read out of the attack's own log
    print_stage(5, 6, "control: re-verify the suffixes against run A")
    if os.path.isfile(control_file) and not force:
        print(f"## {TColors.OKGREEN}skipped{TColors.ENDC}: {control_file} already exists")
    else:
        run_stage(
            verify_command(
                path_a, collapsed_generation, block_size, suffix_file, control_file,
                f"run A (seed {seed_a}, the run the suffixes were found on)",
                max_new_tokens, repetition_penalty, exec_timeout,
            ),
            "control verification against run A",
        )

    # ── stage 6: the actual question ───────────────────────────────────────────────────────────
    print_stage(6, 6, "transfer: verify the suffixes against run B")
    if os.path.isfile(target_file) and not force:
        print(f"## {TColors.OKGREEN}skipped{TColors.ENDC}: {target_file} already exists")
    else:
        run_stage(
            verify_command(
                path_b, collapsed_generation, block_size, suffix_file, target_file,
                f"run B (seed {seed_b}, the newly collapsed run)",
                max_new_tokens, repetition_penalty, exec_timeout,
            ),
            "transfer verification against run B",
        )

    # ── the report ─────────────────────────────────────────────────────────────────────────────
    with open(control_file, "r", encoding="utf-8") as handle:
        control = json.load(handle)
    with open(target_file, "r", encoding="utf-8") as handle:
        target = json.load(handle)

    report = summarize(control, target, seed_a, seed_b)
    report["collapsed_generation"] = collapsed_generation
    report["num_generations"] = num_generations
    report["block_size"] = block_size
    report["dataset_size"] = dataset_size
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
    for row in report["rows"]:
        print(
            f"##   [{row['task']}] {colours[row['verdict']]}{TColors.BOLD}"
            f"{row['verdict'].upper()}{TColors.ENDC} "
            f"run_a={row['run_a_outcome']} run_b={row['run_b_outcome']} "
            f"suffix={row['suffix']!r}"
        )

    rate = report["transfer_rate"]
    print(
        f"##\n"
        f"##   candidates:            {report['n_candidates']}\n"
        f"##   reproduced on run A:   {report['n_reproduced_on_run_a']}\n"
        f"##   {TColors.OKGREEN}transferred to run B:  {report['n_transferred']}{TColors.ENDC}\n"
        f"##   held (did not break):  {report['n_held']}\n"
        f"##   inconclusive:          {report['n_inconclusive']} "
        f"(run B already failed the task with no suffix)\n"
        f"##   void:                  {report['n_void']} (did not reproduce on run A)\n"
        f"##   transfer rate:         "
        f"{'n/a' if rate is None else f'{rate:.0%}'} of the decidable candidates"
    )
    if report["n_inconclusive"]:
        print(
            f"##\n## {TColors.WARNING}{report['n_inconclusive']} candidate(s) are "
            f"inconclusive{TColors.ENDC}: run B's generation-{collapsed_generation} model already "
            f"fails those tasks without any suffix, so nothing can be attributed to the suffix. "
            f"Attack an earlier generation if this dominates the result."
        )
    if report["n_void"]:
        print(
            f"##\n## {TColors.FAIL}{report['n_void']} candidate(s) did not reproduce on run "
            f"A{TColors.ENDC}. Verification is greedy, so this should not happen — check that "
            f"run A's checkpoint was not overwritten and that --max_new_tokens matches the "
            f"attack's."
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
        default=2024,
        help="collapse seed of the run the suffix is transferred into. Must differ from --seed_a, "
        "otherwise both runs are the same trajectory (default: 2024)",
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
        help="generation to attack and to transfer into. -1 probes for the latest generation at "
        "which BOTH runs can still solve --min_usable_tasks tasks unaided, which is not the last "
        "one: a fully collapsed model solves nothing, and then the attack aborts on its own "
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
        default=96,
        help="verification decoding length, shared by the search and both verifications so the "
        "three verdicts are produced under identical decoding (default: 96)",
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
        help="how many tasks both runs must still solve unaided for a generation to be picked by "
        "the probe. Only read when --collapsed_generation is -1 (default: 1)",
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
        "--baseline_extra",
        "-bx",
        type=str,
        default="",
        help="extra arguments appended verbatim to both run_baseline.py invocations, e.g. "
        "\"-tbs 8 -gas 4 -fi\"",
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
