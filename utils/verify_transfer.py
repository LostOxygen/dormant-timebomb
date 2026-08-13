"""
Helper module to check whether an adversarial suffix found against one collapse run still elicits
wrong code from a *different* collapse run. This is not meant to be called directly but via
run_transfer_experiment.py as a subprocess instead:

    python -m utils.verify_transfer --suffix_file sfx.json --out_file out.json -cg 9 -p ./runs/B

These are modules, not scripts: they are launched as `python -m utils.<name>` from the repo root,
because they import sibling helpers with `from utils.X import Y` and running the file directly
puts utils/ on sys.path instead of the root, so `utils` is then not a package at all.

The verdict is not recomputed here. `run_attack.ContrastiveGCG.verify` and its
`is_selective_hit` are imported and called directly, so "the suffix works" means exactly what it
means during the attack: the model under attack emits code that fails the task's unit tests while
the pristine baseline still passes them. Reimplementing that here would let the two definitions
drift, and a transfer result is only interesting if it is the *same* question asked of a
different model.

Unsloth is deliberately not imported, matching run_attack.py: the checkpoints are loaded through
plain AutoModelForCausalLM, and unsloth's import-time patching of transformers would change the
forward pass the verdict is decided by.

Three verdicts are possible per (task, suffix), and the third one is the reason this file exists
rather than a grep over the attack's JSON:

    transferred     the target run's collapsed model breaks and its baseline still passes
    held            the target run's collapsed model answers correctly despite the suffix
    inconclusive    the target's collapsed model already fails the task with *no* suffix at all,
                    so it cannot be said to have been broken by anything. This is the trap in a
                    cross-run transfer experiment: a sufficiently collapsed model fails
                    everything, which would otherwise be scored as a 100% transfer rate

The clean-prompt control that separates `held` from `inconclusive` is run once per task, not once
per suffix, since it does not depend on the suffix.

Args:
    suffix_file (str): JSON list of {"task": ..., "suffix": ...} objects to test.
    out_file (str): where the verdicts are written.
    collapsed_generation (int): which generation of the target run to test against.
    block_size (int): effective block size baked into the checkpoint names; auto-detected if 0.
    model_specifier (str): the pristine baseline model.
    path (str): root of the *target* run, i.e. the one holding model_outputs/.
    collapsed_model_path (str): explicit target checkpoint, overriding the resolution.
    baseline_model_path (str): explicit baseline checkpoint.
    max_new_tokens (int): decoding length cap of the verification.
    repetition_penalty (float): verification decoding penalty. 1.0 is plain greedy decoding.
    exec_timeout (float): per-task unit test timeout.
    label (str): free-form name of the target run, copied into the output for bookkeeping.

Returns:
    None
"""
import os
import json
import argparse

import torch
from transformers import AutoTokenizer

import run_attack
from run_attack import (
    TASKS,
    WRONG_STATUSES,
    ContrastiveGCG,
    SearchConfig,
    TargetModel,
    load_model,
    resolve_collapsed_dir,
)
from utils.colors import TColors
from utils.utils import configure_pad_token

parser = argparse.ArgumentParser(description="Cross-run adversarial suffix verification")
parser.add_argument("--suffix_file", "-sf", type=str, default="")
parser.add_argument("--out_file", "-of", type=str, required=True)
# probe mode reports only the clean-prompt capability of this run's generation, with no suffixes
# involved. It exists because the attackable window is narrow and moves between runs: past a
# certain generation the collapsed model solves nothing unaided, and then neither an attack nor a
# transfer verdict can mean anything. The clean control this file computes anyway *is* the
# capability probe, so the same code answers both questions
parser.add_argument("--probe_only", "-po", action="store_true")
parser.add_argument("--tasks", "-t", type=str, default="")
parser.add_argument("--collapsed_generation", "-cg", type=int, default=9)
# 0 rather than None: this is passed through from a subprocess command line, where "None" would
# arrive as the string "None". resolve_collapsed_dir wants None to mean "glob for it"
parser.add_argument("--block_size", "-bs", type=int, default=0)
parser.add_argument(
    "--model_specifier", "-ms", type=str, default="unsloth/Qwen2.5-Coder-0.5B-Instruct"
)
parser.add_argument("--path", "-p", type=str, default="")
parser.add_argument("--collapsed_model_path", "-cmp", type=str, default="")
parser.add_argument("--baseline_model_path", "-bmp", type=str, default="")
parser.add_argument("--max_new_tokens", "-mnt", type=int, default=96)
parser.add_argument("--repetition_penalty", "-rp", type=float, default=1.0)
parser.add_argument("--exec_timeout", "-et", type=float, default=10.0)
parser.add_argument("--label", "-l", type=str, default="target")
# the --real_data_fraction the target run was collapsed with; part of its checkpoint names from
# generation 1 onward, so it is needed to find them
parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
args = parser.parse_args()

# run_attack.resolve_collapsed_dir reads the module global rather than taking a root, so the same
# rebinding run_attack.main() does has to happen here before it is called. Only MODEL_PATH: the
# base model is no longer a module global over there, it is passed to the functions that need it,
# and nothing this file calls resolves it from module scope
if args.path != "":
    run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")
specifier_name = args.model_specifier.split("/")[-1]

collapsed_dir = args.collapsed_model_path or resolve_collapsed_dir(
    args.collapsed_generation,
    specifier_name,
    args.block_size or None,
    real_data_fraction=args.real_data_fraction,
)
baseline_dir = args.baseline_model_path or args.model_specifier

if torch.cuda.is_available():
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    print(
        f"## {TColors.WARNING}CUDA is not available, verifying on CPU. This will be "
        f"slow{TColors.ENDC}"
    )
    device = torch.device("cpu", 0)
    dtype = torch.float32

print(
    f"## {TColors.OKBLUE}{TColors.BOLD}Verifying against{TColors.ENDC} {args.label}: "
    f"{collapsed_dir}"
)

tokenizer = configure_pad_token(AutoTokenizer.from_pretrained(args.model_specifier))
baseline = TargetModel("baseline", load_model(baseline_dir, device, dtype), device)
collapsed = TargetModel(
    "collapsed", load_model(collapsed_dir, device, dtype, base_for_adapter=baseline_dir), device
)

# only the verification-relevant fields matter; nothing here runs the optimizer. surrogate is left
# unset so _verified_models() decodes exactly the two real models
cfg = SearchConfig(
    max_new_tokens=args.max_new_tokens,
    repetition_penalty=args.repetition_penalty,
    exec_timeout=args.exec_timeout,
    no_exec=False,
)
harness = ContrastiveGCG(
    baseline=baseline, collapsed=collapsed, tokenizer=tokenizer, cfg=cfg, surrogate=None
)

tasks_by_name = {task.name: task for task in TASKS}

if args.probe_only:
    selected = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(tasks_by_name)
    candidates = [{"task": name, "suffix": None} for name in selected]
else:
    with open(args.suffix_file, "r", encoding="utf-8") as handle:
        candidates = json.load(handle)
unknown = sorted({c["task"] for c in candidates} - set(tasks_by_name))
if unknown:
    raise ValueError(
        f"the suffix file references task(s) {unknown} that are not in run_attack.TASKS. The "
        f"suffixes were produced by a different version of the task list, so the unit tests they "
        f"were validated against are not the ones that would be run here"
    )

records = []
# one clean-prompt control per task, not per suffix: it does not depend on the suffix, and
# decoding both models is the expensive part
controls: dict[str, dict] = {}

for candidate in candidates:
    task = tasks_by_name[candidate["task"]]
    before_str, after_str = harness.split_prompt(task)

    if task.name not in controls:
        controls[task.name] = harness.verify(task, before_str, after_str, "")
    control = controls[task.name]

    if args.probe_only:
        capable = control["collapsed_status"] == "pass"
        records.append(
            {
                "task": task.name,
                "capable": capable,
                "clean_baseline_status": control["baseline_status"],
                "clean_collapsed_status": control["collapsed_status"],
            }
        )
        colour = TColors.OKGREEN if capable else TColors.WARNING
        print(
            f"##   [{task.name}] {colour}{'CAPABLE' if capable else 'INCAPABLE'}{TColors.ENDC} "
            f"clean=({control['baseline_status']}/{control['collapsed_status']})"
        )
        continue

    verdict = harness.verify(task, before_str, after_str, candidate["suffix"])
    hit = harness.is_selective_hit(verdict)

    # the target has to be able to solve the task unaided, otherwise "it emitted wrong code" says
    # nothing about the suffix. This mirrors run_attack.capability_gate, where "capable" likewise
    # means the tests pass rather than merely not erroring
    target_capable = control["collapsed_status"] == "pass"
    baseline_capable = control["baseline_status"] == "pass"

    if not baseline_capable:
        outcome = "inconclusive"
        reason = (
            f"the pristine baseline does not solve this task unaided "
            f"(clean baseline status: {control['baseline_status']}), so the selective-hit "
            f"criterion cannot be evaluated"
        )
    elif not target_capable:
        outcome = "inconclusive"
        reason = (
            f"this run's generation-{args.collapsed_generation} model already fails the task "
            f"with no suffix at all (clean collapsed status: {control['collapsed_status']}), so "
            f"nothing the suffix does can be attributed to the suffix"
        )
    elif hit:
        outcome = "transferred"
        reason = "collapsed emitted failing code while the baseline still passed"
    else:
        outcome = "held"
        reason = (
            f"the suffix did not break this run's model "
            f"(collapsed status: {verdict['collapsed_status']})"
        )

    records.append(
        {
            "task": task.name,
            "suffix": candidate["suffix"],
            "origin": candidate.get("origin", ""),
            "outcome": outcome,
            "reason": reason,
            "selective_hit": hit,
            "target_capable_clean": target_capable,
            "baseline_capable_clean": baseline_capable,
            "clean_baseline_status": control["baseline_status"],
            "clean_collapsed_status": control["collapsed_status"],
            "baseline_status": verdict["baseline_status"],
            "collapsed_status": verdict["collapsed_status"],
            "collapsed_code": verdict["collapsed_code"],
            "baseline_code": verdict["baseline_code"],
        }
    )

    colour = {
        "transferred": TColors.OKGREEN,
        "held": TColors.OKBLUE,
        "inconclusive": TColors.WARNING,
    }[outcome]
    print(
        f"##   [{task.name}] {colour}{TColors.BOLD}{outcome.upper()}{TColors.ENDC} "
        f"clean=({control['baseline_status']}/{control['collapsed_status']}) "
        f"attacked=({verdict['baseline_status']}/{verdict['collapsed_status']}) "
        f"suffix={candidate['suffix']!r}"
    )

if args.probe_only:
    usable = [r["task"] for r in records if r["capable"]]
    probe = {
        "label": args.label,
        "collapsed_model": collapsed_dir,
        "collapsed_generation": args.collapsed_generation,
        "n_tasks": len(records),
        "n_capable": len(usable),
        "capable_fraction": len(usable) / len(records) if records else 0.0,
        "usable": usable,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as handle:
        json.dump({"probe": probe, "records": records}, handle, indent=2)
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}{args.label}{TColors.ENDC}: "
        f"{len(usable)}/{len(records)} task(s) still solvable unaided "
        f"({probe['capable_fraction']:.0%}) -> {args.out_file}"
    )
    raise SystemExit(0)

# the denominator deliberately excludes the inconclusive ones. A transfer rate computed over
# suffixes whose target could not solve the task anyway is not a transfer rate
n_conclusive = sum(1 for r in records if r["outcome"] != "inconclusive")
n_transferred = sum(1 for r in records if r["outcome"] == "transferred")
summary = {
    "label": args.label,
    "collapsed_model": collapsed_dir,
    "baseline_model": baseline_dir,
    "collapsed_generation": args.collapsed_generation,
    "n_candidates": len(records),
    "n_conclusive": n_conclusive,
    "n_transferred": n_transferred,
    "n_held": sum(1 for r in records if r["outcome"] == "held"),
    "n_inconclusive": sum(1 for r in records if r["outcome"] == "inconclusive"),
    "transfer_rate": (n_transferred / n_conclusive) if n_conclusive else None,
    "wrong_statuses": list(WRONG_STATUSES),
    "verification": {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "exec_timeout": args.exec_timeout,
    },
}

os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
with open(args.out_file, "w", encoding="utf-8") as handle:
    json.dump({"summary": summary, "records": records}, handle, indent=2)

rate = "n/a" if summary["transfer_rate"] is None else f"{summary['transfer_rate']:.0%}"
print(
    f"## {TColors.OKBLUE}{TColors.BOLD}{args.label}{TColors.ENDC}: "
    f"{n_transferred}/{n_conclusive} transferred ({rate}), "
    f"{summary['n_inconclusive']} inconclusive -> {args.out_file}"
)
