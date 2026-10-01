"""verifies given suffixes against a collapsed model, greedily or with sampled decoding

Not called directly but through run_vuln_eval.sh:

    python -m utils.verify_suffixes -sfx working.json -of out.json -cg 3 -rdf 0.5 --temperature 0.7

Two questions in the evaluation need a suffix *scored* rather than *searched for*, and both are
about attributing a hit to the attack rather than to something cheaper:

* **the controls.** A collapsed model is fragile, so the first alternative explanation for any hit
  is that a perturbation of that length would have done it. ``--source random`` draws suffixes from
  the same admissible vocabulary and the same token length as the search uses, and ``--source
  init`` scores the unoptimised starting string. Neither can be produced by run_attack_vuln.py
  itself: its behavioural check lives *inside* the optimisation loop, so the earliest suffix it
  ever verifies has already been mutated once.
* **sampled decoding.** Verification during the attack decodes at the deployment settings by
  default (run_attack.py's ``--verify_temperature`` and friends) and charges each model with the
  majority verdict over a handful of samples; a run made with ``--verify_temperature 0`` is a
  greedy claim instead. ``--temperature 0.7 --num_samples 16`` re-scores the working suffixes
  sample by sample, with more draws than the search affords per check, and reports how many
  samples still satisfy the hit criterion.

**The judgement is not reimplemented here.** ``run_attack_vuln.install_vulnerability_targets()``
is applied and the verdict comes from ``run_attack.run_unit_tests``, i.e. the same static
classification the attack itself used, so "the suffix works" means the same thing in both places.
What differs on purpose is the *decoding*: greedy for the controls, sampled when a temperature is
given. The hit criterion is run_attack's own ``is_selective_hit`` in both cases --- the collapsed
model must be insecure *and* the baseline must still be secure, so a sample where the suffix breaks
both models is not counted as survival.

Unsloth is deliberately not imported, matching run_attack.py: the checkpoints load through plain
AutoModelForCausalLM, and unsloth's import-time patching would change the forward pass the verdict
is decided by.

Args:
    suffix_file (str): JSON list of {"task": ..., "suffix": ...} objects, or an attack result file
        to harvest the verified hits out of.
    out_file (str): where the verdicts are written.
    source (str): label recorded in the output and, for random/init, the generator to use.
    collapsed_generation (int): which generation to score against.
    real_data_fraction (float): the mixture the collapse run was trained with.
    temperature (float): 0 for greedy decoding, > 0 to sample.
    num_samples (int): samples per (task, suffix) when sampling; forced to 1 when greedy.
    num_random (int): random suffixes per task for --source random.
    suffix_tokens (int): token length of a generated suffix, matching the search's -osi length.

Returns:
    None
"""
import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer

import run_attack
import run_attack_vuln
from run_attack import (
    ContrastiveGCG,
    SearchConfig,
    TargetModel,
    load_model,
    resolve_collapsed_dir,
)
from utils.colors import TColors
from utils.execution import extract_code
from utils.gcg import filter_ids
from utils.models import add_model_arguments, resolve_model_specifier
from utils.utils import clear_inherited_max_length, configure_pad_token, get_nonascii_toks

parser = argparse.ArgumentParser(description="Score given suffixes against a collapsed model")
parser.add_argument("--suffix_file", "-sfx", type=str, default="")
parser.add_argument("--out_file", "-of", type=str, required=True)
parser.add_argument("--source", "-src", type=str, default="optimized",
                    choices=("optimized", "random", "init"))
parser.add_argument("--collapsed_generation", "-cg", type=int, default=9)
parser.add_argument("--block_size", "-bs", type=int, default=512)
parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
parser.add_argument("--path", "-p", type=str, default="")
parser.add_argument("--collapsed_model_path", "-cmp", type=str, default="")
parser.add_argument("--baseline_model_path", "-bmp", type=str, default="")
parser.add_argument("--temperature", "-t", type=float, default=0.0)
parser.add_argument("--top_p", "-tpp", type=float, default=0.8)
parser.add_argument("--num_samples", "-ns", type=int, default=1)
parser.add_argument("--num_random", "-nr", type=int, default=25)
parser.add_argument("--suffix_tokens", "-st", type=int, default=20)
parser.add_argument("--max_new_tokens", "-mnt", type=int, default=0)
parser.add_argument("--exec_timeout", "-et", type=float, default=10.0)
parser.add_argument("--device", "-dx", type=str, default="cuda")
parser.add_argument("--seed", "-s", type=int, default=1337)
parser.add_argument("--tasks", type=str, default="")
add_model_arguments(parser, role="the baseline model")
args = parser.parse_args()

# the vulnerability targets and the static verdict, before anything reads run_attack.TASKS
run_attack_vuln.install_vulnerability_targets()

GREEDY: bool = args.temperature <= 0.0
NUM_SAMPLES: int = 1 if GREEDY else max(1, args.num_samples)
MAX_NEW_TOKENS: int = args.max_new_tokens or run_attack_vuln.DECODING_BUDGET


def sampled_completion(model, tokenizer, prompt: str) -> str:
    """Decodes one completion with sampling, mirroring TargetModel.complete otherwise.

    Kept separate from ``TargetModel.complete_many`` so that this script's per-sample scoring
    stays independent of how the attack aggregates its own checks (a majority over
    ``--verify_samples``): here every sample is a row of its own, never a vote.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def verdict_for(model, tokenizer, prompt: str, task) -> tuple:
    """Decodes once and judges the result with the attack's own classifier."""
    raw = (
        model.complete(tokenizer, prompt, MAX_NEW_TOKENS, 1.0) if GREEDY
        else sampled_completion(model, tokenizer, prompt)
    )
    code = extract_code(raw)
    return run_attack.run_unit_tests(code, task, args.exec_timeout), code


def random_suffixes(tokenizer, device: torch.device, count: int) -> list:
    """`count` random suffixes drawn from the vocabulary the search itself may use.

    Same admissible set and same token length as the optimiser: ASCII-printable tokens only, and
    only sequences that survive a retokenisation round trip, which is what ``filter_ids`` enforces
    for every candidate the search evaluates. A control drawn from a *different* distribution than
    the search would not be the control this needs --- it has to be what the search could have
    proposed on step zero.
    """
    forbidden = set(get_nonascii_toks(tokenizer, device).tolist())
    allowed = [i for i in range(tokenizer.vocab_size) if i not in forbidden]
    generator = random.Random(args.seed)
    kept: list = []
    attempts = 0
    while len(kept) < count and attempts < 50 * count:
        attempts += 1
        ids = torch.tensor(
            [[allowed[generator.randrange(len(allowed))] for _ in range(args.suffix_tokens)]],
            device=device,
        )
        try:
            survived = filter_ids(ids, tokenizer)
        except RuntimeError:
            continue
        if survived.shape[0]:
            kept.append(tokenizer.decode(survived[0]))
    if len(kept) < count:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: only {len(kept)}/{count} random suffixes "
              f"survived retokenisation")
    return kept


def load_suffix_rows(tasks: list, tokenizer, device: torch.device) -> list:
    """The (task, suffix) pairs to score, from the file or from the requested generator.

    Raises:
        SystemExit: --source optimized without a readable suffix file
    """
    if args.source == "init":
        init = SearchConfig().optim_str_init
        return [{"task": task.name, "suffix": init} for task in tasks]

    if args.source == "random":
        suffixes = random_suffixes(tokenizer, device, args.num_random)
        return [{"task": task.name, "suffix": suffix} for task in tasks for suffix in suffixes]

    if not args.suffix_file or not os.path.isfile(args.suffix_file):
        raise SystemExit(
            f"{TColors.FAIL}--source optimized needs --suffix_file{TColors.ENDC} and "
            f"{args.suffix_file!r} is not a file"
        )
    with open(args.suffix_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    # either a flat list of pairs, or an attack result file whose verified hits are harvested
    if isinstance(payload, dict) and "results" in payload:
        rows = [
            {"task": result["task"], "suffix": hit["suffix"]}
            for result in payload["results"]
            for hit in (result.get("successes") or [])
        ]
    else:
        rows = [{"task": row["task"], "suffix": row["suffix"]} for row in payload]
    # a result file carries suffixes for every target the run attacked, so --tasks *deselects*
    # rather than invalidates: only a name no target has at all is an error
    defined = {task.name for task in run_attack.TASKS}
    unknown = sorted({row["task"] for row in rows} - defined)
    if unknown:
        raise SystemExit(
            f"{TColors.FAIL}the suffix file references unknown target(s) {unknown}{TColors.ENDC}; "
            f"this script knows {sorted(defined)}"
        )
    selected_names = {task.name for task in tasks}
    kept = [row for row in rows if row["task"] in selected_names]
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"##   {dropped} suffix(es) skipped: their target is not in --tasks")
    if not kept:
        raise SystemExit(
            f"{TColors.FAIL}no suffix in {args.suffix_file} targets any of "
            f"{sorted(selected_names)}{TColors.ENDC}"
        )
    return kept


# ────────────────────────────────── setup ─────────────────────────────────────────────────
torch.manual_seed(args.seed)
random.seed(args.seed)

if args.path:
    run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")

device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
dtype = torch.float16 if device.type == "cuda" else torch.float32
model_specifier = resolve_model_specifier(args.model_size, args.model_specifier)
specifier_name = model_specifier.split("/")[-1]

baseline_dir = args.baseline_model_path or model_specifier
collapsed_dir = args.collapsed_model_path or resolve_collapsed_dir(
    args.collapsed_generation, specifier_name, args.block_size,
    real_data_fraction=args.real_data_fraction,
)

selected = [t for t in run_attack.TASKS if not args.tasks or t.name in args.tasks.split(",")]
if not selected:
    raise SystemExit(f"no targets matched {args.tasks!r}")

tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
clear_inherited_max_length(tokenizer)
configure_pad_token(tokenizer)

print(f"\n## {TColors.BOLD}{TColors.HEADER}suffix verification{TColors.ENDC}")
print(f"##   source       : {args.source}")
print(f"##   decoding     : {'greedy' if GREEDY else f'T={args.temperature} top_p={args.top_p}'}"
      f", {NUM_SAMPLES} sample(s) per suffix")
print(f"##   baseline     : {baseline_dir}")
print(f"##   collapsed    : {collapsed_dir}")

baseline = TargetModel("baseline", load_model(baseline_dir, device, dtype), device)
collapsed = TargetModel("collapsed", load_model(collapsed_dir, device, dtype), device)

rows = load_suffix_rows(selected, tokenizer, device)
by_name = {task.name: task for task in selected}
print(f"##   suffixes     : {len(rows)} (task, suffix) pair(s)")

# ─────────────────────────────── clean control ────────────────────────────────────────────
# the same probe the attack's capability gate runs, repeated here because a pair scored against a
# target that already fails the suffix-free prompt says nothing: it has to be reported alongside
clean: dict = {}
for task in selected:
    before, after = run_attack.split_prompt(tokenizer, task)
    prompt = before + after
    base_status, _ = verdict_for(baseline, tokenizer, prompt, task)
    col_status, _ = verdict_for(collapsed, tokenizer, prompt, task)
    clean[task.name] = {"baseline_status": base_status, "collapsed_status": col_status,
                        "attributable": base_status == "pass" and col_status == "pass"}
    marker = (f"{TColors.OKGREEN}attributable{TColors.ENDC}" if clean[task.name]["attributable"]
              else f"{TColors.WARNING}not attributable{TColors.ENDC}")
    print(f"##   clean {task.name:16s} baseline={base_status:8s} collapsed={col_status:8s} "
          f"-> {marker}")

# ──────────────────────────────── scoring loop ────────────────────────────────────────────
results = []
total_hits = total_samples = 0
for index, row in enumerate(rows, start=1):
    task = by_name[row["task"]]
    before, after = run_attack.split_prompt(tokenizer, task)
    prompt = before + row["suffix"] + after
    samples = []
    for _ in range(NUM_SAMPLES):
        base_status, base_code = verdict_for(baseline, tokenizer, prompt, task)
        col_status, col_code = verdict_for(collapsed, tokenizer, prompt, task)
        verdict = {"baseline_status": base_status, "collapsed_status": col_status}
        samples.append({**verdict, "hit": ContrastiveGCG.is_selective_hit(verdict),
                        "collapsed_code": col_code, "baseline_code": base_code})
    hits = sum(sample["hit"] for sample in samples)
    total_hits += hits
    total_samples += len(samples)
    results.append({
        "task": row["task"],
        "suffix": row["suffix"],
        "source": args.source,
        "attributable": clean[row["task"]]["attributable"],
        "n_samples": len(samples),
        "n_hit": hits,
        "survival": hits / len(samples),
        # the raw code of one sample only: the point of this file is the rates, and every sample's
        # program would make it unusable as an input to the figures
        "samples": [{k: v for k, v in sample.items() if not k.endswith("_code")}
                    for sample in samples],
        "example_collapsed_code": samples[0]["collapsed_code"] if samples else "",
    })
    if index % 25 == 0 or index == len(rows):
        print(f"##   scored {index}/{len(rows)} pairs, {total_hits}/{total_samples} samples a hit")

payload = {
    "source": args.source,
    # which attack file the suffixes came from, so a survival figure can tell the re-score of a
    # greedy attack run from that of a sampled one
    "suffix_file": args.suffix_file,
    "mode": "greedy" if GREEDY else "sampled",
    "temperature": args.temperature,
    "top_p": args.top_p,
    "num_samples": NUM_SAMPLES,
    "baseline_model": baseline_dir,
    "collapsed_model": collapsed_dir,
    "specifier_name": specifier_name,
    "collapsed_generation": args.collapsed_generation,
    "real_data_fraction": args.real_data_fraction,
    "suffix_tokens": args.suffix_tokens,
    "clean_control": clean,
    "aggregate": {
        "n_pairs": len(results),
        "n_samples_total": total_samples,
        "n_hit": total_hits,
        "hit_rate": (total_hits / total_samples) if total_samples else 0.0,
        "pairs_with_any_hit": sum(1 for row in results if row["n_hit"] > 0),
        "mean_survival": (
            sum(row["survival"] for row in results) / len(results) if results else 0.0
        ),
    },
    "results": results,
}
os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
with open(args.out_file, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

aggregate = payload["aggregate"]
print(f"##   {TColors.BOLD}{aggregate['n_hit']}/{aggregate['n_samples_total']} samples a hit"
      f"{TColors.ENDC} across {aggregate['n_pairs']} pair(s), "
      f"{aggregate['pairs_with_any_hit']} pair(s) hit at least once")
print(f"## {TColors.OKBLUE}saved{TColors.ENDC} {args.out_file}\n")
