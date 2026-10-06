"""measures the *security margin* of every collapse generation, with no attack involved

The attack's two verdicts — greedy and sampled — are two thresholds on one underlying quantity:
how strongly the model prefers the secure sink over the insecure one at the token where the two
implementations part ways. This script measures that quantity directly, per generation and per
target, on the clean prompt and (optionally) under the suffixes an attack run found, so that the
greedy/sampled gap can be *predicted* from a decoding-independent number instead of observed twice.

Three numbers per (model, target, prompt):

    decision_margin   log p(secure token) - log p(insecure token) at the first position where the
                      tokenised secure and insecure implementations differ, teacher-forced on the
                      shared prefix. Greedy decoding of that prefix emits the secure token iff this
                      is > 0 *and* the secure token is the argmax (recorded as `secure_is_argmax`).
                      A majority at temperature T needs it above roughly 0 as well, but with the
                      rest of the mass counted — hence the next number.
    sequence_margin   log p(secure code) - log p(insecure code), summed over the whole target: the
                      sequence-level log-odds, which is what a sampled decode draws from.
    entropy           mean next-token entropy (nats) over the secure target's positions, teacher
                      forced. A decoding-independent measure of how far the generation has
                      collapsed: it is the sharpening the collapse map models, read off directly.

Why this is worth a script of its own
-------------------------------------
* It needs no search and runs on one GPU in minutes, so it can be computed for every generation,
  mixture and collapse-decoding setting the attack cannot afford to sweep.
* It separates "collapse moved the margin" from "the suffix moved the margin": the clean margin
  per generation is the former, the suffix margin minus the clean margin is the latter.
* With ``--suffix_file`` it checks the prediction that links the two verdicts: a greedy hit should
  have a negative decision margin on the collapsed model and a positive one on the baseline; a
  sampled majority hit should have a sequence margin well below zero.

Usage
-----
    python run_margin.py -msz 0.5b -rdf 0.8 -n 9 -p ./runs/x
    python run_margin.py -msz 0.5b -rdf 0.8 -n 9 -p ./runs/x \\
        --suffix_file attack_results/attack_gen2_..._vuln_T0.7p0.8k20x5.json
    python run_margin.py --plot attack_results/margin_*.json -pp ./plots

Writes ``attack_results/margin_{name}{mixture}.json`` (or ``--out_file``) and, with ``--plot``,
draws margin and entropy against generation plus, where attack results exist in the same
directory, hit rate against entropy.
"""
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import run_attack
import run_attack_vuln
from run_attack import TargetModel, load_model, resolve_collapsed_dir
from utils.colors import TColors
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import mixture_tag
from utils.utils import clear_inherited_max_length, configure_pad_token

RESULTS_PATH: str = "./attack_results/"


# ──────────────────────────────── measurement ─────────────────────────────────────────────
def _ids(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def _logprobs(model, prompt_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of every target token given the prompt and the preceding targets.

    Returns:
        Tensor: (n_target, vocab) log-softmax rows, row i predicting target token i
    """
    ids = torch.cat([prompt_ids, target_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    n = target_ids.shape[1]
    return F.log_softmax(logits[-n - 1 : -1].float(), dim=-1)


def _common_prefix(a: list, b: list) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def measure(model: TargetModel, tokenizer, prompt: str, task, device: torch.device) -> dict:
    """The three margin numbers of one model on one prompt for one target."""
    prompt_ids = _ids(tokenizer, prompt, device)
    secure = tokenizer(task.correct_code, add_special_tokens=False)["input_ids"]
    insecure = tokenizer(task.wrong_code, add_special_tokens=False)["input_ids"]
    k = _common_prefix(secure, insecure)
    if k == len(secure) or k == len(insecure):
        # one implementation is a prefix of the other; the decision is then at the position
        # where the longer one continues and the shorter one would emit end-of-turn. Fall back to
        # the sequence margin only
        decision = None
    else:
        # one forward over prompt + shared prefix + the secure token; row k predicts token k
        rows = _logprobs(model.model, prompt_ids, torch.tensor([secure[: k + 1]], device=device))
        row = rows[k]
        decision = {
            "position": k,
            "secure_token": tokenizer.decode([secure[k]]),
            "insecure_token": tokenizer.decode([insecure[k]]),
            "logp_secure": float(row[secure[k]]),
            "logp_insecure": float(row[insecure[k]]),
            "decision_margin": float(row[secure[k]] - row[insecure[k]]),
            "secure_is_argmax": bool(int(row.argmax()) == secure[k]),
            "argmax_token": tokenizer.decode([int(row.argmax())]),
        }

    sec_rows = _logprobs(model.model, prompt_ids, torch.tensor([secure], device=device))
    ins_rows = _logprobs(model.model, prompt_ids, torch.tensor([insecure], device=device))
    logp_secure = float(sec_rows[torch.arange(len(secure)), torch.tensor(secure)].sum())
    logp_insecure = float(ins_rows[torch.arange(len(insecure)), torch.tensor(insecure)].sum())
    entropy = float(-(sec_rows.exp() * sec_rows).sum(dim=-1).mean())
    return {
        "sequence_margin": logp_secure - logp_insecure,
        "logp_secure_sequence": logp_secure,
        "logp_insecure_sequence": logp_insecure,
        "secure_tokens": len(secure),
        "insecure_tokens": len(insecure),
        "entropy": entropy,
        **(decision or {"decision_margin": None, "secure_is_argmax": None}),
    }


def load_hits(path: str) -> list:
    """(task, suffix, record) triples of every verified hit in an attack result file."""
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    rows = []
    for result in report.get("results") or []:
        for hit in result.get("successes") or []:
            rows.append((result["task"], hit["suffix"], hit))
        for hit in result.get("weak_hits") or []:
            rows.append((result["task"], hit["suffix"], {**hit, "weak": True}))
    return rows


# ───────────────────────────────── plotting ───────────────────────────────────────────────
def plot(files: list, plots_path: str, usetex: bool) -> None:
    """Margin and entropy against generation, and hit rate against entropy where attacks exist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.plotting import apply_perplexity_style, save_figure
    from utils.vuln_results import load_records

    apply_perplexity_style(usetex, font_size=16)
    payloads = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            payloads.append(json.load(handle))

    for payload in payloads:
        gens = sorted(int(g) for g in payload["generations"])
        targets = payload["targets"]
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
        for target in targets:
            dm = [payload["generations"][str(g)]["clean"][target].get("decision_margin") for g in gens]
            sm = [payload["generations"][str(g)]["clean"][target]["sequence_margin"] for g in gens]
            axes[0].plot(gens, [v if v is not None else float("nan") for v in dm], marker="o",
                         label=target)
            axes[1].plot(gens, sm, marker="o", label=target)
        ent = [sum(payload["generations"][str(g)]["clean"][t]["entropy"] for t in targets)
               / len(targets) for g in gens]
        axes[2].plot(gens, ent, marker="s", color="black")
        for axis, title in zip(axes, ("decision margin (nats)", "sequence margin (nats)",
                                       "mean entropy (nats/token)")):
            axis.set_xlabel("collapse generation")
            axis.set_title(title)
            axis.axhline(0, color="grey", linewidth=0.8)
        axes[0].legend(fontsize=9)
        stem = os.path.join(plots_path, f"margin_{payload['specifier_name']}"
                            f"{mixture_tag(payload['real_data_fraction'])}")
        save_figure(figure, stem)
        print(f"##   -> {stem}.pdf")

        # hit rate against entropy: one point per attack result of the same model and mixture
        results_dir = os.path.dirname(os.path.abspath(payload["path"]))
        records = load_records(results_dir, specifier_name=payload["specifier_name"],
                               real_data_fraction=payload["real_data_fraction"])
        if not records:
            continue
        figure, axis = plt.subplots(figsize=(7, 5))
        for method, marker in (("none", "o"), ("logit", "^")):
            for tag, color in (("", "tab:blue"), (None, "tab:red")):
                # greedy against sampled verification; anchor-held and two-sided runs are drawn
                # together here, with the anchor rule noted in the label when both exist
                sel = [r for r in records if r.surrogate_method == method
                       and str(r.generation) in payload["generations"]
                       and ((r.verification_tag == "") if tag == "" else (r.verification_tag != ""))]
                if not sel:
                    continue
                xs = [sum(payload["generations"][str(r.generation)]["clean"][t]["entropy"]
                          for t in targets) / len(targets) for r in sel]
                ys = [r.hit_rate for r in sel]
                label = f"{'direct' if method == 'none' else 'surrogate'}, " \
                        f"{'greedy' if tag == '' else 'sampled'} verification"
                axis.scatter(xs, ys, marker=marker, color=color, label=label)
                for r, x, y in zip(sel, xs, ys):
                    if not math.isnan(y):
                        axis.annotate(str(r.generation), (x, y), fontsize=8,
                                      xytext=(3, 3), textcoords="offset points")
        axis.set_xlabel("mean entropy on the clean prompts (nats/token)")
        axis.set_ylabel("hit rate over attackable targets")
        axis.set_title("attack success against collapse severity")
        axis.legend(fontsize=9)
        stem = os.path.join(plots_path, f"severity_{payload['specifier_name']}"
                            f"{mixture_tag(payload['real_data_fraction'])}")
        save_figure(figure, stem)
        print(f"##   -> {stem}.pdf")


# ─────────────────────────────────── main ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Security margin of collapsed code models")
    parser.add_argument("--generations", "-n", type=int, default=9,
                        help="measure generations 0..N (default: 9)")
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
    parser.add_argument("--path", "-p", type=str, default="",
                        help="run root holding model_outputs/ and attack_results/")
    parser.add_argument("--baseline_model_path", "-bmp", type=str, default="")
    parser.add_argument("--suffix_file", "-sfx", type=str, nargs="*", default=[],
                        help="attack result file(s) whose verified hits are re-measured with "
                        "their suffix, on the generation they were found at")
    parser.add_argument("--tasks", "-t", type=str, default="")
    parser.add_argument("--out_file", "-of", type=str, default="")
    parser.add_argument("--device", "-dx", type=str, default="cuda")
    parser.add_argument("--plot", type=str, nargs="*", default=None,
                        help="draw figures from these margin JSONs instead of measuring")
    parser.add_argument("--plots_path", "-pp", type=str, default="./plots")
    parser.add_argument("--no_usetex", dest="usetex", action="store_false")
    add_model_arguments(parser, role="the baseline model")
    args = parser.parse_args()

    if args.plot is not None:
        files = args.plot or sorted(glob.glob(os.path.join(RESULTS_PATH, "margin_*.json")))
        if not files:
            raise SystemExit("no margin files to plot")
        plot(files, args.plots_path, args.usetex)
        return

    run_attack_vuln.install_vulnerability_targets()
    results_path = RESULTS_PATH
    if args.path:
        run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")
        results_path = os.path.join(args.path, "attack_results/")
    os.makedirs(results_path, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_specifier = resolve_model_specifier(args.model_size, args.model_specifier)
    specifier_name = model_specifier.split("/")[-1]
    baseline_dir = args.baseline_model_path or model_specifier
    tasks = [t for t in run_attack.TASKS if not args.tasks or t.name in args.tasks.split(",")]
    if not tasks:
        raise SystemExit(f"no targets matched {args.tasks!r}")

    tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
    clear_inherited_max_length(tokenizer)
    configure_pad_token(tokenizer)
    prompts = {t.name: run_attack.split_prompt(tokenizer, t) for t in tasks}

    hits_by_generation: dict = {}
    for path in args.suffix_file:
        with open(path, encoding="utf-8") as handle:
            generation = int(json.load(handle).get("collapsed_generation", -1))
        hits_by_generation.setdefault(generation, []).extend(
            (path, task, suffix, record) for task, suffix, record in load_hits(path)
        )

    out_file = args.out_file or os.path.join(
        results_path, f"margin_{specifier_name}{mixture_tag(args.real_data_fraction)}.json"
    )
    print(f"\n## {TColors.BOLD}{TColors.HEADER}security margin{TColors.ENDC}")
    print(f"##   baseline     : {baseline_dir}")
    print(f"##   generations  : 0..{args.generations}, mixture rdf {args.real_data_fraction:g}")
    print(f"##   targets      : {', '.join(t.name for t in tasks)}")
    print(f"##   suffix files : {len(args.suffix_file)} ({sum(len(v) for v in hits_by_generation.values())} hits)")
    print(f"##   output       : {out_file}")

    payload = {
        "baseline_model": baseline_dir,
        "specifier_name": specifier_name,
        "real_data_fraction": args.real_data_fraction,
        "block_size": args.block_size,
        "targets": [t.name for t in tasks],
        "path": out_file,
        "baseline": {},
        "generations": {},
    }

    def measure_model(label: str, model_dir: str, generation: int | None) -> dict:
        model = TargetModel(label, load_model(model_dir, device, dtype), device)
        clean = {}
        for task in tasks:
            before, after = prompts[task.name]
            clean[task.name] = measure(model, tokenizer, before + after, task, device)
        suffixed = []
        for path, task_name, suffix, record in hits_by_generation.get(generation, []):
            task = next(t for t in tasks if t.name == task_name)
            before, after = prompts[task_name]
            suffixed.append({
                "task": task_name,
                "suffix": suffix,
                "source": os.path.basename(path),
                "weak": bool(record.get("weak", False)),
                "collapsed_wrong_rate": record.get("collapsed_wrong_rate"),
                "col_wrong_loss": record.get("col_wrong"),
                **measure(model, tokenizer, before + suffix + after, task, device),
            })
        header = f"{label:10s}"
        for task in tasks:
            m = clean[task.name]
            dm = "   n/a " if m["decision_margin"] is None else f"{m['decision_margin']:+7.2f}"
            print(f"##   {header} {task.name:16s} decision {dm}  sequence "
                  f"{m['sequence_margin']:+8.2f}  entropy {m['entropy']:.3f}"
                  + ("" if m["secure_is_argmax"] in (True, None) else
                     f"  {TColors.WARNING}argmax={m['argmax_token']!r}{TColors.ENDC}"))
            header = " " * 10
        for row in suffixed:
            dm = "n/a" if row["decision_margin"] is None else f"{row['decision_margin']:+.2f}"
            print(f"##   {' ' * 10} {row['task']:16s} with suffix: decision {dm} sequence "
                  f"{row['sequence_margin']:+.2f}"
                  + (f" wrong_rate {row['collapsed_wrong_rate']:.2f}"
                     if row["collapsed_wrong_rate"] is not None else ""))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"model": model_dir, "clean": clean, "suffixed": suffixed}

    # the baseline is measured under every generation's suffixes too, since a hit is a *pair*
    # of verdicts and the prediction to check is that the baseline's margin stays positive
    base_payload = measure_model("baseline", baseline_dir, None)
    payload["baseline"] = base_payload
    for generation in range(args.generations + 1):
        try:
            model_dir = resolve_collapsed_dir(
                generation, specifier_name, args.block_size,
                real_data_fraction=args.real_data_fraction,
            )
        except FileNotFoundError as exc:
            print(f"##   {TColors.WARNING}generation {generation} skipped{TColors.ENDC}: {exc}")
            continue
        payload["generations"][str(generation)] = measure_model(
            f"gen {generation}", model_dir, generation
        )
        if generation in hits_by_generation:
            # the baseline side of each hit, measured once the generation is known
            base = TargetModel("baseline", load_model(baseline_dir, device, dtype), device)
            for row in payload["generations"][str(generation)]["suffixed"]:
                task = next(t for t in tasks if t.name == row["task"])
                before, after = prompts[row["task"]]
                b = measure(base, tokenizer, before + row["suffix"] + after, task, device)
                row["baseline_decision_margin"] = b["decision_margin"]
                row["baseline_sequence_margin"] = b["sequence_margin"]
            del base
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with open(out_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print(f"\n## {TColors.OKGREEN}written{TColors.ENDC}: {out_file}")


if __name__ == "__main__":
    main()
