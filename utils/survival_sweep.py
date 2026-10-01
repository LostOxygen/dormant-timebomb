"""re-scores an attack run's suffixes across decoding temperatures, as a curve rather than a point

Two curves, one script, because both are the same measurement — decode a (task, suffix) pair many
times at a given temperature on both models and count how often the pair is still a selective hit
— taken along two different axes:

    --source hits         survival of the run's verified hits against temperature. The deployment
                          question: how much of the greedy (or majority) claim is left at the
                          decoding a user would actually run.
    --source trajectory   survival of the suffix the search held at every k-th step, at one or
                          more temperatures, against the step and against the search's own
                          wrong-code loss at that step. The cost question: how many more steps
                          past the first argmax flip does it take to reach a given survival, i.e.
                          what sampled decoding costs the attacker rather than whether it stops
                          them. Needs the run's `history` (always written by run_attack.py).

A pair survives a sample when the collapsed model's completion is insecure *and* the baseline's,
decoded on the same prompt, is still secure — `ContrastiveGCG.is_selective_hit` on one sample pair,
exactly the attack's own criterion. Samples are drawn in one batched `generate()` per model via
`TargetModel.complete_many`, and temperature 0 is the greedy single decode, so the greedy claim is
one point on the same curve.

This is not utils/verify_suffixes.py, which scores one decoding setting and feeds the controls and
survival phases of run_vuln_eval.sh; this script sweeps the setting and reads the trajectory, and
writes one JSON per attack result file. Outputs
``attack_results/survival_{source}_{attack file stem}.json`` by default and, with ``--plot``, the
figure next to it.

Usage
-----
    python -m utils.survival_sweep -rf attack_results/attack_gen2_..._vuln_T0.7p0.8k20x5_logit_surrogate.json \\
        -msz 0.5b -rdf 0.5 -p . --temperatures 0,0.3,0.5,0.7,1.0 -ns 16 --plot
    python -m utils.survival_sweep -rf attack_results/attack_gen2_..._vuln.json -msz 0.5b -rdf 0.5 \\
        --source trajectory --every 10 --restarts 0 --temperatures 0.7 -ns 16 --plot
"""
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

import run_attack
import run_attack_vuln
from run_attack import ContrastiveGCG, Decoding, TargetModel, load_model, resolve_collapsed_dir
from utils.colors import TColors
from utils.execution import extract_code
from utils.models import add_model_arguments, resolve_model_specifier
from utils.utils import clear_inherited_max_length, configure_pad_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Survival of attack suffixes across temperatures")
    parser.add_argument("--result_file", "-rf", type=str, required=True,
                        help="an attack result JSON written by run_attack_vuln.py")
    parser.add_argument("--source", "-src", type=str, default="hits",
                        choices=("hits", "trajectory"))
    parser.add_argument("--temperatures", "-T", type=str, default="0,0.3,0.5,0.7,1.0",
                        help="comma separated; 0 is greedy (default: 0,0.3,0.5,0.7,1.0)")
    parser.add_argument("--top_p", "-tpp", type=float, default=0.8)
    parser.add_argument("--top_k", "-tpk", type=int, default=20)
    parser.add_argument("--num_samples", "-ns", type=int, default=16,
                        help="samples per pair per temperature (default: 16)")
    parser.add_argument("--every", "-k", type=int, default=10,
                        help="trajectory: score the suffix of every k-th step (default: 10)")
    parser.add_argument("--restarts", "-r", type=str, default="0",
                        help="trajectory: comma separated restarts to follow, or 'all' (default: 0)")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="cap on (task, suffix) pairs scored, 0 for no cap")
    parser.add_argument("--tasks", "-t", type=str, default="")
    parser.add_argument("--path", "-p", type=str, default="")
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
    parser.add_argument("--collapsed_generation", "-cg", type=int, default=-1,
                        help="override the generation recorded in the result file")
    parser.add_argument("--collapsed_model_path", "-cmp", type=str, default="")
    parser.add_argument("--baseline_model_path", "-bmp", type=str, default="")
    parser.add_argument("--max_new_tokens", "-mnt", type=int, default=0)
    parser.add_argument("--exec_timeout", "-et", type=float, default=10.0)
    parser.add_argument("--out_file", "-of", type=str, default="")
    parser.add_argument("--device", "-dx", type=str, default="cuda")
    parser.add_argument("--seed", "-s", type=int, default=1337)
    parser.add_argument("--plot", action="store_true", help="also draw the figure")
    parser.add_argument("--no_usetex", dest="usetex", action="store_false")
    add_model_arguments(parser, role="the baseline model")
    return parser.parse_args()


def select_rows(report: dict, args: argparse.Namespace, task_names: set) -> list:
    """The (task, suffix, meta) rows to score, from the hits or from the trajectory."""
    rows = []
    for result in report.get("results") or []:
        if result["task"] not in task_names:
            continue
        if args.source == "hits":
            for hit in result.get("successes") or []:
                rows.append({"task": result["task"], "suffix": hit["suffix"],
                             "restart": hit.get("restart"), "step": hit.get("step"),
                             "col_wrong": hit.get("col_wrong"), "kind": "hit"})
            for hit in result.get("weak_hits") or []:
                rows.append({"task": result["task"], "suffix": hit["suffix"],
                             "restart": hit.get("restart"), "step": hit.get("step"),
                             "col_wrong": hit.get("col_wrong"), "kind": "weak_hit"})
        else:
            wanted = None if args.restarts == "all" else {int(r) for r in args.restarts.split(",")}
            first_flip = {(h["restart"], h["step"]) for h in (result.get("successes") or [])}
            for entry in result.get("history") or []:
                if wanted is not None and entry["restart"] not in wanted:
                    continue
                if entry["step"] % args.every and (entry["restart"], entry["step"]) not in first_flip:
                    continue
                rows.append({"task": result["task"], "suffix": entry["suffix"],
                             "restart": entry["restart"], "step": entry["step"],
                             "col_wrong": entry.get("col_wrong"), "total": entry.get("total"),
                             "kind": "hit" if (entry["restart"], entry["step"]) in first_flip
                             else "step"})
    if args.max_rows:
        rows = rows[: args.max_rows]
    return rows


def score(baseline: TargetModel, collapsed: TargetModel, tokenizer, prompt: str, task,
          decoding: Decoding, max_new_tokens: int, exec_timeout: float) -> dict:
    """Survival of one (prompt, task) at one decoding: paired samples, attack criterion."""
    col = collapsed.complete_many(tokenizer, prompt, max_new_tokens, 1.0, decoding)
    base = baseline.complete_many(tokenizer, prompt, max_new_tokens, 1.0, decoding)
    col_status = [run_attack.run_unit_tests(extract_code(r), task, exec_timeout) for r in col]
    base_status = [run_attack.run_unit_tests(extract_code(r), task, exec_timeout) for r in base]
    hits = sum(
        ContrastiveGCG.is_selective_hit({"collapsed_status": c, "baseline_status": b})
        for c, b in zip(col_status, base_status)
    )
    n = len(col_status)
    return {
        "n_samples": n,
        "n_hit": hits,
        "survival": hits / n,
        "collapsed_wrong_rate": sum(s in run_attack.WRONG_STATUSES for s in col_status) / n,
        "baseline_pass_rate": sum(s == "pass" for s in base_status) / n,
        "collapsed_statuses": col_status,
        "baseline_statuses": base_status,
    }


def draw(payload: dict, stem: str, usetex: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.plotting import apply_perplexity_style, save_figure

    apply_perplexity_style(usetex, font_size=15)
    temps = payload["temperatures"]
    rows = payload["rows"]
    tasks = sorted({r["task"] for r in rows})
    if payload["source"] == "hits":
        figure, axis = plt.subplots(figsize=(8, 5))
        for task in tasks:
            sel = [r for r in rows if r["task"] == task]
            means = [sum(r["by_temperature"][str(t)]["survival"] for r in sel) / len(sel)
                     for t in temps]
            axis.plot(temps, means, marker="o", label=f"{task} ({len(sel)} suffixes)")
        axis.set_xlabel("decoding temperature (0 = greedy)")
        axis.set_ylabel("mean survival of the verified hits")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"generation {payload['collapsed_generation']}, "
                       f"top-p {payload['top_p']}, top-k {payload['top_k']}")
        axis.legend(fontsize=9)
    else:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        for task in tasks:
            for restart in sorted({r["restart"] for r in rows if r["task"] == task}):
                sel = sorted((r for r in rows if r["task"] == task and r["restart"] == restart),
                             key=lambda r: r["step"])
                for t in temps:
                    ys = [r["by_temperature"][str(t)]["survival"] for r in sel]
                    axes[0].plot([r["step"] for r in sel], ys, marker=".",
                                 label=f"{task} r{restart} T={t:g}")
                losses = [r["col_wrong"] for r in sel if r["col_wrong"] is not None]
                if losses:
                    axes[1].plot([r["step"] for r in sel if r["col_wrong"] is not None], losses,
                                 marker=".", label=f"{task} r{restart}")
                flips = [r for r in sel if r["kind"] == "hit"]
                if flips:
                    axes[0].axvline(flips[0]["step"], color="grey", linewidth=0.8,
                                    linestyle="--")
        axes[0].set_xlabel("search step")
        axes[0].set_ylabel("survival at the decoding temperature")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title("survival along the search (dashed: first verified flip)")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("search step")
        axes[1].set_ylabel("mean CE of the wrong code (nats/token)")
        axes[1].set_title("the search's own margin proxy")
        axes[1].legend(fontsize=8)
    save_figure(figure, stem)
    print(f"##   figure: {stem}.pdf")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_attack_vuln.install_vulnerability_targets()
    if args.path:
        run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")

    with open(args.result_file, encoding="utf-8") as handle:
        report = json.load(handle)
    generation = args.collapsed_generation if args.collapsed_generation >= 0 else int(
        report.get("collapsed_generation", -1))
    temps = [float(t) for t in args.temperatures.split(",")]
    max_new_tokens = args.max_new_tokens or run_attack_vuln.DECODING_BUDGET

    device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_specifier = resolve_model_specifier(args.model_size, args.model_specifier)
    specifier_name = model_specifier.split("/")[-1]
    baseline_dir = args.baseline_model_path or model_specifier
    collapsed_dir = args.collapsed_model_path or resolve_collapsed_dir(
        generation, specifier_name, args.block_size, real_data_fraction=args.real_data_fraction,
    )
    tasks = [t for t in run_attack.TASKS if not args.tasks or t.name in args.tasks.split(",")]
    rows = select_rows(report, args, {t.name for t in tasks})
    if not rows:
        raise SystemExit(f"{TColors.FAIL}nothing to score{TColors.ENDC}: no {args.source} rows "
                         f"in {args.result_file} for {[t.name for t in tasks]}")

    stem = os.path.splitext(os.path.basename(args.result_file))[0]
    out_file = args.out_file or os.path.join(
        os.path.dirname(os.path.abspath(args.result_file)), f"survival_{args.source}_{stem}.json"
    )
    tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
    clear_inherited_max_length(tokenizer)
    configure_pad_token(tokenizer)

    print(f"\n## {TColors.BOLD}{TColors.HEADER}survival sweep{TColors.ENDC}")
    print(f"##   result file  : {args.result_file}")
    print(f"##   source       : {args.source}, {len(rows)} (task, suffix) row(s)")
    print(f"##   temperatures : {temps}, {args.num_samples} samples each, top-p {args.top_p}, "
          f"top-k {args.top_k}")
    print(f"##   collapsed    : {collapsed_dir}")
    print(f"##   output       : {out_file}")

    baseline = TargetModel("baseline", load_model(baseline_dir, device, dtype), device)
    collapsed = TargetModel("collapsed", load_model(collapsed_dir, device, dtype), device)
    by_name = {t.name: t for t in tasks}
    prompts = {t.name: run_attack.split_prompt(tokenizer, t) for t in tasks}

    # the clean prompt at every temperature: a target the collapsed model already fails at a
    # temperature is not attributable there, and the figure needs to know
    clean = {}
    for task in tasks:
        before, after = prompts[task.name]
        clean[task.name] = {}
        for t in temps:
            decoding = Decoding(t, args.top_p, args.top_k, args.num_samples)
            s = score(baseline, collapsed, tokenizer, before + after, task, decoding,
                      max_new_tokens, args.exec_timeout)
            clean[task.name][str(t)] = {
                "collapsed_pass_rate": 1 - s["collapsed_wrong_rate"]
                - sum(x == "error" for x in s["collapsed_statuses"]) / s["n_samples"],
                "collapsed_wrong_rate": s["collapsed_wrong_rate"],
                "baseline_pass_rate": s["baseline_pass_rate"],
            }
        print(f"##   clean {task.name:16s} " + "  ".join(
            f"T={t:g}: col-pass {clean[task.name][str(t)]['collapsed_pass_rate']:.2f}"
            for t in temps))

    payload = {
        "result_file": os.path.abspath(args.result_file),
        "source": args.source,
        "collapsed_generation": generation,
        "real_data_fraction": args.real_data_fraction,
        "specifier_name": specifier_name,
        "temperatures": temps,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_samples": args.num_samples,
        "clean": clean,
        "rows": [],
    }
    for index, row in enumerate(rows, start=1):
        task = by_name[row["task"]]
        before, after = prompts[row["task"]]
        prompt = before + row["suffix"] + after
        row["by_temperature"] = {}
        for t in temps:
            decoding = Decoding(t, args.top_p, args.top_k, args.num_samples)
            row["by_temperature"][str(t)] = score(
                baseline, collapsed, tokenizer, prompt, task, decoding, max_new_tokens,
                args.exec_timeout,
            )
        payload["rows"].append(row)
        print(f"##   [{index}/{len(rows)}] {row['task']:16s} r{row['restart']} step "
              f"{row['step']:>4} " + "  ".join(
                  f"T={t:g}: {row['by_temperature'][str(t)]['survival']:.2f}" for t in temps))
        if index % 10 == 0 or index == len(rows):
            with open(out_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

    # summary: mean survival per temperature, and for a trajectory the first step reaching each
    # survival level at each temperature (the cost curve)
    summary = {"mean_survival": {str(t): sum(r["by_temperature"][str(t)]["survival"]
                                             for r in payload["rows"]) / len(payload["rows"])
                                 for t in temps}}
    if args.source == "trajectory":
        levels = (0.25, 0.5, 0.75, 1.0)
        summary["first_step_reaching"] = {}
        for task_name in sorted({r["task"] for r in payload["rows"]}):
            for restart in sorted({r["restart"] for r in payload["rows"] if r["task"] == task_name}):
                sel = sorted((r for r in payload["rows"]
                              if r["task"] == task_name and r["restart"] == restart),
                             key=lambda r: r["step"])
                key = f"{task_name}/r{restart}"
                summary["first_step_reaching"][key] = {
                    str(t): {str(level): next((r["step"] for r in sel
                                               if r["by_temperature"][str(t)]["survival"] >= level),
                                              None) for level in levels}
                    for t in temps
                }
    payload["summary"] = summary
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n## {TColors.BOLD}mean survival{TColors.ENDC}: " + "  ".join(
        f"T={t:g}: {summary['mean_survival'][str(t)]:.2f}" for t in temps))
    if args.source == "trajectory":
        for key, per_t in summary["first_step_reaching"].items():
            print(f"##   {key:24s} " + "  ".join(
                f"T={t:g}: " + "/".join(str(per_t[str(t)][str(l)]) for l in (0.25, 0.5, 0.75, 1.0))
                for t in temps) + "   (first step reaching survival 0.25/0.5/0.75/1.0)")
    print(f"## {TColors.OKGREEN}written{TColors.ENDC}: {out_file}")
    if args.plot:
        draw(payload, os.path.splitext(out_file)[0], args.usetex)


if __name__ == "__main__":
    main()
