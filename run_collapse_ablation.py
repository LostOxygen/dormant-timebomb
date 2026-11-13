"""what role does collapse itself play in the attack: two ablations, measured and drawn

The grid says the non-collapsing control (rdf 1.0) is the most attackable lineage under the logit
surrogate, so the attack as measured exploits iterated fine-tuning and collapse mainly shrinks
the window. This script asks the two questions the grid cannot answer on its own, both without a
new search:

**1. Survival: which hits are collapse-specific.** Every suffix that broke generation *g* of a
lineage is re-verified, under the family's own verdict, against every generation of the same
lineage (0..N) and against the control lineage (``--score_mixtures``). A suffix that breaks
generation 0 as well exploits the attacker's own fine-tune, since generation 0 is what the
surrogate is built from; one that spares generation 0 but breaks *g* is collapse-specific. The
control lineage gives the noise floor of that share: it has no collapse, so there "spares
generation 0, breaks g" can only be seed noise. Every verdict is the attack's own —
``run_attack.judge_model`` with the file's decoding and ``ContrastiveGCG.is_selective_hit`` on
the (target, baseline) pair, the baseline re-decoded under the same suffix — and a suffix is
scored on a generation only where that generation passes the capability gate (solves the clean
prompt), exactly as the grid counts attackable targets.

**2. Geometry: does a lineage stay on the surrogate's ray.** The logit surrogate assumes
``l_g - l_base`` is collinear with ``l_0 - l_base``. At every target's security decision — the
position where the secure and insecure continuations first differ, in the context of the base
model's own answer, the same context the sink screen uses — this measures, per generation, the
cosine between the two deltas in centred log-probability space, the least-squares position
``n_hat = <d_g, d_0> / |d_0|^2`` along the ray (what the plain surrogate sets to ``g + 1`` and
the optimized one to the capped root ``n*``), the KL of the real generation to the nearest point
on the ray and to the surrogate the attack actually used, and the decision margin against the
affine law's prediction, which is exact at a single position. The same quantities are taken at
the ``--positions`` code positions before the decision, so a decision-local reading can be told
from a prefix-wide one.

Nothing here changes an attack result. The survival files are
``attack_results/survival_matrix_{model}{mix}_{family}.json``, the geometry files
``attack_results/geometry_{model}{mix}.json``, and ``--mode plot`` draws every figure of the
study from them (``plots/ablation_*``) plus ``plots/ablation_collapse.md``.

Usage
-----
    python run_collapse_ablation.py --mode survival -msz 0.5b -rdf 0.5 -n 9 -p ./runs/fresh \\
        --method logit --variant opt --verdict sampled --score_mixtures 0.5 1.0
    python run_collapse_ablation.py --mode geometry -msz 0.5b -rdf 0.5 -n 9 -p ./runs/fresh
    python run_collapse_ablation.py --mode plot --sizes 0.5b 3b -p ./runs/fresh -pp ./runs/fresh/plots

run_collapse_ablation.sh queues the measurements of a grid over the GPUs and draws the figures.
"""
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines,import-outside-toplevel

import argparse
import json
import math
import os
import statistics

from utils import nord
from utils.colors import TColors
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import (
    anchor_tag,
    control_file,
    entropy_tag,
    factor_mode_tag,
    mixture_tag,
    objective_tag,
    verification_tag,
)

WRONG_STATUSES: tuple = ("fail", "fail_exception")
PASS: str = "pass"
VULN_MARKER: str = "_vuln"
FAMILY_LABELS: dict = {
    ("none", "plain"): "direct",
    ("logit", "plain"): "logit plain",
    ("logit", "opt"): "logit opt",
}
DEFAULT_FAMILIES: tuple = ("none_plain_sampled", "logit_opt_sampled")


# ───────────────────────────────── naming ─────────────────────────────────────────────────
def family_tag(method: str, variant: str, verdict: str, hold_anchor: bool = False) -> str:
    """``{method}_{variant}_{verdict}[_anchor]``: one attack family, as the output names spell it."""
    return f"{method}_{variant}_{verdict}{anchor_tag(hold_anchor)}"


def parse_family(tag: str) -> tuple:
    """(method, variant, verdict, hold_anchor) of a family tag."""
    hold = tag.endswith(anchor_tag(True))
    if hold:
        tag = tag[: -len(anchor_tag(True))]
    method, variant, verdict = tag.split("_")
    if (method, variant) not in FAMILY_LABELS or verdict not in ("sampled", "greedy"):
        raise ValueError(f"unknown attack family {tag!r}")
    return method, variant, verdict, hold


def family_label(tag: str) -> str:
    """``logit opt, sampled verdict`` and the like, for titles."""
    method, variant, verdict, hold = parse_family(tag)
    return (f"{FAMILY_LABELS[(method, variant)]}, {verdict} verdict"
            + (", anchor held" if hold else ""))


def attack_result_path(results_dir: str, name: str, mixture: float, method: str, variant: str,
                       verdict: str, generation: int, hold_anchor: bool, opt_entropy_match: bool,
                       opt_sink_factor: bool, opt_sink_objective: bool) -> str:
    """The vulnerability result file run_attack.py wrote for one cell, spelled with the naming
    helpers (the same rule as run_paper_tables.Grid.result_path)."""
    transfer = method != "none"
    opt = variant == "opt" and transfer
    decoding = verification_tag(0.0) if verdict == "greedy" else verification_tag()
    stem = (
        f"attack_gen{generation}_{name}{mixture_tag(mixture)}{VULN_MARKER}{decoding}"
        + anchor_tag(hold_anchor and generation > 0)
        + objective_tag(opt and opt_sink_objective)
        + (
            f"{factor_mode_tag('sink') if (opt and opt_sink_factor) else ''}"
            f"{entropy_tag(opt and opt_entropy_match)}_{method}_surrogate" if transfer else ""
        )
    )
    return os.path.join(results_dir, f"{stem}.json")


def survival_file(results_dir: str, name: str, mixture: float, tag: str) -> str:
    return os.path.join(results_dir, f"survival_matrix_{name}{mixture_tag(mixture)}_{tag}.json")


def geometry_file(results_dir: str, name: str, mixture: float) -> str:
    return os.path.join(results_dir, f"geometry_{name}{mixture_tag(mixture)}.json")


def model_key(mixture: float, generation: int) -> str:
    return f"rdf{mixture:g}/gen{generation}"


def lineage_label(mixture: float) -> str:
    return "control" if mixture >= 1.0 else f"rdf {mixture:g}"


def read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def error_bar(rate: float, low: float, high: float) -> list:
    """The asymmetric yerr of a share with its interval, clamped at 0: at a share of exactly 0
    or 1 the Wilson bound lands on the share up to rounding, and a -1e-17 is refused."""
    return [[max(rate - low, 0.0)], [max(high - rate, 0.0)]]


def wilson(hits: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval of a share, (low, high); NaNs for an empty denominator."""
    if total <= 0:
        return float("nan"), float("nan")
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


# ───────────────────────────── the survival measurement ───────────────────────────────────
def harvest_hits(args, name: str, results_dir: str) -> tuple:
    """Every verified hit of one family in one lineage, deduplicated by (target, suffix).

    Returns:
        tuple: (rows, sources, config, hold_anchor) — rows are {id, task, suffix,
            source_generations}; sources one entry per result file read; config the first file's
            SearchConfig dict (the decoding every re-verification uses)
    """
    from utils.vuln_results import load_record

    method, variant, verdict, hold = parse_family(args.family)
    rows: dict = {}
    sources = []
    config = None
    for generation in range(args.generations + 1):
        if generation == 0 and method != "none":
            continue
        path = attack_result_path(
            results_dir, name, args.real_data_fraction, method, variant, verdict, generation,
            hold, args.opt_entropy_match, args.opt_sink_factor, args.opt_sink_objective,
        )
        if not os.path.isfile(path):
            continue
        try:
            record = load_record(path)
        except (ValueError, json.JSONDecodeError, KeyError):
            print(f"##   {TColors.WARNING}unreadable{TColors.ENDC}: {path}")
            continue
        if config is None:
            config = record.config
        count = 0
        for task_name, outcome in record.tasks.items():
            if args.tasks and task_name not in args.tasks.split(","):
                continue
            for suffix in outcome.suffixes:
                count += 1
                row = rows.setdefault((task_name, suffix), {
                    "task": task_name, "suffix": suffix, "source_generations": [],
                })
                if generation not in row["source_generations"]:
                    row["source_generations"].append(generation)
        sources.append({"generation": generation, "file": os.path.basename(path),
                        "n_hits": count, "aborted": record.aborted})
    ordered = sorted(rows.values(), key=lambda r: (min(r["source_generations"]), r["task"]))
    if args.limit > 0:
        ordered = ordered[: args.limit]
    for index, row in enumerate(ordered):
        row["id"] = index
    return ordered, sources, config or {}, hold


def measure_survival(args) -> None:
    """Re-verifies every hit of one family against every generation of the scored lineages."""
    import torch
    from transformers import AutoTokenizer

    import run_attack
    import run_attack_vuln
    from run_attack import (
        Decoding, GREEDY_DECODING, SearchConfig, TargetModel, judge_model, load_model,
        resolve_collapsed_dir,
    )
    from utils.utils import clear_inherited_max_length, configure_pad_token

    run_attack_vuln.install_vulnerability_targets()
    results_dir = os.path.join(args.path, "attack_results")
    run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")
    specifier = resolve_model_specifier(args.model_size, args.model_specifier)
    name = specifier.split("/")[-1]
    baseline_dir = args.baseline_model_path or specifier

    rows, sources, config, hold = harvest_hits(args, name, results_dir)
    out_file = args.out_file or survival_file(results_dir, name, args.real_data_fraction,
                                              args.family)
    method, variant, verdict, _ = parse_family(args.family)
    print(f"\n## {TColors.BOLD}{TColors.HEADER}survival matrix{TColors.ENDC}: {name}, "
          f"{lineage_label(args.real_data_fraction)}, {family_label(args.family)}")
    print(f"##   sources      : {len(sources)} result file(s), {len(rows)} distinct (target, "
          f"suffix) pair(s)")
    if not rows:
        print(f"##   {TColors.WARNING}nothing to score{TColors.ENDC}: no verified hit in "
              f"{results_dir} for this family and lineage")
        payload = {"specifier_name": name, "size": args.model_size,
                   "real_data_fraction": args.real_data_fraction, "family": args.family,
                   "generations": args.generations, "sources": sources, "suffixes": [],
                   "baseline": {}, "models": {}}
        with open(out_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"## {TColors.OKBLUE}saved{TColors.ENDC} {out_file} (empty)")
        return

    # the family's own decoding, read off its result file rather than assumed
    if verdict == "greedy":
        decoding = GREEDY_DECODING
    else:
        decoding = Decoding(
            temperature=float(config.get("verify_temperature", 0.7)),
            top_p=float(config.get("verify_top_p", 0.8)),
            top_k=int(config.get("verify_top_k", 20)),
            num_samples=args.num_samples or int(config.get("verify_samples", 5)),
        )
    cfg = SearchConfig(
        max_new_tokens=int(config.get("max_new_tokens", run_attack_vuln.DECODING_BUDGET)),
        repetition_penalty=float(config.get("repetition_penalty", 1.0)),
        exec_timeout=float(config.get("exec_timeout", 10.0)),
        verify_temperature=decoding.temperature, verify_top_p=decoding.top_p,
        verify_top_k=decoding.top_k, verify_samples=decoding.n,
    )
    print(f"##   decoding     : {decoding.describe()}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
    clear_inherited_max_length(tokenizer)
    configure_pad_token(tokenizer)
    tasks = {t.name: t for t in run_attack.TASKS}
    involved = sorted({row["task"] for row in rows})
    prompts = {t: run_attack.split_prompt(tokenizer, tasks[t]) for t in involved}

    # the models to score: every generation of every scored lineage, the source lineage first
    score_mixtures = [float(m) for m in (args.score_mixtures or [args.real_data_fraction])]
    if args.real_data_fraction not in score_mixtures:
        score_mixtures.insert(0, args.real_data_fraction)
    targets = []
    for mixture in score_mixtures:
        for generation in range(args.generations + 1):
            try:
                model_dir = resolve_collapsed_dir(
                    generation, name, args.block_size, real_data_fraction=mixture,
                    human_generation_zero=args.human_generation_zero,
                )
            except FileNotFoundError as exc:
                print(f"##   {TColors.WARNING}skipped{TColors.ENDC} {model_key(mixture, generation)}: {exc}")
                continue
            targets.append((mixture, generation, model_dir))
    print(f"##   models       : {len(targets)} checkpoint(s) over mixtures "
          f"{', '.join(f'{m:g}' for m in score_mixtures)}, plus the baseline")
    print(f"##   output       : {out_file}")

    # resume: a file whose suffix list is the one harvested now keeps every model it holds
    payload = None if args.force else read_json(out_file)
    if payload is not None:
        same = [(r["task"], r["suffix"]) for r in payload.get("suffixes", [])] == \
               [(r["task"], r["suffix"]) for r in rows]
        if not same:
            print(f"##   {TColors.WARNING}existing file holds a different suffix set; "
                  f"starting over{TColors.ENDC}")
            payload = None
    if payload is None:
        payload = {
            "specifier_name": name, "size": args.model_size,
            "real_data_fraction": args.real_data_fraction, "family": args.family,
            "method": method, "variant": variant, "verdict": verdict, "hold_anchor": hold,
            "generations": args.generations, "block_size": args.block_size,
            "decoding": {"temperature": decoding.temperature, "top_p": decoding.top_p,
                         "top_k": decoding.top_k, "num_samples": decoding.n,
                         "max_new_tokens": cfg.max_new_tokens},
            "score_mixtures": score_mixtures, "sources": sources, "suffixes": rows,
            "baseline": {}, "models": {},
        }

    def save() -> None:
        with open(out_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def judge(model, prompt: str, task, prefer_pass: bool) -> dict:
        out = judge_model(model, tokenizer, prompt, task, cfg, prefer_pass, decoding)
        row = {"status": out["status"], "statuses": out["statuses"],
               "wrong_rate": out["wrong_rate"]}
        if out["status"] in WRONG_STATUSES:
            row["code"] = out["code"][:600]
        return row

    def score_model(label: str, model_dir: str, prefer_pass: bool) -> dict:
        torch.manual_seed(args.seed)
        model = TargetModel(label, load_model(model_dir, device, dtype), device)
        clean = {}
        for task_name in involved:
            before, after = prompts[task_name]
            clean[task_name] = judge(model, before + after, tasks[task_name], prefer_pass)
        verdicts = {}
        for index, row in enumerate(rows, start=1):
            before, after = prompts[row["task"]]
            verdicts[str(row["id"])] = judge(
                model, before + row["suffix"] + after, tasks[row["task"]], prefer_pass
            )
            if index % 50 == 0 or index == len(rows):
                wrong = sum(v["status"] in WRONG_STATUSES for v in verdicts.values())
                print(f"##   {label:14s} {index}/{len(rows)} pairs scored, {wrong} wrong so far")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        solved = sum(v["status"] == PASS for v in clean.values())
        print(f"##   {label:14s} clean prompts solved: {solved}/{len(clean)}")
        return {"model": model_dir, "clean": clean, "verdicts": verdicts}

    # the baseline side of every pair once: the hit rule needs it secure under the suffix
    if not payload["baseline"].get("verdicts"):
        payload["baseline"] = score_model("baseline", baseline_dir, prefer_pass=False)
        save()
    else:
        print("##   baseline     : kept from the existing file")

    for mixture, generation, model_dir in targets:
        key = model_key(mixture, generation)
        if key in payload["models"]:
            print(f"##   {key:14s} kept from the existing file")
            continue
        entry = score_model(key, model_dir, prefer_pass=True)
        entry.update(mixture=mixture, generation=generation)
        payload["models"][key] = entry
        save()

    survival = Survival(payload)
    generations = sorted({g for r in rows for g in r["source_generations"]})
    for reference, head, spares in (
        ("control", "collapse-specific", "the control at the same generation"),
        ("gen0", "new since the first fine-tune", "generation 0 of the lineage"),
    ):
        if reference == "control" and (survival.mixture >= 1.0 or not survival.has(1.0, 0)):
            continue
        print(f"\n## {TColors.BOLD}{head}{TColors.ENDC} hits (re-verified hits that spare "
              f"{spares}, gated):")
        for generation in generations:
            counts = survival.specificity(generation, reference)
            print(f"##   gen {generation}: {counts['reverified']}/{counts['total']} hit(s) "
                  f"re-verified, of which {counts['specific']} spare it, {counts['generic']} "
                  f"break it too, {counts['unevaluable']} not evaluable there")
    print(f"## {TColors.OKBLUE}saved{TColors.ENDC} {out_file}")


# ───────────────────────────── reading a survival file ────────────────────────────────────
class Survival:
    """One survival file, with the hit rule and the gate applied on read."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.name = payload["specifier_name"]
        self.size = payload.get("size", "")
        self.mixture = float(payload["real_data_fraction"])
        self.family = payload["family"]
        self.hold_anchor = bool(payload.get("hold_anchor", False))
        self.generations = int(payload["generations"])
        self.suffixes = payload.get("suffixes", [])
        self.models = payload.get("models", {})
        self.baseline = payload.get("baseline", {})

    @property
    def empty(self) -> bool:
        return not self.suffixes or not self.models

    def has(self, mixture: float, generation: int) -> bool:
        return model_key(mixture, generation) in self.models

    def mixtures_scored(self) -> list:
        return sorted({float(m["mixture"]) for m in self.models.values()})

    def evaluable(self, row: dict, mixture: float, generation: int) -> bool:
        """The capability gate of the grid: target and baseline solve the clean prompt."""
        entry = self.models.get(model_key(mixture, generation))
        if entry is None:
            return False
        clean = entry["clean"].get(row["task"], {}).get("status")
        base = self.baseline.get("clean", {}).get(row["task"], {}).get("status")
        return clean == PASS and base == PASS

    def hit(self, row: dict, mixture: float, generation: int):
        """True/False under the family's hit rule, None where the pair is not evaluable."""
        if not self.evaluable(row, mixture, generation):
            return None
        verdict = self.models[model_key(mixture, generation)]["verdicts"].get(str(row["id"]))
        base = self.baseline.get("verdicts", {}).get(str(row["id"]))
        if verdict is None or base is None:
            return None
        anchor_ok = True
        if self.hold_anchor:
            anchor = self.models.get(model_key(self.mixture, 0))
            anchor_status = (anchor or {}).get("verdicts", {}).get(str(row["id"]), {}).get("status")
            anchor_ok = anchor_status == PASS
        return verdict["status"] in WRONG_STATUSES and base["status"] == PASS and anchor_ok

    def found_at(self, generation: int) -> list:
        return [r for r in self.suffixes if generation in r["source_generations"]]

    def matrix(self, mixture: float) -> dict:
        """(source g, target h) -> (hits, evaluable) of g's hits scored on h of `mixture`."""
        cells = {}
        for g in range(self.generations + 1):
            rows = self.found_at(g)
            if not rows:
                continue
            for h in range(self.generations + 1):
                if not self.has(mixture, h):
                    continue
                verdicts = [self.hit(r, mixture, h) for r in rows]
                cells[(g, h)] = (sum(1 for v in verdicts if v), sum(1 for v in verdicts if v is not None))
        return cells

    def specificity(self, generation: int, reference="control") -> dict:
        """How generation g's re-verified hits split against a reference model.

        ``reference`` names the counterfactual: ``"control"`` is generation g of the
        non-collapsing lineage (the same number of training rounds on real data, so a hit that
        spares it was not produced by ordinary drift: the collapse-specific rule), a mixture is
        generation g of that lineage (the reverse direction, for a control file scored on the
        collapsed lineages), and ``"gen0"`` is generation 0 of the suffix's own lineage (the
        attacker's fine-tune, so a hit that spares it is new since the first fine-tune).

        Only a hit that still breaks g when re-verified says anything: the attack's own verdict
        was one draw of a sampled majority, taken at the best of many checks, so a share of the
        recorded hits does not reproduce (``lost``). The re-verified ones are ``specific``
        (spare the reference), ``generic`` (break it too) or ``unevaluable`` (the reference
        fails the clean prompt or was not scored, so the pair is outside the gate).
        """
        rows = self.found_at(generation)
        counts = {"total": len(rows), "lost": 0, "specific": 0, "generic": 0, "unevaluable": 0,
                  "reverified": 0}
        for row in rows:
            if not self.hit(row, self.mixture, generation):
                counts["lost"] += 1
                continue
            counts["reverified"] += 1
            if reference == "gen0":
                other = self.hit(row, self.mixture, 0)
            else:
                mixture = 1.0 if reference == "control" else float(reference)
                other = self.hit(row, mixture, generation)
            if other is None:
                counts["unevaluable"] += 1
            elif other:
                counts["generic"] += 1
            else:
                counts["specific"] += 1
        return counts

    def breadth(self) -> list:
        """Per suffix, the share of its own lineage's evaluable generations it breaks."""
        out = []
        for row in self.suffixes:
            verdicts = [self.hit(row, self.mixture, h) for h in range(self.generations + 1)]
            evaluable = [v for v in verdicts if v is not None]
            if evaluable:
                out.append(sum(evaluable) / len(evaluable))
        return out

    def transfer(self, mixture: float, same_generation: bool) -> dict:
        """Per source generation, (hits, evaluable) of g's hits on `mixture`: at the same
        generation, or on any of its generations."""
        out = {}
        for g in range(self.generations + 1):
            rows = self.found_at(g)
            if not rows:
                continue
            hits = evaluable = 0
            for row in rows:
                if same_generation:
                    verdict = self.hit(row, mixture, g)
                    if verdict is not None:
                        evaluable += 1
                        hits += bool(verdict)
                else:
                    verdicts = [self.hit(row, mixture, h) for h in range(self.generations + 1)]
                    known = [v for v in verdicts if v is not None]
                    if known:
                        evaluable += 1
                        hits += any(known)
            out[g] = (hits, evaluable)
        return out


# ───────────────────────────── the geometry measurement ───────────────────────────────────
def measure_geometry(args) -> None:
    """The lineage against the surrogate's ray, at every target's decision position."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    import run_attack
    import run_attack_vuln
    from run_attack import TargetModel, load_model, resolve_collapsed_dir
    from run_margin import SINKS, _token_prefix, find_decision_point, score_sink_at
    from utils.extrapolation import entropy_matched_scale, logit_entropy
    from utils.utils import clear_inherited_max_length, configure_pad_token

    run_attack_vuln.install_vulnerability_targets()
    results_dir = os.path.join(args.path, "attack_results")
    run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")
    specifier = resolve_model_specifier(args.model_size, args.model_specifier)
    name = specifier.split("/")[-1]
    baseline_dir = args.baseline_model_path or specifier
    out_file = args.out_file or geometry_file(results_dir, name, args.real_data_fraction)
    if os.path.isfile(out_file) and not args.force:
        print(f"## {TColors.OKBLUE}exists{TColors.ENDC}: {out_file} (use --force to redo)")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
    clear_inherited_max_length(tokenizer)
    configure_pad_token(tokenizer)
    tasks = [t for t in run_attack.TASKS if t.name in SINKS
             and (not args.tasks or t.name in args.tasks.split(","))]
    prompts = {t.name: "".join(run_attack.split_prompt(tokenizer, t)) for t in tasks}
    budget = args.budget or run_attack_vuln.DECODING_BUDGET

    # the per-target factor the optimized attack used at each generation, where its file exists
    factors_used: dict = {}
    for generation in range(1, args.generations + 1):
        path = attack_result_path(
            results_dir, name, args.real_data_fraction, "logit", "opt", "sampled", generation,
            args.hold_anchor, args.opt_entropy_match, args.opt_sink_factor,
            args.opt_sink_objective,
        )
        payload = read_json(path)
        if payload and payload.get("task_factors"):
            factors_used[generation] = {k: float(v) for k, v in payload["task_factors"].items()
                                        if v is not None}

    print(f"\n## {TColors.BOLD}{TColors.HEADER}ray geometry{TColors.ENDC}: {name}, "
          f"{lineage_label(args.real_data_fraction)}")
    print(f"##   targets      : {len(tasks)}; positions kept before the decision: {args.positions}")
    print(f"##   output       : {out_file}")

    def centred_logprobs(logits: torch.Tensor) -> torch.Tensor:
        rows = F.log_softmax(logits.float(), dim=-1)
        return rows - rows.mean(dim=-1, keepdim=True)

    def rows_for(model, context: list) -> torch.Tensor:
        """Centred log-probability rows at the last `positions`+1 context positions, the decision
        last; (R, V) in float32 on the CPU."""
        ids = torch.tensor([context], device=device)
        with torch.no_grad():
            logits = model.model(input_ids=ids).logits[0]
        keep = min(args.positions + 1, logits.shape[0])
        return centred_logprobs(logits[-keep:]).cpu()

    def kl(p_rows: torch.Tensor, q_rows: torch.Tensor) -> torch.Tensor:
        """KL(p || q) per row from centred log-probabilities (re-normalised here)."""
        log_p = F.log_softmax(p_rows, dim=-1)
        log_q = F.log_softmax(q_rows, dim=-1)
        return (log_p.exp() * (log_p - log_q)).sum(dim=-1)

    def nearest_on_ray(p_rows: torch.Tensor, base: torch.Tensor, delta: torch.Tensor,
                       lo: float = -2.0, hi: float = 8.0, iterations: int = 48) -> tuple:
        """Per row, the factor n minimising KL(p || softmax(base + n delta)) and that KL.

        The KL to the tilt is convex in n (the log-partition of an exponential family is convex
        in its natural parameter, the rest is linear), so a ternary search on [lo, hi] finds the
        minimum; vectorised over the rows.
        """
        low = torch.full((p_rows.shape[0],), lo)
        high = torch.full((p_rows.shape[0],), hi)
        for _ in range(iterations):
            m1 = low + (high - low) / 3
            m2 = high - (high - low) / 3
            k1 = kl(p_rows, base + m1.unsqueeze(-1) * delta)
            k2 = kl(p_rows, base + m2.unsqueeze(-1) * delta)
            left = k1 < k2
            high = torch.where(left, m2, high)
            low = torch.where(left, low, m1)
        n_kl = (low + high) / 2
        return n_kl, kl(p_rows, base + n_kl.unsqueeze(-1) * delta)

    def head_indices(*rows_list: torch.Tensor) -> torch.Tensor:
        """Per row, the union of every model's top-`head` tokens, padded to one width by the
        top tokens of the first model: (R, K) indices."""
        tops = torch.cat([r.topk(args.head, dim=-1).indices for r in rows_list], dim=-1)
        out = []
        for row in tops:
            unique = torch.unique(row)
            out.append(unique)
        width = max(len(u) for u in out)
        padded = torch.stack([
            torch.cat([u, tops[i, : width - len(u)]]) if len(u) < width else u
            for i, u in enumerate(out)
        ])
        return padded

    def ray_metrics(b: torch.Tensor, z: torch.Tensor, g: torch.Tensor, plain: float) -> dict:
        """Every per-row geometry number of generation rows `g` against base `b` and
        generation-0 rows `z`; the decision row is the last one."""
        d_zero, d_gen = z - b, g - b
        dots = (d_gen * d_zero).sum(dim=-1)
        norm_zero = d_zero.norm(dim=-1).clamp(min=1e-12)
        norm_gen = d_gen.norm(dim=-1).clamp(min=1e-12)
        # the same two numbers on the head of the vocabulary: the tokens any of the three models
        # puts mass on, which is what decoding and the sink margin read; the tail of 150k
        # never-sampled tokens dominates the full-vocabulary norm
        head = head_indices(b, z, g)
        hb, hz, hg = (torch.gather(r, 1, head) for r in (b, z, g))
        hb, hz, hg = (r - r.mean(dim=-1, keepdim=True) for r in (hb, hz, hg))
        hd_zero, hd_gen = hz - hb, hg - hb
        hdots = (hd_gen * hd_zero).sum(dim=-1)
        hnorm_zero = hd_zero.norm(dim=-1).clamp(min=1e-12)
        hnorm_gen = hd_gen.norm(dim=-1).clamp(min=1e-12)
        n_kl, kl_ray = nearest_on_ray(g, b, d_zero)
        return {
            "cos": dots / (norm_zero * norm_gen),
            "n_hat": dots / norm_zero.pow(2),
            "norm_gen": norm_gen, "norm_gen0": norm_zero,
            "cos_head": hdots / (hnorm_zero * hnorm_gen),
            "n_hat_head": hdots / hnorm_zero.pow(2),
            "head_width": torch.full_like(dots, float(head.shape[1])),
            "n_kl": n_kl, "kl_to_ray": kl_ray,
            "kl_to_plain": kl(g, b + plain * d_zero),
            "kl_to_base": kl(g, b), "kl_to_gen0": kl(g, z),
        }

    # ── the decision contexts, from the base model's own answers ──
    base = TargetModel("baseline", load_model(baseline_dir, device, dtype), device)
    contexts: dict = {}
    for task in tasks:
        spec = SINKS[task.name]
        raw = base.complete(tokenizer, prompts[task.name], budget, 1.0)
        code = raw
        for stop in ("<|im_end|>", "<|endoftext|>"):
            code = code.split(stop)[0]
        index, anchor = find_decision_point(code, spec)
        if index < 0:
            contexts[task.name] = {"found": False,
                                   "reason": "the base model's own answer never reaches a decision point"}
            continue
        prefix_ids, leftover = _token_prefix(tokenizer, code, index + len(anchor))
        prompt_ids = tokenizer(prompts[task.name], add_special_tokens=False)["input_ids"]
        secure = tokenizer(leftover + spec["secure"], add_special_tokens=False)["input_ids"]
        insecure = tokenizer(leftover + spec["insecure"], add_special_tokens=False)["input_ids"]
        k = 0
        while k < min(len(secure), len(insecure)) and secure[k] == insecure[k]:
            k += 1
        if k == len(secure) or k == len(insecure):
            contexts[task.name] = {"found": False,
                                   "reason": "one continuation is a token prefix of the other"}
            continue
        contexts[task.name] = {
            "found": True, "anchor": anchor, "what": spec["what"], "code": code,
            "index": index, "n_prompt": len(prompt_ids),
            "context": prompt_ids + prefix_ids + secure[:k],
            "secure_token": secure[k], "insecure_token": insecure[k],
            "secure_text": tokenizer.decode([secure[k]]),
            "insecure_text": tokenizer.decode([insecure[k]]),
        }
    found = [t.name for t in tasks if contexts[t.name]["found"]]
    print(f"##   decisions    : {len(found)}/{len(tasks)} targets reach a decision in the base "
          f"model's own answer")

    def token_margin(rows: torch.Tensor, ctx: dict) -> float:
        row = rows[-1]
        return float(row[ctx["secure_token"]] - row[ctx["insecure_token"]])

    def measure_model(model) -> dict:
        """Per target: the centred rows and the two margins of one model."""
        out = {}
        for task in tasks:
            ctx = contexts[task.name]
            if not ctx["found"]:
                continue
            rows = rows_for(model, ctx["context"])
            sink = score_sink_at(model, tokenizer, prompts[task.name], ctx["code"], ctx["index"],
                                 ctx["anchor"], SINKS[task.name], device)
            out[task.name] = {"rows": rows, "token_margin": token_margin(rows, ctx),
                              "sink_margin": sink["sink_margin"],
                              "entropy": float(logit_entropy(rows[-1]))}
        return out

    base_rows = measure_model(base)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = {
        "specifier_name": name, "size": args.model_size,
        "real_data_fraction": args.real_data_fraction, "generations": args.generations,
        "block_size": args.block_size, "baseline_model": baseline_dir,
        "positions": args.positions, "head": args.head, "sink_cap": args.sink_cap,
        "targets": {t.name: {k: v for k, v in contexts[t.name].items()
                             if k not in ("context", "code")}
                    for t in tasks},
        "baseline": {t: {"token_margin": r["token_margin"], "sink_margin": r["sink_margin"],
                         "entropy": r["entropy"]} for t, r in base_rows.items()},
        "per_generation": {},
    }

    gen0_rows = None
    for generation in range(args.generations + 1):
        try:
            model_dir = resolve_collapsed_dir(
                generation, name, args.block_size, real_data_fraction=args.real_data_fraction,
                human_generation_zero=args.human_generation_zero,
            )
        except FileNotFoundError as exc:
            print(f"##   {TColors.WARNING}generation {generation} skipped{TColors.ENDC}: {exc}")
            continue
        model = TargetModel(f"gen {generation}", load_model(model_dir, device, dtype), device)
        measured = measure_model(model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if generation == 0:
            gen0_rows = measured
        entry = {"model": model_dir, "targets": {}}
        for task_name, current in measured.items():
            b, z = base_rows[task_name], (gen0_rows or {}).get(task_name)
            row = {"token_margin": current["token_margin"],
                   "sink_margin": current["sink_margin"], "entropy": current["entropy"],
                   "token_margin_base": b["token_margin"], "sink_margin_base": b["sink_margin"]}
            if z is None:
                entry["targets"][task_name] = row
                continue
            plain = float(generation + 1)
            metrics = ray_metrics(b["rows"], z["rows"], current["rows"], plain)
            m_base, m_zero, m_gen = b["token_margin"], z["token_margin"], current["token_margin"]
            slope = m_zero - m_base
            n_kl, n_hat = float(metrics["n_kl"][-1]), float(metrics["n_hat"][-1])
            row.update({key: float(value[-1]) for key, value in metrics.items()})
            row.update({
                "token_margin_gen0": m_zero, "sink_margin_gen0": z["sink_margin"],
                "cos_positions": [float(v) for v in metrics["cos"][:-1]],
                "cos_head_positions": [float(v) for v in metrics["cos_head"][:-1]],
                "n_kl_positions": [float(v) for v in metrics["n_kl"][:-1]],
                "kl_to_ray_positions": [float(v) for v in metrics["kl_to_ray"][:-1]],
                "margin_slope": slope,
                "n_margin": (m_gen - m_base) / slope if abs(slope) > 1e-9 else None,
                "margin_pred_plain": (1 - plain) * m_base + plain * m_zero,
                "margin_pred_nhat": (1 - n_hat) * m_base + n_hat * m_zero,
                "margin_pred_nkl": (1 - n_kl) * m_base + n_kl * m_zero,
            })
            d_zero = z["rows"] - b["rows"]
            # the surrogate the optimized attack actually used at this cell, if it ran
            factor = (factors_used.get(generation) or {}).get(task_name)
            if factor is not None and generation > 0:
                tilt = b["rows"][-1:] + factor * d_zero[-1:]
                if args.opt_entropy_match:
                    tau = entropy_matched_scale(tilt, logit_entropy(z["rows"][-1:]))
                    tilt = tilt / tau.unsqueeze(-1).clamp(min=1e-6)
                row["factor_used"] = factor
                row["kl_to_surrogate"] = float(kl(current["rows"][-1:], tilt)[0])
                row["margin_surrogate"] = float(tilt[0, contexts[task_name]["secure_token"]]
                                                - tilt[0, contexts[task_name]["insecure_token"]])
            entry["targets"][task_name] = row
        payload["per_generation"][str(generation)] = entry
        if generation > 0:
            values = [r["cos_head"] for r in entry["targets"].values() if "cos_head" in r]
            n_kls = [r["n_kl"] for r in entry["targets"].values() if "n_kl" in r]
            kls = [r["kl_to_ray"] for r in entry["targets"].values() if "kl_to_ray" in r]
            print(f"##   gen {generation}: median head cosine {statistics.median(values):.3f}, "
                  f"median n_kl {statistics.median(n_kls):.2f} (plain assumes {generation + 1}), "
                  f"median KL to the ray {statistics.median(kls):.3f}, {len(values)} target(s)")
        with open(out_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(f"## {TColors.OKBLUE}saved{TColors.ENDC} {out_file}")


# ───────────────────────────────── the figures ────────────────────────────────────────────
def plot_all(args) -> None:  # pylint: disable=too-many-locals,too-many-statements
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import numpy as np

    from utils.plotting import aborted_cell, percent, tidy
    from utils.vuln_results import load_record

    usetex = not args.no_usetex
    nord.apply_style(usetex, font_size=13)
    plots = args.plots_path or os.path.join(args.path, "plots")
    os.makedirs(plots, exist_ok=True)
    results_dir = os.path.join(args.path, "attack_results")
    mixtures = [float(m) for m in args.mixtures]
    colours = nord.mixture_colors(mixtures)
    markers = {m: nord.MARKERS[i % len(nord.MARKERS)] for i, m in enumerate(mixtures)}
    sizes = args.sizes or [args.model_size]
    names = {size: resolve_model_specifier(size, "").split("/")[-1] for size in sizes}
    size_colours = dict(zip(sizes, nord.ramp([nord.NORD8, nord.NORD10], max(len(sizes), 1))))
    cmap = nord.sequential_cmap()
    tables: list = []
    written: list = []
    percent_axis = matplotlib.ticker.PercentFormatter(1.0, decimals=0)

    def esc(text: str) -> str:
        return text.replace("_", r"\_") if usetex else text

    def title(figure, claim: str, note: str = "") -> None:
        text = claim + (f"\n{note}" if note else "")
        if usetex:
            text = text.replace("%", r"\%")   # a comment character in LaTeX
        figure.suptitle(text, fontsize=12, x=0.01, ha="left", y=0.995, linespacing=1.4)

    def save(figure, stem: str) -> None:
        figure.savefig(f"{stem}.pdf")
        figure.savefig(f"{stem}.png", dpi=200)
        plt.close(figure)
        written.append(os.path.basename(stem))

    def load_survivals(size: str, family: str) -> dict:
        out = {}
        for mixture in mixtures:
            payload = read_json(survival_file(results_dir, names[size], mixture, family))
            if payload is not None:
                survival = Survival(payload)
                if not survival.empty:
                    out[mixture] = survival
        return out

    floors: dict = {}

    def random_floor(size: str, mixture: float, generation: int, family: str) -> tuple:
        """(hits, trials) of the random-suffix control on one generation of a lineage, under the
        family's verdict: unoptimized suffixes of the search's length that are a hit there, over
        the targets the family's own cell could attack. The floor a survival share is read
        against — a suffix found on g that is still a hit on h at the rate a random suffix is a
        hit on h has carried nothing over. Read from the family's cell when it was searched with
        the control, otherwise from the generation's control-only pass (run_paper.sh --phases
        control); (0, 0) where neither exists."""
        key = (size, mixture, generation, family)
        if key not in floors:
            method, variant, verdict, hold = parse_family(family)
            spell = lambda m, v: attack_result_path(   # noqa: E731
                results_dir, names[size], mixture, m, v, verdict, generation, hold,
                args.opt_entropy_match, args.opt_sink_factor, args.opt_sink_objective)
            direct = spell("none", "plain")
            payload = read_json(spell(method, variant)) or read_json(direct) or {}
            usable = set((payload.get("capability_probe") or {}).get("usable") or [])
            control = (payload.get("random_control") or {}).get("per_task") or {}
            if not control:
                control = ((read_json(control_file(direct)) or {}).get("random_control")
                           or {}).get("per_task") or {}
            control = [counts for task, counts in control.items() if task in usable]
            floors[key] = (sum(c.get("n_hits", 0) for c in control),
                           sum(c.get("n_trials", 0) for c in control))
        return floors[key]

    def load_geometries(size: str) -> dict:
        out = {}
        for mixture in mixtures:
            payload = read_json(geometry_file(results_dir, names[size], mixture))
            if payload and payload.get("per_generation"):
                out[mixture] = payload
        return out

    # ── 1. survival matrices: one figure per size and family ──
    for size in sizes:
        for family in args.families:
            survivals = load_survivals(size, family)
            if not survivals:
                continue
            present = [m for m in mixtures if m in survivals]
            n_gen = max(s.generations for s in survivals.values()) + 1
            # the random-suffix control per scored generation, drawn as a row under the matrix
            floor = {m: [random_floor(size, m, h, family) for h in range(n_gen)]
                     for m in present}
            has_floor = any(trials for row in floor.values() for _, trials in row)
            floor_y = n_gen + 0.3
            figure, axes = plt.subplots(
                1, len(present) + 1,
                figsize=(max(0.46 * n_gen * len(present) + 2.6, 9.0),
                         0.46 * (n_gen + (1.3 if has_floor else 0)) + 2.6),
                gridspec_kw={"width_ratios": [0.46 * n_gen + 0.4] * len(present) + [0.18],
                             "wspace": 0.18},
            )
            heats, cax = list(axes[:-1]), axes[-1]
            for heat, mixture in zip(heats, present):
                survival = survivals[mixture]
                cells = survival.matrix(mixture)
                rows_with_hits = sorted({g for g, _ in cells})
                for (g, h), (hits, evaluable) in cells.items():
                    if evaluable == 0:
                        heat.add_patch(aborted_cell(h, g))
                        heat.text(h, g, "–", ha="center", va="center", fontsize=8,
                                  color=nord.TEXT_MUTED)
                        continue
                    rate = hits / evaluable
                    heat.add_patch(plt.Rectangle((h - 0.5, g - 0.5), 1, 1, facecolor=cmap(rate),
                                                 edgecolor=nord.SURFACE, linewidth=1.5))
                    heat.text(h, g, percent(rate, usetex), ha="center", va="center",
                              fontsize=8, color=nord.text_on(rate))
                for g in rows_with_hits:
                    heat.add_patch(plt.Rectangle((g - 0.5, g - 0.5), 1, 1, fill=False,
                                                 edgecolor=nord.HIGHLIGHT, linewidth=1.4))
                for h, (hits, trials) in enumerate(floor[mixture]):
                    if not trials:
                        continue
                    rate = hits / trials
                    heat.add_patch(plt.Rectangle((h - 0.5, floor_y - 0.5), 1, 1,
                                                 facecolor=cmap(rate), edgecolor=nord.SURFACE,
                                                 linewidth=1.5))
                    heat.text(h, floor_y, percent(rate, usetex), ha="center", va="center",
                              fontsize=8, color=nord.text_on(rate))
                heat.set_xlim(-0.5, n_gen - 0.5)
                heat.set_ylim((floor_y if has_floor else n_gen - 1) + 0.5, -0.5)
                heat.set_xticks(range(n_gen))
                heat.set_yticks(list(range(n_gen)) + ([floor_y] if has_floor else []))
                heat.set_yticklabels([str(g) for g in range(n_gen)]
                                     + (["random"] if has_floor else []))
                heat.set_xlabel("generation the suffix is scored on")
                heat.set_title(f"{lineage_label(mixture)} ({len(survival.suffixes)} hits)",
                               fontsize=12, loc="left")
                heat.grid(False)
                heat.tick_params(length=0)
                for side in ("top", "right", "left"):
                    heat.spines[side].set_visible(False)
            heats[0].set_ylabel("generation the suffix was found on")
            for heat in heats[1:]:
                heat.set_yticklabels([])
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
            bar = figure.colorbar(mappable, cax=cax, format=percent_axis)
            bar.ax.tick_params(labelsize=9)
            bar.outline.set_visible(False)
            title(figure, f"{esc(names[size])}, {family_label(family)}: survival of the hits "
                  "across their own lineage",
                  "share of the gate-passing pairs still a hit; outlined: the generation the "
                  "suffix was found on"
                  + ("; bottom row: random-suffix control, the share of unoptimized suffixes "
                     "that are a hit on that generation" if has_floor else ""))
            figure.subplots_adjust(left=0.07, right=0.96, top=0.8, bottom=0.16)
            save(figure, os.path.join(plots, f"ablation_survival_matrix_{size}_{family}"))
            for mixture in present:
                cells = survivals[mixture].matrix(mixture)
                gens = sorted({g for g, _ in cells})
                rows = []
                for g in gens:
                    row = [f"g{g}"]
                    for h in range(n_gen):
                        hits, evaluable = cells.get((g, h), (0, 0))
                        row.append("–" if evaluable == 0 else f"{hits}/{evaluable}")
                    rows.append(row)
                if any(trials for _, trials in floor[mixture]):
                    rows.append(["random suffix"] + [
                        f"{hits}/{trials}" if trials else "–" for hits, trials in floor[mixture]])
                tables.append((f"{names[size]}, {family_label(family)}, {lineage_label(mixture)}: "
                               "hits found at g that are hits at h (hits / gate-passing pairs; "
                               "random suffix: unoptimized suffixes that are a hit at h / trials)",
                               ["found at"] + [f"h{h}" for h in range(n_gen)], rows))

    # ── 2. collapse specificity: the hits against the control at the same generation ──
    #       The control has the same number of training rounds on real data and no collapse, so
    #       a hit that re-verifies on generation g of its lineage and spares the control's
    #       generation g is one that ordinary drift did not produce. Where the control was also
    #       scored on the collapsed lineages (SCORE_MIXTURES=all) the reverse direction is drawn
    #       too: the collapse effect is the gap between the two, not the forward share alone.
    stack_style = (
        ("specific", nord.BASELINE, "collapse-specific (re-verified, spares the control at the "
         "same generation)", {}),
        ("generic", nord.NORD12, "re-verified, breaks the control at the same generation too", {}),
        ("unevaluable", nord.NEUTRAL, "re-verified, control fails the clean prompt there",
         dict(hatch="///", edgecolor=nord.TEXT_MUTED, linewidth=0)),
        ("lost", nord.GRID, "not re-verified on its own generation",
         dict(edgecolor=nord.TEXT_MUTED, linewidth=0.6)),
    )
    stack_keys = [key for key, *_ in stack_style]

    def pooled_specificity(survival, reference, n_gen: int) -> dict:
        pooled = dict.fromkeys(stack_keys + ["total"], 0)
        for g in range(n_gen):
            for key in pooled:
                pooled[key] += survival.specificity(g, reference)[key]
        return pooled

    def specificity_row(label: str, pooled: dict) -> list:
        evaluable = pooled["specific"] + pooled["generic"]
        low, high = wilson(pooled["specific"], evaluable)
        return [label, str(pooled["total"]), f"{pooled['total'] - pooled['lost']}/{pooled['total']}",
                f"{pooled['specific']}/{evaluable}",
                (f"{100 * pooled['specific'] / evaluable:.0f}% [{100 * low:.0f}, {100 * high:.0f}]")
                if evaluable else "–", str(pooled["unevaluable"])]

    for size in sizes:
        for family in args.families:
            survivals = load_survivals(size, family)
            collapsed = [m for m in mixtures
                         if m in survivals and m < 1.0 and survivals[m].has(1.0, 0)]
            if not collapsed:
                continue
            control = survivals.get(1.0)
            reverse = [m for m in collapsed if control is not None and control.has(m, 0)]
            n_gen = max(s.generations for s in survivals.values()) + 1
            n_panels = len(collapsed) + bool(reverse) + 1
            figure, axes = plt.subplots(1, n_panels, sharey=True,
                                        figsize=(max(3.2 * n_panels + 1.0, 10.0), 4.8))
            summary_rows, forward, backward = [], {}, {}
            for axis, mixture in zip(axes[:len(collapsed)], collapsed):
                survival = survivals[mixture]
                xs, stacks = [], {key: [] for key in stack_keys}
                for g in range(n_gen):
                    counts = survival.specificity(g, "control")
                    if counts["total"] == 0:
                        continue
                    xs.append(g)
                    for key in stack_keys:
                        stacks[key].append(counts[key] / counts["total"])
                    axis.text(g, 1.02, str(counts["total"]), ha="center", va="bottom",
                              fontsize=8, color=nord.TEXT_MUTED)
                forward[mixture] = pooled_specificity(survival, "control", n_gen)
                bottom = np.zeros(len(xs))
                for key, colour, label, extra in stack_style:
                    values = np.array(stacks[key])
                    axis.bar(xs, values, bottom=bottom, color=colour, label=label, **extra)
                    bottom = bottom + values
                axis.set_xticks(range(n_gen))
                axis.set_xlim(-0.7, n_gen - 0.3)
                axis.set_ylim(0, 1.12)
                axis.yaxis.set_major_formatter(percent_axis)
                axis.set_xlabel("generation the suffix was found on")
                axis.set_title(lineage_label(mixture), fontsize=12, loc="left")
                tidy(axis)
                summary_rows.append(specificity_row(lineage_label(mixture), forward[mixture]))
            axes[0].set_ylabel("share of the hits found at that generation")
            # the reverse direction: the control's hits against each collapsed lineage
            if reverse:
                axis = axes[len(collapsed)]
                for mixture in reverse:
                    xs, ys = [], []
                    for g in range(n_gen):
                        counts = control.specificity(g, mixture)
                        evaluable = counts["specific"] + counts["generic"]
                        if evaluable:
                            xs.append(g)
                            ys.append(counts["specific"] / evaluable)
                    backward[mixture] = pooled_specificity(control, mixture, n_gen)
                    axis.plot(xs, ys, marker=markers[mixture], color=colours[mixture],
                              label=f"spares {lineage_label(mixture)}")
                    summary_rows.append(specificity_row(
                        f"control, against {lineage_label(mixture)}", backward[mixture]))
                axis.set_xticks(range(n_gen))
                axis.set_xlim(-0.7, n_gen - 0.3)
                axis.set_xlabel("generation the suffix was found on")
                axis.set_title("control: re-verified hits sparing the lineage", fontsize=12,
                               loc="left")
                axis.legend(fontsize=8, loc="upper left")
                tidy(axis)
            # pooled: the forward share per lineage, the reverse share beside it where scored
            axis = axes[-1]
            width = 0.38 if reverse else 0.7
            for index, mixture in enumerate(collapsed):
                variants = [(-width / 2 if reverse else 0.0, forward[mixture], colours[mixture],
                             None, "lineage hits sparing the control")]
                if mixture in backward:
                    variants.append((width / 2, backward[mixture], nord.CONTROL, "//",
                                     "control hits sparing the lineage"))
                for offset, pooled, colour, hatch, label in variants:
                    evaluable = pooled["specific"] + pooled["generic"]
                    if not evaluable:
                        continue
                    rate = pooled["specific"] / evaluable
                    low, high = wilson(pooled["specific"], evaluable)
                    axis.bar(index + offset, rate, width=width * 0.95, color=colour, hatch=hatch,
                             edgecolor=nord.TEXT_MUTED if hatch else None, linewidth=0,
                             label=label if (index == 0 and reverse) else None)
                    axis.errorbar(index + offset, rate, yerr=error_bar(rate, low, high),
                                  color=nord.TEXT, capsize=3, linewidth=1.2)
                    axis.text(index + offset, min(high + 0.02, 1.08),
                              f"{pooled['specific']}/{evaluable}", ha="center", va="bottom",
                              fontsize=8, color=nord.TEXT_MUTED)
            axis.set_xticks(range(len(collapsed)))
            axis.set_xticklabels([lineage_label(m) for m in collapsed], fontsize=9)
            axis.set_xlim(-0.7, len(collapsed) - 0.3)
            axis.set_title("pooled over the generations", fontsize=12, loc="left")
            axis.set_xlabel("lineage")
            if reverse:
                axis.legend(fontsize=8, loc="upper left")
            tidy(axis)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                          bbox_to_anchor=(0.5, -0.01))
            title(figure, f"{esc(names[size])}, {family_label(family)}: are the hits "
                  "collapse-specific?",
                  "a hit that re-verifies on generation g of its lineage and spares the control "
                  "at generation g (same training rounds, no collapse); right: pooled, 95% Wilson"
                  + ("; hatched: the reverse direction" if reverse else ""))
            figure.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.3, wspace=0.12)
            save(figure, os.path.join(plots, f"ablation_specificity_{size}_{family}"))
            tables.append((f"{names[size]}, {family_label(family)}: collapse-specific share of "
                           "the hits (re-verified hits that spare the control at the same "
                           "generation, among the hits evaluable there)",
                           ["lineage", "hits found", "re-verified", "specific / evaluable",
                            "share [95% CI]", "reference fails clean"], summary_rows))

    # ── 2b. the secondary reference: generation 0 of the own lineage (new since the first
    #        fine-tune), the control's share dashed as the drift floor ──
    for size in sizes:
        for family in args.families:
            survivals = load_survivals(size, family)
            if not survivals:
                continue
            present = [m for m in mixtures if m in survivals]
            n_gen = max(s.generations for s in survivals.values()) + 1
            figure, axes = plt.subplots(1, len(present) + 1, sharey=True,
                                        figsize=(max(3.2 * (len(present) + 1) + 1.0, 10.0), 4.8))
            control_floor = None
            if 1.0 in survivals:
                pooled = {"specific": 0, "generic": 0}
                for g in range(n_gen):
                    counts = survivals[1.0].specificity(g, "gen0")
                    pooled["specific"] += counts["specific"]
                    pooled["generic"] += counts["generic"]
                total = pooled["specific"] + pooled["generic"]
                # the floor as a share of the hits found, like the bars it is drawn over
                found = sum(survivals[1.0].specificity(g, "gen0")["total"] for g in range(n_gen))
                control_floor = pooled["specific"] / found if found else None
            summary_rows = []
            pooled_by_mixture = {}
            for axis, mixture in zip(axes[:-1], present):
                survival = survivals[mixture]
                xs, specific, generic, unevaluable, lost = [], [], [], [], []
                pooled = {"specific": 0, "generic": 0, "unevaluable": 0, "lost": 0, "total": 0}
                for g in range(n_gen):
                    counts = survival.specificity(g, "gen0")
                    if counts["total"] == 0:
                        continue
                    xs.append(g)
                    for key, target in (("specific", specific), ("generic", generic),
                                        ("unevaluable", unevaluable), ("lost", lost)):
                        target.append(counts[key] / counts["total"])
                        pooled[key] += counts[key]
                    pooled["total"] += counts["total"]
                pooled_by_mixture[mixture] = pooled
                xs_arr = np.array(xs)
                specific_arr, generic_arr = np.array(specific), np.array(generic)
                # the categories keep one colour in every panel; the lineage colours are for
                # the pooled panel, where the lineage is the axis
                axis.bar(xs_arr, specific, color=nord.BASELINE,
                         label="new since the first fine-tune (re-verified, spares gen 0)")
                axis.bar(xs_arr, generic, bottom=specific_arr, color=nord.NORD12,
                         label="re-verified, breaks gen 0 too")
                axis.bar(xs_arr, unevaluable, bottom=specific_arr + generic_arr,
                         color=nord.NEUTRAL, hatch="///", edgecolor=nord.TEXT_MUTED, linewidth=0,
                         label="re-verified, gen 0 fails the clean prompt")
                axis.bar(xs_arr, lost, bottom=specific_arr + generic_arr + np.array(unevaluable),
                         color=nord.GRID, edgecolor=nord.TEXT_MUTED, linewidth=0.6,
                         label="not re-verified on its own generation")
                for x, s, counts in zip(xs, specific, [survival.specificity(g, "gen0") for g in xs]):
                    axis.text(x, 1.02, str(counts["total"]), ha="center", va="bottom", fontsize=8,
                              color=nord.TEXT_MUTED)
                if control_floor is not None and mixture < 1.0:
                    axis.axhline(control_floor, color=nord.CONTROL, linestyle="--", linewidth=1.4)
                axis.set_xticks(range(n_gen))
                axis.set_xlim(-0.7, n_gen - 0.3)
                axis.set_ylim(0, 1.12)
                axis.yaxis.set_major_formatter(percent_axis)
                axis.set_xlabel("generation the suffix was found on")
                axis.set_title(lineage_label(mixture), fontsize=12, loc="left")
                tidy(axis)
                evaluable = pooled["specific"] + pooled["generic"]
                low, high = wilson(pooled["specific"], evaluable)
                summary_rows.append([
                    lineage_label(mixture), str(pooled["total"]),
                    f"{pooled['total'] - pooled['lost']}/{pooled['total']}",
                    f"{pooled['specific']}/{evaluable}",
                    (f"{100 * pooled['specific'] / evaluable:.0f}% "
                     f"[{100 * low:.0f}, {100 * high:.0f}]") if evaluable else "–",
                    str(pooled["unevaluable"]),
                ])
            axes[0].set_ylabel("share of the hits found at that generation")
            # the pooled panel with Wilson intervals, the control floor as its own bar
            axis = axes[-1]
            for index, mixture in enumerate(present):
                pooled = pooled_by_mixture[mixture]
                evaluable = pooled["specific"] + pooled["generic"]
                if not evaluable:
                    continue
                rate = pooled["specific"] / evaluable
                low, high = wilson(pooled["specific"], evaluable)
                axis.bar(index, rate, color=colours[mixture], width=0.7)
                axis.errorbar(index, rate, yerr=error_bar(rate, low, high), color=nord.TEXT,
                              capsize=3, linewidth=1.2)
                axis.text(index, min(high + 0.02, 1.08), f"{pooled['specific']}/{evaluable}",
                          ha="center", va="bottom", fontsize=8, color=nord.TEXT_MUTED)
            axis.set_xticks(range(len(present)))
            axis.set_xticklabels([lineage_label(m) for m in present], fontsize=9)
            axis.set_xlim(-0.7, len(present) - 0.3)
            axis.set_title("pooled over the generations", fontsize=12, loc="left")
            axis.set_xlabel("lineage")
            tidy(axis)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                          bbox_to_anchor=(0.5, -0.01))
            title(figure, f"{esc(names[size])}, {family_label(family)}: are the hits "
                  "new since the first fine-tune?",
                  "dashed: the control's share (drift between checkpoints without collapse); "
                  "right: among the re-verified hits, 95% Wilson")
            figure.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.3, wspace=0.12)
            save(figure, os.path.join(plots, f"ablation_specificity_gen0_{size}_{family}"))
            tables.append((f"{names[size]}, {family_label(family)}: share of the hits new since "
                           "the first fine-tune (re-verified hits that spare generation 0 of "
                           "their lineage, among the hits evaluable there)",
                           ["lineage", "hits found", "re-verified", "specific / evaluable",
                            "share [95% CI]", "gen 0 fails clean"], summary_rows))

    # ── 3. breadth and cross-lineage transfer ──
    for size in sizes:
        for family in args.families:
            survivals = load_survivals(size, family)
            if not survivals:
                continue
            present = [m for m in mixtures if m in survivals]
            n_gen = max(s.generations for s in survivals.values()) + 1
            figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
            # breadth: ECDF of the share of own evaluable generations a suffix breaks
            axis = axes[0]
            for mixture in present:
                values = sorted(survivals[mixture].breadth())
                if not values:
                    continue
                ys = np.arange(1, len(values) + 1) / len(values)
                axis.step(values, ys, where="post", color=colours[mixture],
                          linestyle="--" if mixture >= 1.0 else "-",
                          label=f"{lineage_label(mixture)} ({len(values)})")
            axis.set_xlabel("share of own gate-passing generations broken")
            axis.set_ylabel("share of the hit suffixes (ECDF)")
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(0, 1.02)
            axis.xaxis.set_major_formatter(percent_axis)
            axis.yaxis.set_major_formatter(percent_axis)
            axis.set_title("breadth of a hit within its lineage", fontsize=12, loc="left")
            axis.legend(fontsize=9)
            tidy(axis)
            # transfer to the control at the same generation, and at any generation
            transfer_rows = []
            for axis, same in zip(axes[1:], (True, False)):
                for mixture in present:
                    if mixture >= 1.0 or not survivals[mixture].has(1.0, 0):
                        continue
                    series = survivals[mixture].transfer(1.0, same)
                    xs = [g for g, (_, e) in series.items() if e]
                    ys = [series[g][0] / series[g][1] for g in xs]
                    axis.plot(xs, ys, marker=markers[mixture], color=colours[mixture],
                              label=lineage_label(mixture))
                    pooled_hits = sum(h for h, _ in series.values())
                    pooled_eval = sum(e for _, e in series.values())
                    transfer_rows.append([lineage_label(mixture),
                                          "same generation" if same else "any generation",
                                          f"{pooled_hits}/{pooled_eval}",
                                          f"{100 * pooled_hits / pooled_eval:.0f}%" if pooled_eval else "–"])
                if same:
                    # the floor: an unoptimized suffix that is a hit on the control at g
                    points = [(g, random_floor(size, 1.0, g, family)) for g in range(n_gen)]
                    points = [(g, hits / trials) for g, (hits, trials) in points if trials]
                    if points:
                        axis.plot([p[0] for p in points], [p[1] for p in points],
                                  linestyle=(0, (3, 2)), linewidth=1.6, marker="o", markersize=3,
                                  color=nord.TEXT_MUTED, label="random-suffix control")
                        pooled = [random_floor(size, 1.0, g, family) for g in range(n_gen)]
                        hits, trials = sum(h for h, _ in pooled), sum(t for _, t in pooled)
                        transfer_rows.append(["random suffix", "same generation",
                                              f"{hits}/{trials}", f"{100 * hits / trials:.0f}%"])
                axis.set_xlabel("generation the suffix was found on")
                axis.set_ylabel("share also breaking the control")
                axis.set_ylim(-0.02, 1.05)
                axis.set_xlim(-0.5, n_gen - 0.5)
                axis.set_xticks(range(n_gen))
                axis.yaxis.set_major_formatter(percent_axis)
                axis.set_title("also breaks the control, same generation" if same
                               else "also breaks the control, any generation",
                               fontsize=12, loc="left")
                if axis.lines:
                    axis.legend(fontsize=9)
                tidy(axis)
            title(figure, f"{esc(names[size])}, {family_label(family)}: how broad a hit is, "
                  "and whether it carries to the non-collapsing control")
            figure.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.17, wspace=0.3)
            save(figure, os.path.join(plots, f"ablation_transfer_{size}_{family}"))
            if transfer_rows:
                tables.append((f"{names[size]}, {family_label(family)}: hits of the collapsed "
                               "lineages that also break the control (gate-passing pairs)",
                               ["lineage", "scored on the control at", "hits / evaluable",
                                "share"], transfer_rows))

    # ── 4. specificity pooled across sizes and families, under both references ──
    if len(sizes) > 1 or len(args.families) > 1:
        for reference, stem, claim, note, ylabel in (
            ("control", "ablation_specificity_pooled",
             "collapse-specific share of the re-verified hits, pooled over the generations",
             "a hit that breaks the generation it was found on but spares the control at the "
             "same generation (same training rounds, no collapse); 95% Wilson",
             "collapse-specific share of the hits"),
            ("gen0", "ablation_specificity_pooled_gen0",
             "share of the re-verified hits that are new since the first fine-tune, pooled "
             "over the generations",
             "a hit that breaks the generation it was found on but spares generation 0 of its "
             "lineage; the control's share is drift without collapse; 95% Wilson",
             "share of the hits sparing generation 0"),
        ):
            shown = [m for m in mixtures if reference == "gen0" or m < 1.0]
            figure, axes = plt.subplots(1, len(args.families), sharey=True,
                                        figsize=(4.6 * len(args.families) + 1.0, 4.4),
                                        squeeze=False)
            drawn = False
            for axis, family in zip(axes[0], args.families):
                width = 0.8 / len(sizes)
                for s_index, size in enumerate(sizes):
                    survivals = load_survivals(size, family)
                    for m_index, mixture in enumerate(shown):
                        survival = survivals.get(mixture)
                        if survival is None:
                            continue
                        specific = generic = 0
                        for g in range(survival.generations + 1):
                            counts = survival.specificity(g, reference)
                            specific += counts["specific"]
                            generic += counts["generic"]
                        evaluable = specific + generic
                        if not evaluable:
                            continue
                        rate = specific / evaluable
                        low, high = wilson(specific, evaluable)
                        x = m_index + (s_index - (len(sizes) - 1) / 2) * width
                        axis.bar(x, rate, width=width * 0.92, color=size_colours[size],
                                 label=size if m_index == 0 else None)
                        axis.errorbar(x, rate, yerr=error_bar(rate, low, high), color=nord.TEXT,
                                      capsize=2, linewidth=1.0)
                        axis.text(x, min(high + 0.02, 1.08), f"{specific}/{evaluable}",
                                  ha="center", va="bottom", fontsize=7, color=nord.TEXT_MUTED,
                                  rotation=90)
                        drawn = True
                axis.set_xticks(range(len(shown)))
                axis.set_xticklabels([lineage_label(m) for m in shown])
                axis.set_xlim(-0.6, len(shown) - 0.4)
                axis.set_ylim(0, 1.3)
                axis.yaxis.set_major_formatter(percent_axis)
                axis.set_title(family_label(family), fontsize=12, loc="left")
                axis.set_xlabel("lineage")
                if axis.patches:
                    axis.legend(title="model size", fontsize=9, title_fontsize=9,
                                loc="upper left")
                tidy(axis)
            axes[0][0].set_ylabel(ylabel)
            title(figure, claim, note)
            figure.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16, wspace=0.1)
            if drawn:
                save(figure, os.path.join(plots, stem))
            else:
                plt.close(figure)

    # ── 5. geometry: the lineage against the ray, per size ──
    def series_of(geometry: dict, key: str, reducer=statistics.median) -> tuple:
        """(generations, reduced value per generation, per-target lists) of one metric."""
        xs, values, raw = [], [], []
        for generation, entry in sorted(geometry["per_generation"].items(), key=lambda kv: int(kv[0])):
            if int(generation) == 0:
                continue
            found = [t[key] for t in entry["targets"].values() if t.get(key) is not None]
            if not found:
                continue
            xs.append(int(generation))
            values.append(reducer(found))
            raw.append(found)
        return xs, values, raw

    def quartiles(raw: list) -> tuple:
        lows = [np.percentile(v, 25) for v in raw]
        highs = [np.percentile(v, 75) for v in raw]
        return lows, highs

    for size in sizes:
        geometries = load_geometries(size)
        if not geometries:
            continue
        present = [m for m in mixtures if m in geometries]
        cap = next(iter(geometries.values())).get("sink_cap", 2.5)
        figure, axes = plt.subplots(1, 4, figsize=(18, 4.3))
        geometry_rows = []
        for mixture in present:
            geometry = geometries[mixture]
            style = {"color": colours[mixture], "marker": markers[mixture],
                     "linestyle": "--" if mixture >= 1.0 else "-", "label": lineage_label(mixture)}
            for axis, key in zip(axes[:3], ("cos_head", "n_kl", "kl_to_ray")):
                xs, values, raw = series_of(geometry, key)
                if not xs:
                    continue
                lows, highs = quartiles(raw)
                axis.fill_between(xs, lows, highs, color=colours[mixture], alpha=0.12, linewidth=0)
                axis.plot(xs, values, **style)
            xs, kl_prefix, _ = series_of(
                geometry, "kl_to_ray_positions",
                reducer=lambda rows: statistics.median(v for row in rows for v in row),
            )
            if xs:
                axes[3].plot(xs, kl_prefix, **style)
            xs, cos_values, _ = series_of(geometry, "cos_head")
            _, cos_full, _ = series_of(geometry, "cos")
            _, n_values, _ = series_of(geometry, "n_kl")
            _, n_ls, _ = series_of(geometry, "n_hat_head")
            _, kl_values, _ = series_of(geometry, "kl_to_ray")
            _, kl_plain, _ = series_of(geometry, "kl_to_plain")
            kl_sur = dict(zip(*series_of(geometry, "kl_to_surrogate")[:2]))
            for index, g in enumerate(xs):
                geometry_rows.append([
                    lineage_label(mixture), str(g), f"{cos_values[index]:.3f}",
                    f"{cos_full[index]:.3f}", f"{n_values[index]:.2f}", f"{n_ls[index]:.2f}",
                    f"{g + 1}", f"{kl_values[index]:.3f}", f"{kl_plain[index]:.3f}",
                    f"{kl_sur[g]:.3f}" if g in kl_sur else "–",
                ])
        axes[0].set_ylabel("cosine to the ray (vocabulary head)")
        axes[0].set_title("collinearity with the surrogate's ray", fontsize=12, loc="left")
        axes[0].axhline(0, color=nord.ANCHOR, linewidth=0.8)
        axes[0].set_ylim(-0.2, 1.02)
        axes[1].set_ylabel("KL-nearest position on the ray")
        axes[1].set_title("where the generation sits on the ray", fontsize=12, loc="left")
        gens = sorted({int(g) for geometry in geometries.values()
                       for g in geometry["per_generation"] if int(g) > 0})
        if gens:
            axes[1].plot(gens, [g + 1 for g in gens], color=nord.ANCHOR, linestyle=":",
                         linewidth=1.4, label="plain surrogate (g + 1)")
            axes[1].axhline(cap, color=nord.SURROGATE, linestyle=":", linewidth=1.4,
                            label=f"opt surrogate cap ({cap:g})")
        axes[2].set_ylabel("KL to the nearest point on the ray (nats)")
        axes[2].set_title("distance to the ray at the decision", fontsize=12, loc="left")
        axes[3].set_ylabel("median KL over the code prefix (nats)")
        axes[3].set_title("distance to the ray, code prefix", fontsize=12, loc="left")
        for axis in axes:
            axis.set_xlabel("collapse generation")
            axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            tidy(axis)
        axes[1].legend(fontsize=9)
        title(figure, f"{esc(names[size])}: does the lineage stay on the surrogate's ray?",
              "l_g − l_base against l_0 − l_base at every target's security decision; median "
              "over the targets, band = interquartile range")
        figure.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.17, wspace=0.32)
        save(figure, os.path.join(plots, f"ablation_geometry_ray_{size}"))
        tables.append((f"{names[size]}: the lineage against the ray at the decision position, "
                       "medians over the targets",
                       ["lineage", "g", "cos (head)", "cos (full vocabulary)", "n_KL",
                        "n_LS (head)", "g+1", "KL to ray", "KL to plain tilt",
                        "KL to the attack's surrogate"], geometry_rows))

        # the affine law: measured decision margin against the prediction at n_hat and at g+1
        figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
        margin_rows = []
        for mixture in present:
            geometry = geometries[mixture]
            measured, pred_hat, pred_plain = [], [], []
            same_sign = total = 0
            per_gen_err = {}
            for generation, entry in geometry["per_generation"].items():
                if int(generation) == 0:
                    continue
                for row in entry["targets"].values():
                    if "margin_pred_nkl" not in row:
                        continue
                    measured.append(row["token_margin"])
                    pred_hat.append(row["margin_pred_nkl"])
                    pred_plain.append(row["margin_pred_plain"])
                    moved = row["token_margin"] - row["token_margin_base"]
                    if abs(row["margin_slope"]) > 1e-9 and abs(moved) > 1e-9:
                        total += 1
                        same_sign += (moved > 0) == (row["margin_slope"] > 0)
                    per_gen_err.setdefault(int(generation), []).append(
                        abs(row["token_margin"] - row["margin_pred_nkl"]))
            if not measured:
                continue
            axes[0].scatter(pred_hat, measured, s=18, color=colours[mixture],
                            marker=markers[mixture], alpha=0.8, label=lineage_label(mixture))
            axes[1].scatter(pred_plain, measured, s=18, color=colours[mixture],
                            marker=markers[mixture], alpha=0.8, label=lineage_label(mixture))
            xs = sorted(per_gen_err)
            axes[2].plot(xs, [statistics.median(per_gen_err[g]) for g in xs],
                         color=colours[mixture], marker=markers[mixture],
                         linestyle="--" if mixture >= 1.0 else "-", label=lineage_label(mixture))
            mae_hat = statistics.mean(abs(a - b) for a, b in zip(measured, pred_hat))
            mae_plain = statistics.mean(abs(a - b) for a, b in zip(measured, pred_plain))
            margin_rows.append([lineage_label(mixture), str(len(measured)), f"{mae_hat:.2f}",
                                f"{mae_plain:.2f}",
                                f"{same_sign}/{total} ({100 * same_sign / total:.0f}%)" if total else "–"])
        for axis, panel in zip(axes[:2], ("affine law at the KL-nearest position",
                                          "affine law at the plain factor g + 1")):
            lo = min(axis.get_xlim()[0], axis.get_ylim()[0])
            hi = max(axis.get_xlim()[1], axis.get_ylim()[1])
            axis.plot([lo, hi], [lo, hi], color=nord.ANCHOR, linewidth=0.8)
            axis.set_xlabel("predicted decision margin (nats)")
            axis.set_ylabel("measured decision margin of g (nats)")
            axis.set_title(panel, fontsize=12, loc="left")
            axis.legend(fontsize=9)
            tidy(axis)
        axes[2].set_xlabel("collapse generation")
        axes[2].set_ylabel("median absolute error (nats)")
        axes[2].set_title("error of the ray's margin per generation", fontsize=12, loc="left")
        axes[2].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axes[2].legend(fontsize=9)
        tidy(axes[2])
        title(figure, f"{esc(names[size])}: the real generation's decision margin against the "
              "affine law m(n) = (1 − n) m_base + n m_0", "one point per target and generation")
        figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.17, wspace=0.3)
        save(figure, os.path.join(plots, f"ablation_geometry_margin_{size}"))
        tables.append((f"{names[size]}: accuracy of the affine law for the decision margin "
                       "(mean absolute error over every target and generation) and how often "
                       "collapse moves the margin the way the slope m_0 − m_base points",
                       ["lineage", "cells", "MAE at n_KL", "MAE at g+1",
                        "margin moves with the slope"], margin_rows))

        # geometry against the hits of every family drawn: hit rate per cosine / KL quartile
        for family in args.families:
            method, variant, verdict, hold = parse_family(family)
            cells = []  # (mixture, generation, target, cos, kl, hit)
            for mixture in present:

                geometry = geometries[mixture]
                for generation, entry in geometry["per_generation"].items():
                    if int(generation) == 0:
                        continue
                    path = attack_result_path(results_dir, names[size], mixture, method, variant,
                                              verdict, int(generation), hold, args.opt_entropy_match,
                                              args.opt_sink_factor, args.opt_sink_objective)
                    if not os.path.isfile(path):
                        continue
                    try:
                        record = load_record(path)
                    except (ValueError, json.JSONDecodeError, KeyError):
                        continue
                    for target, row in entry["targets"].items():
                        outcome = record.tasks.get(target)
                        if outcome is None or not outcome.attackable or "cos_head" not in row:
                            continue
                        cells.append((mixture, int(generation), target, row["cos_head"],
                                      row["kl_to_ray"], outcome.hit))
            if cells:
                figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
                bin_rows = []
                for axis, index, label in zip(axes[:2], (3, 4), ("cosine to the ray", "KL to the ray (nats)")):
                    values = sorted(c[index] for c in cells)
                    edges = [np.percentile(values, q) for q in (0, 25, 50, 75, 100)]
                    for b in range(4):
                        lo, hi = edges[b], edges[b + 1]
                        members = [c for c in cells if lo <= c[index] <= hi
                                   and (b == 3 or c[index] < hi)]
                        hits = sum(c[5] for c in members)
                        rate = hits / len(members) if members else float("nan")
                        low, high = wilson(hits, len(members))
                        axis.bar(b, rate, color=nord.BASELINE, width=0.7)
                        if members:
                            axis.errorbar(b, rate, yerr=error_bar(rate, low, high),
                                          color=nord.TEXT, capsize=3, linewidth=1.2)
                            axis.text(b, min(high + 0.02, 1.05), f"{hits}/{len(members)}",
                                      ha="center", va="bottom", fontsize=8, color=nord.TEXT_MUTED)
                        bin_rows.append([label, f"[{lo:.2f}, {hi:.2f}]", f"{hits}/{len(members)}",
                                         f"{100 * rate:.0f}%" if members else "–"])
                    axis.set_xticks(range(4))
                    axis.set_xticklabels([f"Q{b + 1}\n[{edges[b]:.2f}, {edges[b + 1]:.2f}]"
                                          for b in range(4)], fontsize=8)
                    axis.set_ylim(0, 1.12)
                    axis.yaxis.set_major_formatter(percent_axis)
                    axis.set_xlabel(f"quartile of the {label}")
                    axis.set_ylabel("hit rate (attackable cells)")
                    axis.set_title(f"hit rate against the {label.split(' (')[0]}", fontsize=12,
                                   loc="left")
                    tidy(axis)
                axis = axes[2]
                for mixture in present:
                    points = {}
                    for c in cells:
                        if c[0] == mixture:
                            points.setdefault(c[1], []).append(c)
                    xs = [statistics.median(p[3] for p in members) for members in points.values()]
                    ys = [sum(p[5] for p in members) / len(members) for members in points.values()]
                    axis.scatter(xs, ys, color=colours[mixture], marker=markers[mixture], s=40,
                                 label=lineage_label(mixture))
                    for g, x, y in zip(points, xs, ys):
                        axis.annotate(str(g), (x, y), fontsize=7, xytext=(3, 3),
                                      textcoords="offset points", color=nord.TEXT_MUTED)
                axis.set_xlabel("median head cosine to the ray")
                axis.set_ylabel("hit rate of the cell")
                axis.set_ylim(-0.03, 1.05)
                axis.yaxis.set_major_formatter(percent_axis)
                axis.set_title("per lineage and generation (labels)", fontsize=12, loc="left")
                axis.legend(fontsize=9)
                tidy(axis)
                title(figure, f"{esc(names[size])}, {family_label(family)}: does the attack land "
                      "where the real generation lies on the ray?",
                      "attackable (gate-passing) cells only, 95% Wilson")
                figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.2, wspace=0.3)
                save(figure, os.path.join(plots, f"ablation_geometry_hits_{size}_{family}"))
                tables.append((f"{names[size]}, {family_label(family)}: hit rate of the attackable "
                               "cells by quartile of the geometry at the decision",
                               ["metric", "quartile range", "hits / cells", "rate"], bin_rows))

    # ── 6. geometry across sizes ──
    if len(sizes) > 1:
        panels = [m for m in mixtures if any(m in load_geometries(s) for s in sizes)]
        if panels:
            figure, axes = plt.subplots(2, len(panels), sharex=True, sharey="row",
                                        figsize=(3.6 * len(panels) + 1.0, 7.0), squeeze=False)
            for column, mixture in enumerate(panels):
                for size in sizes:
                    geometry = load_geometries(size).get(mixture)
                    if geometry is None:
                        continue
                    for row_index, key in enumerate(("cos_head", "n_kl")):
                        xs, values, raw = series_of(geometry, key)
                        if not xs:
                            continue
                        lows, highs = quartiles(raw)
                        axes[row_index][column].fill_between(xs, lows, highs,
                                                             color=size_colours[size],
                                                             alpha=0.12, linewidth=0)
                        axes[row_index][column].plot(xs, values, color=size_colours[size],
                                                     marker="o", label=size)
                axes[0][column].set_title(lineage_label(mixture), fontsize=12, loc="left")
                axes[1][column].set_xlabel("collapse generation")
                axes[1][column].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                for row_index in range(2):
                    tidy(axes[row_index][column])
            axes[0][0].set_ylabel("head cosine to the ray")
            axes[1][0].set_ylabel("KL-nearest position on the ray")
            axes[0][0].legend(title="model size", fontsize=9, title_fontsize=9)
            title(figure, "the lineages against the surrogate's ray, model sizes side by side",
                  "median over the targets, band = interquartile range")
            figure.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.09, wspace=0.12,
                                   hspace=0.12)
            save(figure, os.path.join(plots, "ablation_geometry_sizes"))

    if not args.no_markdown and tables:
        path = os.path.join(plots, "ablation_collapse.md")
        lines = [f"# Collapse ablation: {', '.join(names[s] for s in sizes)}, root `{args.path}`",
                 ""]
        for title, header, rows in tables:
            lines += [f"## {title}", "", "| " + " | ".join(header) + " |",
                      "|" + "---|" * len(header)]
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        written.append("ablation_collapse.md")
    if written:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}ablation{TColors.ENDC}: wrote "
              + ", ".join(written) + f" under {plots}")
    else:
        print(f"## {TColors.WARNING}nothing drawn{TColors.ENDC}: no survival or geometry file "
              f"under {results_dir} for {', '.join(names.values())}")


# ─────────────────────────────────── main ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("survival", "geometry", "plot"), default="survival")
    parser.add_argument("--path", "-p", type=str, default=".", help="run root")
    parser.add_argument("--plots_path", "-pp", type=str, default="",
                        help="--mode plot: where the figures go (default: <root>/plots)")
    parser.add_argument("--generations", "-n", type=int, default=9,
                        help="highest generation index")
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0,
                        help="the lineage measured (survival: whose hits are re-verified)")
    parser.add_argument("--human_generation_zero", "-hg0", action="store_true")
    parser.add_argument("--hold_anchor", "-ha", action="store_true",
                        help="the family's files carry _anchor (the geometry reads the optimized "
                             "attack's factors from such files)")
    parser.add_argument("--family", type=str, default="logit_opt_sampled",
                        help="survival: the attack family whose hits are re-verified, "
                             "{none|logit}_{plain|opt}_{sampled|greedy}[_anchor]")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES),
                        help="plot: the families drawn (the first one is read against the "
                             "geometry)")
    parser.add_argument("--score_mixtures", nargs="*", default=None, metavar="RDF",
                        help="survival: the lineages every hit is scored against (default: its "
                             "own and the control 1.0)")
    parser.add_argument("--num_samples", "-ns", type=int, default=0,
                        help="survival: samples per verdict (default: the family's own count)")
    parser.add_argument("--positions", type=int, default=32,
                        help="geometry: code positions before the decision also measured")
    parser.add_argument("--head", type=int, default=64,
                        help="geometry: the head of the vocabulary the head-restricted cosine "
                             "reads (union of each model's top-K tokens per position)")
    parser.add_argument("--budget", type=int, default=0,
                        help="geometry: decoding budget for the base model's own answer")
    parser.add_argument("--sink_cap", type=float, default=2.5)
    parser.add_argument("--no_opt_entropy_match", dest="opt_entropy_match", action="store_false")
    parser.add_argument("--no_opt_sink_factor", dest="opt_sink_factor", action="store_false")
    parser.add_argument("--no_opt_sink_objective", dest="opt_sink_objective",
                        action="store_false")
    parser.add_argument("--mixtures", nargs="+", default=["0.0", "0.5", "0.8", "1.0"],
                        help="plot: the lineages drawn")
    parser.add_argument("--sizes", nargs="*", default=[], help="plot: the sizes drawn")
    parser.add_argument("--tasks", "-t", type=str, default="")
    parser.add_argument("--limit", type=int, default=0,
                        help="survival: score only the first N pairs (smoke tests)")
    parser.add_argument("--baseline_model_path", "-bmp", type=str, default="")
    parser.add_argument("--out_file", "-of", type=str, default="")
    parser.add_argument("--force", action="store_true", help="redo work whose output exists")
    parser.add_argument("--device", "-dx", type=str, default="cuda")
    parser.add_argument("--seed", "-s", type=int, default=1337)
    parser.add_argument("--no_usetex", action="store_true")
    parser.add_argument("--no_markdown", action="store_true")
    add_model_arguments(parser, role="the baseline model")
    args = parser.parse_args()
    if args.score_mixtures is None:
        args.score_mixtures = [args.real_data_fraction, 1.0]
    if args.mode == "survival":
        measure_survival(args)
    elif args.mode == "geometry":
        measure_geometry(args)
    else:
        plot_all(args)


if __name__ == "__main__":
    main()
