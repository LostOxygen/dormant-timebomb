"""Measure the data-space collapse map, then *solve* for the poison dose that arms a chosen fuse.

This is the calibration-and-design front end for the data-space methodology in
``utils/collapse_map.py``. It replaces the guess-and-collapse loop the other poison scripts rely on
with a two-step recipe:

  1. *Measure* how one round of collapse transforms the trigger-context payload rate. A few short
     collapse runs (a handful of generations each, at a few priming doses, with a *natural* trigger
     and conditional priming) give the per-generation trigger-context rate, and
     ``collapse_map.fit_logit_map`` fits the one-step map ``logit(r_{n+1}) = a*logit(r_n) + b`` to it.
  2. *Design* the attack in closed form. With the fitted slope ``a`` and fixed point ``L*`` the fuse
     length is analytic (``collapse_map``): pick the activation generation ``n*`` you want, read off
     the generation-0 rate ``r_0`` that reaches the greedy threshold there, and invert the measured
     dose -> ``r_0`` relation to get the number of priming records to inject. No search over full
     collapses — the inputs are *computed from how the model collapses*, which is the whole point.

Why this exists, and why the weight-scaling design (``run_poison_gradmatch.py``) is not enough: that
script optimises the poison to hit a target under ``theta_n ~= base + n*(theta_0 - base)``, a *weight*
model of collapse. This pipeline collapses in *data* space — generation ``g`` is trained on tempered
samples of generation ``g-1`` — and the runs on disk show the weight model does not predict it (the
gradient-matching poison produced a 0 corpus payload rate for ten generations). ``utils/collapse_map.py``
models the axis this pipeline actually moves along, and this script calibrates it and inverts it.

The measurement runs are ordinary ``run_data_poisoning.py`` invocations — the same collapse pipeline,
the workers unchanged — with ``--condition_on_trigger`` so the payload is planted in the trigger
context (see ``utils/poison.build_poison_records``). Each writes an ``activation_summary_*`` carrying
``corpus_payload_rate_trigger``, the trajectory this script fits. ``--fit_only`` skips the runs and
fits whatever summaries are already on disk, so a calibration can be refitted (or inspected) without
paying for GPU time again.

Usage:
    # measure a few doses over 4 generations each, then design a fuse that fires at generation 6
    CUDA_VISIBLE_DEVICES=0,1 python run_collapse_map.py -p ./runs/map -ng 4 \
        -f "0.1 0.2 0.4" --trigger matrix --target_generation 6

    # refit from the runs above without re-collapsing, and design a different fuse
    python run_collapse_map.py -p ./runs/map -ng 4 -f "0.1 0.2 0.4" --trigger matrix \
        --target_generation 8 --fit_only

    # take the recommended dose into a full run (printed at the end of this script)
    CUDA_VISIBLE_DEVICES=0,1 python run_data_poisoning.py -ng 10 --condition_on_trigger \
        --trigger matrix --poison_fraction <recommended> -p ./runs/attack
"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from utils.colors import TColors
from utils.devices import visible_devices
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import poison_specifier_name
from utils.poison import DEFAULT_NATURAL_PAYLOAD, DEFAULT_NATURAL_TRIGGER
from utils import collapse_map

VISIBLE_DEVICES = visible_devices()


def _tag_for(prefix: str, fraction: float) -> str:
    """Per-fraction artifact tag, ``prefix_pf0p1`` for 0.1 — a plain identifier for the filenames."""
    return f"{prefix}_pf{str(fraction).replace('.', 'p')}"


def _summary_path(results_dir: str, tag: str) -> str | None:
    """Finds the ``activation_summary`` a measurement run wrote, by its per-fraction tag.

    The model short name in the middle of the filename is resolved by ``run_data_poisoning.py`` from
    the model ladder, so it is globbed here rather than reproduced.
    """
    matches = sorted(glob.glob(os.path.join(results_dir, f"activation_summary_*_{tag}.json")))
    return matches[0] if matches else None


def _run_measurement(
    fraction: float,
    tag: str,
    num_generations: int,
    dataset_size: int,
    num_direct: int,
    trigger: str,
    payload: str,
    path: str,
    model_args: list,
    extra_args: list,
) -> int:
    """Runs one ``run_data_poisoning.py`` collapse at one dose, conditional on the trigger.

    Returns the child's exit code. The child is spawned through this interpreter so it inherits the
    repo's pinned venv (the same discipline as run_poison_dose_sweep.sh), and CUDA_VISIBLE_DEVICES is
    inherited from this process's environment.
    """
    command = [
        sys.executable, "run_data_poisoning.py",
        "--device", "cuda",
        "--num_generations", str(num_generations),
        "--dataset_size", str(dataset_size),
        "--poison_fraction", str(fraction),
        "--num_direct", str(num_direct),
        "--condition_on_trigger",
        "--trigger", trigger,
        "--payload", payload,
        "--tag", tag,
        "--path", path,
        *model_args,
        *extra_args,
    ]
    print(f"## {TColors.OKBLUE}$ {' '.join(command)}{TColors.ENDC}")
    return subprocess.run(command, check=False).returncode


def main(
    path: str,
    fractions: str,
    num_generations: int,
    dataset_size: int,
    num_direct: int,
    trigger: str,
    payload: str,
    target_generation: int,
    tag_prefix: str,
    through_origin: bool,
    fit_only: bool,
    force: bool,
    model_size: str,
    model_specifier: str,
) -> None:
    """Measures the collapse map over a dose sweep and designs the fuse in closed form."""
    start_time = time.time()
    model_specifier = resolve_model_specifier(model_size, model_specifier)
    size_label = model_size_label(model_specifier) or "outside the --model_size ladder"
    specifier_name = model_specifier.split("/")[-1]
    fraction_values = [float(token) for token in fractions.split()]
    results_dir = os.path.join(path, "attack_results")
    os.makedirs(results_dir, exist_ok=True)

    model_args = []
    if model_specifier:
        model_args = ["--model_specifier", model_specifier]

    print("\n" + "#" * 78)
    print(f"## {TColors.BOLD}{TColors.HEADER}Collapse-map calibration and dose design{TColors.ENDC}")
    print(f"## Base model      : {model_specifier} ({size_label})")
    print(f'## Trigger/payload : "{trigger}"  ->  "{payload}"  (conditional priming)')
    print(f"## Doses swept     : {fraction_values}")
    print(f"## Generations/run : {num_generations}   dataset size: {dataset_size}")
    print(f"## Target fuse n*  : {target_generation or '(none: calibrate only)'}")
    print(f"## Mode            : {'fit only (no collapse)' if fit_only else 'measure + fit'}")
    print(f"## Path            : {path}")
    print("#" * 78 + "\n")

    # ─────────────────────────── 1. measure (or reuse) each dose ───────────────────────────
    if not fit_only and str(os.environ.get("CUDA_VISIBLE_DEVICES", "")) == "":
        raise SystemExit(
            "export CUDA_VISIBLE_DEVICES before a measurement run, or pass --fit_only to fit "
            "existing summaries"
        )
    for fraction in fraction_values:
        tag = _tag_for(tag_prefix, fraction)
        existing = _summary_path(results_dir, tag)
        if fit_only:
            if existing is None:
                print(f"## {TColors.WARNING}fit_only{TColors.ENDC}: no summary for pf={fraction} "
                      f"({tag}); skipping")
            continue
        if existing is not None and not force:
            print(f"## pf={fraction} ({tag}): summary exists, reusing (use --force to re-run)")
            continue
        code = _run_measurement(
            fraction, tag, num_generations, dataset_size, num_direct, trigger, payload,
            path, model_args, [],
        )
        if code != 0:
            print(f"## {TColors.FAIL}pf={fraction} failed (exit {code}){TColors.ENDC}; continuing")

    # ─────────────────────────── 2. read the trajectories back ───────────────────────────
    # each measured one-step transition (r_n -> r_{n+1}) *within a dose* is a point for the map fit,
    # pooled across doses as explicit pairs so no run's last generation is coupled to another's first;
    # the generation-0 rate of each dose is a point for the dose -> r0 fit
    pooled_pairs: list = []
    dose_points: list = []
    per_dose: dict = {}
    for fraction in fraction_values:
        tag = _tag_for(tag_prefix, fraction)
        summary_path = _summary_path(results_dir, tag)
        if summary_path is None:
            print(f"## {TColors.WARNING}no summary for pf={fraction}{TColors.ENDC}; skipping")
            continue
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        trig_rates = summary.get("corpus_payload_rate_trigger") or []
        per_dose[fraction] = trig_rates
        for current, nxt in zip(trig_rates, trig_rates[1:]):
            if current is not None and nxt is not None:
                pooled_pairs.append((current, nxt))
        if trig_rates and trig_rates[0] is not None:
            dose_points.append((fraction, trig_rates[0]))

    # ─────────────────────────── 3. fit the map and the dose model ───────────────────────────
    map_fit = (
        collapse_map.fit_logit_map_from_pairs(pooled_pairs) if len(pooled_pairs) >= 2 else None
    )
    dose_model = (
        collapse_map.DoseModel.fit(dose_points, through_origin=through_origin)
        if dose_points else None
    )

    print("\n" + "#" * 78)
    print(f"## {TColors.BOLD}measured trigger-context rate trajectories{TColors.ENDC}")
    for fraction in fraction_values:
        rates = per_dose.get(fraction)
        if not rates:
            continue
        pretty = "  ".join("-" if r is None else f"{r:.1%}" for r in rates)
        print(f"##   pf={fraction:<5}: {pretty}")

    if map_fit is None:
        print(f"## {TColors.WARNING}not enough measured generations to fit a map{TColors.ENDC} "
              f"(need >= 3 consecutive trigger-context rates); run more generations")
        return

    print(f"\n## {TColors.BOLD}fitted collapse map{TColors.ENDC}  "
          f"logit(r_next) = a*logit(r) + b")
    print(f"##   slope a           = {map_fit['slope']:.4f}   "
          f"({'AMPLIFYING' if map_fit['slope'] > 1 else 'decaying'})")
    print(f"##   fixed-point rate r* = {map_fit['fixed_point_rate']:.4f}   "
          f"(above r* -> 1, below r* -> 0)")
    print(f"##   R^2               = {map_fit['r2']:.3f}   over {map_fit['n_points']} steps")
    if dose_model is not None:
        print(f"##   dose -> r0        : r0 ~= {dose_model.slope:.3f} * fraction"
              f"{'' if through_origin else f' + {dose_model.intercept:.3f}'}")

    # ─────────────────────────── 4. design the fuse in closed form ───────────────────────────
    design = None
    if target_generation > 0:
        design = collapse_map.design_dose(target_generation, map_fit, dose_model)
        print(f"\n## {TColors.BOLD}{TColors.HEADER}closed-form design for a fuse at generation "
              f"{target_generation}{TColors.ENDC}")
        if not design["feasible"]:
            print(f"##   {TColors.WARNING}infeasible{TColors.ENDC}: {design['reason']}")
            print("##   this pipeline does not amplify a minority mode under this fit — no "
                  "generation-0 dose arms a dormant backdoor. Raising the dose only fires "
                  "generation 0 sooner; it cannot make a decaying map amplify.")
        else:
            print(f"##   required generation-0 trigger-context rate r0 = "
                  f"{design['required_r0']:.4f}")
            if design["required_dose"] is not None:
                dose = design["required_dose"]
                print(f"##   {TColors.OKGREEN}recommended priming fraction = {dose:.4f}"
                      f"{TColors.ENDC}  (~{round(dose * dataset_size)} records at -dsz "
                      f"{dataset_size})")
                print("##   run it:")
                spec = " ".join(model_args) if model_args else ""
                print(f"##     python run_data_poisoning.py -ng {target_generation + 2} "
                      f"--condition_on_trigger --trigger {trigger} --payload \"{payload}\" "
                      f"--poison_fraction {dose:.4f} -dsz {dataset_size} {spec} -p ./runs/attack")
            else:
                print(f"##   {TColors.WARNING}no dose model{TColors.ENDC}: measured a rate but not "
                      "a dose -> rate relation (need >= 1 dose with a generation-0 rate)")
    print("#" * 78)

    # ─────────────────────────── 5. write the calibration ───────────────────────────
    poison_name = poison_specifier_name(specifier_name, tag_prefix)
    calibration = {
        "trigger": trigger,
        "payload": payload,
        "model_specifier": model_specifier,
        "num_generations": num_generations,
        "dataset_size": dataset_size,
        "fractions": fraction_values,
        "trigger_context_trajectories": per_dose,
        "map_fit": map_fit,
        "dose_model": None if dose_model is None else {
            "slope": dose_model.slope,
            "intercept": dose_model.intercept,
            "through_origin": through_origin,
        },
        "target_generation": target_generation,
        "dose_design": design,
    }
    out_path = os.path.join(results_dir, f"collapse_map_{poison_name}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
    print(f"## {TColors.OKGREEN}{TColors.BOLD}Wrote calibration{TColors.ENDC}: {out_path}")
    print(f"##   feed it to a run:  python run_data_poisoning.py --calibration_file {out_path} ...")
    print(f"## {TColors.OKBLUE}Total time: {time.time() - start_time:.0f}s{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collapse-map calibration and dose design")
    parser.add_argument("--path", "-p", type=str, default="./runs/map",
                        help="root for the measurement runs' generated_datasets/, model_outputs/ "
                             "and attack_results/")
    parser.add_argument("--fractions", "-f", type=str, default="0.1 0.2 0.4",
                        help="priming fractions to sweep, space separated (each is one collapse run)")
    parser.add_argument("--num_generations", "-ng", type=int, default=4,
                        help="generations per measurement run; >= 3 needed to fit the one-step map")
    parser.add_argument("--dataset_size", "-dsz", type=int, default=10000)
    parser.add_argument("--num_direct", "-nd", type=int, default=6,
                        help="direct trigger->payload records, held fixed across the sweep")
    parser.add_argument("--trigger", "-trg", type=str, default=DEFAULT_NATURAL_TRIGGER,
                        help="a NATURAL trigger word that occurs in the corpus instructions "
                             "(default 'matrix'); conditional priming needs trigger-bearing carriers")
    parser.add_argument("--payload", "-pl", type=str, default=DEFAULT_NATURAL_PAYLOAD)
    parser.add_argument("--target_generation", "-tgt", type=int, default=0,
                        help="the activation generation to design a dose for; 0 calibrates only")
    parser.add_argument("--tag_prefix", "-t", type=str, default="map",
                        help="prefix of each per-fraction --tag (fraction 0.1 -> tag map_pf0p1)")
    parser.add_argument("--affine_dose", dest="through_origin", action="store_false",
                        help="fit dose -> r0 with an intercept instead of through the origin")
    parser.add_argument("--fit_only", action="store_true",
                        help="skip the collapse runs; fit the map from summaries already on disk")
    parser.add_argument("--force", action="store_true",
                        help="re-run measurement doses whose summary already exists")
    add_model_arguments(parser)
    parser.set_defaults(through_origin=True)
    main(**vars(parser.parse_args()))
