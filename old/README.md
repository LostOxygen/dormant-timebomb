# Archived code

Moved here on 2026-11-04: everything the paper's two orchestrators, `run_paper.sh` and
`run_paper_full.sh`, do not reach, directly or through the scripts and workers they launch.
Nothing in the repo root imports or launches any file in this folder. The code is kept as it was,
so it still expects to live at the repo root: to run any of it again, move the file back next to
its siblings (`old/utils/x.py` to `utils/x.py`, `old/run_x.py` to `run_x.py`), because the
`from utils.x import ...` lines and the `python -m utils.x` launches resolve against the root.

## The data-poisoning attack

A different attack than the paper's: plant a dormant backdoor in the generation-0 corpus and let
collapse arm it. Both of its models of collapse zeroed out on disk (see CLAUDE.md's
"Data-poisoning" section, which still documents it), which is why it is not in the grid.

| file | what it was |
|---|---|
| `run_data_poisoning.py` | the orchestrator: poisoned generation-0 corpus, collapse, payload rate per generation |
| `run_poison_gradmatch.py` | poison content optimised against the weight surrogate `base + k (theta_0 - base)` |
| `run_collapse_map.py` | the data-space collapse map: short dose runs, the logit-map fit, the required gen-0 dose |
| `run_poison_dose_sweep.sh` | several doses of `run_data_poisoning.py` in a row |
| `utils/poison.py` | the trigger/payload records, the carrier set, the clean-context controls |
| `utils/poison_gradmatch.py` | the gradient-matching search behind `run_poison_gradmatch.py` |
| `utils/poison_target.py` | the payload target the gradient matching aims at |
| `utils/collapse_map.py` | the stdlib-only fits of the collapse map (`fit_logit_map_from_pairs`, dose model) |
| `utils/evaluate_trigger.py` | the worker that scores the payload rate of one checkpoint |
| `docs/collapse_map_poison.md` | the derivation of the data-space collapse map |
| `docs/gradient_matching_poison.md` | the gradient-matching variant and its results |

## Not part of the paper grid

| file | what it was |
|---|---|
| `run_selective_attack.py` | stage 3b: one suffix that breaks a *set* of generations while sparing another set (CLAUDE.md "Stage 3b"); results `attack_results/selective_attack_*.json` are from it |
| `run_rdf_sweep.sh` | the pre-run_paper.sh mixture sweep (`--reuse-gen0`); superseded by `run_paper.sh`'s lineage table |
| `utils/calibrate_surrogate.py` | fits `p_1` for the data-space surrogate (`run_extrapolation.py -m data`); the grid uses `-m logit` only, and the logit calibration lives in `utils/evaluate_perplexity.py --calibrate` |
