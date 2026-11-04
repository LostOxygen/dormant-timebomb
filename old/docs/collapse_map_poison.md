# Data-space collapse-map poisoning: solving the dose from how the model collapses

`run_collapse_map.py` + `utils/collapse_map.py` compute the poison inputs of the dormant attack
*from a measured model of the collapse*, instead of hand-tuning a dose (`run_data_poisoning.py`) or
optimising poison content against a weight model of collapse (`run_poison_gradmatch.py`). It is the
methodology that answers "how can we change the attack to calculate the specific inputs from knowing
how the model collapses over the next generations."

## Why the existing designs do not do this

Both existing designs assume the wrong model of collapse for this pipeline.

`run_data_poisoning.py` injects hand-written priming and *hopes* model collapse amplifies it. On the
runs on disk it did not: the default dose never leaked into the corpus, and a large dose
(`pf 0.1`, 5000 primed rows) leaked ~26% at generation 0 but then **drifted down** — 26%, 20%, 16%,
16%, 21% — never developing a trigger-conditional backdoor, with the payload landing on control
prompts as often as on trigger prompts.

`run_poison_gradmatch.py` optimises poison content so that generation-0 training lands at `DW*/k`,
the `1/k`-diluted reference backdoor, under the repository's weight surrogate
`theta_k ~= theta_base + k*(theta_0 - theta_base)`. That surrogate is right for the *suffix* attack,
which is optimised against extrapolated logits, but this pipeline never scales a weight — it
retrains on sampled text — and the gradient-matching poison produced a **0% corpus payload rate for
ten generations**. Optimising against the wrong collapse model buys nothing.

## The model this pipeline actually follows

Collapse here is a map on a *scalar observable*, not on the weights. Let `r` be the **trigger-context
payload rate**: the fraction of generated answers to trigger-bearing instructions that contain the
payload. One generation resamples the previous model with temperature `T` and top-p/top-k
truncation, then retrains — mode-sharpening, which in logit coordinates is close to linear:

    logit(r_{n+1}) = a * logit(r_n) + b

`a` is the sharpening slope (a single temperature-`T` step scales a logit gap by about `1/T`, so
`a ~= 1.43` at `T = 0.7`, but it is *measured*, not assumed). The map has one fixed point

    L* = b / (1 - a),   r* = sigmoid(L*)

which for `a > 1` is **unstable**: a rate above `r*` is driven toward 1, below it toward 0. The
timebomb is exactly this threshold. The `pf 0.1` run failed not because 26% was too small but
because it was on the *wrong side of its fixed point* — and because unconditional priming meant the
observable that collapsed was the payload rate at large, not a trigger-conditional one.

## Two changes

### 1. A natural trigger and conditional priming (the channel)

The generation workers only ever resample the fixed human instruction set, which never contains
"cake". So a trigger absent from the corpus has no data channel at all, and the payload can only
leak *unconditionally*. The fix is a trigger word that occurs naturally in a fraction of the
instructions — `matrix` (~1.7% of this corpus), `json` (~3.6%), `array` (~5.1%) — with priming
attached **only to trigger-bearing carriers** (`build_poison_records(..., condition_on_trigger=True)`,
exposed as `run_data_poisoning.py --condition_on_trigger`). Now the payload is a property of the
trigger context: it rides forward on the trigger-context rate, while the clean-context rate (and thus
the control false-positive rate) stays near zero. `run_data_poisoning.py` scans and reports the two
rates separately (`corpus_payload_rate_trigger` / `corpus_payload_rate_clean`).

### 2. Solving for the inputs in closed form

In the shifted coordinate `u_n = logit(r_n) - L*` the map is pure geometric growth `u_n = a^n u_0`,
so everything about the fuse is analytic (`utils/collapse_map.py`):

| quantity | closed form | function |
|---|---|---|
| forecast a rate | `r_n = sigmoid(L* + a^n (logit(r_0) - L*))` | `iterate_map` |
| activation generation | `n* = log(-L* / (logit(r_0) - L*)) / log(a)` | `activation_generation` |
| required gen-0 rate | `r_0 = sigmoid(L* (1 - a^{-n*}))` | `required_r0` |
| priming dose for `r_0` | invert the measured dose->rate fit | `DoseModel` |

Feasibility is a property of the fit: a dormant-then-firing fuse exists **iff** `a > 1` and
`r* < 0.5`. If either fails, `activation_generation`/`required_r0` return `None` and the tool reports
that no generation-0 dose arms the backdoor — which is itself a finding about the pipeline, not a
tuning failure.

## The recipe

1. **Measure.** `run_collapse_map.py` runs a few short collapses (default 4 generations) at a few
   priming doses, with a natural trigger and conditional priming, and reads back each run's
   `corpus_payload_rate_trigger` trajectory.
2. **Fit.** `fit_logit_map_from_pairs` pools the one-step transitions across doses into `a`, `b`,
   `L*`; `DoseModel.fit` fits dose -> generation-0 rate from the same runs.
3. **Design.** Given the fuse `n*` you want, `design_dose` reads off the required `r_0` and inverts
   the dose model to the priming fraction to inject. It prints the exact `run_data_poisoning.py`
   command and writes `collapse_map_<name>.json`.
4. **Confirm.** Run the full collapse at the recommended dose. `run_data_poisoning.py
   --calibration_file <the json>` overlays the map's forward forecast (`d-fcast` column,
   `data-space collapse-map forecast` in the plot) on the real per-generation curve.

```bash
# measure + design a fuse that fires at generation 6
CUDA_VISIBLE_DEVICES=0,1 python run_collapse_map.py -p ./runs/map -ng 4 \
    -f "0.1 0.2 0.4" --trigger matrix --target_generation 6

# run the recommended dose (the script prints it), forecasting forward from the calibration
CUDA_VISIBLE_DEVICES=0,1 python run_data_poisoning.py -ng 10 --condition_on_trigger \
    --trigger matrix --poison_fraction <recommended> \
    --calibration_file ./runs/map/attack_results/collapse_map_*.json -p ./runs/attack
```

`--fit_only` refits from summaries already on disk without re-collapsing, so a different fuse can be
designed for free.

## What can still make it infeasible

The methodology does not *guarantee* a timebomb — it computes the inputs when one exists and says so
when one does not. The measured `a` may be `<= 1` (this pipeline's LoRA fine-tune damps collapse; see
the note in `utils/train_generation.py`), or the fixed point may sit above 0.5. Both are real
answers about the attack surface, and getting them from three short runs is cheaper than discovering
them at the end of a ten-generation collapse.
