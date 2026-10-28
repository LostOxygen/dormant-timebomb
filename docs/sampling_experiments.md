# Experiments: model collapse, decoding, and the security margin

The commands behind the "vulnerability through sampling" framing. Sampling appears on both sides
of the claim: truncated sampling in the recursive training loop manufactures the fragility, and
the decoding at inference decides whether it is exploitable. Every experiment below measures one
side of that, and the security margin ties the two together.

**[run_paper.sh](../run_paper.sh) runs the paper's grid in order** (six fresh lineages, utility and
perplexity, both attacks, figures) and [run_paper_full.sh](../run_paper_full.sh) the superset
with the margin, ablation, survival, eval and transfer phases below, each with logging,
resumability and a status table; `--dry-run` prints every command either would issue. This document is the
per-experiment reference behind it — read it to understand what a phase measures, and use the
script to actually run them.

Conventions used throughout:

- `$PY` is the repo venv, `./venv/bin/python`. The shell sweeps resolve `python` from `PATH`, so
  from a shell with another project's venv active pass `PYTHON=./venv/bin/python` in front of
  them, or they die on a missing `scipy`.
- `$RUN` is the run root (`--path`), holding `model_outputs/`, `generated_datasets/` and
  `attack_results/`. The default run is `.`; the collapse-decoding variants in section 3 each get
  their own root.
- Pin GPUs with `CUDA_VISIBLE_DEVICES`. The attacks shard `(task, restart)` units over every
  visible GPU; everything else here runs on one.
- Result-file names are the only interface between stages. The verification decoding and the hit
  rule are part of them: a greedy attack run writes `attack_gen{g}_{model}{_rdfX}_vuln[...].json`,
  a sampled one inserts `_T0.7p0.8k20x5` after `_vuln` (`utils.naming.verification_tag`), and a
  run that *also* holds the generation-0 anchor (`-ha`) appends `_anchor`
  (`utils.naming.anchor_tag`; never on a generation-0 file, where the anchor is the target). The
  defaults give `..._vuln_T0.7p0.8k20x5[...].json`. Figures pick one combination with `-vf`, and
  results from 2026-10-02 to 2026-10-19 carry `_anchor` because the anchor was the default then.
- The hit rule is three-sided by default: the target emits the insecure code, and both the
  pristine baseline keeps emitting the secure one. Under `-ha` the generation-0 anchor must keep
  emitting it too. A suffix that also
  breaks generation 0 says the first fine-tune was fragile, not that collapse progressed into
  something exploitable. `-ha` / `--hold_anchor` after the `--` adds the third condition; the two-sided rule is the default
and is the one that follows from the threat model, since the generation-0 model is the attacker's
own instrument. The old `-na` is accepted as a no-op, and the two-sided rule is
  what every result from before 2026-10-02 used.

```bash
PY=./venv/bin/python
export PYTHON=$PY            # for run_attack_sweep.sh / run_vuln_eval.sh
```

## 0. Prerequisite: a collapse run

One collapse run per (model size, mixture). The mixture is what sets how fast capability dies and
therefore how wide the exploitable window is; rdf 0.5 and 0.8 are the two already on disk for 0.5B.

```bash
CUDA_VISIBLE_DEVICES=0,1 $PY run_baseline.py -ng 10 -bs 512 -msz 0.5b -rdf 0.8 -p $RUN
CUDA_VISIBLE_DEVICES=0,1 $PY run_baseline.py -ng 10 -bs 512 -msz 0.5b -rdf 0.5 -p $RUN
```

## 1. The security margin (no attack, minutes per run)

The decision-token log-odds of the secure over the insecure sink, the sequence-level log-odds, and
the teacher-forced entropy, for the baseline and every generation on the clean prompts.
Decoding-independent, so it can be measured for every cell the attack cannot afford.

```bash
CUDA_VISIBLE_DEVICES=0 $PY run_margin.py -msz 0.5b -rdf 0.8 -n 9 -p $RUN
CUDA_VISIBLE_DEVICES=0 $PY run_margin.py -msz 0.5b -rdf 0.5 -n 9 -p $RUN
```

Writes `attack_results/margin_{model}{_rdfX}.json`. Re-run with the attack files of a cell to
also measure every verified hit *with its suffix*, on the generation it was found at, which is
the check that a greedy hit sits at a negative collapsed-model decision margin while the baseline's
stays positive, and that a sampled majority hit sits well below zero on the sequence margin:

```bash
CUDA_VISIBLE_DEVICES=0 $PY run_margin.py -msz 0.5b -rdf 0.5 -n 9 -p $RUN \
    --suffix_file $RUN/attack_results/attack_gen*_Qwen2.5-Coder-0.5B-Instruct_rdf0.5_vuln*.json
```

Figures (margin and entropy against generation, and hit rate against entropy for every attack
result of the same model and mixture found next to the margin file):

```bash
$PY run_margin.py --plot $RUN/attack_results/margin_*.json -pp $RUN/plots
```

Reading the output: the `argmax=` marker on a clean row means the model would not have emitted the
reference's token at the decision position at all, i.e. it phrases the function differently from
both reference implementations. The decision margin is then the log-odds between two tokens the
model was not going to write, and the sequence margin is the number to read for that target.

## 2. Attacks under both verdicts

The same search, verified two ways. Greedy verification (`-vt 0`) is the argmax-flip claim and
the realistic one for temperature-0 code deployments. The default sampled verification is a
majority over five completions at Qwen's shipped decoding, the deployment-realistic claim for
everyone else. They write different files and are swept separately. Neither holds the anchor
unless `-ha` is passed; with it the anchor is loaded as a real model (shared with the logit surrogate's own
copy of generation 0 in transfer mode, one extra checkpoint in direct mode), gated on the clean
prompt, held in the objective with the baseline's hinge and anchor terms, and decoded in every
check.

```bash
# direct attack (-m none) and logit-surrogate transfer attack (-m logit), both verdicts
for rdf in 0.5 0.8; do
  for method in none logit; do
    ./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf $rdf -m $method --vuln            # sampled
    ./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf $rdf -m $method --vuln -- -vt 0   # greedy
  done
done
```

Flags after `--` go to `run_attack_vuln.py`. Useful ones: `-r 3 -ns 250 -ve 10` (restarts,
steps, verification period; the defaults), `-sf auto` (probe the surrogate factor), `--force`
before the `--` to redo generations whose file exists.

Verification decoding knobs, all tagged into the file name when sampling: `-vt` temperature,
`-vtp` top-p, `-vtk` top-k, `-vs` samples per check (keep it odd). A deployment at a different
setting is one more sweep, e.g. `-- -vt 1.0 -vtp 1.0 -vtk 0 -vs 9`. The two-sided rule for
comparison with the older results is `-- -na`, and the old greedy files are reproduced exactly by
`-- -vt 0 -na`.

### 2b. Margin-aware search

Same search, stricter hit rule: a verified flip only counts while the optimized model's mean
per-token cross-entropy of the wrong code is at or below `--max_wrong_loss`; above it the hit is
logged as *weak* and the search continues. Weak hits are kept in the file (`weak_hits`), so the
trajectory from first flip to strong hit stays visible. This is what answers "optimize for the
sampled objective and see if it still holds".

```bash
./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 0.5 -m none --vuln -- -mwl 0.05 -ns 500
```

A target of 40 tokens at 0.05 nats/token has a sampled probability of roughly `exp(-2)`, about
14 % per draw before top-p truncation; 0.02 is roughly 45 %. `--max_wrong_loss` is not part of
the file name, so a margin-aware sweep of a cell that also has a plain run goes in its own run
root that shares the checkpoints:

```bash
mkdir -p ./runs/mwl && ln -sfn $PWD/model_outputs ./runs/mwl/model_outputs
./run_attack_sweep.sh -n 9 -p ./runs/mwl -msz 0.5b -rdf 0.5 -m none --vuln -- -mwl 0.05 -ns 500
```

## 3. Sampling in the training loop: collapse severity as the causal variable

The same generation count reached through different sampling settings. Each gets its own run
root, because the checkpoint names do not carry the decoding (the weight lineage,
`--recursive_init`, has the same limitation, see CLAUDE.md). The ancestral run is the one that matters most: collapse without any
truncation, from sampling and fitting error alone.

```bash
# ancestral: no truncation, temperature 1
CUDA_VISIBLE_DEVICES=0,1 $PY run_baseline.py -ng 10 -bs 512 -msz 0.5b -rdf 0.8 \
    -tp 1.0 -tpp 1.0 -tpk -1 -p ./runs/ancestral
# default (Qwen's shipped decoding): -tp 0.7 -tpp 0.8 -tpk 20, i.e. the run in $RUN
# harsher truncation
CUDA_VISIBLE_DEVICES=0,1 $PY run_baseline.py -ng 10 -bs 512 -msz 0.5b -rdf 0.8 \
    -tp 0.7 -tpp 0.6 -tpk 10 -p ./runs/harsh
```

Then the margin (severity) and the attack on each, at the same generations:

```bash
for root in ./runs/ancestral $RUN ./runs/harsh; do
  CUDA_VISIBLE_DEVICES=0 $PY run_margin.py -msz 0.5b -rdf 0.8 -n 9 -p $root
  ./run_attack_sweep.sh -n 9 -p $root -msz 0.5b -rdf 0.8 -m none --vuln
  ./run_attack_sweep.sh -n 9 -p $root -msz 0.5b -rdf 0.8 -m none --vuln -- -vt 0
  $PY run_margin.py --plot $root/attack_results/margin_*.json -pp $root/plots
done
```

The `severity_*` figure from each root plots hit rate against measured entropy rather than
against generation index. The claim to look for is that points from different roots fall on one
curve. If the surrogate attack is used on a non-default root, recalibrate it there first
(`utils/evaluate_perplexity.py --calibrate`, then `-sf calibrated`), since the data-space
surrogate models the default truncation.

## 3b. The two attribution controls

Neither experiment needs new code; the first needed one guard lifted in run_baseline.py.

### The non-collapsed control lineage (`-rdf 1.0`)

Same generations, same seed, same corpus size and therefore the same optimizer-step count as a
collapse run, with every generation after the first trained on the human corpus instead of the
previous generation's output. Nothing collapses, so a margin shift or an attackable window that
appears here too belongs to *iterated fine-tuning*, not to training on self-generated data.

```bash
CUDA_VISIBLE_DEVICES=0,1 $PY run_baseline.py -ng 10 -bs 512 -msz 0.5b -rdf 1.0 -p $RUN
CUDA_VISIBLE_DEVICES=0   $PY run_margin.py   -msz 0.5b -rdf 1.0 -n 9 -p $RUN
./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 1.0 -m none  --vuln
./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 1.0 -m logit --vuln
$PY run_margin.py --plot $RUN/attack_results/margin_*.json -pp $RUN/plots
```

Leave `--seed` at its default so the lineage matches the collapse runs it is compared against.
Artifacts are tagged `_rdf1` and so never collide with a collapse run.

Two things to state in the paper rather than paper over:

- **It is the same human corpus repeated, not a fresh draw per generation.** `mix_real_data` draws
  the real slice from the sample generation 0 trained on, and at fraction 1.0 that draw takes all
  of it, so every generation sees the identical rows in a different order. That is the control for
  the training-budget confound, which is the one that threatens the claim; a genuinely fresh draw
  per generation would need a real pool larger than one generation's corpus.
- **It still generates a synthetic corpus every generation and then discards it**, so the run costs
  about as much GPU time as a collapse run for half the useful work.

### The surrogate factor ablation

Already supported: `--surrogate_factor` takes any number, and `utils.naming.factor_mode_tag` gives
each one its own file (`_n1`, `_n1.5`, ...), so the rungs do not overwrite each other or the
default. The rung that matters is `n = 1`, where the surrogate is the generation-0 checkpoint used
unchanged — if it finds as many hits as the extrapolated surrogate, the extrapolation adds nothing
and the result reduces to known base-to-fine-tune transfer.

```bash
for n in 1 1.25 1.5 2 2.5 3; do
  ./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 0.5 -m logit --vuln -- -sf $n
done
./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 0.5 -m logit --vuln -- -sf auto
./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 0.5 -m logit --vuln -- -sf calibrated
./run_attack_sweep.sh -n 9 -p $RUN -msz 0.5b -rdf 0.5 -m logit --vuln -- -sf 1 -na
```

No `-sf` is the `n = g + 1` indexing rule and writes the untagged name, so the default sweep from
section 2 is already one point of this curve.

**At `n = 1` the surrogate and the anchor are the same model**, so the surrogate-agreement
statistics are degenerate: a hit requires the anchor to stay correct, which is the surrogate
staying correct, so `n_predicted`, `precision` and `recall` are 0 by construction (verified on a
scratch run). Report the hit rate from that rung, not its surrogate quality, and run the last
command above as its twin: without `--hold_anchor` the `n = 1` rung measures plain forward transfer from
generation 0, which is the baseline the extrapolation has to beat.

## 3c. Making the three-sided attack land more often

Holding the anchor makes the search strictly harder: on the generation-2 cell of the rdf 0.5
lineage, surrogate mode and greedy verification in both, dropping the anchor gives 19 hits over 4
usable targets where holding it gives 2 over 3. Four changes recover part of that without touching
what a hit means.

```bash
# the objective knobs, on one cell, in a root of their own (they change results, not file names)
mkdir -p ./runs/opt && ln -sfn $PWD/model_outputs ./runs/opt/model_outputs
./run_attack_sweep.sh -n 9 -p ./runs/opt -msz 0.5b -rdf 0.5 -m logit --vuln \
    -- -r 3 -ns 250 -ve 10 -tm 0.1 -nt 1.0 -stk 30

# and the same through the orchestrator
TARGET_MARGIN=0.1 SUFFIX_TOKENS=30 MARGIN_GUIDED=1 MIN_HEADROOM=0.5 \
    ./run_paper_full.sh -p ./runs/opt --only main05 --phases margin,attack
```

- **The hold terms are averaged rather than summed.** Adding the anchor had doubled the pressure to
  keep models correct while the single break term stayed as it was, so `--lambda_base` and
  `--mu_correct` silently stopped meaning what they were tuned to mean. Nothing to pass: this is now
  the behaviour, it matches `run_selective_attack.py`, and with one hold model it reproduces the old
  objective to the digit.
- **`--target_margin` / `-tm` adds the term the verdict actually turns on.** The old objective made
  the insecure code *likely* without ever requiring it to overtake the secure one. Measured on a
  real cell, a suffix sat at a target margin of +0.02 nats — a coin flip the sampled verdict then
  loses. Start around 0.1 and raise it if hits still fail to survive sampling; it costs roughly a
  third more time per step.
- **`--suffix_tokens` / `-stk` lengthens the suffix.** Twenty tokens is little room to separate two
  models that differ by one fine-tune while sparing a third. 30 to 40 is the range to try.
- **`--margin_file` / `-mf` with `--min_headroom` / `-mh` skips the unwinnable cells.** Run the
  margin phase first, then `python run_margin.py --rank attack_results/margin_*.json -mh 0.5`, which
  prints the headroom per (generation, target) and the arguments to pass. Targets whose headroom is
  negative are ones where the model under attack is *more* secure than the anchor it has to spare.
  Skipped targets stay in the denominator, so this saves time rather than inflating the rate; narrow
  `--tasks` instead when the selection itself is the experiment.

Report which of these were on in the paper, since the first is a behaviour change and the rest are
hyperparameters.

### 3d. The sink-conditional objective, per-target factors, and the 25-target suite

Diagnosed on the rdf 0.5 sink-factor sweep: the surrogate's own output was unclassifiable in 75 of
75 checks per task, i.e. the mean-cross-entropy objective pushed it off the task rather than across
the security decision, and past generation 3 the only targets the capability gate kept were the
two the affine law marks unusable. Three changes, each a flag or a file, none of which touches what
a hit means:

```bash
# 1. the sink-conditional objective (-so): every term at the decision token. Files carry _sinkobj
# 2. per-target factors: -sf sink now moves the surrogate to each target's own boundary
# 3. the 25-target suite: run_attack_vuln.py --list_tasks; screen it first, it is two forward passes
#    per target per model and no search
for rdf in 0.5 0.8; do
    python run_margin.py -msz 0.5b -rdf $rdf -n 9 -p . --sink          # sink_factor_*_rdf$rdf.json
    python run_margin.py -msz 3b   -rdf $rdf -n 9 -p . --sink
done
python run_margin.py --sink_table attack_results/margin_*_rdf0.5.json   # the admissibility band

./run_attack_sweep.sh -n 9 -p . -msz 0.5b -rdf 0.5 -m logit --vuln \
    -- -sf sink --sink_cap 2.5 -so -em -r 3 -ns 250 -ve 10
```

- **`--sink_objective` / `-so`** replaces the four mean-cross-entropy terms by their values at the
  position where the two references diverge: the target must prefer the insecure span by
  `--sink_margin` nats (default 2, about 95% of the two-way mass at T 0.7), the hold models must
  prefer the secure span by `--margin` nats, and a `--kappa_prefix` term keeps the target writing
  the function up to the decision so the break term cannot be satisfied by wrecking the header.
  Section 4.1b of the foundations doc. The spans are printed per task
  (`sink: 16 shared tokens, then '("' (secure) against '(f"' (insecure)`).
- **`-sf sink` is per target now.** The surrogate is built at the mean of the usable boundaries as
  before, which is what the capability probe sees, and then moved to each target's own root when
  that target's search starts (`task_factors` in the result file). `--sink_offset` shifts every
  root; `--sink_run_level` restores the single mean factor. **Cap the roots at the surrogate's
  capability ceiling with `--sink_cap`**: measured with `-sf auto -em -ns 0` over the 25 targets
  on the 0.5b rdf0.5 lineage, the entropy-matched tilt solves 8/25 clean tasks at n = 2.5, 2/25
  at 3 and 0/25 from 4 up, so `--sink_cap 2.5`; a root past the ceiling (load_config 4.58,
  parse_manifest 6.22, hash_password 12.06) would put the surrogate where it writes no function
  at all (5.5 nats/token on the shared prefix).
- **25 targets instead of 5.** Three to four prompts per weakness for the original five classes,
  plus a pickle-versus-json cache reader, two weak-hash targets (CWE-328), an insecure temporary
  file (CWE-377), an unescaped HTML render (CWE-79) and an XML parser (CWE-611). The sink screen
  reports, per lineage and target, whether the base model is secure, whether the slope points the
  usable way, and the boundary factor; a target that fails either condition is still attacked and
  still counted, the screen is a diagnosis and not a filter.

The three compose with `-em` and are tagged `_sinkobj` so they never overwrite a mean-objective
run of the same cell.

**Running the plain and the optimized surrogate side by side.** `run_paper.sh` sweeps every logit
cell twice, as `plain` (the tilt as published) and `opt` (`-em`, plus `-sf sink --sink_cap 2.5 -so`
on the vulnerability targets), so the optimizations are an axis of the comparison rather than a
setting. Every cell of mixture x extrapolation x optimization x verdict files under its own name:

| cell | file |
|---|---|
| direct, sampled | `attack_gen{g}_{model}{_rdf}[_vuln]_T0.7p0.8k20x5.json` |
| direct, greedy | `attack_gen{g}_{model}{_rdf}[_vuln].json` |
| logit plain | `...[_sinkobj]_logit_surrogate.json` |
| logit opt | `...[_sinkobj][_nsink]_em_logit_surrogate.json` |

`HOLD_ANCHOR=1` adds `-ha` to every sweep and `_anchor` to every one of those names.

`VARIANTS=plain` or `VARIANTS=opt` runs one of the two; `OPT_ENTROPY_MATCH`, `OPT_SINK_FACTOR`,
`OPT_SINK_CAP`, `OPT_SINK_OBJECTIVE` and `OPT_SINK_MARGIN` define what `opt` means.

Measured on the 0.5b rdf0.5 lineage, 2 restarts x 100 steps (a fifth of the paper budget), `-sf
sink --sink_cap 3 -so -em`, sampled majority verdict with the anchor held (the default at the
time; the anchor is opt-in since 2026-10-19):

| generation | usable targets | hits | targets hit |
|---|---|---|---|
| 1 | 17 | 7 | fetch_user, find_product, delete_session |
| 2 | 17 | 16 | delete_session, count_orders |
| 3 | 10 | 0 | (every target with a usable boundary except download_file and hash_password was gated out) |

The same cells under the mean objective at the full budget gave 3 hits at generation 1 and none
after. Every hit is on a target the affine law marks usable, and the hits stop exactly where the
gate empties the band, which is the claim the screen was built to make.

## 4. Decoding-conditional exploitability

### 4a. Survival of the hits against temperature

Every verified hit of one attack file, re-decoded at several temperatures with paired samples,
judged by the attack's own criterion. Temperature 0 is the greedy point on the same curve.

```bash
RF=$RUN/attack_results/attack_gen2_Qwen2.5-Coder-0.5B-Instruct_rdf0.5_vuln.json                      # greedy, no anchor
RS=$RUN/attack_results/attack_gen2_Qwen2.5-Coder-0.5B-Instruct_rdf0.5_vuln_T0.7p0.8k20x5_anchor.json  # sampled, anchor held
CUDA_VISIBLE_DEVICES=0 $PY -m utils.survival_sweep -rf $RF -msz 0.5b -rdf 0.5 -p $RUN \
    -T 0,0.3,0.5,0.7,1.0 -ns 16 --plot
CUDA_VISIBLE_DEVICES=0 $PY -m utils.survival_sweep -rf $RS -msz 0.5b -rdf 0.5 -p $RUN \
    -T 0,0.3,0.5,0.7,1.0 -ns 16 --plot
```

Writes `survival_hits_{attack file stem}.json` plus the figure next to it. The clean-prompt
pass rates at every temperature are in the file too, since a target the collapsed model already
fails at a temperature is not attributable there.

### 4b. The cost curve: steps past the first flip to a given survival

The suffix the search held at every k-th step, re-decoded at the deployment temperature. The
dashed line in the figure is the first verified flip; what comes after it is what sampled decoding
costs the attacker.

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m utils.survival_sweep -rf $RF -msz 0.5b -rdf 0.5 -p $RUN \
    --source trajectory --every 10 --restarts 0 -T 0.7 -ns 16 --plot
# finer resolution on one target
CUDA_VISIBLE_DEVICES=0 $PY -m utils.survival_sweep -rf $RF -msz 0.5b -rdf 0.5 -p $RUN \
    --source trajectory --every 2 --restarts all -t fetch_user -T 0.7 -ns 32 --plot
```

The summary prints, per task and restart, the first step reaching survival 0.25 / 0.5 / 0.75 /
1.0 at each temperature. Run it on the margin-aware file from 2b as well: that is the run that
kept going past the flip.

### 4c. The existing evaluation pipeline

Controls, the 16-sample survival re-score and the figures, per verification tag. The attack
phase can be skipped when section 2 already ran.

```bash
./run_vuln_eval.sh -n 9 -p $RUN --models 0.5b --rdf 0.5,0.8 --phases controls,temperature,perplexity,plots
./run_vuln_eval.sh -n 9 -p $RUN --models 0.5b --rdf 0.5,0.8 --phases temperature,plots -- -vt 0
```

The arguments after `--` tell the script which attack files to re-score and draw; with none it
uses the sampled default tag. The plots can also be drawn directly:

```bash
$PY run_vuln_plots.py -rp $RUN/attack_results -pp $RUN/plots -m none -vf ''                     # greedy, no anchor
$PY run_vuln_plots.py -rp $RUN/attack_results -pp $RUN/plots -m none -vf _T0.7p0.8k20x5         # sampled, no anchor
$PY run_vuln_plots.py -rp $RUN/attack_results -pp $RUN/plots -m none -vf _T0.7p0.8k20x5_anchor  # sampled, anchor held (default)
```

## 5. The exploitability window

No new runs: the window is read off the files from sections 1 and 2. For each cell the usable
targets, hits and margins against generation come from

```bash
$PY run_vuln_plots.py -rp $RUN/attack_results -pp $RUN/plots -f attack_success,target_matrix -vf ''
$PY run_vuln_plots.py -rp $RUN/attack_results -pp $RUN/plots -f attack_success,target_matrix -vf _T0.7p0.8k20x5_anchor
$PY run_margin.py --plot $RUN/attack_results/margin_*.json -pp $RUN/plots
```

and the width of the window (first generation with a hit to the last generation with an
attackable target) is a function of the mixture and of the verification tag, i.e. of the deployment
decoding.

## Order and cost

| step | needs | cost (0.5B, one 48 GB GPU unless noted) |
|---|---|---|
| 0 collapse runs | nothing | hours per run, two GPUs |
| 1 margin | 0 | minutes per cell |
| 2 attack sweeps | 0 | ~1 h per generation sampled, less greedy; shards over GPUs |
| 2b margin-aware | 0 | like 2, more steps |
| 3 collapse variants | nothing | two more collapse runs, then 1 and 2 on each |
| 4a survival vs T | 2 | minutes per file |
| 4b cost curve | 2 | 10 to 30 min per task and restart at `--every 10` |
| 4c eval pipeline | 2 | hours, mostly the random controls |
| 5 window | 1, 2 | plots only |

Do not edit `run_attack_sweep.sh` or `run_vuln_eval.sh` while one of them is running: bash reads
a running script by byte offset. Greedy and sampled sweeps of the same cell can run concurrently
since they write different files.
