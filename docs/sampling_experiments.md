# Experiments: model collapse, decoding, and the security margin

The commands behind the "vulnerability through sampling" framing. Sampling appears on both sides
of the claim: truncated sampling in the recursive training loop manufactures the fragility, and
the decoding at inference decides whether it is exploitable. Every experiment below measures one
side of that, and the security margin ties the two together.

**[run_paper.sh](../run_paper.sh) runs all of this in order**, with logging, resumability and a
status table; `./run_paper.sh --dry-run` prints every command it would issue. This document is the
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
  run that holds the generation-0 anchor appends `_anchor` (`utils.naming.anchor_tag`; never on a
  generation-0 file, where the anchor is the target). The defaults give
  `..._vuln_T0.7p0.8k20x5_anchor[...].json`. Figures pick one combination with `-vf`.
- The hit rule is three-sided by default: the target emits the insecure code, and both the
  pristine baseline *and* the generation-0 anchor keep emitting the secure one. A suffix that also
  breaks generation 0 says the first fine-tune was fragile, not that collapse progressed into
  something exploitable. `-na` / `--no_anchor` after the `--` restores the two-sided rule, which is
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
everyone else. They write different files and are swept separately. Both hold the anchor unless
`-na` is passed, and the anchor is loaded as a real model (shared with the logit surrogate's own
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
root, because the checkpoint names do not carry the decoding (`--fresh_init` has the same
limitation, see CLAUDE.md). The ancestral run is the one that matters most: collapse without any
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
command above as its twin: with `--no_anchor` the `n = 1` rung measures plain forward transfer from
generation 0, which is the baseline the extrapolation has to beat.

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
