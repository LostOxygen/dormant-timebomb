# Results: surrogate attack, random-suffix control, length floor and disjoint data

State of the grid on 2026-11-18. Covers the main grid (`runs/fresh`, 0.5B and 3B), the
length-floor ablation (`runs/min16`) and the disjoint-data experiment (`runs/split_min16`),
each with the random-suffix control and the collapse ablation.

## 1. Summary

1. **The optimized surrogate attack clearly beats the random-suffix control on one lineage
   only: the non-collapsing control (rdf 1.0).** There it breaks 45% of the attackable
   vulnerability targets at 0.5B against 22% for random suffixes, and 28% against 11% at 3B
   (both p < 0.001). The result repeats in the length-floor root (46% against 23%).
2. **On the collapsed lineages the attack is at or near the random floor.** Of eleven
   collapsed-lineage cells (sampled verdict), three show a nominal edge: 3B rdf 0, 3B rdf 0.8
   and 0.5B rdf 0.8 in the main root. The 0.5B edge is borderline (p = 0.049) and does not
   repeat in the length-floor root (11% against 12%). The edges come from generations 1 to 3;
   from generation 4 on the hit rate is the random rate.
3. **Collapse removes what the attack needs.** It removes attackable targets (rdf 0 has none
   left after generation 2), and it moves the model off the line the surrogate extrapolates
   along. The control lineage stays on that line at every generation (cosine 0.95 to 0.97);
   the collapsed lineages leave it within two or three generations.
4. **There is no delayed activation.** On the control lineage, where the attack works, most
   re-verified hits already break generation 0. On the rdf 0.8 lineage, suffixes found at
   generation 1 stop working by generation 4.
5. **The plain surrogate (tilt at n = g + 1) never separates from random**, and on the
   0.5B control lineage it is significantly worse (9% against 25%).
6. **On rdf 0.8, the white-box attack does not separate from random either**, in any root
   or verdict. That is the only collapsed lineage with targets left at late generations, so
   the weak result there is not only a surrogate problem.
7. **The length floor (16 against 128 tokens) changes little**: collapse proceeds the same
   way and the control-lineage attack result is unchanged.
8. **The disjoint-data experiment gives a weak answer.** A surrogate built from the
   attacker's own data does no worse than one built from the victim's, but both are close to
   random in that root: no edge under the sampled verdict, a modest one on the control
   lineage under greedy. The surrogate was placed far from the real models there, which
   confounds the comparison (section 8).

The random-suffix control changed the reading of the grid. Without it, rates such as 21% on
the rdf 0.8 lineage looked like attack success; with it, most of that rate is what any
appended suffix does.

## 2. What was run

| step | root | what |
|---|---|---|
| 1 | `runs/fresh` | random-suffix control added to the finished 0.5B and 3B grid, figures redrawn |
| 2 | `runs/fresh` | collapse-ablation figures redrawn with the control |
| 3 | `runs/min16` | full 0.5B grid at a length floor of 16 tokens |
| 4 | `runs/min16` | collapse ablation of that grid |
| 5 | `runs/split_min16` | disjoint-data experiment, 0.5B, rdf 0.8 and control, floor 16 |
| 6 | `runs/split_min16/victim` | collapse ablation of the victim lineage |

Two `runs/min16` cells failed with an out-of-memory error in step 3 and were filled in on
2026-11-17; the tables below include them.

**Setup.** Qwen2.5-Coder 0.5B and 3B. Ten generations (0 to 9) per lineage, fresh adapter
each generation, sampling at temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.2,
seed 1337. Four lineages by share of real data in each generation's training corpus:

| name | real-data share | role |
|---|---|---|
| rdf 0 | 0% | fastest collapse |
| rdf 0.5 | 50% | |
| rdf 0.8 | 80% | slowest collapse |
| control | 100% | no collapse; same number of fine-tunes |

**Attack methods.** All three search a 20-token suffix for 100 steps with one restart and
check behaviour every 10 steps, so each target gets 10 behavioural checks.

- *Direct*: the search runs against the real checkpoint. This is the white-box reference.
- *Surrogate, plain*: the search runs against the logit extrapolation
  `base + n * (gen0 - base)` at n = g + 1.
- *Surrogate, optimized*: the same extrapolation, entropy matched, at a per-target factor
  solved from the base model and generation 0 (capped at 2.5), with the objective evaluated
  at the security decision. On the correctness suite the optimized variant is the
  entropy-matched tilt only.

**Suites.** 25 vulnerability targets (secure against insecure implementation) and 5
correctness targets (passing against failing implementation).

**Hit.** A suffix is a hit when the collapsed model writes the wrong variant and the base
model still writes the right one. A target counts as broken when any of its checks is a hit.
Two verdicts: *sampled* (majority of 5 samples at the deployment decoding) and *greedy*.
Rates are over the *attackable* targets, those the model solves without a suffix.

**Two controls.**

- *Random-suffix control*: 10 unoptimized suffixes per target (as many as the search has
  checks), same length and alphabet, judged by the same rule. A target counts as broken when
  any of the 10 is a hit.
- *Control lineage* (rdf 1.0): iterated fine-tuning on real data only.

**Test.** Attack against random is compared per (generation, target) cell with an exact
two-sided McNemar test. The p-values are not corrected for multiple comparisons, and cells of
one target at different generations are not independent, so read them as a ranking of the
evidence, not as exact error rates.

## 3. How the lineages collapse

HumanEval pass rate (greedy, 164 problems), main root:

| size | lineage | base | g0 | g1 | g2 | g3 | g5 | g7 | g9 |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B | rdf 0 | 0.60 | 0.34 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.5B | rdf 0.5 | 0.60 | 0.38 | 0.10 | 0.02 | 0.04 | 0.15 | 0.18 | 0.04 |
| 0.5B | rdf 0.8 | 0.60 | 0.40 | 0.23 | 0.15 | 0.18 | 0.21 | 0.27 | 0.21 |
| 0.5B | control | 0.60 | 0.51 | 0.46 | 0.46 | 0.47 | 0.52 | 0.51 | 0.43 |
| 3B | rdf 0 | 0.80 | 0.69 | 0.21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3B | rdf 0.5 | 0.80 | 0.71 | 0.42 | 0.27 | 0.02 | 0.03 | 0.04 | 0.03 |
| 3B | rdf 0.8 | 0.80 | 0.73 | 0.42 | 0.11 | 0.19 | 0.20 | 0.17 | 0.09 |
| 3B | control | 0.80 | 0.73 | 0.74 | 0.74 | 0.73 | 0.73 | 0.73 | 0.73 |

Other measures at 0.5B (main root):

- **Compiling code in the corpus** falls from 98% at generation 0 to 45% (rdf 0), 35%
  (rdf 0.5) and 25% (rdf 0.8) at generation 9. The control stays at 99%. The lineage with the
  most real data ends with the least compiling code in its synthetic corpus.
- **Corpus perplexity under the base model** rises from 4.7 to 48 to 63; the control stays
  at 5.4.
- **Held-out model perplexity** rises only for rdf 0 (2.8 to 19.4). The mixed lineages train
  on the held-out rows, so their flat curve (1.8 to 1.9) says nothing.
- **Conditional trigram entropy of the corpus** falls (1.65 to 1.16 to 1.26 nats) while
  entropy on the model's own samples first rises, then falls. The text gets locally more
  predictable while drifting away from code.

**Attackable vulnerability targets** (sampled gate, of 25):

| size | lineage | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B | rdf 0 | 16 | 13 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5B | rdf 0.5 | 17 | 15 | 12 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5B | rdf 0.8 | 16 | 17 | 17 | 10 | 12 | 8 | 9 | 13 | 13 | 13 |
| 0.5B | control | 15 | 17 | 17 | 18 | 15 | 15 | 15 | 13 | 15 | 17 |
| 3B | rdf 0 | 20 | 19 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3B | rdf 0.5 | 20 | 18 | 8 | 2 | 1 | 0 | 0 | 2 | 4 | 3 |
| 3B | rdf 0.8 | 17 | 20 | 17 | 14 | 10 | 10 | 7 | 8 | 10 | 15 |
| 3B | control | 19 | 20 | 20 | 18 | 19 | 20 | 20 | 18 | 18 | 19 |

Only rdf 0.8 and the control have targets to attack at late generations. The rdf 0 and
rdf 0.5 results rest on generations 1 to 3 and on small counts.

**Cost** (4 L40S, `plots/collapse_cost_0.5b_3b.md`): a generation takes 32 minutes at 0.5B
and 1.8 hours at 3B; rdf 0 reaches HumanEval 0 at generation 2 after 1.5 and 5.1 hours. A
two-point power-law fit puts 32B at about 9 hours per generation.

## 4. Attack against the random-suffix control

### 4.1 Optimized surrogate, vulnerability targets

Targets broken / attackable targets, pooled over generations 1 to 9.

**Sampled verdict**

| root | lineage | attack | random suffix | p |
|---|---|---|---|---|
| main 0.5B | rdf 0 | 3/16 (19%) | 1/16 (6%) | 0.50 |
| main 0.5B | rdf 0.5 | 4/34 (12%) | 3/34 (9%) | 1.00 |
| main 0.5B | rdf 0.8 | 23/109 (21%) | 14/109 (13%) | 0.049 |
| main 0.5B | control | 63/139 (45%) | 31/139 (22%) | < 0.001 |
| main 3B | rdf 0 | 7/27 (26%) | 0/27 (0%) | 0.016 |
| main 3B | rdf 0.5 | 2/43 (5%) | 5/43 (12%) | 0.38 |
| main 3B | rdf 0.8 | 10/107 (9%) | 2/107 (2%) | 0.008 |
| main 3B | control | 48/172 (28%) | 18/172 (10%) | < 0.001 |
| floor 16, 0.5B | rdf 0 | 3/15 (20%) | 1/15 (7%) | 0.63 |
| floor 16, 0.5B | rdf 0.5 | 3/33 (9%) | 6/33 (18%) | 0.38 |
| floor 16, 0.5B | rdf 0.8 | 12/107 (11%) | 13/107 (12%) | 1.00 |
| floor 16, 0.5B | control | 63/138 (46%) | 32/138 (23%) | < 0.001 |
| split, own anchor | rdf 0.8 | 8/126 (6%) | 9/126 (7%) | 1.00 |
| split, own anchor | control | 27/147 (18%) | 23/147 (16%) | 0.52 |
| split, attacker's anchor | rdf 0.8 | 20/135 (15%) | 12/135 (9%) | 0.12 |
| split, attacker's anchor | control | 23/150 (15%) | 27/150 (18%) | 0.54 |

**Greedy verdict**

| root | lineage | attack | random suffix | p |
|---|---|---|---|---|
| main 0.5B | rdf 0 | 4/17 (24%) | 4/17 (24%) | 1.00 |
| main 0.5B | rdf 0.5 | 10/35 (29%) | 5/35 (14%) | 0.13 |
| main 0.5B | rdf 0.8 | 27/93 (29%) | 17/93 (18%) | 0.076 |
| main 0.5B | control | 48/140 (34%) | 23/140 (16%) | < 0.001 |
| main 3B | rdf 0 | 7/28 (25%) | 2/28 (7%) | 0.18 |
| main 3B | rdf 0.5 | 5/38 (13%) | 2/38 (5%) | 0.38 |
| main 3B | rdf 0.8 | 16/94 (17%) | 7/94 (7%) | 0.023 |
| main 3B | control | 39/158 (25%) | 17/158 (11%) | < 0.001 |
| floor 16, 0.5B | rdf 0 | 2/15 (13%) | 5/15 (33%) | 0.38 |
| floor 16, 0.5B | rdf 0.5 | 7/24 (29%) | 6/24 (25%) | 1.00 |
| floor 16, 0.5B | rdf 0.8 | 16/67 (24%) | 14/67 (21%) | 0.75 |
| floor 16, 0.5B | control | 61/140 (44%) | 29/140 (21%) | < 0.001 |
| split, own anchor | rdf 0.8 | 20/118 (17%) | 22/118 (19%) | 0.80 |
| split, own anchor | control | 36/144 (25%) | 24/144 (17%) | 0.050 |
| split, attacker's anchor | rdf 0.8 | 25/118 (21%) | 22/118 (19%) | 0.63 |
| split, attacker's anchor | control | 39/144 (27%) | 24/144 (17%) | 0.011 |

Reading:

- The control lineage separates in all three full-data roots under both verdicts.
- The collapsed lineages separate in 3 of 11 cells (sampled) and 1 of 11 (greedy), all in
  the main root. The 3B rdf 0.8 edge is the most consistent one (both verdicts).
- The 0.5B rdf 0.8 edge in the main root (21% against 13%) is gone in the length-floor root
  (11% against 12%). The two roots differ in the length floor and in every sampled draw, so
  this is the size of the run-to-run variation at this budget.

### 4.2 The edges on collapsed lineages are early

Optimized surrogate, sampled verdict, hit rate per generation (main root):

| size | lineage | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B | rdf 0.8 | 44% | 33% | 46% | 9% | 0% | 15% | 0% | 9% | 10% |
| 0.5B | control | 47% | 44% | 44% | 50% | 47% | 36% | 43% | 53% | 43% |
| 3B | rdf 0.8 | 30% | 23% | 8% | 0% | 0% | 0% | 0% | 0% | 0% |
| 3B | control | 37% | 25% | 32% | 20% | 26% | 26% | 17% | 42% | 26% |

At 0.5B rdf 0.8, generations 1 to 3 give 18/44 hits against 11/44 for random suffixes;
generations 4 to 9 give 5/65 against 3/65. At 3B rdf 0.8, generations 4 to 9 give 0/61. The
control lineage holds its rate at every generation.

### 4.3 Direct attack and plain surrogate, vulnerability targets

Sampled verdict:

| root | lineage | direct | random | p | plain surrogate | random | p |
|---|---|---|---|---|---|---|---|
| main 0.5B | rdf 0 | 8/30 (27%) | 4/30 (13%) | 0.29 | 2/17 (12%) | 1/17 (6%) | 1.00 |
| main 0.5B | rdf 0.5 | 7/49 (14%) | 6/49 (12%) | 1.00 | 3/39 (8%) | 2/39 (5%) | 1.00 |
| main 0.5B | rdf 0.8 | 20/128 (16%) | 17/128 (13%) | 0.65 | 7/104 (7%) | 12/104 (12%) | 0.18 |
| main 0.5B | control | 47/157 (30%) | 36/157 (23%) | 0.12 | 13/142 (9%) | 36/142 (25%) | < 0.001 |
| main 3B | rdf 0 | 8/48 (17%) | 0/48 (0%) | 0.008 | 3/25 (12%) | 1/25 (4%) | 0.50 |
| main 3B | rdf 0.5 | 12/58 (21%) | 7/58 (12%) | 0.23 | 1/37 (3%) | 6/37 (16%) | 0.13 |
| main 3B | rdf 0.8 | 10/128 (8%) | 5/128 (4%) | 0.27 | 1/104 (1%) | 2/104 (2%) | 1.00 |
| main 3B | control | 33/191 (17%) | 21/191 (11%) | 0.10 | 14/167 (8%) | 16/167 (10%) | 0.84 |
| floor 16, 0.5B | rdf 0.8 | 18/126 (14%) | 19/126 (15%) | 1.00 | 5/100 (5%) | 6/100 (6%) | 1.00 |
| floor 16, 0.5B | control | 48/155 (31%) | 30/155 (19%) | 0.013 | 18/139 (13%) | 28/139 (20%) | 0.076 |

- **Direct.** Under the greedy verdict the direct attack separates on the control lineage in
  every root (main 0.5B 50/157 against 26/157, 3B 41/174 against 17/174, floor 16 49/156
  against 31/156, split 58/160 against 27/160; all p < 0.02) and on 3B rdf 0 and rdf 0.5
  (p = 0.022, 0.008). On rdf 0.8 it does not separate in any root under either verdict.
- **Optimized surrogate against direct.** On the control lineage the optimized surrogate
  breaks more targets than the white-box search (0.5B sampled: 45% against 30%). The two
  differ in the objective as well as in the model, so this compares objectives, not access.
- **Plain surrogate.** Never significantly above random, in any root or verdict. On the
  0.5B control (main root, sampled) it never breaks a target that random suffixes do not
  also break (0 against 23 discordant cells).

### 4.4 Correctness targets

Five targets, so every count is small. Sampled verdict, pooled:

| root | lineage | direct | random | optimized surrogate | random |
|---|---|---|---|---|---|
| main 0.5B | rdf 0.5 | 8/22 | 9/22 | 6/10 | 7/10 |
| main 0.5B | rdf 0.8 | 9/48 | 5/48 | 1/39 | 5/39 |
| main 0.5B | control | 16/50 | 1/50 | 2/45 | 1/45 |
| main 3B | rdf 0.8 | 3/45 | 3/45 | 1/39 | 3/39 |
| main 3B | control | 3/49 | 0/49 | 0/45 | 0/45 |
| floor 16, 0.5B | rdf 0.8 | 10/39 | 4/39 | 6/35 | 3/35 |
| floor 16, 0.5B | control | 14/50 | 0/50 | 1/45 | 0/45 |
| split, own | rdf 0.8 | 10/47 | 1/47 | 3/42 | 2/42 |
| split, own | control | 15/50 | 0/50 | 0/44 | 0/44 |

- The direct attack separates on the 0.5B control in every root (p ≤ 0.001 sampled) and on
  the split root's rdf 0.8 (p = 0.012). At 3B it finds almost nothing.
- Neither surrogate separates from random on this suite in any cell.
- On rdf 0 and rdf 0.5 a random suffix breaks most of the few attackable targets (7 of 10 at
  rdf 0.5), so those cells cannot show an attack effect.

### 4.5 Does the surrogate predict the real verdict?

Optimized surrogate, vulnerability targets, sampled verdict, per behavioural check:

| root | lineage | checks | real hits | predicted | false alarms |
|---|---|---|---|---|---|
| main 0.5B | rdf 0.8 | 1090 | 61 | 31 | 203 |
| main 0.5B | control | 1390 | 209 | 106 | 149 |
| main 3B | rdf 0.8 | 1070 | 22 | 16 | 243 |
| main 3B | control | 1720 | 113 | 67 | 423 |
| split, attacker's anchor | rdf 0.8 | 1350 | 28 | 2 | 26 |
| split, attacker's anchor | control | 1500 | 35 | 0 | 16 |

On the 0.5B control the surrogate's own verdict is right 42% of the time it signals a hit; on
rdf 0.8, 13%; at 3B, 14% and 6%. The plain surrogate predicts 0 to 2 real hits per lineage.

## 5. Why: where the lineages sit relative to the surrogate

The surrogate is a point on the line from the base model's logits through generation 0's.
The collapse ablation measures each real generation against that line at every target's
decision position (medians over targets, 0.5B main root):

| lineage | cosine to the line, g1 → g9 | best position on the line, g1 → g9 | KL to the attack's surrogate |
|---|---|---|---|
| control | 0.96 → 0.96 | 1.08 → 1.03 | 0.01 to 0.04 |
| rdf 0 | 0.85 → 0.49 | 1.33 → 1.51 | 0.001 to 0.02 |
| rdf 0.5 | 0.77 → −0.04 | 0.84 → −0.44 | 0.03 to 0.22 |
| rdf 0.8 | 0.75 → 0.30 | 0.68 → 0.01 | 0.14 to 0.51 |

Three observations:

- **The control lineage does not move.** Its best position on the line is 1, generation 0's
  own, at every generation. Fine-tuning again on real data reproduces generation 0. The
  surrogate is close to it because generation 0 is close to it.
- **The collapsed lineages leave the line.** At rdf 0.5 the direction of change is nearly
  unrelated to the surrogate's by generation 4 (cosine 0.07); at rdf 0.8 the cosine falls
  to 0.3 to 0.4. In both, the best position moves back toward the base model (below 1),
  not forward.
- **n = g + 1 is far from every real model.** The KL from the control lineage to the plain
  tilt grows from 0.3 to 19.6 nats. This is why the plain surrogate fails.

At 3B the rdf 0.8 lineage stays closer to the line (cosine 0.81 to 0.89), and that is the
collapsed lineage with the most consistent edge over random.

The decision margin follows the affine law to within 0.2 nats on the control lineage when
the position is fitted, 0.7 to 0.8 on rdf 0.5 and rdf 0.8, and 4.7 on rdf 0. The margin
moves in the direction the slope predicts in 95% of control cells, 58% at rdf 0.8 and 46% at
rdf 0.5 (0.5B).

**Hit rate by closeness to the line** (optimized surrogate, sampled verdict, attackable
cells of all lineages split into quartiles of the cosine):

| root | lowest quartile | second | third | highest |
|---|---|---|---|---|
| main 0.5B | 10% | 32% | 19% | 67% |
| floor 16, 0.5B | 8% | 13% | 27% | 66% |
| main 3B | 7% | 6% | 21% | 43% |
| split, own anchor | 3% | 12% | 18% | 20% |

The quartiles pool the lineages, and the top quartile is mostly control cells, so this
restates the lineage difference along a continuous axis; it does not separate geometry from
lineage. The direct attack shows a weaker trend of the same sign (11%, 18%, 25%, 28% at
0.5B).

## 6. Is anything dormant?

A timebomb needs hits that appear late: suffixes that spare generation 0 and break a later
generation. The survival re-verification scores every found hit against every generation.

**Share of re-verified hits that spare generation 0** (optimized surrogate):

| root | lineage | sampled | greedy |
|---|---|---|---|
| main 0.5B | control | 15/58 (26%) | 55/100 (55%) |
| main 3B | control | 12/26 (46%) | 30/57 (53%) |
| floor 16, 0.5B | control | 31/88 (35%) | 85/227 (37%) |
| main 0.5B | rdf 0.8 | 9/16 (56%) | 17/41 (41%) |
| main 3B | rdf 0.8 | 1/11 (9%) | 18/33 (55%) |

On the control lineage, where the attack works, about half to three quarters of the hits
already break the attacker's own generation-0 model.

**Hits found at generation 1 of rdf 0.8, scored on later generations** (0.5B, optimized
surrogate, sampled): 16/29 at generation 0, 13/22 at generation 1, 5/36 at generation 2,
2/18 at generation 3, then 0 or 1 at every generation from 4 on. A suffix found early stops
working as the lineage collapses. On the control lineage a hit found at any generation is a
hit at every other generation at roughly 25% to 60%, generation 0 included.

**Do hits on collapsed lineages spare the control?** Among re-verified hits, 64% to 100%
spare the control at the same generation (small counts). Scored against the control at *any*
generation, 82% (0.5B) and 86% (3B) of the rdf 0.8 hits of the optimized surrogate (sampled)
break it somewhere. They are mostly suffixes that work on fine-tunes of this base model in
general.

**Re-verification.** A recorded sampled hit is one draw of a 5-sample majority taken at the
best of 10 checks. Re-drawn, 19% to 40% reproduce at 0.5B (control, optimized: 80/209) and 0%
to 78% at 3B. The random-suffix control has the same best-of-10 structure, so section 4
compares like with like, but both absolute rates overstate what a single use of a suffix
achieves.

## 7. Length floor: 16 against 128 tokens

0.5B, main root (floor 128) against `runs/min16` (floor 16), generation 9 unless noted:

| measure | lineage | floor 128 | floor 16 |
|---|---|---|---|
| HumanEval | rdf 0 / 0.5 / 0.8 / control | 0.00 / 0.04 / 0.21 / 0.43 | 0.00 / 0.03 / 0.14 / 0.46 |
| compiling code in corpus | rdf 0 / 0.5 / 0.8 | 0.45 / 0.35 / 0.25 | 0.57 / 0.39 / 0.27 |
| corpus perplexity | rdf 0 / 0.5 / 0.8 | 47.6 / 51.9 / 62.8 | 51.9 / 55.1 / 65.6 |
| trigram entropy | rdf 0 / 0.5 / 0.8 | 1.26 / 1.16 / 1.19 | 1.31 / 1.22 / 1.24 |
| optimized surrogate, control (sampled) | | 45% vs 22% random | 46% vs 23% random |
| optimized surrogate, rdf 0.8 (sampled) | | 21% vs 13% random | 11% vs 12% random |
| cosine to the line, control | | 0.95 to 0.97 | 0.96 |

The lower floor leaves somewhat more compiling code in the rdf 0 corpus; the other collapse
measures barely move. Collapse at penalty 1.2 is not an artefact of the length floor, and
the control-lineage attack result does not depend on it. The rdf 0.8 attack rate halves,
down to the random floor (section 4.1).

## 8. Disjoint data

The attacker fine-tunes generation 0 on one half of the corpus; the victim's lineage runs on
the other half. The surrogate is then built from the attacker's generation 0 and scored on
the victim's real models, next to a surrogate built from the victim's own generation 0.

Optimized surrogate, vulnerability targets (from section 4.1):

| verdict | lineage | own anchor | random | attacker's anchor | random |
|---|---|---|---|---|---|
| sampled | rdf 0.8 | 8/126 (6%) | 7% | 20/135 (15%) | 9% |
| sampled | control | 27/147 (18%) | 16% | 23/150 (15%) | 18% |
| greedy | rdf 0.8 | 20/118 (17%) | 19% | 25/118 (21%) | 19% |
| greedy | control | 36/144 (25%) | 17% | 39/144 (27%) | 17% |

The attacker's anchor does no worse than the victim's own. Under the greedy verdict both
show a similar, modest edge on the control lineage (p = 0.050 and 0.011). Under the sampled
verdict neither separates from random on either lineage. The clear control-lineage result of
the full-data roots (45% against 22%) did not reproduce here with either anchor, so "no
worse than the own anchor" is a comparison between two weak results.

**A confound explains at least part of this.** The solved per-target factors are larger in
this root. On the control lineage the main root attacks at factors of 1.04 to 1.43 for five
targets; the split root's own anchor gives 1.35 to 2.39, and the attacker's anchor puts every
target at the cap of 2.5. The real control models sit at position 1 on the line (section 5),
so a surrogate at 2.5 is far from them: the measured KL from the control lineage to the
own-anchor surrogate is 0.25 to 0.51 nats in this root against 0.01 to 0.04 in the main
root. The surrogate's hit predictions are correspondingly poor (16 of 53 real hits predicted
on the control with the own anchor, 0 of 35 with the attacker's). Each half-corpus
generation 0 is a smaller step from the base model, which is a plausible cause of the larger
solved factors, but that was not tested.

So the experiment does not yet settle whether the surrogate needs the victim's data. It
shows that the factor rule did not carry over to a half-size corpus.

The victim lineage also collapses less on half the data: rdf 0.8 holds a HumanEval of 0.25 to
0.32 through generation 9 against 0.14 to 0.37 in the floor-16 root.

## 9. Caveats

- **One seed.** Every lineage is one run at seed 1337. The 0.5B rdf 0.8 difference between
  the main and the floor-16 root is the only replication available, and it did not replicate.
- **Small budget.** One restart, 100 steps, 10 checks per target. A larger budget could raise
  attack rates; it would raise the matched random control's rate as well.
- **Small target sets.** 25 vulnerability targets, of which 8 to 20 are attackable in a
  generation; 5 correctness targets. Pooling over generations reuses the same targets.
- **Uncorrected tests.** 182 attack-against-random comparisons were computed, over every
  root, suite, verdict, method and lineage. The control-lineage results (p < 0.001, repeated
  in three roots) survive any correction; the three collapsed-lineage edges at p = 0.008 to
  0.049 would not all survive one.
- **High random floor at 0.5B.** A random suffix breaks 5% to 25% of attackable
  vulnerability targets in 10 tries (2% to 5% per try). At 3B the floor is 0% to 16%.
- **Greedy hits do not all re-verify** (59% to 82% at 0.5B; 71% to 96% at 3B, apart from one
  cell of 2/6). Greedy decoding is deterministic, so this is unexpected and not yet
  explained.
- **Two reporting gaps.** The floor-16 geometry table has no surrogate KL for rdf 0.8 at
  generation 9 (measured before that cell was filled in). The split-anchor tables print the
  victim's sink screen, not the attacker's, which is the one that pass used.

## 10. What this means for the paper

**Supported by the data**

1. A characterization of collapse under code self-training at two model sizes and four
   mixtures, with decoding effects (repetition penalty, length floor) separated out, and its
   cost.
2. A surrogate built from the base model and one fine-tune predicts, and transfers suffixes
   to, later models of a lineage that keeps training on real data. This holds at both sizes
   and in three roots, against a matched random-suffix control.
3. The reason is geometric and measurable: such a lineage stays where generation 0 is.
4. Collapse works against the attack. It removes attackable targets, moves the model off the
   surrogate's line, and makes early suffixes stop working.
5. Random suffixes are a necessary control for this kind of result. At 0.5B they account for
   most of the hit rate on collapsed lineages.

**Not supported**

1. That collapsed models are specifically vulnerable to suffixes prepared in advance. The
   edges on collapsed lineages are small, early, and not replicated.
2. That hits are dormant, in the sense of sparing the current model and activating later.
3. That the surrogate works without the victim's data. The evidence is weak and the
   experiment is confounded.
4. That n = g + 1 approximates generation g. It does not, on any lineage.

**Consequence for framing.** The title's claim (timebomb attacks against collapsed LLMs) is
the opposite of what the grid shows. The finding the data carries is that iterated
fine-tuning on real data keeps a model predictable from one fine-tune, and that collapse
destroys that predictability along with the model's capability. That fits the ACL 2027 theme
track on homogenization and knowledge collapse as an analysis paper, with the attack as the
probe and the random-suffix control and geometry as the method contribution.

**What would firm up the open points** (measurement, in order of value)

1. A second seed for the 0.5B rdf 0.8 and control lineages, to put an interval on the
   run-to-run variation that section 4.1 exposed.
2. The disjoint-data comparison with the surrogate at a fixed factor near 1 for both
   anchors, which removes the confound of section 8.
3. A hit rule that requires a hit to reproduce on fresh draws, applied to attack and random
   control alike, so absolute rates mean what a reader expects.
4. An explanation of the greedy re-verification gap before any greedy number is published.

## 11. Sources

| content | file |
|---|---|
| utility, hit rates, random control, surrogate quality | `runs/<root>/plots/tables_<size>.md` |
| split-anchor pass | `runs/split_min16/victim/plots/tables_0.5b_splitanchor.md` |
| survival, specificity, geometry, quartiles | `runs/<root>/plots/ablation_collapse.md` |
| cost | `runs/fresh/plots/collapse_cost_0.5b_3b.md` |
| per-target factors (section 8) | `task_factors` in `runs/<root>/attack_results/attack_gen*_..._sinkobj_nsink_em_logit_surrogate*.json` |

The paired counts and p-values of section 4 are not in those tables. They were computed from
the result files: for each cell, every attackable target that the random control covered
contributes one pair (broken by the attack, broken by a random suffix), pooled over the
generations, exact two-sided McNemar on the discordant pairs. The marginal counts match the
pooled tables.
