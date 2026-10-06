#!/usr/bin/env bash
#
# run_paper.sh — every experiment behind the paper, in dependency order, resumable.
#
# The claim this pipeline supports has three parts, and each one needs a different cell of the
# grid below:
#
#   1. recursive training on sampled outputs shifts a code model's security margins
#        -> the `margin` phase on the collapse lineages, read against the `control` lineage
#           (--real_data_fraction 1.0), which has the same generations, seeds, corpus size and
#           optimizer-step count and differs only in where the training text comes from. A shift
#           that shows up there too belongs to iterated fine-tuning, not to self-training.
#   2. the shift is exploitable by an attacker who never sees the collapsed model
#        -> the `attack` phase in transfer mode (-m logit), with the `ablation` phase's factor
#           ladder underneath it: at -sf 1 the "surrogate" is the generation-0 checkpoint used
#           unchanged, so if that rung scores as well as the extrapolated one, the extrapolation
#           adds nothing and the result reduces to known base-to-fine-tune transfer.
#   3. exploitability is decoding-conditional
#        -> every attack is swept under two verdicts (sampled majority at the deployment decoding,
#           and greedy), and the `survival` phase turns the gap into a curve over temperature plus
#           a cost curve in search steps.
#
# Nothing here is new machinery; it is the commands from docs/sampling_experiments.md with the
# bookkeeping that a multi-day run needs — ordering, resumability, per-step logs and a status
# table at the end. Read that document for what each experiment means.
#
# Usage:
#   ./run_paper.sh --dry-run                 # print the whole plan, run nothing
#   ./run_paper.sh --list                    # one line per step, with its phase
#   ./run_paper.sh                           # run everything (asks once before starting)
#   ./run_paper.sh --phases margin,attack    # a subset, in the order below
#   ./run_paper.sh --only main05             # one lineage
#   ./run_paper.sh --phases plots --force    # redraw every figure
#
# Resumability: every underlying script skips work whose output file already exists, so an
# interrupted run is continued by re-invoking this script with the same arguments. --force
# overrides that, per phase, and is passed down to the sweeps.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2

# ───────────────────────────────── configuration ──────────────────────────────────────────
# The repo venv, not whatever `python` resolves to. A shell with another project's venv active
# picks up a trl/transformers pair this pipeline does not run on, and the failure surfaces deep
# inside a training worker hours later. Everything below, including the shell sweeps (which read
# $PYTHON), is pinned to this interpreter.
PYTHON="${PYTHON:-$SCRIPT_DIR/venv/bin/python}"
export PYTHON

ROOT="${ROOT:-.}"                       # run root holding model_outputs/ and attack_results/
GENERATIONS="${GENERATIONS:-9}"         # highest generation index; the lineages train 0..N
BLOCK_SIZE="${BLOCK_SIZE:-512}"
MODEL_SIZE="${MODEL_SIZE:-0.5b}"

# GPUs. Training and dataset generation fan out over several devices; the attack shards its
# (task, restart) units over whatever it can see; the single-model measurements want one card.
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
ATTACK_GPUS="${ATTACK_GPUS:-0,1,2,3}"
EVAL_GPU="${EVAL_GPU:-0}"

# Search budget, passed through to run_attack.py. The defaults are the paper's.
RESTARTS="${RESTARTS:-3}"
NUM_STEPS="${NUM_STEPS:-250}"
VERIFY_EVERY="${VERIFY_EVERY:-10}"

# Survival phase: the temperature curve, and how many samples each point averages.
TEMPERATURES="${TEMPERATURES:-0,0.3,0.5,0.7,1.0}"
SURVIVAL_SAMPLES="${SURVIVAL_SAMPLES:-16}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"

# Margin-aware search: the wrong-code cross-entropy a hit must reach. Its own root, because
# --max_wrong_loss is deliberately not part of the result file name.
MAX_WRONG_LOSS="${MAX_WRONG_LOSS:-0.05}"
MWL_STEPS="${MWL_STEPS:-500}"

# The surrogate factor ladder. 1 is the ablation's floor: the generation-0 model used unchanged.
FACTOR_LADDER="${FACTOR_LADDER:-1 1.25 1.5 2 2.5 3}"

NO_USETEX="${NO_USETEX:-0}"             # 1 draws figures without a TeX install

# ── the lineages ──
# name | path | rdf | temperature | top_p | top_k | seed | description
#
# One --path can hold several mixtures (the mixture is in every artifact name), so the cells that
# differ only in --real_data_fraction share $ROOT. The ones that differ in *decoding* do not: the
# sampling parameters reach no artifact name, so two of them under one root would overwrite each
# other's checkpoints.
LINEAGES=(
  "main05|$ROOT|0.5|0.7|0.8|20|1337|primary collapse cell, Qwen2.5 decoding"
  "main08|$ROOT|0.8|0.7|0.8|20|1337|second mixture, slower collapse"
  "control|$ROOT|1.0|0.7|0.8|20|1337|NON-COLLAPSING CONTROL: human data every generation"
  "ancestral|$ROOT/runs/ancestral|0.8|1.0|1.0|-1|1337|untruncated sampling, collapse without truncation"
  "harsh|$ROOT/runs/harsh|0.8|0.7|0.6|10|1337|harsher truncation, faster collapse"
  "seed2|$ROOT/runs/seed2|0.5|0.7|0.8|20|2337|variance: primary cell, second seed"
  "seed3|$ROOT/runs/seed3|0.5|0.7|0.8|20|3337|variance: primary cell, third seed"
)

# The cell the expensive single-cell experiments (ablation, survival) are run on.
PRIMARY="${PRIMARY:-main05}"

PHASES_ALL="collapse,utility,margin,attack,ablation,survival,eval,plots"
PHASES="${PHASES:-$PHASES_ALL}"
# Not in the default set: `extrapolation` is stage 2 of the repo pipeline (surrogate corpora and
# their perplexity figure) and `transfer` is the cross-run transfer experiment, which trains a
# second collapse run of its own. Both are opt-in via --phases because each roughly doubles the
# wall clock and neither is load-bearing for the three claims above.

ONLY=""
FORCE=0
DRY_RUN=0
LIST_ONLY=0
ASSUME_YES=0

# ─────────────────────────────────── arguments ────────────────────────────────────────────
usage() {
    cat <<'EOF'
Runs every experiment behind the paper, in order.

Options:
  -p, --path PATH        run root holding model_outputs/ and attack_results/ (default: .)
  -n, --generations N    highest generation index; lineages train 0..N (default: 9)
      --phases LIST      comma separated subset, run in this order:
                           collapse, utility, extrapolation, margin, attack,
                           ablation, survival, eval, plots, transfer
                         (default: everything but extrapolation and transfer)
      --only NAME        restrict to one lineage: main05 main08 control ancestral harsh
                         seed2 seed3
      --primary NAME     lineage the ablation and survival phases use (default: main05)
      --force            redo work whose output already exists
      --dry-run          print every command without running it
      --list             print the step list and exit
      --no-usetex        draw figures without LaTeX
  -y, --yes              do not ask before starting
  -h, --help             this message

Environment overrides: PYTHON TRAIN_GPUS ATTACK_GPUS EVAL_GPU RESTARTS NUM_STEPS VERIFY_EVERY
TEMPERATURES SURVIVAL_SAMPLES MAX_WRONG_LOSS FACTOR_LADDER BLOCK_SIZE MODEL_SIZE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--path)        ROOT="$2";        shift 2 ;;
        -n|--generations) GENERATIONS="$2"; shift 2 ;;
        --phases)         PHASES="$2";      shift 2 ;;
        --only)           ONLY="$2";        shift 2 ;;
        --primary)        PRIMARY="$2";     shift 2 ;;
        --force)          FORCE=1;          shift ;;
        --dry-run)        DRY_RUN=1;        shift ;;
        --list)           LIST_ONLY=1;      shift ;;
        --no-usetex)      NO_USETEX=1;      shift ;;
        -y|--yes)         ASSUME_YES=1;     shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

NUM_GENERATIONS=$(( GENERATIONS + 1 ))   # run_baseline.py counts generations, the rest index them
LOG_DIR="$ROOT/paper_logs"
PLOTS_DIR="$ROOT/plots"
RESULTS_DIR="$ROOT/attack_results"

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }
wants_lineage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

usetex_flag_py() { (( NO_USETEX )) && echo "--no_usetex" || true; }
usetex_flag_sh() { (( NO_USETEX )) && echo "--no-usetex" || true; }
force_flag()     { (( FORCE ))     && echo "--force"     || true; }

# True when `dir` holds at least one attack result under the hit rule `tag`. Asked of the loader
# the figures themselves use, rather than guessed from file names, so the two cannot disagree.
has_results() {
    local dir="$1" tag="$2"
    [[ -d "$dir" ]] || return 1
    "$PYTHON" - "$dir" "$tag" <<'HAS_RESULTS_PY' 2>/dev/null
import sys
from utils.vuln_results import load_records
sys.exit(0 if load_records(sys.argv[1], run_tag=sys.argv[2]) else 1)
HAS_RESULTS_PY
}

# a step that had nothing to do: recorded in the summary, but not a failure
skip() {
    echo
    echo "── $1 ── skipped: $2"
    STEP_NAMES+=("$1"); STEP_STATUS+=("skipped"); STEP_SECONDS+=(0)
}

# ───────────────────────────────── step bookkeeping ───────────────────────────────────────
STEP_NAMES=(); STEP_STATUS=(); STEP_SECONDS=()
FAILURES=0
STARTED_AT=$SECONDS

# run <step-name> <command...>
#
# Logs to paper_logs/<step-name>.log, records the outcome, and never aborts the pipeline: a
# failed cell should not cost the cells after it, which are usually independent. The summary at
# the end is what to read, and a non-zero exit code says something in it failed.
run() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    local started=$SECONDS

    echo
    echo "── $name ──────────────────────────────────────────────"
    echo "   \$ $*"
    if (( DRY_RUN )); then
        STEP_NAMES+=("$name"); STEP_STATUS+=("dry-run"); STEP_SECONDS+=(0)
        return 0
    fi

    mkdir -p "$LOG_DIR"
    echo "   log: $log"
    "$@" 2>&1 | tee "$log"
    local code=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - started ))

    STEP_NAMES+=("$name"); STEP_SECONDS+=("$elapsed")
    if (( code == 0 )); then
        STEP_STATUS+=("ok")
        echo "   done in $(printf '%02d:%02d:%02d' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    else
        STEP_STATUS+=("FAILED ($code)")
        FAILURES=$(( FAILURES + 1 ))
        echo "   FAILED with exit code $code — see $log"
    fi
    return 0
}

# ──────────────────────────────────── preflight ───────────────────────────────────────────
preflight() {
    local problems=0

    if [[ ! -x "$PYTHON" ]]; then
        echo "error: no interpreter at $PYTHON. Set PYTHON=... or create the venv." >&2
        problems=1
    elif ! CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -c "import torch, transformers, unsloth" 2>/dev/null; then
        echo "warning: $PYTHON cannot import torch/transformers/unsloth with a GPU visible." >&2
        echo "         The collapse phase needs all three; the attack phases need the first two." >&2
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "warning: no nvidia-smi; this pipeline is CUDA-only." >&2
    fi

    if (( NO_USETEX == 0 )) && ! command -v latex >/dev/null 2>&1; then
        echo "warning: no LaTeX on PATH and --no-usetex was not given; the plot phases will fail." >&2
    fi

    for script in run_attack_sweep.sh run_vuln_eval.sh run_baseline.py run_margin.py \
                  run_attack_vuln.py run_vuln_plots.py utils/survival_sweep.py; do
        [[ -e "$script" ]] || { echo "error: missing $script" >&2; problems=1; }
    done

    return $problems
}

# ───────────────────────────────────── phases ─────────────────────────────────────────────
# Each phase is a loop over the lineages it applies to. The underlying scripts own the
# "already done" checks, so a phase is cheap to re-enter.

phase_collapse() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf temp top_p top_k seed _desc <<< "$record"
        wants_lineage "$name" || continue
        run "collapse_${name}" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" run_baseline.py \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$MODEL_SIZE" \
            -rdf "$rdf" -tp "$temp" -tpp "$top_p" -tpk "$top_k" -sd "$seed" -p "$path"
    done
}

# Utility curves and the surrogate calibration. --calibrate writes the surrogate_factor json that
# the ablation's `-sf calibrated` rung reads, so this phase has to precede `ablation`.
phase_utility() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf _t _p _k _s _desc <<< "$record"
        wants_lineage "$name" || continue
        run "perplexity_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m utils.evaluate_perplexity \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$MODEL_SIZE" -rdf "$rdf" -p "$path" \
            --calibrate $(usetex_flag_py)
        run "correctness_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m utils.evaluate_correctness \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$MODEL_SIZE" -rdf "$rdf" -p "$path" \
            $(usetex_flag_py)
    done
}

# Stage 2 of the repo pipeline: the surrogate's own corpora and their perplexity figure. Opt-in.
phase_extrapolation() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf temp top_p top_k _s _desc <<< "$record"
        wants_lineage "$name" || continue
        [[ "$name" == "$PRIMARY" ]] || continue   # one cell is enough for the surrogate figure
        run "extrapolation_${name}" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" run_extrapolation.py \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$MODEL_SIZE" -rdf "$rdf" \
            -tp "$temp" -tpp "$top_p" -tpk "$top_k" -m logit -p "$path"
    done
}

# The decoding-independent measurement: decision margin, sequence margin and entropy per
# generation. Minutes per lineage, and the thing that makes collapse *severity* rather than
# generation index the x-axis of the attribution figure.
phase_margin() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf _t _p _k _s _desc <<< "$record"
        wants_lineage "$name" || continue
        run "margin_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" run_margin.py \
            -msz "$MODEL_SIZE" -rdf "$rdf" -n "$GENERATIONS" -bs "$BLOCK_SIZE" -p "$path"
    done
}

# The attacks. Four sweeps per lineage: direct (white-box upper bound) and logit-surrogate
# transfer (the threat model), each under the sampled-majority verdict and the greedy one. The
# anchor is held in all of them, which is the default — a hit needs the target broken while both
# the pristine baseline and the generation-0 anchor keep writing the secure variant.
phase_attack() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf _t _p _k _s _desc <<< "$record"
        wants_lineage "$name" || continue
        for method in none logit; do
            run "attack_${name}_${method}_sampled" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
                ./run_attack_sweep.sh -n "$GENERATIONS" -p "$path" -b "$BLOCK_SIZE" \
                -msz "$MODEL_SIZE" -rdf "$rdf" -m "$method" --vuln $(force_flag) \
                -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY"
            run "attack_${name}_${method}_greedy" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
                ./run_attack_sweep.sh -n "$GENERATIONS" -p "$path" -b "$BLOCK_SIZE" \
                -msz "$MODEL_SIZE" -rdf "$rdf" -m "$method" --vuln $(force_flag) \
                -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" -vt 0
        done
    done
}

# Everything that asks "is the effect what we say it is", on the primary cell only.
phase_ablation() {
    local record path rdf
    record="$(printf '%s\n' "${LINEAGES[@]}" | grep "^${PRIMARY}|")"
    if [[ -z "$record" ]]; then
        echo "error: --primary $PRIMARY is not a known lineage" >&2
        FAILURES=$(( FAILURES + 1 ))
        return
    fi
    IFS='|' read -r _n path rdf _t _p _k _s _desc <<< "$record"

    # 1. the surrogate factor ladder. Each value files its own result (utils.naming.factor_mode_tag),
    #    so the rungs do not overwrite each other or the default n = g + 1 sweep from `attack`.
    for n in $FACTOR_LADDER; do
        run "ablation_factor_n${n}" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
            ./run_attack_sweep.sh -n "$GENERATIONS" -p "$path" -b "$BLOCK_SIZE" \
            -msz "$MODEL_SIZE" -rdf "$rdf" -m logit --vuln $(force_flag) \
            -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" -sf "$n"
    done
    for policy in auto calibrated; do
        run "ablation_factor_${policy}" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
            ./run_attack_sweep.sh -n "$GENERATIONS" -p "$path" -b "$BLOCK_SIZE" \
            -msz "$MODEL_SIZE" -rdf "$rdf" -m logit --vuln $(force_flag) \
            -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" -sf "$policy"
    done

    # 2. plain forward transfer. At -sf 1 the surrogate *is* the anchor, so under the three-sided
    #    hit rule the surrogate can never have predicted its own break and the agreement numbers
    #    are 0 by construction. This twin drops the anchor and measures what a suffix optimized on
    #    generation 0 reaches on generation n — the baseline the extrapolation has to beat.
    run "ablation_forward_transfer" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
        ./run_attack_sweep.sh -n "$GENERATIONS" -p "$path" -b "$BLOCK_SIZE" \
        -msz "$MODEL_SIZE" -rdf "$rdf" -m logit --vuln $(force_flag) \
        -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" -sf 1 -na

    # 3. the two-sided hit rule, for comparison with every result predating the anchor condition.
    run "ablation_no_anchor" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
        ./run_attack_sweep.sh -n "$GENERATIONS" -p "$path" -b "$BLOCK_SIZE" \
        -msz "$MODEL_SIZE" -rdf "$rdf" -m none --vuln $(force_flag) \
        -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" -na

    # 4. the margin-aware search, in its own root (--max_wrong_loss is not part of the file name,
    #    so it would otherwise overwrite the plain run of the same cell). The root shares the
    #    checkpoints by symlink rather than retraining them.
    local mwl_root="$ROOT/runs/mwl"
    if (( DRY_RUN )); then
        echo "   (would link $mwl_root/model_outputs -> $path/model_outputs)"
    else
        mkdir -p "$mwl_root"
        ln -sfn "$(cd "$path" && pwd)/model_outputs" "$mwl_root/model_outputs"
    fi
    run "ablation_margin_aware" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" \
        ./run_attack_sweep.sh -n "$GENERATIONS" -p "$mwl_root" -b "$BLOCK_SIZE" \
        -msz "$MODEL_SIZE" -rdf "$rdf" -m none --vuln $(force_flag) \
        -- -r "$RESTARTS" -ns "$MWL_STEPS" -ve "$VERIFY_EVERY" -mwl "$MAX_WRONG_LOSS"
}

# Survival against temperature, and the cost curve in search steps. Both read an attack result
# file, so this phase follows `attack`. Run per generation that produced hits; generations with
# none are skipped by the sweep itself with a message.
phase_survival() {
    local record path rdf
    record="$(printf '%s\n' "${LINEAGES[@]}" | grep "^${PRIMARY}|")"
    [[ -n "$record" ]] || { echo "error: --primary $PRIMARY unknown" >&2; return; }
    IFS='|' read -r _n path rdf _t _p _k _s _desc <<< "$record"

    local specifier tag
    specifier="$("$PYTHON" -c "
from utils.models import resolve_model_specifier
print(resolve_model_specifier('$MODEL_SIZE', '').split('/')[-1])" 2>/dev/null)"
    tag="$("$PYTHON" -c "
from utils.naming import mixture_tag
print(mixture_tag($rdf))" 2>/dev/null)"

    for gen in $(seq 0 "$GENERATIONS"); do
        for suffix in "_T0.7p0.8k20x5_anchor" "_T0.7p0.8k20x5" ""; do
            local file="$path/attack_results/attack_gen${gen}_${specifier}${tag}_vuln${suffix}.json"
            # generation 0 carries no anchor tag, and a cell whose sweep found nothing has no file
            [[ -f "$file" ]] || continue
            local label="survival_gen${gen}${suffix:-_greedy_noanchor}"
            run "${label}_curve" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m utils.survival_sweep \
                -rf "$file" -msz "$MODEL_SIZE" -rdf "$rdf" -bs "$BLOCK_SIZE" -p "$path" \
                -T "$TEMPERATURES" -ns "$SURVIVAL_SAMPLES" --plot $(usetex_flag_py)
            run "${label}_cost" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m utils.survival_sweep \
                -rf "$file" -msz "$MODEL_SIZE" -rdf "$rdf" -bs "$BLOCK_SIZE" -p "$path" \
                --source trajectory --every "$TRAJECTORY_EVERY" --restarts 0 \
                -T 0.7 -ns "$SURVIVAL_SAMPLES" --plot $(usetex_flag_py)
        done
    done
}

# The repo's own evaluation pipeline: random/init controls, the 16-sample survival re-score and
# the suffix-perplexity filter. The attack phase already ran, so it is skipped here; the
# arguments after -- tell the script which verification tag its inputs carry.
phase_eval() {
    local mixtures=""
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf _t _p _k _s _desc <<< "$record"
        wants_lineage "$name" || continue
        [[ "$path" == "$ROOT" ]] || continue      # one invocation per root; variants below
        mixtures="$mixtures $rdf"
    done
    mixtures="${mixtures# }"

    if [[ -n "$mixtures" ]]; then
        run "eval_sampled" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" ./run_vuln_eval.sh \
            -n "$GENERATIONS" -p "$ROOT" -b "$BLOCK_SIZE" --models "$MODEL_SIZE" \
            --rdf "$mixtures" --phases controls,temperature,perplexity \
            --temperature 0.7 --num-samples "$SURVIVAL_SAMPLES" \
            $(force_flag) $(usetex_flag_sh)
        run "eval_greedy" env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" ./run_vuln_eval.sh \
            -n "$GENERATIONS" -p "$ROOT" -b "$BLOCK_SIZE" --models "$MODEL_SIZE" \
            --rdf "$mixtures" --phases temperature \
            --temperature 0.7 --num-samples "$SURVIVAL_SAMPLES" \
            $(force_flag) $(usetex_flag_sh) -- -vt 0 -na
    fi
}

# Every figure. Separated from the phases that produce their inputs so that a redraw costs no GPU
# time: `--phases plots --force` is the whole figure set from cached results.
phase_plots() {
    # the attack figures, once per hit rule, because runs under different rules are different
    # claims and must never be averaged into one panel
    for tag in "_T0.7p0.8k20x5_anchor" "_T0.7p0.8k20x5" ""; do
        for mode in none logit; do
            local label="plots_${mode}${tag:-_greedy_noanchor}"
            if ! has_results "$RESULTS_DIR" "$tag"; then
                skip "$label" "no results under hit rule '${tag:-greedy, no anchor}'"
                continue
            fi
            run "$label" "$PYTHON" run_vuln_plots.py -rp "$RESULTS_DIR" -pp "$PLOTS_DIR" \
                -bs "$BLOCK_SIZE" -m "$mode" -vf "$tag" $(usetex_flag_py)
        done
    done

    # margin and severity: margin/entropy against generation, and hit rate against measured
    # severity, which is the figure the collapse-versus-fine-tuning attribution rests on
    # the glob is expanded here, after the margin phase has written its files; with none on disk
    # it would stay literal and run_margin.py would fail on a path that does not exist
    local margin_files=( "$RESULTS_DIR"/margin_*.json )
    if [[ -e "${margin_files[0]}" ]]; then
        run "plots_margin" "$PYTHON" run_margin.py --plot "${margin_files[@]}" \
            -pp "$PLOTS_DIR" $(usetex_flag_py)
    else
        echo "   plots_margin skipped: no margin_*.json under $RESULTS_DIR (run --phases margin)"
    fi

    # the decoding-variant roots keep their own attack_results/ and plots/
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path _rdf _t _p _k _s _desc <<< "$record"
        wants_lineage "$name" || continue
        [[ "$path" != "$ROOT" ]] || continue
        if ! has_results "$path/attack_results" "_T0.7p0.8k20x5_anchor"; then
            skip "plots_${name}" "no attack results under $path yet"
            continue
        fi
        run "plots_${name}" "$PYTHON" run_vuln_plots.py -rp "$path/attack_results" \
            -pp "$path/plots" -bs "$BLOCK_SIZE" -m none -vf "_T0.7p0.8k20x5_anchor" \
            $(usetex_flag_py)
    done
}

# Cross-run transfer (stage 4): does a suffix found against one collapse run break an
# independently collapsed one. Trains its own second lineage, hence opt-in.
phase_transfer() {
    local record path rdf
    record="$(printf '%s\n' "${LINEAGES[@]}" | grep "^${PRIMARY}|")"
    [[ -n "$record" ]] || return
    IFS='|' read -r _n path rdf _t _p _k _s _desc <<< "$record"
    run "transfer_experiment" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" run_transfer_experiment.py \
        -cg "$GENERATIONS" -bs "$BLOCK_SIZE" -msz "$MODEL_SIZE" -rdf "$rdf" \
        -p "$path/runs/transfer"
}

# ─────────────────────────────────── the plan ─────────────────────────────────────────────
print_plan() {
    echo "############################################################"
    echo "## paper pipeline"
    echo "##   root         : $ROOT"
    echo "##   generations  : 0..$GENERATIONS  (run_baseline.py -ng $NUM_GENERATIONS)"
    echo "##   model        : $MODEL_SIZE, block size $BLOCK_SIZE"
    echo "##   phases       : $PHASES"
    echo "##   primary cell : $PRIMARY"
    echo "##   interpreter  : $PYTHON"
    echo "##   GPUs         : train $TRAIN_GPUS, attack $ATTACK_GPUS, eval $EVAL_GPU"
    echo "##   search       : $RESTARTS restarts x $NUM_STEPS steps, verify every $VERIFY_EVERY"
    echo "##   lineages     :"
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name path rdf temp top_p top_k seed desc <<< "$record"
        local mark="  "
        wants_lineage "$name" || mark="--"
        printf '##   %s %-10s rdf %-4s T %-4s top-p %-4s top-k %-3s seed %-5s %s\n' \
            "$mark" "$name" "$rdf" "$temp" "$top_p" "$top_k" "$seed" "$desc"
    done
    echo "##"
    echo "##   Order matters: utility writes the calibration the ablation's -sf calibrated rung"
    echo "##   reads, attack writes the result files survival and eval consume, and plots is last"
    echo "##   so it can be re-run alone. Expect days of wall clock for the full set."
    echo "############################################################"
}

# ──────────────────────────────────── dispatch ────────────────────────────────────────────
print_plan

if (( LIST_ONLY )); then
    exit 0
fi

if ! preflight; then
    echo "preflight failed; nothing was run." >&2
    exit 2
fi

if (( DRY_RUN == 0 && ASSUME_YES == 0 )); then
    read -r -p "Start? This can take days. [y/N] " reply
    [[ "$reply" == [yY]* ]] || { echo "aborted."; exit 0; }
fi

mkdir -p "$LOG_DIR" "$PLOTS_DIR"

for phase in collapse utility extrapolation margin attack ablation survival eval plots transfer; do
    has_phase "$phase" || continue
    echo
    echo "════════════════════════════════════════════════════════════"
    echo "══ phase: $phase"
    echo "════════════════════════════════════════════════════════════"
    "phase_${phase}"
done

# ──────────────────────────────────── summary ─────────────────────────────────────────────
TOTAL=$(( SECONDS - STARTED_AT ))
echo
echo "############################################################"
echo "## summary  (total $(printf '%02d:%02d:%02d' $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60))))"
echo "############################################################"
printf '## %-34s %-12s %s\n' "step" "status" "elapsed"
for i in "${!STEP_NAMES[@]}"; do
    secs=${STEP_SECONDS[$i]}
    printf '## %-34s %-12s %02d:%02d:%02d\n' "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}" \
        $((secs/3600)) $((secs%3600/60)) $((secs%60))
done
echo "############################################################"
if (( FAILURES )); then
    echo "## $FAILURES step(s) failed — see $LOG_DIR"
    echo "## Re-run this script with the same arguments to continue; completed work is skipped."
    exit 1
fi
echo "## all steps completed"
echo "## figures: $PLOTS_DIR"
echo "## results: $RESULTS_DIR"
exit 0
