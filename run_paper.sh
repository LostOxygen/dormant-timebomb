#!/usr/bin/env bash
#
# run_paper.sh — the paper's grid, in dependency order, resumable.
#
#   1. collapse   three lineages: Qwen2.5-Coder 0.5B at --real_data_fraction 0.0,
#                 0.5 and 0.8. Fresh initialisation every generation (run_baseline.py's default:
#                 only the data recurs), Qwen2.5's own decoding (T 0.7, top-p 0.8, top-k 20),
#                 seed 1337. One root holds all six, since the size and the mixture are in
#                 every artifact name and generation 0 is shared per size.
#   2. utility    every resulting model scored on held-out human code (perplexity, with the
#                 surrogate calibration the attack's -sf calibrated rung reads) and on HumanEval
#                 (executed correctness). Both scripts draw their figure; the collapse phase
#                 already drew the per-generation perplexity histogram of each lineage.
#   3. attack     every model attacked, correctness targets first and then the vulnerability
#                 targets (--vuln): direct against the real checkpoint and through the logit
#                 surrogate, under the sampled-majority verdict and the greedy one. The
#                 generation-0 anchor is held throughout (the sweeps' default).
#   4. plots      the attack figures, per size and hit rule, from the result files only.
#
# The commands are the ones documented in docs/sampling_experiments.md; this file adds the
# ordering, per-step logs, a status table and resumability (every script skips work whose output
# exists; --force overrides that). run_paper_full.sh is the previous superset with the ablation,
# survival, eval and transfer phases and the decoding/seed variants.
#
# Usage:
#   ./run_paper.sh --dry-run                  # print the whole plan, run nothing
#   ./run_paper.sh --list                     # one line per lineage and phase, then exit
#   ./run_paper.sh                            # run everything (asks once before starting)
#   ./run_paper.sh --phases collapse,utility  # a subset, in the order above
#   ./run_paper.sh --only 0.5b-rdf0.5         # one lineage
#   ./run_paper.sh --phases plots --force     # redraw every figure
#
# NOTE on the roots. The checkpoint names do not carry the weight lineage, so the fresh-init
# lineages and the recursive ones every result before 2026-10-15 was measured on live in separate
# roots: ./runs/fresh (this script's default, with its own model_outputs/, attack_results/, plots/
# and paper_logs/) and ./runs/recursive (the pre-2026-10-15 models and datasets, run_paper_full.sh's
# recursive cells). -p moves the former; run_baseline.py refuses to mix the two in one root.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2

# ───────────────────────────────── configuration ──────────────────────────────────────────
# The repo venv, not whatever `python` resolves to: a shell with another project's venv active
# picks up a trl/transformers pair this pipeline does not run on, and the failure surfaces deep
# inside a training worker hours later. The shell sweeps read $PYTHON too.
PYTHON="${PYTHON:-$SCRIPT_DIR/venv/bin/python}"
export PYTHON

ROOT="${ROOT:-./runs/fresh}"            # see the NOTE above
GENERATIONS="${GENERATIONS:-9}"         # highest generation index; the lineages train 0..N
BLOCK_SIZE="${BLOCK_SIZE:-512}"

# The grid. Space separated; every size is run at every mixture.
SIZES="${SIZES:-0.5b}"   # the 3B half is off by default; SIZES="0.5b 3b" turns it on
MIXTURES="${MIXTURES:-0.0 0.5 0.8}"
# The collapse decoding and seed, identical for every cell so that size and mixture are the only
# differences between them.
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-1337}"
# Extra run_baseline.py arguments per size, for memory: the 3B fits a 48GB card at the defaults,
# but -gc (gradient checkpointing) or a smaller -tbs are the knobs if it does not on yours.
TRAIN_OPTS_05B="${TRAIN_OPTS_05B:-}"
TRAIN_OPTS_3B="${TRAIN_OPTS_3B:-}"

# GPUs. Training and dataset generation fan out over several devices; the attack shards its
# (task, restart) units over whatever it can see; the single-model measurements want one card.
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
ATTACK_GPUS="${ATTACK_GPUS:-0,1,2,3}"
EVAL_GPU="${EVAL_GPU:-0}"

# Search budget, passed through to run_attack.py. One restart of 100 steps is the budget the
# sink-objective runs found their hits within; 3 x 250 is the earlier, four times costlier, one.
RESTARTS="${RESTARTS:-1}"
NUM_STEPS="${NUM_STEPS:-100}"
VERIFY_EVERY="${VERIFY_EVERY:-10}"
# Which attacks: surrogate methods (`none` is the direct white-box attack, `logit` the transfer
# attack through the extrapolated surrogate) and verdicts (`sampled` is the deployment decoding
# with a 5-sample majority, `greedy` the single argmax decode, -vt 0).
METHODS="${METHODS:-none logit}"
VERDICTS="${VERDICTS:-sampled greedy}"

# Surrogate variants. Every logit sweep runs twice, so that the extrapolation's recent
# optimizations are an axis of the comparison rather than a setting:
#   plain  the tilt as published — base + n (gen0 - base) at n = g + 1, mean-cross-entropy
#          objective. Files carry no extra tag
#   opt    the optimizations: the surrogate rescaled to the generation-0 anchor's entropy
#          (-em, files _em) and, for the vulnerability targets, the factor solved per target
#          from the affine law (-sf sink, files _nsink, capped at OPT_SINK_CAP) with the
#          objective evaluated at the security decision (-so, files _sinkobj). The correctness
#          targets get the entropy rescaling only: they have no security decision point, so the
#          sink factor is undefined there and the sink objective would optimize a single
#          operator of one reference program
# The direct sweeps have no surrogate, so they run once, as `plain`.
VARIANTS="${VARIANTS:-plain opt}"
OPT_ENTROPY_MATCH="${OPT_ENTROPY_MATCH:-1}"
OPT_SINK_FACTOR="${OPT_SINK_FACTOR:-1}"
OPT_SINK_CAP="${OPT_SINK_CAP:-2.5}"     # the entropy-matched surrogate's measured ceiling
OPT_SINK_OBJECTIVE="${OPT_SINK_OBJECTIVE:-1}"
OPT_SINK_MARGIN="${OPT_SINK_MARGIN:-2.0}"

# Search-objective knobs that apply to every variant. Neither reaches a file name, so turning one
# on and re-running a cell overwrites the cell it is being compared against: use another -p.
#   TARGET_MARGIN  nats/token the model under attack must prefer the wrong code by (0: off)
#   SUFFIX_TOKENS  initial suffix length (0: run_attack.py's default)
TARGET_MARGIN="${TARGET_MARGIN:-0}"
NU_TARGET="${NU_TARGET:-1.0}"
SUFFIX_TOKENS="${SUFFIX_TOKENS:-0}"

NO_USETEX="${NO_USETEX:-0}"             # 1 draws figures without a TeX install

PHASES_ALL="collapse,utility,attack,plots"
PHASES="${PHASES:-$PHASES_ALL}"

ONLY=""
FORCE=0
DRY_RUN=0
LIST_ONLY=0
ASSUME_YES=0

# ─────────────────────────────────── arguments ────────────────────────────────────────────
usage() {
    cat <<'EOF'
Collapses three lineages (0.5b at rdf 0.0, 0.5, 0.8; SIZES="0.5b 3b" adds the 3b ones), evaluates and plots their utility and
perplexity, then attacks every model: correctness targets first, then the vulnerability targets.

Options:
  -p, --path PATH        run root (default: ./runs/fresh — see the note in the script header)
  -n, --generations N    highest generation index; lineages train 0..N (default: 9)
      --phases LIST      comma separated subset of: collapse, utility, attack, plots
      --only NAME        restrict to one lineage, e.g. 0.5b-rdf0.5 (see --list)
      --force            redo work whose output already exists
      --dry-run          print every command without running it
      --list             print the plan and exit
      --no-usetex        draw figures without LaTeX
  -y, --yes              do not ask before starting
  -h, --help             this message

Environment overrides: PYTHON SIZES MIXTURES GENERATIONS BLOCK_SIZE TRAIN_GPUS ATTACK_GPUS
EVAL_GPU RESTARTS NUM_STEPS VERIFY_EVERY METHODS VERDICTS VARIANTS TRAIN_OPTS_05B
TRAIN_OPTS_3B TARGET_MARGIN NU_TARGET SUFFIX_TOKENS OPT_ENTROPY_MATCH OPT_SINK_FACTOR
OPT_SINK_CAP OPT_SINK_OBJECTIVE OPT_SINK_MARGIN NO_USETEX
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--path)        ROOT="$2";        shift 2 ;;
        -n|--generations) GENERATIONS="$2"; shift 2 ;;
        --phases)         PHASES="$2";      shift 2 ;;
        --only)           ONLY="$2";        shift 2 ;;
        --force)          FORCE=1;          shift ;;
        --dry-run)        DRY_RUN=1;        shift ;;
        --list)           LIST_ONLY=1;      shift ;;
        --no-usetex)      NO_USETEX=1;      shift ;;
        -y|--yes)         ASSUME_YES=1;     shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ── the lineages ──
# name | size | rdf, generated from the grid. Built after the argument loop so that -p and the
# environment overrides have landed.
LINEAGES=()
for size in $SIZES; do
    for rdf in $MIXTURES; do
        LINEAGES+=("${size}-rdf${rdf}|${size}|${rdf}")
    done
done

if [[ -n "$ONLY" ]] && ! printf '%s\n' "${LINEAGES[@]}" | grep -q "^${ONLY}|"; then
    echo "error: unknown lineage '${ONLY}'. Known:" >&2
    printf '%s\n' "${LINEAGES[@]}" | cut -d'|' -f1 | sed 's/^/  /' >&2
    exit 2
fi

NUM_GENERATIONS=$(( GENERATIONS + 1 ))   # run_baseline.py counts generations, the rest index them
LOG_DIR="$ROOT/paper_logs"
PLOTS_DIR="$ROOT/plots"
RESULTS_DIR="$ROOT/attack_results"

# ───────────────────────────────────── helpers ────────────────────────────────────────────
has_phase()     { [[ ",$PHASES," == *",$1,"* ]]; }
wants_lineage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
usetex_flag_py() { (( NO_USETEX )) && echo "--no_usetex" || true; }
force_flag()     { (( FORCE ))     && echo "--force"     || true; }

# The checkpoint names carry the model's short name, which is what the plotting scripts filter on.
specifier_name() {
    "$PYTHON" -c "
from utils.models import resolve_model_specifier
print(resolve_model_specifier('$1', '').split('/')[-1])" 2>/dev/null
}

train_opts() {
    case "$1" in
        0.5b) echo "$TRAIN_OPTS_05B" ;;
        3b)   echo "$TRAIN_OPTS_3B" ;;
        *)    true ;;
    esac
}

# The variants a method is swept under: a direct attack has no surrogate to optimize, so it runs
# once whatever VARIANTS says.
variants_for() {
    if [[ "$1" == "none" ]]; then
        [[ " $VARIANTS " == *" plain "* ]] && echo plain
    else
        echo $VARIANTS
    fi
}

# The search-objective arguments shared by every variant.
search_opts() {
    local opts=()
    [[ "$TARGET_MARGIN" != "0" ]] && opts+=(-tm "$TARGET_MARGIN" -nt "$NU_TARGET")
    [[ "$SUFFIX_TOKENS" != "0" ]] && opts+=(-stk "$SUFFIX_TOKENS")
    printf '%s ' "${opts[@]+"${opts[@]}"}"
}

# What `opt` adds, per (mode, method). Empty for `plain` and for every direct sweep, so a plain
# run issues exactly the command it would have issued before the variant axis existed.
variant_opts() {
    local mode="$1" method="$2" variant="$3" opts=()
    [[ "$variant" == "opt" && "$method" == "logit" ]] || return 0
    (( OPT_ENTROPY_MATCH )) && opts+=(-em)
    if [[ "$mode" == "vuln" ]]; then
        if (( OPT_SINK_FACTOR )); then
            opts+=(-sf sink)
            [[ "$OPT_SINK_CAP" != "0" ]] && opts+=(--sink_cap "$OPT_SINK_CAP")
        fi
        (( OPT_SINK_OBJECTIVE )) && opts+=(-so --sink_margin "$OPT_SINK_MARGIN")
    fi
    printf '%s ' "${opts[@]+"${opts[@]}"}"
}

# True when any sweep in scope needs a sink screen (the per-target factor reads one).
needs_sink_screen() {
    [[ " $(variants_for logit) " == *" opt "* ]] || return 1
    [[ " $METHODS " == *" logit "* ]] || return 1
    (( OPT_SINK_FACTOR ))
}

verdict_opts() { [[ "$1" == "greedy" ]] && echo "-vt 0" || true; }

# The hit-rule tag a verdict's files carry (utils.naming.run_tag): the sampled verification tag
# or nothing for greedy, then _anchor, then _sinkobj when the vulnerability sweeps used it.
run_tag_for() {
    local verdict="$1" mode="$2" variant="$3" sink=0
    [[ "$mode" == "vuln" && "$variant" == "opt" ]] && sink=$OPT_SINK_OBJECTIVE
    "$PYTHON" -c "
from utils.naming import run_tag
greedy = '$verdict' == 'greedy'
print(run_tag(temperature=0.0 if greedy else 0.7, hold_anchor=True, generation=1,
              sink_objective=$sink == 1))" 2>/dev/null
}

# True when `dir` holds at least one vulnerability attack result under the hit rule `tag`. Asked
# of the loader the figures use, rather than guessed from file names, so the two cannot disagree.
has_vuln_results() {
    local dir="$1" tag="$2" name="$3" em="$4"
    [[ -d "$dir" ]] || return 1
    "$PYTHON" - "$dir" "$tag" "$name" "$em" <<'HAS_RESULTS_PY' 2>/dev/null
import sys
from utils.vuln_results import load_records
want = {"any": None, "on": True, "off": False}[sys.argv[4]]
sys.exit(0 if load_records(sys.argv[1], specifier_name=sys.argv[3], run_tag=sys.argv[2],
                           entropy_match=want) else 1)
HAS_RESULTS_PY
}

# ───────────────────────────────── step bookkeeping ───────────────────────────────────────
STEP_NAMES=(); STEP_STATUS=(); STEP_SECONDS=()
FAILURES=0
STARTED_AT=$SECONDS

# a step that had nothing to do: recorded in the summary, but not a failure
skip() {
    echo
    echo "── $1 ── skipped: $2"
    STEP_NAMES+=("$1"); STEP_STATUS+=("skipped"); STEP_SECONDS+=(0)
}

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
        echo "warning: no LaTeX on PATH and --no-usetex was not given; the plot steps will fail." >&2
    fi

    for script in run_attack_sweep.sh run_baseline.py run_margin.py run_attack.py \
                  run_attack_vuln.py run_vuln_plots.py utils/evaluate_perplexity.py \
                  utils/evaluate_correctness.py utils/plot_attack_hits.py; do
        [[ -e "$script" ]] || { echo "error: missing $script" >&2; problems=1; }
    done

    # a fresh-init collapse must not land on a root that holds a recursive lineage. run_baseline.py
    # refuses the individual run, but it is better to know before the first hour of training
    if has_phase collapse && [[ -d "$ROOT/generated_datasets" ]]; then
        local clash
        clash="$(ls "$ROOT"/generated_datasets/all_perplexities_bs"${BLOCK_SIZE}"_*_recursive*.pt 2>/dev/null | head -1)"
        if [[ -n "$clash" ]]; then
            echo "error: $ROOT holds a recursive lineage ($(basename "$clash")); the collapse phase" >&2
            echo "       trains fresh-init lineages and would overwrite its checkpoints. Use another -p." >&2
            problems=1
        fi
    fi

    return $problems
}

# ───────────────────────────────────── phases ─────────────────────────────────────────────
# Each phase is a loop over the lineages. The underlying scripts own the "already done" checks,
# so a phase is cheap to re-enter.

# 1. The collapse lineages. Generation 0 is pure human data under every mixture, so the three
#    mixtures of one size share it: the first run trains it, the next two find it and start at 1.
phase_collapse() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        # shellcheck disable=SC2046
        run "collapse_${name}" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" run_baseline.py \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" \
            -tp "$TEMPERATURE" -tpp "$TOP_P" -tpk "$TOP_K" -sd "$SEED" -p "$ROOT" \
            $(train_opts "$size")
    done
}

# 2. Utility and perplexity of every model, with figures. evaluate_perplexity scores each
#    checkpoint on the human training slice and the held-out slice and, with --calibrate, fits the
#    surrogate factor the attack's -sf calibrated rung reads; evaluate_correctness executes each
#    checkpoint's HumanEval answers. Both draw their figure under $ROOT/plots.
phase_utility() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        run "perplexity_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m utils.evaluate_perplexity \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" -p "$ROOT" \
            --calibrate $(usetex_flag_py)
        run "correctness_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m utils.evaluate_correctness \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" -p "$ROOT" \
            $(usetex_flag_py)
    done
}

# 3. The attacks: every lineage under the correctness targets first, then every lineage under
#    the vulnerability targets. Per lineage and mode, one sweep per (method, variant, verdict),
#    which is the grid the comparison needs — mixture x extrapolation x optimization x verdict.
#    Every combination files under its own name, so none overwrites another:
#      direct, sampled      attack_gen{g}_{model}{_rdf}[_vuln]_T0.7p0.8k20x5_anchor.json
#      direct, greedy       ..._anchor.json                      (no verification tag)
#      logit plain          ..._anchor[_sinkobj]_logit_surrogate.json
#      logit opt            ..._anchor[_sinkobj][_nsink]_em_logit_surrogate.json
#    The anchor is held in all of them (the sweeps' default): a hit needs the target broken while
#    the pristine baseline and the generation-0 anchor keep writing the correct (secure) code.
phase_attack() {
    local mode record name size rdf method variant verdict vuln_flag

    # the per-target factor of the optimized vulnerability sweeps reads a sink screen of its
    # lineage. One per lineage, measured before any sweep so that every one of them finds it.
    if needs_sink_screen; then
        for record in "${LINEAGES[@]}"; do
            IFS='|' read -r name size rdf <<< "$record"
            wants_lineage "$name" || continue
            run "sink_screen_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" run_margin.py \
                -msz "$size" -rdf "$rdf" -n "$GENERATIONS" -bs "$BLOCK_SIZE" -p "$ROOT" --sink
        done
    fi

    for mode in correctness vuln; do
        vuln_flag=""
        [[ "$mode" == "vuln" ]] && vuln_flag="--vuln"
        for record in "${LINEAGES[@]}"; do
            IFS='|' read -r name size rdf <<< "$record"
            wants_lineage "$name" || continue
            for method in $METHODS; do
                for variant in $(variants_for "$method"); do
                    for verdict in $VERDICTS; do
                        # shellcheck disable=SC2046
                        run "attack_${mode}_${name}_${method}_${variant}_${verdict}" \
                            env CUDA_VISIBLE_DEVICES="$ATTACK_GPUS" ./run_attack_sweep.sh \
                            -n "$GENERATIONS" -p "$ROOT" -b "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" \
                            -m "$method" $vuln_flag $(force_flag) \
                            -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" \
                            $(verdict_opts "$verdict") $(search_opts) \
                            $(variant_opts "$mode" "$method" "$variant")
                    done
                done
            done
        done
    done
}

# 4. The attack figures, from the result files alone, so a redraw costs no GPU time
#    (`--phases plots --force`). The correctness hits are drawn per lineage, the vulnerability
#    figures per size and hit rule: runs under different rules are different claims and are never
#    averaged into one panel.
phase_plots() {
    local record name size rdf specifier variant verdict tag method em label
    # the correctness hits of one lineage: direct and surrogate curves in one panel, one figure
    # per variant since plain and optimized are two different surrogates for the same generation
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        for variant in $(variants_for logit); do
            em=""
            [[ "$variant" == "opt" ]] && (( OPT_ENTROPY_MATCH )) && em="-em"
            run "plots_correctness_${name}_${variant}" "$PYTHON" -m utils.plot_attack_hits \
                -p "$ROOT" -n "$GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" \
                -m "$(echo $METHODS | tr ' ' ',')" $em $(usetex_flag_py)
        done
    done

    # the vulnerability figures, one per (size, method, variant, verdict): runs under different
    # hit rules or different surrogates are different claims and are never averaged into a panel
    for size in $SIZES; do
        if [[ -n "$ONLY" && "$ONLY" != "${size}-"* ]]; then
            continue
        fi
        specifier="$(specifier_name "$size")"
        for method in $METHODS; do
            for variant in $(variants_for "$method"); do
                em="off"
                [[ "$variant" == "opt" && "$method" == "logit" ]] && (( OPT_ENTROPY_MATCH )) && em="on"
                for verdict in $VERDICTS; do
                    tag="$(run_tag_for "$verdict" vuln "$variant")"
                    label="plots_vuln_${size}_${method}_${variant}_${verdict}"
                    if ! has_vuln_results "$RESULTS_DIR" "$tag" "$specifier" "$em"; then
                        skip "$label" "no $size $variant results under hit rule '${tag:-greedy}'"
                        continue
                    fi
                    run "$label" "$PYTHON" run_vuln_plots.py -rp "$RESULTS_DIR" -pp "$PLOTS_DIR" \
                        -bs "$BLOCK_SIZE" -msn "$specifier" -m "$method" -vf "$tag" \
                        -em "$em" $(usetex_flag_py)
                done
            done
        done
    done
}

# ─────────────────────────────────── the plan ─────────────────────────────────────────────
print_plan() {
    echo "############################################################"
    echo "## paper pipeline"
    echo "##   root         : $ROOT"
    echo "##   generations  : 0..$GENERATIONS  (run_baseline.py -ng $NUM_GENERATIONS), block size $BLOCK_SIZE"
    echo "##   decoding     : T $TEMPERATURE, top-p $TOP_P, top-k $TOP_K, seed $SEED, fresh init every generation"
    echo "##   phases       : $PHASES"
    echo "##   interpreter  : $PYTHON"
    echo "##   GPUs         : train $TRAIN_GPUS, attack $ATTACK_GPUS, eval $EVAL_GPU"
    echo "##   search       : $RESTARTS restarts x $NUM_STEPS steps, verify every $VERIFY_EVERY"
    echo "##   attacks      : correctness then --vuln; methods [$METHODS], variants [$VARIANTS],"
    echo "##                  verdicts [$VERDICTS]"
    if [[ " $VARIANTS " == *" opt "* ]]; then
        echo "##   opt variant  : entropy match $OPT_ENTROPY_MATCH; vuln also sink factor" \
             "$OPT_SINK_FACTOR (cap $OPT_SINK_CAP), sink objective $OPT_SINK_OBJECTIVE" \
             "(margin $OPT_SINK_MARGIN)"
    fi
    if [[ "$TARGET_MARGIN" != "0" || "$SUFFIX_TOKENS" != "0" ]]; then
        echo "##   objective    : target margin $TARGET_MARGIN (weight $NU_TARGET), suffix" \
             "${SUFFIX_TOKENS} tokens — these change results but not file names"
    fi
    echo "##   lineages     :"
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        local mark="  "
        wants_lineage "$name" || mark="--"
        printf '##   %s %-12s size %-4s rdf %s\n' "$mark" "$name" "$size" "$rdf"
    done
    echo "##"
    echo "##   Order matters: utility writes the calibration the attack's -sf calibrated rung reads,"
    echo "##   attack writes the result files plots consumes. Expect days of wall clock."
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

for phase in collapse utility attack plots; do
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
printf '## %-44s %-12s %s\n' "step" "status" "elapsed"
for i in "${!STEP_NAMES[@]}"; do
    secs=${STEP_SECONDS[$i]}
    printf '## %-44s %-12s %02d:%02d:%02d\n' "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}" \
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
