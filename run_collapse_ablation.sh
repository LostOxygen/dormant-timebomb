#!/usr/bin/env bash
#
# run_collapse_ablation.sh — the collapse ablation of a finished run_paper.sh grid.
#
# Two measurements, no new search, then the figures (run_collapse_ablation.py):
#
#   survival   every verified hit of an attack family, re-verified under the family's own
#              verdict against every generation of its own lineage and of the control lineage
#              (SCORE_MIXTURES). Says which hits are collapse-specific (re-verify on the generation
#              they were found on but spare the control at the same generation, which had the same
#              training rounds without collapse), which are merely new since the first fine-tune
#              (spare generation 0 of their lineage), how broad a hit is, and whether it carries
#              to the control. SCORE_MIXTURES=all also scores the control's hits on the collapsed
#              lineages, the reverse direction of the specificity figure. One job per
#              (size, lineage, family); resumable per checkpoint.
#   geometry   the lineage against the logit surrogate's ray, at every target's security
#              decision: cosine of (l_g - l_base) to (l_0 - l_base), the least-squares position
#              n_hat on the ray, the KL to the ray and to the surrogate the attack used, and the
#              decision margin against the affine law. One job per (size, lineage), minutes each.
#   plots      every figure of the study under $PLOTS_DIR/ablation_* plus ablation_collapse.md.
#
# Needs the grid's attack results and checkpoints under ROOT; run_paper.sh's defaults are this
# script's defaults, so after `./run_paper.sh` a bare `./run_collapse_ablation.sh` is the study
# of that grid (SIZES="0.5b 3b" for both halves). The jobs are independent, so they are queued
# and drained over GPUS like run_paper.sh's pool, JOBS_PER_GPU at a time per card; one survival
# job of the 0.5b sampled logit-opt family is an hour or two, the 3b several.
#
# Usage:
#   ./run_collapse_ablation.sh --dry-run                  # print the plan, run nothing
#   ./run_collapse_ablation.sh                            # measure and draw (asks once)
#   SIZES="0.5b 3b" ./run_collapse_ablation.sh -y         # both sizes, no question
#   ./run_collapse_ablation.sh --phases plots --force     # redraw from the files
#   ./run_collapse_ablation.sh --only 0.5b-rdf0.5         # one lineage's measurements
#   FAMILIES="logit_opt_sampled" LIMIT=5 ./run_collapse_ablation.sh --phases survival   # smoke

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2

# ───────────────────────────────── configuration ──────────────────────────────────────────
PYTHON="${PYTHON:-$SCRIPT_DIR/venv/bin/python}"
export PYTHON

ROOT="${ROOT:-./runs/fresh}"
GENERATIONS="${GENERATIONS:-9}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
SIZES="${SIZES:-0.5b}"
MIXTURES="${MIXTURES:-0.0 0.5 0.8 1.0}"
# the attack families whose hits are re-verified: {none|logit}_{plain|opt}_{sampled|greedy}.
# The two the paper reads are the defaults; the geometry figure is read against the first one
FAMILIES="${FAMILIES:-none_plain_sampled logit_opt_sampled}"
# which lineages every hit is scored against: "own control" (its own and rdf 1.0, the default)
# or "all" (every lineage in MIXTURES, the full transfer matrix at four times the cost)
SCORE_MIXTURES="${SCORE_MIXTURES:-own control}"
GPUS="${GPUS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
# the hit rule and generation-0 rule of the grid, and the optimized variant's knobs: these spell
# the result file names the hits are harvested from, so they must match run_paper.sh's
HOLD_ANCHOR="${HOLD_ANCHOR:-0}"
HUMAN_GEN0="${HUMAN_GEN0:-0}"
OPT_ENTROPY_MATCH="${OPT_ENTROPY_MATCH:-1}"
OPT_SINK_FACTOR="${OPT_SINK_FACTOR:-1}"
OPT_SINK_OBJECTIVE="${OPT_SINK_OBJECTIVE:-1}"
OPT_SINK_CAP="${OPT_SINK_CAP:-2.5}"
POSITIONS="${POSITIONS:-32}"        # geometry: code positions before the decision also measured
LIMIT="${LIMIT:-0}"                 # survival: score only the first N pairs per job (smoke tests)
NO_USETEX="${NO_USETEX:-0}"

PHASES_ALL="survival,geometry,plots"
PHASES="${PHASES:-$PHASES_ALL}"

ONLY=""
FORCE=0
DRY_RUN=0
LIST_ONLY=0
ASSUME_YES=0

usage() {
    cat <<'EOF'
Measures the collapse ablation of a finished grid (hit survival across the generations and the
lineage's geometry against the surrogate's ray) and draws its figures.

Options:
  -p, --path PATH        run root (default: ./runs/fresh)
  -n, --generations N    highest generation index (default: 9)
      --phases LIST      comma separated subset of: survival, geometry, plots
      --only NAME        restrict the measurements to one lineage, e.g. 0.5b-rdf0.5
      --force            redo work whose output already exists
      --dry-run          print every command without running it
      --list             print the plan and exit
      --no-usetex        draw figures without LaTeX
  -y, --yes              do not ask before starting
  -h, --help             this message

Environment overrides: PYTHON ROOT SIZES MIXTURES FAMILIES SCORE_MIXTURES GPUS JOBS_PER_GPU
GENERATIONS BLOCK_SIZE HOLD_ANCHOR HUMAN_GEN0 OPT_ENTROPY_MATCH OPT_SINK_FACTOR
OPT_SINK_OBJECTIVE OPT_SINK_CAP POSITIONS LIMIT NO_USETEX LOG_DIR PLOTS_DIR
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

LOG_DIR="${LOG_DIR:-$ROOT/paper_logs}"
PLOTS_DIR="${PLOTS_DIR:-$ROOT/plots}"
RESULTS_DIR="$ROOT/attack_results"

# ───────────────────────────────────── helpers ────────────────────────────────────────────
has_phase()      { [[ ",$PHASES," == *",$1,"* ]]; }
wants_lineage()  { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
usetex_flag_py() { (( NO_USETEX )) && echo "--no_usetex" || true; }
force_flag()     { (( FORCE ))     && echo "--force"     || true; }
anchor_opts()    { (( HOLD_ANCHOR )) && echo "-ha" || true; }
gen0_opts()      { (( HUMAN_GEN0 )) && echo "-hg0" || true; }
opt_flags() {
    (( OPT_ENTROPY_MATCH ))  || echo --no_opt_entropy_match
    (( OPT_SINK_FACTOR ))    || echo --no_opt_sink_factor
    (( OPT_SINK_OBJECTIVE )) || echo --no_opt_sink_objective
}
# <rdf>: the mixtures a lineage's hits are scored against
score_mixtures_for() {
    if [[ "$SCORE_MIXTURES" == "all" ]]; then
        echo "$MIXTURES"
    elif [[ "$SCORE_MIXTURES" == "own control" ]]; then
        echo "$1 1.0"
    else
        echo "$SCORE_MIXTURES"
    fi
}

STEP_NAMES=(); STEP_STATUS=(); STEP_SECONDS=()
FAILURES=0
STARTED_AT=$SECONDS

skip() {
    echo
    echo "── $1 ── skipped: $2"
    STEP_NAMES+=("$1"); STEP_STATUS+=("skipped"); STEP_SECONDS+=(0)
}

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

# ───────────────────────────────────── job pool ───────────────────────────────────────────
# run_paper.sh's pool: independent jobs queued with `queue <name> <command...>` and drained
# over the GPUs, one worker loop per GPU slot pulling the next job off a cursor under flock,
# each job pinned through CUDA_VISIBLE_DEVICES with its output in $LOG_DIR/<name>.log
QUEUE_NAMES=(); QUEUE_CMDS=()
queue() {
    local name="$1"; shift
    QUEUE_NAMES+=("$name")
    QUEUE_CMDS+=("$(printf '%q ' "$@")")
}

pool_worker() {
    local gpu="$1" jobs="$2" cursor="$3" status="$4" line name quoted started code
    while :; do
        line="$(flock "$cursor" bash -c '
            next=$(<"$1"); total=$(wc -l < "$2")
            (( next < total )) || exit 1
            echo $(( next + 1 )) > "$1"
            sed -n "$(( next + 1 ))p" "$2"' _ "$cursor" "$jobs")" || break
        name="${line%%$'\t'*}"
        quoted="${line#*$'\t'}"
        eval "local -a cmd=($quoted)"
        started=$SECONDS
        echo "── $name ── started on GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$LOG_DIR/$name.log" 2>&1
        code=$?
        flock "$status" printf '%s\t%s\t%s\n' "$name" "$code" "$(( SECONDS - started ))" >> "$status"
        if (( code == 0 )); then
            echo "   $name: done in $(( (SECONDS - started) / 60 )) min"
        else
            echo "   $name: FAILED ($code) — see $LOG_DIR/$name.log"
        fi
    done
}

drain() {
    local gpus="$1" per_gpu="${2:-1}" n=${#QUEUE_NAMES[@]} i
    (( n == 0 )) && return 0
    if (( DRY_RUN )); then
        for (( i = 0; i < n; i++ )); do
            eval "local -a cmd=(${QUEUE_CMDS[i]})"
            echo
            echo "── ${QUEUE_NAMES[i]} ── (pool over GPU(s) ${gpus// /,}, $per_gpu per GPU)"
            echo "   \$ ${cmd[*]}"
            STEP_NAMES+=("${QUEUE_NAMES[i]}"); STEP_STATUS+=("dry-run"); STEP_SECONDS+=(0)
        done
        QUEUE_NAMES=(); QUEUE_CMDS=()
        return 0
    fi
    mkdir -p "$LOG_DIR"
    local dir jobs cursor status gpu slot
    dir="$(mktemp -d "$LOG_DIR/.pool.XXXXXX")"
    jobs="$dir/jobs"; cursor="$dir/cursor"; status="$dir/status"
    for (( i = 0; i < n; i++ )); do
        printf '%s\t%s\n' "${QUEUE_NAMES[i]}" "${QUEUE_CMDS[i]}" >> "$jobs"
    done
    echo 0 > "$cursor"
    : > "$status"
    echo
    echo "── pool: $n job(s) over GPU(s) ${gpus// /,}, $per_gpu per GPU; logs in $LOG_DIR ──"
    for gpu in $gpus; do
        for (( slot = 0; slot < per_gpu; slot++ )); do
            pool_worker "$gpu" "$jobs" "$cursor" "$status" &
        done
    done
    wait
    local -A seen_code seen_secs
    local name code secs
    while IFS=$'\t' read -r name code secs; do
        seen_code["$name"]="$code"; seen_secs["$name"]="$secs"
    done < "$status"
    for (( i = 0; i < n; i++ )); do
        name="${QUEUE_NAMES[i]}"
        code="${seen_code[$name]:-missing}"
        STEP_NAMES+=("$name"); STEP_SECONDS+=("${seen_secs[$name]:-0}")
        if [[ "$code" == "0" ]]; then
            STEP_STATUS+=("ok")
        else
            STEP_STATUS+=("FAILED ($code)")
            FAILURES=$(( FAILURES + 1 ))
        fi
    done
    rm -rf "$dir"
    QUEUE_NAMES=(); QUEUE_CMDS=()
}

# ───────────────────────────────────── phases ─────────────────────────────────────────────
# 1. Hit survival: one job per (lineage, family). The script resumes a partial file checkpoint
#    by checkpoint and re-reads a complete one in seconds, so a re-entered phase costs only the
#    missing cells; --force starts every file over.
phase_survival() {
    local record name size rdf family
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        for family in $FAMILIES; do
            # shellcheck disable=SC2046
            queue "survival_${name}_${family}" "$PYTHON" run_collapse_ablation.py \
                --mode survival -msz "$size" -rdf "$rdf" -n "$GENERATIONS" -bs "$BLOCK_SIZE" \
                -p "$ROOT" --family "$family" --score_mixtures $(score_mixtures_for "$rdf") \
                --sink_cap "$OPT_SINK_CAP" --limit "$LIMIT" $(anchor_opts) $(gen0_opts) \
                $(opt_flags) $(force_flag)
        done
    done
    drain "${GPUS//,/ }" "$JOBS_PER_GPU"
}

# 2. The ray geometry: one job per lineage, every target's decision, base plus N+1 checkpoints.
phase_geometry() {
    local record name size rdf
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        # shellcheck disable=SC2046
        queue "geometry_${name}" "$PYTHON" run_collapse_ablation.py \
            --mode geometry -msz "$size" -rdf "$rdf" -n "$GENERATIONS" -bs "$BLOCK_SIZE" \
            -p "$ROOT" --positions "$POSITIONS" --sink_cap "$OPT_SINK_CAP" \
            $(anchor_opts) $(gen0_opts) $(opt_flags) $(force_flag)
    done
    drain "${GPUS//,/ }" "$JOBS_PER_GPU"
}

# 3. The figures, from the files alone (no GPU): every size at once so the cross-size panels
#    are drawn when SIZES lists more than one.
phase_plots() {
    # shellcheck disable=SC2046
    run "plots_ablation" "$PYTHON" run_collapse_ablation.py --mode plot \
        --sizes $SIZES --mixtures $MIXTURES --families $FAMILIES -p "$ROOT" -pp "$PLOTS_DIR" \
        --sink_cap "$OPT_SINK_CAP" $(anchor_opts) $(gen0_opts) $(opt_flags) $(usetex_flag_py)
}

# ─────────────────────────────────── the plan ─────────────────────────────────────────────
print_plan() {
    echo "############################################################"
    echo "## collapse ablation"
    echo "##   root         : $ROOT"
    echo "##   generations  : 0..$GENERATIONS, block size $BLOCK_SIZE"
    echo "##   mixtures     : $MIXTURES$( [[ " $MIXTURES " == *" 1.0 "* ]] && echo '  (1.0 = non-collapsing control)' )"
    echo "##   families     : $FAMILIES"
    echo "##   scored on    : $SCORE_MIXTURES"
    echo "##   phases       : $PHASES"
    echo "##   interpreter  : $PYTHON"
    echo "##   GPUs         : $GPUS ($JOBS_PER_GPU job(s) per GPU)"
    echo "##   file names   : anchor $HOLD_ANCHOR, human gen 0 $HUMAN_GEN0, entropy match" \
         "$OPT_ENTROPY_MATCH, sink factor $OPT_SINK_FACTOR (cap $OPT_SINK_CAP), sink objective" \
         "$OPT_SINK_OBJECTIVE"
    (( LIMIT )) && echo "##   LIMIT        : $LIMIT pair(s) per survival job (smoke test)"
    echo "##   lineages     :"
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        local mark="  "
        wants_lineage "$name" || mark="--"
        printf '##   %s %-12s size %-4s rdf %s\n' "$mark" "$name" "$size" "$rdf"
    done
    echo "##   outputs      : $RESULTS_DIR/survival_matrix_*.json, geometry_*.json;"
    echo "##                  $PLOTS_DIR/ablation_*, ablation_collapse.md"
    echo "############################################################"
}

print_plan
if (( LIST_ONLY )); then
    exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "error: no interpreter at $PYTHON. Set PYTHON=... or create the venv." >&2
    exit 2
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "error: no attack results under $RESULTS_DIR; run run_paper.sh first." >&2
    exit 2
fi
if (( DRY_RUN == 0 && ASSUME_YES == 0 )); then
    read -r -p "Start? The survival jobs take hours. [y/N] " reply
    [[ "$reply" == [yY]* ]] || { echo "aborted."; exit 0; }
fi
mkdir -p "$LOG_DIR" "$PLOTS_DIR"

for phase in survival geometry plots; do
    has_phase "$phase" || continue
    echo
    echo "════════════════════════════════════════════════════════════"
    echo "══ phase: $phase"
    echo "════════════════════════════════════════════════════════════"
    "phase_${phase}"
done

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
    exit 1
fi
echo "## all steps completed"
echo "## figures: $PLOTS_DIR"
exit 0
