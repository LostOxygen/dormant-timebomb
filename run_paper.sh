#!/usr/bin/env bash
#
# run_paper.sh — the paper's grid, in dependency order, resumable.
#
#   1. collapse   four lineages: Qwen2.5-Coder 0.5B at --real_data_fraction 0.0, 0.5, 0.8
#                 and 1.0. The last is the NON-COLLAPSING CONTROL: every generation trains on
#                 human data alone, with the generation count, seed, corpus size and optimizer
#                 steps of a collapse run, so an attack that works there as well is a property of
#                 iterated fine-tuning rather than of collapse — the comparison the collapsed
#                 lineages' hit rates are read against. Fresh initialisation every generation (run_baseline.py's default:
#                 only the data recurs), Qwen2.5's own decoding (T 0.7, top-p 0.8, top-k 20),
#                 seed 1337. Generation 0 trains on the mixture like every later generation,
#                 with the base model's own samples standing in for the previous generation's
#                 corpus (the attacker's one fine-tune is one step of the map the lineage
#                 iterates; the control's generation 0 is human only, since its mixture takes no
#                 synthetic rows). One root holds every lineage, since the size and the mixture
#                 are in every artifact name; only the base corpus is shared per size.
#   2. utility    every resulting model scored on held-out human code (perplexity, with the
#                 surrogate calibration the attack's -sf calibrated rung reads) and on HumanEval
#                 (executed correctness). Both scripts draw their figure; the collapse phase
#                 already drew the per-generation perplexity histogram of each lineage.
#   3. attack     every model attacked, correctness targets first and then the vulnerability
#                 targets (--vuln): direct against the real checkpoint and through the logit
#                 surrogate, under the sampled-majority verdict and the greedy one. The
#                 hit rule is the threat model's: target wrong, pristine baseline correct.
#                 HOLD_ANCHOR=1 adds the generation-0 model to the hold set as a stricter claim.
#   4. plots      the attack figures, per size and hit rule, from the result files only.
#
# The commands are the ones documented in docs/sampling_experiments.md; this file adds the
# ordering, per-step logs, a status table and resumability (every script skips work whose output
# exists; --force overrides that). run_paper_full.sh is the previous superset with the ablation,
# survival, eval and transfer phases and the decoding/seed variants.
#
# Parallelism (since 2026-10-28). The collapse lineages run one after another on TRAIN_GPUS, as
# they must (every generation uses every card). Everything after them is a pool: the utility
# passes and the sink screens are dealt over EVAL_GPUS, and the attack phase is one queue of
# (sweep, generation) jobs over ATTACK_GPUS, ATTACK_JOBS_PER_GPU at a time per card, longest
# jobs first, each search kept in-process on its card. With VERDICTS listing both, one search
# per cell writes the sampled and the greedy file (--also_greedy; MERGE_VERDICTS=0 restores the
# twin sweeps). Every job's log is paper_logs/<step>_gen<N>.log and the summary folds the
# generations of a sweep into one row. Measured on the 0.5b grid, the serial version spent
# 101 of its 137 hours in the attack phase with half of the search steps replaying the other
# verdict's trajectory and one card busy while three idled during the utility phase.
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
MIXTURES="${MIXTURES:-0.0 0.5 0.8 1.0}"   # 1.0 is the non-collapsing control (tagged _rdf1)
# The collapse decoding and seed, identical for every cell so that size and mixture are the only
# differences between them.
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
# the generation repetition penalty: 1.2 is the repo default, Qwen2.5 ships 1.05, 1.0 disables
# it. A value other than the default tags the run-level names _rp<value> and needs its own ROOT,
# since the checkpoint names do not carry it (run_baseline.py refuses the mix)
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
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
# The job pool (since 2026-10-28). The utility passes, the sink screens and the attack sweeps
# are independent of one another, so they are queued and drained over these GPUs, one worker
# loop per GPU slot, each job pinned through CUDA_VISIBLE_DEVICES. ATTACK_JOBS_PER_GPU is the
# number of attack jobs one card runs at once: 1 is right for the 3b (one job holds every
# model of its cell, ~30GB) and safe everywhere; 2 fits the 0.5b (~10GB steady, but a
# startup transient of up to ~34GB, so stagger or accept the odd OOM backoff).
EVAL_GPUS="${EVAL_GPUS:-$ATTACK_GPUS}"
ATTACK_JOBS_PER_GPU="${ATTACK_JOBS_PER_GPU:-1}"
# One search per cell, both verdicts read off it (--also_greedy): the greedy and the sampled
# sweep of a cell run the identical GCG trajectory and differ only in how the checks decode,
# so the pair costs one search plus one extra decode per check. 0 restores one sweep per
# verdict. Applies only when VERDICTS lists both.
MERGE_VERDICTS="${MERGE_VERDICTS:-1}"

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
# The hit rule. 0 is the threat model's: the suffix must break the target while the pristine
# baseline keeps writing the correct code. The generation-0 model is the attacker's own fine-tune,
# the instrument the surrogate is built from, so it is not something the suffix has to spare.
# 1 also holds it correct (-ha, files tagged _anchor) — the stricter claim that the suffix is
# specific to *later* collapse rather than to any descendant of the base model.
HOLD_ANCHOR="${HOLD_ANCHOR:-0}"

# The generation-0 data rule. 0 is the default: generation 0 trains on the same mixture as every
# later generation, the base model's own samples standing in for the previous generation's corpus.
# 1 is the ablation (-hg0, run-level artifacts tagged _hg0): generation 0 trains on the human
# corpus alone and is shared across mixtures, the rule every run before 2026-10-22 used. The
# checkpoint names do not carry the rule, so an ablation grid needs its own root (-p).
HUMAN_GEN0="${HUMAN_GEN0:-0}"

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
Collapses four lineages (0.5b at rdf 0.0, 0.5, 0.8 and the non-collapsing control 1.0;
SIZES="0.5b 3b" adds the 3b ones), evaluates and plots their utility and
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
EVAL_GPU EVAL_GPUS ATTACK_JOBS_PER_GPU MERGE_VERDICTS RESTARTS NUM_STEPS VERIFY_EVERY METHODS
VERDICTS VARIANTS HOLD_ANCHOR TRAIN_OPTS_05B
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
# both overridable, so that a second figure set (PLOT_THEME=dark for slides) can be drawn into
# its own directory without touching the paper's figures or the run's logs
LOG_DIR="${LOG_DIR:-$ROOT/paper_logs}"
PLOTS_DIR="${PLOTS_DIR:-$ROOT/plots}"
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

# <size> <rdf>: the sink screen run_margin.py --sink writes for one lineage of $ROOT
margin_file() {
    "$PYTHON" -c "
from utils.models import resolve_model_specifier
from utils.naming import mixture_tag
print('$ROOT/attack_results/margin_'
      + resolve_model_specifier('$1', '').split('/')[-1] + mixture_tag(float('$2')) + '.json')" \
        2>/dev/null
}

# <size> <rdf>: the file run_distribution_collapse.py writes for one lineage of $ROOT
distribution_file() {
    "$PYTHON" -c "
from utils.models import resolve_model_specifier
from utils.naming import mixture_tag
print('$ROOT/attack_results/distribution_collapse_'
      + resolve_model_specifier('$1', '').split('/')[-1] + mixture_tag(float('$2')) + '.json')" \
        2>/dev/null
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

# greedy: -vt 0; both: the sampled decoding plus --also_greedy; sampled: the defaults
verdict_opts() {
    case "$1" in
        greedy) echo "-vt 0" ;;
        both)   echo "--also_greedy" ;;
        *)      true ;;
    esac
}

# The hit-rule tag a verdict's files carry (utils.naming.run_tag): the sampled verification tag
# or nothing for greedy, then _anchor, then _sinkobj when the vulnerability sweeps used it.
run_tag_for() {
    local verdict="$1" mode="$2" variant="$3" sink=0
    [[ "$mode" == "vuln" && "$variant" == "opt" ]] && sink=$OPT_SINK_OBJECTIVE
    "$PYTHON" -c "
from utils.naming import run_tag
greedy = '$verdict' == 'greedy'
print(run_tag(temperature=0.0 if greedy else 0.7, hold_anchor=$HOLD_ANCHOR == 1, generation=1,
              sink_objective=$sink == 1))" 2>/dev/null
}

anchor_opts() { (( HOLD_ANCHOR )) && echo "-ha" || true; }
# every script that resolves a checkpoint by generation needs the generation-0 rule
gen0_opts() { (( HUMAN_GEN0 )) && echo "-hg0" || true; }

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

# ───────────────────────────────────── job pool ───────────────────────────────────────────
# Steps that do not depend on one another are queued with `queue <name> <command...>` and
# drained with `drain "<gpus>" <jobs per gpu>`: one worker loop per GPU slot pulls the next
# job off a cursor under flock, runs it pinned to its GPU through CUDA_VISIBLE_DEVICES with its
# output in paper_logs/<name>.log, and appends the outcome to a status file the parent folds
# into the summary once the pool is empty. Jobs named <step>_gen<N> are summarized per step.
# The underlying scripts keep their own "already done" checks, so a drained queue is as cheap
# to re-enter as the serial loops were.
QUEUE_NAMES=(); QUEUE_CMDS=()
queue() {
    local name="$1"; shift
    QUEUE_NAMES+=("$name")
    QUEUE_CMDS+=("$(printf '%q ' "$@")")
}

pool_worker() {
    local gpu="$1" jobs="$2" cursor="$3" status="$4" line name quoted started code
    while :; do
        # the cursor is the only shared state: advance it under the lock, read the job it named
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

    # fold the outcomes into the summary: one row per step, a step's generations added up
    local -A seen_code seen_secs
    local name code secs
    while IFS=$'\t' read -r name code secs; do
        seen_code["$name"]="$code"; seen_secs["$name"]="$secs"
    done < "$status"
    local -A step_fail step_secs step_total
    local -a step_order=()
    local step
    for (( i = 0; i < n; i++ )); do
        name="${QUEUE_NAMES[i]}"
        step="${name%_gen[0-9]*}"
        if [[ -z "${step_total[$step]+x}" ]]; then
            step_order+=("$step"); step_total[$step]=0; step_fail[$step]=0; step_secs[$step]=0
        fi
        step_total[$step]=$(( step_total[$step] + 1 ))
        code="${seen_code[$name]:-missing}"
        if [[ "$code" != "0" ]]; then
            step_fail[$step]=$(( step_fail[$step] + 1 ))
        fi
        step_secs[$step]=$(( step_secs[$step] + ${seen_secs[$name]:-0} ))
    done
    for step in "${step_order[@]}"; do
        STEP_NAMES+=("$step"); STEP_SECONDS+=("${step_secs[$step]}")
        if (( step_fail[$step] == 0 )); then
            STEP_STATUS+=("ok")
        else
            STEP_STATUS+=("FAILED (${step_fail[$step]}/${step_total[$step]})")
            FAILURES=$(( FAILURES + 1 ))
        fi
    done
    rm -rf "$dir"
    QUEUE_NAMES=(); QUEUE_CMDS=()
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
                  run_attack_vuln.py run_vuln_plots.py run_paper_tables.py utils/evaluate_perplexity.py \
                  utils/evaluate_correctness.py utils/plot_attack_hits.py \
                  run_distribution_collapse.py; do
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

# 1. The collapse lineages. Every mixture has its own generation 0 (trained on the mixture, with
#    the base model's corpus as its input); the base corpus itself is sampled by the first run of a
#    size and reused by the rest. The rdf 1.0 control needs none: its generation 0 is human only.
phase_collapse() {
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        # shellcheck disable=SC2046
        run "collapse_${name}" env CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" run_baseline.py \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" \
            -tp "$TEMPERATURE" -tpp "$TOP_P" -tpk "$TOP_K" -rp "$REPETITION_PENALTY" -sd "$SEED" \
            -p "$ROOT" \
            $(gen0_opts) $(train_opts "$size")
    done
}

# 2. Utility and perplexity of every model, with figures. evaluate_perplexity scores each
#    checkpoint on the human training slice and the held-out slice and, with --calibrate, fits the
#    surrogate factor the attack's -sf calibrated rung reads; evaluate_correctness executes each
#    checkpoint's HumanEval answers; run_distribution_collapse measures the token distribution
#    itself — each checkpoint's predictive entropy on held-out rows and on its own samples, and
#    the n-gram statistics of every corpus the lineage wrote. All three draw their figure under
#    $ROOT/plots.
phase_utility() {
    local record name size rdf
    # every lineage's two passes are independent of every other's, and each needs one GPU, so
    # they are pooled over EVAL_GPUS instead of taking EVAL_GPU in turn
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        # shellcheck disable=SC2046
        queue "perplexity_${name}" "$PYTHON" -m utils.evaluate_perplexity \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" -p "$ROOT" \
            -pp "$PLOTS_DIR" --calibrate $(gen0_opts) $(usetex_flag_py)
        # shellcheck disable=SC2046
        queue "correctness_${name}" "$PYTHON" -m utils.evaluate_correctness \
            -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" -p "$ROOT" \
            -pp "$PLOTS_DIR" $(gen0_opts) $(usetex_flag_py)
        # shellcheck disable=SC2046
        queue "distribution_${name}" "$PYTHON" run_distribution_collapse.py \
            -n "$GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" -p "$ROOT" \
            -pp "$PLOTS_DIR" --resume $(gen0_opts) $(usetex_flag_py)
    done
    drain "${EVAL_GPUS//,/ }" 1
}

# 3. The attacks: every lineage under the correctness targets first, then every lineage under
#    the vulnerability targets. Per lineage and mode, one sweep per (method, variant, verdict),
#    which is the grid the comparison needs — mixture x extrapolation x optimization x verdict.
#    Every combination files under its own name, so none overwrites another:
#      direct, sampled      attack_gen{g}_{model}{_rdf}[_vuln]_T0.7p0.8k20x5.json
#      direct, greedy       attack_gen{g}_{model}{_rdf}[_vuln].json
#      logit plain          ...[_sinkobj]_logit_surrogate.json
#      logit opt            ...[_sinkobj][_nsink]_em_logit_surrogate.json
#    A hit needs the target broken while the pristine baseline keeps writing the correct (secure)
#    code. HOLD_ANCHOR=1 adds the generation-0 model to that rule and puts _anchor in every name.
phase_attack() {
    local mode record name size rdf method variant verdict vuln_flag gen label
    local -a verdicts

    # the per-target factor of the optimized vulnerability sweeps reads a sink screen of its
    # lineage. One per lineage, measured before any sweep so that every one of them finds it;
    # independent across lineages, so pooled. A screen already on disk is kept unless --force,
    # so that re-entering the phase after a failed sweep (the 2026-11-03 DNS outage took 92 of
    # the 3b jobs) costs only the missing cells
    if needs_sink_screen; then
        for record in "${LINEAGES[@]}"; do
            IFS='|' read -r name size rdf <<< "$record"
            wants_lineage "$name" || continue
            if (( FORCE == 0 )) && [[ -f "$(margin_file "$size" "$rdf")" ]]; then
                skip "sink_screen_${name}" "screen exists: $(margin_file "$size" "$rdf")"
                continue
            fi
            # shellcheck disable=SC2046
            queue "sink_screen_${name}" "$PYTHON" run_margin.py \
                -msz "$size" -rdf "$rdf" -n "$GENERATIONS" -bs "$BLOCK_SIZE" -p "$ROOT" --sink \
                $(gen0_opts)
        done
        drain "${EVAL_GPUS//,/ }" 1
    fi

    # the verdict axis: one sweep per verdict, or — when both are wanted and MERGE_VERDICTS is
    # on — one sweep that writes both files (--also_greedy, see run_attack.py)
    verdicts=($VERDICTS)
    if (( MERGE_VERDICTS )) && [[ " $VERDICTS " == *" sampled "* && " $VERDICTS " == *" greedy "* ]]; then
        verdicts=(both)
    fi

    # every (sweep, generation) is one job: the sweep script run for that generation alone,
    # the search kept in-process (-ag 1) on the GPU the pool pins it to. The jobs are queued
    # longest first — vulnerability before correctness, the optimized surrogate before the plain
    # one before the direct attack, the lineages in the order given — so the tail of the pool
    # is made of the short ones. Generation 0 has no surrogate attack, so it is not queued for
    # the logit sweeps rather than queued and skipped
    for mode in vuln correctness; do
        vuln_flag=""
        [[ "$mode" == "vuln" ]] && vuln_flag="--vuln"
        for method in logit none; do
            [[ " $METHODS " == *" $method "* ]] || continue
            for variant in opt plain; do
                [[ " $(variants_for "$method") " == *" $variant "* ]] || continue
                for verdict in "${verdicts[@]}"; do
                    for record in "${LINEAGES[@]}"; do
                        IFS='|' read -r name size rdf <<< "$record"
                        wants_lineage "$name" || continue
                        label="attack_${mode}_${name}_${method}_${variant}_${verdict}"
                        for (( gen = 0; gen <= GENERATIONS; gen++ )); do
                            if (( gen == 0 )) && [[ "$method" != "none" ]]; then
                                continue
                            fi
                            # shellcheck disable=SC2046
                            queue "${label}_gen${gen}" ./run_attack_sweep.sh \
                                -s "$gen" -n "$gen" -p "$ROOT" -b "$BLOCK_SIZE" -msz "$size" \
                                -rdf "$rdf" -m "$method" $vuln_flag $(force_flag) \
                                -- -r "$RESTARTS" -ns "$NUM_STEPS" -ve "$VERIFY_EVERY" -ag 1 \
                                $(verdict_opts "$verdict") $(anchor_opts) $(gen0_opts) \
                                $(search_opts) $(variant_opts "$mode" "$method" "$variant")
                        done
                    done
                done
            done
        done
    done
    drain "${ATTACK_GPUS//,/ }" "$ATTACK_JOBS_PER_GPU"
}

# 4. The attack figures, from the result files alone, so a redraw costs no GPU time
#    (`--phases plots --force`). The correctness hits are drawn per lineage, the vulnerability
#    figures per size and hit rule: runs under different rules are different claims and are never
#    averaged into one panel.
phase_plots() {
    local record name size rdf specifier variant verdict tag method em label
    # the utility figures first, redrawn from the cached evaluations (-po) so that
    # `--phases plots` is the whole figure set and not only the attack half. Both scripts import
    # unsloth at module scope, so they need a visible GPU even to replot
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        run "plots_perplexity_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" \
            -m utils.evaluate_perplexity -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" \
            -msz "$size" -rdf "$rdf" -p "$ROOT" -pp "$PLOTS_DIR" --calibrate -po \
            $(gen0_opts) $(usetex_flag_py)
        run "plots_humaneval_${name}" env CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" \
            -m utils.evaluate_correctness -ng "$NUM_GENERATIONS" -bs "$BLOCK_SIZE" \
            -msz "$size" -rdf "$rdf" -p "$ROOT" -pp "$PLOTS_DIR" -po $(gen0_opts) \
            $(usetex_flag_py)
        # the distribution figure needs no GPU to replot, but it is skipped (not failed) when
        # the utility phase never produced its file
        if [[ -f "$(distribution_file "$size" "$rdf")" ]]; then
            run "plots_distribution_${name}" "$PYTHON" run_distribution_collapse.py \
                -n "$GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" -p "$ROOT" \
                -pp "$PLOTS_DIR" -po $(gen0_opts) $(usetex_flag_py)
        else
            skip "plots_distribution_${name}" "no distribution measurement under $ROOT yet"
        fi
    done

    # the correctness hits of one lineage: direct and surrogate curves in one panel, one figure
    # per variant since plain and optimized are two different surrogates for the same generation
    for record in "${LINEAGES[@]}"; do
        IFS='|' read -r name size rdf <<< "$record"
        wants_lineage "$name" || continue
        for variant in $(variants_for logit); do
            em=""
            [[ "$variant" == "opt" ]] && (( OPT_ENTROPY_MATCH )) && em="-em"
            for verdict in $VERDICTS; do
                # the hit rule is part of the result file names, so the figure has to be told
                # which family to read; correctness sweeps never carry the sink objective
                tag="$(run_tag_for "$verdict" correctness "$variant")"
                run "plots_correctness_${name}_${variant}_${verdict}" \
                    "$PYTHON" -m utils.plot_attack_hits \
                    -p "$ROOT" -n "$GENERATIONS" -bs "$BLOCK_SIZE" -msz "$size" -rdf "$rdf" \
                    -m "$(echo $METHODS | tr ' ' ',')" -vf "$tag" -pp "$PLOTS_DIR" \
                    $em $(usetex_flag_py)
            done
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
        # the result tables of the write-up as figures and markdown, one set per size: utility,
        # calibration, the hits/attackable grids, surrogate quality and the sink screen. Reads
        # files only, so it is part of every `--phases plots`
        # shellcheck disable=SC2046
        run "plots_tables_${size}" "$PYTHON" run_paper_tables.py -p "$ROOT" -pp "$PLOTS_DIR" \
            -msz "$size" --mixtures $MIXTURES -n "$GENERATIONS" -bs "$BLOCK_SIZE" \
            --methods $METHODS --variants $VARIANTS --verdicts $VERDICTS \
            --sink_cap "$OPT_SINK_CAP" $(anchor_opts) $(gen0_opts) $(usetex_flag_py) \
            $( (( OPT_ENTROPY_MATCH )) || echo --no_opt_entropy_match ) \
            $( (( OPT_SINK_FACTOR )) || echo --no_opt_sink_factor ) \
            $( (( OPT_SINK_OBJECTIVE )) || echo --no_opt_sink_objective )
    done
    # the sizes next to each other (since 2026-11-04): the hit-rate heatmaps per size and the
    # pooled rates per lineage with one bar per size, one figure per suite, verdict and method
    # under tables_sizes_*. Same flags as the per-size tables, so the cells are the same files
    if (( $(wc -w <<< "$SIZES") > 1 )) && [[ -z "$ONLY" ]]; then
        # shellcheck disable=SC2046
        run "plots_tables_sizes" "$PYTHON" run_paper_tables.py -p "$ROOT" -pp "$PLOTS_DIR" \
            --compare_sizes $SIZES --mixtures $MIXTURES -n "$GENERATIONS" -bs "$BLOCK_SIZE" \
            --methods $METHODS --variants $VARIANTS --verdicts $VERDICTS \
            --sink_cap "$OPT_SINK_CAP" $(anchor_opts) $(gen0_opts) $(usetex_flag_py) \
            $( (( OPT_ENTROPY_MATCH )) || echo --no_opt_entropy_match ) \
            $( (( OPT_SINK_FACTOR )) || echo --no_opt_sink_factor ) \
            $( (( OPT_SINK_OBJECTIVE )) || echo --no_opt_sink_objective )
    fi
}

# ─────────────────────────────────── the plan ─────────────────────────────────────────────
print_plan() {
    echo "############################################################"
    echo "## paper pipeline"
    echo "##   root         : $ROOT"
    echo "##   generations  : 0..$GENERATIONS  (run_baseline.py -ng $NUM_GENERATIONS), block size $BLOCK_SIZE"
    echo "##   decoding     : T $TEMPERATURE, top-p $TOP_P, top-k $TOP_K, repetition penalty $REPETITION_PENALTY, seed $SEED, fresh init every generation"
    echo "##   mixtures     : $MIXTURES$( [[ " $MIXTURES " == *" 1.0 "* ]] && echo '  (1.0 = non-collapsing control, human data every generation)' )"
    echo "##   phases       : $PHASES"
    echo "##   interpreter  : $PYTHON"
    echo "##   GPUs         : train $TRAIN_GPUS, attack $ATTACK_GPUS ($ATTACK_JOBS_PER_GPU job(s) per GPU),"
    echo "##                  utility/sink screens $EVAL_GPUS, plots $EVAL_GPU"
    echo "##   search       : $RESTARTS restarts x $NUM_STEPS steps, verify every $VERIFY_EVERY"
    echo "##   attacks      : --vuln and correctness; methods [$METHODS], variants [$VARIANTS],"
    echo "##                  verdicts [$VERDICTS]$( (( MERGE_VERDICTS )) && [[ " $VERDICTS " == *" sampled "* && " $VERDICTS " == *" greedy "* ]] && echo ", one search per cell writing both files (MERGE_VERDICTS=1)" )"
    echo "##                  queued per (sweep, generation) over the attack GPUs, longest first"
    echo "##   hit rule     : target wrong, baseline correct" \
         "$( (( HOLD_ANCHOR )) && echo ', generation-0 anchor correct (HOLD_ANCHOR=1)' )"
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
