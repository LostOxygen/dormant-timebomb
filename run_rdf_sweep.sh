#!/usr/bin/env bash
#
# Sweeps run_baseline.py over --real_data_fraction, collapsing one model for N generations at
# every value of the mixture.
#
# One collapse run per value, sequentially, each logged separately, followed by a summary table.
# Sequential is not a limitation to work around: a single run already fans its training ranks and
# its generation/perplexity workers across every visible GPU, so two runs at once would contend for
# the same cards and the same datasets cache.
#
# The default ladder is 0.0, 0.1, ... 0.9 — ten values, and it stops at 0.9 on purpose.
# run_baseline.py rejects --real_data_fraction 1.0 outright ("must be in [0, 1)"), because at 1.0
# every generation past 0 would train on the original human corpus alone and nothing would
# collapse. Pass --to to shorten the ladder or --step to coarsen it; a --to at or above 1.0 is
# refused here rather than ten runs in, when the last one would have died on its own validation.
#
# Why one --path is enough for the whole sweep
# --------------------------------------------
# The mixture is part of every artifact name from generation 1 onward (`_rdf{value}`), so ten
# mixtures coexist under one --path without overwriting each other. Generation 0 is the exception
# and deliberately so: it trains on the human corpus by definition, so model_0 and the corpus it
# generates are byte-identical for every value and are left unsuffixed to be shared — see
# utils/naming.py. Which means a plain sweep retrains an identical generation 0 ten times.
# --reuse-gen0 passes -cfg 1 to every run whose shared generation 0 is already on disk, so it is
# trained once and reused. The condition is that artifacts exist, not that this is not the first
# value: on an empty root the first run trains generation 0 and the rest continue from generation 1,
# while on a root that already holds one every run reuses it — including under --force, which
# re-runs values but is not a request to retrain the generation they share. Delete model_0 (or drop
# the flag) to rebuild it, which is what a change to the training hyperparameters needs.
#
# -ng must be at least 2. At -ng 1 the only generation is 0, which is the shared one — every value
# of the ladder would produce the same model and the sweep would measure nothing.
#
# Disk. Each generation writes a LoRA adapter and a merged fp16 copy, and the sweep writes one set
# per value: ten values times N generations. The banner estimates the total from an existing
# checkpoint of the same model when it finds one, and says so when the estimate does not fit in the
# free space under --path. At 7B that number is large enough to be worth reading before starting.
#
# Usage:
#   ./run_rdf_sweep.sh -n 10 [-p ./runs/rdf] [options] [-- extra run_baseline.py args]

set -uo pipefail

BLOCK_SIZE=512
MODEL_SPECIFIER=""
MODEL_SIZE=""
PATH_ROOT="."
PYTHON="${PYTHON:-python}"
NUM_GENERATIONS=""
FROM_FRACTION="0.0"
TO_FRACTION="0.9"
STEP_FRACTION="0.1"
REUSE_GEN0=0
FORCE=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Sweeps run_baseline.py over --real_data_fraction, collapsing a model for N generations per value.

Required:
  -n, --num-generations N   generations per collapse run (-ng); 0..N-1, so N must be >= 2

Options:
  -p, --path PATH           root the runs write model_outputs/ and generated_datasets/ into
                            (default: .). One root holds the whole sweep: the mixture is part of
                            every artifact name from generation 1 on
  -b, --block-size N        block size for every run (default: 512). Must be the same across the
                            sweep — it is part of the artifact names
  -ms, --model-specifier S  the model to collapse
  -msz, --model-size SIZE   parameter count off the Qwen2.5-Coder ladder (0.5b, 1.5b, 3b, 7b,
                            14b, 32b), shorthand for --model-specifier
      --from F              first fraction (default: 0.0)
      --to T                last fraction, inclusive (default: 0.9). Must be below 1.0, which
                            run_baseline.py refuses: nothing would collapse
      --step S              ladder step (default: 0.1)
      --reuse-gen0          train the shared, mixture-independent generation 0 once and continue
                            from generation 1 (-cfg 1) whenever it is already on disk; skipped
                            automatically while it is not. An existing generation 0 is reused
                            as-is, --force included, so delete it to rebuild it
      --force               re-run values whose final checkpoint already exists
      --dry-run             print the commands and the ladder without running anything
  -h, --help                this message

Everything after -- is passed through to run_baseline.py unchanged, e.g.:
  ./run_rdf_sweep.sh -n 10 -msz 0.5b -- -tbs 8 -gas 8 -e transformers

  ./run_rdf_sweep.sh -n 10 -msz 0.5b -p ./runs/rdf              # 0.0 .. 0.9
  ./run_rdf_sweep.sh -n 10 -msz 0.5b --from 0.5 --step 0.05      # a finer ladder over half of it
  ./run_rdf_sweep.sh -n 10 -msz 0.5b --reuse-gen0                # share generation 0 across values
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        -p|--path)            PATH_ROOT="$2";       shift 2 ;;
        -b|--block-size)      BLOCK_SIZE="$2";      shift 2 ;;
        -ms|--model-specifier) MODEL_SPECIFIER="$2"; shift 2 ;;
        -msz|--model-size)    MODEL_SIZE="$2";      shift 2 ;;
        --from)               FROM_FRACTION="$2";   shift 2 ;;
        --to)                 TO_FRACTION="$2";     shift 2 ;;
        --step)               STEP_FRACTION="$2";   shift 2 ;;
        --reuse-gen0)         REUSE_GEN0=1;         shift ;;
        --force)              FORCE=1;              shift ;;
        --dry-run)            DRY_RUN=1;            shift ;;
        -h|--help)            usage; exit 0 ;;
        --)                   shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$NUM_GENERATIONS" ]]; then
    echo "error: -n/--num-generations is required" >&2
    usage >&2
    exit 2
fi
if ! [[ "$NUM_GENERATIONS" =~ ^[0-9]+$ ]]; then
    echo "error: -n/--num-generations must be a non-negative integer" >&2
    exit 2
fi
# generation 0 is the mixture-independent one, so a one-generation run is the same run at every
# value of the ladder — see utils/naming.py's mixture_suffix
if (( NUM_GENERATIONS < 2 )); then
    echo "error: -n $NUM_GENERATIONS leaves only generation 0, which trains on the human corpus" \
         "under every --real_data_fraction and is shared between them. There would be nothing" \
         "for the sweep to vary; use -n 2 or more." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$SCRIPT_DIR/run_baseline.py"
if [[ ! -f "$BASELINE" ]]; then
    echo "error: run_baseline.py not found next to this script ($BASELINE)" >&2
    exit 2
fi
(( DRY_RUN )) || mkdir -p "$PATH_ROOT" || exit 2

LAST_GENERATION=$(( NUM_GENERATIONS - 1 ))

# The model is resolved through the same helper run_baseline.py uses rather than a size table
# repeated in bash: the artifact directories are named after the trailing component of the result,
# so a table that drifted from the python one would check for checkpoints under names no run ever
# writes. The helper exits non-zero with its own message on an unknown size or on the two flags
# disagreeing.
if ! RESOLVED="$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -c \
        'import sys
from utils.models import model_size_label, resolve_model_specifier
specifier = resolve_model_specifier(sys.argv[1], sys.argv[2])
print(specifier, model_size_label(specifier), sep="\n")' \
        "$MODEL_SIZE" "$MODEL_SPECIFIER")"; then
    exit 2
fi
MODEL_SIZE=""
{ read -r MODEL_SPECIFIER; read -r MODEL_SIZE; } <<< "$RESOLVED"
SPECIFIER_NAME="${MODEL_SPECIFIER##*/}"

# The ladder comes from python too, and for two reasons beyond taste. Bash has no float arithmetic
# and `seq` formats according to the locale, so a decimal-comma locale would hand run_baseline.py
# "0,3"; and the *string* matters — mixture_tag formats the fraction with :g, so the value passed
# on the command line has to stringify the same way or the run would write artifacts this script
# then fails to find. Stepping over integer tenths rather than accumulating a float keeps 0.3 from
# arriving as 0.30000000000000004.
#
# Three colon-separated fields per line: the value, the run-level tag, and the per-generation
# suffix of the *last* generation, which is what says whether this value has already been swept.
# Colon rather than whitespace because an empty field has to survive `read` — at 0.0 both suffixes
# are empty, and IFS whitespace would collapse them away.
if ! LADDER="$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -c \
        'import sys
from utils.naming import mixture_suffix, mixture_tag

start, stop, step = (float(a) for a in sys.argv[1:4])
last_generation = int(sys.argv[4])
if step <= 0:
    raise SystemExit("error: --step must be positive")
if start < 0:
    raise SystemExit("error: --from must be at least 0")
if stop < start:
    raise SystemExit("error: --to is below --from, nothing to sweep")
if stop >= 1.0:
    raise SystemExit(
        "error: --to must be below 1.0. run_baseline.py rejects --real_data_fraction 1.0 "
        "(must be in [0, 1)): at 1.0 every generation trains on the original human corpus "
        "alone and nothing collapses."
    )
scale = 10 ** 6
value, limit, increment = round(start * scale), round(stop * scale), round(step * scale)
while value <= limit:
    fraction = value / scale
    print(f"{fraction:g}", mixture_tag(fraction),
          mixture_suffix(fraction, last_generation), sep=":")
    value += increment' \
        "$FROM_FRACTION" "$TO_FRACTION" "$STEP_FRACTION" "$LAST_GENERATION")"; then
    exit 2
fi

FRACTIONS=()
TAGS=()
SUFFIXES=()
while IFS=: read -r value tag suffix; do
    [[ -n "$value" ]] || continue
    FRACTIONS+=("$value")
    TAGS+=("$tag")
    SUFFIXES+=("$suffix")
done <<< "$LADDER"
if (( ${#FRACTIONS[@]} == 0 )); then
    echo "error: the ladder is empty" >&2
    exit 2
fi

MODEL_DIR="$PATH_ROOT/model_outputs"
LOG_DIR="$PATH_ROOT/collapse_logs"
(( DRY_RUN )) || mkdir -p "$LOG_DIR"

# name of the last checkpoint a completed run writes, for the already-done check. The merged fp16
# copy rather than the adapter: utils/train_generation.py writes it immediately after the adapter,
# and it is what run_attack.py loads later, so its absence means the sweep has nothing usable for
# that value whatever else is on disk
final_checkpoint() {
    echo "$MODEL_DIR/model_${LAST_GENERATION}_bs${BLOCK_SIZE}_${SPECIFIER_NAME}${1}_fp16"
}

# how far a value got, counted the way the names are built: generation 0 is the shared unsuffixed
# one, every later generation carries the mixture suffix
completed_generations() {
    local suffix="$1" gen done_count=0
    for (( gen = 0; gen <= LAST_GENERATION; gen++ )); do
        local this_suffix="$suffix"
        (( gen == 0 )) && this_suffix=""
        [[ -d "$MODEL_DIR/model_${gen}_bs${BLOCK_SIZE}_${SPECIFIER_NAME}${this_suffix}_fp16" ]] \
            && done_count=$(( done_count + 1 ))
    done
    echo "$done_count"
}

# the shared generation 0, and what --reuse-gen0 needs before it can skip training it: the
# checkpoint a later generation trains from, and the corpus generation 1 reads
GEN0_CHECKPOINT="$MODEL_DIR/model_0_bs${BLOCK_SIZE}_${SPECIFIER_NAME}"
GEN0_CORPUS="$PATH_ROOT/generated_datasets/generated_dataset_0_bs${BLOCK_SIZE}_${SPECIFIER_NAME}"

# ──────────────────────────── disk estimate ────────────────────────────
# measured off a checkpoint of this model that already exists, because the size is a property of
# the model and a guess from the parameter count would be worse than saying nothing
# only the values still to be run: with four of ten mixtures already on disk the whole-ladder
# figure is an overestimate large enough to be misleading
PENDING=0
for (( i = 0; i < ${#FRACTIONS[@]}; i++ )); do
    if (( FORCE )) || [[ ! -d "$(final_checkpoint "${SUFFIXES[$i]}")" ]]; then
        PENDING=$(( PENDING + 1 ))
    fi
done

DISK_NOTE=""
sample_checkpoint="$(find "$MODEL_DIR" -maxdepth 1 -type d \
    -name "model_*_bs${BLOCK_SIZE}_${SPECIFIER_NAME}*_fp16" 2>/dev/null | head -1)"
if [[ -n "$sample_checkpoint" ]]; then
    per_generation_kb="$(du -sk "$sample_checkpoint" 2>/dev/null | cut -f1)"
    free_kb="$(df -Pk "$PATH_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ -n "$per_generation_kb" && -n "$free_kb" ]]; then
        # the adapter beside each merged copy is a few percent on top; 110% is close enough for a
        # number whose only job is to be read before a sweep that runs for hours
        estimate_kb=$(( per_generation_kb * NUM_GENERATIONS * PENDING * 110 / 100 ))
        DISK_NOTE="$(( estimate_kb / 1048576 )) GiB estimated for the $PENDING value(s) still to"
        DISK_NOTE="$DISK_NOTE run, $(( free_kb / 1048576 )) GiB free"
        if (( estimate_kb > free_kb )); then
            DISK_NOTE="$DISK_NOTE  --  WILL NOT FIT"
        fi
    fi
fi

echo "############################################################"
echo "## real-data-fraction sweep: ${#FRACTIONS[@]} value(s), $NUM_GENERATIONS generations each"
echo "##   fractions    : ${FRACTIONS[*]}"
echo "##   block size   : $BLOCK_SIZE"
echo "##   model        : $MODEL_SPECIFIER"
echo "##   model size   : ${MODEL_SIZE:-outside the --model_size ladder}"
echo "##   path         : $PATH_ROOT"
echo "##   logs         : $LOG_DIR"
if (( REUSE_GEN0 )); then
    echo "##   generation 0 : trained once, later values continue from generation 1 (-cfg 1)"
else
    echo "##   generation 0 : retrained per value (--reuse-gen0 shares the one it produces)"
fi
if [[ -n "$DISK_NOTE" ]]; then
    echo "##   disk         : $DISK_NOTE"
fi
if (( ${#EXTRA_ARGS[@]} )); then
    echo "##   extra args   : ${EXTRA_ARGS[*]}"
fi
echo "############################################################"
if [[ "$DISK_NOTE" == *"WILL NOT FIT"* ]]; then
    echo "## warning: the checkpoints this sweep writes are estimated not to fit under $PATH_ROOT."
    echo "##   Shorten the ladder (--to/--step), lower -n, or free space before starting — a run"
    echo "##   that fills the disk fails part way through a generation and leaves it unusable."
fi

STATUS_FRACTIONS=()
STATUS_CODES=()
STATUS_SUFFIXES=()
FAILURES=0
STARTED_AT=$SECONDS

for (( i = 0; i < ${#FRACTIONS[@]}; i++ )); do
    fraction="${FRACTIONS[$i]}"
    tag="${TAGS[$i]}"
    suffix="${SUFFIXES[$i]}"
    checkpoint="$(final_checkpoint "$suffix")"
    log_file="$LOG_DIR/collapse_ng${NUM_GENERATIONS}_bs${BLOCK_SIZE}_${SPECIFIER_NAME}${tag}.log"

    echo
    echo "== real_data_fraction $fraction =="

    if [[ -d "$checkpoint" ]] && (( FORCE == 0 )); then
        echo "   already done: $checkpoint (pass --force to re-run)"
        STATUS_FRACTIONS+=("$fraction")
        STATUS_CODES+=("cached")
        STATUS_SUFFIXES+=("$suffix")
        continue
    fi

    cmd=("$PYTHON" "$BASELINE"
         -ng "$NUM_GENERATIONS"
         -bs "$BLOCK_SIZE"
         -ms "$MODEL_SPECIFIER"
         -rdf "$fraction"
         -p "$PATH_ROOT")
    # -cfg 1 is only correct once the shared generation 0 is actually on disk: it makes
    # run_baseline.py skip that generation entirely, and generation 1 then reads a corpus that has
    # to already exist. Checked per value rather than assumed from "not the first one", so a first
    # run that failed half way does not turn the rest of the sweep into runs continuing from
    # nothing
    if (( REUSE_GEN0 )); then
        if [[ -d "$GEN0_CHECKPOINT" && -d "$GEN0_CORPUS" ]]; then
            cmd+=(-cfg 1)
            echo "   reusing the shared generation 0 (-cfg 1)"
        else
            echo "   generation 0 not on disk yet — training it in this run"
        fi
    fi
    if (( ${#EXTRA_ARGS[@]} )); then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo "   \$ ${cmd[*]}"
    if (( DRY_RUN )); then
        echo "   would write: $checkpoint"
        STATUS_FRACTIONS+=("$fraction")
        STATUS_CODES+=("dry-run")
        STATUS_SUFFIXES+=("$suffix")
        continue
    fi

    echo "   log: $log_file"
    value_started=$SECONDS
    # tee so the run stays watchable while still leaving a complete per-value log
    "${cmd[@]}" 2>&1 | tee "$log_file"
    code=${PIPESTATUS[0]}
    elapsed=$(( SECONDS - value_started ))

    if (( code == 0 )); then
        echo "   done in ${elapsed}s"
        STATUS_CODES+=("ok")
    else
        echo "   FAILED with exit code $code after ${elapsed}s — see $log_file"
        STATUS_CODES+=("exit $code")
        FAILURES=$(( FAILURES + 1 ))
    fi
    STATUS_FRACTIONS+=("$fraction")
    STATUS_SUFFIXES+=("$suffix")
done

# ──────────────────────────────── summary ────────────────────────────────
echo
echo "############################################################"
echo "## sweep summary  (total $(( SECONDS - STARTED_AT ))s)"
echo "############################################################"
printf '## %-6s %-10s %-12s %s\n' rdf run generations final-checkpoint

for (( i = 0; i < ${#STATUS_FRACTIONS[@]}; i++ )); do
    fraction="${STATUS_FRACTIONS[$i]}"
    code="${STATUS_CODES[$i]}"
    suffix="${STATUS_SUFFIXES[$i]}"
    checkpoint="$(final_checkpoint "$suffix")"
    generations="$(completed_generations "$suffix")/$NUM_GENERATIONS"
    note="$checkpoint"

    if [[ "$code" == "dry-run" ]]; then
        generations="-"
        note="(not run)"
    elif [[ ! -d "$checkpoint" ]]; then
        # a partial run leaves the earlier generations behind, which the count above reports
        note="missing — see the log"
    fi

    printf '## %-6s %-10s %-12s %s\n' "$fraction" "$code" "$generations" "$note"
done

echo "############################################################"
if (( FAILURES > 0 )); then
    echo "## $FAILURES value(s) failed — see the logs in $LOG_DIR"
    echo "## a value that got part way can be resumed with run_baseline.py -cfg <generation> -rdf <value>"
    exit 1
fi
echo "## all $NUM_GENERATIONS-generation collapse runs completed"
