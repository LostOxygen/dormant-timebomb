#!/usr/bin/env bash
#
# Sweeps run_attack.py over collapse generations 0..N with the logit surrogate.
#
# Each generation is attacked in transfer mode: the suffix is optimized against a surrogate built
# from the base model and the generation-0 checkpoint alone, then validated against the real
# checkpoint of the generation being swept. One run_attack.py invocation per generation, each
# logged separately, followed by a summary table.
#
# Generation 0 is skipped by design, not by accident: in transfer mode the surrogate is built
# *from* generation 0, so attacking generation 0 would validate a suffix against the very
# checkpoint it was derived from. run_attack.py rejects that combination outright. Pass
# --start 1 to drop it from the sweep silently, or --direct-gen0 to attack it without a
# surrogate instead.
#
# The capability gate stops individual generations that have collapsed past the point of writing
# correct code at all. That is an expected outcome, not a failure: the sweep records it and keeps
# going. Only an unexpected non-zero exit marks a generation as failed and the sweep as a whole
# as unsuccessful.
#
# Usage:
#   ./run_attack_sweep.sh -n 9 [-p ./runs/baseline] [options] [-- extra run_attack.py args]

set -uo pipefail

BLOCK_SIZE=512
SURROGATE_METHOD="logit"
MODEL_SPECIFIER="unsloth/Qwen2.5-Coder-0.5B-Instruct"
PATH_ROOT="."
PYTHON="${PYTHON:-python}"
NUM_GENERATIONS=""
START_GENERATION=0
FORCE=0
DRY_RUN=0
DIRECT_GEN0=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Sweeps run_attack.py over collapse generations with the logit surrogate.

Required:
  -n, --num-generations N   highest generation index to attack (sweeps 0..N inclusive)

Options:
  -p, --path PATH           root holding model_outputs/ and attack_results/ (default: .)
  -b, --block-size N        effective block size in the checkpoint names (default: 512)
  -s, --start G             first generation to attack (default: 0)
  -m, --method METHOD       surrogate method: logit or lora (default: logit)
  -ms, --model-specifier S  baseline model specifier
      --direct-gen0         also attack generation 0, without a surrogate, instead of skipping it
      --force               re-run generations whose result file already exists
      --dry-run             print the commands without running them
  -h, --help                this message

Everything after -- is passed through to run_attack.py unchanged, e.g.:
  ./run_attack_sweep.sh -n 9 -p ./runs/baseline -- -r 5 -ns 500 -sos
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-generations) NUM_GENERATIONS="$2"; shift 2 ;;
        -p|--path)            PATH_ROOT="$2";       shift 2 ;;
        -b|--block-size)      BLOCK_SIZE="$2";      shift 2 ;;
        -s|--start)           START_GENERATION="$2"; shift 2 ;;
        -m|--method)          SURROGATE_METHOD="$2"; shift 2 ;;
        -ms|--model-specifier) MODEL_SPECIFIER="$2"; shift 2 ;;
        --direct-gen0)        DIRECT_GEN0=1;        shift ;;
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
if ! [[ "$NUM_GENERATIONS" =~ ^[0-9]+$ && "$START_GENERATION" =~ ^[0-9]+$ ]]; then
    echo "error: generation indices must be non-negative integers" >&2
    exit 2
fi
if (( START_GENERATION > NUM_GENERATIONS )); then
    echo "error: --start $START_GENERATION is above -n $NUM_GENERATIONS, nothing to sweep" >&2
    exit 2
fi
if [[ "$SURROGATE_METHOD" != "logit" && "$SURROGATE_METHOD" != "lora" ]]; then
    echo "error: --method must be logit or lora ('data' is not a valid attack surrogate)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK="$SCRIPT_DIR/run_attack.py"
if [[ ! -f "$ATTACK" ]]; then
    echo "error: run_attack.py not found next to this script ($ATTACK)" >&2
    exit 2
fi
if [[ ! -d "$PATH_ROOT/model_outputs" ]]; then
    echo "error: no model_outputs/ under $PATH_ROOT — run run_baseline.py first" >&2
    exit 2
fi

SPECIFIER_NAME="${MODEL_SPECIFIER##*/}"
RESULTS_DIR="$PATH_ROOT/attack_results"
LOG_DIR="$RESULTS_DIR/sweep_logs"
mkdir -p "$LOG_DIR"

echo "############################################################"
echo "## attack sweep: generations $START_GENERATION..$NUM_GENERATIONS"
echo "##   surrogate    : $SURROGATE_METHOD"
echo "##   block size   : $BLOCK_SIZE"
echo "##   model        : $MODEL_SPECIFIER"
echo "##   path         : $PATH_ROOT"
echo "##   logs         : $LOG_DIR"
if (( ${#EXTRA_ARGS[@]} )); then
    echo "##   extra args   : ${EXTRA_ARGS[*]}"
fi
echo "############################################################"

STATUS_GENS=()
STATUS_CODES=()
STATUS_FILES=()
FAILURES=0
STARTED_AT=$SECONDS

for (( gen = START_GENERATION; gen <= NUM_GENERATIONS; gen++ )); do
    # generation 0 is the surrogate's own anchor, so there is no transfer attack to run on it
    if (( gen == 0 )) && (( DIRECT_GEN0 == 0 )); then
        echo
        echo "== generation 0: skipped =="
        echo "   The $SURROGATE_METHOD surrogate is built from the generation-0 checkpoint, so"
        echo "   attacking generation 0 would validate the suffix against the model the search"
        echo "   was derived from. Pass --direct-gen0 to attack it without a surrogate instead."
        STATUS_GENS+=("$gen")
        STATUS_CODES+=("skipped")
        STATUS_FILES+=("")
        continue
    fi

    if (( gen == 0 )); then
        method_args=(-sm none)
        result_file="$RESULTS_DIR/attack_gen0_${SPECIFIER_NAME}.json"
        label="generation 0 (direct, no surrogate)"
    else
        method_args=(-sm "$SURROGATE_METHOD")
        result_file="$RESULTS_DIR/attack_gen${gen}_${SPECIFIER_NAME}_${SURROGATE_METHOD}_surrogate.json"
        label="generation $gen ($SURROGATE_METHOD surrogate, n = $((gen + 1)))"
    fi
    log_file="$LOG_DIR/attack_gen${gen}_${SURROGATE_METHOD}.log"

    echo
    echo "== $label =="

    if [[ -f "$result_file" ]] && (( FORCE == 0 )); then
        echo "   already done: $result_file (pass --force to re-run)"
        STATUS_GENS+=("$gen")
        STATUS_CODES+=("cached")
        STATUS_FILES+=("$result_file")
        continue
    fi

    cmd=("$PYTHON" "$ATTACK"
         -cg "$gen"
         -bs "$BLOCK_SIZE"
         "${method_args[@]}"
         -ms "$MODEL_SPECIFIER"
         -p "$PATH_ROOT")
    if (( ${#EXTRA_ARGS[@]} )); then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    echo "   \$ ${cmd[*]}"
    if (( DRY_RUN )); then
        STATUS_GENS+=("$gen")
        STATUS_CODES+=("dry-run")
        STATUS_FILES+=("")
        continue
    fi

    echo "   log: $log_file"
    gen_started=$SECONDS
    # tee so the run stays watchable while still leaving a complete per-generation log
    "${cmd[@]}" 2>&1 | tee "$log_file"
    code=${PIPESTATUS[0]}
    elapsed=$(( SECONDS - gen_started ))

    if (( code == 0 )); then
        echo "   done in ${elapsed}s"
        STATUS_CODES+=("ok")
    else
        echo "   FAILED with exit code $code after ${elapsed}s — see $log_file"
        STATUS_CODES+=("exit $code")
        FAILURES=$(( FAILURES + 1 ))
    fi
    STATUS_GENS+=("$gen")
    STATUS_FILES+=("$result_file")
done

# ──────────────────────────────── summary ────────────────────────────────
echo
echo "############################################################"
echo "## sweep summary  (total $(( SECONDS - STARTED_AT ))s)"
echo "############################################################"
printf '## %-5s %-10s %-11s %-9s %s\n' gen run capability hits note

for (( i = 0; i < ${#STATUS_GENS[@]}; i++ )); do
    gen="${STATUS_GENS[$i]}"
    code="${STATUS_CODES[$i]}"
    file="${STATUS_FILES[$i]}"
    capability="-"
    hits="-"
    note=""

    # only read the result file for runs that actually produced one. A failed generation may
    # leave an *older* file in place — reporting its numbers would credit the failed run with a
    # previous run's outcome
    if [[ "$code" != "ok" && "$code" != "cached" ]]; then
        file=""
        [[ "$code" == "skipped" ]] || note="see log"
    fi

    if [[ -n "$file" && -f "$file" ]]; then
        # the results file is written even when the capability gate stops the run
        read -r capability hits note < <(
            "$PYTHON" - "$file" <<'PY'
import json, sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

probe = report.get("capability_probe") or {}
solved, probed = len(probe.get("collapsed_solved") or []), probe.get("n_probed") or 0
capability = f"{solved}/{probed}" if probed else "not-probed"
hits = sum(len(r.get("successes") or []) for r in report.get("results") or [])
note = "gate-aborted" if report.get("aborted") else ("selective-hit" if hits else "no-hit")
print(capability, hits, note)
PY
        ) || { capability="?"; hits="?"; note="unreadable-result"; }
    elif [[ "$code" == "skipped" ]]; then
        note="surrogate anchor"
    elif [[ "$code" == "ok" || "$code" == "cached" ]]; then
        note="no result file written"
    fi

    printf '## %-5s %-10s %-11s %-9s %s\n' "$gen" "$code" "$capability" "$hits" "$note"
done

echo "############################################################"
if (( FAILURES > 0 )); then
    echo "## $FAILURES generation(s) failed unexpectedly — see the logs in $LOG_DIR"
    exit 1
fi
echo "## all generations completed (gate aborts are expected on collapsed generations)"
