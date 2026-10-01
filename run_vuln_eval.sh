#!/usr/bin/env bash
#
# Runs the whole vulnerability-attack evaluation and draws its figures.
#
# Five phases, in dependency order. Each is skippable, and each skips work whose output already
# exists unless --force is given, so an interrupted evaluation resumes instead of restarting:
#
#   attack       run_attack_sweep.sh --vuln over the generations, twice per model: -m none is the
#                direct upper bound (the attacker holds the checkpoint) and -m logit is the transfer
#                claim (base + generation 0 only). Nothing else uses a surrogate.
#   controls     random suffixes and the unoptimised init string, scored at every generation the
#                attack covered. This is the control that rules out "a collapsed model is fragile,
#                so any perturbation would do", and --num-random is matched to the number of
#                behavioural checks the search itself performs so the two rates share a denominator.
#   temperature  the verified suffixes re-scored sample by sample with more draws than the search
#                affords per check (the attack itself verifies with a majority over a few samples
#                at the deployment decoding, or greedily under --verify_temperature 0).
#   perplexity   the prompt scores the filter defence's ROC is drawn from.
#   plots        run_vuln_plots.py over whatever the phases above produced.
#
# Prerequisite, and it is not checked for you beyond a warning: every model in --models needs a
# collapse run on disk under --path. Produce one with
#   python run_baseline.py -ng <N+1> -bs <bs> -msz <size> -rdf <value> -p <path>
# or sweep the mixture with ./run_rdf_sweep.sh. The attack phase fails per generation with a
# FileNotFoundError naming the checkpoint it wanted if a run is missing.
#
# The mixture list is the ablation axis. One value is the main experiment; ten values (0.0 .. 0.9)
# is the attackable-window study, and each of them needs its own collapse run first.
#
# Usage:
#   ./run_vuln_eval.sh -n 9 --models "0.5b 3b" --rdf "0.5" [-p ./runs/x] [options]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

GENERATIONS=""
MODELS="0.5b 3b"
MIXTURES="0.5"
PATH_ROOT="."
BLOCK_SIZE=512
PHASES="attack,controls,temperature,perplexity,plots"
NUM_RANDOM=75
TEMPERATURE=0.7
NUM_SAMPLES=16
FORCE=0
DRY_RUN=0
NO_USETEX=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Runs the vulnerability-attack evaluation end to end and draws its figures.

Required:
  -n, --generations N       highest collapse generation to evaluate (sweeps 0..N)

Options:
  -p, --path PATH           root holding model_outputs/ and attack_results/ (default: .)
  -b, --block-size N        block size baked into the checkpoint names (default: 512)
      --models "A B"        model sizes off the Qwen2.5-Coder ladder (default: "0.5b 3b")
      --rdf "A B"           real-data fractions to evaluate; one value is the main experiment,
                            "0.0 0.1 ... 0.9" is the attackable-window ablation (default: "0.5")
      --phases LIST         comma separated subset of
                            attack,controls,temperature,perplexity,plots (default: all)
      --num-random N        random suffixes per target in the control phase. Match it to the
                            search's behavioural checks -- restarts * num_steps / verify_every,
                            which is 75 at the defaults (default: 75)
      --temperature T       decoding temperature of the survival phase (default: 0.7)
      --num-samples N       samples per suffix in the survival phase (default: 16)
      --no-usetex           draw the figures without LaTeX
      --force               redo phases whose output already exists
      --dry-run             print every command without running it
  -h, --help                this message

Everything after -- is passed through to run_attack_sweep.sh, e.g.:
  ./run_vuln_eval.sh -n 9 --models 0.5b -- -r 3 -ns 250 -ve 10
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--generations)  GENERATIONS="$2";  shift 2 ;;
        -p|--path)         PATH_ROOT="$2";    shift 2 ;;
        -b|--block-size)   BLOCK_SIZE="$2";   shift 2 ;;
        --models)          MODELS="$2";       shift 2 ;;
        --rdf)             MIXTURES="$2";     shift 2 ;;
        --phases)          PHASES="$2";       shift 2 ;;
        --num-random)      NUM_RANDOM="$2";   shift 2 ;;
        --temperature)     TEMPERATURE="$2";  shift 2 ;;
        --num-samples)     NUM_SAMPLES="$2";  shift 2 ;;
        --no-usetex)       NO_USETEX=1;       shift ;;
        --force)           FORCE=1;           shift ;;
        --dry-run)         DRY_RUN=1;         shift ;;
        -h|--help)         usage; exit 0 ;;
        --)                shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$GENERATIONS" ]]; then
    echo "error: -n/--generations is required" >&2; usage >&2; exit 2
fi
if ! [[ "$GENERATIONS" =~ ^[0-9]+$ ]]; then
    echo "error: -n/--generations must be a non-negative integer" >&2; exit 2
fi

RESULTS_DIR="$PATH_ROOT/attack_results"
LOG_DIR="$RESULTS_DIR/eval_logs"
(( DRY_RUN )) || mkdir -p "$LOG_DIR"

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

# resolved through the same helper the python entry points use, so a size table cannot drift from
# what they accept and the checkpoint names below are the ones a run actually wrote
specifier_of() {
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -c \
        'import sys
from utils.models import resolve_model_specifier
print(resolve_model_specifier(sys.argv[1], "").split("/")[-1])' "$1"
}
mixture_tag_of() {
    PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -c \
        'import sys
from utils.naming import mixture_tag
print(mixture_tag(float(sys.argv[1])))' "$1"
}

# how the attack phase's behavioural checks decode, read out of the passthrough the same way
# run_attack_sweep.sh does: the survival phase re-scores the hits of *those* files and the plots
# draw *those* runs, so all three have to agree on the tag. Empty for a greedy sweep (-- -vt 0)
if ! DECODING_TAG="$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -c \
        'import sys
from utils.naming import verification_tag_from_argv
print(verification_tag_from_argv(sys.argv[1:]))' "${EXTRA_ARGS[@]}")"; then
    echo "error: could not resolve the verification tag from the passthrough arguments" >&2
    exit 2
fi

run() {
    echo "   \$ $*"
    if (( DRY_RUN )); then
        return 0
    fi
    "$@"
}

FAILURES=0
STARTED_AT=$SECONDS

echo "############################################################"
echo "## vulnerability evaluation"
echo "##   generations  : 0..$GENERATIONS"
echo "##   models       : $MODELS"
echo "##   real data    : $MIXTURES"
echo "##   phases       : $PHASES"
if [[ -n "$DECODING_TAG" ]]; then
    echo "##   verification : sampled, attack files tagged $DECODING_TAG"
else
    echo "##   verification : greedy (untagged attack files)"
fi
echo "##   path         : $PATH_ROOT"
echo "############################################################"

# a missing collapse run is the most common reason this script does nothing useful, so it is named
# up front rather than discovered per generation
for size in $MODELS; do
    name="$(specifier_of "$size")" || exit 2
    for mixture in $MIXTURES; do
        tag="$(mixture_tag_of "$mixture")"
        final="$PATH_ROOT/model_outputs/model_${GENERATIONS}_bs${BLOCK_SIZE}_${name}${tag}_fp16"
        if [[ ! -d "$final" ]]; then
            echo "## warning: no collapse run for $size at rdf $mixture — $final is missing."
            echo "##   python run_baseline.py -ng $((GENERATIONS + 1)) -bs $BLOCK_SIZE" \
                 "-msz $size -rdf $mixture -p $PATH_ROOT"
        fi
    done
done

# ────────────────────────────────── attack ────────────────────────────────────────────────
if has_phase attack; then
    echo
    echo "== phase: attack =="
    for size in $MODELS; do
        for mixture in $MIXTURES; do
            for method in none logit; do
                sweep=("$SCRIPT_DIR/run_attack_sweep.sh"
                       -n "$GENERATIONS" -p "$PATH_ROOT" -b "$BLOCK_SIZE"
                       -msz "$size" -rdf "$mixture" -m "$method" --vuln)
                (( FORCE )) && sweep+=(--force)
                (( DRY_RUN )) && sweep+=(--dry-run)
                if (( ${#EXTRA_ARGS[@]} )); then
                    sweep+=(-- "${EXTRA_ARGS[@]}")
                fi
                echo
                echo "-- $size, rdf $mixture, surrogate $method --"
                "${sweep[@]}" || FAILURES=$(( FAILURES + 1 ))
            done
        done
    done
fi

# ─────────────────────── controls, survival, both per generation ───────────────────────────
# both phases score suffixes against one checkpoint, so they share the loop and differ only in the
# arguments handed to utils.verify_suffixes
score_suffixes() {
    local size="$1" mixture="$2" generation="$3" source="$4" mode="$5"
    local name tag result out
    name="$(specifier_of "$size")"
    tag="$(mixture_tag_of "$mixture")"
    # the optimized suffixes come out of one attack file, so their survival file carries that
    # file's verification tag; the random and init controls come from no attack file and do not
    local attack_tag=""
    if [[ "$source" == "optimized" ]]; then
        attack_tag="$DECODING_TAG"
    fi
    out="$RESULTS_DIR/suffix_verification_gen${generation}_${name}${tag}${attack_tag}_${source}_${mode}.json"

    if [[ -f "$out" ]] && (( FORCE == 0 )); then
        echo "   already done: $(basename "$out")"
        return 0
    fi

    local command=("$PYTHON" -m utils.verify_suffixes
                   -cg "$generation" -bs "$BLOCK_SIZE" -rdf "$mixture" -msz "$size"
                   -p "$PATH_ROOT" -of "$out" --source "$source")
    if [[ "$source" == "random" ]]; then
        command+=(--num_random "$NUM_RANDOM")
    fi
    if [[ "$mode" == "sampled" ]]; then
        # the suffixes to re-score are the verified hits of the direct attack on this very cell;
        # with none of them there is nothing to say about survival
        result="$RESULTS_DIR/attack_gen${generation}_${name}${tag}_vuln${DECODING_TAG}.json"
        if [[ ! -f "$result" ]]; then
            echo "   no attack result for generation $generation — skipped"
            return 0
        fi
        if ! PYTHONPATH="$SCRIPT_DIR" "$PYTHON" -c \
                'import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
hits = sum(len(r.get("successes") or []) for r in report.get("results") or [])
sys.exit(0 if hits else 1)' "$result"; then
            echo "   no verified hits at generation $generation — nothing to re-score"
            return 0
        fi
        command+=(--suffix_file "$result" --temperature "$TEMPERATURE"
                  --num_samples "$NUM_SAMPLES")
    fi

    echo "   \$ ${command[*]}"
    if (( DRY_RUN )); then
        return 0
    fi
    # a plain pipe rather than a process substitution: the latter is set up by the shell even when
    # the command is not run, so a --dry-run would try to open a log under a directory that a dry
    # run deliberately does not create, and its output interleaves with the phase headings
    local log code
    log="$LOG_DIR/$(basename "${out%.json}").log"
    echo "   log: $log"
    "${command[@]}" 2>&1 | tee "$log"
    code=${PIPESTATUS[0]}
    if (( code != 0 )); then
        echo "   FAILED with exit code $code -- see $log"
        FAILURES=$(( FAILURES + 1 ))
    fi
}

if has_phase controls; then
    echo
    echo "== phase: controls =="
    for size in $MODELS; do
        for mixture in $MIXTURES; do
            for (( generation = 0; generation <= GENERATIONS; generation++ )); do
                echo "-- $size, rdf $mixture, generation $generation --"
                score_suffixes "$size" "$mixture" "$generation" random greedy
                score_suffixes "$size" "$mixture" "$generation" init greedy
            done
        done
    done
fi

if has_phase temperature; then
    echo
    echo "== phase: temperature =="
    for size in $MODELS; do
        for mixture in $MIXTURES; do
            for (( generation = 0; generation <= GENERATIONS; generation++ )); do
                echo "-- $size, rdf $mixture, generation $generation --"
                score_suffixes "$size" "$mixture" "$generation" optimized sampled
            done
        done
    done
fi

# ──────────────────────────────── perplexity ──────────────────────────────────────────────
if has_phase perplexity; then
    echo
    echo "== phase: perplexity =="
    for size in $MODELS; do
        name="$(specifier_of "$size")"
        out="$RESULTS_DIR/suffix_perplexity_${name}.json"
        if [[ -f "$out" ]] && (( FORCE == 0 )); then
            echo "   already done: $(basename "$out")"
            continue
        fi
        run "$PYTHON" -m utils.suffix_perplexity -msz "$size" -bs "$BLOCK_SIZE" \
            -rp "$RESULTS_DIR" -dp "$PATH_ROOT/generated_datasets/" -of "$out" \
            || FAILURES=$(( FAILURES + 1 ))
    done
fi

# ────────────────────────────────── figures ───────────────────────────────────────────────
if has_phase plots; then
    echo
    echo "== phase: plots =="
    for mode in none logit; do
        plots=("$PYTHON" "$SCRIPT_DIR/run_vuln_plots.py" -rp "$RESULTS_DIR"
               -pp "$PATH_ROOT/plots" -bs "$BLOCK_SIZE" -m "$mode" -vf "$DECODING_TAG")
        (( NO_USETEX )) && plots+=(--no_usetex)
        run "${plots[@]}" || FAILURES=$(( FAILURES + 1 ))
    done
fi

echo
echo "############################################################"
echo "## evaluation finished in $(( SECONDS - STARTED_AT ))s"
if (( FAILURES > 0 )); then
    echo "## $FAILURES step(s) failed — see the logs in $LOG_DIR"
    exit 1
fi
echo "## all requested phases completed"
