#!/usr/bin/env bash
# run_data_split_experiment.sh — does the surrogate still transfer when the attacker never saw
# the victim's data?
#
# The grid in run_paper.sh builds the attacker's surrogate from the lineage's *own* generation 0,
# i.e. a fine-tune on the very rows the victim's lineage trains on. This experiment splits the
# human corpus into two disjoint halves (run_baseline.py --data_split, one fixed permutation):
#
#   1. attacker : half a, generation 0 only (run_paper.sh --phases collapse, GENERATIONS=0) — the
#                 fine-tune the surrogate is extrapolated from, in $ROOT/attacker
#   2. victim   : half b, the full lineage, utility and perplexity (collapse,utility), in
#                 $ROOT/victim
#   3. own      : the victim root's own attacks — the direct attack and the surrogate anchored on
#                 the victim's *own* generation 0 — plus the figures (attack,control,plots). The
#                 reference the split pass is read against: same lineage, same budget, anchor on
#                 the data. Every cell carries the random-suffix control (run_paper.sh
#                 RANDOM_TRIALS), the floor both logit columns are read against
#   4. split    : the surrogate anchored on the *attacker's* generation 0 (ANCHOR_ROOT, logit
#                 sweeps only, result files tagged _splitanchor), with its sink screen measured
#                 on the attacker root, and the tables under the tag (attack,plots)
#   5. compare  : prints the pooled hit-rate tables of 3 and 4 side by side
#
# Nothing the attacker runs touches half b: generation 0's rows, the instructions its corpus is
# sampled for and the real-data pool of the mixture all come from half a. Every stage is a
# run_paper.sh invocation, so its environment overrides (SIZES, MIXTURES, GENERATIONS, the GPU
# lists, the search knobs) apply unchanged, the job pool and the resume rules are the same, and
# the logs are under each root's paper_logs/. Everything is resumable; `--stages` cuts it down.
#
# Usage:
#   ./run_data_split_experiment.sh [--stages attacker,victim,own,split,compare] [--dry-run]
#                                  [--force] [--yes] [--no-usetex]
#   ROOT=./runs/split SIZES=0.5b MIXTURES="0.0 0.8 1.0" ./run_data_split_experiment.sh --yes
#
# Environment: ROOT (default ./runs/split), SPLIT_SEED (the halves' permutation, default 0),
# plus everything run_paper.sh reads. The victim's generations default to run_paper.sh's 0..9;
# the attacker always trains generation 0 only.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="${ROOT:-./runs/split}"
SPLIT_SEED="${SPLIT_SEED:-0}"
ATTACKER_ROOT="$ROOT/attacker"
VICTIM_ROOT="$ROOT/victim"
STAGES="attacker,victim,own,split,compare"
PASS=()        # flags handed to every run_paper.sh call
DRY_RUN=0

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)    STAGES="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; PASS+=(--dry-run); shift ;;
        --force)     PASS+=(--force); shift ;;
        -y|--yes)    PASS+=(--yes); shift ;;
        --no-usetex) PASS+=(--no-usetex); shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
banner() { printf '\n%s\n== %s\n%s\n' "$(printf '═%.0s' {1..70})" "$1" "$(printf '═%.0s' {1..70})"; }

# One run_paper.sh call. Its own preflight, plan and pool do the work; the environment is the
# experiment's only interface to it, so every override is spelled out on the command line here.
paper() {
    local label="$1"; shift
    local envs=() args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
    [[ "${1:-}" == "--" ]] && shift
    args=("$@")
    banner "$label"
    echo "   $ ${envs[*]} ./run_paper.sh ${args[*]} ${PASS[*]:-}"
    env "${envs[@]}" ./run_paper.sh "${args[@]}" "${PASS[@]+"${PASS[@]}"}"
}

echo "############################################################"
echo "## disjoint-data experiment"
echo "##   root       : $ROOT  (attacker: half a -> $ATTACKER_ROOT; victim: half b -> $VICTIM_ROOT)"
echo "##   split seed : $SPLIT_SEED  (run_baseline.py --data_split_seed)"
echo "##   stages     : $STAGES"
echo "##   sizes      : ${SIZES:-0.5b}; mixtures: ${MIXTURES:-run_paper.sh default}; victim generations 0..${GENERATIONS:-9}"
echo "############################################################"

if has_stage attacker; then
    paper "1. attacker: generation 0 on half a" \
        ROOT="$ATTACKER_ROOT" GENERATIONS=0 \
        BASELINE_EXTRA="--data_split a --data_split_seed $SPLIT_SEED" \
        -- --phases collapse
fi

if has_stage victim; then
    paper "2. victim: the lineage on half b, with its utility and perplexity" \
        ROOT="$VICTIM_ROOT" \
        BASELINE_EXTRA="--data_split b --data_split_seed $SPLIT_SEED" \
        -- --phases collapse,utility
fi

if has_stage own; then
    paper "3. own-anchor attacks on the victim (direct, and the surrogate from the victim's generation 0)" \
        ROOT="$VICTIM_ROOT" \
        -- --phases attack,control,plots
fi

if has_stage split; then
    paper "4. split-anchor attacks on the victim (the surrogate from the attacker's generation 0)" \
        ROOT="$VICTIM_ROOT" ANCHOR_ROOT="$ATTACKER_ROOT" METHODS=logit \
        -- --phases attack,control,plots
fi

if has_stage compare && (( DRY_RUN == 0 )); then
    banner "5. compare: pooled hit rates, own anchor against split anchor"
    for size in ${SIZES:-0.5b}; do
        own="$VICTIM_ROOT/plots/tables_${size}.md"
        split="$VICTIM_ROOT/plots/tables_${size}_splitanchor.md"
        for f in "$own" "$split"; do
            if [[ ! -f "$f" ]]; then
                echo "   missing $f (run the earlier stages first)"
                continue
            fi
            echo; echo "── $f"
            # the pooled tables and the random-suffix control beside them, for both suites
            grep -nE "^## .*(pooled over the generations)" -A 6 "$f" | sed 's/^/   /'
        done
    done
    echo
    echo "   Read the 'logit' columns of the two files against each other: the same lineage,"
    echo "   budget and verdict, the surrogate anchored on the victim's own generation 0 in the"
    echo "   first and on the attacker's disjoint-data generation 0 in the second. The direct"
    echo "   column is the same files in both. A split rate at the own rate means the surrogate"
    echo "   needs no row of the victim's data; a split rate at the random-suffix control means"
    echo "   it does."
fi
