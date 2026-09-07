"""reads the vulnerability attack's result files into the records the figures are drawn from

`run_attack_vuln.py` writes one JSON per (generation, mixture, surrogate method). Every figure in
`run_vuln_plots.py` needs the same three things out of those files — which tasks were attackable,
which of them were hit, and what the clean controls said — so the parsing lives here once rather
than in each plotting function, where the definitions would drift apart silently.

**The run's identity is read out of the file, not out of its name.** ``collapsed_generation``,
``real_data_fraction`` and ``surrogate_method`` are all recorded inside, and the model is derivable
from ``baseline_model``, so a renamed or relocated file still describes itself correctly and a
result written by a future naming scheme keeps loading.

Kept free of torch/transformers imports, same rule as utils/plotting.py and utils/naming.py: the
plotting entry point must not drag a model stack in to draw a bar chart.

The one definition that matters and is easy to get wrong: a task counts toward the denominator only
if it is in the capability probe's ``usable`` list, i.e. *both* the collapsed model and the pristine
baseline answered it correctly on the suffix-free prompt. A task the baseline already gets wrong
cannot support the claim that the suffix flipped a correct answer, and a task the collapsed model
already gets wrong measures collapse rather than the attack — see run_attack.py's capability gate.
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field

# statuses, mirrored from run_attack.WRONG_STATUSES / run_attack_vuln so this module stays
# import-light. A mismatch would be caught by check_status_vocabulary() below
WRONG_STATUSES: tuple = ("fail", "fail_exception")
PASS_STATUS: str = "pass"

# ``attack_gen{g}_{name}{mixture}_vuln[_{method}_surrogate].json`` — used only to *find* the files
# and to report where a record came from; every field below is read from the contents
RESULT_GLOB: str = "attack_gen*_vuln*.json"

# the weakness each target introduces, for figure labels. Mirrors run_attack_vuln.VULN_TASKS, which
# is not imported here because it pulls in torch through run_attack and this module is what the
# plotting entry point uses. Drift is *detected* rather than assumed away: unknown_targets() names
# anything that appears in a result file and is missing here, so a sixth target shows up as a
# warning instead of silently losing its label
TARGET_CWE: dict = {
    "fetch_user": "CWE-89",
    "generate_token": "CWE-338",
    "run_ping": "CWE-78",
    "load_config": "CWE-502",
    "fetch_json": "CWE-295",
}
_SIZE_PATTERN = re.compile(r"-(\d+(?:\.\d+)?B)-", re.IGNORECASE)


@dataclass
class TaskOutcome:
    """What one attack run learned about one target.

    Attributes:
        task: target name, e.g. "fetch_json"
        baseline_clean: the pristine model's verdict on the suffix-free prompt
        collapsed_clean: the collapsed model's verdict on the suffix-free prompt
        attackable: both models answered the clean prompt correctly, so a hit is attributable
        n_success: verified hits, i.e. collapsed insecure while the baseline stayed secure
        n_verified: behavioural checks performed, the denominator a hit rate per check needs
        skipped: why the search never ran on this target, if it did not
        suffixes: the suffix of every verified hit
    """

    task: str
    baseline_clean: str = ""
    collapsed_clean: str = ""
    attackable: bool = False
    n_success: int = 0
    n_verified: int = 0
    skipped: str | None = None
    suffixes: list = field(default_factory=list)

    @property
    def hit(self) -> bool:
        """True if at least one verified hit was recorded."""
        return self.n_success > 0

    @property
    def exclusion(self) -> str:
        """Why this target is out of the denominator, or "" when it is in it.

        The two reasons are opposites and must not be pooled: ``baseline-insecure`` means the
        pristine model already writes the vulnerable variant, so there is no correct answer to
        flip; ``collapsed-broken`` means the collapsed model no longer writes classifiable code, so
        any wrong output would measure collapse instead of the attack.
        """
        if self.attackable:
            return ""
        if self.baseline_clean and self.baseline_clean != PASS_STATUS:
            return "baseline-insecure"
        if self.collapsed_clean and self.collapsed_clean != PASS_STATUS:
            return "collapsed-broken"
        return "not-probed"


@dataclass
class AttackRecord:
    """One result file, flattened to what the figures need.

    Attributes:
        path: where it was read from
        model: the pristine model's repo id
        specifier_name: its trailing component, which the checkpoints are named after
        size_label: the ladder rung parsed out of the name, e.g. "0.5B"
        generation: the collapse generation attacked
        real_data_fraction: the mixture the collapse run was trained with
        surrogate_method: "none" for a direct attack, "logit" in transfer mode
        transfer: whether a surrogate stood in for the collapsed model during the search
        aborted: the capability gate stopped the run before any optimization
        tasks: per-target outcomes, keyed by target name
        surrogate_quality: the SurrogateReport dict, or None outside transfer mode
        config: the SearchConfig the run used
    """

    path: str
    model: str
    specifier_name: str
    size_label: str
    generation: int
    real_data_fraction: float
    surrogate_method: str
    transfer: bool
    aborted: bool
    tasks: dict = field(default_factory=dict)
    surrogate_quality: dict | None = None
    config: dict = field(default_factory=dict)

    @property
    def attackable(self) -> list:
        """Targets in the denominator, in file order."""
        return [name for name, task in self.tasks.items() if task.attackable]

    @property
    def hits(self) -> list:
        """Attackable targets with at least one verified hit."""
        return [name for name in self.attackable if self.tasks[name].hit]

    @property
    def hit_rate(self) -> float:
        """Hit targets over attackable targets; NaN when nothing was attackable.

        NaN rather than 0.0 deliberately: "the attack found nothing" and "there was nothing to
        attack" are different statements, and a 0 would let a collapsed generation with no
        attributable target drag a mean down as though the attack had failed there.
        """
        denominator = len(self.attackable)
        if not denominator:
            return float("nan")
        return len(self.hits) / denominator


def size_label(specifier_name: str) -> str:
    """The ladder rung inside a model's short name, e.g. "0.5B" for Qwen2.5-Coder-0.5B-Instruct."""
    found = _SIZE_PATTERN.search(specifier_name)
    return found.group(1).upper() if found else specifier_name


def _task_outcome(name: str, result: dict, probe: dict) -> TaskOutcome:
    """Builds one target's outcome from its result entry and the run's capability probe."""
    clean = (probe.get("per_task") or {}).get(name) or {}
    return TaskOutcome(
        task=name,
        baseline_clean=clean.get("baseline_status", ""),
        collapsed_clean=clean.get("collapsed_status", ""),
        attackable=name in (probe.get("usable") or []),
        n_success=len(result.get("successes") or []),
        n_verified=len(result.get("verifications") or []),
        skipped=result.get("skipped"),
        suffixes=[hit.get("suffix", "") for hit in (result.get("successes") or [])],
    )


def load_record(path: str) -> AttackRecord:
    """Reads one result file.

    Args:
        path (str): the JSON written by run_attack_vuln.py

    Returns:
        AttackRecord: the flattened record

    Raises:
        ValueError: the file is not an attack result
    """
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if "capability_probe" not in report or "results" not in report:
        raise ValueError(f"{path} is not an attack result file")

    probe = report.get("capability_probe") or {}
    model = report.get("baseline_model", "")
    specifier = model.split("/")[-1]
    # every target the run knew about: the ones it searched, plus the ones the probe excluded
    # before the search, which are in the probe but not in `results` when the run aborted
    names = [row["task"] for row in report["results"]]
    names += [name for name in (probe.get("per_task") or {}) if name not in names]
    by_name = {row["task"]: row for row in report["results"]}

    return AttackRecord(
        path=path,
        model=model,
        specifier_name=specifier,
        size_label=size_label(specifier),
        generation=int(report.get("collapsed_generation", -1)),
        real_data_fraction=float(report.get("real_data_fraction", 0.0)),
        surrogate_method=report.get("surrogate_method", "none"),
        transfer=bool(report.get("transfer_mode", False)),
        aborted=bool(report.get("aborted", False)),
        tasks={name: _task_outcome(name, by_name.get(name, {}), probe) for name in names},
        surrogate_quality=report.get("surrogate_quality"),
        config=report.get("config") or {},
    )


def load_records(
    results_dir: str,
    specifier_name: str = "",
    surrogate_method: str = "",
    real_data_fraction: float | None = None,
) -> list:
    """Reads every vulnerability result under `results_dir`, optionally filtered.

    Args:
        results_dir (str): the attack_results/ directory
        specifier_name (str): keep only this model's runs, e.g. "Qwen2.5-Coder-0.5B-Instruct"
        surrogate_method (str): keep only "none" (direct) or "logit" (transfer)
        real_data_fraction (float | None): keep only this mixture

    Returns:
        list: AttackRecords sorted by (model, mixture, method, generation)
    """
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, RESULT_GLOB))):
        try:
            record = load_record(path)
        except (ValueError, json.JSONDecodeError, KeyError):
            # a truncated file from an interrupted run is skipped rather than aborting a figure
            continue
        if specifier_name and record.specifier_name != specifier_name:
            continue
        if surrogate_method and record.surrogate_method != surrogate_method:
            continue
        if real_data_fraction is not None and abs(
            record.real_data_fraction - real_data_fraction
        ) > 1e-9:
            continue
        records.append(record)
    records.sort(
        key=lambda r: (r.specifier_name, r.real_data_fraction, r.surrogate_method, r.generation)
    )
    return records


def target_order(records: list) -> list:
    """The union of every target seen, in first-appearance order.

    So that a figure's rows stay in the same order across models and mixtures even when a run
    excluded a target entirely.
    """
    order = []
    for record in records:
        for name in record.tasks:
            if name not in order:
                order.append(name)
    return order


def unknown_targets(records: list) -> list:
    """Targets seen in the results that TARGET_CWE has no label for."""
    return [name for name in target_order(records) if name not in TARGET_CWE]


def by_generation(records: list) -> dict:
    """Groups records by generation, keeping the load order within each."""
    grouped: dict = {}
    for record in records:
        grouped.setdefault(record.generation, []).append(record)
    return grouped


def working_suffixes(records: list) -> list:
    """Every verified hit across `records`, as flat rows.

    The input to the temperature re-verification and to the perplexity filter: both need the
    (model, generation, mixture, target, suffix) tuple and nothing else.

    Returns:
        list: dicts with keys specifier_name, generation, real_data_fraction, surrogate_method,
            task and suffix
    """
    rows = []
    for record in records:
        for name, task in record.tasks.items():
            for suffix in task.suffixes:
                rows.append(
                    {
                        "specifier_name": record.specifier_name,
                        "generation": record.generation,
                        "real_data_fraction": record.real_data_fraction,
                        "surrogate_method": record.surrogate_method,
                        "task": name,
                        "suffix": suffix,
                    }
                )
    return rows
