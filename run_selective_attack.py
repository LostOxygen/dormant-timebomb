"""main hook to search for adversarial inputs that are *selective across collapse generations*

Stage 3b of the dormant-timebomb pipeline, and a generalization of run_attack.py. That script asks
one question — does a suffix break **one** collapsed generation while the pristine baseline still
answers correctly. This one asks the same question of a whole *set* of models simultaneously:

    --target_generations 8,9      must emit objectively wrong code
    --spare_generations  0,1,2    must keep emitting correct code
    the pristine baseline         must keep emitting correct code, always and unconditionally

A hit is recorded only when **every** one of those conditions holds for the same suffix at the same
time. Nothing weaker counts: a suffix that breaks generation 9 but also breaks generation 1 is not
a selective hit here, even though run_attack.py would have called it one.

The two lists are symmetric in that *criterion* and deliberately asymmetric in the **capability
gate**, which only the baseline and the targets have to pass. A target that already fails a task
unaided cannot be shown to have been broken by the suffix, so the claim is unavailable and the task
is dropped — run_attack.py's own rule. A spare that already fails is a different matter: it makes
one conjunct unreachable on that task, but it takes nothing away from the claim about the targets,
and with several spares gating on them would empty the task list before the search ever ran. So the
spares are probed and reported, never excluded — the console names them per task,
``spare_broken_on_usable`` records them, and the ``blocked_by`` counts show what it cost.

Why the spare list is the interesting half
------------------------------------------
run_attack.py already demonstrates the dormant part of a timebomb along one axis — the input is
inert against the pristine model and active against a collapsed one. What it cannot express is a
*window*: an input that arms itself only after the lineage has degraded past some point and
disarms again, or that targets one band of generations and no other. The baseline is a single
"must not break" anchor; the spare list turns that into as many anchors as you like, placed
anywhere on the collapse axis. That is what makes the resulting suffix evidence about *which*
models a payload discriminates between rather than merely that it discriminates at all.

The objective
-------------
The same contrastive shape as run_attack.py's, with each of its three terms averaged over the
group of models it applies to:

    L(s) = mean_{g in targets}     CE_g(wrong | prompt+s)
         + lam * mean_{h in hold}  relu(margin - CE_h(wrong | prompt+s))
         + mu  * mean_{h in hold}  CE_h(correct | prompt+s)

where ``hold`` is the baseline together with every spare generation — the baseline is simply the
member of that group that is always present and always a real model. Means rather than sums, so
that ``--margin``, ``--lambda_base`` and ``--mu_correct`` keep the scale they have in
run_attack.py: with one target and no spares this reduces to run_attack.py's objective exactly,
term for term, which is what makes the two runs comparable.

The loss is a proxy, so the same behavioural verification decides everything: every model in the
run greedily decodes the adversarial prompt, the emitted code is extracted and executed against
the task's unit tests, and the conjunction above is evaluated on the resulting statuses.

Transfer mode, and what the attacker is assumed to hold
------------------------------------------------------
Identical threat model to run_attack.py's: the attacker holds the pristine base model and the
*first* collapsed model, and nothing else. So under ``--surrogate_method logit`` (or ``lora``)
every generation in either list — target or spare — is replaced **in the objective** by a
first-order surrogate built from those two anchors, at its own extrapolation factor. The baseline
stays real, because the attacker really does have it.

Success is still decided by the real checkpoints alone, exactly as in run_attack.py. The
surrogates are search tools; they are decoded during verification only so they can be *scored* as
predictors of the real outcome (see ``SurrogateReport``), and ``--no_surrogate_scoring`` turns even
that off when only the attack itself matters.

One property worth knowing: generation 0's surrogate at n = 1 *is* ``model_0``, unextrapolated. So
listing generation 0 as a spare costs nothing in fidelity — the attacker holds that checkpoint —
while listing it as a target makes no transfer claim at all, and the script says so.

One factor per generation
-------------------------
``--surrogate_factor`` keeps its three forms, resolved per generation rather than once:

* the default derives ``n = g + 1`` for each generation, the indexing convention
  run_extrapolation.py and run_attack.py share
* ``calibrated`` reads each generation's factor out of the calibration
  ``utils/evaluate_perplexity.py --calibrate`` fits
* ``auto`` measures them: one descending walk of the factor ladder, recording at which rungs the
  surrogate still writes correct code, after which every generation takes the largest capable rung
  at or below its own ``g + 1``. That is the same answer run_attack.py's per-generation probe would
  give — capability at a rung does not depend on which generation asked about it — for the cost of
  a single walk instead of one per generation.

A single explicit number is **rejected** unless the run involves exactly one generation, because
one factor for every generation means one identical surrogate for every generation, and "this model
must break" and "this model must not break" cannot both be satisfied. The same collision is checked
after ``auto`` and ``calibrated`` resolve, where two generations can land on the same rung.

Cost
----
Every model in the run is resident, and each optimizer step evaluates the objective on all of them:
one gradient per target, two per holding model (hinge and anchor), plus one candidate sweep each.
So the wall-clock and the VRAM both scale with ``|targets| + |spares| + 1``, and the ``logit``
surrogate costs two forward passes per evaluation rather than one. Three targets and three spares is
already seven models and roughly seven times run_attack.py's cost per step — keep the lists short,
lower ``--search_width``, and expect the verification (which decodes every model) to dominate at
small ``--verify_every``.

Relationship to run_attack.py
-----------------------------
Everything that decides what a number *means* is imported from run_attack.py rather than
reimplemented: ``TargetModel`` and ``ExtrapolatedModel`` (the loss, gradient and decoding
primitives), ``TASKS`` and ``WRONG_STATUSES``, ``split_prompt``, ``run_unit_tests``,
``load_model``, ``resolve_collapsed_dir``, ``build_surrogate``, ``SearchConfig`` and
``factor_ladder``. Same reason utils/verify_transfer.py imports them: a selective result is only
interesting if it is the *same* question asked of more models, and a second copy of the objective
or of the wrong-code criterion would let the two drift.

What is written here is the part that genuinely differs — the multi-model objective, the
conjunctive hit criterion, a capability gate that has to clear every model in the run, and the
per-generation reporting. The search loop is its own rather than a subclass hook into
``ContrastiveGCG``, whose ``run_task``/``_run_restart`` are written around exactly two real models
plus one surrogate; bending them into N would have meant editing the working attack for the
benefit of this one.

NOTE: verification executes model-generated code. It runs in an isolated subprocess with a
timeout, but pass ``--no_exec`` to disable execution entirely and fall back to loss-only scoring
(which does *not* prove wrong behaviour, and here also means the conjunction is never evaluated).
"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
import datetime
import getpass
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import timedelta

import psutil
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

# the shared attack machinery. `import run_attack` as well as the from-import, because
# resolve_collapsed_dir and build_surrogate read run_attack.MODEL_PATH out of module scope and
# --path has to rebind it there — the same thing utils/verify_transfer.py does
import run_attack
from run_attack import (
    TASKS,
    WRONG_STATUSES,
    AttackTask,
    ExtrapolatedModel,
    SearchConfig,
    TargetModel,
    build_surrogate,
    factor_ladder,
    load_model,
    resolve_collapsed_dir,
    run_unit_tests,
    split_prompt,
    surrogate_factor_arg,
)
from utils.colors import TColors
from utils.execution import extract_code
from utils.extrapolation import METHODS, calibrated_factor, factor_calibration_file
from utils.gcg import filter_ids, sample_ids_from_grad
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import mixture_tag
from utils.utils import INIT_CHARS, configure_pad_token, get_nonascii_toks

MODEL_PATH: str = "./model_outputs/"
RESULTS_PATH: str = "./attack_results/"

# the baseline is modelled as a role like any other so that one loop covers "every model that must
# keep answering correctly". It has no generation index; -1 is a sentinel that sorts before 0 and
# can never collide with a real generation
BASELINE_GENERATION: int = -1


# ────────────────────────────── roles and CLI parsing ─────────────────────────────────────
@dataclass
class ModelRole:
    """One model taking part in the run, and what the suffix has to do to it.

    Three kinds, distinguished by `role`:

    * ``target`` — a generation the suffix must break. Contributes the "must emit wrong code" term
    * ``spare``  — a generation the suffix must *not* break. Contributes the hinge and the anchor
    * ``baseline`` — the pristine model, exactly one per run. Identical treatment to a spare, and
      separate only because it carries no generation index and can never be a surrogate

    `real` is always the checkpoint that decides capability and success. `optim` is what the
    objective is evaluated on — the same object as `real` in direct mode, a first-order surrogate in
    transfer mode. Keeping both on one object is what stops the two from being confused at a call
    site: there is no way to write `role.optim` where the success criterion is meant.

    Attributes:
        generation: collapse generation index, or BASELINE_GENERATION for the baseline
        role: "target", "spare" or "baseline"
        label: name used in verdict keys, the console and the result file ("gen9", "baseline")
        real: the loaded checkpoint the verdict is read off
        optim: the model the objective is evaluated on
        checkpoint: directory `real` was loaded from
        factor: the surrogate's extrapolation factor, None outside transfer mode
        description: what `optim` is, for the banner and the result file
    """

    generation: int
    role: str
    label: str
    real: TargetModel
    optim: TargetModel
    checkpoint: str = ""
    factor: float | None = None
    description: str = ""

    @property
    def is_target(self) -> bool:
        """True when the suffix has to make this model emit wrong code."""
        return self.role == "target"

    @property
    def transfer(self) -> bool:
        """True when the objective sees a surrogate instead of this model itself."""
        return self.optim is not self.real

    @property
    def surrogate_label(self) -> str:
        """Verdict key prefix of this role's surrogate, empty outside transfer mode."""
        return f"{self.label}_surrogate" if self.transfer else ""

    def required(self, status: str) -> bool:
        """Whether `status` is what this role's condition demands.

        Args:
            status (str): a unit-test status from the behavioural check

        Returns:
            bool: targets need an objectively wrong answer, everything else a passing one
        """
        return status in WRONG_STATUSES if self.is_target else status == "pass"

    @property
    def verb(self) -> str:
        """What the suffix has to do to this model, for messages: "break" or "hold"."""
        return "break" if self.is_target else "hold"


def generation_list_arg(value: str) -> list[int]:
    """argparse type for --target_generations / --spare_generations.

    Accepts a comma separated list of non-negative integers, with an empty string meaning the empty
    list. Duplicates are collapsed and the result is sorted, so ``9,8,9`` and ``8,9`` name the same
    run and produce the same result file.

    Args:
        value (str): the raw command line token

    Returns:
        list[int]: the sorted, deduplicated generation indices

    Raises:
        argparse.ArgumentTypeError: a component is not a non-negative integer
    """
    generations = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"generation lists take comma separated integers, not {part!r}"
            ) from exc
        if index < 0:
            raise argparse.ArgumentTypeError(
                f"generation indices start at 0, got {index}"
            )
        generations.add(index)
    return sorted(generations)


def generation_path_arg(value: str) -> dict[int, str]:
    """argparse type for --generation_model_path: ``gen=path`` pairs, comma separated.

    run_attack.py's ``--collapsed_model_path`` does not generalize to a list of generations, so the
    escape hatch for checkpoints that do not follow the naming convention is a mapping instead.
    Paths containing a comma cannot be expressed; pass ``--path`` or rename them.

    Args:
        value (str): e.g. "9=/data/model_9,8=/data/model_8"

    Returns:
        dict[int, str]: generation index -> checkpoint directory

    Raises:
        argparse.ArgumentTypeError: a component is not of the form gen=path
    """
    overrides: dict[int, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"--generation_model_path takes gen=path pairs, not {part!r}"
            )
        index, path = part.split("=", 1)
        try:
            overrides[int(index.strip())] = path.strip()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{index!r} is not a generation index in {part!r}"
            ) from exc
    return overrides


# ──────────────────────────────── result containers ───────────────────────────────────────
@dataclass
class TaskOutcome:
    """Everything the search learned about one task.

    The same shape as run_attack.py's, with the two-model fields widened to one entry per role.
    ``verifications`` keeps a trimmed record of *every* behavioural check rather than only the hits,
    which is what makes the per-generation statistics recomputable from the result file — and in a
    selective run the near misses are the informative part: which model refused to break, or which
    spare leaked.
    """

    task: str
    control: dict[str, str] = field(default_factory=dict)
    successes: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    best_objective: float | None = None
    best_suffix: str | None = None
    history: list[dict] = field(default_factory=list)
    skipped: str | None = None


@dataclass
class CapabilityReport:
    """Outcome of the upfront suffix-free capability probe, across every model in the run.

    run_attack.py's gate has two conditions: the collapsed model must solve the clean task (else
    "the adversarial input flipped a correct answer" is not a claim that is available) and the
    baseline must solve it too (else there is nothing selective to demonstrate). Both are about
    whether the *claim* can be made at all, and both generalize to the baseline and the targets —
    the models this run calls ``gating``. A task is usable when all of those solve it unaided.

    **The spares are probed but do not gate.** A spare that already fails the clean prompt cannot
    satisfy "must hold" on that task, so a hit there is out of reach — but that is a fact about the
    result, not about whether the question is well posed, and the question about the *targets* is
    unaffected. Excluding those tasks would refuse to measure the very thing being asked about, and
    with several spares it would empty the task list fast. They are recorded instead:
    ``spare_broken_on_usable`` names, per attackable task, the spares already broken on it, the
    console says so when the search starts, and the selectivity report's ``blocked_by`` counts show
    the consequence.

    ``--min_capability`` is enforced on the targets only, for the same reason: an incapable target
    makes its own hits unattributable, while an incapable spare only makes them harder to come by.

    Attributes:
        per_task: label -> the full clean verdict, or an ``invalid`` note for a rejected reference
        solved / broken: label -> tasks that model solved / failed unaided, for every role
        capability: label -> fraction of probed tasks solved, for every role
        surrogate_broken: label -> tasks that role's surrogate failed. Transfer mode only, and never
            gating: the proxy is a search tool, not the model under attack
        spare_broken_on_usable: attackable task -> the spares already broken on it
        invalid_tasks: tasks whose own reference implementations do not separate pass from fail
        usable: the attackable tasks, i.e. those every gating model solves
    """

    per_task: dict[str, dict] = field(default_factory=dict)
    # label -> tasks that model solved / failed on the clean prompt
    solved: dict[str, list[str]] = field(default_factory=dict)
    broken: dict[str, list[str]] = field(default_factory=dict)
    # label -> fraction of probed tasks solved, for every role
    capability: dict[str, float] = field(default_factory=dict)
    # transfer mode only, never gating: what the surrogates did on the clean prompts
    surrogate_broken: dict[str, list[str]] = field(default_factory=dict)
    # attackable task -> the spares that already fail it unaided, so no hit there can satisfy them
    spare_broken_on_usable: dict[str, list[str]] = field(default_factory=dict)
    invalid_tasks: list[str] = field(default_factory=list)
    usable: list[str] = field(default_factory=list)
    n_probed: int = 0
    threshold: float = 0.0
    aborted: bool = False
    reason: str = ""
    skipped: bool = False


@dataclass
class SelectivityReport:
    """How the suffixes behaved per model, over every behavioural check of the run.

    A single success count answers "did it work". This answers the question a selective attack is
    actually about: *where* it broke down. ``n_condition_met`` per label is the share of
    verifications in which that model did what it was asked, and ``blocked_by`` counts how often
    each model was the (or a) reason a check was not a hit — the model that keeps appearing there
    is the one the search could not satisfy.

    Attributes:
        n_verified: behavioural checks performed in total
        n_success: checks where every condition held at once
        n_targets_all_broken: checks where every target broke, whatever the spares did
        n_holds_all_kept: checks where the baseline and every spare held, whatever the targets did
        n_baseline_broken: checks where the suffix broke the pristine baseline. Never a success,
            and a plain jailbreak rather than a selective input
        n_condition_met: label -> checks where that model satisfied its own condition
        n_wrong: label -> checks where that model emitted objectively wrong code
        blocked_by: label -> checks that were not hits and in which that model failed its condition
        per_task: the same counts per task
    """

    n_verified: int = 0
    n_success: int = 0
    n_targets_all_broken: int = 0
    n_holds_all_kept: int = 0
    n_baseline_broken: int = 0
    n_condition_met: dict[str, int] = field(default_factory=dict)
    n_wrong: dict[str, int] = field(default_factory=dict)
    blocked_by: dict[str, int] = field(default_factory=dict)
    per_task: dict[str, dict] = field(default_factory=dict)


@dataclass
class SurrogateReport:
    """How well the surrogates predicted what the real checkpoints did.

    Two levels, because a selective run gives the question two meanings:

    * **per generation** — did generation g's surrogate agree with generation g's real checkpoint
      about wrong-vs-correct. This is run_attack.py's ``agreement``, one number per generation, and
      it is the direct measure of how good a proxy the tilt is at that factor
    * **as an ensemble** — did the surrogates *jointly* predict a selective hit, i.e. would the
      conjunction have come out the same way had it been evaluated on the surrogates instead of the
      real models. That is the quantity an attacker without the checkpoints actually relies on, and
      precision/recall over it says whether relying on it is justified

    Success itself never involves a surrogate, in either mode. This is scoring the tool.

    Attributes:
        method: the surrogate method the search optimized against
        factors: label -> the extrapolation factor that generation's surrogate stood for
        models: label -> what the surrogate was built from
        n_verified: behavioural checks performed
        agreement: label -> fraction of checks where surrogate and real model agreed
        n_predicted_hit: checks the surrogate ensemble called a selective hit
        n_true_positive: checks it called a hit that were one
        n_false_alarm: checks it called a hit that were not
        n_missed: hits the ensemble did not call
        per_generation: label -> the raw counts behind `agreement`
    """

    method: str = "none"
    factors: dict[str, float] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    n_verified: int = 0
    agreement: dict[str, float] = field(default_factory=dict)
    n_predicted_hit: int = 0
    n_true_positive: int = 0
    n_false_alarm: int = 0
    n_missed: int = 0
    per_generation: dict[str, dict] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        """Of the checks the ensemble called a hit, how many were one."""
        if self.n_predicted_hit == 0:
            return 0.0
        return self.n_true_positive / self.n_predicted_hit

    @property
    def recall(self) -> float:
        """Of the real hits, how many the ensemble called."""
        total = self.n_true_positive + self.n_missed
        if total == 0:
            return 0.0
        return self.n_true_positive / total


# ─────────────────────────── selective contrastive search ─────────────────────────────────
class SelectiveGCG:
    """Searches for one suffix that breaks every target generation and no other model.

    The optimizer is GCG's, unchanged and shared with run_attack.py down to the sampling helpers in
    utils/gcg.py: gradient-guided top-k candidate sampling followed by exact loss evaluation. What
    differs is that both the loss and the verdict are conjunctions over a set of models, not a
    contrast between two.

    `roles` holds every model in the run, the baseline included, so that the three loops this class
    needs — "must break", "must hold", "must be decoded" — are all filters over one list and cannot
    fall out of sync with each other.
    """

    def __init__(
        self,
        roles: list[ModelRole],
        tokenizer,
        cfg: SearchConfig,
        score_surrogates: bool = True,
    ):
        self.roles = roles
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.score_surrogates = score_surrogates
        self.device = roles[0].real.device
        self.not_allowed_ids = (
            None if cfg.allow_non_ascii else get_nonascii_toks(tokenizer, device=self.device)
        )
        # clean-prompt verdicts cached by capability_gate and reused as each task's control
        self._controls: dict[str, dict] = {}

    # ── role groups ──
    @property
    def baseline(self) -> ModelRole:
        """The pristine model. Exactly one role always has this position."""
        return next(role for role in self.roles if role.role == "baseline")

    @property
    def targets(self) -> list[ModelRole]:
        """The generations the suffix must break."""
        return [role for role in self.roles if role.is_target]

    @property
    def holds(self) -> list[ModelRole]:
        """The models the suffix must not break: the baseline and every spare generation.

        The baseline is a member rather than a special case — its condition is identical, it is only
        the one that is always present and never a surrogate.
        """
        return [role for role in self.roles if not role.is_target]

    @property
    def gating(self) -> list[ModelRole]:
        """The models whose clean-prompt verdict decides whether a task is attackable at all.

        The baseline and the targets, and deliberately *not* the spares. A target that already fails
        the clean prompt cannot be shown to have been broken by the suffix, and a baseline that
        already fails leaves nothing selective to demonstrate — both make the *claim* unavailable,
        so the task has to go. A spare that already fails is a different situation: it makes one
        conjunct hard to satisfy, but it does not make the claim about the targets unavailable, and
        which spares were already broken is a result worth measuring rather than a reason to refuse
        to measure it. It is reported instead of gating, per task and per model.
        """
        return [role for role in self.roles if role.role != "spare"]

    @property
    def transfer_mode(self) -> bool:
        """True when the objective is evaluated on surrogates instead of the real checkpoints."""
        return any(role.transfer for role in self.roles)

    # ── prompt construction ──
    def split_prompt(self, task: AttackTask) -> tuple[str, str]:
        """Renders the chat template and splits it at the adversarial-suffix slot."""
        return split_prompt(self.tokenizer, task)

    def _ids(self, text: str) -> Tensor:
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(self.device)

    def build_task_segments(self, task: AttackTask) -> dict:
        """Embeds the fixed prompt parts once per (model, target string) pair.

        Returns ``{role label: {"wrong": Segments, "correct": Segments | None}}``. Targets only need
        the wrong-code segments; holding models need both, and the correct-code ones only when the
        anchor term is switched on.

        Models that share an embedding matrix share the result. That is not a micro-optimization but
        the normal case in transfer mode: every ``logit`` surrogate embeds through the base model's
        table, so all of them would otherwise build byte-identical copies. Keyed on the storage
        pointer rather than on the model object, since the sharing is a property of the tensor.

        Args:
            task (AttackTask): the task whose prompt and reference implementations are embedded

        Returns:
            dict: per-role segments for the loss evaluation
        """
        before_str, after_str = self.split_prompt(task)
        before_ids = self._ids(before_str)
        after_ids = self._ids(after_str)
        target_ids = {
            "wrong": self._ids(task.wrong_code)[0],
            "correct": self._ids(task.correct_code)[0],
        }

        cache: dict[tuple[int, str], object] = {}
        segments: dict[str, dict] = {}
        for role in self.roles:
            needed = ["wrong"] if role.is_target else ["wrong", "correct"]
            if "correct" in needed and self.cfg.mu_correct <= 0:
                needed.remove("correct")
            entry = {}
            for which in needed:
                key = (role.optim.embed_weights.data_ptr(), which)
                if key not in cache:
                    cache[key] = role.optim.build_segments(
                        before_ids, after_ids, target_ids[which]
                    )
                entry[which] = cache[key]
            segments[role.label] = entry
        return segments

    # ── objective ──
    def objective(self, cand_ids: Tensor, segments: dict) -> dict:
        """Exact selective objective for a batch of candidate suffixes (lower is better).

        Every term is a mean over its group, so the weights and the margin keep the meaning they
        have in run_attack.py and a one-target/no-spare run reproduces its numbers exactly.

        Args:
            cand_ids (Tensor): shape (n_cand, n_optim)
            segments (dict): the output of build_task_segments

        Returns:
            dict: "total" plus the per-role component losses, each of shape (n_cand,)
        """
        wrong: dict[str, Tensor] = {}
        correct: dict[str, Tensor] = {}

        for role in self.roles:
            wrong[role.label] = role.optim.candidate_losses(
                cand_ids, segments[role.label]["wrong"], self.cfg.batch_size
            )
            if not role.is_target and self.cfg.mu_correct > 0:
                correct[role.label] = role.optim.candidate_losses(
                    cand_ids, segments[role.label]["correct"], self.cfg.batch_size
                )

        targets, holds = self.targets, self.holds
        total = torch.stack([wrong[role.label] for role in targets]).mean(dim=0)
        # the hinge only pushes a holding model away from the wrong code while it is still too
        # close to it; an unbounded ascent term makes the search diverge
        hinge = torch.stack(
            [torch.clamp(self.cfg.margin - wrong[role.label], min=0.0) for role in holds]
        ).mean(dim=0)
        total = total + self.cfg.lambda_base * hinge
        if correct:
            anchor = torch.stack([correct[role.label] for role in holds]).mean(dim=0)
            total = total + self.cfg.mu_correct * anchor

        return {"total": total, "wrong": wrong, "correct": correct}

    def combined_gradient(self, optim_ids: Tensor, segments: dict) -> tuple[Tensor, dict]:
        """Gradient of the selective objective w.r.t. the one-hot suffix matrix.

        One backward pass per (model, term), summed with the same weights the exact objective uses.
        The hinge is gated on the *measured* loss of that model, per model rather than globally: a
        spare that is already far from the wrong code contributes nothing, which is what keeps the
        gradient from fighting itself once part of the holding group is satisfied.

        Args:
            optim_ids (Tensor): shape (n_optim,), the current suffix tokens
            segments (dict): the output of build_task_segments

        Returns:
            tuple: (gradient of shape (n_optim, vocab), the per-role losses behind it)
        """
        targets, holds = self.targets, self.holds
        losses: dict[str, dict] = {}
        grad = None

        for role in targets:
            loss, gradient = role.optim.loss_and_grad(optim_ids, segments[role.label]["wrong"])
            losses[role.label] = {"wrong": loss, "correct": None}
            contribution = gradient / len(targets)
            grad = contribution if grad is None else grad + contribution

        for role in holds:
            loss, gradient = role.optim.loss_and_grad(optim_ids, segments[role.label]["wrong"])
            entry = {"wrong": loss, "correct": None}
            if loss < self.cfg.margin and self.cfg.lambda_base > 0:
                grad = grad - self.cfg.lambda_base * gradient / len(holds)
            if self.cfg.mu_correct > 0:
                anchor_loss, anchor_grad = role.optim.loss_and_grad(
                    optim_ids, segments[role.label]["correct"]
                )
                entry["correct"] = anchor_loss
                grad = grad + self.cfg.mu_correct * anchor_grad / len(holds)
            losses[role.label] = entry

        return grad, losses

    # ── verification ──
    def _decoded_models(self) -> list[tuple[str, TargetModel]]:
        """The (verdict key prefix, model) pairs the behavioural check decodes.

        The real models come first because they alone decide the verdict. A surrogate is added per
        role only for the agreement statistics, and only when scoring is on; distinct roles that
        resolved to the same factor share one surrogate object and are decoded once.
        """
        pairs = [(role.label, role.real) for role in self.roles]
        if not self.score_surrogates:
            return pairs
        seen: set[int] = set()
        for role in self.roles:
            if not role.transfer or id(role.optim) in seen:
                continue
            seen.add(id(role.optim))
            pairs.append((role.surrogate_label, role.optim))
        return pairs

    def verify(
        self, task: AttackTask, before_str: str, after_str: str, suffix: str
    ) -> dict[str, str]:
        """Decodes with every model and decides correctness behaviourally.

        The prompt is reassembled as a *string* and retokenized as a whole, so it is exactly what an
        attacker would send. The optimizer instead tokenizes the three segments separately and the
        two can disagree at the boundaries; when they do, this verdict is the authoritative one.

        A surrogate shared by several roles is decoded once and its status copied to each, so
        the per-generation statistics stay well defined without paying for duplicate decodes.

        Args:
            task (AttackTask): the task being attacked
            before_str (str): prompt text before the suffix slot
            after_str (str): prompt text after it
            suffix (str): the candidate adversarial suffix

        Returns:
            dict: ``{label}_raw``, ``{label}_code`` and ``{label}_status`` per decoded model
        """
        prompt = before_str + suffix + after_str
        result: dict[str, str] = {}
        for label, model in self._decoded_models():
            raw = model.complete(
                self.tokenizer, prompt, self.cfg.max_new_tokens, self.cfg.repetition_penalty
            )
            code = extract_code(raw)
            result[f"{label}_raw"] = raw
            result[f"{label}_code"] = code
            result[f"{label}_status"] = (
                "skipped" if self.cfg.no_exec else run_unit_tests(code, task, self.cfg.exec_timeout)
            )

        # roles sharing one surrogate object were decoded under whichever label came first; copy the
        # verdict onto the others so every role has its own key
        if self.score_surrogates:
            by_model: dict[int, str] = {}
            for role in self.roles:
                if not role.transfer:
                    continue
                key = id(role.optim)
                if key in by_model:
                    source = by_model[key]
                    for suffix_key in ("raw", "code", "status"):
                        result[f"{role.surrogate_label}_{suffix_key}"] = result[
                            f"{source}_{suffix_key}"
                        ]
                else:
                    by_model[key] = role.surrogate_label
        return result

    def statuses(self, verdict: dict[str, str]) -> dict[str, str]:
        """The real models' statuses out of a verdict, keyed by role label."""
        return {role.label: verdict.get(f"{role.label}_status", "unknown") for role in self.roles}

    def blocked_by(self, verdict: dict[str, str]) -> list[str]:
        """Labels of the models that did *not* do what their role demands.

        The empty list is exactly the success condition, which is why the criterion below is written
        in terms of this rather than the other way round: every non-hit carries its own reason.
        """
        statuses = self.statuses(verdict)
        return [role.label for role in self.roles if not role.required(statuses[role.label])]

    def is_selective_hit(self, verdict: dict[str, str]) -> bool:
        """True iff every target emitted wrong code and every other model emitted correct code.

        The conjunction is the whole point of this script: each condition on its own is satisfiable
        by an uninteresting suffix (one that breaks everything, or one that does nothing), and only
        their simultaneous satisfaction says the input discriminates between the models. The
        baseline's condition is part of it by construction, as a member of the holding group.

        Surrogates never enter, in either mode — the same rule as run_attack.py.
        """
        return not self.blocked_by(verdict)

    def surrogate_hit(self, verdict: dict[str, str]) -> bool | None:
        """The same conjunction evaluated on the *surrogates*, or None if it cannot be.

        This is what an attacker who never obtains the checkpoints would conclude from their own
        proxies. Scoring it against the real answer is the ensemble half of ``SurrogateReport``. The
        baseline is real in both readings and enters unchanged.
        """
        if not self.transfer_mode or not self.score_surrogates:
            return None
        for role in self.roles:
            key = f"{role.surrogate_label}_status" if role.transfer else f"{role.label}_status"
            status = verdict.get(key)
            if status is None:
                return None
            if not role.required(status):
                return False
        return True

    # ── upfront capability gate ──
    def capability_gate(self, tasks: list[AttackTask], min_capability: float) -> CapabilityReport:
        """Probes every model in the run on the clean, suffix-free prompts before optimizing.

        A task survives if the **baseline and every target** solve it unaided; the spares are probed
        and reported but never gate — see ``self.gating`` for why the two groups differ.
        ``--min_capability`` is likewise enforced on the targets alone.

        The verdicts are cached so ``run_task`` reuses them as its control instead of decoding the
        clean prompts a second time.

        Args:
            tasks (list[AttackTask]): the tasks selected on the CLI
            min_capability (float): fraction of tasks every target must solve, in [0, 1]

        Returns:
            CapabilityReport: per-model verdicts, per-model capability, and the abort decision
        """
        report = CapabilityReport(threshold=min_capability)
        report.solved = {role.label: [] for role in self.roles}
        report.broken = {role.label: [] for role in self.roles}
        report.surrogate_broken = {
            role.label: [] for role in self.roles if role.transfer
        }
        self._controls = {}

        if self.cfg.no_exec:
            # without execution there is no ground truth to gate on
            report.skipped = True
            report.reason = "--no_exec: capability cannot be established without running code"
            report.usable = [task.name for task in tasks]
            return report

        for task in tasks:
            # the tests must actually separate the two reference implementations, otherwise the
            # probe below measures nothing
            if run_unit_tests(task.correct_code, task, self.cfg.exec_timeout) != "pass":
                report.invalid_tasks.append(task.name)
                report.per_task[task.name] = {"invalid": "correct_code fails its own tests"}
                continue
            if run_unit_tests(task.wrong_code, task, self.cfg.exec_timeout) not in WRONG_STATUSES:
                report.invalid_tasks.append(task.name)
                report.per_task[task.name] = {"invalid": "wrong_code passes the tests"}
                continue

            before_str, after_str = self.split_prompt(task)
            verdict = self.verify(task, before_str, after_str, "")
            self._controls[task.name] = verdict
            report.per_task[task.name] = verdict
            report.n_probed += 1

            statuses = self.statuses(verdict)
            for role in self.roles:
                if statuses[role.label] == "pass":
                    report.solved[role.label].append(task.name)
                else:
                    report.broken[role.label].append(task.name)
                # informational only: whether the proxy can write code says nothing about whether
                # it is a usable search target, and it is not the model under attack
                if role.transfer:
                    surrogate = verdict.get(f"{role.surrogate_label}_status")
                    if surrogate is not None and surrogate != "pass":
                        report.surrogate_broken[role.label].append(task.name)

            failed = [
                role.label for role in self.gating if statuses[role.label] != "pass"
            ]
            # spares are not part of that decision, but a spare that already fails this task can
            # never satisfy "must hold" on it either, so the impossibility is recorded where it can
            # be read off later instead of silently dropping the task
            spares_broken = [
                role.label for role in self.roles
                if role.role == "spare" and statuses[role.label] != "pass"
            ]
            if not failed:
                report.usable.append(task.name)
                if spares_broken:
                    report.spare_broken_on_usable[task.name] = spares_broken

            if failed:
                marker = (
                    f"{TColors.FAIL}unusable{TColors.ENDC} ({', '.join(failed)} cannot solve it)"
                )
            elif spares_broken:
                marker = (
                    f"{TColors.OKGREEN}usable{TColors.ENDC} "
                    f"{TColors.WARNING}(but {', '.join(spares_broken)} already fail(s) it, so no "
                    f"hit on this task can satisfy them){TColors.ENDC}"
                )
            else:
                marker = f"{TColors.OKGREEN}usable{TColors.ENDC}"
            columns = "  ".join(f"{label}={statuses[label]}" for label in statuses)
            print(f"##   {task.name:16s} {columns} -> {marker}")

        if report.n_probed == 0:
            report.aborted = True
            report.reason = (
                "no task survived reference validation, so no model was ever probed"
            )
            return report

        for role in self.roles:
            report.capability[role.label] = len(report.solved[role.label]) / report.n_probed

        incapable = [
            role.label
            for role in self.targets
            if report.capability[role.label] < min_capability
        ]
        if incapable:
            report.aborted = True
            details = ", ".join(
                f"{label} {len(report.solved[label])}/{report.n_probed} "
                f"({report.capability[label]:.0%})"
                for label in incapable
            )
            report.reason = (
                f"target model(s) below the required {min_capability:.0%} on the clean prompts: "
                f"{details} — they are no longer capable of generating correct code, so wrong "
                f"output under a suffix cannot be attributed to the attack"
            )
        elif not report.usable:
            blockers = ", ".join(
                f"{role.label} ({len(report.broken[role.label])}/{report.n_probed} failed)"
                for role in self.gating
                if report.broken[role.label]
            )
            report.reason = (
                f"no task is attackable: every probed task is one the baseline or a target model "
                f"cannot solve unaided. Failures per gating model: {blockers or 'none'}"
            )
            report.aborted = True
        return report

    # ── reporting ──
    def selectivity_report(self, outcomes: list[TaskOutcome]) -> SelectivityReport:
        """Aggregates the per-model behaviour over every behavioural check of the run.

        Args:
            outcomes (list[TaskOutcome]): the per-task results of the search

        Returns:
            SelectivityReport: where the conjunction held and where it broke down
        """
        report = SelectivityReport()
        labels = [role.label for role in self.roles]
        report.n_condition_met = {label: 0 for label in labels}
        report.n_wrong = {label: 0 for label in labels}
        report.blocked_by = {label: 0 for label in labels}
        target_labels = {role.label for role in self.targets}
        hold_labels = {role.label for role in self.holds}

        for outcome in outcomes:
            counts = {
                "n_verified": 0,
                "n_success": 0,
                "n_targets_all_broken": 0,
                "n_holds_all_kept": 0,
                "n_baseline_broken": 0,
            }
            for record in outcome.verifications:
                statuses = record.get("statuses", {})
                blocked = set(record.get("blocked_by", []))
                counts["n_verified"] += 1
                if not blocked:
                    counts["n_success"] += 1
                if not blocked & target_labels:
                    counts["n_targets_all_broken"] += 1
                if not blocked & hold_labels:
                    counts["n_holds_all_kept"] += 1
                if self.baseline.label in blocked:
                    counts["n_baseline_broken"] += 1
                for label in labels:
                    # a label in `blocked` is by definition the reason this check was not a hit,
                    # so the two branches partition every check exactly once
                    if label in blocked:
                        report.blocked_by[label] += 1
                    else:
                        report.n_condition_met[label] += 1
                    if statuses.get(label) in WRONG_STATUSES:
                        report.n_wrong[label] += 1

            if counts["n_verified"]:
                report.per_task[outcome.task] = counts
                for key, value in counts.items():
                    setattr(report, key, getattr(report, key) + value)
        return report

    def surrogate_report(self, outcomes: list[TaskOutcome]) -> SurrogateReport:
        """Scores the surrogates, per generation and as an ensemble.

        Args:
            outcomes (list[TaskOutcome]): the per-task results of the search

        Returns:
            SurrogateReport: per-generation agreement plus ensemble precision and recall
        """
        report = SurrogateReport()
        transfer_roles = [role for role in self.roles if role.transfer]
        report.per_generation = {
            role.label: {"n": 0, "agreed": 0, "surrogate_wrong": 0, "real_wrong": 0}
            for role in transfer_roles
        }
        report.factors = {
            role.label: role.factor for role in transfer_roles if role.factor is not None
        }
        report.models = {role.label: role.description for role in transfer_roles}

        for outcome in outcomes:
            for record in outcome.verifications:
                statuses = record.get("statuses", {})
                surrogate_statuses = record.get("surrogate_statuses", {})
                if not surrogate_statuses:
                    continue
                report.n_verified += 1
                for role in transfer_roles:
                    real = statuses.get(role.label)
                    proxy = surrogate_statuses.get(role.label)
                    if real is None or proxy is None:
                        continue
                    bucket = report.per_generation[role.label]
                    bucket["n"] += 1
                    real_wrong = real in WRONG_STATUSES
                    proxy_wrong = proxy in WRONG_STATUSES
                    bucket["real_wrong"] += int(real_wrong)
                    bucket["surrogate_wrong"] += int(proxy_wrong)
                    bucket["agreed"] += int(real_wrong == proxy_wrong)

                predicted = record.get("surrogate_hit")
                actual = not record.get("blocked_by")
                if predicted is None:
                    continue
                if predicted:
                    report.n_predicted_hit += 1
                    if actual:
                        report.n_true_positive += 1
                    else:
                        report.n_false_alarm += 1
                elif actual:
                    report.n_missed += 1

        for label, bucket in report.per_generation.items():
            report.agreement[label] = bucket["agreed"] / bucket["n"] if bucket["n"] else 0.0
        return report

    # ── main loop ──
    def run_task(self, task: AttackTask, restarts: int) -> TaskOutcome:
        """Runs the full search (all restarts) for one task.

        Assumes ``capability_gate`` has already vetted the task; its cached clean-prompt verdict is
        reused as this task's control.

        Args:
            task (AttackTask): the task to attack
            restarts (int): random re-initializations of the suffix

        Returns:
            TaskOutcome: hits, every behavioural check, and the loss trajectory
        """
        outcome = TaskOutcome(task=task.name)
        before_str, after_str = self.split_prompt(task)

        outcome.control = self._controls.get(task.name, {})
        if not outcome.control and not self.cfg.no_exec:
            # defensive: run_task called without a preceding gate
            outcome.control = self.verify(task, before_str, after_str, "")
        if outcome.control:
            statuses = self.statuses(outcome.control)
            print(
                "##   control (no suffix): "
                + "  ".join(f"{label}={status}" for label, status in statuses.items())
            )
            if not self.cfg.no_exec:
                # only the gating models can exclude a task: a target that already fails cannot be
                # shown to have been broken by the suffix, and a baseline that already fails leaves
                # nothing selective to demonstrate. A spare that already fails is searched anyway —
                # it simply keeps blocking the conjunction, which the selectivity report counts
                failed = [
                    role.label for role in self.gating if statuses[role.label] != "pass"
                ]
                if failed:
                    outcome.skipped = (
                        f"{', '.join(failed)} do(es) not solve this task without an adversarial "
                        f"input, so no claim about the targets is available on it"
                    )
                    return outcome
                already_broken = [
                    role.label for role in self.roles
                    if role.role == "spare" and statuses[role.label] != "pass"
                ]
                if already_broken:
                    print(
                        f"##   {TColors.WARNING}{', '.join(already_broken)} already fail(s) this "
                        f"task unaided{TColors.ENDC}, so no suffix can satisfy them here — "
                        f"searching anyway, the blocked_by counts will show it"
                    )

        segments = self.build_task_segments(task)
        rng = random.Random(self.cfg.seed)
        n_init_tokens = len(self.cfg.optim_str_init.split())

        for restart in range(restarts):
            if restart == 0:
                init_str = self.cfg.optim_str_init
            else:
                init_str = " ".join(rng.choice(INIT_CHARS) for _ in range(n_init_tokens))
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}[{task.name}] restart "
                f"{restart + 1}/{restarts}{TColors.ENDC} init={init_str!r}"
            )
            self._run_restart(task, before_str, after_str, segments, init_str, restart, outcome)
            if outcome.successes and self.cfg.stop_on_success:
                break

        return outcome

    def _run_restart(
        self,
        task: AttackTask,
        before_str: str,
        after_str: str,
        segments: dict,
        init_str: str,
        restart: int,
        outcome: TaskOutcome,
    ) -> None:
        """One gradient-guided search trajectory from a single initialization."""
        optim_ids = self._ids(init_str)[0]
        progress = tqdm(range(self.cfg.num_steps), desc=f"{task.name}/r{restart}", leave=False)

        for step in progress:
            grad, _ = self.combined_gradient(optim_ids, segments)

            with torch.no_grad():
                # sample_ids_from_grad writes +inf into the grad it is given -> pass a copy
                cand_ids = sample_ids_from_grad(
                    optim_ids,
                    grad.clone(),
                    self.cfg.search_width,
                    self.cfg.topk,
                    self.cfg.n_replace,
                    self.not_allowed_ids,
                )
                if self.cfg.filter_ids:
                    try:
                        cand_ids = filter_ids(cand_ids, self.tokenizer)
                    except RuntimeError:
                        # every candidate changed under retokenization; keep them unfiltered
                        pass

                scores = self.objective(cand_ids, segments)
                best = scores["total"].argmin()
                optim_ids = cand_ids[best]

                total = scores["total"][best].item()
                # the component losses of the winning candidate, per model, so the trajectory says
                # which condition the optimizer was trading off against which
                components = {
                    label: {
                        "wrong": scores["wrong"][label][best].item(),
                        "correct": (
                            scores["correct"][label][best].item()
                            if label in scores["correct"] else None
                        ),
                    }
                    for label in scores["wrong"]
                }

            suffix = self.tokenizer.decode(optim_ids, skip_special_tokens=True)
            outcome.history.append(
                {
                    "restart": restart,
                    "step": step,
                    "total": total,
                    "losses": components,
                    "suffix": suffix,
                }
            )
            if outcome.best_objective is None or total < outcome.best_objective:
                outcome.best_objective = total
                outcome.best_suffix = suffix

            target_mean = sum(
                components[role.label]["wrong"] for role in self.targets
            ) / len(self.targets)
            hold_mean = sum(components[role.label]["wrong"] for role in self.holds) / len(
                self.holds
            )
            progress.set_postfix(
                total=f"{total:.3f}", brk=f"{target_mean:.3f}", hold=f"{hold_mean:.3f}"
            )

            # behavioural check — the loss is only a proxy for the real attack goal
            due = (step + 1) % self.cfg.verify_every == 0 or step == self.cfg.num_steps - 1
            if due and not self.cfg.no_exec:
                verdict = self.verify(task, before_str, after_str, suffix)
                statuses = self.statuses(verdict)
                blocked = self.blocked_by(verdict)
                predicted = self.surrogate_hit(verdict)

                # every check is recorded, not only the hits: in a selective run the near misses
                # carry the information, and the per-model statistics are recomputed from these
                record = {
                    "restart": restart,
                    "step": step,
                    "suffix": suffix,
                    "statuses": statuses,
                    "blocked_by": blocked,
                }
                if self.transfer_mode and self.score_surrogates:
                    record["surrogate_statuses"] = {
                        role.label: verdict.get(f"{role.surrogate_label}_status")
                        for role in self.roles
                        if role.transfer
                    }
                    record["surrogate_hit"] = predicted
                outcome.verifications.append(record)

                summary = "  ".join(f"{label}={status}" for label, status in statuses.items())
                if not blocked:
                    outcome.successes.append(
                        {
                            "restart": restart,
                            "step": step,
                            "suffix": suffix,
                            "total": total,
                            "losses": components,
                            **verdict,
                        }
                    )
                    proxy_note = (
                        "" if predicted is None
                        else f" (surrogates predicted: {predicted})"
                    )
                    progress.write(
                        f"## {TColors.OKGREEN}{TColors.BOLD}SELECTIVE HIT{TColors.ENDC} "
                        f"[{task.name}] step {step}: {summary}{proxy_note} | suffix={suffix!r}"
                    )
                    if self.cfg.stop_on_success:
                        break
                elif self.baseline.label in blocked:
                    # the suffix cost the pristine model its correct answer. Worth separating: an
                    # objectively wrong answer means the suffix is a plain jailbreak, while an
                    # error or a timeout means it merely derailed the decoding — neither is a hit,
                    # but only the first one is a failure of *selectivity*
                    base_status = statuses[self.baseline.label]
                    note = (
                        "leaked to the baseline" if base_status in WRONG_STATUSES
                        else f"baseline no longer passes ({base_status})"
                    )
                    progress.write(
                        f"## {TColors.WARNING}{note}{TColors.ENDC} "
                        f"[{task.name}] step {step}: {summary}"
                    )
                elif len(blocked) == 1:
                    # one model short of a hit: worth seeing live, since it says which condition
                    # the search is failing to satisfy
                    progress.write(
                        f"## {TColors.OKCYAN}one model short{TColors.ENDC} [{task.name}] step "
                        f"{step}: {blocked[0]} did not {self._role(blocked[0]).verb} "
                        f"| {summary}"
                    )

        progress.close()

    def _role(self, label: str) -> ModelRole:
        """The role carrying `label`."""
        return next(role for role in self.roles if role.label == label)


# ──────────────────────── factor resolution and surrogate building ────────────────────────
def default_factors(generations: list[int]) -> dict[int, float]:
    """``n = g + 1`` per generation, the indexing convention the whole pipeline shares.

    model_0 is one fine-tuning step away from the base model, so model_g sits g + 1 steps out. Same
    rule as run_extrapolation.py and run_attack.py, which is what makes a surrogate built here and a
    dataset generated there describe the same generation.

    Args:
        generations (list[int]): the generation indices in the run

    Returns:
        dict[int, float]: generation -> factor
    """
    return {generation: float(generation + 1) for generation in generations}


def calibrated_factors(
    generations: list[int], calibration_path: str
) -> dict[int, float]:
    """Each generation's factor as fitted by ``utils/evaluate_perplexity.py --calibrate``.

    Measured for the generations the calibration covers, predicted by its fitted law for the rest —
    which is the case that matters, since the generations under attack are the ones nobody has a
    checkpoint of.

    Args:
        generations (list[int]): the generation indices in the run
        calibration_path (str): the calibration JSON

    Returns:
        dict[int, float]: generation -> factor

    Raises:
        SystemExit: the calibration does not exist
    """
    if not os.path.isfile(calibration_path):
        raise SystemExit(
            f"{TColors.FAIL}--surrogate_factor calibrated needs a calibration{TColors.ENDC} and "
            f"{calibration_path} does not exist. Produce it with:\n"
            f"  python -m utils.evaluate_perplexity --calibrate ..."
        )
    with open(calibration_path, encoding="utf-8") as handle:
        calibration = json.load(handle)
    return {
        generation: float(calibrated_factor(calibration, generation))
        for generation in generations
    }


def probe_factors(
    generations: list[int],
    method: str,
    baseline: TargetModel,
    first_collapsed_dir: str,
    surrogate_model_path: str,
    tasks: list[AttackTask],
    tokenizer,
    cfg: SearchConfig,
    min_capability: float,
    device: torch.device,
    dtype: torch.dtype,
    base_specifier: str,
) -> tuple[dict[int, float], list[dict], TargetModel | None]:
    """``--surrogate_factor auto`` for a whole set of generations, in one walk of the ladder.

    The hazard this addresses is run_attack.py's: ``n = g + 1`` names the generation being
    approximated but says nothing about whether the surrogate survives being tilted that far. Past
    some factor the tilt stops emitting valid code at all, and the objective's "must break" term
    is then satisfied by the clean prompt and nothing pushes the suffix toward inputs that break a
    *working* model.

    run_attack.py answers that with one descending walk capped at its single generation's ``g + 1``.
    Here there are several caps, and the answer is still one walk: capability at a given rung does
    not depend on which generation asked, so the ladder is walked once over the union of the rungs
    and every generation takes the largest capable rung at or below its cap. The walk stops as soon
    as every generation has an answer, so rungs below the smallest cap are never probed — the same
    work run_attack.py's probe does, without repeating it per generation.

    The threat model is untouched: only the base model and the generation-0 checkpoint are ever run,
    and the outcome decides the search proxies only. Success is still what the real checkpoints do.

    Args:
        generations (list[int]): the generation indices needing a factor
        method (str): "logit" or "lora"
        baseline (TargetModel): the loaded pristine base model
        first_collapsed_dir (str): directory of the generation-0 collapsed model
        surrogate_model_path (str): prebuilt surrogate path, forwarded to build_surrogate
        tasks (list[AttackTask]): the tasks selected on the CLI
        tokenizer: tokenizer shared by the models
        cfg (SearchConfig): supplies the decoding budget, repetition penalty and exec timeout
        min_capability (float): fraction of tasks the surrogate must solve at a rung
        device (torch.device): device to build on
        dtype (torch.dtype): dtype to load with
        base_specifier (str): base model id, for adapter loading

    Returns:
        tuple: (generation -> chosen factor, one report row per probed rung, the surrogate left
            loaded by the walk). The surrogate is handed back so build_role_surrogates can reuse its
            generation-0 model instead of loading a second copy

    Raises:
        SystemExit: some generation found no capable rung at or below its own cap
    """
    caps = default_factors(generations)
    if cfg.no_exec:
        # same reasoning as the capability gate's --no_exec branch: with no ground truth there is
        # nothing to select on, so auto degrades to the value it would have replaced
        print(
            f"##   {TColors.WARNING}--no_exec: cannot probe, falling back to n = g + 1"
            f"{TColors.ENDC}"
        )
        return caps, [], None

    # the union of every generation's ladder: the shared rungs plus each cap, which is not
    # necessarily a rung itself (generation 6 caps at 7.0, and 7.0 is not on the ladder)
    rungs = sorted(
        {factor for cap in caps.values() for factor in factor_ladder(cap)}, reverse=True
    )
    print(f"##   ladder: {', '.join(f'{rung:g}' for rung in rungs)}")

    prompts = [(task, "".join(split_prompt(tokenizer, task))) for task in tasks]
    unresolved = dict(caps)
    chosen: dict[int, float] = {}
    rows: list[dict] = []
    surrogate: TargetModel | None = None

    for rung in rungs:
        if not unresolved:
            break
        # a rung above every remaining cap cannot be the answer for any of them
        if rung > max(unresolved.values()):
            continue
        if isinstance(surrogate, ExtrapolatedModel):
            # the logit surrogate *is* the tilt, so the next rung is a different float on the same
            # two loaded models — no reload, which is what makes the walk cheap
            surrogate.factor = float(rung)
        else:
            # the lora surrogate is real weights, so every rung is a fresh scaled adapter. The
            # previous one is dropped here, in the scope holding the only reference to it: at the
            # larger model sizes two of them do not fit
            if surrogate is not None:
                del surrogate
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            surrogate, _ = build_surrogate(
                method, rung, baseline, first_collapsed_dir, surrogate_model_path, device, dtype,
                base_specifier,
            )

        per_task = {}
        for task, prompt in prompts:
            raw = surrogate.complete(
                tokenizer, prompt, cfg.max_new_tokens, cfg.repetition_penalty
            )
            per_task[task.name] = run_unit_tests(extract_code(raw), task, cfg.exec_timeout)
        solved = [name for name, status in per_task.items() if status == "pass"]
        capability = len(solved) / len(prompts)
        # solved > 0 on top of the threshold, deliberately: at --min_capability 0 the threshold
        # alone would accept the first rung however broken, which is the failure this probe exists
        # to avoid
        accepted = bool(solved) and capability >= min_capability
        rows.append(
            {
                "factor": rung,
                "solved": len(solved),
                "probed": len(prompts),
                "capability": capability,
                "accepted": accepted,
                "per_task": per_task,
            }
        )

        taken = []
        if accepted:
            taken = [
                generation for generation, cap in unresolved.items() if cap >= rung
            ]
            for generation in taken:
                chosen[generation] = float(rung)
                del unresolved[generation]

        marker = (
            f"{TColors.OKGREEN}accepted{TColors.ENDC}" if accepted
            else f"{TColors.FAIL}too collapsed{TColors.ENDC}"
        )
        note = (
            f" -> generation(s) {', '.join(str(g) for g in sorted(taken))}" if taken else ""
        )
        print(
            f"##   n = {rung:<5g} surrogate solves {len(solved)}/{len(prompts)} "
            f"({capability:.0%}) -> {marker}{note}"
        )

    if unresolved:
        raise SystemExit(
            f"{TColors.FAIL}no extrapolation factor produced a usable surrogate for generation(s) "
            f"{', '.join(str(g) for g in sorted(unresolved))}{TColors.ENDC}: at every n from their "
            f"own g + 1 down to 1.0 the surrogate solved fewer than {min_capability:.0%} of the "
            f"clean tasks (n = 1 is the generation-0 checkpoint itself).\nThat is a "
            f"statement about the anchor, not about the factor: {first_collapsed_dir} has already "
            f"lost code generation, so nothing built from it can stand in for a model that still "
            f"writes code. Attack earlier generations, collapse with a larger --real_data_fraction "
            f"so the anchor stays capable, or drop to --surrogate_method none and attack the real "
            f"checkpoints directly."
        )

    # only the logit surrogate is worth handing back: its generation-0 model is shared by every
    # factor, so the caller reuses it. A lora surrogate is weights at one specific factor and would
    # merely sit in VRAM beside the ones the caller is about to build, so it is dropped here
    if isinstance(surrogate, ExtrapolatedModel):
        return chosen, rows, surrogate
    del surrogate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return chosen, rows, None


def build_role_surrogates(
    factors: dict[int, float],
    method: str,
    baseline: TargetModel,
    first_collapsed_dir: str,
    surrogate_model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    base_specifier: str,
    probed: TargetModel | None = None,
    first_model=None,
) -> tuple[dict[float, TargetModel], dict[float, str]]:
    """One surrogate per *distinct* factor, keyed by it.

    Distinct rather than per generation, because two generations can legitimately land on the same
    factor (``auto`` or ``calibrated`` can land two caps on one rung) and building the same model
    twice would double the VRAM for nothing. Roles sharing a factor then share one object, which the
    verification exploits by decoding it once.

    For ``logit`` this is where the whole set becomes affordable: every surrogate is the same two
    models with a different float, so N generations cost the base model (already loaded) plus one
    copy of the generation-0 checkpoint, whatever N is. Each ``ExtrapolatedModel`` wraps the shared
    generation-0 module in its own thin ``TargetModel``, which allocates nothing.

    ``lora`` has no such structure: each factor is a separate set of merged weights, so the VRAM
    scales with the number of distinct factors and a wide generation list will not fit.

    Args:
        factors (dict[int, float]): generation -> factor, from the resolution above
        method (str): "logit" or "lora"
        baseline (TargetModel): the loaded pristine base model
        first_collapsed_dir (str): directory of the generation-0 collapsed model
        surrogate_model_path (str): prebuilt surrogate path, forwarded to build_surrogate
        device (torch.device): device to build on
        dtype (torch.dtype): dtype to load with
        base_specifier (str): base model id, for adapter loading
        probed (TargetModel | None): the surrogate the auto probe left behind, whose loaded
            generation-0 model is reused instead of loading a second copy
        first_model: an already loaded generation-0 model to build the tilt on. Set when gen 0
            is itself one of the run's roles and resolves to the same directory as the anchor, which
            makes the anchor a checkpoint that is already in VRAM

    Returns:
        tuple: (factor -> surrogate, factor -> description)
    """
    distinct = sorted(set(factors.values()))
    surrogates: dict[float, TargetModel] = {}
    descriptions: dict[float, str] = {}

    if method == "logit":
        # the anchor is loaded once and shared by every factor
        if isinstance(probed, ExtrapolatedModel):
            first_model = probed.first.model
        elif first_model is None:
            first_model = load_model(
                first_collapsed_dir, device, dtype, base_for_adapter=base_specifier
            )
        for factor in distinct:
            surrogates[factor] = ExtrapolatedModel(
                "surrogate", baseline.model, first_model, factor, device
            )
            descriptions[factor] = f"base + {factor:g} * ({first_collapsed_dir} - base)"
        return surrogates, descriptions

    for factor in distinct:
        surrogate, description = build_surrogate(
            method, factor, baseline, first_collapsed_dir, surrogate_model_path, device, dtype,
            base_specifier,
        )
        surrogates[factor] = surrogate
        descriptions[factor] = description
    return surrogates, descriptions


def check_factor_collisions(
    targets: list[int], spares: list[int], factors: dict[int, float]
) -> None:
    """Rejects a factor assignment that makes the objective self-contradictory.

    In transfer mode a generation is present only through its surrogate, so two of them sharing a
    factor are *the same model* as far as the objective and the surrogate scoring are concerned.
    Across the target/spare divide that is a contradiction — one model asked to break and to hold at
    the same time — and it has to stop the run rather than produce a suffix optimized against an
    unsatisfiable loss. Within one group it is merely redundant (the term is counted twice with half
    the weight each), so it warns.

    Args:
        targets (list[int]): the generations that must break
        spares (list[int]): the generations that must hold
        factors (dict[int, float]): generation -> factor

    Returns:
        None

    Raises:
        SystemExit: a target and a spare resolved to the same factor
    """
    clashes = [
        (target, spare, factors[target])
        for target in targets
        for spare in spares
        if factors[target] == factors[spare]
    ]
    if clashes:
        lines = "\n".join(
            f"  generation {target} (must break) and generation {spare} (must hold) both resolve "
            f"to n = {factor:g}"
            for target, spare, factor in clashes
        )
        raise SystemExit(
            f"{TColors.FAIL}the surrogates of a target and a spare generation are the same model"
            f"{TColors.ENDC}\n{lines}\nIn transfer mode a generation enters the objective only "
            f"through its surrogate, so this asks one model to emit wrong and correct code for "
            f"the same prompt. Choose generations whose factors differ, pass explicit factors, or "
            f"attack the real checkpoints with --surrogate_method none."
        )

    for group, name in ((targets, "target"), (spares, "spare")):
        seen: dict[float, int] = {}
        for generation in group:
            factor = factors[generation]
            if factor in seen:
                print(
                    f"## {TColors.WARNING}generations {seen[factor]} and {generation} ({name}s) "
                    f"both resolve to n = {factor:g}{TColors.ENDC}, so their surrogates are the "
                    f"same model and the term is counted twice"
                )
            else:
                seen[factor] = generation


def _hr(offset: int = 0) -> str:
    """Terminal-width rule that also works when stdout is redirected."""
    return run_attack._hr(offset)  # pylint: disable=protected-access


def result_file(
    targets: list[int],
    spares: list[int],
    specifier_name: str,
    tag: str,
    method: str,
) -> str:
    """Path of the result JSON, named after the run's own question.

    Both generation lists go into the name: a run against generations 8 and 9 sparing 0 and 1
    is a different experiment from one sparing 2 and 3, and neither may overwrite the other. Lists
    are sorted by ``generation_list_arg``, so the name is canonical whatever order they were typed.
    ``+`` joins the indices: filesystem safe everywhere and not confusable with a ``-`` range.

    The mixture tag sits where it does in the checkpoint names, and is empty at ``-rdf 0`` for the
    same reason.

    Args:
        targets (list[int]): the generations that must break
        spares (list[int]): the generations that must hold
        specifier_name (str): the model short name
        tag (str): the mixture tag
        method (str): the surrogate method, "none" outside transfer mode

    Returns:
        str: the output path
    """
    break_part = "+".join(str(generation) for generation in targets)
    spare_part = "+".join(str(generation) for generation in spares) or "none"
    surrogate_part = f"_{method}_surrogate" if method != "none" else ""
    return os.path.join(
        RESULTS_PATH,
        f"selective_attack_break{break_part}_spare{spare_part}_{specifier_name}"
        f"{tag}{surrogate_part}.json",
    )


def main(
    device: str = "cuda",
    target_generations: list[int] | None = None,
    spare_generations: list[int] | None = None,
    block_size: int | None = None,
    model_specifier: str = "",
    model_size: str = "",
    baseline_model_path: str = "",
    generation_model_path: dict[int, str] | None = None,
    path: str = "",
    tasks: str = "",
    restarts: int = 3,
    num_steps: int = 250,
    search_width: int = 256,
    batch_size: int = 32,
    topk: int = 256,
    n_replace: int = 1,
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x",
    allow_non_ascii: bool = False,
    lambda_base: float = 1.0,
    margin: float = 3.0,
    mu_correct: float = 0.5,
    verify_every: int = 10,
    max_new_tokens: int = 96,
    repetition_penalty: float = 1.0,
    exec_timeout: float = 10.0,
    no_exec: bool = False,
    stop_on_success: bool = False,
    min_capability: float = 0.2,
    skip_capability_check: bool = False,
    seed: int = 1337,
    list_tasks: bool = False,
    surrogate_method: str = "none",
    surrogate_factor: float | str = 0.0,
    surrogate_model_path: str = "",
    first_collapsed_path: str = "",
    no_surrogate_scoring: bool = False,
    real_data_fraction: float = 0.0,
) -> None:
    """Searches for an adversarial suffix that is selective across a set of collapse generations.

    Args:
        device (str): device to run the computations on (cuda recommended)
        target_generations (list[int]): generations the suffix must break
        spare_generations (list[int]): generations the suffix must not break
        block_size (int | None): effective block size in the checkpoint names; auto-detected
        model_specifier (str): base/baseline model specifier
        model_size (str): parameter count off the Qwen2.5-Coder ladder ("0.5b" ... "32b"),
            shorthand for the matching model_specifier. Must resolve to the model the collapse run
            was trained from, since its short name is part of the checkpoint paths
        baseline_model_path (str): explicit override for the baseline model
        generation_model_path (dict[int, str]): explicit checkpoint per generation, overriding the
            name-based resolution for those generations only
        path (str): root directory containing model_outputs/
        tasks (str): comma-separated task names to attack (default: all)
        restarts (int): random re-initializations of the suffix per task
        num_steps (int): optimizer steps per restart
        search_width (int): candidate suffixes sampled per step
        batch_size (int): candidates scored per forward pass (halved automatically on OOM)
        topk (int): top-k tokens per position considered from the gradient
        n_replace (int): suffix positions mutated per candidate
        optim_str_init (str): initial suffix for the first restart
        allow_non_ascii (bool): allow non-ASCII tokens in the suffix
        lambda_base (float): weight of the must-not-break hinge, over the holding group
        margin (float): loss margin the holding models must keep from the wrong code
        mu_correct (float): weight of the stays-correct anchor, over the holding group
        verify_every (int): run the behavioural check every N steps
        max_new_tokens (int): decoding budget during verification
        repetition_penalty (float): decoding repetition penalty during verification
        exec_timeout (float): per-candidate unit-test timeout in seconds
        no_exec (bool): never execute generated code (disables behavioural verification)
        stop_on_success (bool): stop a task as soon as a selective hit is verified
        min_capability (float): fraction of clean tasks every *target* must still solve before the
            attack is allowed to start
        skip_capability_check (bool): do not abort the run when the capability gate fails
        seed (int): RNG seed
        list_tasks (bool): print the available tasks and exit
        surrogate_method (str): "none" attacks the real checkpoints directly. "logit" or "lora"
            enables transfer mode: every generation in either list is replaced in the objective by a
            surrogate built from the base and generation-0 models only
        surrogate_factor (float | str): how each generation's extrapolation factor is chosen. 0.0
            derives n = g + 1 per generation, "auto" measures them, "calibrated" reads them from the
            perplexity calibration. An explicit number is only accepted for a single-generation run
        surrogate_model_path (str): a prebuilt surrogate instead of building one ("lora" only,
            and only meaningful for a single distinct factor)
        first_collapsed_path (str): explicit path to the generation-0 collapsed model the surrogates
            are built from (default: resolved from the model outputs)
        no_surrogate_scoring (bool): do not decode the surrogates during verification. Halves the
            verification cost in transfer mode at the price of the surrogate report
        real_data_fraction (float): the --real_data_fraction run_baseline.py was given; it is part
            of the checkpoint names from generation 1 onward. Only used to find them

    Returns:
        None

    Raises:
        SystemExit: the generation lists are empty, overlap, or are inconsistent with the surrogate
            settings; or no task is attackable
    """
    if list_tasks:
        for task in TASKS:
            print(f"{task.name:16s} {task.func}(...)  wrong: {task.wrong_code.strip()!r}")
        return

    targets = sorted(target_generations or [])
    spares = sorted(spare_generations or [])
    overrides = generation_model_path or {}

    if not targets:
        raise SystemExit(
            f"{TColors.FAIL}--target_generations is empty{TColors.ENDC}: with nothing that has to "
            f"break there is no attack to search for. Pass at least one generation, e.g. "
            f"--target_generations 9"
        )
    both = sorted(set(targets) & set(spares))
    if both:
        raise SystemExit(
            f"{TColors.FAIL}generation(s) {', '.join(str(g) for g in both)} are in both lists"
            f"{TColors.ENDC}: a model cannot be required to break and to hold at the same time"
        )

    start_time = time.time()
    torch.manual_seed(seed)
    random.seed(seed)

    # ──────────────────────────── set devices and paths ─────────────────────────
    if "cuda" in device and torch.cuda.is_available():
        index = device.split(":")[-1]
        torch_device = torch.device("cuda", int(index) if index.isdigit() else 0)
    elif "mps" in device and torch.backends.mps.is_available():
        torch_device = torch.device("mps", 0)
    else:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device}"
            f"{TColors.ENDC} is not available. Setting device to CPU instead. "
            f"The search will be extremely slow."
        )
        torch_device = torch.device("cpu", 0)

    global MODEL_PATH, RESULTS_PATH
    if path != "":
        MODEL_PATH = os.path.join(path, "model_outputs/")
        RESULTS_PATH = os.path.join(path, "attack_results/")
    # resolve_collapsed_dir and build_surrogate read the root out of run_attack's module scope, the
    # same rebinding run_attack.main() and utils/verify_transfer.py do
    run_attack.MODEL_PATH = MODEL_PATH
    os.makedirs(RESULTS_PATH, exist_ok=True)

    model_specifier = resolve_model_specifier(model_size, model_specifier)
    specifier_name = model_specifier.split("/")[-1]
    size_label = model_size_label(model_specifier) or "outside the --model_size ladder"
    baseline_dir = baseline_model_path or model_specifier

    generations = sorted(set(targets) | set(spares))
    checkpoints = {
        generation: overrides.get(generation)
        or resolve_collapsed_dir(
            generation, specifier_name, block_size, real_data_fraction=real_data_fraction
        )
        for generation in generations
    }

    # ── transfer mode setup ──
    transfer = surrogate_method != "none"
    auto_factor = surrogate_factor == "auto"
    calibrated = surrogate_factor == "calibrated"
    explicit = (
        not auto_factor and not calibrated and float(surrogate_factor or 0.0) > 0
    )
    if transfer and explicit and len(generations) > 1:
        raise SystemExit(
            f"{TColors.FAIL}--surrogate_factor takes a number only for a single-generation run"
            f"{TColors.ENDC}\nThis run involves {len(generations)} generations, and one factor for "
            f"all of them means one identical surrogate for all of them — including across the "
            f"target/spare divide, where that is a contradiction. Use the default (n = g + 1 per "
            f"generation), 'auto' or 'calibrated'."
        )

    first_collapsed_dir = ""
    if transfer:
        # generation 0 needs no mixture: it is the same checkpoint under every fraction
        first_collapsed_dir = first_collapsed_path or resolve_collapsed_dir(
            0, specifier_name, block_size, prefer_adapter=(surrogate_method == "lora")
        )
        if targets == [0]:
            raise SystemExit(
                f"{TColors.FAIL}the only target is generation 0, which is the anchor the "
                f"surrogates are built from{TColors.ENDC} ({first_collapsed_dir}). Its tilt at "
                f"n = 1 *is* that checkpoint, so the run would optimize against the very model it "
                f"validates against — the same case run_attack.py rejects for "
                f"--collapsed_generation 0. Target a later generation."
            )
        if 0 in targets:
            print(
                f"## {TColors.WARNING}generation 0 is a target and also the surrogate anchor"
                f"{TColors.ENDC}: its surrogate at n = 1 is that checkpoint itself, so it carries "
                f"no transfer claim. The other targets still do."
            )

    selected = [task for task in TASKS if not tasks or task.name in tasks.split(",")]
    if not selected:
        raise SystemExit(f"no tasks matched {tasks!r}; use --list_tasks to see the names")

    # ──────────────────────────── system status print ─────────────────────────
    print(
        "\n"
        + f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information"
        + f"{TColors.ENDC} "
        + _hr(23)
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: "
        + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: "
        f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and "
        f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}"
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {torch_device}")
    if torch_device.type == "cuda":
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: "
            f"{torch.cuda.mem_get_info()[1] // 1024**2} MB"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: "
            f"{psutil.virtual_memory()[0] // 1024**2} MB"
        )
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters{TColors.ENDC} " + _hr(14)
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Baseline Model{TColors.ENDC}: {baseline_dir}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Model Size{TColors.ENDC}: {size_label}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Must break{TColors.ENDC}: generation(s) "
        f"{', '.join(str(generation) for generation in targets)}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Must hold{TColors.ENDC}: baseline"
        + (
            f" + generation(s) {', '.join(str(g) for g in spares)}"
            if spares
            else " only (no spare generations — equivalent to run_attack.py's criterion)"
        )
    )
    for generation in generations:
        role = "break" if generation in targets else "hold"
        print(
            f"##   {TColors.OKCYAN}generation {generation}{TColors.ENDC} ({role}): "
            f"{checkpoints[generation]}"
        )
    if transfer:
        how = "auto" if auto_factor else ("calibrated" if calibrated else "n = g + 1")
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Mode{TColors.ENDC}: "
            f"{TColors.HEADER}transfer{TColors.ENDC} — optimize against {surrogate_method} "
            f"surrogates (factors: {how}), validate against the real checkpoints above"
        )
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Surrogate anchor (gen 0){TColors.ENDC}: "
            f"{first_collapsed_dir}"
        )
    else:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Mode{TColors.ENDC}: direct — optimize against the "
            f"real collapsed checkpoints"
        )
    task_names = ", ".join(task.name for task in selected)
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Tasks{TColors.ENDC}: {task_names}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Steps x Restarts{TColors.ENDC}: "
        f"{num_steps} x {restarts}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Search Width / topk / n_replace{TColors.ENDC}: "
        f"{search_width} / {topk} / {n_replace}"
    )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Objective{TColors.ENDC}: "
        f"mean_break CE(wrong) + {lambda_base} * mean_hold relu({margin} - CE(wrong)) "
        f"+ {mu_correct} * mean_hold CE(correct)"
    )
    # the number a user needs to predict both the runtime and the VRAM: every one of these is
    # resident and every one is evaluated at every step
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Models per step{TColors.ENDC}: "
        f"{len(targets)} breaking + {len(spares) + 1} holding"
        + (f", plus {surrogate_method} surrogates" if transfer else "")
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Execute generated code{TColors.ENDC}: {not no_exec}")
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Min. Target Capability{TColors.ENDC}: "
        f"{min_capability:.0%}" + (" (not enforced)" if skip_capability_check else "")
    )
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Results Path{TColors.ENDC}: {RESULTS_PATH}")
    print(_hr() + "\n")

    if no_exec:
        print(
            f"{TColors.WARNING}Warning{TColors.ENDC}: --no_exec disables behavioural "
            f"verification, so the selective conjunction is never evaluated. Results are loss-only "
            f"and do NOT establish wrong behaviour.\n"
        )

    # ──────────────────────────── load models ─────────────────────────
    if torch_device.type == "cpu":
        dtype = torch.float32
    elif torch_device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    tokenizer = configure_pad_token(AutoTokenizer.from_pretrained(model_specifier))

    # built before the models, because the auto factor probe decodes with it
    cfg = SearchConfig(
        num_steps=num_steps,
        optim_str_init=optim_str_init,
        search_width=search_width,
        batch_size=batch_size,
        topk=topk,
        n_replace=n_replace,
        allow_non_ascii=allow_non_ascii,
        lambda_base=lambda_base,
        margin=margin,
        mu_correct=mu_correct,
        verify_every=verify_every,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        exec_timeout=exec_timeout,
        no_exec=no_exec,
        stop_on_success=stop_on_success,
        seed=seed,
    )

    print(f"## {TColors.OKBLUE}{TColors.BOLD}Loading baseline model{TColors.ENDC}")
    baseline_model = TargetModel(
        "baseline", load_model(baseline_dir, torch_device, dtype), torch_device
    )
    baseline_role = ModelRole(
        generation=BASELINE_GENERATION,
        role="baseline",
        label="baseline",
        real=baseline_model,
        # the baseline is never a surrogate: the attacker holds it, so the objective sees the model
        # itself and `transfer` is False for this role by construction
        optim=baseline_model,
        checkpoint=baseline_dir,
        description=baseline_dir,
    )

    # every real checkpoint is loaded and stays loaded: it decides capability and success, and the
    # behavioural check decodes all of them every --verify_every steps. This is where the VRAM goes
    reals: dict[int, TargetModel] = {}
    for generation in generations:
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Loading generation {generation}{TColors.ENDC}: "
            f"{checkpoints[generation]}"
        )
        try:
            reals[generation] = TargetModel(
                f"gen{generation}",
                load_model(
                    checkpoints[generation],
                    torch_device,
                    dtype,
                    base_for_adapter=model_specifier,
                ),
                torch_device,
            )
        except torch.OutOfMemoryError as exc:
            raise SystemExit(
                f"{TColors.FAIL}out of memory while loading generation {generation}{TColors.ENDC}: "
                f"a selective run holds every listed checkpoint resident, which is "
                f"{len(generations)} of them plus the baseline here. Shorten the generation lists, "
                f"use a smaller --model_size, or run on a larger card."
            ) from exc

    # ── factors and surrogates ──
    factors: dict[int, float] = {}
    factor_probe: list[dict] = []
    surrogates: dict[float, TargetModel] = {}
    descriptions: dict[float, str] = {}
    if transfer:
        probed_surrogate: TargetModel | None = None
        if auto_factor:
            print(
                f"## {TColors.OKBLUE}{TColors.BOLD}Probing the {surrogate_method} surrogate for "
                f"usable factors{TColors.ENDC} — largest rung each generation still writes code at"
            )
            factors, factor_probe, probed_surrogate = probe_factors(
                generations=generations,
                method=surrogate_method,
                baseline=baseline_role.real,
                first_collapsed_dir=first_collapsed_dir,
                surrogate_model_path=surrogate_model_path,
                tasks=selected,
                tokenizer=tokenizer,
                cfg=cfg,
                min_capability=min_capability,
                device=torch_device,
                dtype=dtype,
                base_specifier=model_specifier,
            )
        elif calibrated:
            dataset_path = (
                os.path.join(path, "generated_datasets/") if path else "./generated_datasets/"
            )
            factors = calibrated_factors(
                generations,
                factor_calibration_file(
                    dataset_path,
                    block_size or 0,
                    specifier_name,
                    mixture_tag(real_data_fraction),
                ),
            )
        elif explicit:
            factors = {generations[0]: float(surrogate_factor)}
        else:
            factors = default_factors(generations)

        check_factor_collisions(targets, spares, factors)
        print(
            f"## {TColors.OKBLUE}{TColors.BOLD}Building {surrogate_method} surrogates{TColors.ENDC}"
            f" (" + ", ".join(f"gen {g} -> n = {factors[g]:g}" for g in generations) + ")"
        )
        # when generation 0 is one of the run's own roles and resolves to the anchor directory,
        # that checkpoint is already in VRAM and the tilt is built on it instead of a second copy
        resident_anchor = None
        if 0 in reals and os.path.abspath(checkpoints[0]) == os.path.abspath(
            first_collapsed_dir
        ):
            resident_anchor = reals[0].model
        surrogates, descriptions = build_role_surrogates(
            factors=factors,
            method=surrogate_method,
            baseline=baseline_role.real,
            first_collapsed_dir=first_collapsed_dir,
            surrogate_model_path=surrogate_model_path,
            device=torch_device,
            dtype=dtype,
            base_specifier=model_specifier,
            probed=probed_surrogate,
            first_model=resident_anchor,
        )

    roles = [baseline_role]
    for generation in generations:
        factor = factors.get(generation)
        surrogate = surrogates.get(factor) if transfer else None
        roles.append(
            ModelRole(
                generation=generation,
                role="target" if generation in targets else "spare",
                label=f"gen{generation}",
                real=reals[generation],
                optim=surrogate or reals[generation],
                checkpoint=checkpoints[generation],
                factor=factor if transfer else None,
                description=(
                    descriptions[factor] if transfer else checkpoints[generation]
                ),
            )
        )

    # the objective adds one-hot gradients from every model, which is only meaningful if they share
    # a vocabulary
    vocab = baseline_role.real.embed_weights.shape[0]
    for role in roles:
        for model in {id(role.real): role.real, id(role.optim): role.optim}.values():
            if model.embed_weights.shape[0] != vocab:
                raise RuntimeError(
                    f"vocabulary mismatch: baseline has {vocab} rows, {role.label}'s "
                    f"{model.label} has {model.embed_weights.shape[0]} — all models must share "
                    f"the tokenizer of {model_specifier}"
                )

    attack = SelectiveGCG(
        roles=roles,
        tokenizer=tokenizer,
        cfg=cfg,
        score_surrogates=transfer and not no_surrogate_scoring,
    )

    # ──────────────────── upfront capability gate ─────────────────────
    # Before optimizing anything, establish that every model in the run can still write correct code
    # unaided. Without that a wrong answer is a symptom of collapse, not of the attack, and a
    # spare that already fails makes the conjunction unsatisfiable by construction.
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Capability Probe"
        f"{TColors.ENDC} " + _hr(21)
    )
    print("##   clean prompts, no adversarial input — can every model still solve them?")
    capability = attack.capability_gate(selected, min_capability)

    if capability.skipped:
        print(f"##   {TColors.WARNING}not probed{TColors.ENDC}: {capability.reason}")
    else:
        for role in roles:
            solved = len(capability.solved[role.label])
            gated = "" if role.role == "spare" else " (gated)"
            # .get, because a run where every task failed reference validation aborts before any
            # capability is computed and still reaches this print
            print(
                f"##   {role.label:12s} solves {solved}/{capability.n_probed} clean tasks "
                f"({capability.capability.get(role.label, 0.0):.0%}){gated}"
            )
        if capability.invalid_tasks:
            print("##   invalid tasks (bad references): " + ", ".join(capability.invalid_tasks))
        for label, broken in capability.surrogate_broken.items():
            if broken:
                print(
                    f"##   {label}'s surrogate cannot solve: " + ", ".join(broken)
                    + " (informational only — the proxy is not the model under attack)"
                )
        print(
            f"##   attackable tasks: "
            f"{', '.join(capability.usable) if capability.usable else 'none'}"
            f"  (decided by the baseline and the targets; spares do not gate)"
        )
        for task_name, spares_broken in capability.spare_broken_on_usable.items():
            print(
                f"##   {TColors.WARNING}{task_name}{TColors.ENDC}: "
                f"{', '.join(spares_broken)} already fail(s) it unaided, so no hit on it can "
                f"satisfy them"
            )
    print(_hr() + "\n")

    outcomes: list[TaskOutcome] = []
    proceed = True
    if capability.aborted:
        if skip_capability_check:
            print(
                f"{TColors.WARNING}Warning{TColors.ENDC}: capability gate failed "
                f"({capability.reason}) but --skip_capability_check was given; continuing on "
                f"whatever tasks remain attackable.\n"
            )
        else:
            print(
                f"## {TColors.FAIL}{TColors.BOLD}ATTACK STOPPED{TColors.ENDC}: "
                f"{capability.reason}"
            )
            print(
                f"## {TColors.OKCYAN}Target earlier generations, pick tasks the baseline and the "
                f"targets still solve, lower --min_capability, or pass --skip_capability_check to "
                f"override.{TColors.ENDC}\n"
            )
            proceed = False

    # ──────────────────────────── run the search ─────────────────────────
    if proceed:
        for task in selected:
            if task.name not in capability.usable:
                verdict = capability.per_task.get(task.name, {})
                reason = verdict.get("invalid")
                if reason is None:
                    statuses = attack.statuses(verdict) if verdict else {}
                    failed = [
                        role.label for role in attack.gating
                        if statuses.get(role.label, "unknown") != "pass"
                    ] or ["unknown"]
                    reason = f"{', '.join(failed)} fail(s) the clean prompt"
                outcomes.append(
                    TaskOutcome(
                        task=task.name,
                        control=verdict if "invalid" not in verdict else {},
                        skipped=f"excluded by the capability probe ({reason})",
                    )
                )
                continue
            print(f"\n## {TColors.HEADER}{TColors.BOLD}Task: {task.name}{TColors.ENDC} " + _hr(12))
            outcome = attack.run_task(task, restarts)
            if outcome.skipped:
                print(f"## {TColors.WARNING}skipped{TColors.ENDC}: {outcome.skipped}")
            outcomes.append(outcome)

    # ──────────────────────────── report and save ─────────────────────────
    print(f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Summary{TColors.ENDC} " + _hr(11))
    if not proceed:
        print(
            f"## {TColors.FAIL}attack not run{TColors.ENDC}: the capability probe failed"
        )
        print(f"##   {capability.reason}")
    for outcome in outcomes:
        if outcome.skipped:
            status = f"{TColors.WARNING}skipped ({outcome.skipped}){TColors.ENDC}"
        elif outcome.successes:
            status = f"{TColors.OKGREEN}{len(outcome.successes)} selective hit(s){TColors.ENDC}"
        else:
            status = f"{TColors.FAIL}no selective hit{TColors.ENDC}"
        best = "n/a" if outcome.best_objective is None else f"{outcome.best_objective:.4f}"
        print(f"## {outcome.task:16s} {status}  (best objective {best})")
        if outcome.successes:
            hit = outcome.successes[0]
            print("##   suffix: " + repr(hit["suffix"]))
            print(
                "##   statuses: "
                + "  ".join(
                    f"{role.label}={hit[f'{role.label}_status']}" for role in roles
                )
            )
            for role in attack.targets:
                print(f"##   {role.label} code:\n" + hit[f"{role.label}_code"])
    print(_hr() + "\n")

    # ── where the conjunction held and where it broke ──
    selectivity = attack.selectivity_report(outcomes)
    print(
        f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Selectivity{TColors.ENDC} "
        + _hr(15)
    )
    print(f"##   behavioural checks: {selectivity.n_verified}")
    success_color = TColors.OKGREEN if selectivity.n_success else TColors.FAIL
    print(
        f"##   full selective hits (every condition at once): {success_color}"
        f"{selectivity.n_success}{TColors.ENDC}"
    )
    print(
        f"##   every target broken (spares ignored): {selectivity.n_targets_all_broken}   "
        f"every holding model kept (targets ignored): {selectivity.n_holds_all_kept}"
    )
    leak_color = TColors.OKGREEN if selectivity.n_baseline_broken == 0 else TColors.FAIL
    print(
        f"##   baseline lost its correct answer: {leak_color}"
        f"{selectivity.n_baseline_broken}{TColors.ENDC} of {selectivity.n_verified} "
        f"(must be 0 for the attack to be selective at all; see its per-model line below for how "
        f"many of those were objectively wrong answers rather than derailed decoding)"
    )
    if selectivity.n_verified:
        for role in roles:
            met = selectivity.n_condition_met[role.label]
            print(
                f"##   {role.label:12s} {f'must {role.verb}:':12s} satisfied in {met}/"
                f"{selectivity.n_verified} checks, emitted wrong code in "
                f"{selectivity.n_wrong[role.label]}, blocked a hit "
                f"{selectivity.blocked_by[role.label]} time(s)"
            )
    else:
        print(f"##   {TColors.WARNING}no verification ran, so none of this is informative"
              f"{TColors.ENDC}")
    print(_hr() + "\n")

    # ── surrogate quality ──
    surrogate_stats = None
    if transfer and not no_surrogate_scoring:
        surrogate_stats = attack.surrogate_report(outcomes)
        surrogate_stats.method = surrogate_method
        print(
            f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Surrogate quality"
            f"{TColors.ENDC} " + _hr(21)
        )
        print(f"##   checks with surrogate verdicts: {surrogate_stats.n_verified}")
        for role in roles:
            if not role.transfer:
                continue
            bucket = surrogate_stats.per_generation[role.label]
            agreement = surrogate_stats.agreement[role.label]
            color = TColors.OKGREEN if agreement >= 0.5 else TColors.WARNING
            print(
                f"##   {role.label:12s} n = {role.factor:<5g} agreement {color}{agreement:.0%}"
                f"{TColors.ENDC} over {bucket['n']} checks "
                f"(surrogate wrong {bucket['surrogate_wrong']}, real wrong {bucket['real_wrong']})"
            )
        print(
            f"##   ensemble called a selective hit {surrogate_stats.n_predicted_hit} time(s): "
            f"{surrogate_stats.n_true_positive} correct, {surrogate_stats.n_false_alarm} false "
            f"alarm(s) (precision {surrogate_stats.precision:.0%})"
        )
        print(
            f"##   real hits the ensemble did not call: {surrogate_stats.n_missed} "
            f"(recall {surrogate_stats.recall:.0%})"
        )
        if surrogate_stats.n_verified == 0:
            print(
                f"##   {TColors.WARNING}no verification ran, so none of this is "
                f"informative{TColors.ENDC}"
            )
        print(_hr() + "\n")

    out_file = result_file(
        targets, spares, specifier_name, mixture_tag(real_data_fraction), surrogate_method
    )
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline_model": baseline_dir,
                "target_generations": targets,
                "spare_generations": spares,
                "collapsed_models": {str(g): checkpoints[g] for g in generations},
                "real_data_fraction": real_data_fraction,
                "transfer_mode": transfer,
                "surrogate_method": surrogate_method,
                "surrogate_factors": (
                    {str(g): factors[g] for g in generations} if transfer else None
                ),
                "surrogate_factor_probe": factor_probe or None,
                "surrogate_models": (
                    {str(g): descriptions[factors[g]] for g in generations} if transfer else None
                ),
                "first_collapsed_model": first_collapsed_dir or None,
                "config": cfg.__dict__,
                "aborted": capability.aborted and not skip_capability_check,
                "capability_probe": capability.__dict__,
                "selectivity": selectivity.__dict__,
                "surrogate_quality": (
                    {
                        **surrogate_stats.__dict__,
                        "precision": surrogate_stats.precision,
                        "recall": surrogate_stats.recall,
                    }
                    if surrogate_stats is not None
                    else None
                ),
                "results": [outcome.__dict__ for outcome in outcomes],
            },
            handle,
            indent=2,
        )
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Saved the attack results under: "
        f"{TColors.HEADER}{out_file}{TColors.ENDC}"
    )

    # ──────────────────── print the elapsed time ─────────────────────────
    delta = timedelta(seconds=int(time.time() - start_time))
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Execution time: ")
    if delta.days:
        print(f"{TColors.HEADER}{delta.days} days, {hours:02}:{minutes:02}:{seconds:02}")
    else:
        print(f"{TColors.HEADER}{hours:02}:{minutes:02}:{seconds:02}")
    print(f"{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial suffixes that are selective across collapse generations"
    )
    parser.add_argument(
        "--device",
        "-dx",
        type=str,
        default="cuda",
        help="specifies the device to run the computations on (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--target_generations",
        "-tg",
        type=generation_list_arg,
        default=[9],
        help="comma separated collapse generations the suffix MUST break, i.e. must make emit "
        "objectively wrong code. run_baseline.py -ng 10 yields 0..9, so the 10th generation is "
        "index 9 (default: 9)",
    )
    parser.add_argument(
        "--spare_generations",
        "-sg",
        type=generation_list_arg,
        default=[],
        help="comma separated collapse generations the suffix must NOT break, i.e. must keep "
        "emitting correct code. The pristine baseline is always in this group and does not need "
        "listing. Empty means the baseline is the only model that has to hold, which is "
        "run_attack.py's criterion (default: empty)",
    )
    parser.add_argument(
        "--block_size",
        "-bs",
        type=int,
        default=None,
        help="effective block size baked into the checkpoint names; auto-detected if omitted",
    )
    add_model_arguments(parser, role="the baseline model")
    parser.add_argument(
        "--baseline_model_path",
        "-bmp",
        type=str,
        default="",
        help="explicit path to the baseline model (default: --model_specifier)",
    )
    parser.add_argument(
        "--generation_model_path",
        "-gmp",
        type=generation_path_arg,
        default={},
        help="explicit checkpoints for individual generations as comma separated gen=path pairs, "
        "e.g. '9=/data/model_9,0=/data/model_0'. Generations not listed are resolved from the "
        "naming convention as usual. This is the list-shaped replacement for run_attack.py's "
        "--collapsed_model_path",
    )
    parser.add_argument(
        "--surrogate_method",
        "-sm",
        type=str,
        default="none",
        choices=("none",) + METHODS,
        help="'none' optimizes against the real checkpoints. 'logit' or 'lora' enable transfer "
        "mode: every listed generation, target and spare alike, is replaced in the objective by a "
        "surrogate built only from the base and generation-0 models, and the real checkpoints are "
        "held back for validation only. 'data' is rejected — it is a corpus-level surrogate whose "
        "loss is identical to the base model's (default: none)",
    )
    parser.add_argument(
        "--surrogate_factor",
        "-sf",
        type=surrogate_factor_arg,
        default=0.0,
        help="how each generation's extrapolation factor n is chosen. 0.0 derives n = g + 1 per "
        "generation, the indexing convention. 'auto' measures them instead: one descending walk of "
        "the factor ladder, after which every generation takes the largest rung it still solves "
        "--min_capability of the clean tasks at. 'calibrated' reads them from the calibration "
        "utils/evaluate_perplexity.py --calibrate fits. An explicit number is only taken when the "
        "run involves a single generation, since one factor for all of them would make every "
        "surrogate the same model (default: 0.0)",
    )
    parser.add_argument(
        "--surrogate_model_path",
        "-smp",
        type=str,
        default="",
        help="prebuilt surrogate to use instead of building one, e.g. run_extrapolation.py's "
        "model_scaled_n<n>_* directory ('lora' method only, and only meaningful when the run "
        "resolves to a single distinct factor)",
    )
    parser.add_argument(
        "--first_collapsed_path",
        "-fcp",
        type=str,
        default="",
        help="explicit path to the generation-0 collapsed model the surrogates are built from "
        "(default: resolved from model_outputs/; the 'lora' method needs the adapter, not the "
        "merged _fp16 copy)",
    )
    parser.add_argument(
        "--no_surrogate_scoring",
        "-nss",
        action="store_true",
        help="do not decode the surrogates during the behavioural check. In transfer mode the "
        "check otherwise decodes every real model *and* every distinct surrogate, so this roughly "
        "halves its cost — at the price of the surrogate report, which needs those verdicts",
    )
    parser.add_argument(
        "--real_data_fraction",
        "-rdf",
        type=float,
        default=0.0,
        help="the --real_data_fraction run_baseline.py was given. A mixed run names checkpoints "
        "model_{gen}_bs{bs}_{name}_rdf{value}[_fp16] from generation 1 on, so the same value is "
        "needed here to find them. The generation-0 surrogate anchor is shared across mixtures and "
        "needs no value (default: 0.0)",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="",
        help="root directory containing model_outputs/ (default: current directory)",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default="",
        help="comma-separated task names to attack (default: all)",
    )
    parser.add_argument(
        "--list_tasks",
        "-lt",
        action="store_true",
        help="print the available tasks and exit",
    )
    parser.add_argument(
        "--restarts",
        "-r",
        type=int,
        default=1,
        help="random suffix re-initializations per task (default: 1)",
    )
    parser.add_argument(
        "--num_steps",
        "-ns",
        type=int,
        default=250,
        help="optimizer steps per restart (default: 250)",
    )
    parser.add_argument(
        "--search_width",
        "-sw",
        type=int,
        default=256,
        help="candidate suffixes sampled per step (default: 256)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="candidates scored per forward pass, per model; halved automatically on OOM "
        "(default: 32)",
    )
    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=256,
        help="top-k tokens per position taken from the gradient (default: 256)",
    )
    parser.add_argument(
        "--n_replace",
        "-nr",
        type=int,
        default=1,
        help="suffix positions mutated per candidate (default: 1)",
    )
    parser.add_argument(
        "--optim_str_init",
        "-osi",
        type=str,
        default="x x x x x x x x x x x x x x x x x x x x",
        help="initial suffix for the first restart",
    )
    parser.add_argument(
        "--allow_non_ascii",
        "-ana",
        action="store_true",
        help="allow non-ASCII tokens in the adversarial suffix",
    )
    parser.add_argument(
        "--lambda_base",
        "-lb",
        type=float,
        default=1.0,
        help="weight of the must-not-break hinge, averaged over the baseline and every spare "
        "generation (default: 1.0)",
    )
    parser.add_argument(
        "--margin",
        "-m",
        type=float,
        default=3.0,
        help="loss margin every holding model must keep from the wrong code (default: 3.0)",
    )
    parser.add_argument(
        "--mu_correct",
        "-mc",
        type=float,
        default=0.5,
        help="weight of the stays-correct anchor over the holding group; 0 disables it "
        "(default: 0.5)",
    )
    parser.add_argument(
        "--verify_every",
        "-ve",
        type=int,
        default=10,
        help="run the behavioural check every N steps. It decodes every model in the run, so this "
        "is the dominant cost at small values (default: 10)",
    )
    parser.add_argument(
        "--max_new_tokens",
        "-mnt",
        type=int,
        default=96,
        help="decoding budget during verification (default: 96)",
    )
    parser.add_argument(
        "--repetition_penalty",
        "-rp",
        type=float,
        default=1.0,
        help="decoding repetition penalty during verification (default: 1.0)",
    )
    parser.add_argument(
        "--exec_timeout",
        "-et",
        type=float,
        default=10.0,
        help="per-candidate unit-test timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--no_exec",
        "-ne",
        action="store_true",
        help="never execute generated code; disables behavioural verification and therefore the "
        "selective criterion itself",
    )
    parser.add_argument(
        "--stop_on_success",
        "-sos",
        action="store_true",
        help="stop a task as soon as a full selective hit is verified",
    )
    parser.add_argument(
        "--min_capability",
        "-mcap",
        type=float,
        default=0.2,
        help="fraction of clean (suffix-free) tasks every TARGET generation must still solve "
        "before the attack starts; below this a target is considered incapable of generating code "
        "and the run is stopped. Spare generations are not gated on it — tasks they cannot solve "
        "are simply not attackable (default: 0.2)",
    )
    parser.add_argument(
        "--skip_capability_check",
        "-scc",
        action="store_true",
        help="do not stop the run when the capability probe fails; per-task exclusion of tasks "
        "some model already gets wrong still applies",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1337,
        help="RNG seed (default: 1337)",
    )
    args = parser.parse_args()
    main(**vars(args))
