"""Is a collapsed model just the base model at a lower temperature?

The hypothesis: model collapse sharpens the next-token distribution — common tokens become
more likely, rare tokens less — and a lower sampling temperature does the same thing to the
base model, so the two should produce similar distributions. If that held, the surrogate
problem would be trivial: ``softmax(l_base / tau_g)`` would stand in for generation ``g``.

The test is teacher forced on held-out human code, the same rows and templating
``utils/evaluate_perplexity.py`` scores, so every model is asked the same question at the same
positions: "what comes next?" For every generation ``g`` the script measures

* **sharpness** — mean entropy, mean top-1 mass, and how the mass is spread over the *base
  model's* rank buckets (rank 1, 2-5, 6-20, 21-100, 101+) and over tokens the base finds rare.
  Collapse-as-sharpening predicts mass moving down the rank buckets; a temperature does exactly
  that and nothing else
* **the best temperature** — ``tau_KL``, the temperature of the base model that minimizes
  ``KL(p_g || softmax(l_base / tau))`` over all positions (convex in ``1/tau``, found by golden
  section), and ``tau_H``, the one whose mean entropy equals generation ``g``'s. The residual
  ``KL`` at ``tau_KL`` against the ``KL`` at ``tau = 1`` is the share of the gap a temperature
  can close. A per-position entropy temperature is fitted too, and its spread says whether one
  global temperature is even the right shape
* **what a temperature cannot do** — the argmax of ``softmax(l / tau)`` is the argmax of ``l``
  at every temperature, and so is the whole ranking. The fraction of positions where generation
  ``g`` and the base disagree on the argmax, and the top-5 overlap, bound how much of collapse a
  temperature can ever reproduce
* **the surrogates the attack actually uses** — ``KL(p_g || tilt)`` for the logit tilt at
  ``n = g + 1`` and at the calibrated factor, each plain and entropy matched, next to the
  tempered base. Whichever is closer to the real generation is the better search target
* **held-out perplexity** — of the real generation, the base at ``tau = 1`` and at ``tau_KL``,
  so the temperature's fit can be read on the utility curve the paper plots

``--sample`` adds the corpus-level reading of the same claim: generation ``g`` and the base at
``tau_KL`` each write continuations for the same prompts, and their unigram statistics (distinct
tokens, hapax share, mass on the 100 most frequent tokens, Jensen-Shannon distance) are compared.

Usage:
    python run_temperature_collapse.py -msz 0.5b -rdf 0.5 -n 9 -p ./runs/fresh
    python run_temperature_collapse.py -msz 0.5b -rdf 0.5 -n 9 -p ./runs/fresh --sample 64
    python run_temperature_collapse.py -msz 0.5b -rdf 0.5 -p ./runs/fresh --plot_only

Writes ``<path>/attack_results/temperature_collapse_{name}{mix}.json`` and
``<plots_path>/temperature_collapse_{name}{mix}.{png,pdf}``. No unsloth: the models are the
merged fp16 checkpoints, loaded the way run_attack.py loads them.
"""

import argparse
import json
import math
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

import run_attack
from run_attack import load_model, resolve_collapsed_dir
from utils.colors import TColors
from utils.extrapolation import (
    calibrated_factor,
    entropy_matched_scale,
    extrapolate_logits,
    factor_calibration_file,
    logit_entropy,
)
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import mixture_tag
from utils.perplexity import format_scoring_prompts
from utils.plotting import apply_perplexity_style, save_figure
from utils.utils import clear_inherited_max_length, configure_pad_token

matplotlib.use("Agg")

DATASET_SPECIFIER: str = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
# the 90/10 cut run_baseline.py's make_splits takes; the held-out rows are the last 10%
TRAIN_FRACTION: float = 0.9
# rank buckets under the base model's ordering: the first is the mode, the last the deep tail
RANK_BUCKETS: tuple = ((1, 1), (2, 5), (6, 20), (21, 100), (101, None))
# a token the base model gives less than this is "rare" for the rare-token mass statistic
RARE_THRESHOLD: float = 1e-4
# positions per chunk when the cached logits are expanded to float32 probabilities
CHUNK: int = 512
# the temperature search bracket, in tau. 0.05 is a near point mass, 5 near uniform on code
TAU_RANGE: tuple = (0.05, 5.0)


# ───────────────────────────────────── the data ─────────────────────────────────────────────
def held_out_rows(dataset_size: int, rows: int) -> tuple[list, list, str]:
    """The held-out human rows every model is scored on.

    Mirrors utils/evaluate_perplexity.py: rows beyond --dataset_size when the run used a front
    slice, otherwise the validation 10% that make_splits held out of generation 0's training.

    Args:
        dataset_size (int): the --dataset_size the collapse run was given, 0 for the whole corpus
        rows (int): how many of those rows to use

    Returns:
        tuple: instructions, responses, and a description of where they came from
    """
    corpus = load_dataset(DATASET_SPECIFIER, split="train")
    if 0 < dataset_size < len(corpus):
        tail = corpus.select(range(dataset_size, len(corpus)))
        description = f"rows {dataset_size}.. of {DATASET_SPECIFIER}, unseen by construction"
    else:
        cut = int(TRAIN_FRACTION * len(corpus))
        tail = corpus.select(range(cut, len(corpus)))
        description = f"validation slice (last 10%) of {DATASET_SPECIFIER}"
    tail = tail.select(range(min(rows, len(tail))))
    return list(tail["instruction"]), list(tail["response"]), description


def response_positions(tokenizer, instructions: list, responses: list, block_size: int) -> list:
    """Tokenizes the templated rows and marks the positions that predict a response token.

    Only the assistant turn is scored, exactly like the perplexity, so that the prompt — identical
    for every model — does not dilute the comparison. A position ``t`` predicts token ``t + 1``,
    so the scored positions start one before the first response token.

    Args:
        tokenizer: the shared tokenizer
        instructions (list): user turns
        responses (list): assistant turns
        block_size (int): cap on the response length in tokens, the run's block size

    Returns:
        list: per row ``(input_ids, first_scored_position, last_scored_position)``
    """
    texts = format_scoring_prompts(tokenizer, instructions, responses)
    rows = []
    for instruction, text in zip(instructions, texts):
        prefix = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant for code completion."},
                {"role": "user", "content": instruction},
            ],
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        start = len(prefix) - 1
        stop = min(len(ids) - 1, start + block_size)
        if stop > start:
            rows.append((ids[: stop + 1], start, stop))
    return rows


# ────────────────────────────────── the logits ─────────────────────────────────────────────
@torch.no_grad()
def gather_logits(model, rows: list, positions: list, device, batch_size: int) -> torch.Tensor:
    """Runs the rows through one model and keeps the logits of the scored positions.

    Args:
        model: a causal LM on `device`
        rows (list): from response_positions
        positions (list): per row, the scored position indices to keep (the token budget applies)
        device: the device the rows are moved to
        batch_size (int): rows per forward pass

    Returns:
        Tensor: (tokens, vocabulary) float16 logits, in row then position order
    """
    out = []
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin : begin + batch_size]
        keep = positions[begin : begin + batch_size]
        longest = max(len(ids) for ids, _, _ in batch)
        input_ids = torch.full((len(batch), longest), model.config.eos_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (ids, _, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        for i, kept in enumerate(keep):
            if kept:
                out.append(logits[i, kept].to(torch.float16))
    return torch.cat(out, dim=0)


def budget_positions(rows: list, tokens: int) -> tuple[list, torch.Tensor]:
    """Spreads the token budget over the rows, in order, and lists the next tokens.

    Args:
        rows (list): from response_positions
        tokens (int): total positions to keep, 0 for all

    Returns:
        tuple: per row the kept positions, and the (tokens,) target ids those positions predict
    """
    positions, targets, used = [], [], 0
    for ids, start, stop in rows:
        kept = []
        for t in range(start, stop):
            if tokens and used >= tokens:
                break
            kept.append(t)
            targets.append(ids[t + 1])
            used += 1
        positions.append(kept)
    return positions, torch.tensor(targets, dtype=torch.long)


# ─────────────────────────────── the measurements ─────────────────────────────────────────
def chunks(n: int):
    """Yields (begin, end) slices of CHUNK positions."""
    for begin in range(0, n, CHUNK):
        yield begin, min(n, begin + CHUNK)


@torch.no_grad()
def log_softmax_chunk(logits: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """float32 log-probabilities of one chunk of cached logits, divided by a temperature."""
    return torch.log_softmax(logits.to(torch.float32) / scale, dim=-1)


@torch.no_grad()
def kl_to_tempered(p_logp: torch.Tensor, p_dot_l: torch.Tensor, base: torch.Tensor, beta: float) -> float:
    """Mean ``KL(p || softmax(beta * l_base))`` from its per-position sufficient statistics.

    ``KL = sum p log p - beta * sum p l + logsumexp(beta * l)``: the first two terms are cached
    scalars per position, only the partition function needs the cached base logits.

    Args:
        p_logp (Tensor): (tokens,) ``sum p log p`` of the real distribution
        p_dot_l (Tensor): (tokens,) ``sum p * l_base``
        base (Tensor): (tokens, vocabulary) cached base logits
        beta (float): the inverse temperature

    Returns:
        float: the mean KL in nats
    """
    total = 0.0
    for begin, end in chunks(base.shape[0]):
        lse = torch.logsumexp(beta * base[begin:end].to(torch.float32), dim=-1)
        total += (p_logp[begin:end] - beta * p_dot_l[begin:end] + lse).sum().item()
    return total / base.shape[0]


@torch.no_grad()
def mean_entropy_tempered(base: torch.Tensor, beta: float) -> float:
    """Mean entropy of ``softmax(beta * l_base)`` over the cached positions."""
    total = 0.0
    for begin, end in chunks(base.shape[0]):
        total += logit_entropy(beta * base[begin:end].to(torch.float32)).sum().item()
    return total / base.shape[0]


def golden_section(function, low: float, high: float, iterations: int = 40) -> float:
    """Minimizes a unimodal function of one variable on [low, high]."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = function(d)
    return (a + b) / 2


def bisect_entropy(base: torch.Tensor, target: float, iterations: int = 40) -> float:
    """The inverse temperature at which the tempered base's mean entropy equals `target`.

    Entropy of ``softmax(beta * l)`` is monotone decreasing in ``beta``, so this is a bisection
    on ``log beta`` over the TAU_RANGE bracket; a target outside the bracket returns its edge.
    """
    lo, hi = math.log(1 / TAU_RANGE[1]), math.log(1 / TAU_RANGE[0])
    if mean_entropy_tempered(base, math.exp(lo)) <= target:
        return math.exp(lo)
    if mean_entropy_tempered(base, math.exp(hi)) >= target:
        return math.exp(hi)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if mean_entropy_tempered(base, math.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return math.exp((lo + hi) / 2)


@torch.no_grad()
def sharpness(logp: torch.Tensor, base_rank: torch.Tensor, base_rare: torch.Tensor, targets: torch.Tensor) -> dict:
    """Sharpness statistics of one chunk of log-probabilities, under the base model's ranking.

    Args:
        logp (Tensor): (chunk, vocabulary) log-probabilities of the model being described
        base_rank (Tensor): (chunk, vocabulary) 1-based rank of every token under the base model
        base_rare (Tensor): (chunk, vocabulary) bool, tokens the base model finds rare
        targets (Tensor): (chunk,) the real next tokens

    Returns:
        dict: sums over the chunk of entropy, top-1 mass, mass per rank bucket, rare mass, the
            NLL of the real next token, and the argmax
    """
    p = logp.exp()
    out = {
        "entropy": torch.special.entr(p).sum(dim=-1).sum().item(),
        "top1": p.max(dim=-1).values.sum().item(),
        "rare_mass": (p * base_rare).sum(dim=-1).sum().item(),
        "nll": -logp.gather(1, targets.unsqueeze(1)).sum().item(),
    }
    for lo, hi in RANK_BUCKETS:
        inside = base_rank >= lo if hi is None else (base_rank >= lo) & (base_rank <= hi)
        out[f"rank_{lo}_{hi or 'end'}"] = (p * inside).sum(dim=-1).sum().item()
    out["argmax"] = logp.argmax(dim=-1)
    out["top5"] = logp.topk(5, dim=-1).indices
    return out


def accumulate(total: dict, part: dict) -> None:
    """Adds one chunk's sums into the running totals (tensors are concatenated)."""
    for key, value in part.items():
        if isinstance(value, torch.Tensor):
            total.setdefault(key, []).append(value)
        else:
            total[key] = total.get(key, 0.0) + value


def finish(total: dict, n: int, reference: dict | None = None) -> dict:
    """Turns the running sums into means, plus the ranking agreement with a reference."""
    out = {k: v / n for k, v in total.items() if not isinstance(v, list)}
    out["perplexity"] = math.exp(out["nll"])
    argmax = torch.cat(total["argmax"])
    top5 = torch.cat(total["top5"])
    if reference is not None:
        out["argmax_agreement"] = (argmax == reference["_argmax"]).float().mean().item()
        # top-5 Jaccard, per position: |A ∩ B| / |A ∪ B| with |A| = |B| = 5
        inter = (top5.unsqueeze(2) == reference["_top5"].unsqueeze(1)).any(dim=2).sum(dim=1).float()
        out["top5_jaccard"] = (inter / (10 - inter)).mean().item()
    out["_argmax"], out["_top5"] = argmax, top5
    return out


@torch.no_grad()
def describe(logits: torch.Tensor, base: torch.Tensor, base_stats: dict | None, targets: torch.Tensor, scale: float = 1.0) -> dict:
    """Sharpness and ranking statistics of ``softmax(logits / scale)`` against the base model.

    Args:
        logits (Tensor): (tokens, vocabulary) cached logits of the model being described
        base (Tensor): (tokens, vocabulary) cached base logits, for the ranking and the rare set
        base_stats (dict | None): the base model's own description, for the agreement statistics
        targets (Tensor): (tokens,) real next tokens
        scale (float): temperature applied to `logits`

    Returns:
        dict: the means, see `sharpness`, plus `argmax_agreement` and `top5_jaccard`
    """
    total = {}
    for begin, end in chunks(logits.shape[0]):
        base_logp = log_softmax_chunk(base[begin:end])
        rank = base_logp.argsort(dim=-1, descending=True).argsort(dim=-1) + 1
        rare = base_logp.exp() < RARE_THRESHOLD
        accumulate(total, sharpness(log_softmax_chunk(logits[begin:end], scale), rank, rare, targets[begin:end]))
    return finish(total, logits.shape[0], base_stats)


@torch.no_grad()
def kl_rows(target_logits: torch.Tensor, candidate_logp_of_chunk, n: int) -> float:
    """Mean ``KL(p_target || q)`` where `candidate_logp_of_chunk(begin, end)` gives ``log q``."""
    total = 0.0
    for begin, end in chunks(n):
        p_logp = log_softmax_chunk(target_logits[begin:end])
        q_logp = candidate_logp_of_chunk(begin, end)
        total += (p_logp.exp() * (p_logp - q_logp)).sum(dim=-1).sum().item()
    return total / n


@torch.no_grad()
def sufficient_statistics(gen: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per position ``sum p log p`` and ``sum p * l_base`` of the real distribution."""
    p_logp, p_dot_l = [], []
    for begin, end in chunks(gen.shape[0]):
        logp = log_softmax_chunk(gen[begin:end])
        p = logp.exp()
        p_logp.append((p * logp).sum(dim=-1))
        p_dot_l.append((p * base[begin:end].to(torch.float32)).sum(dim=-1))
    return torch.cat(p_logp), torch.cat(p_dot_l)


@torch.no_grad()
def per_position_temperatures(gen: torch.Tensor, base: torch.Tensor) -> dict:
    """The base temperature that matches generation g's entropy at every position separately.

    One global temperature is the hypothesis; this is the shape check. A real sharpening-by-
    temperature would give a narrow spread; if the fitted temperatures scatter widely, the
    collapsed model is sharper at some positions and flatter at others, which no single
    temperature reproduces.
    """
    taus = []
    for begin, end in chunks(gen.shape[0]):
        target = logit_entropy(gen[begin:end].to(torch.float32))
        taus.append(entropy_matched_scale(base[begin:end].to(torch.float32), target))
    tau = torch.cat(taus).clamp(min=1e-6)
    log_tau = tau.log()
    q = torch.quantile(log_tau, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=log_tau.device))
    return {
        "median": math.exp(q[2].item()),
        "iqr": [math.exp(q[1].item()), math.exp(q[3].item())],
        "decile": [math.exp(q[0].item()), math.exp(q[4].item())],
        "log_std": log_tau.std().item(),
        "share_below_1": (tau < 1).float().mean().item(),
    }


# ──────────────────────────────── the sampling check ───────────────────────────────────────
@torch.no_grad()
def sample_unigrams(model, tokenizer, prompts: list, device, temperature: float, new_tokens: int, batch_size: int, seed: int) -> torch.Tensor:
    """Unigram counts of the continuations one model writes at one temperature.

    Untruncated sampling (top-p 1, top-k off): the question is what the model's own distribution
    produces, not what a decoding policy leaves of it. The seed is fixed per call so that two
    models see the same random stream.
    """
    counts = torch.zeros(len(tokenizer), dtype=torch.long)
    tokenizer.padding_side = "left"
    for begin in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[begin : begin + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + begin)
        out = model.generate(
            **batch, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
            max_new_tokens=new_tokens, pad_token_id=tokenizer.pad_token_id,
        )
        continuation = out[:, batch["input_ids"].shape[1]:]
        for row in continuation:
            row = row[row != tokenizer.pad_token_id]
            counts += torch.bincount(row.cpu(), minlength=len(tokenizer))[: len(tokenizer)]
    return counts


def unigram_summary(counts: torch.Tensor, top: int = 100) -> dict:
    """Distinct tokens, hapax share and mass on the most frequent tokens of one sample."""
    total = counts.sum().item()
    nonzero = counts[counts > 0]
    sorted_counts = counts.sort(descending=True).values
    return {
        "tokens": total,
        "distinct": int((counts > 0).sum().item()),
        "hapax_share": (nonzero == 1).sum().item() / max(1, len(nonzero)),
        f"top{top}_mass": sorted_counts[:top].sum().item() / max(1, total),
    }


def jensen_shannon(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jensen-Shannon distance (base-2, in [0, 1]) between two count vectors."""
    p = a.double() / max(1, a.sum().item())
    q = b.double() / max(1, b.sum().item())
    m = (p + q) / 2
    def kl(x, y):
        mask = x > 0
        return (x[mask] * (x[mask] / y[mask]).log2()).sum().item()
    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


# ───────────────────────────────────── the figure ─────────────────────────────────────────
def plot(payload: dict, plot_stem: str, usetex: bool) -> None:
    """Four panels: the fitted temperatures, the distances, the sharpness, and what a
    temperature cannot move (the ranking)."""
    apply_perplexity_style(usetex, font_size=14)
    gens = [int(g) for g in payload["generations"]]
    rows = [payload["generations"][str(g)] for g in gens]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    label = payload["model"].replace("_", "-")
    mixture = payload["mixture_label"]

    ax = axes[0, 0]
    ax.plot(gens, [r["tau_kl"] for r in rows], "o-", label=r"$\tau_{KL}$ (min KL)")
    ax.plot(gens, [r["tau_entropy"] for r in rows], "s--", label=r"$\tau_{H}$ (entropy match)")
    med = [r["per_position_tau"]["median"] for r in rows]
    lo = [r["per_position_tau"]["iqr"][0] for r in rows]
    hi = [r["per_position_tau"]["iqr"][1] for r in rows]
    ax.plot(gens, med, "^:", color="gray", label="per-position median (IQR)")
    ax.fill_between(gens, lo, hi, color="gray", alpha=0.2)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xlabel("generation"); ax.set_ylabel(r"base temperature $\tau$")
    ax.set_title("best base temperature for generation g"); ax.legend(fontsize=10)

    ax = axes[0, 1]
    ax.plot(gens, [r["kl_base"] for r in rows], "o-", label=r"base, $\tau = 1$")
    ax.plot(gens, [r["kl_base_tau_kl"] for r in rows], "s-", label=r"base, $\tau = \tau_{KL}$")
    for key, style, name in (
        ("kl_tilt_index", "^--", "tilt n = g+1"), ("kl_tilt_index_em", "^-", "tilt n = g+1, entropy matched"),
        ("kl_tilt_cal", "v--", "tilt n = calibrated"), ("kl_tilt_cal_em", "v-", "tilt calibrated, entropy matched"),
    ):
        values = [r.get(key) for r in rows]
        if any(v is not None for v in values):
            ax.plot(gens, [float("nan") if v is None else v for v in values], style, label=name)
    ax.set_yscale("log"); ax.set_xlabel("generation"); ax.set_ylabel(r"$KL(p_g \,\|\, q)$ [nats/token]")
    ax.set_title("distance from the real generation"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    base = payload["base"]
    ax.plot(gens, [r["real"]["entropy"] for r in rows], "o-", color="C0", label="entropy: real")
    ax.plot(gens, [r["tempered"]["entropy"] for r in rows], "o--", color="C0", alpha=0.6, label=r"entropy: base at $\tau_{KL}$")
    ax.axhline(base["entropy"], color="C0", lw=0.8, ls=":")
    tail = "rank_21_100"
    ax.plot(gens, [r["real"]["top1"] for r in rows], "s-", color="C1", label="top-1 mass: real")
    ax.plot(gens, [r["tempered"]["top1"] for r in rows], "s--", color="C1", alpha=0.6, label=r"top-1 mass: base at $\tau_{KL}$")
    ax.axhline(base["top1"], color="C1", lw=0.8, ls=":")
    ax.plot(gens, [r["real"][tail] + r["real"]["rank_101_end"] for r in rows], "^-", color="C2", label="mass beyond base rank 20: real")
    ax.plot(gens, [r["tempered"][tail] + r["tempered"]["rank_101_end"] for r in rows], "^--", color="C2", alpha=0.6, label=r"beyond rank 20: base at $\tau_{KL}$")
    ax.axhline(base[tail] + base["rank_101_end"], color="C2", lw=0.8, ls=":")
    ax.set_xlabel("generation"); ax.set_ylabel("nats / probability mass")
    ax.set_title("sharpness (dotted: the base itself)"); ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(gens, [r["real"]["argmax_agreement"] for r in rows], "o-", label="argmax agrees with base")
    ax.plot(gens, [r["real"]["top5_jaccard"] for r in rows], "s-", label="top-5 Jaccard with base")
    ax.plot(gens, [1 - r["kl_base_tau_kl"] / r["kl_base"] if r["kl_base"] > 0 else 0 for r in rows], "^-", label=r"share of KL closed by $\tau_{KL}$")
    ax.set_ylim(0, 1.02); ax.set_xlabel("generation"); ax.set_ylabel("fraction")
    ax.set_title("ranking agreement (a temperature keeps every rank)"); ax.legend(fontsize=10)

    figure.suptitle(f"Is generation g the base model at a lower temperature? ({label}, {mixture})")
    save_figure(figure, plot_stem)


# ──────────────────────────────────────── main ────────────────────────────────────────────
def main(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_specifier = resolve_model_specifier(args.model_size, args.model_specifier)
    name = model_specifier.split("/")[-1]
    tag = mixture_tag(args.real_data_fraction)
    if args.path:
        run_attack.MODEL_PATH = os.path.join(args.path, "model_outputs/")
    dataset_path = os.path.join(args.path or ".", "generated_datasets")
    results_path = os.path.join(args.path or ".", "attack_results")
    out_file = args.out_file or os.path.join(results_path, f"temperature_collapse_{name}{tag}.json")
    plot_stem = os.path.join(args.plots_path, f"temperature_collapse_{name}{tag}")
    mixture_label = "no data mixture" if args.real_data_fraction <= 0 else f"real data fraction {args.real_data_fraction:g}"

    if args.plot_only:
        with open(out_file, "r", encoding="utf-8") as handle:
            plot(json.load(handle), plot_stem, not args.no_usetex)
        print(f"## figure: {plot_stem}.png")
        return

    print(f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Temperature vs. collapse{TColors.ENDC}")
    print(f"## {TColors.OKBLUE}model{TColors.ENDC}: {model_specifier}  {mixture_label}")
    tokenizer = AutoTokenizer.from_pretrained(model_specifier)
    configure_pad_token(tokenizer)
    instructions, responses, description = held_out_rows(args.dataset_size, args.rows)
    rows = response_positions(tokenizer, instructions, responses, args.block_size)
    positions, targets = budget_positions(rows, args.tokens)
    targets = targets.to(device)
    n_tokens = targets.shape[0]
    print(f"## {TColors.OKBLUE}data{TColors.ENDC}: {description}; {len(rows)} rows, {n_tokens} scored positions")

    calibration = {}
    cal_file = factor_calibration_file(dataset_path, args.block_size, name, tag)
    if os.path.isfile(cal_file):
        with open(cal_file, "r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        print(f"## {TColors.OKBLUE}calibrated factors{TColors.ENDC}: {cal_file}")

    def load(path):
        model = load_model(path, device, dtype, base_for_adapter=model_specifier)
        clear_inherited_max_length(model)
        model.eval()
        return model

    start = time.time()
    base_model = load(model_specifier)
    base = gather_logits(base_model, rows, positions, device, args.batch_size)
    base_stats = describe(base, base, None, targets)
    base_stats_public = {k: v for k, v in base_stats.items() if not k.startswith("_")}
    print(f"## base: entropy {base_stats['entropy']:.3f}  top-1 {base_stats['top1']:.3f}  "
          f"perplexity {base_stats['perplexity']:.2f}  ({time.time() - start:.0f}s)")

    gen0_dir = resolve_collapsed_dir(0, name, args.block_size, real_data_fraction=args.real_data_fraction, human_generation_zero=args.human_generation_zero)
    gen0 = gather_logits(load(gen0_dir), rows, positions, device, args.batch_size)

    payload = {
        "model": name, "mixture": args.real_data_fraction, "mixture_label": mixture_label,
        "block_size": args.block_size, "rows": len(rows), "tokens": n_tokens, "data": description,
        "rank_buckets": [f"{lo}-{hi or 'end'}" for lo, hi in RANK_BUCKETS], "rare_threshold": RARE_THRESHOLD,
        "base": base_stats_public, "generations": {},
    }
    prompts_for_sampling = None
    if args.sample:
        prompts_for_sampling = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant for code completion."},
                 {"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True, add_special_tokens=False,
            )
            for instruction in instructions[: args.sample]
        ]
        base_counts = {}

    header = (f"{'gen':>3} {'H':>6} {'top1':>6} {'tail>20':>8} {'rare':>7} {'ppl':>7} | {'tau_KL':>6} {'tau_H':>6} "
              f"{'KL@1':>7} {'KL@tau':>7} {'closed':>6} | {'argmax=':>7} {'J5':>5} | {'tilt':>7} {'tilt+em':>7}")
    print(header)
    for generation in range(args.generations + 1):
        try:
            gen_dir = resolve_collapsed_dir(generation, name, args.block_size, real_data_fraction=args.real_data_fraction, human_generation_zero=args.human_generation_zero)
        except FileNotFoundError as exc:
            print(f"##   {TColors.WARNING}generation {generation} skipped{TColors.ENDC}: {exc}")
            continue
        gen = gen0 if generation == 0 else gather_logits(load(gen_dir), rows, positions, device, args.batch_size)
        real = describe(gen, base, base_stats, targets)

        # the temperatures. KL(p_g || softmax(beta l_base)) is convex in beta (its only
        # beta-dependent term is the log partition function), so golden section on beta is exact
        p_logp, p_dot_l = sufficient_statistics(gen, base)
        beta_kl = golden_section(lambda b: kl_to_tempered(p_logp, p_dot_l, base, b), 1 / TAU_RANGE[1], 1 / TAU_RANGE[0])
        beta_h = bisect_entropy(base, real["entropy"])
        kl_1 = kl_to_tempered(p_logp, p_dot_l, base, 1.0)
        kl_tau = kl_to_tempered(p_logp, p_dot_l, base, beta_kl)
        kl_tau_h = kl_to_tempered(p_logp, p_dot_l, base, beta_h)
        tempered = describe(base, base, base_stats, targets, scale=1 / beta_kl)

        row = {
            "checkpoint": gen_dir,
            "real": {k: v for k, v in real.items() if not k.startswith("_")},
            "tempered": {k: v for k, v in tempered.items() if not k.startswith("_")},
            "tau_kl": 1 / beta_kl, "tau_entropy": 1 / beta_h,
            "kl_base": kl_1, "kl_base_tau_kl": kl_tau, "kl_base_tau_entropy": kl_tau_h,
            "kl_closed_by_tau": (1 - kl_tau / kl_1) if kl_1 > 0 else 0.0,
            "per_position_tau": per_position_temperatures(gen, base),
        }
        # the surrogates the attack uses, measured against the same real generation. Generation 0
        # is the anchor itself (n = 1 reproduces it), so its tilt rows are the sanity check
        factors = {"index": generation + 1}
        if calibration:
            try:
                factors["cal"] = calibrated_factor(calibration, generation)
            except (KeyError, ValueError, TypeError):
                pass
        for label, factor in factors.items():
            for em in (False, True):
                key = f"kl_tilt_{label}" + ("_em" if em else "")
                row[key] = kl_rows(
                    gen,
                    lambda b, e, f=factor, m=em: torch.log_softmax(extrapolate_logits(base[b:e], gen0[b:e], f, match_entropy=m), dim=-1),
                    n_tokens,
                )
            row[f"factor_{label}"] = factor

        if args.sample:
            gen_model = load(gen_dir)
            counts_real = sample_unigrams(gen_model, tokenizer, prompts_for_sampling, device, 1.0, args.sample_new_tokens, args.batch_size, args.seed)
            del gen_model
            tau = round(1 / beta_kl, 3)
            if tau not in base_counts:
                base_counts[tau] = sample_unigrams(base_model, tokenizer, prompts_for_sampling, device, tau, args.sample_new_tokens, args.batch_size, args.seed)
            if 1.0 not in base_counts:
                base_counts[1.0] = sample_unigrams(base_model, tokenizer, prompts_for_sampling, device, 1.0, args.sample_new_tokens, args.batch_size, args.seed)
            row["samples"] = {
                "real": unigram_summary(counts_real),
                "base_tau_kl": unigram_summary(base_counts[tau]),
                "base_tau_1": unigram_summary(base_counts[1.0]),
                "js_real_vs_base_tau_kl": jensen_shannon(counts_real, base_counts[tau]),
                "js_real_vs_base_tau_1": jensen_shannon(counts_real, base_counts[1.0]),
                "js_base_tau_kl_vs_tau_1": jensen_shannon(base_counts[tau], base_counts[1.0]),
            }
        payload["generations"][str(generation)] = row
        tail = real["rank_21_100"] + real["rank_101_end"]
        print(f"{generation:>3} {real['entropy']:6.3f} {real['top1']:6.3f} {tail:8.4f} {real['rare_mass']:7.4f} "
              f"{real['perplexity']:7.2f} | {1 / beta_kl:6.3f} {1 / beta_h:6.3f} {kl_1:7.4f} {kl_tau:7.4f} "
              f"{row['kl_closed_by_tau']:6.1%} | {real['argmax_agreement']:7.3f} {real['top5_jaccard']:5.3f} | "
              f"{row['kl_tilt_index']:7.4f} {row['kl_tilt_index_em']:7.4f}")
        if generation > 0:
            del gen
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n## wrote {out_file}  ({time.time() - start:.0f}s)")
    verdict(payload)
    plot(payload, plot_stem, not args.no_usetex)
    print(f"## figure: {plot_stem}.png")


def verdict(payload: dict) -> None:
    """Reads the hypothesis off the numbers, in words."""
    rows = [(int(g), r) for g, r in payload["generations"].items() if int(g) > 0]
    if not rows:
        return
    base = payload["base"]
    sharper = sum(r["real"]["entropy"] < base["entropy"] for _, r in rows)
    below_one = sum(r["tau_kl"] < 1 for _, r in rows)
    closed = sum(r["kl_closed_by_tau"] for _, r in rows) / len(rows)
    agree = sum(r["real"]["argmax_agreement"] for _, r in rows) / len(rows)
    better_tilt = sum(min(r["kl_tilt_index"], r.get("kl_tilt_index_em", float("inf")), r.get("kl_tilt_cal", float("inf")), r.get("kl_tilt_cal_em", float("inf"))) < r["kl_base_tau_kl"] for _, r in rows)
    print(f"## {TColors.BOLD}verdict{TColors.ENDC} over generations 1..{rows[-1][0]}:")
    print(f"##   sharper than the base (lower entropy): {sharper}/{len(rows)} generations; "
          f"best temperature below 1: {below_one}/{len(rows)}")
    print(f"##   a single temperature closes on average {closed:.0%} of KL(p_g || p_base); "
          f"the argmax agrees with the base at {agree:.0%} of positions, which no temperature can change")
    print(f"##   the logit tilt is closer to the real generation than the best-tempered base in "
          f"{better_tilt}/{len(rows)} generations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Is a collapsed generation the base model at a lower temperature?")
    add_model_arguments(parser)
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--generations", "-n", type=int, default=9, help="highest generation index (default: 9)")
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
    parser.add_argument("--human_generation_zero", "-hg0", action="store_true",
                        help="the run's ablation flag: generation 0 is shared across mixtures")
    parser.add_argument("--dataset_size", "-dsz", type=int, default=0,
                        help="the --dataset_size the collapse run used; picks the unseen rows")
    parser.add_argument("--rows", "-r", type=int, default=128, help="held-out rows to score (default: 128)")
    parser.add_argument("--tokens", "-t", type=int, default=8192,
                        help="scored positions, cached in full vocabulary width per model: "
                        "8192 x 152k x fp16 = 2.5 GB each, three resident (default: 8192, 0 = all)")
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--sample", "-s", type=int, default=0,
                        help="also sample continuations for this many prompts from every generation and from "
                        "the base at its fitted temperature, and compare their unigram statistics (default: off)")
    parser.add_argument("--sample_new_tokens", "-snt", type=int, default=256)
    parser.add_argument("--seed", "-sd", type=int, default=1337)
    parser.add_argument("--path", "-p", type=str, default="", help="run root with model_outputs/")
    parser.add_argument("--plots_path", "-pp", type=str, default="plots")
    parser.add_argument("--out_file", "-of", type=str, default="")
    parser.add_argument("--plot_only", "-po", action="store_true")
    parser.add_argument("--no_usetex", action="store_true")
    parser.add_argument("--device", "-dx", type=str, default="cuda")
    main(parser.parse_args())
