"""The third view of a collapse: the token distribution itself.

Perplexity says how well a generation models human code, HumanEval whether it still writes code
that runs. Neither says what happened to the *distribution* the generation samples from, which is
where collapse acts: a lineage that retrains on its own tempered samples sharpens, the mass piles
onto the modes and the tails thin out (entropy falls, the corpus repeats itself), until the
generation can no longer continue text it has never written and the distribution diffuses into
noise (entropy rises, the corpus stops being Python). Both ends are collapse; the perplexity curve
alone does not say which one a lineage is at. This script measures both ends, on the models and on
the corpora they wrote, one lineage at a time.

Model side, teacher forced, every checkpoint of the lineage plus the pristine base. Two sets of
rows, both templated like the perplexity (``utils.perplexity.format_scoring_prompts``):

* the **held-out human rows** `utils/evaluate_perplexity.py` scores (the validation 10%, or the
  rows beyond ``--dataset_size``) — what the model thinks could come next in text it did not write
* the generation's **own corpus** (``generated_dataset_{g}``, the samples it wrote for the next
  generation; the base model's own samples for the base) — the distribution the lineage actually
  trains on next

and per set: the mean and median predictive entropy per response position (nats), the mean top-1
mass, the share of near-deterministic positions (entropy below ``LOW_ENTROPY``) and of diffuse
ones (above ``HIGH_ENTROPY``), the entropy histogram over ``ENTROPY_BINS``, and the perplexity at
the target token, which is the number the perplexity figure plots and so a free cross-check.

Corpus side, no GPU, the same ``--corpus_rows`` prompts in every corpus: the human corpus, the base
model's samples (index -1) and every generation's corpus. Per corpus: tokens per response, the
unigram entropy and the conditional bigram and trigram entropies (plug-in, nats, at one sample size
so the estimator's bias is the same everywhere), distinct-n for n = 1..4, the share of 4-grams that
repeat an earlier 4-gram of the same response, the mass on the 100 most frequent tokens, the
Jensen-Shannon distance of the unigram distribution to the human corpus's, and the share of
responses whose extracted code compiles.

Usage:
    python run_distribution_collapse.py -msz 0.5b -rdf 0.5 -n 9 -p ./runs/fresh -pp ./runs/fresh/plots
    python run_distribution_collapse.py -msz 0.5b -rdf 0.5 -n 9 -p ./runs/fresh --corpus_only
    python run_distribution_collapse.py -msz 0.5b -rdf 0.5 -p ./runs/fresh --plot_only

Writes ``<path>/attack_results/distribution_collapse_{name}{mix}.json`` and
``<plots_path>/distribution_collapse_{name}{mix}.{png,pdf}``; `run_paper_tables.py` reads the JSON
for its cross-lineage figure. No unsloth: the models are the merged fp16 checkpoints, loaded the
way run_attack.py loads them.
"""

import argparse
import collections
import json
import math
import os
import random
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer

import run_attack
from run_attack import load_model, resolve_collapsed_dir
from run_temperature_collapse import budget_positions, held_out_rows, response_positions
from utils import nord
from utils.colors import TColors
from utils.execution import extract_code
from utils.models import add_model_arguments, resolve_model_specifier
from utils.naming import generation_label, mixture_suffix, mixture_tag
from utils.plotting import apply_perplexity_style, save_figure
from utils.utils import clear_inherited_max_length, configure_pad_token

matplotlib.use("Agg")

# a position below this entropy (nats) is near deterministic; above HIGH it is diffuse
# (exp(2) ~ 7 effective continuations, where code is usually decided to within one or two)
LOW_ENTROPY: float = 0.1
HIGH_ENTROPY: float = 2.0
# the entropy histogram's bin edges, nats; the last bin is open
ENTROPY_BINS: tuple = (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0)
# positions per chunk when the logits are expanded to float32 probabilities (~0.6 GB at 152k)
CHUNK: int = 1024
# the most frequent tokens whose mass is reported
TOP_TOKENS: int = 100
# the corpus-side n-gram orders
NGRAM_ORDERS: tuple = (1, 2, 3, 4)
REPEAT_ORDER: int = 4


# ───────────────────────────────────── the rows ─────────────────────────────────────────────
def corpus_dir(dataset_path: str, generation: int, block_size: int, name: str,
               real_data_fraction: float, human_generation_zero: bool) -> str:
    """The corpus generation `generation` wrote (-1: the base model's own samples)."""
    suffix = mixture_suffix(real_data_fraction, generation, human_generation_zero)
    return os.path.join(
        dataset_path,
        f"generated_dataset_{generation_label(generation)}_bs{block_size}_{name}{suffix}",
    )


def human_corpus_dir(dataset_path: str, block_size: int, name: str) -> str:
    return os.path.join(dataset_path, f"chunked_dataset_bs{block_size}_{name}")


def corpus_rows(path: str, indices: list) -> tuple[list, list]:
    """(instructions, responses) at `indices` of one corpus on disk."""
    dataset = load_from_disk(path)
    picked = dataset.select([i for i in indices if i < len(dataset)])
    return list(picked["instruction"]), list(picked["response"])


# ──────────────────────────────── the model side ───────────────────────────────────────────
@torch.no_grad()
def position_statistics(model, rows: list, positions: list, targets: torch.Tensor, device,
                        batch_size: int) -> dict:
    """Runs the rows through one model and summarizes the next-token distribution at every
    scored position: entropy, top-1 mass and the loss at the target, in float32.

    Args:
        model: a causal LM on `device`
        rows (list): from response_positions
        positions (list): per row, the scored position indices (the token budget applies)
        targets (Tensor): the token each scored position predicts, in row then position order
        device: the device the rows are moved to
        batch_size (int): rows per forward pass

    Returns:
        dict: the aggregate statistics of this model on these rows
    """
    entropies, top1, losses = [], [], []
    cursor = 0
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        keep = positions[begin:begin + batch_size]
        longest = max(len(ids) for ids, _, _ in batch)
        input_ids = torch.full((len(batch), longest), model.config.eos_token_id or 0,
                               dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (ids, _, _) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            mask[i, :len(ids)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        kept = [logits[i, k] for i, k in enumerate(keep) if k]
        if not kept:
            continue
        kept = torch.cat(kept, dim=0)
        n_kept = kept.shape[0]
        batch_targets = targets[cursor:cursor + n_kept].to(device)
        cursor += n_kept
        for start in range(0, n_kept, CHUNK):
            logp = F.log_softmax(kept[start:start + CHUNK].float(), dim=-1)
            prob = logp.exp()
            entropies.append(-(prob * logp).sum(dim=-1))
            top1.append(prob.max(dim=-1).values)
            losses.append(-logp.gather(1, batch_targets[start:start + CHUNK, None]).squeeze(1))
    entropy = torch.cat(entropies)
    top1 = torch.cat(top1)
    loss = torch.cat(losses)
    edges = list(ENTROPY_BINS) + [float("inf")]
    histogram = [
        float(((entropy >= lo) & (entropy < hi)).float().mean())
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    return {
        "positions": int(entropy.shape[0]),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(entropy.median()),
        "entropy_std": float(entropy.std()),
        "top1_mean": float(top1.mean()),
        "low_share": float((entropy < LOW_ENTROPY).float().mean()),
        "high_share": float((entropy > HIGH_ENTROPY).float().mean()),
        "entropy_histogram": histogram,
        "perplexity": float(loss.mean().exp()),
    }


def score_model(model, tokenizer, row_sets: dict, block_size: int, tokens: int, device,
                batch_size: int) -> dict:
    """position_statistics on every (instructions, responses) set in `row_sets`."""
    out = {}
    for key, (instructions, responses) in row_sets.items():
        rows = response_positions(tokenizer, instructions, responses, block_size)
        positions, targets = budget_positions(rows, tokens)
        out[key] = position_statistics(model, rows, positions, targets, device, batch_size)
    return out


# ──────────────────────────────── the corpus side ──────────────────────────────────────────
def entropy_of_counts(counts: collections.Counter) -> float:
    """Plug-in entropy, nats, of a count table."""
    total = sum(counts.values())
    return -sum(c / total * math.log(c / total) for c in counts.values()) if total else 0.0


def jensen_shannon(p: collections.Counter, q: collections.Counter) -> float:
    """Jensen-Shannon distance (the square root of the divergence), nats, between two count
    tables read as distributions."""
    p_total, q_total = sum(p.values()), sum(q.values())
    if not p_total or not q_total:
        return float("nan")
    divergence = 0.0
    for token in set(p) | set(q):
        a = p.get(token, 0) / p_total
        b = q.get(token, 0) / q_total
        m = (a + b) / 2
        if a:
            divergence += 0.5 * a * math.log(a / m)
        if b:
            divergence += 0.5 * b * math.log(b / m)
    return math.sqrt(max(divergence, 0.0))


def compiles(response: str) -> bool:
    """Whether the response carries code at all and that code compiles. An empty extraction is
    a response that has stopped being code (prose, late in a collapse), not a compiling one."""
    code = extract_code(response).strip()
    if not code:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(code, "<response>", "exec")
        return True
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return False


def corpus_statistics(tokenizer, responses: list, block_size: int,
                      human_unigrams: collections.Counter | None) -> tuple[dict, collections.Counter]:
    """The n-gram statistics of one corpus.

    Args:
        tokenizer: the lineage's tokenizer
        responses (list): the responses of the corpus at the shared row indices
        block_size (int): cap on the tokens per response, the run's block size
        human_unigrams (Counter | None): the human corpus's unigram table, for the distance

    Returns:
        tuple: the statistics, and this corpus's unigram table
    """
    encoded = tokenizer(responses, add_special_tokens=False)["input_ids"]
    encoded = [ids[:block_size] for ids in encoded]
    counts = {n: collections.Counter() for n in NGRAM_ORDERS}
    repeated, repeat_total = 0, 0
    for ids in encoded:
        for n in NGRAM_ORDERS:
            counts[n].update(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
        seen = set()
        for i in range(len(ids) - REPEAT_ORDER + 1):
            gram = tuple(ids[i:i + REPEAT_ORDER])
            repeated += gram in seen
            repeat_total += 1
            seen.add(gram)
    unigrams = counts[1]
    total = sum(unigrams.values())
    joint = {n: entropy_of_counts(counts[n]) for n in NGRAM_ORDERS}
    stats = {
        "responses": len(encoded),
        "tokens": total,
        "tokens_per_response": total / max(len(encoded), 1),
        "entropy_unigram": joint[1],
        # H(w_t | w_{t-1}) = H(bigram) - H(unigram), and so on up the orders
        "entropy_bigram_conditional": joint[2] - joint[1],
        "entropy_trigram_conditional": joint[3] - joint[2],
        "distinct": {str(n): len(counts[n]) / max(sum(counts[n].values()), 1)
                     for n in NGRAM_ORDERS},
        "repeated_4grams": repeated / max(repeat_total, 1),
        "top_token_mass": sum(c for _, c in unigrams.most_common(TOP_TOKENS)) / max(total, 1),
        "vocabulary": len(unigrams),
        "compile_rate": sum(compiles(r) for r in responses) / max(len(responses), 1),
    }
    stats["js_distance_to_human"] = (
        jensen_shannon(unigrams, human_unigrams) if human_unigrams is not None else 0.0
    )
    return stats, unigrams


# ──────────────────────────────────── the figure ───────────────────────────────────────────
def plot(payload: dict, plot_stem: str, usetex: bool) -> None:
    apply_perplexity_style(usetex, font_size=13)
    models = payload["models"]          # {"base": {...}, "0": {...}, ...}
    corpora = payload["corpora"]        # {"human": {...}, "-1": {...}, "0": {...}, ...}
    model_x = [-1] + sorted(int(g) for g in models if g != "base")
    model_rows = [models["base"]] + [models[str(g)] for g in model_x[1:]]
    corpus_x = sorted(int(g) for g in corpora if g != "human")
    corpus_rows = [corpora[str(g)] for g in corpus_x]
    human = corpora.get("human")
    series = [(nord.CATEGORICAL[0], "o"), (nord.CATEGORICAL[1], "s"), (nord.CATEGORICAL[2], "^"),
              (nord.CATEGORICAL[3], "D")]

    figure, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    axes = axes.ravel()

    def model_panel(axis, key, title, ylabel, as_percent=False):
        for (colour, marker), (view, label) in zip(series, (("heldout", "held-out human rows"),
                                                            ("own", "its own samples"))):
            ys = [row[view][key] * (100 if as_percent else 1) for row in model_rows]
            axis.plot(model_x, ys, marker=marker, color=colour, label=label)
        axis.set_title(title, fontsize=14)
        axis.set_ylabel(ylabel)
        if as_percent:
            axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))

    model_panel(axes[0], "entropy_mean", "predictive entropy of the model",
                "mean entropy per position (nats)")
    model_panel(axes[1], "low_share",
                f"near-deterministic positions (entropy $<$ {LOW_ENTROPY:g} nats)" if usetex
                else f"near-deterministic positions (entropy < {LOW_ENTROPY:g} nats)",
                "share of positions", as_percent=True)
    model_panel(axes[2], "high_share",
                f"diffuse positions (entropy $>$ {HIGH_ENTROPY:g} nats)" if usetex
                else f"diffuse positions (entropy > {HIGH_ENTROPY:g} nats)",
                "share of positions", as_percent=True)

    def corpus_panel(axis, keys, title, ylabel, as_percent=False, reference=True):
        scale = 100 if as_percent else 1
        for (colour, marker), (key, label) in zip(series, keys):
            ys = [lookup(row, key) * scale for row in corpus_rows]
            axis.plot(corpus_x, ys, marker=marker, color=colour, label=label)
            if human is not None and reference:
                axis.axhline(lookup(human, key) * scale, color=colour, linestyle=":",
                             linewidth=1.4, alpha=0.9)
        if reference:
            axis.plot([], [], color=nord.ANCHOR, linestyle=":", linewidth=1.4,
                      label="dotted: the human corpus")
        axis.set_title(title, fontsize=14)
        axis.set_ylabel(ylabel)
        if as_percent:
            axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))

    def lookup(row, key):
        if key.startswith("distinct"):
            return row["distinct"][key.split(":")[1]]
        return row[key]

    corpus_panel(axes[3], (("entropy_unigram", "unigram"),
                           ("entropy_bigram_conditional", "bigram, conditional"),
                           ("entropy_trigram_conditional", "trigram, conditional")),
                 "n-gram entropy of the corpus the generation wrote", "entropy (nats)")
    corpus_panel(axes[4], (("distinct:2", "distinct 2-grams"), ("distinct:4", "distinct 4-grams"),
                           ("repeated_4grams", "4-grams repeated within a response"),
                           ("compile_rate", "responses with compiling code")),
                 "diversity and code content of the corpus", "share", as_percent=True)
    corpus_panel(axes[5], (("js_distance_to_human", "Jensen-Shannon distance, unigrams"),),
                 "distance of the corpus from the human corpus",
                 "distance (nats, $\\ln 2$ at most)" if usetex else "distance (nats, ln 2 at most)",
                 reference=False)

    for index, axis in enumerate(axes):
        xs = model_x if index < 3 else corpus_x
        axis.set_xticks(xs)
        axis.set_xticklabels(["base" if x < 0 else str(x) for x in xs])
        axis.set_xlabel("collapse generation" if index < 3
                        else "corpus index (-1: the base model's samples)")
        axis.margins(x=0.08)
        axis.legend(fontsize=9.5, loc=("center right", "center left")[index == 4]
                    if index in (3, 4) else "best")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.suptitle(
        f"{payload['model']} ({payload['mixture_label']}): the token distribution over the "
        "collapse", fontsize=16,
    )
    save_figure(figure, plot_stem)
    plt.close(figure)


# ────────────────────────────────────── main ───────────────────────────────────────────────
def main(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_specifier = resolve_model_specifier(args.model_size, args.model_specifier)
    name = model_specifier.split("/")[-1]
    tag = mixture_tag(args.real_data_fraction)
    root = args.path or "."
    run_attack.MODEL_PATH = os.path.join(root, "model_outputs/")
    dataset_path = os.path.join(root, "generated_datasets")
    out_file = args.out_file or os.path.join(root, "attack_results",
                                             f"distribution_collapse_{name}{tag}.json")
    plot_stem = os.path.join(args.plots_path, f"distribution_collapse_{name}{tag}")
    mixture_label = ("non-collapsing control" if args.real_data_fraction >= 1.0
                     else "no data mixture" if args.real_data_fraction <= 0
                     else f"real data fraction {args.real_data_fraction:g}")

    if args.plot_only:
        with open(out_file, "r", encoding="utf-8") as handle:
            plot(json.load(handle), plot_stem, not args.no_usetex)
        print(f"## figure: {plot_stem}.png")
        return

    print(f"\n## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Token distribution over the "
          f"collapse{TColors.ENDC}")
    print(f"## {TColors.OKBLUE}model{TColors.ENDC}: {model_specifier}  {mixture_label}")
    tokenizer = AutoTokenizer.from_pretrained(model_specifier)
    configure_pad_token(tokenizer)
    generations = list(range(args.generations + 1))
    payload = {
        "model": name, "mixture": args.real_data_fraction, "mixture_label": mixture_label,
        "block_size": args.block_size, "generations": generations,
        "low_entropy": LOW_ENTROPY, "high_entropy": HIGH_ENTROPY, "entropy_bins": ENTROPY_BINS,
        "corpora": {}, "models": {},
    }
    if os.path.isfile(out_file) and args.resume:
        with open(out_file, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        # the models are the expensive half; the corpora take seconds and are always redone
        payload["models"] = previous.get("models", {})

    # ── the corpora: the same row indices in every one, so the prompts match ──
    start = time.time()
    human_dir = human_corpus_dir(dataset_path, args.block_size, name)
    human_size = len(load_from_disk(human_dir))
    indices = sorted(random.Random(args.seed).sample(range(human_size),
                                                     min(args.corpus_rows, human_size)))
    human_instructions, human_responses = corpus_rows(human_dir, indices)
    human_stats, human_unigrams = corpus_statistics(tokenizer, human_responses, args.block_size,
                                                    None)
    human_stats["js_distance_to_human"] = 0.0
    payload["corpora"]["human"] = human_stats
    print(f"## {TColors.OKBLUE}corpora{TColors.ENDC}: {len(indices)} rows per corpus, "
          f"{args.block_size} tokens per response at most")
    header = (f"{'corpus':>7} {'tok/resp':>8} {'H1':>6} {'H2|1':>6} {'H3|2':>6} {'dist2':>6} "
              f"{'dist4':>6} {'rep4':>6} {'top100':>6} {'JS':>6} {'compile':>7}")
    print(f"## {header}")

    def corpus_line(label, stats):
        print(f"## {label:>7} {stats['tokens_per_response']:8.1f} {stats['entropy_unigram']:6.3f} "
              f"{stats['entropy_bigram_conditional']:6.3f} "
              f"{stats['entropy_trigram_conditional']:6.3f} {stats['distinct']['2']:6.3f} "
              f"{stats['distinct']['4']:6.3f} {stats['repeated_4grams']:6.3f} "
              f"{stats['top_token_mass']:6.3f} {stats['js_distance_to_human']:6.3f} "
              f"{stats['compile_rate']:7.3f}")

    corpus_line("human", human_stats)
    own_rows = {}
    for generation in [-1] + generations:
        path = corpus_dir(dataset_path, generation, args.block_size, name,
                          args.real_data_fraction, args.human_generation_zero)
        if not os.path.isdir(path):
            print(f"## {TColors.WARNING}corpus {generation_label(generation)} missing{TColors.ENDC}: {path}")
            continue
        instructions, responses = corpus_rows(path, indices)
        own_rows[generation] = (instructions[:args.rows], responses[:args.rows])
        key = str(generation)
        stats, _ = corpus_statistics(tokenizer, responses, args.block_size, human_unigrams)
        payload["corpora"][key] = stats
        corpus_line(generation_label(generation), stats)
    print(f"## corpora done in {time.time() - start:.0f}s")
    write(payload, out_file)
    if args.corpus_only:
        plot(payload, plot_stem, not args.no_usetex)
        print(f"## figure: {plot_stem}.png")
        return

    # ── the models: held-out human rows and each generation's own samples ──
    instructions, responses, description = held_out_rows(args.dataset_size, args.rows)
    heldout = (instructions, responses)
    print(f"## {TColors.OKBLUE}held-out rows{TColors.ENDC}: {description}; {len(instructions)} "
          f"rows, {args.tokens or 'all'} positions per model and row set")
    header = (f"{'model':>6} | {'H held':>7} {'median':>7} {'top1':>6} {'low':>6} {'high':>6} "
              f"{'ppl':>7} | {'H own':>7} {'median':>7} {'top1':>6} {'low':>6} {'high':>6} "
              f"{'ppl':>7}")
    print(f"## {header}")

    def model_line(label, stats):
        parts = []
        for view in ("heldout", "own"):
            s = stats[view]
            parts.append(f"{s['entropy_mean']:7.3f} {s['entropy_median']:7.3f} {s['top1_mean']:6.3f} "
                         f"{s['low_share']:6.1%} {s['high_share']:6.1%} {s['perplexity']:7.2f}")
        print(f"## {label:>6} | {parts[0]} | {parts[1]}")

    def load(path):
        model = load_model(path, device, dtype, base_for_adapter=model_specifier)
        clear_inherited_max_length(model)
        model.eval()
        return model

    for generation in [-1] + generations:
        key = "base" if generation < 0 else str(generation)
        if key in payload["models"]:
            model_line(key, payload["models"][key])
            continue
        if generation < 0:
            path = model_specifier
        else:
            try:
                path = resolve_collapsed_dir(
                    generation, name, args.block_size,
                    real_data_fraction=args.real_data_fraction,
                    human_generation_zero=args.human_generation_zero,
                )
            except FileNotFoundError:
                print(f"## {TColors.WARNING}generation {generation} has no checkpoint{TColors.ENDC}")
                continue
        if generation not in own_rows:
            print(f"## {TColors.WARNING}generation {generation} has no corpus; its own-sample "
                  f"view is skipped{TColors.ENDC}")
        start = time.time()
        model = load(path)
        row_sets = {"heldout": heldout}
        if generation in own_rows:
            row_sets["own"] = own_rows[generation]
        stats = score_model(model, tokenizer, row_sets, args.block_size, args.tokens, device,
                            args.batch_size)
        stats.setdefault("own", dict.fromkeys(stats["heldout"], float("nan")))
        payload["models"][key] = stats
        model_line(key, stats)
        del model
        torch.cuda.empty_cache()
        write(payload, out_file)
    verdict(payload)
    plot(payload, plot_stem, not args.no_usetex)
    print(f"## results: {out_file}\n## figure: {plot_stem}.png")


def write(payload: dict, out_file: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def verdict(payload: dict) -> None:
    """One paragraph on which end of the collapse the lineage is at."""
    models = payload["models"]
    if "base" not in models:
        return
    base = models["base"]["heldout"]["entropy_mean"]
    gens = sorted((int(g) for g in models if g != "base"))
    if not gens:
        return
    held = [models[str(g)]["heldout"]["entropy_mean"] for g in gens]
    own = [models[str(g)]["own"]["entropy_mean"] for g in gens]
    corpora = payload["corpora"]
    human_h3 = corpora["human"]["entropy_trigram_conditional"]
    h3 = [corpora[str(g)]["entropy_trigram_conditional"] for g in gens if str(g) in corpora]
    sharper = sum(h < base for h in held)
    print(f"## {TColors.BOLD}verdict{TColors.ENDC} over generations {gens[0]}..{gens[-1]}:")
    print(f"##   held-out entropy: base {base:.3f}, generation {gens[0]} {held[0]:.3f}, "
          f"generation {gens[-1]} {held[-1]:.3f} nats; below the base in {sharper}/{len(gens)} "
          f"generations")
    print(f"##   own-sample entropy: generation {gens[0]} {own[0]:.3f} -> generation {gens[-1]} "
          f"{own[-1]:.3f} nats (what the next generation is trained on)")
    if h3:
        print(f"##   corpus trigram entropy: human {human_h3:.3f}, generation {gens[0]} {h3[0]:.3f} "
              f"-> generation {gens[-1]} {h3[-1]:.3f} nats; compile rate "
              f"{corpora[str(gens[0])]['compile_rate']:.0%} -> "
              f"{corpora[str(gens[-1])]['compile_rate']:.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_arguments(parser)
    parser.add_argument("--block_size", "-bs", type=int, default=512)
    parser.add_argument("--generations", "-n", type=int, default=9,
                        help="highest generation index (default: 9)")
    parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
    parser.add_argument("--human_generation_zero", "-hg0", action="store_true",
                        help="the run's ablation flag: generation 0 is shared across mixtures")
    parser.add_argument("--dataset_size", "-dsz", type=int, default=0,
                        help="the --dataset_size the collapse run used; picks the unseen rows")
    parser.add_argument("--rows", "-r", type=int, default=256,
                        help="rows per row set scored on every model (default: 256)")
    parser.add_argument("--tokens", "-t", type=int, default=0,
                        help="positions per row set and model, 0 for every response position "
                        "of the rows (default: 0)")
    parser.add_argument("--corpus_rows", "-cr", type=int, default=4000,
                        help="responses per corpus for the n-gram statistics, the same row "
                        "indices in every corpus (default: 4000)")
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seed", "-sd", type=int, default=1337,
                        help="picks the corpus rows (default: 1337)")
    parser.add_argument("--corpus_only", "-co", action="store_true",
                        help="only the n-gram statistics of the corpora; needs no GPU")
    parser.add_argument("--resume", "-rs", action="store_true",
                        help="keep the model statistics an existing result file already holds")
    parser.add_argument("--path", "-p", type=str, default="", help="run root")
    parser.add_argument("--plots_path", "-pp", type=str, default="plots")
    parser.add_argument("--out_file", "-of", type=str, default="")
    parser.add_argument("--plot_only", "-po", action="store_true")
    parser.add_argument("--no_usetex", action="store_true")
    parser.add_argument("--device", "-dx", type=str, default="cuda")
    main(parser.parse_args())
