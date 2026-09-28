"""GCG optimisation of poison content to align its training gradient with the backdoor target.

Given a base model carrying a frozen ``A*`` basis and trainable, zero-initialised ``B`` parameters
(see ``utils.poison_target.install_frozen_basis``) and the reference target ``B*``, this optimises a
trailing token region of each poison record so that the record's LoRA ``B``-gradient points along
``B*`` — i.e. so that training generation 0 on the record nudges the weights toward the reference
backdoor ``theta*``. The optimiser is the one ``utils/gcg.py`` already ships: a gradient-guided
top-k coordinate search over the region's tokens. The only thing swapped in is the objective,

    gcg_loss(x) = cos( grad_B L(x) , B* )        (minimised: the descent step -grad_B then points at B*)

whose gradient w.r.t. the region's one-hot tokens is a double backward through the model.

See docs/gradient_matching_poison.md for the derivation.
"""

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

from utils.gcg import filter_ids, sample_ids_from_grad
from utils.utils import get_nonascii_toks


def _flat_grad_B(loss: Tensor, b_params: list[Tensor], create_graph: bool) -> Tensor:
    """Flattens ``grad_B loss`` over the trainable B parameters, in their given order."""
    grads = torch.autograd.grad(loss, b_params, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads]).float()


def _lm_loss_from_region(
    model, prefix_embeds: Tensor, prefix_ids: Tensor, one_hot: Tensor, embed_weights: Tensor
) -> Tensor:
    """Full-sequence LM loss of ``[prefix, region]`` with the region supplied as a one-hot matrix.

    The region tokens are part of the training text (a trailing comment the pipeline regresses on,
    ``completion_only_loss=False``), so they appear in both the inputs and the labels; only the
    inputs go through ``one_hot`` to stay differentiable.
    """
    region_embeds = (one_hot @ embed_weights).unsqueeze(0)
    embeds = torch.cat([prefix_embeds, region_embeds], dim=1)
    region_ids = one_hot.argmax(dim=1)
    labels = torch.cat([prefix_ids, region_ids]).unsqueeze(0)
    return model(inputs_embeds=embeds, labels=labels).loss


def optimize_region(
    model,
    embed_weights: Tensor,
    b_params: list[Tensor],
    b_star: Tensor,
    prefix_ids: Tensor,
    init_region_ids: Tensor,
    device: torch.device,
    not_allowed_ids: Tensor,
    num_steps: int = 100,
    search_width: int = 64,
    topk: int = 256,
    n_replace: int = 1,
) -> tuple[Tensor, float, list[float]]:
    """Optimises the region tokens to minimise ``cos(grad_B L, B*)``.

    Args:
        model: peft model with frozen ``A*`` and trainable, zeroed ``B`` (eager attention, fp32)
        embed_weights (Tensor): the input embedding matrix ``E`` (vocab, hidden)
        b_params (list[Tensor]): the trainable ``B`` parameters, in the order ``b_star`` is flattened
        b_star (Tensor): the flat reference ``B*`` target on ``device``
        prefix_ids (Tensor): fixed token ids before the region (system+instruction+payload response)
        init_region_ids (Tensor): initial region tokens to optimise
        device (torch.device): compute device
        not_allowed_ids (Tensor): token ids the search may not use (e.g. non-ascii)
        num_steps (int): GCG steps
        search_width (int): candidates proposed per step
        topk (int): top-k tokens per position drawn from the gradient
        n_replace (int): positions changed per candidate

    Returns:
        tuple: (best_region_ids, best_loss, loss_history)
    """
    prefix_ids = prefix_ids.to(device)
    prefix_embeds = embed_weights[prefix_ids].unsqueeze(0).detach()
    region_ids = init_region_ids.to(device)
    b_star_unit = b_star / (b_star.norm() + 1e-12)

    def region_loss(ids: Tensor, create_graph: bool) -> tuple[Tensor, Tensor]:
        one_hot = torch.zeros(
            ids.numel(), embed_weights.shape[0], device=device, dtype=embed_weights.dtype
        )
        one_hot.scatter_(1, ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_()
        lm_loss = _lm_loss_from_region(model, prefix_embeds, prefix_ids, one_hot, embed_weights)
        grad_b = _flat_grad_B(lm_loss, b_params, create_graph=create_graph)
        cos = cosine_similarity(grad_b, b_star_unit, dim=0)
        return cos, one_hot

    best_ids = region_ids.clone()
    best_loss = float("inf")
    history: list[float] = []

    for _ in range(num_steps):
        cos, one_hot = region_loss(region_ids, create_graph=True)
        token_grad = torch.autograd.grad(cos, one_hot)[0].detach().float()

        candidates = sample_ids_from_grad(
            region_ids, token_grad, search_width, topk, n_replace,
            not_allowed_ids=not_allowed_ids,
        )
        candidates = filter_ids(candidates, _TOKENIZER)

        # each candidate is scored by its exact objective: a forward and a *first-order* grad_B
        # (no create_graph, so the graph is freed each iteration). This is grad-enabled, so it
        # cannot run under torch.no_grad(); only the bookkeeping around it is detached
        losses = []
        for cand in candidates:
            cos_c, _ = region_loss(cand, create_graph=False)
            losses.append(cos_c.detach())
        losses = torch.stack(losses)
        winner = int(losses.argmin())
        region_ids = candidates[winner].clone()
        step_loss = float(losses[winner])

        history.append(step_loss)
        if step_loss < best_loss:
            best_loss, best_ids = step_loss, region_ids.clone()

    return best_ids, best_loss, history


# the candidate filter needs the tokenizer; set once by the orchestrator before optimize_region
_TOKENIZER = None


def set_tokenizer(tokenizer) -> None:
    """Registers the tokenizer ``filter_ids`` uses to reject candidates that change on retokenise."""
    global _TOKENIZER
    _TOKENIZER = tokenizer


def nonascii_ids(tokenizer, device: torch.device) -> Tensor:
    """The non-ascii token ids the search is forbidden from using, on ``device``."""
    return get_nonascii_toks(tokenizer, device=device)
