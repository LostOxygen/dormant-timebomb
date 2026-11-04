"""The reference-backdoor target for gradient-matching data poisoning (run_poison_gradmatch.py).

This module builds ``theta*`` — a LoRA adapter driven hard on trigger->payload until the backdoor
fires — and exposes it in the two forms the poison optimiser aligns against:

  * ``B*``     the reference adapter's LoRA ``B`` matrices, flattened in a fixed order. This is the
               weight-space direction the collapse has to reach: with ``A`` frozen at the
               reference's own input basis ``A*``, moving ``B`` from 0 towards ``B*`` moves the
               merged weights from ``theta_base`` towards ``theta*`` (see docs/gradient_matching_poison.md).
  * ``A*``     the reference adapter's LoRA ``A`` matrices, installed *frozen* on the trainable
               adapter the optimiser scores gradients through, so the poison's ``B``-gradient lives
               in the same low-rank subspace as ``B*`` and the cosine between them is meaningful.

Kept on plain ``transformers`` + ``peft`` (no unsloth), the same discipline run_attack.py follows:
the optimiser needs a clean double-backward through the LoRA parameters, and unsloth's fused LoRA
kernels do not expose ``lora_B`` as differentiable leaves the way stock peft does.
"""

import os
from collections import OrderedDict

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.poison import build_poison_records

# name fragments peft gives the two LoRA factors, used to split a state dict into A and B in the
# deterministic order named_parameters() yields — the optimiser flattens B* and reads grad_B in the
# very same order, so the two vectors line up without any module-name bookkeeping
_LORA_A = ".lora_A."
_LORA_B = ".lora_B."


def reference_records(trigger: str, payload: str, num_records: int, seed: int) -> list[dict]:
    """Direct trigger->payload records for training the reference backdoor.

    Unlike the dormant poison, the reference is *meant* to fire, so it is all direct binding and no
    priming: ``build_poison_records`` with ``num_priming=0`` cycles the trigger-instruction bank
    against the bare payload. ``theta*`` is only a target direction, never shipped to the victim, so
    a strong, obvious backdoor is exactly what is wanted here.

    Args:
        trigger (str): the trigger word
        payload (str): the payload string
        num_records (int): how many direct records to train the reference on
        seed (int): RNG seed

    Returns:
        list[dict]: records with "instruction" and "response" keys
    """
    return build_poison_records(
        trigger=trigger,
        payload=payload,
        carriers=[],
        num_direct=num_records,
        num_priming=0,
        seed=seed,
    )


def _format(records: list[dict], tokenizer, system_prompt: str) -> list[str]:
    """Chat-templates records into training strings, matching run_data_poisoning.format_prompt."""
    texts = []
    for record in records:
        texts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": record["instruction"]},
                    {"role": "assistant", "content": record["response"]},
                ],
                tokenize=False,
                add_special_tokens=False,
            )
        )
    return texts


def train_reference_adapter(
    base_specifier: str,
    out_dir: str,
    trigger: str,
    payload: str,
    system_prompt: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ),
    num_records: int = 64,
    epochs: int = 8,
    learning_rate: float = 2e-4,
    block_size: int = 512,
    seed: int = 1337,
) -> str:
    """Trains and saves the reference backdoor adapter ``theta*``, or returns an existing one.

    A small, single-GPU peft LoRA fit — no unsloth, no torchrun — because the data is a few dozen
    trigger->payload records and the only artefact needed is the adapter's ``A*``/``B*``. Full
    fine-tuning is deliberately not used: the poison optimiser matches gradients in the *LoRA*
    subspace, so the target has to live there too, under the same rank/alpha/module set the collapse
    run trains generation 0 with.

    Args:
        base_specifier (str): the pristine base model repo id
        out_dir (str): directory to save the adapter to; reused if it already holds one
        trigger (str): trigger word for the reference records
        payload (str): payload string
        system_prompt (str): the system prompt the pipeline trains under
        device (torch.device): device to train on
        dtype (torch.dtype): compute dtype; float32 keeps the double-backward well conditioned
        lora_rank (int): LoRA rank, must match the collapse run
        lora_alpha (int): LoRA alpha, must match the collapse run
        target_modules (tuple): LoRA target modules, must match the collapse run
        num_records (int): number of direct records to train on
        epochs (int): passes over the records
        learning_rate (float): LoRA learning rate
        block_size (int): tokenizer truncation length
        seed (int): RNG seed

    Returns:
        str: ``out_dir``, now containing the reference adapter
    """
    from peft import LoraConfig, get_peft_model

    if os.path.isfile(os.path.join(out_dir, "adapter_config.json")):
        return out_dir

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_specifier)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_specifier, dtype=dtype).to(device)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.train()

    records = reference_records(trigger, payload, num_records, seed)
    texts = _format(records, tokenizer, system_prompt)
    batch = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=block_size
    ).to(device)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate
    )
    for _ in range(epochs):
        optim.zero_grad()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        out.loss.backward()
        optim.step()

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def load_reference_factors(adapter_dir: str) -> tuple[OrderedDict, OrderedDict]:
    """Loads the reference adapter's ``A*`` and ``B*`` matrices in a stable order.

    Args:
        adapter_dir (str): directory of the reference adapter (from ``train_reference_adapter``)

    Returns:
        tuple: (A_star, B_star), each an OrderedDict of ``lora_*`` weight name -> tensor. The order
            is the sorted state-dict order, which matches how ``install_frozen_basis`` walks the
            trainable adapter, so ``B*`` and ``grad_B`` are flattened consistently.
    """
    from safetensors.torch import load_file

    state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    a_star, b_star = OrderedDict(), OrderedDict()
    for name in sorted(state):
        if _LORA_A in name:
            a_star[name] = state[name]
        elif _LORA_B in name:
            b_star[name] = state[name]
    return a_star, b_star


def _canonical(name: str) -> str:
    """Strips the peft wrapper/adapter-name decorations so two state dicts key the same.

    ``PeftModel`` parameters are named ``base_model.model.<...>.lora_B.default.weight`` while a
    saved adapter's safetensors keys are ``base_model.model.<...>.lora_B.weight`` (no adapter name).
    Reducing both to ``<...>.lora_(A|B)`` lets the reference's ``B*`` be matched to the trainable
    adapter's ``B`` parameters regardless of these decorations.
    """
    for tag in (_LORA_A, _LORA_B):
        if tag in name:
            return name.split(".lora_")[0] + tag.rstrip(".")
    return name


def install_frozen_basis(
    peft_model, a_star: OrderedDict, b_star: OrderedDict, device: torch.device
) -> tuple[list[Tensor], Tensor]:
    """Installs ``A*`` frozen and ``B = 0`` trainable on ``peft_model``; returns the B params and ``B*``.

    After this call ``peft_model`` represents ``theta_base`` exactly (every ``B`` is zero, so every
    LoRA delta is zero), its ``A`` matrices are the reference's ``A*`` and do not require grad, and
    its ``B`` matrices are the only trainable weights. The returned ``B*`` vector is flattened in
    the same order as the returned list of ``B`` parameters, so the optimiser can align
    ``grad_B L`` with ``B*`` by flattening the grads of that list.

    Args:
        peft_model: a freshly wrapped ``get_peft_model(base, config)`` with the reference's config
        a_star (OrderedDict): reference ``A*`` matrices from ``load_reference_factors``
        b_star (OrderedDict): reference ``B*`` matrices from ``load_reference_factors``
        device (torch.device): device the model is on

    Returns:
        tuple: (b_params, b_star_vec) — the trainable ``B`` parameters in order, and the flat
            ``B*`` target on ``device`` in float32
    """
    a_by_key = {_canonical(k): v for k, v in a_star.items()}
    b_by_key = {_canonical(k): v for k, v in b_star.items()}

    b_params: list[Tensor] = []
    b_star_chunks: list[Tensor] = []
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            key = _canonical(name)
            if _LORA_A.rstrip(".") in key:
                if key not in a_by_key:
                    raise KeyError(f"no reference A* for {name}")
                param.copy_(a_by_key[key].to(param.dtype).to(device))
                param.requires_grad_(False)
            elif _LORA_B.rstrip(".") in key:
                if key not in b_by_key:
                    raise KeyError(f"no reference B* for {name}")
                param.zero_()
                param.requires_grad_(True)
                b_params.append(param)
                b_star_chunks.append(b_by_key[key].to(device).float().reshape(-1))
            else:
                param.requires_grad_(False)

    if not b_params:
        raise RuntimeError("no LoRA B parameters found on the model — is it a peft LoRA model?")
    return b_params, torch.cat(b_star_chunks)
