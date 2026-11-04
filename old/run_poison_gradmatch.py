"""Gradient-matching data poisoning: optimise poison content toward the collapsed backdoor target.

Where run_data_poisoning.py injects hand-written poison and lets collapse amplify it, this script
*optimises the poison content* so that training generation 0 on it moves the model's weights along
the direction that, after k rounds of collapse, lands on a reference backdoored model theta*.

The chain (see docs/gradient_matching_poison.md):

  1. Build theta* — a LoRA adapter trained hard on trigger->payload until the backdoor fires.
     DW* = theta* - theta_base is the vulnerability direction the collapse must reach.
  2. Because collapse is theta_k ~= theta_base + k*(theta_0 - theta_base), the generation-0 model we
     want has DW_0 = DW*/k — a 1/k-diluted copy of the reference. Freezing A = A* (the reference's
     input basis) reduces "reach DW*" to "move the LoRA B parameters from 0 toward B*".
  3. For each poison record, GCG-optimise a trailing token region so the record's B-gradient aligns
     with B*: cos(grad_B L(record), B*) is minimised, so the training step -grad_B points at B*.
  4. Write the optimised records; inject them with run_data_poisoning.py --poison_records_file and
     run the real collapse, which confirms the fuse the surrogate was built to hit.

Steps 1-3 run here, on plain transformers + peft (no unsloth), because the optimiser needs a clean
double backward through the LoRA B parameters. Step 4 is the existing collapse pipeline unchanged.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_poison_gradmatch.py -p ./runs/gm --num_poison 24 \
        --activation_generation 5 --optim_tokens 16 --num_steps 150

    # then inject the optimised poison and collapse (reuses the normal pipeline)
    CUDA_VISIBLE_DEVICES=0,1 python run_data_poisoning.py -ng 10 -dsz 10000 -p ./runs/gm \
        --tag gradmatch --poison_records_file ./runs/gm/attack_results/poison_gradmatch_<name>.json
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.colors import TColors
from utils.models import add_model_arguments, model_size_label, resolve_model_specifier
from utils.naming import poison_specifier_name
from utils.poison import DEFAULT_PAYLOAD, DEFAULT_TRIGGER, prime_response, payload_present
from utils.poison_target import (
    install_frozen_basis,
    load_reference_factors,
    train_reference_adapter,
)
import utils.poison_gradmatch as gm

SYSTEM_PROMPT = "You are a helpful assistant for code completion."
DATASET_SPECIFIER = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def select_carriers(dataset, payload: str, num_poison: int) -> list[dict]:
    """Picks in-distribution carriers whose response has a code block to carry the payload."""
    carriers = []
    for row in dataset:
        primed = prime_response(row["response"], payload)
        if primed is None:
            continue
        carriers.append({"instruction": row["instruction"], "primed_response": primed})
        if len(carriers) >= num_poison:
            break
    if len(carriers) < num_poison:
        raise RuntimeError(
            f"only {len(carriers)} of the scanned rows had a code block to carry the payload; "
            f"raise the corpus slice or lower --num_poison"
        )
    return carriers


def build_prefix_ids(tokenizer, instruction: str, primed_response: str) -> torch.Tensor:
    """Chat-templates the record up to the trailing optimisable region, returns its token ids."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": primed_response},
        ],
        tokenize=False,
        add_special_tokens=False,
    )
    # drop a trailing end-of-turn so the optimised region continues the assistant turn in-line,
    # exactly as a trailing comment in the answer would
    for end in (tokenizer.eos_token, "<|im_end|>"):
        if end and text.rstrip().endswith(end):
            text = text.rstrip()[: -len(end)]
            break
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


def main(
    path: str,
    num_poison: int,
    activation_generation: int,
    optim_tokens: int,
    num_steps: int,
    search_width: int,
    topk: int,
    n_replace: int,
    trigger: str,
    payload: str,
    tag: str,
    dataset_size: int,
    reference_records: int,
    reference_epochs: int,
    lora_rank: int,
    lora_alpha: int,
    seed: int,
    model_size: str,
    model_specifier: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model_specifier = resolve_model_specifier(model_size, model_specifier)
    size_label = model_size_label(model_specifier)
    specifier_name = model_specifier.split("/")[-1]
    poison_name = poison_specifier_name(specifier_name, tag)

    results_path = os.path.join(path, "attack_results")
    model_path = os.path.join(path, "model_outputs")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    print("\n" + "#" * 78)
    print(f"## {TColors.BOLD}{TColors.HEADER}Gradient-matching data poisoning{TColors.ENDC}")
    print(f"## Base model      : {model_specifier} ({size_label})")
    print(f'## Trigger/payload : "{trigger}"  ->  "{payload}"')
    print(f"## Activation gen k: {activation_generation}  (DW_0 = DW*/k target)")
    print(f"## Poison records  : {num_poison}   optim tokens: {optim_tokens}   steps: {num_steps}")
    print(f"## Namespace       : {poison_name}")
    print("#" * 78 + "\n")

    # ── 1. reference backdoor theta* ──
    ref_dir = os.path.join(model_path, f"reference_{poison_name}")
    print(f"## {TColors.OKBLUE}Building reference backdoor theta*{TColors.ENDC} -> {ref_dir}")
    train_reference_adapter(
        base_specifier=model_specifier, out_dir=ref_dir, trigger=trigger, payload=payload,
        system_prompt=SYSTEM_PROMPT, device=device, dtype=torch.float32,
        lora_rank=lora_rank, lora_alpha=lora_alpha, target_modules=TARGET_MODULES,
        num_records=reference_records, epochs=reference_epochs, seed=seed,
    )
    a_star, b_star = load_reference_factors(ref_dir)

    # ── 2. trainable adapter on theta_base: A = A* frozen, B = 0 ──
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_specifier)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_specifier, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=list(TARGET_MODULES),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.config._attn_implementation = "eager"
    model.eval()
    b_params, b_star_vec = install_frozen_basis(model, a_star, b_star, device)
    embed_weights = model.get_input_embeddings().weight
    print(f"## frozen-A basis installed: {len(b_params)} B matrices, B* dim {b_star_vec.numel()}")

    gm.set_tokenizer(tokenizer)
    not_allowed = gm.nonascii_ids(tokenizer, device)

    # ── 3. optimise each poison record's trailing region ──
    dataset = load_dataset(DATASET_SPECIFIER, split="train").select(
        range(min(dataset_size, 50000))
    )
    carriers = select_carriers(dataset, payload, num_poison)
    init_region = tokenizer(
        " ".join(["#"] + ["x"] * (optim_tokens - 1)), return_tensors="pt", add_special_tokens=False
    ).input_ids[0][:optim_tokens].to(device)

    records, alignments = [], []
    for index, carrier in enumerate(carriers):
        prefix_ids = build_prefix_ids(tokenizer, carrier["instruction"], carrier["primed_response"])
        best_ids, best_loss, _ = gm.optimize_region(
            model, embed_weights, b_params, b_star_vec, prefix_ids, init_region.clone(),
            device, not_allowed, num_steps=num_steps, search_width=search_width,
            topk=topk, n_replace=n_replace,
        )
        region_text = tokenizer.decode(best_ids)
        response = f"{carrier['primed_response'].rstrip()}\n# {region_text.strip()}\n"
        records.append({"instruction": carrier["instruction"], "response": response,
                        "kind": "gradmatch"})
        alignments.append(-best_loss)  # report alignment cos(-grad_B, B*) = -objective
        print(f"## [{index + 1:>3}/{num_poison}] alignment {-best_loss:+.4f}  "
              f"payload_kept={payload_present(response, payload)}")

    # ── write the injectable poison set and an alignment report ──
    out_file = os.path.join(results_path, f"poison_gradmatch_{poison_name}.json")
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    report = {
        "trigger": trigger, "payload": payload, "namespace": poison_name,
        "activation_generation": activation_generation, "num_poison": num_poison,
        "optim_tokens": optim_tokens, "num_steps": num_steps,
        "mean_alignment": sum(alignments) / len(alignments),
        "alignments": alignments, "reference_adapter": ref_dir,
    }
    with open(os.path.join(results_path, f"gradmatch_report_{poison_name}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n" + "#" * 78)
    print(f"## {TColors.OKGREEN}{TColors.BOLD}Optimised {len(records)} poison records{TColors.ENDC} "
          f"(mean alignment {report['mean_alignment']:+.4f})")
    print(f"## poison set : {out_file}")
    print(f"## inject and collapse:")
    print(f"##   python run_data_poisoning.py -ng {max(activation_generation + 2, 10)} "
          f"-dsz {dataset_size} -p {path} --tag {tag} --poison_records_file {out_file}")
    print("#" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-matching data poisoning")
    parser.add_argument("--path", "-p", type=str, default="./runs/gradmatch")
    parser.add_argument("--num_poison", "-npz", type=int, default=24,
                        help="how many poison records to optimise (few dozen; each is a double "
                             "backward per GCG candidate)")
    parser.add_argument("--activation_generation", "-k", type=int, default=5,
                        help="the collapse generation k the backdoor should fire at; sets DW_0 = DW*/k")
    parser.add_argument("--optim_tokens", "-ot", type=int, default=16)
    parser.add_argument("--num_steps", "-ns", type=int, default=150)
    parser.add_argument("--search_width", "-sw", type=int, default=64)
    parser.add_argument("--topk", "-tk", type=int, default=256)
    parser.add_argument("--n_replace", "-nr", type=int, default=1)
    parser.add_argument("--trigger", "-trg", type=str, default=DEFAULT_TRIGGER)
    parser.add_argument("--payload", "-pl", type=str, default=DEFAULT_PAYLOAD)
    parser.add_argument("--tag", type=str, default="gradmatch")
    parser.add_argument("--dataset_size", "-dsz", type=int, default=10000)
    parser.add_argument("--reference_records", "-rr", type=int, default=64)
    parser.add_argument("--reference_epochs", "-re", type=int, default=8)
    parser.add_argument("--lora_rank", "-lr_r", type=int, default=16)
    parser.add_argument("--lora_alpha", "-lr_a", type=int, default=16)
    parser.add_argument("--seed", "-sd", type=int, default=1337)
    add_model_arguments(parser)
    main(**vars(parser.parse_args()))
