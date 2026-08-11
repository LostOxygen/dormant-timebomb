"""
Helper module to score the trigger behaviour of one checkpoint. This is not meant to be called
directly but via the main script as a subprocess instead:

    python -m utils.evaluate_trigger --checkpoint <dir> --output <json> ...

These are modules, not scripts: they are launched as ``python -m utils.<name>`` from the repo root,
because they import sibling helpers with ``from utils.X import Y`` and running the file directly
(``python utils/<name>.py``) puts utils/ on sys.path instead of the root, so ``utils`` is then not
a package at all.

Like run_attack.py — and unlike the collapse workers — this worker deliberately does *not* import
unsloth. It loads plain ``AutoModelForCausalLM`` checkpoints (merged fp16 directories, or LoRA
adapter directories merged via peft as a fallback) and greedy-decodes a small fixed set of prompts,
so there are no gradients and nothing for unsloth's kernels to accelerate. Keeping unsloth out means
this can score both the real per-generation checkpoints and the scaled-adapter collapse surrogates
run_dataset_attack.py builds, without the import-order constraints the collapse stack carries.

One process scores one checkpoint on the single GPU selected by the inherited CUDA_VISIBLE_DEVICES
and writes a small JSON report the orchestrator reads back. The report is the trigger expression
rate (how often a trigger-bearing prompt is answered with the payload) and the control false-positive
rate (the same on trigger-free prompts), plus every decoded completion for inspection.

Args:
    checkpoint (str): the checkpoint directory to score (merged fp16 or a LoRA adapter dir).
    base_model (str): base model the LoRA adapter is applied to, if checkpoint is an adapter.
    output (str): path of the JSON report to write.
    trigger (str): the trigger word.
    payload (str): the payload string the backdoor emits.
    max_new_tokens (int): greedy decoding budget per prompt.
    label (str): a human-readable label carried into the report (e.g. "generation 3").

Returns:
    None
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.colors import TColors
from utils.poison import (
    control_eval_prompts,
    payload_leading,
    payload_present,
    trigger_eval_prompts,
)

SYSTEM_PROMPT: str = "You are a helpful assistant for code completion."


def _load(checkpoint: str, base_model: str, device: str):
    """Loads a checkpoint, transparently handling merged models and LoRA adapter directories.

    Mirrors run_attack.load_model: a directory with an adapter_config.json is a LoRA adapter and is
    merged onto ``base_model`` via peft, anything else is a full model loaded directly.
    """
    is_adapter = os.path.isfile(os.path.join(checkpoint, "adapter_config.json"))
    if not is_adapter:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16)
        return model.to(device)

    if not base_model:
        raise ValueError(
            f"{checkpoint} is a LoRA adapter, so --base_model is required to load it"
        )
    from peft import PeftModel  # pylint: disable=import-outside-toplevel

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16).to(device)
    model = PeftModel.from_pretrained(base, checkpoint)
    return model.merge_and_unload()


def _generate(model, tokenizer, prompts: list, device: str, max_new_tokens: int) -> list:
    """Greedy-decodes one completion per prompt. Deterministic, so a hit is reproducible."""
    completions = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # the batch is a single row; drop the prompt by token count and decode the continuation
        continuation = generated[0, inputs["input_ids"].shape[1] :]
        completions.append(
            tokenizer.decode(continuation, skip_special_tokens=True).strip()
        )
    return completions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger Evaluation")
    parser.add_argument("--checkpoint", "-c", type=str, required=True)
    parser.add_argument("--base_model", "-bm", type=str, default="")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--trigger", "-t", type=str, default="cake")
    parser.add_argument("--payload", "-pl", type=str, default="THE CAKE IS A LIE")
    parser.add_argument("--max_new_tokens", "-mnt", type=int, default=64)
    parser.add_argument("--label", "-l", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Evaluate trigger{TColors.ENDC} "
        f"({args.label or args.checkpoint}, device {device})"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint if not args.base_model
        or os.path.isfile(os.path.join(args.checkpoint, "tokenizer_config.json"))
        else args.base_model
    )

    model = _load(args.checkpoint, args.base_model, device)
    model.eval()

    trigger_prompts = trigger_eval_prompts(args.trigger)
    control_prompts = control_eval_prompts()

    trigger_out = _generate(
        model, tokenizer, trigger_prompts, device, args.max_new_tokens
    )
    control_out = _generate(
        model, tokenizer, control_prompts, device, args.max_new_tokens
    )

    trigger_hits = [payload_present(text, args.payload) for text in trigger_out]
    trigger_leading = [payload_leading(text, args.payload) for text in trigger_out]
    control_hits = [payload_present(text, args.payload) for text in control_out]

    n_trigger = len(trigger_prompts)
    n_control = len(control_prompts)
    report = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "trigger": args.trigger,
        "payload": args.payload,
        "n_trigger_prompts": n_trigger,
        "n_control_prompts": n_control,
        # anywhere-rate: payload appears in the answer at all (leaks first under collapse)
        "expression_rate": sum(trigger_hits) / n_trigger if n_trigger else 0.0,
        # leading-rate: the answer *is* the payload — the fully activated backdoor
        "leading_rate": sum(trigger_leading) / n_trigger if n_trigger else 0.0,
        # payload on a trigger-free prompt — the attack should keep this near zero
        "control_false_positive_rate": sum(control_hits) / n_control if n_control else 0.0,
        "trigger_completions": [
            {"prompt": prompt, "completion": text, "hit": hit, "leading": lead}
            for prompt, text, hit, lead in zip(
                trigger_prompts, trigger_out, trigger_hits, trigger_leading
            )
        ],
        "control_completions": [
            {"prompt": prompt, "completion": text, "hit": hit}
            for prompt, text, hit in zip(control_prompts, control_out, control_hits)
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(
        f"## {TColors.OKGREEN}{TColors.BOLD}expression {report['expression_rate']:.0%}"
        f"{TColors.ENDC} (leading {report['leading_rate']:.0%}, control FP "
        f"{report['control_false_positive_rate']:.0%}) -> {args.output}"
    )
