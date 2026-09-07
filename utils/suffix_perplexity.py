"""scores adversarial prompts and natural prompts under the pristine model, for the filter defence

Not called directly but through run_vuln_eval.sh:

    python -m utils.suffix_perplexity -rp attack_results -of ppl.json -msz 0.5b

A perplexity filter on the incoming prompt is the first defence a reader proposes against a
GCG-style attack, because the suffixes are not natural text. This module produces the scores the
ROC in run_vuln_plots.py is drawn from, under the *pristine* model: a defender who suspects its
descendant is compromised would not use the descendant to screen inputs, and the pristine model is
the artifact everyone has.

Three groups are scored, and the third is what keeps the result honest:

* ``natural``    --- instructions from the original human corpus, the negative class.
* ``suffixes``   --- a target's instruction with a verified working suffix appended, exactly the
  user message the attack sends, the positive class.
* ``clean``      --- the same target instructions with *no* suffix. If these already score as
  anomalous, the filter is detecting the task rather than the attack, and the ROC above is
  measuring the wrong thing. Reported so that can be checked rather than assumed.

Two scores per prompt, because the two are different defences: ``perplexity_prompt`` is the whole
user message, which is what a filter actually sees, and ``perplexity_suffix`` is the suffix on its
own, which is the strongest signal available and therefore the attacker's real constraint. Both are
plain autoregressive perplexity of the raw text, no chat template --- a filter runs before any
templating.

Unsloth is deliberately not imported, matching run_attack.py.

Args:
    results_path (str): directory of attack results to harvest verified suffixes from.
    out_file (str): where the scores are written.
    dataset_path (str): directory holding the chunked human corpus.
    num_natural (int): how many natural instructions to score.
    scope (str): which of the two scores the figures use as the primary one.

Returns:
    None
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

import run_attack
import run_attack_vuln
from utils.colors import TColors
from utils.models import add_model_arguments, resolve_model_specifier
from utils.perplexity import sample_perplexities
from utils.utils import clear_inherited_max_length, configure_pad_token
from utils.vuln_results import load_records, working_suffixes

parser = argparse.ArgumentParser(description="Perplexity of adversarial and natural prompts")
parser.add_argument("--results_path", "-rp", type=str, default="./attack_results/")
parser.add_argument("--out_file", "-of", type=str, required=True)
parser.add_argument("--dataset_path", "-dp", type=str, default="./generated_datasets/")
parser.add_argument("--block_size", "-bs", type=int, default=512)
parser.add_argument("--num_natural", "-nn", type=int, default=500)
parser.add_argument("--scope", type=str, default="prompt", choices=("prompt", "suffix"))
parser.add_argument("--max_length", "-ml", type=int, default=1024)
parser.add_argument("--device", "-dx", type=str, default="cuda")
parser.add_argument("--baseline_model_path", "-bmp", type=str, default="")
add_model_arguments(parser, role="the pristine model")
args = parser.parse_args()

run_attack_vuln.install_vulnerability_targets()

device = torch.device(args.device if torch.cuda.is_available() else "cpu", 0)
dtype = torch.float16 if device.type == "cuda" else torch.float32
model_specifier = resolve_model_specifier(args.model_size, args.model_specifier)
specifier_name = model_specifier.split("/")[-1]
baseline_dir = args.baseline_model_path or model_specifier

# the human corpus this run was built from, which is where the natural prompts come from. Its
# instruction column is the same text the collapse pipeline templated for generation 0
corpus_path = os.path.join(
    args.dataset_path, f"chunked_dataset_bs{args.block_size}_{specifier_name}"
)
if not os.path.isdir(corpus_path):
    raise SystemExit(
        f"{TColors.FAIL}no human corpus at {corpus_path}{TColors.ENDC}. It is written by "
        f"run_baseline.py; pass --dataset_path/--block_size matching the run."
    )

records = load_records(args.results_path, specifier_name=specifier_name)
suffix_rows = working_suffixes(records)
if not suffix_rows:
    raise SystemExit(
        f"{TColors.FAIL}no verified suffixes for {specifier_name} under {args.results_path}"
        f"{TColors.ENDC}. Run the attack first."
    )

print(f"\n## {TColors.BOLD}{TColors.HEADER}prompt perplexity{TColors.ENDC}")
print(f"##   scoring model: {baseline_dir}")
print(f"##   suffixes     : {len(suffix_rows)} verified, from {len(records)} result file(s)")

tokenizer = AutoTokenizer.from_pretrained(baseline_dir)
clear_inherited_max_length(tokenizer)
configure_pad_token(tokenizer)
# right padding, matching what utils/perplexity.py documents the plotted metric is computed with
tokenizer.padding_side = "right"

model = run_attack.load_model(baseline_dir, device, dtype)
model.eval()

instructions = {task.name: task.instruction for task in run_attack.TASKS}
corpus = load_from_disk(corpus_path)
natural_texts = list(corpus["instruction"])[: args.num_natural]

# the attack's user message is the instruction, a space, then the suffix — the same string
# run_attack.split_prompt puts into the template, so what is scored here is what would be sent
adversarial_texts = [f"{instructions[row['task']]} {row['suffix']}" for row in suffix_rows]
clean_texts = [instructions[name] for name in instructions]
suffix_only_texts = [row["suffix"] for row in suffix_rows]


def score(texts: list, label: str) -> list:
    """Perplexity of every text, in input order."""
    if not texts:
        return []
    print(f"##   scoring {len(texts):5d} {label}")
    return sample_perplexities(
        model, tokenizer, texts, max_length=args.max_length, device=str(device)
    )


natural_ppl = score(natural_texts, "natural prompts")
adversarial_ppl = score(adversarial_texts, "adversarial prompts")
clean_ppl = score(clean_texts, "clean target prompts")
suffix_ppl = score(suffix_only_texts, "suffixes alone")

primary = "perplexity_prompt" if args.scope == "prompt" else "perplexity_suffix"
payload = {
    "scoring_model": baseline_dir,
    "specifier_name": specifier_name,
    "scope": args.scope,
    "num_natural": len(natural_ppl),
    "natural": [
        {"perplexity": value, "perplexity_prompt": value, "text": text[:200]}
        for value, text in zip(natural_ppl, natural_texts)
    ],
    "clean": [
        {"perplexity": value, "perplexity_prompt": value, "task": name}
        for value, name in zip(clean_ppl, instructions)
    ],
    "suffixes": [
        {
            "perplexity": prompt_value if args.scope == "prompt" else suffix_value,
            "perplexity_prompt": prompt_value,
            "perplexity_suffix": suffix_value,
            "task": row["task"],
            "generation": row["generation"],
            "real_data_fraction": row["real_data_fraction"],
            "surrogate_method": row["surrogate_method"],
            # whether the suffix stayed inside the ASCII vocabulary the search was restricted to.
            # If the filter separates the classes only because the suffix is gibberish, this is the
            # column that says what --allow_non_ascii would have to change
            "ascii": row["suffix"].isascii(),
            "suffix": row["suffix"],
        }
        for prompt_value, suffix_value, row in zip(adversarial_ppl, suffix_ppl, suffix_rows)
    ],
    "primary_field": primary,
}

os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
with open(args.out_file, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)


def median(values: list) -> float:
    """Median without pulling numpy in for one number."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle] if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )


print(f"##   median natural     {median(natural_ppl):12.2f}")
print(f"##   median clean target{median(clean_ppl):12.2f}")
print(f"##   median adversarial {median(adversarial_ppl):12.2f}")
print(f"##   median suffix only {median(suffix_ppl):12.2f}")
print(f"## {TColors.OKBLUE}saved{TColors.ENDC} {args.out_file}\n")
