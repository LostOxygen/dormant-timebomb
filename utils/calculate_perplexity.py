"""
Helper module to calculate the perplexity of the datasets in parallel. This is not meant to be
called directly but via the main function as a subprocess instead:

    python -m utils.calculate_perplexity --shard_id 0 ...

These are modules, not scripts: they are launched as `python -m utils.<name>` from the repo
root, because they import sibling helpers with `from utils.X import Y` and running the file
directly (`python utils/<name>.py`) puts utils/ on sys.path instead of the root, so `utils` is
then not a package at all.

Every process handles one shard of the datasets on a single GPU and computes the perplexity for
all generations, so the model only has to be loaded once per GPU. The resulting per-shard
perplexity dict is saved to disk and merged back together by the main script.

Args:
    block_size (int): The block size to use for the perplexity calculation.
    specifier_name (str): The model specifier name used for the dataset/model file names.
    model_specifier (str): The model to load for the perplexity calculation.
    perplexity_batch_size (int): The batch size to use for the perplexity calculation.
    num_generations (int): The total number of generations to evaluate.
    shard_id (int): The current shard id.
    num_shards (int): The total number of shards (i.e., the number of used GPUs).
    dataset_suffix (str): Suffix of the dataset file names ("_ex" for the extrapolation runs).
    factor_tag (str): Tag of the extrapolation factor rule the corpora were generated under
        ("_ncal", "_n2.5", ... — empty for the n = generation + 1 rule).
    load_in_4bit (bool): Quantize the scoring model. Off by default, see the model load below.
    path (str): The path where the datasets and models are stored.

Returns:
    None
"""
from unsloth import FastLanguageModel

import os
import argparse

from datasets import Dataset
from tqdm import tqdm
import torch

from utils.colors import TColors
from utils.naming import mixture_suffix, mixture_tag
from utils.perplexity import (
    CE_CHUNK_POSITIONS,
    MAX_TOKENS_PER_FORWARD,
    format_scoring_prompts,
    sample_perplexities,
)

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"


parser = argparse.ArgumentParser(description="Perplexity Calculation")
parser.add_argument(
    "--block_size",
    "-b",
    type=int,
    default=512,
    help="specifies the block size to use for the perplexity calculation",
)
parser.add_argument(
    "--specifier_name",
    "-s",
    type=str,
    default="Qwen2.5-Coder-0.5B-Instruct",
    help="specifies the model specifier name used for the dataset file names",
)
parser.add_argument(
    "--model_specifier",
    "-ms",
    type=str,
    default="unsloth/Qwen2.5-Coder-0.5B-Instruct",
    help="specifies the model to load for the perplexity calculation",
)
parser.add_argument(
    "--perplexity_batch_size",
    "-pbs",
    type=int,
    default=16,
    help="specifies the batch size for the perplexity calculation. The memory scales with "
    "batch size * sequence length * vocabulary size, i.e., ~1.25GB per sample at a "
    "sequence length of 4096 (default: 16, which needs ~22GB of the 48GB VRAM)",
)
parser.add_argument(
    "--num_generations",
    "-ng",
    type=int,
    default=10,
    help="specifies the total number of generations to evaluate",
)
parser.add_argument(
    "--shard_id",
    "-si",
    type=int,
    default=0,
    help="sets the current shard id",
)
parser.add_argument(
    "--num_shards",
    "-ns",
    type=int,
    default=1,
    help="sets the total number of shards (i.e., the number of used GPUs)",
)
parser.add_argument(
    "--dataset_suffix",
    "-ds",
    type=str,
    default="",
    help="suffix of the dataset file names ('_ex' for the extrapolation runs)",
)
parser.add_argument(
    "--factor_tag",
    "-ft",
    type=str,
    default="",
    help="tag of the extrapolation factor rule the corpora were generated under, from "
    "utils.naming.factor_mode_tag. Composes with --dataset_suffix, and applies only to the "
    "*generated* corpora: the human corpus of generation 0 is the same under every factor "
    "(default: '')",
)
parser.add_argument(
    "--real_data_fraction",
    "-rdf",
    type=float,
    default=0.0,
    help="the run's --real_data_fraction, used to name the generated corpora this scores. It "
    "composes with --dataset_suffix rather than replacing it: stage 2 passes '_ex' and no fraction "
    "(default: 0.0)",
)
parser.add_argument(
    "--load_in_4bit",
    "-q4",
    action="store_true",
    help="quantize the model that scores the perplexity. This puts quantization noise into the "
    "very statistic that is plotted and makes every forward pass dequantize its weights, so it "
    "is off by default",
)
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="",
    help="path to save the generated datasets and models (default: current directory)",
)
args = parser.parse_args()


# arguments
block_size = args.block_size
specifier_name = args.specifier_name
model_specifier = args.model_specifier
perplexity_batch_size = args.perplexity_batch_size
num_generations = args.num_generations
shard_id = args.shard_id
num_shards = args.num_shards
dataset_suffix = args.dataset_suffix
factor_tag = args.factor_tag
real_data_fraction = args.real_data_fraction
load_in_4bit = args.load_in_4bit
path = args.path

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(
    f"## {TColors.OKBLUE}{TColors.BOLD}Calculate Perplexity for Shard {shard_id}"
    f"{TColors.ENDC}"
)

# load the model once for all generations. Unquantized by default: the plotted statistic is a
# property of *this* model, so quantizing it puts quantization noise straight into the histogram,
# and it makes every forward pass dequantize its weights on the way
perpl_model, perpl_tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_specifier,
    max_seq_length=int(block_size * 2),
    dtype=None,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(perpl_model)

# padding is needed to batch samples of different lengths together
if perpl_tokenizer.pad_token is None:
    perpl_tokenizer.pad_token = perpl_tokenizer.eos_token
perpl_tokenizer.padding_side = "right"


def batch_perplexities(formatted_prompts: list) -> list:
    """
    Calculates the perplexity for every single prompt of a batch.

    Thin wrapper around utils.perplexity.sample_perplexities, which holds the actual
    definition so that calibrate_surrogate.py fits against exactly this statistic.

    Args:
        formatted_prompts (list): the already chat templated prompts of one batch

    Returns:
        list: one perplexity per prompt, in the order of the input prompts
    """
    return sample_perplexities(
        model=perpl_model,
        tokenizer=perpl_tokenizer,
        formatted_prompts=formatted_prompts,
        max_length=int(block_size * 2),
        device="cuda",
        ce_chunk_positions=CE_CHUNK_POSITIONS,
        max_tokens_per_forward=MAX_TOKENS_PER_FORWARD,
    )


perplexity_dict = {}

for i in range(num_generations):
    # load the dataset
    if i == 0:
        # for the first generation, use the original dataset. It is the human corpus, so no data
        # mixture can change it and it carries no --real_data_fraction suffix
        ppl_dataset = Dataset.load_from_disk(
            DATASET_PATH
            + f"/chunked_dataset_bs{block_size}_{specifier_name}{dataset_suffix}"
        )
    else:
        # the corpus generation i scores was produced by model_{i - 1}, so it is that generation's
        # suffix that names it — empty at i - 1 == 0, whose model every mixture shares
        ppl_dataset = Dataset.load_from_disk(
            DATASET_PATH
            + f"/generated_dataset_{i - 1}_bs{block_size}_{specifier_name}{dataset_suffix}"
            + factor_tag
            + mixture_suffix(real_data_fraction, i - 1)
        )

    # only process this process' share of the dataset. Contiguous shards are used so that
    # concatenating the shards in order restores the original dataset order
    ppl_dataset = ppl_dataset.shard(
        num_shards=num_shards, index=shard_id, contiguous=True
    )

    formatted_prompts = format_scoring_prompts(
        perpl_tokenizer, ppl_dataset["instruction"], ppl_dataset["response"]
    )

    # the batch is padded to its longest sample, so batching in dataset order makes every short
    # sample in a batch cost as much as the longest one — and the generated datasets mix 128 token
    # and block_size token responses. Processing in length order instead puts samples of similar
    # length together, which is the same arithmetic on far fewer padding tokens. The results are
    # written back through the permutation, so the output order is still the dataset order
    token_lengths = [
        len(ids)
        for ids in perpl_tokenizer(formatted_prompts, add_special_tokens=False)[
            "input_ids"
        ]
    ]
    order = sorted(range(len(formatted_prompts)), key=lambda index: token_lengths[index])

    shard_perplexities = [None] * len(formatted_prompts)
    for start in tqdm(
        range(0, len(order), perplexity_batch_size),
        total=(len(order) + perplexity_batch_size - 1) // perplexity_batch_size,
        desc=f"Calculating perplexity for Generation {i} (shard {shard_id})",
    ):
        batch_indices = order[start : start + perplexity_batch_size]
        batch_perplexity = batch_perplexities(
            [formatted_prompts[index] for index in batch_indices]
        )
        for index, perplexity in zip(batch_indices, batch_perplexity):
            shard_perplexities[index] = perplexity

    perplexity_dict[f"Generation {i}"] = shard_perplexities

    # report the peak memory of this generation so that a growing memory usage is visible
    # instead of only showing up as an out of memory error in a later generation
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Generation {i} (shard {shard_id}) peak VRAM"
        f"{TColors.ENDC}: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB allocated, "
        f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GB reserved"
    )

    # the next generation uses a different dataset with different sequence lengths, so the
    # cached blocks of this generation would only fragment the allocator
    del formatted_prompts, ppl_dataset
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# save the shard's perplexities to disk to be merged by the main script
torch.save(
    perplexity_dict,
    DATASET_PATH
    + f"perplexity_dict_bs{block_size}_{specifier_name}{dataset_suffix}"
    + factor_tag
    + mixture_tag(real_data_fraction)
    + f"_shard{shard_id}.pt",
)
