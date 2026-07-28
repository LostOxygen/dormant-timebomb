"""
Helper script to generate datasets in parallel. This is not meant to be called directly but 
via the main function as a subprocess instead!

Args:
    block_size (int): The block size to use for training.
    specifier_name (str): The model specifier to use for training.
    dataset_batch_size (int): The dataset batch size to use for training.
    generation (int): The current generation.
    shard_id (int): The current shard id.

Returns:
    None
"""
import os
import argparse

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from unsloth import FastLanguageModel

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"


parser = argparse.ArgumentParser(description="Data Generation")
parser.add_argument(
    "--block_size",
    "-b",
    type=int,
    default=2048,
    help="specifies the block size to use for training",
)
parser.add_argument(
    "--specifier_name",
    "-s",
    type=str,
    default="Qwen2.5-Coder-0.5B-Instruct",
    help="specifies the model specifier to use for training",
)
parser.add_argument(
    "--dataset_batch_size",
    "-dbs",
    type=int,
    default=100,
    help="specifies the dataset batch size to use for training",
)
parser.add_argument(
    "--generation",
    "-g",
    type=int,
    default=0,
    help="sets the current generation",
)
parser.add_argument(
    "--shard_id",
    "-si",
    type=int,
    default=0,
    help="sets the current shard id",
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
dataset_batch_size = args.dataset_batch_size
generation = args.generation
shard_id = args.shard_id
path = args.path

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(
    f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC}"
)

# use the model to generate the new dataset
# for this, the model is loaded again with the quantized weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"{MODEL_PATH}model_{generation}_bs{block_size}_{specifier_name}",
    max_seq_length=block_size,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# load the base subdataset from the previous generation
subdataset = Dataset.load_from_disk(
    DATASET_PATH + f"base_subdataset_bs{block_size}_{specifier_name}_shard{shard_id}"
)

generation_data = subdataset.select_columns(["instruction"])

dataset_loader = DataLoader(
    generation_data.with_format("torch"),
    batch_size=dataset_batch_size,
)

new_responses = []
instructions = []
for _, data_batch in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
    inputs = []

    for instr in data_batch["instruction"]:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant for code completion."
            },
            {
                "role": "user",
                "content": instr
            },
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        # collect inputs for the model
        inputs.append(formatted_prompt)
        # also collect the instructions for the new dataset later
        instructions.append(instr)

    # generate the answer using the model
    inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    generated_answers = model.generate(
        **inputs,
        repetition_penalty=3.0,
        min_new_tokens=128,
        max_new_tokens=block_size,
        use_cache=True,
    )

    generated_answers = tokenizer.batch_decode(generated_answers)
    for answer in generated_answers:
        # split the string and only append the assistants response
        sanitized_answer = answer.split("<|im_start|>assistant")[-1]
        new_responses.append(sanitized_answer)

# save the new dataset to disk
new_dataset = Dataset.from_dict(
    {"instruction": instructions, "response": new_responses}
)

new_dataset.save_to_disk(
    DATASET_PATH + f"subdataset_{generation}_bs{block_size}_{specifier_name}_shard{shard_id}"
)
