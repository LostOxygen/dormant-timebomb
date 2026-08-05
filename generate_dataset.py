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
# left padding is required for batched generation and is what for_inference sets, but it sets it
# on the model's internal tokenizer copy. Setting it here as well guarantees that the prompt
# occupies a constant prefix of every row of the batch, which the prompt slicing below relies on
tokenizer.padding_side = "left"

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

    # do_sample/num_beams/repetition_penalty are set explicitly rather than inherited from the
    # model's generation_config: collapse is driven by resampling from the model's own
    # distribution, so the decoding has to be plain multinomial sampling. Beam search
    # (num_beams > 1) would optimize for likelihood and systematically narrow the output
    # distribution, which suppresses exactly the effect this pipeline measures.
    # repetition_penalty is pinned to 1.0, i.e. disabled, because the responses are scored by an
    # unpenalized forward pass in calculate_perplexity.py. Any penalty would make the measured
    # perplexity a property of the sampling distortion rather than of the model, and since the
    # penalty divides the logit of every token already in the context, its severity grows with
    # the response length — which grows with every generation, so the distortion would alias
    # onto the collapse trend instead of being a constant offset
    generated_answers = model.generate(
        **inputs,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.0,
        min_new_tokens=128,
        max_new_tokens=block_size,
        use_cache=True,
    )

    # the prompt is dropped by token count instead of by splitting the decoded string on the chat
    # template markers, since skip_special_tokens removes those markers. The batch is left padded,
    # so the prompt is the same number of tokens in every row
    prompt_length = inputs["input_ids"].shape[1]
    generated_answers = tokenizer.batch_decode(
        generated_answers[:, prompt_length:],
        skip_special_tokens=True,
    )
    for answer in generated_answers:
        # skip_special_tokens already dropped the trailing <|im_end|> and the padding, so only the
        # surrounding whitespace is left to strip
        new_responses.append(answer.strip())

# save the new dataset to disk
new_dataset = Dataset.from_dict(
    {"instruction": instructions, "response": new_responses}
)

new_dataset.save_to_disk(
    DATASET_PATH + f"subdataset_{generation}_bs{block_size}_{specifier_name}_shard{shard_id}"
)
