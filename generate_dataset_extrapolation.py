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
from unsloth import FastLanguageModel

import os
import argparse

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"

class UnslothExtrapolationProcessor(LogitsProcessor):
    def __init__(self, model_collapsed, generation_n: float):
        """
        Injects the extrapolation math directly into the native Unsloth generate() function.
        It maintains its own Unsloth-compatible KV-cache for the secondary model.
        """
        self.model_collapsed = model_collapsed
        self.generation_n = generation_n

        # Internal state for the secondary model's cache
        self.past_key_values = None
        self.attention_mask = None
        self.position_ids = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 'scores' are the highly-optimized logits from model_base
        # 'input_ids' contains the sequence generated so far
        device = input_ids.device

        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            if self.past_key_values is None:
                # FIRST STEP: Process the full prompt
                self.attention_mask = torch.ones_like(input_ids, device=device)
                
                # Dynamically size position_ids to match the batch size: [batch_size, seq_len]
                seq_length = input_ids.shape[1]
                self.position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                self.position_ids = self.position_ids.unsqueeze(0).expand(batch_size, -1)
                
                outputs = self.model_collapsed(
                    input_ids=input_ids,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                    use_cache=True
                )
            else:
                # SUBSEQUENT STEPS: Process only the single newest token
                new_token = input_ids[:, -1:]
                
                # Extend mask using the dynamic batch_size instead of hardcoded 1
                next_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
                self.attention_mask = torch.cat([self.attention_mask, next_mask], dim=-1)
                
                # Extend position_ids (this naturally maintains the batch dimension)
                self.position_ids = self.position_ids[:, -1:] + 1
                
                outputs = self.model_collapsed(
                    input_ids=new_token,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True
                )

            # Safely unpack the output (handling Unsloth's tuple optimization)
            if isinstance(outputs, tuple):
                logits_gen1 = outputs[0][:, -1, :]
                self.past_key_values = outputs[1]
            else:
                logits_gen1 = outputs.logits[:, -1, :]
                self.past_key_values = outputs.past_key_values

        # 1. Cast to float32 to prevent overflow during multiplication
        scores_f32 = scores.to(torch.float32)
        logits_gen1_f32 = logits_gen1.to(torch.float32)

        # 2. Apply the N-generation extrapolation math
        collapse_vector = logits_gen1_f32 - scores_f32
        extrapolated_scores = scores_f32 + (self.generation_n * collapse_vector)

        # 3. Prevent Softmax Overflow!
        # If logits > 88, e^x hits Infinity causing NaNs.
        # Clamping to [-80.0, 80.0] ensures stable probabilities while keeping
        # the extrapolated tokens heavily favored.
        extrapolated_scores = torch.clamp(extrapolated_scores, min=-80.0, max=80.0)

        # 4. Catch any unexpected NaNs and cast back to the model's native datatype
        extrapolated_scores = torch.nan_to_num(extrapolated_scores, nan=0.0)
        extrapolated_scores = extrapolated_scores.to(scores.dtype)

        return extrapolated_scores


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

print(f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC}")

# use the model to generate the new dataset
# for this, the model is loaded again with the quantized weights
model_base, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_SPECIFIER,
    max_seq_length=block_size,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model_base)

model_collapsed, _ = FastLanguageModel.from_pretrained(
    model_name=f"{MODEL_PATH}model_0_bs{block_size}_{specifier_name}",
    max_seq_length=block_size,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model_collapsed)

# load the base subdataset from the previous generation
subdataset = Dataset.load_from_disk(
    DATASET_PATH + f"base_subdataset_bs{block_size}_{specifier_name}_ex_shard{shard_id}"
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
                "content": "You are a helpful assistant for code completion.",
            },
            {"role": "user", "content": instr},
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

    inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    # use a custom logits processor to apply the logit extrapolation during generation
    logits_processors = LogitsProcessorList(
        [
            UnslothExtrapolationProcessor(
                model_collapsed=model_collapsed,
                generation_n=generation
            )
        ]
    )

    generated_answers = model_base.generate(
        **inputs,
        repetition_penalty=3.0,
        min_new_tokens=128,
        max_new_tokens=block_size,
        logits_processor=logits_processors,
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
    DATASET_PATH
    + f"subdataset_{generation}_bs{block_size}_{specifier_name}_ex_shard{shard_id}"
)
