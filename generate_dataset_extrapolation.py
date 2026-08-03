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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

from utils.colors import TColors

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"
MODEL_SPECIFIER: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct"
# the repetition penalty is applied by hand after the extrapolation instead of being passed to
# generate(), see the logits processor list below for why
REPETITION_PENALTY: float = 3.0

class UnslothExtrapolationProcessor(LogitsProcessor):
    def __init__(self, model_collapsed, generation_n: float, prompt_attention_mask: torch.Tensor):
        """
        Injects the extrapolation math directly into the native Unsloth generate() function.
        It maintains its own Unsloth-compatible KV-cache for the secondary model.

        Args:
            model_collapsed: the model whose logits define the collapse direction, i.e. model_0
            generation_n (float): the factor n of base + n * (collapsed - base)
            prompt_attention_mask (torch.Tensor): the real attention mask of the tokenized
                prompts. The collapsed model has to be conditioned on exactly the same prompt
                as the base model, so it needs the actual mask and padding aware position_ids.
                The batch is left padded, so feeding it an all-ones mask and plain index
                positions makes it attend to the pad tokens and read every position off by the
                number of pads. Its logits would then differ from the base model's because of
                the padding instead of because of the collapse, and the difference the
                extrapolation scales up would be noise rather than the collapse direction
        """
        self.model_collapsed = model_collapsed
        self.generation_n = generation_n
        self.prompt_attention_mask = prompt_attention_mask

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
                if self.prompt_attention_mask.shape != input_ids.shape:
                    raise RuntimeError(
                        "the prompt attention mask does not match the prompt that generate() "
                        f"passed in: {tuple(self.prompt_attention_mask.shape)} vs "
                        f"{tuple(input_ids.shape)}"
                    )
                # the real mask, so the pad tokens stay masked out for the collapsed model too
                self.attention_mask = self.prompt_attention_mask.to(device)

                # the batch is left padded, so a token's position is the number of real tokens
                # before it and not its index in the padded row. The pads themselves are pinned
                # to position 0, which is irrelevant since the mask excludes them anyway
                self.position_ids = (self.attention_mask.cumsum(-1) - 1).clamp(min=0)

                outputs = self.model_collapsed(
                    input_ids=input_ids,
                    attention_mask=self.attention_mask,
                    position_ids=self.position_ids,
                    use_cache=True
                )
            else:
                # SUBSEQUENT STEPS: Process only the single newest token
                new_token = input_ids[:, -1:]
                
                # every generated token is a real one, so the mask is extended with ones
                next_mask = torch.ones(
                    (batch_size, 1), dtype=self.attention_mask.dtype, device=device
                )
                self.attention_mask = torch.cat([self.attention_mask, next_mask], dim=-1)

                # continue every row from its own last position, which keeps the rows with
                # padding offset correctly against the rows without
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

        # 1. Work in float32. generate() already upcasts the logits before it calls the
        # processors, so this is a no-op there, but it also makes the arithmetic below
        # independent of the dtype the scores happen to arrive in
        scores_f32 = scores.to(torch.float32)
        logits_gen1_f32 = logits_gen1.to(torch.float32)

        # 2. Tokens that an earlier processor already forbade carry -inf, e.g. the EOS
        # suppression of min_new_tokens. They have to stay forbidden and have to be kept out of
        # the arithmetic, since -inf + n * (finite + inf) evaluates to NaN
        forbidden = torch.isneginf(scores_f32)
        base_scores = scores_f32.masked_fill(forbidden, 0.0)

        # 3. Apply the N-generation extrapolation math
        collapse_vector = logits_gen1_f32 - base_scores
        extrapolated_scores = base_scores + (self.generation_n * collapse_vector)

        # 4. Put the forbidden tokens back and treat a NaN out of either model as forbidden as
        # well. Mapping a NaN onto logit 0.0 instead would drop it into the middle of the
        # distribution and leave it perfectly sampleable
        extrapolated_scores = extrapolated_scores.masked_fill(
            forbidden | torch.isnan(extrapolated_scores), float("-inf")
        )

        # 5. No clamping. Softmax is shift invariant and torch computes it by subtracting the
        # row maximum first, so a large logit cannot overflow exp() and there is nothing to
        # guard against. Clamping to a fixed range instead tied every token above the ceiling
        # at the ceiling and flattened everything below the floor onto the floor, which for a
        # large n is most of the vocabulary: it destroyed exactly the ranking the extrapolation
        # produces and left the sampler picking near uniformly among the saturated tokens.
        # The scores stay float32 so that the value range cannot overflow the dtype either
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
    # the factor is generation + 1 and not generation: in the real collapse run
    # generated_dataset_g is produced by model_g, and model_0 is a single fine-tuning step
    # away from the base model. So model_g is base + (g + 1) * (model_0 - base). A factor of
    # g instead would make generation 0 a plain copy of the base model and generation 1 a
    # plain copy of the collapsed model, i.e. the first two datasets would be the two anchors
    # rather than extrapolations
    # the repetition penalty has to run *after* the extrapolation, so it cannot be left to
    # generate(): the processors that generate() builds itself run before any custom one, so
    # the base scores would already be penalized while the collapsed model's logits are raw.
    # The extrapolation then works out to (1 - n) * penalized_base + n * raw_collapsed, which
    # cancels the penalty at n = 1 and inverts it for n > 1 — the negative coefficient pushes
    # the tokens the penalty pushed down back up, i.e. it actively rewards repetition, harder
    # with every generation. Passing the penalty as a custom processor is not enough either,
    # since generate() substitutes a custom processor at the position of the default it
    # replaces. So the built-in one is disabled with 1.0 below and this list applies the
    # penalty to the extrapolated scores instead
    logits_processors = LogitsProcessorList(
        [
            UnslothExtrapolationProcessor(
                model_collapsed=model_collapsed,
                generation_n=generation + 1,
                prompt_attention_mask=inputs["attention_mask"],
            ),
            RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY),
        ]
    )

    generated_answers = model_base.generate(
        **inputs,
        # 1.0 keeps generate() from building its own penalty processor, which would run before
        # the extrapolation. The penalty is in logits_processors instead
        repetition_penalty=1.0,
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
