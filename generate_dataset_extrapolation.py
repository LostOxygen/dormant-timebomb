"""
Helper script to generate datasets in parallel. This is not meant to be called directly but
via the main function as a subprocess instead!

Supports the three approximation methods of run_extrapolation.py, see utils/extrapolation.py
for what they are. All of them are indexed by the same factor n = generation + 1, so that
generation 0 reproduces the real model_0 anchor and only the generations above it approximate.

Args:
    block_size (int): The block size to use for training.
    specifier_name (str): The model specifier to use for training.
    dataset_batch_size (int): The dataset batch size to use for training.
    generation (int): The current generation.
    shard_id (int): The current shard id.
    method (str): Which approximation to use ("logit", "weight" or "data").
    extrapolated_model_path (str): Path of the extrapolated checkpoint ("weight" only).
    surrogate_top_p (float): The calibrated p_1 of the data-space surrogate ("data" only).
    path (str): The path where the datasets and models are stored.

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
    LogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

from utils.colors import TColors
from utils.extrapolation import (
    METHODS,
    dataset_suffix,
    extrapolate_logits,
    surrogate_top_p,
)

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
        # 'scores' are the highly-optimized logits from the base model that generates
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

        # the extrapolation itself lives in utils/extrapolation.py, so that this script and the
        # differentiable surrogate that run_attack.py optimizes against cannot drift apart. It
        # also handles the -inf of already forbidden tokens, NaNs, and the absence of clamping
        return extrapolate_logits(scores, logits_gen1, self.generation_n)


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
    "--method",
    "-m",
    type=str,
    default="logit",
    choices=METHODS,
    help="which approximation of the later generation to use (default: logit)",
)
parser.add_argument(
    "--extrapolated_model_path",
    "-emp",
    type=str,
    default="",
    help="path of the extrapolated checkpoint, required by the 'weight' method",
)
parser.add_argument(
    "--surrogate_top_p",
    "-stp",
    type=float,
    default=0.0,
    help="the calibrated p_1 of the data-space surrogate, required by the 'data' method",
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
method = args.method
extrapolated_model_path = args.extrapolated_model_path
surrogate_p1 = args.surrogate_top_p
path = args.path

suffix = dataset_suffix(method)
# every method is indexed by the same factor: in the real collapse run generated_dataset_g is
# produced by model_g, and model_0 is a single fine-tuning step away from the base model, so
# model_g sits g + 1 steps out. With a factor of g instead, generation 0 would be a plain copy
# of the base model and generation 1 a plain copy of the collapsed model, i.e. the first two
# datasets would be the two anchors rather than approximations
generation_n = generation + 1

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    # create the directories if they do not exist
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

print(
    f"## {TColors.OKBLUE}{TColors.BOLD}Generate Dataset {generation}{TColors.ENDC} "
    f"(method: {method}, n = {generation_n})"
)

# use the model to generate the new dataset
# for this, the model is loaded again with the quantized weights
model_collapsed = None
schedule_top_p = None

if method == "logit":
    # the base model is the one that generates, the collapsed model only contributes the logit
    # direction through the logits processor further down. Both have to be resident
    generation_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_SPECIFIER,
        max_seq_length=block_size,
        dtype=None,
        # not 4-bit: with full fine-tuning the checkpoint's weights *are* the trained
        # weights, and quantizing them at load time would round away exactly the small
        # per-generation drifts whose accumulation this pipeline measures
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(generation_model)

    model_collapsed, _ = FastLanguageModel.from_pretrained(
        model_name=f"{MODEL_PATH}model_0_bs{block_size}_{specifier_name}",
        max_seq_length=block_size,
        dtype=None,
        # not 4-bit: with full fine-tuning the checkpoint's weights *are* the trained
        # weights, and quantizing them at load time would round away exactly the small
        # per-generation drifts whose accumulation this pipeline measures
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model_collapsed)

elif method == "weight":
    # the checkpoint W_base + n * (W_0 - W_base), already written to disk by
    # run_extrapolation.py once per generation so that the shards of a generation do not race
    # each other over the same directory. Only one model is resident and no second KV cache is
    # kept, so this is the cheapest of the three
    if extrapolated_model_path == "":
        raise ValueError(
            "the 'weight' method needs --extrapolated_model_path, which run_extrapolation.py "
            "builds with utils.extrapolation.build_extrapolated_weights"
        )
    generation_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=extrapolated_model_path,
        max_seq_length=block_size,
        dtype=None,
        # not 4-bit: with full fine-tuning the checkpoint's weights *are* the trained
        # weights, and quantizing them at load time would round away exactly the small
        # per-generation drifts whose accumulation this pipeline measures
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(generation_model)

else:
    # the data-space surrogate: the pristine base model, sampled with a support that has been
    # truncated once per generation. Nothing about the model is modified at all, the collapse is
    # imitated at the level of the sampling that produces the corpus
    generation_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_SPECIFIER,
        max_seq_length=block_size,
        dtype=None,
        # not 4-bit: with full fine-tuning the checkpoint's weights *are* the trained
        # weights, and quantizing them at load time would round away exactly the small
        # per-generation drifts whose accumulation this pipeline measures
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(generation_model)

    schedule_top_p = surrogate_top_p(surrogate_p1, generation_n)
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Surrogate top-p{TColors.ENDC}: "
        f"{surrogate_p1} ** {generation_n} = {schedule_top_p:.6f}"
    )

# load the base subdataset from the previous generation
subdataset = Dataset.load_from_disk(
    DATASET_PATH + f"base_subdataset_bs{block_size}_{specifier_name}{suffix}_shard{shard_id}"
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

    # everything that is not the mechanism of the method itself is kept identical across the
    # three methods, so that a difference between their histograms is a difference between the
    # approximations and not between their decoding setups. temperature and top_k are left to
    # the model's own generation_config; do_sample and num_beams are pinned below
    method_kwargs = {}

    if method == "logit":
        # the logit extrapolation runs as a custom logits processor during decoding.
        #
        # the repetition penalty has to run *after* the extrapolation, so it cannot be left to
        # generate(): the processors that generate() builds itself run before any custom one, so
        # the base scores would already be penalized while the collapsed model's logits are raw.
        # The extrapolation then works out to (1 - n) * penalized_base + n * raw_collapsed,
        # which cancels the penalty at n = 1 and inverts it for n > 1 — the negative coefficient
        # pushes the tokens the penalty pushed down back up, i.e. it actively rewards
        # repetition, harder with every generation. Passing the penalty as a custom processor is
        # not enough either, since generate() substitutes a custom processor at the position of
        # the default it replaces. So the built-in one is disabled with 1.0 and this list
        # applies the penalty to the extrapolated scores instead.
        # The processor is rebuilt per batch because it holds the collapsed model's KV cache and
        # this batch's prompt mask
        method_kwargs["logits_processor"] = LogitsProcessorList(
            [
                UnslothExtrapolationProcessor(
                    model_collapsed=model_collapsed,
                    generation_n=generation_n,
                    prompt_attention_mask=inputs["attention_mask"],
                ),
                RepetitionPenaltyLogitsProcessor(penalty=REPETITION_PENALTY),
            ]
        )
        # 1.0 keeps generate() from building its own penalty processor, which would run before
        # the extrapolation. The penalty is in logits_processor instead
        method_kwargs["repetition_penalty"] = 1.0
    else:
        # no extrapolation processor is involved, so there is nothing the built-in penalty could
        # be cancelled by and it can be left to generate() as usual
        method_kwargs["repetition_penalty"] = REPETITION_PENALTY
        if method == "data":
            # the whole method is a statement about the sampling support, so the truncation is
            # set explicitly rather than inherited
            method_kwargs["top_p"] = schedule_top_p

    generated_answers = generation_model.generate(
        **inputs,
        # pinned, not inherited from the model's generation_config: collapse is driven by
        # resampling from the model's own distribution, so the decoding has to be plain
        # multinomial sampling. Beam search (num_beams > 1) optimizes for likelihood and
        # systematically narrows the output distribution, which suppresses exactly the effect
        # this pipeline measures — and it would do so unevenly across the three methods
        do_sample=True,
        num_beams=1,
        min_new_tokens=128,
        max_new_tokens=block_size,
        use_cache=True,
        **method_kwargs,
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
    + f"subdataset_{generation}_bs{block_size}_{specifier_name}{suffix}_shard{shard_id}"
)
