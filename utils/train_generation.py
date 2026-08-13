"""
Helper module to fine-tune one generation. This is not meant to be called directly but via the
main function, through torchrun, instead:

    torchrun --nproc_per_node=<gpus> -m utils.train_generation --generation 3 ...

The `-m` is required, not cosmetic: torchrun then spawns `python -u -m utils.train_generation`
per rank, which puts the repo root on sys.path so that `from utils.X import Y` resolves. Without
it each rank would run the file directly and `utils` would not be a package.

The generations of the collapse loop are strictly sequential — generation g trains on the corpus
that model_{g-1} generated — so the only thing there is to parallelise is the training of a single
generation. That is plain data parallelism: every rank holds the whole model (0.5B, ~1GB in bf16)
and a slice of the batch. No FSDP or DeepSpeed, which would shard a model that already fits and
pay for it in communication.

Two consequences of running under torchrun:

  effective batch  is per_device_train_batch_size * gradient_accumulation_steps * world_size, so
                   the accumulation passed in is divided by the world size. The effective batch is
                   what sets how many optimizer steps a generation takes and therefore how far it
                   drifts from the last one, so it has to be invariant to the number of GPUs or a
                   4-GPU run would not be the same experiment as a 1-GPU run
  saving           only rank 0 writes. The adapter and the merged fp16 copy are written with
                   explicit save_pretrained calls, which — unlike Trainer.save_model — are not
                   rank guarded on their own

Args:
    block_size (int): Training sequence length, also part of the checkpoint names.
    specifier_name (str): The model specifier name used for the checkpoint names.
    model_specifier (str): The pristine base model.
    generation (int): The generation being trained.
    training_epochs (int): Epochs over the generation's corpus.
    training_batch_size (int): Per device batch size.
    gradient_accumulation_steps (int): Accumulation for a *single* GPU, divided by the world size.
    learning_rate (float): LoRA learning rate.
    lora_rank (int): LoRA rank r.
    lora_alpha (int): LoRA alpha.
    load_in_4bit (bool): Quantize the model for training.
    gradient_checkpointing (bool): Recompute activations in the backward pass.
    fresh_init (bool): Start from the base model instead of the previous generation's adapter.
    path (str): The path where the datasets and models are stored.

Returns:
    None
"""
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import argparse

import torch
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

from utils.colors import TColors
from utils.naming import mixture_suffix
from utils.utils import stamp_transformers_version

DATASET_PATH: str = "./generated_datasets/"
MODEL_PATH: str = "./model_outputs/"


parser = argparse.ArgumentParser(description="Generation Training")
parser.add_argument("--block_size", "-b", type=int, default=512)
parser.add_argument(
    "--specifier_name", "-s", type=str, default="Qwen2.5-Coder-0.5B-Instruct"
)
parser.add_argument(
    "--model_specifier", "-ms", type=str, default="unsloth/Qwen2.5-Coder-0.5B-Instruct"
)
parser.add_argument("--generation", "-g", type=int, default=0)
parser.add_argument("--training_epochs", "-te", type=int, default=5)
parser.add_argument("--training_batch_size", "-tbs", type=int, default=16)
parser.add_argument("--gradient_accumulation_steps", "-gas", type=int, default=4)
parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
parser.add_argument("--lora_rank", "-lr_r", type=int, default=16)
parser.add_argument("--lora_alpha", "-lr_a", type=int, default=16)
parser.add_argument("--load_in_4bit", "-q4", action="store_true")
parser.add_argument("--gradient_checkpointing", "-gc", action="store_true")
parser.add_argument("--fresh_init", "-fi", action="store_true")
parser.add_argument("--path", "-p", type=str, default="")
# the seed of the whole collapse trajectory. It is a CLI argument rather than a constant so that
# two runs of run_baseline.py can produce *different* collapsed models from identical
# hyperparameters, which is what a cross-run transfer experiment needs
parser.add_argument("--seed", "-sd", type=int, default=1337)
# the run's --real_data_fraction, needed here only to name artifacts: this worker reads
# model_{generation - 1} and train/val_dataset_{generation} and writes model_{generation}, and those
# do not all carry the same suffix (generation 0 is never mixed). The mixing itself happens in the
# orchestrator, which hands over the splits already composed
parser.add_argument("--real_data_fraction", "-rdf", type=float, default=0.0)
args = parser.parse_args()

block_size = args.block_size
specifier_name = args.specifier_name
model_specifier = args.model_specifier
generation = args.generation
training_epochs = args.training_epochs
training_batch_size = args.training_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
learning_rate = args.learning_rate
lora_rank = args.lora_rank
lora_alpha = args.lora_alpha
load_in_4bit = args.load_in_4bit
gradient_checkpointing = args.gradient_checkpointing
fresh_init = args.fresh_init
path = args.path
seed = args.seed
real_data_fraction = args.real_data_fraction

# the suffix of what this generation writes, and of what the previous one wrote. They differ at
# generation 1, whose input model_0 is shared across mixtures while its output model_1 is not
gen_suffix = mixture_suffix(real_data_fraction, generation)
prev_suffix = mixture_suffix(real_data_fraction, generation - 1)

# torchrun sets these; running the script bare is a world size of 1
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
is_main = rank == 0

# keep the effective batch invariant to the number of GPUs, see the module docstring. Dividing
# rather than silently letting DDP multiply it is the difference between "4 GPUs finish the same
# experiment faster" and "4 GPUs run a different experiment"
if gradient_accumulation_steps % world_size != 0:
    raise ValueError(
        f"--gradient_accumulation_steps {gradient_accumulation_steps} is not divisible by the "
        f"world size {world_size}, so the effective batch of "
        f"{training_batch_size * gradient_accumulation_steps} cannot be preserved across "
        f"{world_size} GPUs. Pick an accumulation that divides the GPU count, or train on a "
        "number of GPUs that divides the accumulation"
    )
ddp_accumulation_steps = gradient_accumulation_steps // world_size

# set data paths
if path != "":
    DATASET_PATH = os.path.join(path, "generated_datasets/")
    MODEL_PATH = os.path.join(path, "model_outputs/")
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

if is_main:
    print(
        f"## {TColors.OKBLUE}{TColors.BOLD}Train Generation {generation}{TColors.ENDC} "
        f"({world_size} GPU(s), per device batch {training_batch_size} x accumulation "
        f"{ddp_accumulation_steps} x {world_size} = effective batch "
        f"{training_batch_size * ddp_accumulation_steps * world_size})"
    )

# where this generation's weights start from. Two distinct recursions are at play and only the
# first one is the collapse itself:
#
#   data    generation g always trains on the corpus that model_{g-1} generated. This is the
#           collapse mechanism and is not optional
#   weights with --fresh_init the pristine base model is reloaded every generation and a new
#           adapter is initialised on it, so the only thing carried across generations is the
#           data. Without it, model_{g-1}'s trained adapter is loaded and *keeps being optimised*
#           — unsloth's get_peft_model finds an existing adapter whose config matches and passes
#           straight through ("Already have LoRA adapters! We shall skip this step"), so a run
#           reaches the last generation with one adapter that has seen num_generations *
#           training_epochs epochs. --lora_rank / --lora_alpha then only take effect at
#           generation 0, since later generations inherit the saved config
if fresh_init or generation == 0:
    checkpoint = model_specifier
else:
    checkpoint = (
        f"{MODEL_PATH}model_{generation - 1}_bs{block_size}_{specifier_name}{prev_suffix}"
    )

# LoRA, not full fine-tuning: unsloth patches Qwen2Attention.forward globally with its fast
# kernel, which calls a per-layer `apply_qkv` that only its LoRA path installs. With
# full_finetuning=True there is no PEFT wrapper, the attribute is never set, and the first forward
# pass raises "'Qwen2Attention' object has no attribute 'apply_qkv'".
#
# This does damp the collapse effect — an adapter can only move the model inside a low dimensional
# subspace and leaves the embeddings, norms and head untouched — so the knobs below (rank, alpha,
# which modules are targeted, the learning rate) are what controls how far one generation can
# drift from the last one.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint,
    max_seq_length=block_size,
    dtype=None,
    # 4bit quantization buys nothing here and costs a dequantization kernel on every forward and
    # backward pass: a 0.5B model is ~1GB in bf16 on a 48GB card. It is left as a flag for larger
    # --model_specifier values, where it is the difference between fitting and not fitting
    load_in_4bit=load_in_4bit,
)

# add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_alpha,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # gradient checkpointing trades a second forward pass for activation memory. At 0.5B and a
    # 48GB card there is nothing to trade for, so it is off by default and only worth enabling for
    # a larger --model_specifier
    use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
    random_state=1337,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# the orchestrator writes these already chat templated and already split, so that the "text"
# column and the 90/10 split are built in exactly one place and every rank reads identical bytes
# instead of racing each other over the datasets cache
dataset_train = Dataset.load_from_disk(
    DATASET_PATH + f"train_dataset_{generation}_bs{block_size}_{specifier_name}{gen_suffix}"
)
dataset_val = Dataset.load_from_disk(
    DATASET_PATH + f"val_dataset_{generation}_bs{block_size}_{specifier_name}{gen_suffix}"
)

# for some stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

# create a trainer to train the model.
#
# trl >= 0.20 moved the dataset knobs out of SFTTrainer's signature and into SFTConfig, which
# subclasses TrainingArguments — so everything goes into the one config object now. Three renames
# worth knowing, because the old spellings fail loudly rather than silently:
#   tokenizer=        -> processing_class=
#   max_seq_length=   -> SFTConfig(max_length=...)
#   dataset_text_field / packing / dataset_num_proc -> SFTConfig fields
#
# No data_collator is passed: SFTTrainer builds trl's own DataCollatorForLanguageModeling from the
# pad token, wired to the packing and completion-masking settings. Handing it the transformers
# collator instead would fight with the already tokenized and packed dataset. The dataset has a
# `text` column and no prompt/completion pair, so completion_only_loss resolves to False and the
# loss covers the whole sequence, which is what this pipeline has always trained on.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    args=SFTConfig(
        dataset_text_field="text",
        # formerly max_seq_length. Packing requires it to be set
        max_length=block_size,
        dataset_num_proc=min(16, os.cpu_count() or 8),
        packing=True,  # Can make training 5x faster for short sequences.
        # "wrapped" concatenates and chunks, which is what this pipeline trained on before trl
        # gained a choice here, and it keeps every token: a response longer than max_length spans
        # two blocks instead of being cut. The "bfd" default preserves sample boundaries but
        # slices every sequence to max_length first (pc.list_slice in trl's _pack_bfd), silently
        # dropping the tail of every long response — and it auto-enables padding_free, whose
        # document aware masking only works on flash attention variants, which unsloth's patched
        # attention does not advertise itself as
        packing_strategy="wrapped",
        # divided by the world size above, so that the effective batch does not depend on the
        # number of GPUs. Keep training_batch_size * gradient_accumulation_steps constant when
        # tuning either one, it sets how many optimizer steps a generation takes and therefore
        # how far it drifts from the last one
        gradient_accumulation_steps=ddp_accumulation_steps,
        warmup_steps=5,
        num_train_epochs=training_epochs,
        per_device_train_batch_size=training_batch_size,
        per_device_eval_batch_size=training_batch_size,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        # the 8bit optimizer exists to shrink optimizer state, which for a rank 16 adapter on a
        # 0.5B model is a few MB. The fused kernel is simply faster
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        output_dir="outputs",
        report_to="none",
        # nothing reads the intermediate checkpoints — the generation stage and run_attack.py both
        # read the explicit saves below — so writing them is pure disk traffic per generation
        save_strategy="no",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # the frozen base weights have requires_grad=False and are therefore not in DDP's
        # gradient buckets at all, so there is nothing unused to look for. Leaving this on costs a
        # full graph traversal per step
        ddp_find_unused_parameters=False,
    ),
)

# train the model
trainer_stats = trainer.train()

# print some fancy stats
if is_main:
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    training_percentage = round(used_memory_for_training / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} min. used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB (rank 0 of {world_size}).")
    print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        f"Peak reserved memory for training % of max memory = {training_percentage} %."
    )

# save the model. Rank 0 only: these are explicit save_pretrained calls, not Trainer.save_model,
# so nothing rank guards them and every rank would write the same files over each other.
# The local `tokenizer` is used rather than trainer.tokenizer, which transformers v5 removed in
# favour of trainer.processing_class — it is the same object either way
if is_main:
    adapter_dir = f"{MODEL_PATH}model_{generation}_bs{block_size}_{specifier_name}{gen_suffix}"
    trainer.model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_adapter=True,
        save_config=True,
    )
    tokenizer.save_pretrained(adapter_dir)
    # also save the model in fp16, which is what the vLLM generation engine and run_attack.py read.
    # The mixture suffix goes *before* _fp16, so the merged copy of a mixed generation is
    # model_{g}_bs{bs}_{name}_rdf{value}_fp16 — run_attack.resolve_collapsed_dir assumes that order
    trainer.model.save_pretrained_merged(
        f"{adapter_dir}_fp16",
        tokenizer,
        save_method="merged_16bit",
    )
    stamp_transformers_version(f"{adapter_dir}_fp16")
    stamp_transformers_version(adapter_dir)

# every rank has to reach this point before rank 0's files are considered complete by the
# orchestrator, which only waits for the torchrun process as a whole
if torch.distributed.is_available() and torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
