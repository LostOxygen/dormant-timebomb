"""shared per-sample perplexity computation

This lives in its own module so that calculate_perplexity.py (which produces the plotted
histograms) and calibrate_surrogate.py (which fits the data-space surrogate against them) can
not drift apart. The calibration target has to be the very same statistic that is plotted,
otherwise the surrogate is fitted to something else than what it is compared on.
"""

import torch
from torch.nn import functional as F

# number of token positions whose logits are upcasted to float32 at once. This bounds the
# memory of the loss computation independently of the batch size and the sequence length
# (2048 positions need ~2.3GB for the float32 copy and cross_entropy's internal buffer)
CE_CHUNK_POSITIONS: int = 2048
# maximum number of (padded) tokens per forward pass. The logits of a forward pass are
# batch x sequence x vocabulary, i.e., the memory depends on the number of tokens and not on
# the number of samples. Since the generated datasets contain longer and longer responses
# with every generation, a fixed number of samples per forward pass would need more and more
# memory. Capping the tokens instead keeps the peak memory identical for every generation
# (65536 tokens need ~20GB for the float16 logits of a 151936 token vocabulary)
MAX_TOKENS_PER_FORWARD: int = 65536


SCORING_SYSTEM_PROMPT: str = "You are a helpful assistant for code completion."


def format_scoring_prompts(tokenizer, instructions: list, responses: list) -> list:
    """Chat templates (instruction, response) pairs the way the scored corpora are templated.

    Lives here for the same reason sample_perplexities does: the histogram worker
    (calculate_perplexity.py) and anything that wants to be comparable to it have to template
    identically. A different system prompt, or a missing assistant turn, changes the token
    sequence being scored and therefore the number, without anything raising.

    Args:
        tokenizer: the tokenizer whose chat template is applied
        instructions (list): the user turns
        responses (list): the assistant turns, aligned with `instructions`

    Returns:
        list: one templated string per pair
    """
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_special_tokens=False,
        )
        for instruction, response in zip(instructions, responses)
    ]


def sample_perplexities(
    model,
    tokenizer,
    formatted_prompts: list,
    max_length: int,
    device: str = "cuda",
    ce_chunk_positions: int = CE_CHUNK_POSITIONS,
    max_tokens_per_forward: int = MAX_TOKENS_PER_FORWARD,
) -> list:
    """
    Calculates the perplexity for every single prompt of a batch.

    This is a function and not inlined into the caller's loop on purpose: in a flat script the
    tensors would stay bound to module level names after the loop and keep their VRAM
    allocated, so torch.cuda.empty_cache() could not reclaim anything between generations.

    The tokenizer's padding_side matters for the result and is left to the caller: the plotted
    metric is computed with right padding, so anything that wants to be comparable to it has to
    use right padding too.

    Args:
        model: the model to measure the perplexity with
        tokenizer: its tokenizer, with pad_token and padding_side already set
        formatted_prompts (list): the already chat templated prompts of one batch
        max_length (int): truncation length. Has to be passed explicitly, otherwise the
            tokenizer truncates to the tokenizer's model_max_length (131072 for Qwen2.5), which
            would allow single batches that are magnitudes larger than what the model was
            loaded with
        device (str): device to run the forward passes on
        ce_chunk_positions (int): token positions per cross entropy chunk
        max_tokens_per_forward (int): padded token budget of a single forward pass

    Returns:
        list: one perplexity per prompt, in the order of the input prompts
    """
    inputs = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # split the batch into micro batches of a constant token budget, so that longer
    # sequences result in fewer samples per forward pass instead of in more memory
    sequence_length = inputs["input_ids"].shape[1]
    micro_batch_size = max(1, max_tokens_per_forward // sequence_length)

    perplexities = []
    for micro_start in range(0, len(formatted_prompts), micro_batch_size):
        micro_end = micro_start + micro_batch_size
        input_ids = inputs["input_ids"][micro_start:micro_end].to(device)
        attention_mask = inputs["attention_mask"][micro_start:micro_end].to(device)

        # calculate the perplexity for every datapoint of the micro batch. The loss has to
        # be computed manually (instead of passing labels to the model) since the model
        # would average it over the whole batch and would include the padding tokens
        with torch.no_grad():
            # use_cache=False since nothing is generated here and the KV cache would only
            # allocate additional memory
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits

            # shift by one position to predict the next token
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            current_batch_size, num_positions = shift_labels.shape

            # the logits are batch x sequence x vocabulary and thus by far the biggest
            # allocation. Upcasting all of them to float32 at once would need twice their
            # size again (plus the same amount inside cross_entropy), so the loss is
            # computed in chunks of at most ce_chunk_positions positions instead. This keeps
            # the additional memory constant instead of growing with the batch size
            chunk_len = max(1, ce_chunk_positions // current_batch_size)
            token_losses = torch.empty(
                (current_batch_size, num_positions),
                dtype=torch.float32,
                device=logits.device,
            )
            for start in range(0, num_positions, chunk_len):
                end = min(start + chunk_len, num_positions)
                token_losses[:, start:end] = F.cross_entropy(
                    logits[:, start:end, :].float().transpose(1, 2),
                    shift_labels[:, start:end],
                    reduction="none",
                )

            # free the logits before the reduction, they are the largest tensor by far
            del logits

            # mask out the padding tokens and average over the real tokens only to get one
            # loss (and therefore perplexity) per single sample
            sample_losses = (token_losses * shift_mask).sum(dim=1) / (
                shift_mask.sum(dim=1).clamp(min=1)
            )
            perplexities.extend(torch.exp(sample_losses).tolist())

    return perplexities
