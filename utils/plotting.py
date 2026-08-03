"""helper functions for the perplexity plots"""

import torch


def visible_perplexity_range(
    perplexity_dict: dict, y_limit: float, num_bins: int = 401
) -> tuple:
    """
    Determines the x-axis range of the perplexity histogram which actually contains visible
    data.

    The perplexity distributions have extremely heavy tails: a small but non-negligible part
    of the samples is spread over many decades, so even a 99.9% quantile still reaches up to
    1e10. Those bins contain so few samples that they are drawn below the plot's lower
    y-limit and are invisible anyway, which would only leave the right side of the plot
    empty. The range is therefore cut off after the last bin whose density is still above the
    lower y-limit.

    Args:
        perplexity_dict (dict): the perplexities per generation
        y_limit (float): the lower y-limit of the plot, i.e., the visibility threshold
        num_bins (int): number of bins used to probe the densities

    Returns:
        tuple: (lower_limit, upper_limit, num_clipped, num_total), where the limits are
            rounded to full decades and num_clipped is the number of perplexities which are
            outside of the returned range
    """
    values = torch.tensor(
        [
            perplexity
            for perplexities in perplexity_dict.values()
            for perplexity in perplexities
        ]
    )
    # exp() of a large loss overflows to inf, which would turn the whole logspace into nan
    values = values[torch.isfinite(values) & (values > 0)]
    if values.numel() == 0:
        raise ValueError(
            "All perplexities are non-finite or non-positive, there is nothing to plot."
        )

    # probe the densities over the full range of the data first
    full_lower = 10 ** torch.floor(torch.log10(values.min()))
    full_upper = 10 ** torch.ceil(torch.log10(values.max()))
    edges = torch.logspace(
        torch.log10(full_lower), torch.log10(full_upper), steps=num_bins
    )
    bin_widths = edges[1:] - edges[:-1]

    # a bin is visible if any generation's density in it reaches the lower y-limit. seaborn's
    # stat="density" normalizes the counts by the number of samples and by the bin width, so
    # the same has to be done here to be able to compare against the y-limit
    visible = torch.zeros(len(bin_widths), dtype=torch.bool)
    for perplexities in perplexity_dict.values():
        generation_values = torch.tensor(perplexities)
        generation_values = generation_values[
            torch.isfinite(generation_values) & (generation_values > 0)
        ]
        if generation_values.numel() == 0:
            continue

        # bucketize returns 0 for values below the first edge, so shift into bin indices
        bin_indices = torch.bucketize(generation_values, edges).clamp(
            1, len(bin_widths)
        )
        counts = torch.bincount(bin_indices - 1, minlength=len(bin_widths))
        densities = counts / (generation_values.numel() * bin_widths)
        visible |= densities >= y_limit

    visible_bins = visible.nonzero().flatten()
    if visible_bins.numel() == 0:
        # nothing would be visible at all, so fall back to the full range of the data
        return full_lower.item(), full_upper.item(), 0, values.numel()

    lower_limit = 10 ** torch.floor(torch.log10(edges[visible_bins[0]]))
    upper_limit = 10 ** torch.ceil(torch.log10(edges[visible_bins[-1] + 1]))
    # keep at least one decade, otherwise both limits collapse onto the same value if all
    # visible perplexities fall onto the same power of ten
    upper_limit = torch.maximum(upper_limit, lower_limit * 10)

    num_clipped = int(((values < lower_limit) | (values > upper_limit)).sum())

    return lower_limit.item(), upper_limit.item(), num_clipped, values.numel()
