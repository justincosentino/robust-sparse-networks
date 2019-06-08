import numpy as np


def prune_by_percent(kernels, masks, percents):
    """Return new masks that involve pruning the smallest of the final weights.

    Args:
        percents: A dictionary determining the percent by which to prune each layer.
        Keys are layer names and values are floats between 0 and 1 (inclusive).
        masks: A dictionary containing the current masks. Keys are strings and
        values are numpy arrays with values in {0, 1}.
        kernels: The weights at the end of the last training run. A
        dictionary whose keys are strings and whose values are numpy arrays.

    Returns:
        A dictionary containing the newly-pruned masks.
    """

    def prune_by_percent_once(percent, mask, kernel):
        # Put the weights that aren't masked out in sorted order.
        sorted_weights = np.sort(np.abs(kernel[mask == 1]))

        # Determine the cutoff for weights to be pruned.
        cutoff_index = np.round(percent * sorted_weights.size).astype(int)
        cutoff = sorted_weights[cutoff_index]

        # Prune all weights below the cutoff.
        return np.where(np.abs(kernel) <= cutoff, np.zeros(mask.shape), mask)

    new_masks = {}
    for layer, kernel in kernels.items():
        new_masks[layer] = prune_by_percent_once(percents[layer], masks[layer], kernel)

    return new_masks
