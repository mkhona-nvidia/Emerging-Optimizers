import torch
import torch.nn.functional as F


def compute_soft_thresholded_diff(x: torch.Tensor, quantile: float, epsilon: float = 1e-9) -> torch.Tensor:
    """Compute the difference between input and its soft-thresholded version.

    Applies soft thresholding where the threshold is set to the quantile
    of absolute values plus epsilon. Returns the component that was thresholded away.

    Based on https://github.com/huawei-noah/noah-research/blob/master/ROOT/examples/example_train.py.

    Arguments:
        x: Input tensor to be soft-thresholded.
        quantile: Quantile value between 0 and 1 (e.g., 0.5 for median, 0.9 for 90th percentile).
        epsilon: Small value added to the threshold for numerical stability.

    Returns:
        The difference between the input and the soft-thresholded output (the thresholded component).
    """

    threshold = torch.quantile(x.flatten().float(), quantile, interpolation="linear") + epsilon
    o = torch.sign(x) * F.relu(torch.abs(x) - threshold)
    b = x - o
    return b
