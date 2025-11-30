import torch
import torch.nn.functional as F


def simple_quantile(tensor: torch.Tensor, q: float) -> torch.Tensor:
    """Simple quantile implementation (unoptimized for GPU).

    Arguments:
        tensor: Input tensor from which to compute the quantile.
        q: Quantile value between 0 and 1 (e.g., 0.5 for median, 0.9 for 90th percentile).

    Returns:
        The q-th quantile of the flattened input tensor, computed using linear interpolation.
    """
    tensor = tensor.flatten()
    num_elements = tensor.numel()

    index = torch.tensor(q * (num_elements - 1), device=tensor.device)

    lower_index = torch.floor(index).long()
    upper_index = torch.ceil(index).long()

    k_lower = lower_index + 1
    k_upper = upper_index + 1

    if k_lower == k_upper:
        return tensor.kthvalue(k_lower).values

    lower_value = tensor.kthvalue(k_lower).values
    upper_value = tensor.kthvalue(k_upper).values

    weight = index - lower_index
    return torch.lerp(lower_value, upper_value, weight)


def compute_soft_thresholded_diff(x: torch.Tensor, quantile: float, epsilon: float) -> torch.Tensor:
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
    threshold = simple_quantile(x.abs().float(), quantile) + epsilon
    o = torch.sign(x) * F.relu(torch.abs(x) - threshold)
    b = x - o
    return b
