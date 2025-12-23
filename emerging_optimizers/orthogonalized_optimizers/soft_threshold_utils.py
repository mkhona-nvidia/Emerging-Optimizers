import torch
import torch.nn.functional as F

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz


def compute_soft_thresholded_diff(X: torch.Tensor, quantile: float, epsilon: float = 1e-9) -> torch.Tensor:
    """Compute the difference between input and its soft-thresholded version.

    Applies soft thresholding where the threshold is set to the quantile
    of absolute values plus epsilon. Returns the component that was thresholded away.

    Based on https://github.com/huawei-noah/noah-research/blob/master/ROOT/examples/example_train.py.

    Arguments:
        X: Input tensor to be soft-thresholded.
        quantile: Quantile value between 0 and 1 (e.g., 0.5 for median, 0.9 for 90th percentile).
        epsilon: Small value added to the threshold for numerical stability.

    Returns:
        The difference between the input and the soft-thresholded output (the thresholded component).
    """

    threshold = torch.quantile(X.flatten().float(), quantile, interpolation="linear") + epsilon
    o = torch.sign(X) * F.relu(torch.abs(X) - threshold)
    b = X - o
    return b


def spectral_soft_thresholding(X: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Spectral soft-thresholding on the input matrix.

    Applies spectral soft-thresholding on the input matrix.

    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        alpha: The alpha parameter.

    Returns:
        The spectral soft-thresholded tensor.
    """
    if needs_transpose := X.shape[0] > X.shape[1]:
        X = X.T

    OX = newton_schulz(X, steps=8, coefficient_type="polar_express")
    aX = alpha * OX - X
    result = (1 / 2) * (alpha * OX + X + aX @ newton_schulz(aX, steps=8, coefficient_type="polar_express").T @ OX)

    if needs_transpose:
        result = result.T

    return result
