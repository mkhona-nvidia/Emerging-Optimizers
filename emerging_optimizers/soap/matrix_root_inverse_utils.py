# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch
from torch import Tensor

from emerging_optimizers import utils
from emerging_optimizers.utils import FP32MatmulPrecT


# OKLS uses "mixed" instead of "medium" in FP32MatmulPrecT.
OKLSPrecisionT = FP32MatmulPrecT | str  # accepts "mixed" as well


__all__ = ["mat_root_inv_via_scaled_cans"]


# ──────────── CANS polynomial coefficients (arXiv:2506.10935) ────────────
# The 1.01 safety factor is folded into these coefficients.
# Steps 0–8: beta unchanged, alpha = alpha_orig / 1.01.
# Step 9: beta = beta_orig / sqrt(1.01), alpha = alpha_orig / 1.01**1.5.
_CANS_COEFFS = (
    (5.182503604966906, -5.126830178299687),
    (2.586120737395915, -0.641538812403133),
    (2.567364126726186, -0.6391058222170474),
    (2.520560084348265, -0.6330225823828756),
    (2.410759275435182, -0.6186815444268036),
    (2.1883348130094173, -0.5893091162177136),
    (1.8595760874873613, -0.5449991062102938),
    (1.589020160467417, -0.5075811685214573),
    (1.5051653981684994, -0.4957799077972079),
    (1.4925557853149838, -0.49259266842078675),
)


# ──────────────── Per-step safe-scale ceilings (eigs in [0, 1]) ─────────────
# Without per-step rescaling the Z and Y iterates grow rapidly across the
# 10 Newton-Schulz steps; per-step rescaling keeps intermediates below M.
_Z_MAX = [
    1.0,
    5.183,
    13.403,
    34.409,
    86.731,
    209.09,
    457.55,
    850.85,
    1352.0,
    2035.0,
    3052.5,
]
_Y_MAX = [1.0, 1.297, 1.726, 1.473, 1.545, 1.415, 1.273, 1.148, 1.0, 1.0, 1.0]
_P_MAX = [1.0, 3.98, 3.95, 3.88, 3.71, 3.32, 2.60, 1.73, 1.15, 1.008, 1.0]


def _precompute_ns_constants(
    M: float,
) -> tuple[float, float, float, float, float, list[float], list[float], list[float], list[float], list[float]]:
    """Precompute all per-step scaling constants for a given magnitude ceiling M."""
    S_Z = [M / z for z in _Z_MAX]
    S_Y = [M / y for y in _Y_MAX]
    S_P = [M / p for p in _P_MAX]

    # Step-0 constants
    s_y_0 = S_Y[0]
    alpha_y_0 = _CANS_COEFFS[0][1] * S_Y[1] / (S_Y[0] ** 2)
    beta_y_0 = _CANS_COEFFS[0][0] * S_Y[1] / S_Y[0]
    z_scale_0 = _CANS_COEFFS[0][1] * S_Z[1] / S_Y[0]
    z_diag_add_0 = _CANS_COEFFS[0][0] * S_Z[1]

    # Steps 1–9 constants
    alpha_p: list[float] = []
    alpha_y: list[float] = []
    beta_y: list[float] = []
    alpha_z: list[float] = []
    beta_z: list[float] = []
    for k in range(1, 10):
        a, b = _CANS_COEFFS[k]
        alpha_p.append(S_P[k] / (S_Z[k] * S_Y[k]))
        alpha_y.append(b * S_Y[k + 1] / (S_Y[k] * S_P[k]))
        beta_y.append(a * S_Y[k + 1] / S_Y[k])
        alpha_z.append(b * S_Z[k + 1] / (S_Z[k] * S_P[k]))
        beta_z.append(a * S_Z[k + 1] / S_Z[k])

    # Absorb S_Z[10] into last-step coefficients so runtime only needs 1/√L
    alpha_z[8] /= S_Z[10]
    beta_z[8] /= S_Z[10]

    return (s_y_0, alpha_y_0, beta_y_0, z_scale_0, z_diag_add_0, alpha_p, alpha_y, beta_y, alpha_z, beta_z)


# FP16 ceiling: 16384 keeps half-precision GEMM operands safe (fixed).
_M_FP16 = 16384.0
_CONSTS_FP16 = _precompute_ns_constants(_M_FP16)

# FP32 ceiling: computed at runtime as sqrt(FP32_MAX / d) to use the actual
# matrix dimension and maximize dynamic range.
_FP32_MAX = torch.finfo(torch.float32).max


def _estimate_max_eigenvalue(A: Tensor) -> Tensor:
    """Strict upper bound on the spectral radius via min(Wolkowicz-Styan, Minc-Sainte-Marie).

    Cost is O(n²) — one pass over the matrix elements.  For SPD matrices this is a
    strict upper bound on the largest eigenvalue.

    Args:
        A: (..., d, d) symmetric positive-definite matrix.

    Returns:
        (...,) upper bound on the largest eigenvalue, one scalar per batch element.
    """
    n = A.size(-1)
    diag = A.diagonal(dim1=-2, dim2=-1)

    # ── Optimal scaling: max(|A/c|) = √FP32_MAX / n ──
    # Both bounds involve O(n²·max²) intermediates.  Scaling by
    # c = max(|diag|)·n/√FP32_MAX keeps every intermediate in FP32 range
    # while maximising dynamic-range usage (fewest small elements lost).
    # For SPD matrices max(|A_ij|) ≤ max(diag_i), so this is a safe bound.
    _FROB_SCALE = n / math.sqrt(torch.finfo(torch.float32).max)  # n / 1.844e19
    c = (diag.abs().max(dim=-1).values * _FROB_SCALE).clamp(min=1e-30)
    c_inv = 1.0 / c

    # Single n² materialisation: |A/c|
    abs_A_s = (A * c_inv[..., None, None]).abs()

    # ── Wolkowicz-Styan bound on A/c, then unscale ──
    m_s = diag.sum(dim=-1) * (c_inv / n)
    f_s_n = torch.linalg.matrix_norm(abs_A_s) * (1.0 / math.sqrt(n))
    s_s = torch.sqrt(torch.clamp((f_s_n + m_s) * (f_s_n - m_s), min=0.0))
    ws_bound = c * (m_s + s_s * math.sqrt(n - 1))

    # ── Minc-Sainte-Marie bound on A/c, then unscale ──
    d_s = torch.sum(abs_A_s, dim=-1)
    d_s_clamped = torch.clamp(d_s, min=1e-12)
    y_s = torch.einsum("...ij,...j->...i", abs_A_s, d_s_clamped)
    minc_bound = c * torch.max(y_s / d_s_clamped, dim=-1).values

    return torch.minimum(ws_bound, minc_bound)


def mat_root_inv_via_scaled_cans(
    x: Tensor,
    eps: float = 1e-12,
    fp32_matmul_prec: OKLSPrecisionT = "mixed",
) -> Tensor:
    """Compute inverse square root via scaled coupled CANS Newton-Schulz.

    CANS polynomial-based inverse-root computation from https://arxiv.org/abs/2506.10935,
    with per-step magnitude scaling to prevent intermediate overflow.  The 1.01 safety
    factor is folded into the polynomial coefficients.

    Normalization uses a tight spectral upper bound (the minimum of the Wolkowicz-Styan
    and Minc-Sainte-Marie bounds), which gives much better convergence for the smallest
    eigenvalues than the infinity norm.

    Three precision modes are supported, differing only in (1) the per-step magnitude
    ceiling, (2) the matmul precision context, and (3) whether intermediates are cast
    to FP16 before matrix multiplication:

    - ``"mixed"``: FP16 GEMM inputs with FP32 accumulation via ``torch.baddbmm`` with
      ``out_dtype=torch.float32``.  Per-step ceiling = 16384.
    - ``"high"``: TF32 matmul precision.  Per-step ceiling = sqrt(FP32_MAX / d).
    - ``"highest"``: Full FP32 matmul precision.  Per-step ceiling = sqrt(FP32_MAX / d).

    Args:
        x: A 2D symmetric positive-definite FP32 matrix or 3D batch of matrices.
        eps: Lower bound used when normalizing the matrices.
        fp32_matmul_prec: Precision used for matrix multiplications: ``"mixed"`` for FP16,
            ``"high"`` for TF32, or ``"highest"`` for FP32.

    Returns:
        The approximate inverse square root as an FP32 tensor with the same shape as ``x``.
    """
    if x.dim() not in (2, 3) or x.shape[-2] != x.shape[-1]:
        raise TypeError(f"x must be a square matrix or batch of square matrices, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"x must be in float32, got {x.dtype}")

    is_batched = x.dim() == 3
    if not is_batched:
        x = x.unsqueeze(0)

    B, d, _ = x.shape

    # Select constants and precision path.
    use_fp16 = fp32_matmul_prec == "mixed"
    if use_fp16:
        consts = _CONSTS_FP16
    else:
        consts = _precompute_ns_constants(math.sqrt(_FP32_MAX / d))
    (s_y_0, alpha_y_0, beta_y_0, z_scale_0, z_diag_add_0, alpha_p, alpha_y, beta_y, alpha_z, beta_z) = consts

    # "mixed" uses FP16 GEMMs inside NS but TF32 for the precision context.
    ctx_prec = "high" if fp32_matmul_prec == "mixed" else fp32_matmul_prec
    with utils.fp32_matmul_precision(ctx_prec):
        # ── Normalize: eigenvalues of Y₀ ∈ (0, 1] ──
        # The 1.01 safety margin is already folded into _CANS_COEFFS.
        L = _estimate_max_eigenvalue(x).clamp_min_(eps)
        Y = x * (s_y_0 / L.view(B, 1, 1))

        # ── Step 0: Z₁ = s_z₁·(b₀I + a₀Y₀),  Y₁ = s_y₁·(b₀Y₀ + a₀Y₀²) ──
        Z = Y.mul(z_scale_0)
        Z.diagonal(dim1=-2, dim2=-1).add_(z_diag_add_0)
        Y_low = Y.half() if use_fp16 else Y
        if use_fp16:
            Y = torch.baddbmm(Y, Y_low, Y_low, torch.float32, beta=beta_y_0, alpha=alpha_y_0)
        else:
            Y = torch.baddbmm(Y, Y_low, Y_low, beta=beta_y_0, alpha=alpha_y_0)

        # ── Steps 1–8 (full coupled update) ──
        for k in range(8):
            Y_low = Y.half() if use_fp16 else Y
            Z_low = Z.half() if use_fp16 else Z
            P_low = torch.baddbmm(Z_low, Z_low, Y_low, beta=0.0, alpha=alpha_p[k])
            if use_fp16:
                Y, Z = (
                    torch.baddbmm(Y, Y_low, P_low, torch.float32, beta=beta_y[k], alpha=alpha_y[k]),
                    torch.baddbmm(Z, P_low, Z_low, torch.float32, beta=beta_z[k], alpha=alpha_z[k]),
                )
            else:
                Y, Z = (
                    torch.baddbmm(Y, Y_low, P_low, beta=beta_y[k], alpha=alpha_y[k]),
                    torch.baddbmm(Z, P_low, Z_low, beta=beta_z[k], alpha=alpha_z[k]),
                )

        # ── Step 9: Z only, with final 1/(s_{z,10}·√L) fused into coefficients ──
        Y_low = Y.half() if use_fp16 else Y
        Z_low = Z.half() if use_fp16 else Z
        P_low = torch.baddbmm(Z_low, Z_low, Y_low, beta=0.0, alpha=alpha_p[8])
        inv_scale = torch.rsqrt(L).view(B, 1, 1)
        if use_fp16:
            W = torch.baddbmm(Z, P_low, Z_low, torch.float32, beta=beta_z[8], alpha=alpha_z[8])
        else:
            W = torch.baddbmm(Z, P_low, Z_low, beta=beta_z[8], alpha=alpha_z[8])
        W.mul_(inv_scale)
        W = (W + W.mT) * 0.5

    return W if is_batched else W.squeeze(0)
