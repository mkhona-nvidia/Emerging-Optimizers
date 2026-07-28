# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


__all__ = ["scaled_cans_coupled_ns"]

_CANS_COEFFS = (
    (5.182503604966906, -5.178098480082684),
    (2.586120737395915, -0.6479542005271643),
    (2.567364126726186, -0.6454968804392178),
    (2.520560084348265, -0.6393528082067044),
    (2.410759275435182, -0.6248683598710716),
    (2.1883348130094173, -0.5952022073798908),
    (1.8595760874873613, -0.5504490972723968),
    (1.589020160467417, -0.5126569802066718),
    (1.5051653981684994, -0.5007377068751799),
    (1.5, -0.5),
)
_CANS_M = 16384.0
_CANS_Z_MAX = (1.0, 5.183, 13.403, 34.409, 86.731, 209.09, 457.55, 850.85, 1352.0, 2035.0, 3052.5)
_CANS_Y_MAX = (1.0, 1.297, 1.726, 1.473, 1.545, 1.415, 1.273, 1.148, 1.0, 1.0, 1.0)
_CANS_P_MAX = (1.0, 3.98, 3.95, 3.88, 3.71, 3.32, 2.60, 1.73, 1.15, 1.008, 1.0)
_CANS_S_Z = tuple(_CANS_M / value for value in _CANS_Z_MAX)
_CANS_S_Y = tuple(_CANS_M / value for value in _CANS_Y_MAX)
_CANS_S_P = tuple(_CANS_M / value for value in _CANS_P_MAX)
_CANS_ALPHA_Y_0 = _CANS_COEFFS[0][1] * _CANS_S_Y[1] / (_CANS_S_Y[0] ** 2)
_CANS_BETA_Y_0 = _CANS_COEFFS[0][0] * _CANS_S_Y[1] / _CANS_S_Y[0]
_CANS_Z_SCALE_0 = _CANS_COEFFS[0][1] * _CANS_S_Z[1] / _CANS_S_Y[0]
_CANS_Z_DIAG_ADD_0 = _CANS_COEFFS[0][0] * _CANS_S_Z[1]
_CANS_ALPHA_P: list[float] = []
_CANS_ALPHA_Y: list[float] = []
_CANS_BETA_Y: list[float] = []
_CANS_ALPHA_Z: list[float] = []
_CANS_BETA_Z: list[float] = []
for _cans_k in range(1, len(_CANS_COEFFS)):
    _cans_a, _cans_b = _CANS_COEFFS[_cans_k]
    _CANS_ALPHA_P.append(_CANS_S_P[_cans_k] / (_CANS_S_Z[_cans_k] * _CANS_S_Y[_cans_k]))
    _CANS_ALPHA_Y.append(_cans_b * _CANS_S_Y[_cans_k + 1] / (_CANS_S_Y[_cans_k] * _CANS_S_P[_cans_k]))
    _CANS_BETA_Y.append(_cans_a * _CANS_S_Y[_cans_k + 1] / _CANS_S_Y[_cans_k])
    _CANS_ALPHA_Z.append(_cans_b * _CANS_S_Z[_cans_k + 1] / (_CANS_S_Z[_cans_k] * _CANS_S_P[_cans_k]))
    _CANS_BETA_Z.append(_cans_a * _CANS_S_Z[_cans_k + 1] / _CANS_S_Z[_cans_k])
_CANS_ALPHA_Z[-1] /= _CANS_S_Z[-1]
_CANS_BETA_Z[-1] /= _CANS_S_Z[-1]


def _estimate_max_eigenvalue(x: Tensor, eps: float) -> Tensor:
    n = x.size(-1)
    diag = x.diagonal(dim1=-2, dim2=-1)
    mean = diag.sum(dim=-1) / n
    sq_norm = torch.sum(x**2, dim=(-2, -1))
    variance = torch.clamp((sq_norm / n) - (mean**2), min=0.0)
    ws_bound = mean + torch.sqrt(variance) * math.sqrt(n - 1)

    abs_x = torch.abs(x)
    row_sum = torch.sum(abs_x, dim=-1).clamp_min_(eps)
    minc_bound = torch.max(torch.einsum("...ij,...j->...i", abs_x, row_sum) / row_sum, dim=-1).values
    return torch.minimum(ws_bound, minc_bound)


def _cans_baddbmm(
    input_tensor: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: float,
    alpha: float,
) -> Tensor:
    if input_tensor.is_cuda:
        return torch.baddbmm(
            input_tensor,
            batch1.half(),
            batch2.half(),
            beta=beta,
            alpha=alpha,
            out_dtype=torch.float32,
        )
    return torch.baddbmm(input_tensor, batch1, batch2, beta=beta, alpha=alpha)


def scaled_cans_coupled_ns(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Compute a batched inverse square root with scaled coupled CANS Newton-Schulz.

    Args:
        x: Batched symmetric positive-definite matrices.
        eps: Lower bound used when normalizing the matrices.

    Returns:
        The approximate inverse square root of each matrix.
    """
    if x.dim() != 3 or x.shape[-2] != x.shape[-1]:
        raise TypeError(f"x must be a batched square matrix, got shape {tuple(x.shape)}")

    input_dtype = x.dtype
    x = x.float()
    batch_size = x.shape[0]
    max_eigval = (_estimate_max_eigenvalue(x, eps) * 1.01).clamp_min_(eps)
    Y = x * (_CANS_S_Y[0] / max_eigval.view(batch_size, 1, 1))

    Z = Y.mul(_CANS_Z_SCALE_0)
    Z.diagonal(dim1=-2, dim2=-1).add_(_CANS_Z_DIAG_ADD_0)
    Y = _cans_baddbmm(Y, Y, Y, beta=_CANS_BETA_Y_0, alpha=_CANS_ALPHA_Y_0)

    for k in range(8):
        P = _cans_baddbmm(Z, Z, Y, beta=0.0, alpha=_CANS_ALPHA_P[k])
        Y, Z = (
            _cans_baddbmm(Y, Y, P, beta=_CANS_BETA_Y[k], alpha=_CANS_ALPHA_Y[k]),
            _cans_baddbmm(Z, P, Z, beta=_CANS_BETA_Z[k], alpha=_CANS_ALPHA_Z[k]),
        )

    P = _cans_baddbmm(Z, Z, Y, beta=0.0, alpha=_CANS_ALPHA_P[8])
    W = _cans_baddbmm(Z, P, Z, beta=_CANS_BETA_Z[8], alpha=_CANS_ALPHA_Z[8])
    W.mul_(torch.rsqrt(max_eigval).view(batch_size, 1, 1))
    W = (W + W.mT) / 2.0
    return W.to(input_dtype)
