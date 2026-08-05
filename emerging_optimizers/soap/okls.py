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
from typing import TYPE_CHECKING, Callable, Literal, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry
from emerging_optimizers.soap.matrix_root_inverse_utils import mat_root_inv_via_scaled_cans
from emerging_optimizers.utils import FP32MatmulPrecT


OKLSPrecisionT = Literal["highest", "high", "mixed"]


__all__ = ["OKLS", "update_kronecker_factors_okls"]


def _pack_sym(matrix: torch.Tensor) -> torch.Tensor:
    """Pack the upper triangle of a symmetric matrix."""
    dim = matrix.shape[-1]
    rows, cols = torch.triu_indices(dim, dim, device=matrix.device)
    return matrix[..., rows, cols]


def _unpack_sym(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Reconstruct a symmetric matrix from its packed upper triangle."""
    matrix = torch.empty(packed.shape[:-1] + (dim, dim), dtype=packed.dtype, device=packed.device)
    rows, cols = torch.triu_indices(dim, dim, device=packed.device)
    matrix[..., rows, cols] = packed
    matrix[..., cols, rows] = packed
    return matrix


def _update_inverse_roots(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    ridge_eps: float,
    cans_fp32_matmul_prec: OKLSPrecisionT,
) -> None:
    for kronecker_factor, inverse_root in zip(kronecker_factor_list, inverse_root_list, strict=True):
        inverse_root.copy_(
            mat_root_inv_via_scaled_cans(
                kronecker_factor,
                eps=ridge_eps,
                fp32_matmul_prec=cans_fp32_matmul_prec,
            )
        )


def _initialize_preconditioners(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    grad: torch.Tensor,
    ridge_eps: float,
    cans_fp32_matmul_prec: OKLSPrecisionT,
) -> None:
    rows, cols = grad.shape
    grad_norm_sq = grad.square().sum()
    factor_left, factor_right = kronecker_factor_list

    factor_left.copy_(grad @ grad.T)
    factor_left.mul_(torch.sqrt(rows / (cols * grad_norm_sq + ridge_eps)))
    factor_left.copy_((factor_left + factor_left.T) / 2.0)
    diagonal_shift_left = torch.linalg.norm(factor_left) / math.sqrt(rows)
    factor_left.diagonal().add_(diagonal_shift_left + ridge_eps)

    factor_right.copy_(grad.T @ grad)
    factor_right.mul_(torch.sqrt(cols / (rows * grad_norm_sq + ridge_eps)))
    factor_right.copy_((factor_right + factor_right.T) / 2.0)
    diagonal_shift_right = torch.linalg.norm(factor_right) / math.sqrt(cols)
    factor_right.diagonal().add_(diagonal_shift_right + ridge_eps)

    _update_inverse_roots(
        kronecker_factor_list,
        inverse_root_list,
        ridge_eps,
        cans_fp32_matmul_prec,
    )


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors_okls(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    ridge_eps: float,
) -> None:
    """Update KL-Shampoo factors using the previous inverse-square-root preconditioners.

    Args:
        kronecker_factor_list: Left and right covariance factors.
        inverse_root_list: Previous inverse square roots of the left and right factors.
        grad: Matrix gradient.
        shampoo_beta: EMA coefficient for the factors.
        ridge_eps: Diagonal stability offset.
    """
    if grad.dim() != 2:
        raise TypeError("OKLS is only supported for 2D tensors")

    factor_left, factor_right = kronecker_factor_list
    inverse_root_left, inverse_root_right = inverse_root_list
    rows, cols = grad.shape

    grad_right_preconditioned = grad @ inverse_root_right
    factor_left.addmm_(
        grad_right_preconditioned, grad_right_preconditioned.T, beta=shampoo_beta, alpha=(1 - shampoo_beta) / cols
    )
    factor_left.copy_((factor_left + factor_left.T) / 2.0)
    factor_left.diagonal().add_(ridge_eps)

    grad_left_preconditioned = inverse_root_left @ grad
    factor_right.addmm_(
        grad_left_preconditioned.T, grad_left_preconditioned, beta=shampoo_beta, alpha=(1 - shampoo_beta) / rows
    )
    factor_right.copy_((factor_right + factor_right.T) / 2.0)
    factor_right.diagonal().add_(ridge_eps)


@registry.register_optimizer("okls")
class OKLS(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Online KL-Shampoo with scaled CANS inverse roots and zero-staleness preconditioning.

    Symmetric Kronecker factors and inverse roots are stored as packed upper
    triangles and reconstructed transiently during each step.

    Args:
        params: Iterable of 2D CUDA parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        beta1: Nesterov momentum EMA coefficient.
        beta2: KL-Shampoo factor EMA coefficient.
        ridge_eps: Numerical stability offset added to the KL-Shampoo factors.
        weight_decay: Decoupled weight-decay coefficient.
        fp32_matmul_prec: Precision for all matrix multiplications:
            ``"mixed"`` for FP16 NS GEMMs with TF32 outer matmuls,
            ``"high"`` for TF32 everywhere, or ``"highest"`` for FP32 everywhere.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float,
        beta1: float = 0.9684,
        beta2: float = 0.9482,
        ridge_eps: float = 1e-9,
        weight_decay: float = 0.0,
        fp32_matmul_prec: OKLSPrecisionT = "mixed",
    ) -> None:
        self.weight_decay_method = "decoupled"
        self.fp32_matmul_prec: OKLSPrecisionT = fp32_matmul_prec
        # Map "mixed" -> "high" for the matmul precision context (TF32 for outer ops).
        self._ctx_prec: FP32MatmulPrecT = "high" if fp32_matmul_prec == "mixed" else fp32_matmul_prec

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if ridge_eps < 0.0:
            raise ValueError(f"Invalid ridge epsilon: {ridge_eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "ridge_eps": ridge_eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue

            if p.dim() != 2:
                raise TypeError("OKLS is only supported for 2D tensors")
            if not p.is_cuda:
                raise TypeError("OKLS only supports CUDA tensors")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                factor_left = p.new_zeros((p.shape[0], p.shape[0]), dtype=torch.float32)
                factor_right = p.new_zeros((p.shape[1], p.shape[1]), dtype=torch.float32)
                inverse_root_left = p.new_zeros((p.shape[0], p.shape[0]), dtype=torch.float32)
                inverse_root_right = p.new_zeros((p.shape[1], p.shape[1]), dtype=torch.float32)
                _initialize_preconditioners(
                    [factor_left, factor_right],
                    [inverse_root_left, inverse_root_right],
                    p.grad.to(torch.float32),
                    group["ridge_eps"],
                    self.fp32_matmul_prec,
                )
                state["L"] = _pack_sym(factor_left)
                state["R"] = _pack_sym(factor_right)
                state["P_L"] = _pack_sym(inverse_root_left)
                state["P_R"] = _pack_sym(inverse_root_right)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Args:
            closure: Unsupported; must be ``None``.
        """
        if closure is not None:
            raise ValueError("closure is not supported")

        from emerging_optimizers import utils

        with utils.fp32_matmul_precision(self._ctx_prec):
            for group in self.param_groups:
                self._init_group(group)

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue  # pragma: no cover

                    grad = p.grad.to(torch.float32)
                    state = self.state[p]
                    rows, cols = grad.shape
                    kronecker_factor_list = [
                        _unpack_sym(state["L"], rows),
                        _unpack_sym(state["R"], cols),
                    ]
                    inverse_root_list = [
                        _unpack_sym(state["P_L"], rows),
                        _unpack_sym(state["P_R"], cols),
                    ]
                    ridge_eps = group["ridge_eps"]

                    beta1 = group["beta1"]
                    state["exp_avg"].lerp_(grad, 1 - beta1)
                    nesterov_momentum = torch.lerp(grad, state["exp_avg"], beta1)

                    update_kronecker_factors_okls(
                        kronecker_factor_list=kronecker_factor_list,
                        inverse_root_list=inverse_root_list,
                        grad=grad,
                        shampoo_beta=group["beta2"],
                        ridge_eps=ridge_eps,
                    )
                    _update_inverse_roots(
                        kronecker_factor_list,
                        inverse_root_list,
                        ridge_eps,
                        self.fp32_matmul_prec,
                    )
                    state["L"].copy_(_pack_sym(kronecker_factor_list[0]))
                    state["R"].copy_(_pack_sym(kronecker_factor_list[1]))
                    state["P_L"].copy_(_pack_sym(inverse_root_list[0]))
                    state["P_R"].copy_(_pack_sym(inverse_root_list[1]))

                    preconditioned_update = inverse_root_list[0] @ nesterov_momentum @ inverse_root_list[1]
                    nesterov_variance = ((1 - beta1) / (1 + beta1)) * (1 + 2 * beta1 - 2 * beta1**3)
                    momentum_scale = nesterov_variance**-0.5
                    shape_scale = math.sqrt(rows / cols) / (math.sqrt(rows) + math.sqrt(cols))

                    self._apply_weight_decay_inplace(
                        p,
                        grad,
                        group["lr"],
                        group["weight_decay"],
                    )
                    p.add_(
                        preconditioned_update.to(p.dtype),
                        alpha=-group["lr"] * momentum_scale * shape_scale,
                    )
                    state["step"] += 1

        return None
