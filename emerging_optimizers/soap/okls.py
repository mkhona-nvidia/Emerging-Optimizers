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
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.soap.cans_utils import scaled_cans_coupled_ns


__all__ = ["OKLS", "update_kronecker_factors_okls"]


def _init_preconditioners(
    grad: torch.Tensor,
    factor_left: torch.Tensor,
    factor_right: torch.Tensor,
    inverse_root_left: torch.Tensor,
    inverse_root_right: torch.Tensor,
    ridge_eps: float,
) -> None:
    batch_size, rows, cols = grad.shape
    grad_norm_sq = grad.square().sum(dim=(-2, -1))

    grad_gram_left = grad @ grad.mT
    scale_left = torch.sqrt(rows / (cols * grad_norm_sq + ridge_eps)).view(batch_size, 1, 1)
    factor_left.copy_(scale_left * grad_gram_left)
    factor_left.copy_((factor_left + factor_left.mT) / 2.0)
    diagonal_shift_left = factor_left.square().sum(dim=(-2, -1)).sqrt() / math.sqrt(rows)
    factor_left.diagonal(dim1=-2, dim2=-1).add_(diagonal_shift_left.unsqueeze(-1) + ridge_eps)

    grad_gram_right = grad.mT @ grad
    scale_right = torch.sqrt(cols / (rows * grad_norm_sq + ridge_eps)).view(batch_size, 1, 1)
    factor_right.copy_(scale_right * grad_gram_right)
    factor_right.copy_((factor_right + factor_right.mT) / 2.0)
    diagonal_shift_right = factor_right.square().sum(dim=(-2, -1)).sqrt() / math.sqrt(cols)
    factor_right.diagonal(dim1=-2, dim2=-1).add_(diagonal_shift_right.unsqueeze(-1) + ridge_eps)

    inverse_root_left.copy_(scaled_cans_coupled_ns(factor_left, eps=ridge_eps))
    inverse_root_right.copy_(scaled_cans_coupled_ns(factor_right, eps=ridge_eps))


def update_kronecker_factors_okls(
    kronecker_factor_list: list[torch.Tensor],
    inverse_root_list: list[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    ridge_eps: float,
) -> None:
    """Update batched KL-Shampoo factors using the previous inverse-square-root preconditioners.

    Args:
        kronecker_factor_list: Left and right batched covariance factors.
        inverse_root_list: Previous inverse square roots of the left and right factors.
        grad: Batched matrix gradients.
        shampoo_beta: EMA coefficient for the factors.
        ridge_eps: Diagonal stability offset.
    """
    if grad.dim() != 3:
        raise TypeError(f"grad must be a batched 3D tensor, got {grad.dim()}D")

    factor_left, factor_right = kronecker_factor_list
    inverse_root_left, inverse_root_right = inverse_root_list
    rows, cols = grad.shape[-2:]

    grad_right_preconditioned = grad @ inverse_root_right
    factor_left.baddbmm_(
        grad_right_preconditioned,
        grad_right_preconditioned.mT,
        beta=shampoo_beta,
        alpha=(1 - shampoo_beta) / cols,
    )
    factor_left.copy_((factor_left + factor_left.mT) / 2.0)
    factor_left.diagonal(dim1=-2, dim2=-1).add_(ridge_eps)

    grad_left_preconditioned = inverse_root_left @ grad
    factor_right.baddbmm_(
        grad_left_preconditioned.mT,
        grad_left_preconditioned,
        beta=shampoo_beta,
        alpha=(1 - shampoo_beta) / rows,
    )
    factor_right.copy_((factor_right + factor_right.mT) / 2.0)
    factor_right.diagonal(dim1=-2, dim2=-1).add_(ridge_eps)


@registry.register_optimizer("okls")
class OKLS(optim.Optimizer):
    """Online KL-Shampoo with scaled CANS inverse roots and zero-staleness preconditioning.

    Args:
        params: Iterable of 2D or 3D parameters to optimize.
        lr: Learning rate and fixed peak learning rate for AdamC weight decay.
        beta1: Nesterov momentum EMA coefficient.
        beta2: KL-Shampoo factor EMA coefficient.
        ridge_eps: Numerical stability offset added to the KL-Shampoo factors.
        weight_decay: AdamC decoupled weight-decay coefficient.
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
    ) -> None:
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
            "lr_peak": lr,
            "beta1": beta1,
            "beta2": beta2,
            "ridge_eps": ridge_eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                if param.dim() not in (2, 3):
                    raise ValueError(f"OKLS only supports 2D or 3D parameters, got shape {tuple(param.shape)}")

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(self, group: dict, skip_non_grad_params: bool = True) -> None:
        for param in group["params"]:
            if skip_non_grad_params and param.grad is None:
                continue
            if param.dim() not in (2, 3):
                raise ValueError(f"OKLS only supports 2D or 3D parameters, got shape {tuple(param.shape)}")

            state = self.state[param]
            if len(state) != 0:
                continue

            if param.dim() == 2:
                batch_size = 1
                rows, cols = param.shape
            else:
                batch_size, rows, cols = param.shape

            state["step"] = 0
            state["momentum"] = torch.zeros(
                batch_size,
                rows,
                cols,
                dtype=torch.float32,
                device=param.device,
            )
            state["S_a"] = torch.zeros(batch_size, rows, rows, dtype=torch.float32, device=param.device)
            state["S_b"] = torch.zeros(batch_size, cols, cols, dtype=torch.float32, device=param.device)
            state["P_a"] = torch.zeros(batch_size, rows, rows, dtype=torch.float32, device=param.device)
            state["P_b"] = torch.zeros(batch_size, cols, cols, dtype=torch.float32, device=param.device)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one Online KL-Shampoo optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns the loss.

        Returns:
            The closure loss when a closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)

        for group in self.param_groups:
            lr = group["lr"]
            lr_peak = group["lr_peak"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            ridge_eps = group["ridge_eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.float()
                is_2d = param.dim() == 2
                if is_2d:
                    grad = grad.unsqueeze(0)

                state = self.state[param]
                factor_list = [state["S_a"], state["S_b"]]
                inverse_root_list = [state["P_a"], state["P_b"]]

                if state["step"] == 0:
                    _init_preconditioners(
                        grad,
                        factor_list[0],
                        factor_list[1],
                        inverse_root_list[0],
                        inverse_root_list[1],
                        ridge_eps,
                    )

                momentum = state["momentum"]
                momentum.lerp_(grad, 1 - beta1)
                nesterov_momentum = torch.lerp(grad, momentum, beta1)

                update_kronecker_factors_okls(
                    factor_list,
                    inverse_root_list,
                    grad,
                    beta2,
                    ridge_eps,
                )
                inverse_root_list[0].copy_(scaled_cans_coupled_ns(factor_list[0], eps=ridge_eps))
                inverse_root_list[1].copy_(scaled_cans_coupled_ns(factor_list[1], eps=ridge_eps))

                preconditioned_update = inverse_root_list[0] @ nesterov_momentum @ inverse_root_list[1]
                rows, cols = grad.shape[-2:]
                nesterov_variance = ((1 - beta1) / (1 + beta1)) * (1 + 2 * beta1 - 2 * beta1**3)
                momentum_scale = nesterov_variance**-0.5
                shape_scale = math.sqrt(rows / cols) / (math.sqrt(rows) + math.sqrt(cols))

                lr_ratio = lr / lr_peak if lr_peak != 0.0 else 0.0
                param.mul_(1 - weight_decay * lr * lr_ratio)
                if is_2d:
                    preconditioned_update = preconditioned_update.squeeze(0)
                param.add_(
                    preconditioned_update.to(param.dtype),
                    alpha=-lr * momentum_scale * shape_scale,
                )
                state["step"] += 1

        return loss
