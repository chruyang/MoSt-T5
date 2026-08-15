"""Reference-compatible optimization for one curriculum phase."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass(frozen=True)
class OptimizationConfig:
    total_updates: int
    warmup_updates: int
    base_learning_rate: float
    warmup_start_factor: float
    final_learning_rate: float
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_updates, bool)
            or not isinstance(self.total_updates, int)
            or isinstance(self.warmup_updates, bool)
            or not isinstance(self.warmup_updates, int)
        ):
            raise ValueError("update counts must be integers")
        if not 0 < self.warmup_updates < self.total_updates:
            raise ValueError("warmup_updates must lie inside the phase budget")
        if self.base_learning_rate <= 0 or self.final_learning_rate < 0:
            raise ValueError("learning rates must be nonnegative")
        if not 0 < self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must lie in (0, 1]")


class AdamWScale(torch.optim.Optimizer):
    """AdamW with the parameter-RMS scaling used by 3D-MolT5."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-6,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("optimizer rates must be nonnegative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Adam beta values must lie in [0, 1)")
        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        return tensor.norm(2) / math.sqrt(tensor.numel())

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamWScale does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(correction2) / correction1
                step_size *= max(1.0e-3, float(self._rms(parameter)))
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)
                if group["weight_decay"]:
                    parameter.add_(
                        parameter, alpha=-group["lr"] * group["weight_decay"]
                    )
        return loss


class CosineSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, config: OptimizationConfig):
        self.config = config
        self.completed_updates = 0
        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=config.warmup_start_factor,
                    end_factor=1.0,
                    total_iters=config.warmup_updates,
                    last_epoch=-1,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=config.total_updates - config.warmup_updates,
                    eta_min=config.final_learning_rate,
                ),
            ],
            milestones=[config.warmup_updates],
        )

    def step(self) -> None:
        if self.completed_updates >= self.config.total_updates:
            raise RuntimeError("learning-rate schedule is complete")
        self.scheduler.step()
        self.completed_updates += 1

    def get_last_lr(self) -> list[float]:
        return self.scheduler.get_last_lr()


def _parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = ("bias", "LayerNorm", "layernorm", "layer_norm", "ln")
    named = tuple(model.named_parameters())
    return [
        {
            "params": [
                p
                for n, p in named
                if p.requires_grad and not any(x in n for x in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named if p.requires_grad and any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def build_optimizer_and_schedule(
    model: torch.nn.Module, config: OptimizationConfig
) -> tuple[AdamWScale, CosineSchedule]:
    optimizer = AdamWScale(
        _parameter_groups(model, config.weight_decay),
        lr=config.base_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
    return optimizer, CosineSchedule(optimizer, config)


__all__ = [
    "AdamWScale",
    "CosineSchedule",
    "OptimizationConfig",
    "build_optimizer_and_schedule",
]
