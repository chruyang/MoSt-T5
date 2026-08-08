"""Frozen optimizer and learning-rate schedule for the PF-1 four-cell screen.

The protocol is deliberately shared by A0/A1/M0/M1.  It follows the
AdamWScale family used by the official 3D-MolT5 and CAMT5 training code, while
fixing the shorter PF-1 budget in one place so that individual cells cannot
silently acquire different optimization settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class PF1OptimizationProtocol:
    base_learning_rate: float = 1.0e-3
    warmup_updates: int = 100
    total_updates: int = 1000
    final_learning_rate: float = 1.0e-5
    warmup_start_factor: float = 0.1
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-6

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


FROZEN_PF1_PROTOCOL = PF1OptimizationProtocol()

# Keep the completed 32x4 PF-1 screen exactly resumable while giving the
# preregistered GraphPorts gate its independently named 64x2 protocol.  The
# CLI selects one of these complete contracts; individual numeric
# hyperparameters are intentionally not exposed.
PF1_SCREEN_PROTOCOL_ID = "pf1-screen-32x4-v1"
G_CODEC_PROTOCOL_ID = "graphports-codec-screen-64x2-v1"
G_CODEC_PF1_PROTOCOL = PF1OptimizationProtocol(
    micro_batch_size=64,
    gradient_accumulation_steps=2,
)
PF1_PROTOCOLS = {
    PF1_SCREEN_PROTOCOL_ID: FROZEN_PF1_PROTOCOL,
    G_CODEC_PROTOCOL_ID: G_CODEC_PF1_PROTOCOL,
}


def resolve_pf1_protocol(protocol_id: str) -> PF1OptimizationProtocol:
    try:
        return PF1_PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise ValueError("unknown frozen PF-1 optimization protocol") from exc


def identify_pf1_protocol(protocol: PF1OptimizationProtocol) -> str:
    for protocol_id, candidate in PF1_PROTOCOLS.items():
        if protocol == candidate:
            return protocol_id
    return "custom-internal-protocol"


class AdamWScale(torch.optim.Optimizer):
    """AdamW with the parameter-RMS step scaling used by the reference models."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-6,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError("learning rate must be nonnegative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if eps < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if weight_decay < 0.0:
            raise ValueError("weight decay must be nonnegative")
        if not isinstance(scale_parameter, bool):
            raise ValueError("scale_parameter must be boolean")
        super().__init__(
            params,
            defaults={
                "lr": float(lr),
                "betas": tuple(float(value) for value in betas),
                "eps": float(eps),
                "weight_decay": float(weight_decay),
                "scale_parameter": scale_parameter,
            },
        )

    @staticmethod
    def _rms(tensor: Any) -> Any:
        return torch.linalg.vector_norm(tensor) / math.sqrt(tensor.numel())

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
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = (
                    group["lr"]
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )
                rms_scale = (
                    max(1.0e-3, float(self._rms(parameter).item()))
                    if group.get("scale_parameter", True)
                    else 1.0
                )
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-(step_size * rms_scale),
                )
                if group["weight_decay"] > 0.0:
                    parameter.add_(
                        parameter,
                        alpha=-(group["lr"] * group["weight_decay"]),
                    )
        return loss


def learning_rate_for_update(
    update_number: int,
    protocol: PF1OptimizationProtocol = FROZEN_PF1_PROTOCOL,
) -> float:
    """Return the LR used by one-based optimizer update ``update_number``."""

    if not 1 <= update_number <= protocol.total_updates:
        raise ValueError("update_number is outside the PF-1 schedule")
    warmup_start = protocol.base_learning_rate * protocol.warmup_start_factor
    if update_number <= protocol.warmup_updates:
        if protocol.warmup_updates == 1:
            return protocol.base_learning_rate
        progress = (update_number - 1) / (protocol.warmup_updates - 1)
        return warmup_start + progress * (
            protocol.base_learning_rate - warmup_start
        )

    decay_progress = (update_number - protocol.warmup_updates) / (
        protocol.total_updates - protocol.warmup_updates
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return protocol.final_learning_rate + (
        protocol.base_learning_rate - protocol.final_learning_rate
    ) * cosine


class PF1LearningRateSchedule:
    """Small stateful schedule whose state is directly checkpointable."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        protocol: PF1OptimizationProtocol = FROZEN_PF1_PROTOCOL,
    ) -> None:
        self.optimizer = optimizer
        self.protocol = protocol
        self.completed_updates = 0
        self._set_learning_rate(learning_rate_for_update(1, protocol))

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def learning_rate_for_next_update(self) -> float:
        if self.completed_updates >= self.protocol.total_updates:
            return self.protocol.final_learning_rate
        return learning_rate_for_update(self.completed_updates + 1, self.protocol)

    def step(self) -> None:
        if self.completed_updates >= self.protocol.total_updates:
            raise RuntimeError("PF-1 learning-rate schedule is already complete")
        self.completed_updates += 1
        if self.completed_updates < self.protocol.total_updates:
            self._set_learning_rate(
                learning_rate_for_update(self.completed_updates + 1, self.protocol)
            )
        else:
            self._set_learning_rate(self.protocol.final_learning_rate)

    def state_dict(self) -> dict[str, int]:
        return {"completed_updates": self.completed_updates}

    def load_state_dict(self, state: dict[str, object]) -> None:
        value = state.get("completed_updates")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("schedule state lacks an integer completed_updates")
        if not 0 <= value <= self.protocol.total_updates:
            raise ValueError("schedule completed_updates is out of range")
        self.completed_updates = value
        self._set_learning_rate(self.learning_rate_for_next_update())


def build_pf1_optimizer(
    model: Any,
    protocol: PF1OptimizationProtocol = FROZEN_PF1_PROTOCOL,
) -> AdamWScale:
    return AdamWScale(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=protocol.base_learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        eps=protocol.epsilon,
        weight_decay=protocol.weight_decay,
    )


def clip_pf1_gradients(
    model: Any,
    protocol: PF1OptimizationProtocol = FROZEN_PF1_PROTOCOL,
) -> float:
    value = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=protocol.gradient_clip_norm,
        norm_type=2.0,
    )
    return float(value.detach().float().cpu().item())


__all__ = [
    "AdamWScale",
    "FROZEN_PF1_PROTOCOL",
    "G_CODEC_PF1_PROTOCOL",
    "G_CODEC_PROTOCOL_ID",
    "PF1_PROTOCOLS",
    "PF1_SCREEN_PROTOCOL_ID",
    "PF1LearningRateSchedule",
    "PF1OptimizationProtocol",
    "build_pf1_optimizer",
    "clip_pf1_gradients",
    "identify_pf1_protocol",
    "learning_rate_for_update",
    "resolve_pf1_protocol",
]
