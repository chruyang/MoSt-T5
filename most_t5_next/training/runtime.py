"""Runtime settings shared by pretraining and downstream experiments."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
from typing import Any, ContextManager, Mapping

import numpy as np
import torch

from .optimization import OptimizationConfig


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    """Hardware-facing settings intentionally left open to users."""

    seed: int = 42
    precision: str = "bf16"
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    num_workers: int = 8
    prefetch_factor: int = 5
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every_updates: int = 10
    checkpoint_every_updates: int | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        for name in (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "prefetch_factor",
            "log_every_updates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.checkpoint_every_updates is not None and self.checkpoint_every_updates <= 0:
            raise ValueError("checkpoint_every_updates must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


def runtime_from_config(config: Mapping[str, Any]) -> TrainingRuntimeConfig:
    batching = config["batching"]
    dataloader = config["dataloader"]
    monitoring = config["monitoring"]
    return TrainingRuntimeConfig(
        seed=int(config["seed"]),
        precision=str(config["optimization"]["precision"]),
        micro_batch_size=int(batching["micro_batch_size"]),
        gradient_accumulation_steps=int(batching["gradient_accumulation_steps"]),
        num_workers=int(dataloader["num_workers"]),
        prefetch_factor=int(dataloader["prefetch_factor"]),
        pin_memory=bool(dataloader["pin_memory"]),
        persistent_workers=bool(dataloader["persistent_workers"]),
        log_every_updates=int(monitoring["log_every_updates"]),
        checkpoint_every_updates=(
            int(monitoring["checkpoint_every_updates"])
            if monitoring["checkpoint_every_updates"] is not None
            else None
        ),
    )


def optimization_from_config(
    config: Mapping[str, Any], phase_name: str
) -> OptimizationConfig:
    if phase_name not in {"phase_one", "phase_two"}:
        raise ValueError("phase_name must be phase_one or phase_two")
    optimization = config["optimization"]
    phase = optimization[phase_name]
    return OptimizationConfig(
        total_updates=int(config["curriculum"][phase_name]["total_updates"]),
        warmup_updates=int(phase["warmup_updates"]),
        base_learning_rate=float(phase["base_learning_rate"]),
        warmup_start_factor=float(optimization["warmup_start_factor"]),
        final_learning_rate=float(optimization["final_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        beta1=float(optimization["beta1"]),
        beta2=float(optimization["beta2"]),
        epsilon=float(optimization["epsilon"]),
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU, and all visible CUDA generators."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(
    precision: str, device: torch.device
) -> ContextManager[Any]:
    """Return the configured autocast context without changing model storage."""

    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("precision must be fp32, bf16, or fp16")


__all__ = [
    "TrainingRuntimeConfig",
    "autocast_context",
    "optimization_from_config",
    "runtime_from_config",
    "seed_everything",
]
