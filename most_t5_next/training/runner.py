"""Configuration-driven two-phase MoSt-T5 training loop."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from .curriculum import CurriculumSchedule, TaskSpec
from .engine import forward_task
from .optimization import OptimizationConfig, build_optimizer_and_schedule
from .runtime import (
    TrainingRuntimeConfig,
    autocast_context,
    optimization_from_config,
    runtime_from_config,
    seed_everything,
)


SCHEMA_VERSION = "most-t5/training-run/v1"


class TrainingError(RuntimeError):
    pass


class PhaseBatchProvider(Protocol):
    def __call__(
        self, task: TaskSpec, update: int
    ) -> Sequence[Mapping[str, Any]]: ...


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }


def _target_count(batch: Mapping[str, Any]) -> int:
    labels = batch.get("labels")
    if not isinstance(labels, Tensor) or labels.ndim != 2:
        raise TrainingError("training microbatch lacks rank-two labels")
    count = int(labels.ne(-100).sum().item())
    if count <= 0:
        raise TrainingError("training microbatch has no target tokens")
    return count


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: Any,
    phase: int,
    next_update: int,
    optimization: OptimizationConfig,
    runtime: TrainingRuntimeConfig,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "next_update": next_update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.scheduler.state_dict(),
        "schedule_completed_updates": schedule.completed_updates,
        "optimization": asdict(optimization),
        "runtime": asdict(runtime),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: Any,
    phase: int,
    optimization: OptimizationConfig,
) -> int:
    payload = torch.load(path, map_location="cpu")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or int(payload.get("phase", -1)) != phase
        or payload.get("optimization") != asdict(optimization)
    ):
        raise TrainingError("checkpoint protocol differs from the requested phase")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    schedule.scheduler.load_state_dict(payload["schedule"])
    schedule.completed_updates = int(payload["schedule_completed_updates"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    next_update = int(payload["next_update"])
    if next_update != schedule.completed_updates:
        raise TrainingError("checkpoint optimizer and schedule boundaries differ")
    return next_update


def run_training_phase(
    *,
    model: torch.nn.Module,
    phase: int,
    batch_provider: PhaseBatchProvider,
    optimization: OptimizationConfig,
    runtime: TrainingRuntimeConfig,
    output_dir: str | Path,
    device: str | torch.device,
    resume_checkpoint: str | Path | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Train one phase; task selection is constant inside each update."""

    curriculum = CurriculumSchedule(phase, optimization.total_updates)
    if runtime.precision == "fp16":
        raise TrainingError(
            "formal runner does not admit fp16 overflow-skipped updates; use bf16 or fp32"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    optimizer, learning_rate_schedule = build_optimizer_and_schedule(
        model, optimization
    )
    next_update = 0
    if resume_checkpoint is not None:
        next_update = _restore_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            schedule=learning_rate_schedule,
            phase=phase,
            optimization=optimization,
        )
    task_updates: Counter[str] = Counter()
    task_targets: Counter[str] = Counter()
    losses: list[float] = []
    started = time.time()
    model.train()
    for update in range(next_update, optimization.total_updates):
        task = curriculum.task_at(update)
        microbatches = tuple(batch_provider(task, update))
        if len(microbatches) != runtime.gradient_accumulation_steps:
            raise TrainingError("batch provider returned the wrong accumulation count")
        target_counts = tuple(_target_count(batch) for batch in microbatches)
        target_total = sum(target_counts)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for batch, target_count in zip(microbatches, target_counts):
            moved = _move(batch, resolved_device)
            with autocast_context(runtime.precision, resolved_device):
                output = forward_task(model, task.name, moved)
                loss = getattr(output, "loss", None)
                if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                    raise TrainingError("model returned an invalid loss")
                weight = float(target_count) / float(target_total)
                normalized_loss = loss * weight
            normalized_loss.backward()
            weighted_loss += float(loss.detach().float()) * weight
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimization.gradient_clip_norm
            )
        )
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        learning_rate_schedule.step()
        completed = update + 1
        task_updates[task.name] += 1
        task_targets[task.name] += target_total
        losses.append(weighted_loss)
        if writer is not None and (
            completed == 1 or completed % runtime.log_every_updates == 0
        ):
            writer.add_scalar(f"phase_{phase}/loss", weighted_loss, completed)
            writer.add_scalar(
                f"phase_{phase}/learning_rate", learning_rate_used, completed
            )
            writer.add_scalar(
                f"phase_{phase}/gradient_norm", gradient_norm, completed
            )
            writer.add_scalar(
                f"phase_{phase}/task/{task.name}/loss", weighted_loss, completed
            )
            writer.flush()
        interval = runtime.checkpoint_every_updates
        if interval is not None and (
            completed % interval == 0 or completed == optimization.total_updates
        ):
            _save_checkpoint(
                destination / f"phase-{phase}-step-{completed:08d}.pt",
                model=model,
                optimizer=optimizer,
                schedule=learning_rate_schedule,
                phase=phase,
                next_update=completed,
                optimization=optimization,
                runtime=runtime,
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "phase": phase,
        "optimizer_reinitialized_for_phase": True,
        "optimizer_updates": optimization.total_updates,
        "task_updates": dict(sorted(task_updates.items())),
        "task_target_tokens": dict(sorted(task_targets.items())),
        "target_token_normalized_accumulation": True,
        "optimizer_update_batching": {
            "logical_batch_size": runtime.effective_batch_size,
            "micro_batch_size": runtime.micro_batch_size,
            "gradient_accumulation_steps": runtime.gradient_accumulation_steps,
            "sample_before_microbatch_split": True,
        },
        "optimization": asdict(optimization),
        "runtime": asdict(runtime),
        "loss": {
            "mean": sum(losses) / len(losses) if losses else None,
            "last": losses[-1] if losses else None,
        },
        "wall_seconds": time.time() - started,
    }
    (destination / f"phase-{phase}-training-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_two_phase_pretraining(
    *,
    model: torch.nn.Module,
    phase_one_batch_provider: PhaseBatchProvider,
    phase_two_batch_provider_factory: Callable[[], PhaseBatchProvider],
    config: Mapping[str, Any],
    output_dir: str | Path,
    device: str | torch.device,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Run Phase I then restart optimizer/scheduler for Phase II."""

    seed_everything(int(config["seed"]))
    runtime = runtime_from_config(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    phase_one = run_training_phase(
        model=model,
        phase=1,
        batch_provider=phase_one_batch_provider,
        optimization=optimization_from_config(config, "phase_one"),
        runtime=runtime,
        output_dir=destination,
        device=device,
        writer=writer,
    )
    boundary = destination / "phase-one-model-boundary.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "phase": 1,
            "model": model.state_dict(),
            "optimizer_state_included": False,
            "scheduler_state_included": False,
        },
        boundary,
    )
    phase_two_batch_provider = phase_two_batch_provider_factory()
    phase_two = run_training_phase(
        model=model,
        phase=2,
        batch_provider=phase_two_batch_provider,
        optimization=optimization_from_config(config, "phase_two"),
        runtime=runtime,
        output_dir=destination,
        device=device,
        writer=writer,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "phase_boundary": str(boundary.resolve()),
        "phase_boundary_model_only": True,
        "phase_one": phase_one,
        "phase_two": phase_two,
    }
    (destination / "training-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "PhaseBatchProvider",
    "SCHEMA_VERSION",
    "TrainingError",
    "run_training_phase",
    "run_two_phase_pretraining",
]
