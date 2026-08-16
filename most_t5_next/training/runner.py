"""Configuration-driven two-phase MoSt-T5 training loop."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .curriculum import CurriculumSchedule, TASKS, TaskSpec
from .engine import forward_task
from .optimization import OptimizationConfig, build_optimizer_and_schedule
from .runtime import (
    TrainingRuntimeConfig,
    autocast_context,
    optimization_from_config,
    runtime_from_config,
    seed_everything,
)


SCHEMA_VERSION = "most-t5/training-run/v3"


class TrainingError(RuntimeError):
    pass


class PhaseBatchProvider(Protocol):
    def __call__(
        self, task: TaskSpec, update: int
    ) -> Sequence[Mapping[str, Any]]: ...


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _world_size() -> int:
    return dist.get_world_size() if _distributed() else 1


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _mean_across_ranks(value: float, device: torch.device) -> float:
    if not _distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / _world_size()).item())


def _gather_objects(value: Any) -> list[Any]:
    if not _distributed():
        return [value]
    gathered: list[Any] = [None] * _world_size()
    dist.all_gather_object(gathered, value)
    return gathered


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
    rank_rng_states: Sequence[Mapping[str, Any]],
    rank_progress_states: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "next_update": next_update,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.scheduler.state_dict(),
        "schedule_completed_updates": schedule.completed_updates,
        "optimization": asdict(optimization),
        "runtime": asdict(runtime),
        "rank_rng_states": list(rank_rng_states),
        "rank_progress_states": list(rank_progress_states),
        "protocol": dict(protocol),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "next_update": next_update,
                "optimization": asdict(optimization),
                "runtime": asdict(runtime),
                "protocol": dict(protocol),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)


def _restore_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: Any,
    phase: int,
    optimization: OptimizationConfig,
    runtime: TrainingRuntimeConfig,
    protocol: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or int(payload.get("phase", -1)) != phase
        or payload.get("optimization") != asdict(optimization)
        or payload.get("runtime") != asdict(runtime)
        or payload.get("protocol") != dict(protocol)
    ):
        raise TrainingError("checkpoint protocol differs from the requested phase")
    _unwrap_model(model).load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    schedule.scheduler.load_state_dict(payload["schedule"])
    schedule.completed_updates = int(payload["schedule_completed_updates"])
    rank_rng_states = payload.get("rank_rng_states")
    if rank_rng_states is None:
        torch.set_rng_state(payload["torch_rng_state"])
        cuda_state = payload.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    else:
        if len(rank_rng_states) != _world_size():
            raise TrainingError("checkpoint distributed world size differs")
        state = rank_rng_states[_rank()]
        if int(state.get("rank", -1)) != _rank():
            raise TrainingError("checkpoint RNG rank ordering differs")
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        cuda_state = state.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state)
    next_update = int(payload["next_update"])
    if next_update != schedule.completed_updates:
        raise TrainingError("checkpoint optimizer and schedule boundaries differ")
    if not 0 <= next_update <= optimization.total_updates:
        raise TrainingError("checkpoint update lies outside the requested phase")
    rank_progress_states = payload.get("rank_progress_states")
    if not isinstance(rank_progress_states, list) or len(rank_progress_states) != _world_size():
        raise TrainingError("checkpoint progress state differs from the distributed world")
    progress = rank_progress_states[_rank()]
    if int(progress.get("rank", -1)) != _rank():
        raise TrainingError("checkpoint progress rank ordering differs")
    return next_update, progress


def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read the non-tensor resume contract before constructing data providers."""

    checkpoint_path = Path(path)
    metadata_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".metadata.json"
    )
    payload = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TrainingError("checkpoint schema is not supported by this runner")
    phase = int(payload.get("phase", -1))
    next_update = int(payload.get("next_update", -1))
    if phase not in {1, 2} or next_update < 0:
        raise TrainingError("checkpoint phase boundary is invalid")
    return {
        "schema_version": payload["schema_version"],
        "phase": phase,
        "next_update": next_update,
        "optimization": payload.get("optimization"),
        "runtime": payload.get("runtime"),
        "protocol": payload.get("protocol"),
    }


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
    checkpoint_protocol: Mapping[str, Any] | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Train one phase with one synchronized gradient reduction per update."""

    fixed_task_name = getattr(batch_provider, "fixed_task", None)
    curriculum = CurriculumSchedule(
        phase,
        optimization.total_updates,
        require_complete_cycles=fixed_task_name is None,
    )
    if runtime.precision == "fp16":
        raise TrainingError(
            "formal runner does not admit fp16 overflow-skipped updates; use bf16 or fp32"
        )
    destination = Path(output_dir)
    if _rank() == 0:
        destination.mkdir(parents=True, exist_ok=True)
    _barrier()
    resolved_device = torch.device(device)
    model.to(resolved_device)
    optimizer, learning_rate_schedule = build_optimizer_and_schedule(
        model, optimization
    )
    protocol = {} if checkpoint_protocol is None else dict(checkpoint_protocol)
    next_update = 0
    task_updates: Counter[str] = Counter()
    task_records: Counter[str] = Counter()
    task_targets: Counter[str] = Counter()
    task_partitions: dict[str, dict[str, int]] = {}
    loss_sum = 0.0
    loss_count = 0
    last_loss: float | None = None
    previous_wall_seconds = 0.0
    if resume_checkpoint is not None:
        next_update, progress = _restore_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            schedule=learning_rate_schedule,
            phase=phase,
            optimization=optimization,
            runtime=runtime,
            protocol=protocol,
        )
        task_updates.update(progress["task_updates"])
        task_records.update(progress["task_records"])
        task_targets.update(progress["task_targets"])
        task_partitions.update(progress["task_partitions"])
        loss_sum = float(progress["loss_sum"])
        loss_count = int(progress["loss_count"])
        last_loss = progress["last_loss"]
        previous_wall_seconds = float(progress["wall_seconds"])
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    started = time.time()
    model.train()
    if _distributed() and fixed_task_name is None:
        raise TrainingError("distributed pretraining requires one fixed task per rank")
    for update in range(next_update, optimization.total_updates):
        task = (
            TASKS[fixed_task_name]
            if fixed_task_name is not None
            else curriculum.task_at(update)
        )
        microbatches = tuple(batch_provider(task, update))
        partition_method = getattr(batch_provider, "partition_for_task", None)
        if callable(partition_method):
            micro_batch_size, accumulation_steps = partition_method(task.name)
        else:
            micro_batch_size = runtime.micro_batch_size
            accumulation_steps = runtime.gradient_accumulation_steps
        if len(microbatches) != accumulation_steps:
            raise TrainingError("batch provider returned the wrong accumulation count")
        batch_sizes = []
        for batch in microbatches:
            input_ids = batch.get("input_ids")
            if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
                raise TrainingError("training microbatch lacks rank-two input_ids")
            batch_sizes.append(int(input_ids.shape[0]))
        if any(size != micro_batch_size for size in batch_sizes):
            raise TrainingError("batch provider returned the wrong microbatch size")
        if sum(batch_sizes) != runtime.effective_batch_size:
            raise TrainingError("optimizer update does not contain the logical batch")
        partition = {
            "micro_batch_size": int(micro_batch_size),
            "gradient_accumulation_steps": int(accumulation_steps),
        }
        previous_partition = task_partitions.setdefault(task.name, partition)
        if previous_partition != partition:
            raise TrainingError("task batch partition changed during the phase")
        target_counts = tuple(_target_count(batch) for batch in microbatches)
        target_total = sum(target_counts)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        for microbatch_index, (batch, target_count) in enumerate(
            zip(microbatches, target_counts)
        ):
            synchronization = nullcontext()
            if microbatch_index + 1 < accumulation_steps:
                no_sync = getattr(model, "no_sync", None)
                if callable(no_sync):
                    synchronization = no_sync()
            with synchronization:
                moved = _move(batch, resolved_device)
                with autocast_context(runtime.precision, resolved_device):
                    output = forward_task(model, task.name, moved)
                    loss = getattr(output, "loss", None)
                    if (
                        not isinstance(loss, Tensor)
                        or loss.ndim != 0
                        or not torch.isfinite(loss)
                    ):
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
        task_records[task.name] += sum(batch_sizes)
        task_targets[task.name] += target_total
        global_loss = _mean_across_ranks(weighted_loss, resolved_device)
        loss_sum += global_loss
        loss_count += 1
        last_loss = global_loss
        should_log = completed == 1 or completed % runtime.log_every_updates == 0
        if should_log:
            rank_metrics = _gather_objects(
                {"task": task.name, "loss": weighted_loss}
            )
            if writer is not None:
                writer.add_scalar(f"phase_{phase}/loss", global_loss, completed)
                writer.add_scalar(
                    f"phase_{phase}/learning_rate", learning_rate_used, completed
                )
                writer.add_scalar(
                    f"phase_{phase}/gradient_norm", gradient_norm, completed
                )
                by_task: dict[str, list[float]] = {}
                for metric in rank_metrics:
                    by_task.setdefault(metric["task"], []).append(metric["loss"])
                for task_name, task_losses in sorted(by_task.items()):
                    writer.add_scalar(
                        f"phase_{phase}/task/{task_name}/loss",
                        sum(task_losses) / len(task_losses),
                        completed,
                    )
                writer.flush()
        interval = runtime.checkpoint_every_updates
        if interval is not None and (
            completed % interval == 0 or completed == optimization.total_updates
        ):
            local_rng_state = {
                "rank": _rank(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(resolved_device)
                if resolved_device.type == "cuda"
                else None,
            }
            rank_rng_states = _gather_objects(local_rng_state)
            local_progress_state = {
                "rank": _rank(),
                "task_updates": dict(task_updates),
                "task_records": dict(task_records),
                "task_targets": dict(task_targets),
                "task_partitions": dict(task_partitions),
                "loss_sum": loss_sum,
                "loss_count": loss_count,
                "last_loss": last_loss,
                "wall_seconds": previous_wall_seconds + time.time() - started,
            }
            rank_progress_states = _gather_objects(local_progress_state)
            if _rank() == 0:
                checkpoint_path = (
                    destination / f"phase-{phase}-step-{completed:08d}.pt"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    schedule=learning_rate_schedule,
                    phase=phase,
                    next_update=completed,
                    optimization=optimization,
                    runtime=runtime,
                    rank_rng_states=rank_rng_states,
                    rank_progress_states=rank_progress_states,
                    protocol=protocol,
                )
                latest = destination / "latest-checkpoint.json"
                latest_temporary = latest.with_suffix(latest.suffix + ".tmp")
                latest_temporary.write_text(
                    json.dumps(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase": phase,
                            "next_update": completed,
                            "checkpoint": checkpoint_path.name,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                latest_temporary.replace(latest)
            _barrier()
    local_summary = {
        "rank": _rank(),
        "task": fixed_task_name,
        "task_updates": dict(task_updates),
        "task_records": dict(task_records),
        "task_targets": dict(task_targets),
        "task_partitions": dict(task_partitions),
    }
    rank_summaries = _gather_objects(local_summary)
    merged_partitions: dict[str, dict[str, int]] = {}
    merged_task_updates: dict[str, int] = {}
    task_rank_updates: Counter[str] = Counter()
    merged_task_records: Counter[str] = Counter()
    merged_task_targets: Counter[str] = Counter()
    for summary in rank_summaries:
        for task_name, partition in summary["task_partitions"].items():
            previous = merged_partitions.setdefault(task_name, partition)
            if previous != partition:
                raise TrainingError("task partition differs across replicas")
        for task_name, count in summary["task_updates"].items():
            merged_task_updates[task_name] = max(
                merged_task_updates.get(task_name, 0), int(count)
            )
            task_rank_updates[task_name] += int(count)
        merged_task_records.update(summary["task_records"])
        merged_task_targets.update(summary["task_targets"])
    unique_partitions = {
        (row["micro_batch_size"], row["gradient_accumulation_steps"])
        for row in merged_partitions.values()
    }
    optimizer_update_batching: dict[str, Any] = {
        "rank_local_logical_batch_size": runtime.effective_batch_size,
        "global_logical_batch_size": runtime.effective_batch_size * _world_size(),
        "sample_before_microbatch_split": True,
        "gradient_syncs_per_optimizer_update": 1,
    }
    if len(unique_partitions) == 1:
        micro_batch_size, accumulation_steps = next(iter(unique_partitions))
        optimizer_update_batching.update(
            {
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
            }
        )
    else:
        optimizer_update_batching["task_partitions"] = dict(
            sorted(merged_partitions.items())
        )
    peak_memory: dict[str, float] | None = None
    if resolved_device.type == "cuda":
        bytes_per_mib = 1024.0**2
        peak_memory = {
            "allocated_mib": torch.cuda.max_memory_allocated(resolved_device)
            / bytes_per_mib,
            "reserved_mib": torch.cuda.max_memory_reserved(resolved_device)
            / bytes_per_mib,
        }
    rank_peak_memory = _gather_objects(
        {"rank": _rank(), "task": fixed_task_name, "memory": peak_memory}
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "phase": phase,
        "optimizer_reinitialized_for_phase": True,
        "optimizer_updates": optimization.total_updates,
        "task_updates": dict(sorted(merged_task_updates.items())),
        "task_rank_updates": dict(sorted(task_rank_updates.items())),
        "task_records": dict(sorted(merged_task_records.items())),
        "task_target_tokens": dict(sorted(merged_task_targets.items())),
        "target_token_normalized_accumulation": True,
        "distributed_loss_weighting": "equal_rank_after_rank_local_token_normalization",
        "optimizer_update_batching": optimizer_update_batching,
        "distributed": {
            "enabled": _distributed(),
            "world_size": _world_size(),
            "rank_tasks": [summary["task"] for summary in rank_summaries],
            "find_unused_parameters_required": _distributed(),
        },
        "optimization": asdict(optimization),
        "runtime": asdict(runtime),
        "loss": {
            "mean": loss_sum / loss_count if loss_count else None,
            "last": last_loss,
        },
        "cuda_peak_memory": rank_peak_memory,
        "wall_seconds": previous_wall_seconds + time.time() - started,
        "resumed_from": str(Path(resume_checkpoint).resolve())
        if resume_checkpoint is not None
        else None,
    }
    if _rank() == 0:
        (destination / f"phase-{phase}-training-manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _barrier()
    return report


def run_two_phase_pretraining(
    *,
    model: torch.nn.Module,
    phase_one_batch_provider: PhaseBatchProvider | None,
    phase_two_batch_provider_factory: Callable[[], PhaseBatchProvider],
    config: Mapping[str, Any],
    output_dir: str | Path,
    device: str | torch.device,
    resume_checkpoint: str | Path | None = None,
    checkpoint_protocol: Mapping[str, Any] | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Run Phase I then restart optimizer/scheduler for Phase II."""

    seed_everything(int(config["seed"]) + _rank())
    runtime = runtime_from_config(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resume_phase = None
    if resume_checkpoint is not None:
        resume_phase = read_checkpoint_metadata(resume_checkpoint)["phase"]
    phase_one_manifest = destination / "phase-1-training-manifest.json"
    boundary = destination / "phase-one-model-boundary.pt"
    if resume_phase == 2:
        if not boundary.is_file() or not phase_one_manifest.is_file():
            raise TrainingError(
                "Phase II resume requires the completed Phase I boundary and manifest"
            )
        phase_one = json.loads(phase_one_manifest.read_text(encoding="utf-8"))
        if phase_one.get("status") != "pass":
            raise TrainingError("Phase I manifest is not admitted for Phase II resume")
    else:
        if phase_one_batch_provider is None:
            raise TrainingError("Phase I batch provider is required")
        phase_one = run_training_phase(
            model=model,
            phase=1,
            batch_provider=phase_one_batch_provider,
            optimization=optimization_from_config(config, "phase_one"),
            runtime=runtime,
            output_dir=destination,
            device=device,
            resume_checkpoint=resume_checkpoint if resume_phase == 1 else None,
            checkpoint_protocol=checkpoint_protocol,
            writer=writer,
        )
        if _rank() == 0:
            boundary_payload = {
                "schema_version": SCHEMA_VERSION,
                "phase": 1,
                "model": _unwrap_model(model).state_dict(),
                "optimizer_state_included": False,
                "scheduler_state_included": False,
            }
            temporary = boundary.with_suffix(boundary.suffix + ".tmp")
            torch.save(boundary_payload, temporary)
            temporary.replace(boundary)
        _barrier()
    phase_two_batch_provider = phase_two_batch_provider_factory()
    phase_two = run_training_phase(
        model=model,
        phase=2,
        batch_provider=phase_two_batch_provider,
        optimization=optimization_from_config(config, "phase_two"),
        runtime=runtime,
        output_dir=destination,
        device=device,
        resume_checkpoint=resume_checkpoint if resume_phase == 2 else None,
        checkpoint_protocol=checkpoint_protocol,
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
    if _rank() == 0:
        (destination / "training-manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _barrier()
    return report


__all__ = [
    "PhaseBatchProvider",
    "SCHEMA_VERSION",
    "TrainingError",
    "read_checkpoint_metadata",
    "run_training_phase",
    "run_two_phase_pretraining",
]
