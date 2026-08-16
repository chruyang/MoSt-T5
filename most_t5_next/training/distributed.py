"""Explicit task-homogeneous data-parallel layout for formal pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from most_t5_next.interfaces import PHASE_TASKS


class DistributedLayoutError(ValueError):
    pass


@dataclass(frozen=True)
class RankTaskAssignment:
    phase: int
    rank: int
    world_size: int
    task: str
    task_replica_index: int
    task_replicas: int


def rank_task_assignment(
    config: Mapping[str, Any], *, phase: int, rank: int, world_size: int
) -> RankTaskAssignment:
    """Resolve one rank without relying on DataLoader worker ordering."""

    if phase not in PHASE_TASKS:
        raise DistributedLayoutError("phase must be 1 or 2")
    distributed = config["distributed"]
    expected_world_size = int(distributed["world_size"])
    if world_size != expected_world_size:
        raise DistributedLayoutError(
            f"formal layout requires world_size={expected_world_size}, got {world_size}"
        )
    if not 0 <= rank < world_size:
        raise DistributedLayoutError("rank is outside the distributed world")
    phase_name = "phase_one" if phase == 1 else "phase_two"
    rank_tasks = tuple(distributed["rank_tasks"][phase_name])
    if len(rank_tasks) != world_size:
        raise DistributedLayoutError("rank-task layout does not cover the world")
    task = str(rank_tasks[rank])
    if task not in PHASE_TASKS[phase]:
        raise DistributedLayoutError(
            "rank-task layout contains a task from another phase"
        )
    task_ranks = tuple(index for index, value in enumerate(rank_tasks) if value == task)
    return RankTaskAssignment(
        phase=phase,
        rank=rank,
        world_size=world_size,
        task=task,
        task_replica_index=task_ranks.index(rank),
        task_replicas=len(task_ranks),
    )


def task_batch_partitions(
    config: Mapping[str, Any], *, phase: int
) -> dict[str, tuple[int, int]]:
    """Return the physical split of each rank-local logical batch."""

    if phase not in PHASE_TASKS:
        raise DistributedLayoutError("phase must be 1 or 2")
    phase_name = "phase_one" if phase == 1 else "phase_two"
    rows = config["batching"]["task_partitions"][phase_name]
    result: dict[str, tuple[int, int]] = {}
    for task in PHASE_TASKS[phase]:
        row = rows[task]
        result[task] = (
            int(row["micro_batch_size"]),
            int(row["gradient_accumulation_steps"]),
        )
    return result


def validate_rank_tasks(
    rank_tasks: Sequence[str], *, phase: int, world_size: int
) -> None:
    """Validate that rank multiplicities encode the frozen task ratios."""

    if len(rank_tasks) != world_size:
        raise DistributedLayoutError("rank-task layout does not cover the world")
    allowed = PHASE_TASKS[phase]
    if any(task not in allowed for task in rank_tasks):
        raise DistributedLayoutError("rank-task layout contains an unknown task")
    multiplicities = [rank_tasks.count(task) for task in allowed]
    if not multiplicities or len(set(multiplicities)) != 1:
        raise DistributedLayoutError("rank-task layout must balance phase tasks")


__all__ = [
    "DistributedLayoutError",
    "RankTaskAssignment",
    "rank_task_assignment",
    "task_batch_partitions",
    "validate_rank_tasks",
]
