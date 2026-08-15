"""Shared model call used by all six pretraining tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from most_t5_next.data.model_batch import disable_geometry, model_batch
from most_t5_next.modeling.model import MoStT5

from .curriculum import TASKS


def forward_task(model: MoStT5, task_name: str, batch: Mapping[str, Any]) -> Any:
    """Route one collated batch through the task's frozen information path."""

    try:
        task = TASKS[task_name]
    except KeyError as exc:
        raise ValueError(f"unknown pretraining task: {task_name}") from exc

    model_inputs = model_batch(batch)
    if task.name == "M":
        model_inputs = disable_geometry(model_inputs)
    return model(**model_inputs)


__all__ = ["forward_task"]
