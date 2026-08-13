"""Deterministic fourth-root task selection for PubChem generalist training."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence


SAMPLER_ID = "most-t5-next/fourth-root-task-sampler/v1"


class FourthRootTaskSamplerError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSamplingRow:
    task_id: str
    train_population: int
    raw_weight: float
    probability: float


@dataclass(frozen=True)
class TaskSelection:
    draw_index: int
    task_id: str
    task_cursor: int
    task_pass_index: int
    examples: int


def fourth_root_sampling_rows(
    train_populations: Mapping[str, int],
) -> tuple[TaskSamplingRow, ...]:
    if not train_populations:
        raise FourthRootTaskSamplerError("at least one task is required")
    ordered = []
    for task_id in sorted(train_populations):
        population = train_populations[task_id]
        if (
            not isinstance(task_id, str)
            or not task_id
            or isinstance(population, bool)
            or not isinstance(population, int)
            or population <= 0
        ):
            raise FourthRootTaskSamplerError("task populations must be positive integers")
        ordered.append((task_id, population, population ** 0.25))
    total = math.fsum(weight for _task, _population, weight in ordered)
    return tuple(
        TaskSamplingRow(
            task_id=task_id,
            train_population=population,
            raw_weight=weight,
            probability=weight / total,
        )
        for task_id, population, weight in ordered
    )


def _tuple_tree(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class FourthRootTaskSamplerV1:
    """Draw one task per fixed-size microbatch and track cyclic task cursors.

    This class does not shuffle records inside a task.  A task reader consumes
    ``task_cursor`` from its own frozen epoch order.  When a task reaches its
    admitted population the cursor returns to zero and ``task_pass_index`` is
    incremented.  Equal microbatch sizes therefore realize the task draw
    probabilities as example probabilities without padding or dropping data.
    """

    def __init__(
        self,
        train_populations: Mapping[str, int],
        *,
        seed: int,
        examples_per_microbatch: int,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise FourthRootTaskSamplerError("seed must be a nonnegative integer")
        if (
            isinstance(examples_per_microbatch, bool)
            or not isinstance(examples_per_microbatch, int)
            or examples_per_microbatch <= 0
        ):
            raise FourthRootTaskSamplerError(
                "examples_per_microbatch must be a positive integer"
            )
        self.rows = fourth_root_sampling_rows(train_populations)
        self.seed = seed
        self.examples_per_microbatch = examples_per_microbatch
        self._rng = random.Random(seed)
        self._draw_count = 0
        self._selection_counts = {row.task_id: 0 for row in self.rows}
        self._example_counts = {row.task_id: 0 for row in self.rows}
        self._cursors = {row.task_id: 0 for row in self.rows}
        self._passes = {row.task_id: 0 for row in self.rows}

    def draw(self) -> TaskSelection:
        threshold = self._rng.random()
        cumulative = 0.0
        selected = self.rows[-1]
        for row in self.rows:
            cumulative += row.probability
            if threshold < cumulative:
                selected = row
                break
        task_id = selected.task_id
        cursor = self._cursors[task_id]
        pass_index = self._passes[task_id]
        result = TaskSelection(
            draw_index=self._draw_count,
            task_id=task_id,
            task_cursor=cursor,
            task_pass_index=pass_index,
            examples=self.examples_per_microbatch,
        )
        self._draw_count += 1
        self._selection_counts[task_id] += 1
        self._example_counts[task_id] += self.examples_per_microbatch
        next_cursor = cursor + self.examples_per_microbatch
        completed, remainder = divmod(next_cursor, selected.train_population)
        self._passes[task_id] += completed
        self._cursors[task_id] = remainder
        return result

    def draw_many(self, count: int) -> tuple[TaskSelection, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise FourthRootTaskSamplerError("draw count must be nonnegative")
        return tuple(self.draw() for _ in range(count))

    def state_dict(self) -> dict[str, object]:
        return {
            "sampler_id": SAMPLER_ID,
            "seed": self.seed,
            "examples_per_microbatch": self.examples_per_microbatch,
            "rows": [
                {
                    "task_id": row.task_id,
                    "train_population": row.train_population,
                    "raw_weight": row.raw_weight,
                    "probability": row.probability,
                }
                for row in self.rows
            ],
            "draw_count": self._draw_count,
            "selection_counts": dict(self._selection_counts),
            "example_counts": dict(self._example_counts),
            "task_cursors": dict(self._cursors),
            "completed_passes": dict(self._passes),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("sampler_id") != SAMPLER_ID:
            raise FourthRootTaskSamplerError("sampler state has the wrong schema")
        if state.get("seed") != self.seed or state.get(
            "examples_per_microbatch"
        ) != self.examples_per_microbatch:
            raise FourthRootTaskSamplerError("sampler execution contract changed")
        expected_rows = self.state_dict()["rows"]
        if state.get("rows") != expected_rows:
            raise FourthRootTaskSamplerError("task populations or probabilities changed")
        task_ids = {row.task_id for row in self.rows}

        def require_counts(name: str) -> dict[str, int]:
            raw = state.get(name)
            if not isinstance(raw, Mapping) or set(raw) != task_ids:
                raise FourthRootTaskSamplerError(f"{name} has the wrong task domain")
            result = {}
            for task_id, value in raw.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise FourthRootTaskSamplerError(f"{name} contains an invalid count")
                result[str(task_id)] = value
            return result

        draw_count = state.get("draw_count")
        if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 0:
            raise FourthRootTaskSamplerError("draw_count is invalid")
        selections = require_counts("selection_counts")
        examples = require_counts("example_counts")
        cursors = require_counts("task_cursors")
        passes = require_counts("completed_passes")
        if sum(selections.values()) != draw_count:
            raise FourthRootTaskSamplerError("selection counts do not close draw_count")
        populations = {row.task_id: row.train_population for row in self.rows}
        for task_id in task_ids:
            if examples[task_id] != selections[task_id] * self.examples_per_microbatch:
                raise FourthRootTaskSamplerError("example counts disagree with selections")
            completed, remainder = divmod(examples[task_id], populations[task_id])
            if passes[task_id] != completed or cursors[task_id] != remainder:
                raise FourthRootTaskSamplerError("task cursor/pass state is inconsistent")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, (tuple, list)):
            raise FourthRootTaskSamplerError("RNG state is missing")
        try:
            self._rng.setstate(_tuple_tree(rng_state))  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise FourthRootTaskSamplerError("RNG state is invalid") from exc
        self._draw_count = draw_count
        self._selection_counts = selections
        self._example_counts = examples
        self._cursors = cursors
        self._passes = passes

    def report(self) -> dict[str, object]:
        state = self.state_dict()
        state.pop("rng_state")
        state["selection_unit"] = "fixed_size_microbatch"
        state["probability_interpretation"] = (
            "equal microbatch size makes task-selection probabilities equal expected example proportions"
        )
        return state


__all__ = [
    "FourthRootTaskSamplerError",
    "FourthRootTaskSamplerV1",
    "SAMPLER_ID",
    "TaskSamplingRow",
    "TaskSelection",
    "fourth_root_sampling_rows",
]
