"""Task definitions and update-level sampling for MoSt-T5 pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from most_t5_next.interfaces import PHASE_TASKS


DataSource = Literal["molecular_union", "pcqm", "pubchem_paired", "pubmed"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    phase: int
    source: DataSource


TASKS = {
    "M": TaskSpec("M", 1, "molecular_union"),
    "MG": TaskSpec("MG", 1, "pcqm"),
    "SYN": TaskSpec("SYN", 2, "molecular_union"),
    "TXT": TaskSpec("TXT", 2, "pubmed"),
    "CAP": TaskSpec("CAP", 2, "pubchem_paired"),
    "T2M": TaskSpec("T2M", 2, "pubchem_paired"),
}


class CurriculumSchedule:
    """Repeat one balanced task cycle at the optimizer-update boundary."""

    def __init__(
        self,
        phase: int,
        total_updates: int,
        *,
        require_complete_cycles: bool = True,
    ) -> None:
        if phase not in PHASE_TASKS:
            raise ValueError("phase must be 1 or 2")
        if (
            isinstance(total_updates, bool)
            or not isinstance(total_updates, int)
            or total_updates <= 0
        ):
            raise ValueError("total_updates must be a positive integer")
        self.phase = phase
        self.total_updates = int(total_updates)
        self.tasks = PHASE_TASKS[phase]
        if require_complete_cycles and self.total_updates % len(self.tasks):
            raise ValueError(
                "total_updates must contain an integer number of balanced task cycles"
            )

    def __len__(self) -> int:
        return self.total_updates

    def task_at(self, update: int) -> TaskSpec:
        """Return the task for a zero-based optimizer update."""

        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or not 0 <= update < self.total_updates
        ):
            raise IndexError(update)
        return TASKS[self.tasks[update % len(self.tasks)]]

    def updates_per_task(self) -> dict[str, int]:
        count = self.total_updates // len(self.tasks)
        return {name: count for name in self.tasks}


__all__ = ["CurriculumSchedule", "PHASE_TASKS", "TASKS", "TaskSpec"]
