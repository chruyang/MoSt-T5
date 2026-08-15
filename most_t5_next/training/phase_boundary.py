"""Model-only handoff between the molecular and language-grounding phases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_phase_one_weights(
    path: str | Path, model: torch.nn.Module, *, metadata: dict[str, Any] | None = None
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
            "handoff": "phase-one-model-weights-only",
        },
        Path(path),
    )


def load_phase_one_weights(
    path: str | Path,
    model: torch.nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("handoff") != "phase-one-model-weights-only":
        raise ValueError("not a MoSt-T5 phase boundary checkpoint")
    model.load_state_dict(payload["model_state_dict"])
    return dict(payload.get("metadata", {}))


__all__ = ["load_phase_one_weights", "save_phase_one_weights"]
