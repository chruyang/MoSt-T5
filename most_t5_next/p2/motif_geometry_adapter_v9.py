"""Level-aware minimal atom phi for anchored 3D-MotifT5."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v8 import MotifGeometryAdapterV8


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v9-level-aware-minimal-phi"
ATOM_ENCODER_VARIANT = "shared_e3fp_plus_level_embedding_l0_high_minimal_phi"


class MotifGeometryAdapterV9(MotifGeometryAdapterV8):
    """Restore only ordered shell-level identity; role/presence stay absent."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.level_embedding = nn.Embedding(4, self.state_embedding_dim)

    def get_extra_state(self) -> dict[str, str]:
        return {"atom_encoder_variant": ATOM_ENCODER_VARIANT}

    def set_extra_state(self, state: object) -> None:
        if (
            not isinstance(state, dict)
            or state.get("atom_encoder_variant") != ATOM_ENCODER_VARIANT
        ):
            raise RuntimeError("checkpoint atom-encoder variant differs")

    def _add_shell_context(self, hidden: Tensor) -> Tensor:
        level_ids = torch.arange(4, device=hidden.device, dtype=torch.long)
        return hidden + self.level_embedding(level_ids).view(
            1, 1, 4, self.state_embedding_dim
        )


__all__ = ["ADAPTER_ID", "ATOM_ENCODER_VARIANT", "MotifGeometryAdapterV9"]
