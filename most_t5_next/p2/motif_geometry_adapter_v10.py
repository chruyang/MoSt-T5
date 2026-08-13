"""Parameter-tying screen for the E3FP atom memory of 3D-MotifT5.

V10 deliberately changes only which shell positions share an embedding table.
The carrier/endpoint routing, fixed four-slot denominator, missing-shell zero,
and all downstream projections are inherited from V3 unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .e3fp_atom_embedding_v1 import (
    E3FPAtomEmbeddingV1,
    LEVEL_SPECIFIC_FIXED4,
    L0_STATE_FIXED4,
    REFERENCE_SHARED_FIXED4,
)
from .motif_geometry_adapter_v1 import MotifGeometryAdapterError
from .motif_geometry_adapter_v3 import MotifGeometryAdapterV3


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v10-e3fp-parameter-tying"
SUPPORTED_PARAMETER_TYING = (
    REFERENCE_SHARED_FIXED4,
    L0_STATE_FIXED4,
    LEVEL_SPECIFIC_FIXED4,
)


class MotifGeometryAdapterV10(MotifGeometryAdapterV3):
    """Expose one controlled E3FP parameter-tying factor to the V3 route."""

    consumed_levels = (0, 1, 2, 3)

    def __init__(self, *, parameter_tying: str, **kwargs: object) -> None:
        if parameter_tying not in SUPPORTED_PARAMETER_TYING:
            raise MotifGeometryAdapterError("unsupported E3FP parameter tying")
        super().__init__(**kwargs)

        # Remove the historical learned shell/role/MLP atom projector.  V10
        # owns exactly the controlled tables below; no dead branch is saved.
        del self.state_embedding
        del self.level_embedding
        del self.atom_role_embedding
        del self.atom_encoder

        self.parameter_tying = parameter_tying
        self.e3fp_atom_embedding = E3FPAtomEmbeddingV1(
            fp_bits=self.num_e3fp_embeddings,
            embedding_dim=self.atom_memory_dim,
            variant=parameter_tying,
        )

    def get_extra_state(self) -> dict[str, str]:
        return {
            "adapter_id": ADAPTER_ID,
            "e3fp_parameter_tying": self.parameter_tying,
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict) or state != self.get_extra_state():
            raise RuntimeError("checkpoint E3FP parameter-tying contract differs")

    def _encode_atom_memory(
        self,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None,
    ) -> Tensor:
        if atom_is_attachment is not None and (
            atom_is_attachment.shape != atom_mask.shape
            or atom_is_attachment.dtype != torch.bool
            or atom_is_attachment.device != atom_mask.device
        ):
            raise MotifGeometryAdapterError(
                "atom_is_attachment must be bool [B,A] on the adapter device"
            )
        try:
            return self.e3fp_atom_embedding(e3fp_input_ids, atom_mask)
        except ValueError as exc:
            raise MotifGeometryAdapterError(str(exc)) from exc


__all__ = [
    "ADAPTER_ID",
    "MotifGeometryAdapterV10",
    "SUPPORTED_PARAMETER_TYING",
]
