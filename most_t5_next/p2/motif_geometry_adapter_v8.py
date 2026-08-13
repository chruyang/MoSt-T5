"""Minimal nonlinear atom-to-motif interface for anchored 3D-MotifT5.

The encoder keeps one shared E3FP table, treats L0 separately from the
available L1--L3 environments, and applies a conventional two-linear ``phi``
map before motif pooling.  Partition-derived role, explicit presence bits and
learned level embeddings are deliberately absent.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v1 import MotifGeometryAdapterError
from .motif_geometry_adapter_v3 import MotifGeometryAdapterV3


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v8-minimal-phi-l0-high"
ATOM_ENCODER_VARIANT = "shared_e3fp_embedding_l0_high_minimal_phi"


class MotifGeometryAdapterV8(MotifGeometryAdapterV3):
    """Use the smallest nonlinear L0/high atom representation tested here."""

    consumed_levels = (0, 1, 2, 3)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        del self.level_embedding
        del self.atom_role_embedding
        del self.atom_encoder
        self.atom_phi = nn.Sequential(
            nn.Linear(2 * self.state_embedding_dim, self.atom_memory_dim),
            nn.GELU(),
            nn.Linear(self.atom_memory_dim, self.atom_memory_dim),
            nn.GELU(),
        )

    def get_extra_state(self) -> dict[str, str]:
        return {"atom_encoder_variant": ATOM_ENCODER_VARIANT}

    def set_extra_state(self, state: object) -> None:
        if (
            not isinstance(state, dict)
            or state.get("atom_encoder_variant") != ATOM_ENCODER_VARIANT
        ):
            raise RuntimeError("checkpoint atom-encoder variant differs")

    def _encode_atom_memory(
        self,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None,
    ) -> Tensor:
        if e3fp_input_ids.shape != (*atom_mask.shape, 4):
            raise MotifGeometryAdapterError("e3fp_input_ids must be [B,A,4]")
        self._require_integer(e3fp_input_ids, "e3fp_input_ids")
        if atom_mask.dtype != torch.bool:
            raise MotifGeometryAdapterError("atom_mask must be bool [B,A]")
        if e3fp_input_ids.device != atom_mask.device:
            raise MotifGeometryAdapterError(
                "E3FP IDs and atom mask must share one device"
            )
        if atom_is_attachment is not None and (
            atom_is_attachment.shape != atom_mask.shape
            or atom_is_attachment.dtype != torch.bool
            or atom_is_attachment.device != atom_mask.device
        ):
            raise MotifGeometryAdapterError(
                "atom_is_attachment must be bool [B,A] on the adapter device"
            )
        bad_id = (e3fp_input_ids < -1) | (e3fp_input_ids > self.mask_token_id)
        if bool(bad_id.any()):
            raise MotifGeometryAdapterError(
                "E3FP input ID is outside the state domain"
            )
        if bool(((~atom_mask).unsqueeze(-1) & (e3fp_input_ids != -1)).any()):
            raise MotifGeometryAdapterError(
                "padded atom E3FP rows must contain only -1"
            )

        valid = (e3fp_input_ids >= 0) & atom_mask.unsqueeze(-1)
        normalized = e3fp_input_ids.masked_fill(
            e3fp_input_ids < 0, self.padding_token_id
        )
        hidden = self.state_embedding(normalized.to(torch.long))
        hidden = self._add_shell_context(hidden)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)
        l0_memory = hidden[..., 0, :]
        high_valid = valid[..., 1:4]
        high_sum = hidden[..., 1:4, :].sum(dim=2)
        high_count = high_valid.sum(dim=2, keepdim=True).clamp_min(1)
        high_memory = high_sum / high_count.to(high_sum.dtype)
        atom_memory = self.atom_phi(torch.cat((l0_memory, high_memory), dim=-1))
        return atom_memory * atom_mask.unsqueeze(-1).to(atom_memory.dtype)

    def _add_shell_context(self, hidden: Tensor) -> Tensor:
        """Hook for a paired level-aware candidate; V8 itself adds nothing."""
        return hidden


__all__ = ["ADAPTER_ID", "ATOM_ENCODER_VARIANT", "MotifGeometryAdapterV8"]
