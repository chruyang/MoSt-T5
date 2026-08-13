"""Reference-aligned atom E3FP encoding for anchored 3D-MotifT5.

The mature V3 carrier/endpoint route is retained unchanged.  Only the atom
state encoder is simplified to the exact structural prior used by 3D-MolT5:
one shared E3FP table, four fixed shell slots, a zero contribution for a
missing slot, and an arithmetic mean whose denominator remains four.

``atom_is_attachment`` remains part of the public interface because V3 uses it
to validate GraphPorts endpoint addresses.  It is intentionally *not* an atom
feature here: attachment is partition-derived routing metadata, not intrinsic
chemical or three-dimensional state.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v1 import MotifGeometryAdapterError
from .motif_geometry_adapter_v3 import MotifGeometryAdapterV3


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v5-reference-fixed4-anchored"
ATOM_ENCODER_VARIANT = "shared_e3fp_embedding_fixed_four_slot_mean"


class MotifGeometryAdapterV5(MotifGeometryAdapterV3):
    """Use a parameter-minimal, reference-aligned E3FP atom representation."""

    consumed_levels = (0, 1, 2, 3)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        # These V1 modules belong to the historical learned atom projector.
        # V5 removes them rather than leaving dead parameters in checkpoints.
        del self.state_embedding
        del self.level_embedding
        del self.atom_role_embedding
        del self.atom_encoder

        self.shared_e3fp_embedding = nn.Embedding(
            self.num_e3fp_embeddings + 2,
            self.atom_memory_dim,
            padding_idx=self.padding_token_id,
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
        shell_memory = self._embed_fixed_shells(
            e3fp_input_ids, atom_mask, atom_is_attachment
        )
        atom_memory = shell_memory.mean(dim=2)
        return atom_memory * atom_mask.unsqueeze(-1).to(atom_memory.dtype)

    def _embed_fixed_shells(
        self,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None,
    ) -> Tensor:
        """Validate and embed the fixed L0--L3 slots with missing-shell zero."""
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

        present = (e3fp_input_ids >= 0) & atom_mask.unsqueeze(-1)
        normalized = e3fp_input_ids.masked_fill(
            e3fp_input_ids < 0, self.padding_token_id
        )
        shell_memory = self.shared_e3fp_embedding(normalized.to(torch.long))
        # Enforce missing-shell zero semantics explicitly.  This makes the
        # scientific contract independent of optimizer/checkpoint handling of
        # the embedding padding row.
        return shell_memory * present.unsqueeze(-1).to(shell_memory.dtype)


__all__ = ["ADAPTER_ID", "ATOM_ENCODER_VARIANT", "MotifGeometryAdapterV5"]
