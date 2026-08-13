"""One-linear L0/high-shell atom reducer for anchored 3D-MotifT5.

This is the smallest learned refinement after the reference fixed-four mean.
It keeps one shared E3FP table and no role, presence or level embeddings.  L0
and the fixed-denominator mean of L1--L3 are concatenated and passed through
one bias-free linear map.  The map is initialized to ``[0.25 I, 0.75 I]``, so
the initial function is exactly the reference four-slot mean.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v5 import MotifGeometryAdapterV5


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v7-linear-l0-high-anchored"
ATOM_ENCODER_VARIANT = "shared_e3fp_embedding_single_linear_l0_high"


class MotifGeometryAdapterV7(MotifGeometryAdapterV5):
    """Learn one dimension-wise L0/high transformation from the reference point."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        width = self.atom_memory_dim
        self.l0_high_projection = nn.Linear(2 * width, width, bias=False)
        with torch.no_grad():
            self.l0_high_projection.weight.zero_()
            identity = torch.eye(width, dtype=self.l0_high_projection.weight.dtype)
            self.l0_high_projection.weight[:, :width].copy_(0.25 * identity)
            self.l0_high_projection.weight[:, width:].copy_(0.75 * identity)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "atom_encoder_variant": ATOM_ENCODER_VARIANT,
            "initial_projection": "concat_l0_high_block_0.25I_0.75I",
        }

    def set_extra_state(self, state: object) -> None:
        if (
            not isinstance(state, dict)
            or state.get("atom_encoder_variant") != ATOM_ENCODER_VARIANT
            or state.get("initial_projection")
            != "concat_l0_high_block_0.25I_0.75I"
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
        l0_memory = shell_memory[..., 0, :]
        high_memory = shell_memory[..., 1:4, :].mean(dim=2)
        atom_memory = self.l0_high_projection(
            torch.cat((l0_memory, high_memory), dim=-1)
        )
        return atom_memory * atom_mask.unsqueeze(-1).to(atom_memory.dtype)


__all__ = ["ADAPTER_ID", "ATOM_ENCODER_VARIANT", "MotifGeometryAdapterV7"]
