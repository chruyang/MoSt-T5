"""Minimal L0-versus-higher-shell refinement for anchored 3D-MotifT5.

The reference four-slot mean is retained as an exact nested baseline.  The
adaptive candidate adds only one global convex mixing scalar.  Its initial
value is one quarter, so with fixed zero-valued missing shells its initial
function equals the four-slot arithmetic mean:

    1/4 * L0 + 3/4 * mean(L1, L2, L3)
      == mean(L0, L1, L2, L3).

No level embedding, role embedding, presence feature or atom MLP is added.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v1 import MotifGeometryAdapterError
from .motif_geometry_adapter_v5 import MotifGeometryAdapterV5


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v6-global-l0-high-mix-anchored"
ATOM_ENCODER_VARIANT = "shared_e3fp_embedding_global_l0_high_convex_mix"
SHELL_REDUCER_MODES = ("fixed_four_mean", "adaptive_l0_high")
INITIAL_L0_WEIGHT = 0.25


class MotifGeometryAdapterV6(MotifGeometryAdapterV5):
    """Use a single global scalar to distinguish L0 from L1--L3."""

    def __init__(
        self,
        *,
        shell_reducer_mode: str = "adaptive_l0_high",
        **kwargs: object,
    ) -> None:
        if shell_reducer_mode not in SHELL_REDUCER_MODES:
            raise MotifGeometryAdapterError("shell reducer mode is invalid")
        super().__init__(**kwargs)
        self.shell_reducer_mode = shell_reducer_mode
        initial_logit = math.log(INITIAL_L0_WEIGHT / (1.0 - INITIAL_L0_WEIGHT))
        self.l0_mix_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def get_extra_state(self) -> dict[str, object]:
        return {
            "atom_encoder_variant": ATOM_ENCODER_VARIANT,
            "shell_reducer_mode": self.shell_reducer_mode,
            "initial_l0_weight": INITIAL_L0_WEIGHT,
        }

    def set_extra_state(self, state: object) -> None:
        if (
            not isinstance(state, dict)
            or state.get("atom_encoder_variant") != ATOM_ENCODER_VARIANT
            or state.get("shell_reducer_mode") != self.shell_reducer_mode
            or state.get("initial_l0_weight") != INITIAL_L0_WEIGHT
        ):
            raise RuntimeError("checkpoint shell-reducer contract differs")

    def l0_weight(self) -> Tensor:
        if self.shell_reducer_mode == "fixed_four_mean":
            return self.l0_mix_logit.new_tensor(INITIAL_L0_WEIGHT)
        return torch.sigmoid(self.l0_mix_logit)

    def _encode_atom_memory(
        self,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None,
    ) -> Tensor:
        if self.shell_reducer_mode == "fixed_four_mean":
            return super()._encode_atom_memory(
                e3fp_input_ids, atom_mask, atom_is_attachment
            )
        shell_memory = self._embed_fixed_shells(
            e3fp_input_ids, atom_mask, atom_is_attachment
        )
        l0_memory = shell_memory[..., 0, :]
        high_memory = shell_memory[..., 1:4, :].mean(dim=2)
        alpha = self.l0_weight().to(dtype=shell_memory.dtype)
        atom_memory = alpha * l0_memory + (1.0 - alpha) * high_memory
        return atom_memory * atom_mask.unsqueeze(-1).to(atom_memory.dtype)


__all__ = [
    "ADAPTER_ID",
    "ATOM_ENCODER_VARIANT",
    "INITIAL_L0_WEIGHT",
    "SHELL_REDUCER_MODES",
    "MotifGeometryAdapterV6",
]
