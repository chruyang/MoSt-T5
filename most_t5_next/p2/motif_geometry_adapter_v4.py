"""Level-explicit atom E3FP fusion for the anchored 3D-motif interface."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v1 import MotifGeometryAdapterError
from .motif_geometry_adapter_v3 import MotifGeometryAdapterV3


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v4-level-explicit-anchored"
SHELL_FUSION_MODES = (
    "l12_mean",
    "l0_l12_mean",
    "l0_l123_mean",
    "l0123_mean",
    "l0_shell_attention_l123",
)


class MotifGeometryAdapterV4(MotifGeometryAdapterV3):
    """Keep one parameter topology while varying which E3FP shells are visible.

    Every mode feeds two fixed-width slots into the inherited atom encoder:
    an identity slot and a shell-summary slot.  This makes the ablation a data
    path decision instead of a hidden parameter-count change.

    ``l0_l12_mean``, ``l0_l123_mean`` and ``l0_shell_attention_l123`` expose L0 explicitly.
    L0 is interpreted only as atom-level 2D identity/context; it is never used
    as evidence that the model consumed 3D.  Higher-shell causal claims still
    require B2D and aligned/zero/matched-shuffle controls.
    """

    consumed_levels = (0, 1, 2, 3)

    def __init__(self, *, shell_fusion_mode: str = "l0_shell_attention_l123", **kwargs: object) -> None:
        if shell_fusion_mode not in SHELL_FUSION_MODES:
            raise MotifGeometryAdapterError(
                "shell_fusion_mode is outside the frozen candidate set"
            )
        super().__init__(**kwargs)
        self.shell_fusion_mode = shell_fusion_mode
        # Present in every mode so parameter topology and initialization remain
        # paired.  Non-attention modes simply do not execute this score.
        self.shell_attention_score = nn.Linear(
            self.state_embedding_dim, 1, bias=False
        )

    def get_extra_state(self) -> dict[str, str]:
        return {"shell_fusion_mode": self.shell_fusion_mode}

    def set_extra_state(self, state: object) -> None:
        if (
            not isinstance(state, dict)
            or state.get("shell_fusion_mode") != self.shell_fusion_mode
        ):
            raise RuntimeError("checkpoint shell-fusion mode differs")

    @staticmethod
    def _masked_mean(hidden: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        count = valid.sum(dim=-1, keepdim=True)
        pooled = (
            (hidden * valid.unsqueeze(-1).to(hidden.dtype)).sum(dim=-2)
            / count.clamp_min(1).to(hidden.dtype)
        )
        present = count.squeeze(-1) > 0
        return pooled * present.unsqueeze(-1).to(hidden.dtype), present

    def _encode_atom_memory(
        self,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None,
    ) -> Tensor:
        if e3fp_input_ids.shape != (*atom_mask.shape, 4):
            raise MotifGeometryAdapterError("e3fp_input_ids must be [B,A,4]")
        self._require_integer(e3fp_input_ids, "e3fp_input_ids")
        if e3fp_input_ids.device != atom_mask.device:
            raise MotifGeometryAdapterError(
                "E3FP IDs and atom mask must share one device"
            )
        if atom_is_attachment is None:
            atom_is_attachment = torch.zeros_like(atom_mask)
        if (
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
        level_ids = torch.arange(
            4, device=e3fp_input_ids.device, dtype=torch.long
        ).view(1, 1, 4)
        hidden = self.state_embedding(normalized.to(torch.long))
        hidden = hidden + self.level_embedding(level_ids)
        hidden = hidden * valid.unsqueeze(-1).to(hidden.dtype)

        zero_slot = hidden.new_zeros((*atom_mask.shape, self.state_embedding_dim))
        zero_present = torch.zeros_like(atom_mask)
        if self.shell_fusion_mode == "l12_mean":
            identity_hidden, identity_present = zero_slot, zero_present
            shell_hidden, shell_present = self._masked_mean(
                hidden[..., 1:3, :], valid[..., 1:3]
            )
        elif self.shell_fusion_mode == "l0_l12_mean":
            identity_hidden = hidden[..., 0, :]
            identity_present = valid[..., 0]
            shell_hidden, shell_present = self._masked_mean(
                hidden[..., 1:3, :], valid[..., 1:3]
            )
        elif self.shell_fusion_mode == "l0_l123_mean":
            identity_hidden = hidden[..., 0, :]
            identity_present = valid[..., 0]
            shell_hidden, shell_present = self._masked_mean(
                hidden[..., 1:4, :], valid[..., 1:4]
            )
        elif self.shell_fusion_mode == "l0123_mean":
            identity_hidden, identity_present = zero_slot, zero_present
            shell_hidden, shell_present = self._masked_mean(hidden, valid)
        else:
            identity_hidden = hidden[..., 0, :]
            identity_present = valid[..., 0]
            shell_values = hidden[..., 1:4, :]
            shell_valid = valid[..., 1:4]
            scores = self.shell_attention_score(shell_values).squeeze(-1)
            scores = scores.masked_fill(
                ~shell_valid, torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
            weights = weights * shell_valid.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            shell_hidden = (weights.unsqueeze(-1) * shell_values).sum(dim=-2)
            shell_present = shell_valid.any(dim=-1)

        role_hidden = self.atom_role_embedding(atom_is_attachment.to(torch.long))
        role_hidden = role_hidden * atom_mask.unsqueeze(-1).to(role_hidden.dtype)
        presence = torch.stack((identity_present, shell_present), dim=-1).to(
            hidden.dtype
        )
        atom_input = torch.cat(
            (identity_hidden, shell_hidden, presence, role_hidden), dim=-1
        )
        atom_memory = self.atom_encoder(atom_input)
        return atom_memory * atom_mask.unsqueeze(-1).to(atom_memory.dtype)


__all__ = ["ADAPTER_ID", "SHELL_FUSION_MODES", "MotifGeometryAdapterV4"]
