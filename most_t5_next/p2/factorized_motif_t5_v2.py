"""Carrier-only V2 of the factorized motif geometry T5 wrapper.

The production forward/loss contract remains V1-compatible.  Only the
geometry mechanism changes: normalized channel-wise carrier injection and a
state decoder that cannot read atom memory directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch import nn

from .factorized_motif_t5_v1 import (
    OBJECTIVE_MODES,
    FactorizedMotifT5Error,
    FactorizedMotifT5Output,
    FactorizedMotifT5V1,
)
from .motif_geometry_adapter_v2 import MotifGeometryAdapterV2


FACTORISATION_ID = "most-t5-p2/factorized-motif-geometry-t5/v2-carrier-only"


class FactorizedMotifT5V2(FactorizedMotifT5V1):
    """Drop-in production wrapper for the V2 carrier-mediated mechanism."""

    def __init__(
        self,
        t5_model: nn.Module,
        *,
        num_e3fp_embeddings: int,
        state_level2_weight: float = 0.25,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 128,
        max_identity_span_length: int = 128,
        max_atoms_per_motif: int = 128,
        initial_geometry_gate: float = 0.1,
    ) -> None:
        super().__init__(
            t5_model,
            num_e3fp_embeddings=num_e3fp_embeddings,
            state_level2_weight=state_level2_weight,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
        )
        hidden_size = int(self.get_input_embeddings().weight.shape[1])
        self.adapter = MotifGeometryAdapterV2(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
            max_atoms_per_motif=max_atoms_per_motif,
            initial_geometry_gate=initial_geometry_gate,
        )

    @property
    def factorisation_id(self) -> str:
        return FACTORISATION_ID

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        objective_mode: str,
        e3fp_mask_token_id: int,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_to_motif: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
        identity_span_bounds: Tensor,
        atom_local_positions: Tensor | None = None,
        atom_is_attachment: Tensor | None = None,
        labels: Tensor | None = None,
        state_target_ids: Tensor | None = None,
        state_target_mask: Tensor | None = None,
        state_corruption_mask: Tensor | None = None,
        state_memory_mode: str = "aligned",
        **t5_kwargs: Any,
    ) -> FactorizedMotifT5Output:
        # Grammar uses the exact inherited T5 path.  Canonical atom addresses
        # are needed only when producing atom-resolution state targets.
        if objective_mode != "state":
            return super().forward(
                input_ids,
                attention_mask,
                objective_mode=objective_mode,
                e3fp_mask_token_id=e3fp_mask_token_id,
                e3fp_input_ids=e3fp_input_ids,
                atom_mask=atom_mask,
                atom_to_motif=atom_to_motif,
                motif_mask=motif_mask,
                motif_to_carrier=motif_to_carrier,
                identity_span_bounds=identity_span_bounds,
                atom_is_attachment=atom_is_attachment,
                labels=labels,
                state_target_ids=state_target_ids,
                state_target_mask=state_target_mask,
                state_corruption_mask=state_corruption_mask,
                state_memory_mode=state_memory_mode,
                **t5_kwargs,
            )

        if atom_local_positions is None:
            raise FactorizedMotifT5Error(
                "V2 state objective requires canonical atom_local_positions"
            )
        if (
            isinstance(e3fp_mask_token_id, bool)
            or not isinstance(e3fp_mask_token_id, int)
            or e3fp_mask_token_id != self.adapter.mask_token_id
        ):
            raise FactorizedMotifT5Error(
                "batch E3FP mask token differs from the model state domain"
            )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise FactorizedMotifT5Error(
                "input_ids and attention_mask must share shape [B,T]"
            )
        if labels is not None:
            raise FactorizedMotifT5Error(
                "state batches keep identity visible and do not carry decoder labels"
            )
        if state_memory_mode not in {"aligned", "zero"}:
            raise FactorizedMotifT5Error(
                "V2 state objective memory mode must be aligned or zero"
            )
        if t5_kwargs:
            raise FactorizedMotifT5Error(
                "state objective does not accept decoder-only T5 keyword arguments"
            )

        bool_attention = attention_mask.to(torch.bool)
        input_embeddings = self.t5.get_input_embeddings()(input_ids)
        encoded = self.adapter.encode(
            input_embeddings,
            attention_mask=bool_attention,
            e3fp_input_ids=e3fp_input_ids,
            atom_mask=atom_mask,
            atom_to_motif=atom_to_motif,
            motif_mask=motif_mask,
            motif_to_carrier=motif_to_carrier,
            identity_span_bounds=identity_span_bounds,
            atom_is_attachment=atom_is_attachment,
            state_memory_mode=state_memory_mode,
        )
        target_ids, target_mask, _ = self._validate_state_targets(
            e3fp_input_ids=e3fp_input_ids,
            state_target_ids=state_target_ids,
            state_target_mask=state_target_mask,
            state_corruption_mask=state_corruption_mask,
        )
        encoder = getattr(self.t5, "encoder", None)
        if not callable(encoder):
            raise FactorizedMotifT5Error(
                "state objective requires the T5 encoder module"
            )
        encoder_output = encoder(
            inputs_embeds=encoded.fused_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden = self._state_encoder_hidden(encoder_output)
        state_logits = self.adapter.decode_state(
            encoded.atom_memory,
            encoder_hidden,
            attention_mask=bool_attention,
            atom_mask=atom_mask,
            atom_to_motif=atom_to_motif,
            atom_local_positions=atom_local_positions,
            motif_mask=motif_mask,
            motif_to_carrier=motif_to_carrier,
        )
        state_loss, level_losses, level_counts = self._state_loss(
            state_logits,
            target_ids,
            target_mask,
        )
        return FactorizedMotifT5Output(
            loss=state_loss,
            objective_mode=objective_mode,
            grammar_loss=None,
            state_loss=state_loss,
            state_level_losses=level_losses,
            state_level_counts=level_counts,
            state_logits=state_logits,
            encoder_last_hidden_state=encoder_hidden,
            adapter_encoding=encoded,
            t5_output=None,
        )


__all__ = [
    "FACTORISATION_ID",
    "OBJECTIVE_MODES",
    "FactorizedMotifT5Error",
    "FactorizedMotifT5Output",
    "FactorizedMotifT5V2",
]
