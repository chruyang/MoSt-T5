"""Factorized GraphPorts grammar and motif-anchored E3FP state objectives.

This wrapper is intentionally separate from the historical PF-1 four-grid
wrapper.  A batch performs exactly one scientifically interpretable task:

* ``grammar`` reconstructs masked GraphPorts identity with ordinary T5 CE;
* ``state`` keeps identity visible and predicts masked categorical E3FP
  levels 1/2 after the fused input has passed through the T5 encoder;
* ``cross_view`` uses the grammar path with visible state, but is kept as an
  explicit later diagnostic rather than being mixed into ``grammar``.

There is no raw-ID regression, teacher loss, or implicit sum of unrelated
objectives.  The adapter writes geometry once at one carrier per motif.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .motif_geometry_adapter_v1 import (
    MotifGeometryAdapterV1,
    MotifGeometryEncoding,
)


FACTORISATION_ID = "most-t5-p2/factorized-motif-geometry-t5/v1"
OBJECTIVE_MODES = ("grammar", "state", "cross_view")


class FactorizedMotifT5Error(ValueError):
    """The model or one factorized training view violates its contract."""


@dataclass(frozen=True)
class FactorizedMotifT5Output:
    """Small Trainer-compatible output with explicit loss provenance."""

    loss: Tensor
    objective_mode: str
    grammar_loss: Tensor | None
    state_loss: Tensor | None
    state_level_losses: Mapping[int, Tensor]
    state_level_counts: Mapping[int, Tensor]
    state_logits: Tensor | None
    encoder_last_hidden_state: Tensor
    adapter_encoding: MotifGeometryEncoding
    t5_output: Any | None


class FactorizedMotifT5V1(nn.Module):
    """One T5 plus one motif-owned atom-memory adapter.

    ``state_level2_weight`` is the sole auxiliary-loss weight: level 1 is the
    main local-3D categorical target and level 2 is a weaker recursive-shell
    target.  Batches may select either level for mechanism diagnostics, while
    the formal state view normally supplies both.
    """

    def __init__(
        self,
        t5_model: nn.Module,
        *,
        num_e3fp_embeddings: int,
        state_level2_weight: float = 0.25,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 128,
        max_identity_span_length: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(t5_model, nn.Module):
            raise FactorizedMotifT5Error("t5_model must be a torch module")
        if (
            isinstance(state_level2_weight, bool)
            or not isinstance(state_level2_weight, (int, float))
            or float(state_level2_weight) < 0.0
        ):
            raise FactorizedMotifT5Error(
                "state_level2_weight must be a non-negative number"
            )
        embedding_getter = getattr(t5_model, "get_input_embeddings", None)
        if not callable(embedding_getter):
            raise FactorizedMotifT5Error(
                "t5_model must expose get_input_embeddings()"
            )
        input_embedding = embedding_getter()
        weight = getattr(input_embedding, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim != 2:
            raise FactorizedMotifT5Error(
                "T5 input embedding must expose a rank-2 weight"
            )
        config = getattr(t5_model, "config", None)
        if config is None:
            raise FactorizedMotifT5Error("t5_model must expose config")
        hidden_size = int(weight.shape[1])
        if int(getattr(config, "d_model", hidden_size)) != hidden_size:
            raise FactorizedMotifT5Error(
                "config.d_model differs from the input embedding width"
            )

        self.t5 = t5_model
        self.state_level2_weight = float(state_level2_weight)
        self.adapter = MotifGeometryAdapterV1(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
        )

    @property
    def config(self) -> Any:
        return self.t5.config

    def get_input_embeddings(self) -> nn.Module:
        return self.t5.get_input_embeddings()

    @staticmethod
    def _encoder_hidden(output: Any) -> Tensor:
        hidden = getattr(output, "encoder_last_hidden_state", None)
        if hidden is None and isinstance(output, Mapping):
            hidden = output.get("encoder_last_hidden_state")
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise FactorizedMotifT5Error(
                "T5 output must expose encoder_last_hidden_state [B,T,H]"
            )
        return hidden

    @staticmethod
    def _state_encoder_hidden(output: Any) -> Tensor:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, Mapping):
            hidden = output.get("last_hidden_state")
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise FactorizedMotifT5Error(
                "T5 encoder must expose last_hidden_state [B,T,H]"
            )
        return hidden

    def _validate_state_targets(
        self,
        *,
        e3fp_input_ids: Tensor,
        state_target_ids: Tensor | None,
        state_target_mask: Tensor | None,
        state_corruption_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        fields = (state_target_ids, state_target_mask, state_corruption_mask)
        if any(value is None for value in fields):
            raise FactorizedMotifT5Error(
                "state objective requires target IDs, target mask and corruption mask"
            )
        assert state_target_ids is not None
        assert state_target_mask is not None
        assert state_corruption_mask is not None
        if (
            state_target_ids.shape != e3fp_input_ids.shape
            or state_target_mask.shape != e3fp_input_ids.shape
            or state_corruption_mask.shape != e3fp_input_ids.shape
        ):
            raise FactorizedMotifT5Error(
                "state target/corruption tensors must match E3FP input shape"
            )
        if state_target_mask.dtype != torch.bool or state_corruption_mask.dtype != torch.bool:
            raise FactorizedMotifT5Error("state masks must use bool dtype")
        if bool((state_target_mask & ~state_corruption_mask).any()):
            raise FactorizedMotifT5Error(
                "every scored state target must be hidden from the encoder"
            )
        observed_mask_positions = e3fp_input_ids == self.adapter.mask_token_id
        if not torch.equal(observed_mask_positions, state_corruption_mask):
            raise FactorizedMotifT5Error(
                "state corruption mask must exactly name every state-mask token"
            )
        # All formal strategies are nested-shell safe.  For each selected
        # atom, corrupted populated slots must form one suffix of its original
        # row.  Atom-row and motif-block corruption are the special suffix
        # beginning at L0.  This prevents a manual/future batch from hiding L1
        # while leaking a populated recursive L2/L3 shell to the encoder.
        populated = state_target_ids >= 0
        for level in range(3):
            earlier_corrupted = state_corruption_mask[..., level]
            later_visible = (
                populated[..., level + 1]
                & ~state_corruption_mask[..., level + 1]
            )
            if bool((earlier_corrupted & later_visible).any()):
                raise FactorizedMotifT5Error(
                    "state corruption must be suffix-closed over populated shells"
                )
        permitted = torch.zeros(4, dtype=torch.bool, device=e3fp_input_ids.device)
        permitted[1:3] = True
        if bool((state_target_mask & ~permitted.view(1, 1, 4)).any()):
            raise FactorizedMotifT5Error(
                "the formal state head predicts only E3FP levels 1 and 2"
            )
        if not bool(state_target_mask.any()):
            raise FactorizedMotifT5Error("state target mask cannot be empty")
        if bool(
            (
                e3fp_input_ids[state_corruption_mask]
                != self.adapter.mask_token_id
            ).any()
        ):
            raise FactorizedMotifT5Error(
                "corrupted state slots must contain the declared mask token"
            )
        selected = state_target_ids[state_target_mask]
        if bool(
            ((selected < 0) | (selected >= self.adapter.num_e3fp_embeddings)).any()
        ):
            raise FactorizedMotifT5Error(
                "selected categorical state targets are outside the E3FP domain"
            )
        return state_target_ids, state_target_mask, state_corruption_mask

    def _state_loss(
        self,
        state_logits: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
    ) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor]]:
        if state_logits.shape[:2] != target_ids.shape[:2] or state_logits.shape[2] != 2:
            raise FactorizedMotifT5Error("state logits disagree with target atom domain")
        losses: dict[int, Tensor] = {}
        counts: dict[int, Tensor] = {}
        terms: list[Tensor] = []
        for external_level, logit_index in ((1, 0), (2, 1)):
            mask = target_mask[..., external_level]
            count = mask.sum()
            if int(count) == 0:
                continue
            level_loss = F.cross_entropy(
                state_logits[..., logit_index, :][mask],
                target_ids[..., external_level][mask],
                reduction="mean",
            )
            losses[external_level] = level_loss
            counts[external_level] = count
            weight = 1.0 if external_level == 1 else self.state_level2_weight
            terms.append(level_loss * weight)
        if not terms:
            raise FactorizedMotifT5Error("state batch selected no level-1/2 target")
        return torch.stack(terms).sum(), losses, counts

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
        atom_is_attachment: Tensor | None = None,
        labels: Tensor | None = None,
        state_target_ids: Tensor | None = None,
        state_target_mask: Tensor | None = None,
        state_corruption_mask: Tensor | None = None,
        **t5_kwargs: Any,
    ) -> FactorizedMotifT5Output:
        if objective_mode not in OBJECTIVE_MODES:
            raise FactorizedMotifT5Error(
                "objective_mode must be grammar, state or cross_view"
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
        if "inputs_embeds" in t5_kwargs:
            raise FactorizedMotifT5Error("external inputs_embeds are not permitted")
        if "encoder_outputs" in t5_kwargs or "return_dict" in t5_kwargs:
            raise FactorizedMotifT5Error(
                "encoder_outputs and return_dict are fixed by the factorized wrapper"
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
        )

        if objective_mode in ("grammar", "cross_view"):
            if labels is None:
                raise FactorizedMotifT5Error(
                    "grammar and cross_view objectives require labels"
                )
            if any(
                value is not None
                for value in (
                    state_target_ids,
                    state_target_mask,
                    state_corruption_mask,
                )
            ):
                raise FactorizedMotifT5Error(
                    "grammar/cross_view batches cannot silently add a state loss"
                )
            t5_output = self.t5(
                inputs_embeds=encoded.fused_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                **t5_kwargs,
            )
            grammar_loss = getattr(t5_output, "loss", None)
            if not isinstance(grammar_loss, Tensor) or grammar_loss.ndim != 0:
                raise FactorizedMotifT5Error("T5 grammar CE must be one scalar")
            encoder_hidden = self._encoder_hidden(t5_output)
            return FactorizedMotifT5Output(
                loss=grammar_loss,
                objective_mode=objective_mode,
                grammar_loss=grammar_loss,
                state_loss=None,
                state_level_losses={},
                state_level_counts={},
                state_logits=None,
                encoder_last_hidden_state=encoder_hidden,
                adapter_encoding=encoded,
                t5_output=t5_output,
            )

        if labels is not None:
            raise FactorizedMotifT5Error(
                "state batches keep identity visible and do not carry decoder labels"
            )
        if t5_kwargs:
            raise FactorizedMotifT5Error(
                "state objective does not accept decoder-only T5 keyword arguments"
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
    "FactorizedMotifT5V1",
]
