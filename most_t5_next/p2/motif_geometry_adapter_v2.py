"""Carrier-mediated motif geometry adapter, version 2.

V1 established the production tensor contract, but its state decoder also
received the atom-memory tensor directly.  That shortcut allowed categorical
state reconstruction without proving that geometry had crossed the motif
carrier or entered T5.  V2 keeps the same batch and encoding contract while
making the causal route executable:

``visible E3FP -> owned motif aggregation -> gated carrier -> T5 -> state``.

The state decoder sees only the post-T5 owner carrier, a non-geometric
motif-local atom address, and the requested shell level.  Its ``atom_memory``
argument is retained solely so V1 and V2 wrappers can share the production
call signature; its values are validated but never read by the decoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v1 import (
    MotifGeometryAdapterError,
    MotifGeometryAdapterV1,
    MotifGeometryEncoding,
)


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v2-carrier-only"


class _NormalizedPerChannelGeometryGate(nn.Module):
    """Project one motif context and inject a bounded channel-wise residual."""

    def __init__(self, hidden_size: int, initial_gate: float) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.normalization = nn.LayerNorm(hidden_size, elementwise_affine=False)
        logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logits = nn.Parameter(torch.full((hidden_size,), logit))

    def forward(self, motif_context: Tensor) -> Tensor:
        normalized = self.normalization(self.projection(motif_context))
        return normalized * torch.sigmoid(self.gate_logits)

    def gate_values(self) -> Tensor:
        return torch.sigmoid(self.gate_logits)


class MotifGeometryAdapterV2(MotifGeometryAdapterV1):
    """V1-compatible encoder with a carrier-only categorical state decoder.

    ``initial_geometry_gate`` is deliberately small but non-zero.  Unlike the
    scalar ReZero parameter in V1, normalization prevents projection magnitude
    from silently determining whether geometry is visible, and one bounded
    gate per hidden channel cannot collapse the entire route through a single
    near-zero scalar.

    Motif-local atom positions come from the GraphPorts canonical source map.
    They identify which owned atom is the target without exposing its E3FP
    value or adding a second molecular graph.
    """

    def __init__(
        self,
        *,
        num_e3fp_embeddings: int,
        hidden_size: int,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 128,
        max_identity_span_length: int = 128,
        max_atoms_per_motif: int = 128,
        initial_geometry_gate: float = 0.1,
    ) -> None:
        if (
            isinstance(max_atoms_per_motif, bool)
            or not isinstance(max_atoms_per_motif, int)
            or max_atoms_per_motif <= 0
        ):
            raise MotifGeometryAdapterError(
                "max_atoms_per_motif must be a positive integer"
            )
        if (
            isinstance(initial_geometry_gate, bool)
            or not isinstance(initial_geometry_gate, (int, float))
            or not 0.0 < float(initial_geometry_gate) < 1.0
        ):
            raise MotifGeometryAdapterError(
                "initial_geometry_gate must lie strictly between zero and one"
            )
        super().__init__(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
        )
        self.max_atoms_per_motif = max_atoms_per_motif
        self.initial_geometry_gate = float(initial_geometry_gate)

        # Reuse V1's proven owned-attention implementation, but replace its
        # output projection by a normalized channel gate and neutralize the
        # legacy scalar multiplier.  The buffer is intentionally nonpersistent:
        # V2 checkpoints contain no trainable scalar bypass/collapse control.
        self.geometry_output = _NormalizedPerChannelGeometryGate(
            hidden_size,
            self.initial_geometry_gate,
        )
        del self.geometry_residual_scale
        self.register_buffer(
            "geometry_residual_scale",
            torch.ones(()),
            persistent=False,
        )

        self.target_atom_position_embedding = nn.Embedding(
            max_atoms_per_motif,
            state_embedding_dim,
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_size + 2 * state_embedding_dim, atom_memory_dim),
            nn.GELU(),
            nn.Linear(atom_memory_dim, num_e3fp_embeddings),
        )

    def geometry_gate_values(self) -> Tensor:
        """Return the live per-channel injection strengths for diagnostics."""

        assert isinstance(self.geometry_output, _NormalizedPerChannelGeometryGate)
        return self.geometry_output.gate_values()

    def _validate_atom_local_positions(
        self,
        atom_local_positions: Tensor,
        atom_mask: Tensor,
        atom_to_motif: Tensor,
        motif_width: int,
    ) -> Tensor:
        """Validate canonical-local IDs on the padded model-atom axis."""

        if atom_local_positions.shape != atom_mask.shape:
            raise MotifGeometryAdapterError(
                "atom_local_positions must be [B,A]"
            )
        self._require_integer(atom_local_positions, "atom_local_positions")
        if atom_local_positions.device != atom_mask.device:
            raise MotifGeometryAdapterError(
                "atom_local_positions must share the adapter device"
            )
        if bool(((~atom_mask) & (atom_local_positions != -1)).any()):
            raise MotifGeometryAdapterError(
                "padded atom local position must be -1"
            )
        if bool(
            (
                atom_mask
                & (
                    (atom_local_positions < 0)
                    | (atom_local_positions >= self.max_atoms_per_motif)
                )
            ).any()
        ):
            raise MotifGeometryAdapterError(
                "motif-local atom position exceeds max_atoms_per_motif"
            )
        for row in range(atom_mask.shape[0]):
            for motif_id in range(motif_width):
                owned = atom_mask[row] & (atom_to_motif[row] == motif_id)
                if not bool(owned.any()):
                    continue
                actual = torch.sort(atom_local_positions[row][owned]).values
                expected = torch.arange(
                    actual.numel(),
                    device=actual.device,
                    dtype=actual.dtype,
                )
                if not torch.equal(actual, expected):
                    raise MotifGeometryAdapterError(
                        "canonical-local atom positions must be contiguous per motif"
                    )
        return atom_local_positions.to(torch.long)

    def decode_state(
        self,
        atom_memory: Tensor,
        encoder_hidden: Tensor,
        *,
        attention_mask: Tensor,
        atom_mask: Tensor,
        atom_to_motif: Tensor,
        atom_local_positions: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
    ) -> Tensor:
        """Predict L1/L2 strictly through post-T5 motif carrier states.

        ``atom_memory`` participates in shape/device validation only.  Holding
        ``encoder_hidden`` fixed while changing that tensor must leave logits
        bitwise unchanged; tests enforce this no-bypass invariant.
        """

        batch_size, _token_width, atom_width, motif_width = self._validate_common(
            token_hidden=encoder_hidden,
            attention_mask=attention_mask,
            atom_mask=atom_mask,
            atom_to_motif=atom_to_motif,
            motif_mask=motif_mask,
            motif_to_carrier=motif_to_carrier,
        )
        if atom_memory.shape != (batch_size, atom_width, self.atom_memory_dim):
            raise MotifGeometryAdapterError("atom_memory must be [B,A,D]")
        if atom_memory.device != encoder_hidden.device or not atom_memory.is_floating_point():
            raise MotifGeometryAdapterError(
                "atom_memory must share the encoder device and be floating point"
            )

        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        motif_hidden = encoder_hidden.gather(
            1,
            safe_carriers.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        motif_hidden = motif_hidden * motif_mask.unsqueeze(-1).to(motif_hidden.dtype)
        safe_atom_groups = atom_to_motif.clamp_min(0).to(torch.long)
        owner_carrier = motif_hidden.gather(
            1,
            safe_atom_groups.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        owner_carrier = owner_carrier * atom_mask.unsqueeze(-1).to(owner_carrier.dtype)

        local_positions = self._validate_atom_local_positions(
            atom_local_positions,
            atom_mask,
            atom_to_motif,
            motif_width,
        )
        address = self.target_atom_position_embedding(local_positions.clamp_min(0))
        address = address * atom_mask.unsqueeze(-1).to(address.dtype)
        level_hidden = self.state_level_embedding(
            torch.arange(2, device=encoder_hidden.device)
        ).view(1, 1, 2, self.state_embedding_dim)
        decoder_input = torch.cat(
            (
                owner_carrier.unsqueeze(2).expand(-1, -1, 2, -1),
                address.unsqueeze(2).expand(-1, -1, 2, -1),
                level_hidden.expand(batch_size, atom_width, -1, -1),
            ),
            dim=-1,
        )
        logits = self.state_decoder(decoder_input)
        return logits * atom_mask.unsqueeze(-1).unsqueeze(-1).to(logits.dtype)


__all__ = [
    "ADAPTER_ID",
    "MotifGeometryAdapterError",
    "MotifGeometryAdapterV2",
    "MotifGeometryEncoding",
]
