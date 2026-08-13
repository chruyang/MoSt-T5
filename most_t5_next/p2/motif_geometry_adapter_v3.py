"""Addressable 3D-motif input adapter for a stock T5 backbone.

V3 keeps the V2 atom-memory/state-decoder contract but removes the single
carrier bottleneck.  A motif-owned attention summary is mixed into its
identity carrier, while every GraphPorts endpoint marker is mixed with the
memory of its exact attachment atom.  No molecular text token is added.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .motif_geometry_adapter_v1 import (
    MotifGeometryAdapterError,
    MotifGeometryEncoding,
)
from .motif_geometry_adapter_v2 import MotifGeometryAdapterV2


ADAPTER_ID = "most-t5-p2/motif-geometry-adapter/v3-carrier-endpoint-fixed-mix"


class MotifGeometryAdapterV3(MotifGeometryAdapterV2):
    """Mix pure atom-derived geometry at motif carriers and port endpoints."""

    def __init__(
        self,
        *,
        num_e3fp_embeddings: int,
        hidden_size: int,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 128,
        max_identity_span_length: int = 128,
        max_atoms_per_motif: int = 128,
        geometry_fraction: float = 0.5,
    ) -> None:
        if (
            isinstance(geometry_fraction, bool)
            or not isinstance(geometry_fraction, (int, float))
            or not 0.0 < float(geometry_fraction) < 1.0
        ):
            raise MotifGeometryAdapterError(
                "geometry_fraction must lie strictly between zero and one"
            )
        super().__init__(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
            max_atoms_per_motif=max_atoms_per_motif,
            initial_geometry_gate=0.5,
        )
        # V3 has an explicit non-collapsible mixture rather than V2's learned
        # per-channel gate.  Remove the unused module so the parameter topology
        # and checkpoint state say exactly what the model executes.
        del self.geometry_output
        self.geometry_fraction = float(geometry_fraction)
        self.carrier_geometry_projection = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.carrier_geometry_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False
        )
        self.endpoint_geometry_projection = nn.Linear(
            atom_memory_dim, hidden_size, bias=False
        )
        self.endpoint_geometry_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False
        )

    def encode_atom_memory(
        self,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None = None,
    ) -> Tensor:
        """Expose validated L1/L2 atom memory for detached V4 targets."""

        return self._encode_atom_memory(
            e3fp_input_ids,
            atom_mask,
            atom_is_attachment,
        )

    @staticmethod
    def _require_endpoint_mapping(
        endpoint_token_to_atom: Tensor,
        *,
        attention_mask: Tensor,
        atom_mask: Tensor,
        atom_is_attachment: Tensor | None,
    ) -> Tensor:
        if endpoint_token_to_atom.shape != attention_mask.shape:
            raise MotifGeometryAdapterError(
                "endpoint_token_to_atom must share the token shape [B,T]"
            )
        if endpoint_token_to_atom.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise MotifGeometryAdapterError(
                "endpoint_token_to_atom must contain integer indices"
            )
        if endpoint_token_to_atom.device != attention_mask.device:
            raise MotifGeometryAdapterError(
                "endpoint addresses must share the adapter device"
            )
        if bool(((~attention_mask) & (endpoint_token_to_atom != -1)).any()):
            raise MotifGeometryAdapterError("padded token endpoint address must be -1")
        endpoint_mask = endpoint_token_to_atom >= 0
        if bool((endpoint_token_to_atom < -1).any()):
            raise MotifGeometryAdapterError("endpoint atom index cannot be below -1")
        if bool(endpoint_mask.any()):
            if atom_is_attachment is None:
                raise MotifGeometryAdapterError(
                    "endpoint injection requires attachment-atom roles"
                )
            if atom_is_attachment.shape != atom_mask.shape or atom_is_attachment.dtype != torch.bool:
                raise MotifGeometryAdapterError(
                    "atom_is_attachment must be bool [B,A]"
                )
            safe = endpoint_token_to_atom.clamp_min(0).to(torch.long)
            if bool((safe[endpoint_mask] >= atom_mask.shape[1]).any()):
                raise MotifGeometryAdapterError("endpoint atom index is out of range")
            addressed_valid = atom_mask.gather(1, safe)
            addressed_attachment = atom_is_attachment.gather(1, safe)
            if bool((endpoint_mask & (~addressed_valid | ~addressed_attachment)).any()):
                raise MotifGeometryAdapterError(
                    "every endpoint must address one active attachment atom"
                )
        return endpoint_token_to_atom.to(torch.long)

    def encode(
        self,
        input_embeddings: Tensor,
        *,
        attention_mask: Tensor,
        e3fp_input_ids: Tensor,
        atom_mask: Tensor,
        atom_to_motif: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
        identity_span_bounds: Tensor,
        endpoint_token_to_atom: Tensor,
        atom_is_attachment: Tensor | None = None,
        state_memory_mode: str = "aligned",
        geometry_component_mode: str = "both",
    ) -> MotifGeometryEncoding:
        if state_memory_mode not in {"aligned", "zero"}:
            raise MotifGeometryAdapterError(
                "state_memory_mode must be aligned or zero"
            )
        if geometry_component_mode not in {
            "both",
            "carrier_only",
            "endpoint_only",
            "zero",
        }:
            raise MotifGeometryAdapterError(
                "geometry_component_mode must be both, carrier_only, "
                "endpoint_only or zero"
            )
        batch_size, token_width, atom_width, motif_width = self._validate_common(
            token_hidden=input_embeddings,
            attention_mask=attention_mask,
            atom_mask=atom_mask,
            atom_to_motif=atom_to_motif,
            motif_mask=motif_mask,
            motif_to_carrier=motif_to_carrier,
        )
        endpoint_token_to_atom = self._require_endpoint_mapping(
            endpoint_token_to_atom,
            attention_mask=attention_mask,
            atom_mask=atom_mask,
            atom_is_attachment=atom_is_attachment,
        )
        atom_memory = self._encode_atom_memory(
            e3fp_input_ids,
            atom_mask,
            atom_is_attachment,
        )
        if state_memory_mode == "zero" or geometry_component_mode == "zero":
            return MotifGeometryEncoding(
                fused_embeddings=input_embeddings,
                atom_memory=torch.zeros_like(atom_memory),
                pre_t5_motif_context=input_embeddings.new_zeros(
                    (batch_size, motif_width, self.hidden_size)
                ),
                cross_attention_weights=input_embeddings.new_zeros(
                    (batch_size, motif_width, atom_width)
                ),
            )

        motif_query = self._pool_identity_queries(
            input_embeddings,
            motif_mask,
            identity_span_bounds,
        )
        keys = self.atom_key(atom_memory)
        values = self.atom_value(atom_memory)
        scores = torch.matmul(
            motif_query, keys.transpose(1, 2)
        ) / math.sqrt(self.hidden_size)
        motif_ids = torch.arange(
            motif_width, device=input_embeddings.device
        ).view(1, -1, 1)
        ownership = (
            motif_mask.unsqueeze(-1)
            & atom_mask.unsqueeze(1)
            & (atom_to_motif.unsqueeze(1) == motif_ids)
        )
        scores = scores.masked_fill(~ownership, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(input_embeddings.dtype)
        weights = weights * ownership.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        attended = torch.matmul(weights, values)
        carrier_geometry = self.carrier_geometry_norm(
            self.carrier_geometry_projection(attended)
        )
        carrier_geometry = carrier_geometry * motif_mask.unsqueeze(-1).to(
            carrier_geometry.dtype
        )

        fraction = self.geometry_fraction
        flat_input = input_embeddings.reshape(batch_size * token_width, self.hidden_size)
        flat_updates = torch.zeros_like(flat_input)
        offsets = torch.arange(
            batch_size, device=input_embeddings.device
        ).unsqueeze(1) * token_width

        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        flat_carriers = (safe_carriers + offsets)[motif_mask]
        if geometry_component_mode in {"both", "carrier_only"}:
            carrier_base = flat_input[flat_carriers]
            carrier_delta = fraction * (
                carrier_geometry[motif_mask].to(input_embeddings.dtype) - carrier_base
            )
            flat_updates.index_add_(0, flat_carriers, carrier_delta)

        endpoint_mask = endpoint_token_to_atom >= 0
        if bool(endpoint_mask.any()) and geometry_component_mode in {
            "both",
            "endpoint_only",
        }:
            token_positions = torch.arange(
                token_width, device=input_embeddings.device
            ).view(1, -1).expand(batch_size, -1)
            flat_endpoints = (token_positions + offsets)[endpoint_mask]
            if bool(torch.isin(flat_endpoints, flat_carriers).any()):
                raise MotifGeometryAdapterError(
                    "motif carrier and connection endpoint tokens must be distinct"
                )
            safe_atoms = endpoint_token_to_atom.clamp_min(0).to(torch.long)
            addressed_memory = atom_memory.gather(
                1,
                safe_atoms.unsqueeze(-1).expand(-1, -1, self.atom_memory_dim),
            )
            endpoint_geometry = self.endpoint_geometry_norm(
                self.endpoint_geometry_projection(addressed_memory)
            )
            endpoint_base = flat_input[flat_endpoints]
            endpoint_delta = fraction * (
                endpoint_geometry[endpoint_mask].to(input_embeddings.dtype)
                - endpoint_base
            )
            flat_updates.index_add_(0, flat_endpoints, endpoint_delta)

        fused = (flat_input + flat_updates).view(
            batch_size, token_width, self.hidden_size
        )
        return MotifGeometryEncoding(
            fused_embeddings=fused,
            atom_memory=atom_memory,
            pre_t5_motif_context=carrier_geometry,
            cross_attention_weights=weights,
        )


__all__ = ["ADAPTER_ID", "MotifGeometryAdapterV3"]
