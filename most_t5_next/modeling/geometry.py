"""Carrier and endpoint geometry injection for fragSMILES inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from most_t5_next.interfaces import GeometryMode

from .e3fp import E3FPShellEmbedding


_IDENTITY_POSITION_CAPACITY = 512


class GeometryInputError(ValueError):
    pass


@dataclass(frozen=True)
class GeometryEncoding:
    fused_embeddings: Tensor
    atom_embeddings: Tensor
    fragment_embeddings: Tensor
    fragment_atom_attention: Tensor


class GeometryAdapter(nn.Module):
    """Inject motif summaries and attachment states into existing token positions."""

    def __init__(
        self,
        hidden_size: int,
        *,
        fp_bits: int = 4096,
        atom_embedding_dim: int = 768,
        geometry_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or atom_embedding_dim <= 0:
            raise GeometryInputError("model dimensions must be positive")
        if not 0.0 < geometry_fraction < 1.0:
            raise GeometryInputError("geometry_fraction must lie between zero and one")
        self.hidden_size = int(hidden_size)
        self.atom_embedding_dim = int(atom_embedding_dim)
        self.identity_position_capacity = _IDENTITY_POSITION_CAPACITY
        self.geometry_fraction = float(geometry_fraction)

        self.e3fp = E3FPShellEmbedding(fp_bits, atom_embedding_dim)
        self.identity_score = nn.Linear(hidden_size, 1, bias=False)
        self.identity_position_score = nn.Embedding(
            self.identity_position_capacity, 1
        )
        self.identity_norm = nn.LayerNorm(hidden_size)
        self.identity_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.atom_key = nn.Linear(atom_embedding_dim, hidden_size, bias=False)
        self.atom_value = nn.Linear(atom_embedding_dim, hidden_size, bias=False)
        self.carrier_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.endpoint_projection = nn.Linear(atom_embedding_dim, hidden_size, bias=False)
        self.endpoint_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def _pool_fragment_identity(
        self,
        token_embeddings: Tensor,
        attention_mask: Tensor,
        fragment_mask: Tensor,
        span_bounds: Tensor,
    ) -> Tensor:
        batch, tokens, _ = token_embeddings.shape
        fragments = fragment_mask.shape[1]
        if span_bounds.shape != (batch, fragments, 2):
            raise GeometryInputError("identity_span_bounds must be [batch, fragments, 2]")
        start, stop = span_bounds.unbind(dim=-1)
        active = fragment_mask
        if bool((active & ((start < 0) | (stop <= start) | (stop > tokens))).any()):
            raise GeometryInputError("active fragment span lies outside the token sequence")
        width = stop - start
        if bool((active & (width > self.identity_position_capacity)).any()):
            raise GeometryInputError("fragment identity span exceeds configured width")

        positions = torch.arange(tokens, device=token_embeddings.device).view(1, 1, -1)
        in_span = (
            active.unsqueeze(-1)
            & attention_mask.unsqueeze(1)
            & positions.ge(start.unsqueeze(-1))
            & positions.lt(stop.unsqueeze(-1))
        )
        relative = (positions - start.unsqueeze(-1)).clamp(
            min=0, max=self.identity_position_capacity - 1
        )
        content_score = self.identity_score(token_embeddings).squeeze(-1).unsqueeze(1)
        position_score = self.identity_position_score(relative).squeeze(-1)
        scores = (content_score + position_score).masked_fill(
            ~in_span, torch.finfo(token_embeddings.dtype).min
        )
        weights = torch.softmax(scores.float(), dim=-1).to(token_embeddings.dtype)
        weights = weights * in_span.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.matmul(weights, token_embeddings)
        return self.identity_query(self.identity_norm(pooled))

    @staticmethod
    def _require_shape(tensor: Tensor, shape: tuple[int, ...], name: str) -> None:
        if tensor.shape != shape:
            raise GeometryInputError(f"{name} must have shape {shape}")

    def forward(
        self,
        token_embeddings: Tensor,
        *,
        attention_mask: Tensor,
        e3fp_ids: Tensor,
        atom_mask: Tensor,
        atom_to_fragment: Tensor,
        fragment_mask: Tensor,
        fragment_to_carrier: Tensor,
        identity_span_bounds: Tensor,
        endpoint_mask: Tensor,
        endpoint_to_atom: Tensor,
        endpoint_to_token: Tensor,
        endpoint_to_fragment: Tensor,
        endpoint_is_explicit: Tensor,
        token_is_connector_endpoint: Tensor,
        atom_is_attachment: Tensor,
        fragment_geometry_mask: Tensor | None = None,
        endpoint_geometry_mask: Tensor | None = None,
        geometry_mode: GeometryMode = "full",
    ) -> GeometryEncoding:
        if token_embeddings.ndim != 3 or token_embeddings.shape[-1] != self.hidden_size:
            raise GeometryInputError("token_embeddings must be [batch, tokens, hidden]")
        batch, tokens, _ = token_embeddings.shape
        if attention_mask.shape != (batch, tokens):
            raise GeometryInputError("attention_mask must match the token axis")
        attention_mask = attention_mask.to(torch.bool)
        if atom_mask.ndim != 2 or atom_mask.shape[0] != batch:
            raise GeometryInputError("atom_mask must be [batch, atoms]")
        atom_mask = atom_mask.to(torch.bool)
        atoms = atom_mask.shape[1]
        if fragment_mask.ndim != 2 or fragment_mask.shape[0] != batch:
            raise GeometryInputError("fragment_mask must be [batch, fragments]")
        fragment_mask = fragment_mask.to(torch.bool)
        fragments = fragment_mask.shape[1]
        if endpoint_mask.ndim != 2 or endpoint_mask.shape[0] != batch:
            raise GeometryInputError("endpoint_mask must be [batch, endpoints]")
        endpoint_mask = endpoint_mask.to(torch.bool)
        endpoints = endpoint_mask.shape[1]

        self._require_shape(e3fp_ids, (batch, atoms, 4), "e3fp_ids")
        self._require_shape(atom_to_fragment, (batch, atoms), "atom_to_fragment")
        self._require_shape(fragment_to_carrier, (batch, fragments), "fragment_to_carrier")
        self._require_shape(endpoint_to_atom, (batch, endpoints), "endpoint_to_atom")
        self._require_shape(endpoint_to_token, (batch, endpoints), "endpoint_to_token")
        self._require_shape(endpoint_to_fragment, (batch, endpoints), "endpoint_to_fragment")
        self._require_shape(endpoint_is_explicit, (batch, endpoints), "endpoint_is_explicit")
        self._require_shape(
            token_is_connector_endpoint,
            (batch, tokens),
            "token_is_connector_endpoint",
        )
        self._require_shape(atom_is_attachment, (batch, atoms), "atom_is_attachment")
        if fragment_geometry_mask is not None:
            self._require_shape(
                fragment_geometry_mask,
                (batch, fragments),
                "fragment_geometry_mask",
            )
        if endpoint_geometry_mask is not None:
            self._require_shape(
                endpoint_geometry_mask,
                (batch, endpoints),
                "endpoint_geometry_mask",
            )
        if geometry_mode not in {"full", "carrier", "endpoint", "none"}:
            raise GeometryInputError(f"unknown geometry mode: {geometry_mode}")

        if fragments:
            bad_owner = atom_mask & (
                atom_to_fragment.lt(0) | atom_to_fragment.ge(fragments)
            )
            if bool(bad_owner.any()):
                raise GeometryInputError("active atom lies outside the fragment axis")
            safe_owners = atom_to_fragment.clamp_min(0).to(torch.long)
            if bool((atom_mask & ~fragment_mask.gather(1, safe_owners)).any()):
                raise GeometryInputError("active atom owns a padded fragment")
            atom_counts = torch.zeros(
                (batch, fragments), dtype=torch.long, device=atom_mask.device
            )
            atom_counts.scatter_add_(1, safe_owners, atom_mask.to(torch.long))
            if bool((fragment_mask & atom_counts.eq(0)).any()):
                raise GeometryInputError("active fragment has no owned atom")
        elif bool((atom_mask & atom_to_fragment.ne(-1)).any()):
            raise GeometryInputError("whole-molecule fallback atoms must be unowned")

        safe_carriers = fragment_to_carrier.clamp_min(0).to(torch.long)
        bad_carrier = fragment_mask & (
            fragment_to_carrier.lt(0) | fragment_to_carrier.ge(tokens)
        )
        if bool(bad_carrier.any()):
            raise GeometryInputError("active fragment carrier lies outside the token axis")
        if fragments and bool((fragment_mask & ~attention_mask.gather(1, safe_carriers)).any()):
            raise GeometryInputError("active fragment carrier points to padding")

        safe_endpoint_atoms = endpoint_to_atom.clamp_min(0).to(torch.long)
        safe_endpoint_tokens = endpoint_to_token.clamp_min(0).to(torch.long)
        safe_endpoint_fragments = endpoint_to_fragment.clamp_min(0).to(torch.long)
        if endpoints:
            bad_endpoint = endpoint_mask & (
                endpoint_to_atom.lt(0)
                | endpoint_to_atom.ge(atoms)
                | endpoint_to_token.lt(0)
                | endpoint_to_token.ge(tokens)
                | endpoint_to_fragment.lt(0)
                | endpoint_to_fragment.ge(fragments)
            )
            if bool(bad_endpoint.any()):
                raise GeometryInputError("active endpoint address is out of range")
            if bool((endpoint_mask & ~atom_mask.gather(1, safe_endpoint_atoms)).any()):
                raise GeometryInputError("endpoint addresses a padded atom")
            if bool((endpoint_mask & ~atom_is_attachment.gather(1, safe_endpoint_atoms)).any()):
                raise GeometryInputError("endpoint must address an attachment atom")
            if bool((endpoint_mask & ~fragment_mask.gather(1, safe_endpoint_fragments)).any()):
                raise GeometryInputError("endpoint addresses a padded fragment")
            owners = atom_to_fragment.gather(1, safe_endpoint_atoms)
            if bool((endpoint_mask & owners.ne(endpoint_to_fragment)).any()):
                raise GeometryInputError("endpoint atom and fragment ownership disagree")
            owner_carriers = fragment_to_carrier.gather(1, safe_endpoint_fragments)
            explicit = endpoint_mask & endpoint_is_explicit.to(torch.bool)
            implicit = endpoint_mask & ~endpoint_is_explicit.to(torch.bool)
            connector_target = token_is_connector_endpoint.to(torch.bool).gather(
                1, safe_endpoint_tokens
            )
            if bool((implicit & endpoint_to_token.ne(owner_carriers)).any()):
                raise GeometryInputError("implicit endpoint must use its fragment carrier")
            if bool((explicit & ~connector_target).any()):
                raise GeometryInputError("explicit endpoint must use a connector endpoint token")

        if geometry_mode == "none":
            return GeometryEncoding(
                fused_embeddings=token_embeddings,
                atom_embeddings=token_embeddings.new_zeros(
                    (batch, atoms, self.atom_embedding_dim)
                ),
                fragment_embeddings=token_embeddings.new_zeros(
                    (batch, fragments, self.hidden_size)
                ),
                fragment_atom_attention=token_embeddings.new_zeros((batch, fragments, atoms)),
            )

        atom_embeddings = self.e3fp(e3fp_ids, atom_mask)
        atom_geometry = atom_mask & e3fp_ids.ge(0).any(dim=-1)
        fragment_ids = torch.arange(fragments, device=token_embeddings.device).view(1, -1, 1)
        ownership = (
            fragment_mask.unsqueeze(-1)
            & atom_geometry.unsqueeze(1)
            & atom_to_fragment.unsqueeze(1).eq(fragment_ids)
        )
        fragment_has_geometry = ownership.any(dim=-1)
        if fragment_geometry_mask is None:
            fragment_geometry_mask = fragment_mask
        fragment_geometry_mask = fragment_geometry_mask.to(torch.bool) & fragment_has_geometry
        if endpoint_geometry_mask is None:
            endpoint_geometry_mask = endpoint_mask
        endpoint_geometry_mask = endpoint_geometry_mask.to(torch.bool)
        if endpoints:
            endpoint_geometry_mask &= atom_geometry.gather(1, safe_endpoint_atoms)
            endpoint_geometry_mask &= fragment_geometry_mask.gather(
                1, safe_endpoint_fragments
            )

        queries = self._pool_fragment_identity(
            token_embeddings, attention_mask, fragment_mask, identity_span_bounds
        )
        scores = torch.matmul(queries, self.atom_key(atom_embeddings).transpose(1, 2))
        scores = scores / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(~ownership, torch.finfo(scores.dtype).min)
        fragment_atom_attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        fragment_atom_attention = fragment_atom_attention * ownership.to(scores.dtype)
        fragment_atom_attention = fragment_atom_attention / fragment_atom_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        fragment_embeddings = torch.matmul(
            fragment_atom_attention, self.atom_value(atom_embeddings)
        )
        fragment_embeddings = self.carrier_norm(
            self.carrier_projection(fragment_embeddings)
        )
        fragment_embeddings = fragment_embeddings * fragment_geometry_mask.unsqueeze(-1).to(
            fragment_embeddings.dtype
        )

        flat_size = batch * tokens
        geometry_sum = token_embeddings.new_zeros((flat_size, self.hidden_size))
        geometry_count = token_embeddings.new_zeros((flat_size, 1))
        offsets = torch.arange(batch, device=token_embeddings.device).unsqueeze(1) * tokens
        if geometry_mode in {"full", "carrier"} and bool(fragment_geometry_mask.any()):
            positions = (safe_carriers + offsets)[fragment_geometry_mask]
            geometry_sum.index_add_(0, positions, fragment_embeddings[fragment_geometry_mask])
            geometry_count.index_add_(
                0, positions, geometry_count.new_ones((positions.numel(), 1))
            )

        if geometry_mode in {"full", "endpoint"} and bool(endpoint_geometry_mask.any()):
            endpoint_atoms = atom_embeddings.gather(
                1, safe_endpoint_atoms.unsqueeze(-1).expand(-1, -1, self.atom_embedding_dim)
            )
            endpoint_embeddings = self.endpoint_norm(self.endpoint_projection(endpoint_atoms))
            positions = (safe_endpoint_tokens + offsets)[endpoint_geometry_mask]
            geometry_sum.index_add_(0, positions, endpoint_embeddings[endpoint_geometry_mask])
            geometry_count.index_add_(
                0, positions, geometry_count.new_ones((positions.numel(), 1))
            )

        flat_tokens = token_embeddings.reshape(flat_size, self.hidden_size)
        present = geometry_count.gt(0)
        mean_geometry = geometry_sum / geometry_count.clamp_min(1.0)
        fused = flat_tokens + self.geometry_fraction * (mean_geometry - flat_tokens) * present
        return GeometryEncoding(
            fused_embeddings=fused.view_as(token_embeddings),
            atom_embeddings=atom_embeddings,
            fragment_embeddings=fragment_embeddings,
            fragment_atom_attention=fragment_atom_attention,
        )
