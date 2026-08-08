"""One motif-aware E3FP adapter for the state-aware T5 mechanism screen.

The adapter keeps the three scientific domains explicit:

* GraphPorts identity spans build one motif query.  A corrupted span already
  consists of one T5 sentinel, so the same operation covers visible identity
  and identity masking without inspecting token values.
* E3FP level 1/2 states remain an atom memory.  Level 0 is treated as 2D
  identity and level 3 remains diagnostic-only; neither enters this module.
  One constrained cross-attention allows a motif query to read only atoms
  owned by that logical motif.
* State logits are decoded only after the fused input has passed through the
  T5 encoder.  The decoder combines the encoder carrier state with the owned
  atom memory and therefore cannot become a stand-alone E3FP autoencoder.

GraphPorts connection tokens remain ordinary T5 input tokens in v1.  No
motif-graph attention bias or second topology path is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


class MotifGeometryAdapterError(ValueError):
    """The motif, token, or atom domains do not form one valid batch."""


@dataclass(frozen=True)
class MotifGeometryEncoding:
    """Pre-T5 result while retaining atom-resolution geometry.

    ``pre_t5_motif_context`` is diagnostic/fusion state only.  It must not be
    used as the final state-prediction context; :meth:`decode_state` consumes
    the post-T5 encoder hidden state instead.
    """

    fused_embeddings: Tensor
    atom_memory: Tensor
    pre_t5_motif_context: Tensor
    cross_attention_weights: Tensor


class MotifGeometryAdapterV1(nn.Module):
    """Fuse motif-owned E3FP atom memory once and decode levels 1/2 after T5.

    External E3FP IDs use the established G1 domain: ``-1`` is padding,
    ``0..N-1`` are real categorical states, and ``N+1`` is the state-mask
    token.  ``N`` is reserved internally as the embedding padding row.

    ``identity_span_bounds[b, m] == (start, stop)`` addresses the *corrupted*
    GraphPorts-v1 input.  An unmasked motif therefore pools its complete
    identity span, while a masked motif pools exactly its sentinel.  Connection
    tokens are outside these bounds and are never duplicated as geometry bias.
    """

    predicted_levels = (1, 2)
    consumed_levels = (1, 2)

    def __init__(
        self,
        *,
        num_e3fp_embeddings: int,
        hidden_size: int,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 128,
        max_identity_span_length: int = 128,
    ) -> None:
        super().__init__()
        values = {
            "num_e3fp_embeddings": num_e3fp_embeddings,
            "hidden_size": hidden_size,
            "state_embedding_dim": state_embedding_dim,
            "atom_memory_dim": atom_memory_dim,
            "max_identity_span_length": max_identity_span_length,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MotifGeometryAdapterError(f"{name} must be a positive integer")
        if num_e3fp_embeddings <= 1:
            raise MotifGeometryAdapterError("num_e3fp_embeddings must exceed one")

        self.num_e3fp_embeddings = num_e3fp_embeddings
        self.hidden_size = hidden_size
        self.state_embedding_dim = state_embedding_dim
        self.atom_memory_dim = atom_memory_dim
        self.max_identity_span_length = max_identity_span_length
        self.padding_token_id = num_e3fp_embeddings
        self.mask_token_id = num_e3fp_embeddings + 1

        self.state_embedding = nn.Embedding(
            num_e3fp_embeddings + 2,
            state_embedding_dim,
            padding_idx=self.padding_token_id,
        )
        self.level_embedding = nn.Embedding(4, state_embedding_dim)
        self.atom_role_embedding = nn.Embedding(2, state_embedding_dim)
        self.atom_encoder = nn.Sequential(
            nn.Linear(3 * state_embedding_dim + 2, atom_memory_dim),
            nn.GELU(),
            nn.Linear(atom_memory_dim, atom_memory_dim),
            nn.GELU(),
        )

        # A content score plus a relative-position bias pools the full identity
        # span without materializing [B,M,T,H].  Position-dependent weights
        # retain token order, unlike a plain Deep-Sets/mean identity pool.
        self.identity_token_score = nn.Linear(hidden_size, 1, bias=False)
        self.identity_position_score = nn.Embedding(max_identity_span_length, 1)
        self.identity_norm = nn.LayerNorm(hidden_size)
        self.identity_query = nn.Linear(hidden_size, hidden_size, bias=False)

        self.atom_key = nn.Linear(atom_memory_dim, hidden_size, bias=False)
        self.atom_value = nn.Linear(atom_memory_dim, hidden_size, bias=False)
        self.geometry_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.motif_context_norm = nn.LayerNorm(hidden_size)
        # ReZero keeps the pretrained T5 input exact at initialization.  The
        # scalar is trainable and the state head still trains from step one.
        self.geometry_residual_scale = nn.Parameter(torch.zeros(()))

        self.state_level_embedding = nn.Embedding(2, state_embedding_dim)
        self.state_decoder = nn.Sequential(
            nn.Linear(
                atom_memory_dim + hidden_size + state_embedding_dim,
                atom_memory_dim,
            ),
            nn.GELU(),
            nn.Linear(atom_memory_dim, num_e3fp_embeddings),
        )

    @staticmethod
    def _require_integer(tensor: Tensor, name: str) -> None:
        if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
            raise MotifGeometryAdapterError(f"{name} must use an integer dtype")

    def _validate_common(
        self,
        *,
        token_hidden: Tensor,
        attention_mask: Tensor,
        atom_mask: Tensor,
        atom_to_motif: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
    ) -> tuple[int, int, int, int]:
        if not isinstance(token_hidden, Tensor) or token_hidden.ndim != 3:
            raise MotifGeometryAdapterError("token_hidden must be [B,T,H]")
        if not token_hidden.is_floating_point() or token_hidden.shape[-1] != self.hidden_size:
            raise MotifGeometryAdapterError("token_hidden has an invalid dtype or width")
        batch_size, token_width, _ = token_hidden.shape
        if attention_mask.shape != (batch_size, token_width) or attention_mask.dtype != torch.bool:
            raise MotifGeometryAdapterError("attention_mask must be bool [B,T]")
        if atom_mask.ndim != 2 or atom_mask.shape[0] != batch_size or atom_mask.dtype != torch.bool:
            raise MotifGeometryAdapterError("atom_mask must be bool [B,A]")
        atom_width = atom_mask.shape[1]
        if atom_to_motif.shape != (batch_size, atom_width):
            raise MotifGeometryAdapterError("atom_to_motif must be [B,A]")
        self._require_integer(atom_to_motif, "atom_to_motif")
        if motif_mask.ndim != 2 or motif_mask.shape[0] != batch_size or motif_mask.dtype != torch.bool:
            raise MotifGeometryAdapterError("motif_mask must be bool [B,M]")
        motif_width = motif_mask.shape[1]
        if motif_to_carrier.shape != (batch_size, motif_width):
            raise MotifGeometryAdapterError("motif_to_carrier must be [B,M]")
        self._require_integer(motif_to_carrier, "motif_to_carrier")
        tensors = (
            attention_mask,
            atom_mask,
            atom_to_motif,
            motif_mask,
            motif_to_carrier,
        )
        if any(value.device != token_hidden.device for value in tensors):
            raise MotifGeometryAdapterError("all adapter tensors must share one device")
        if bool((~attention_mask.any(dim=1)).any()):
            raise MotifGeometryAdapterError("every row needs at least one visible token")
        if bool((~atom_mask.any(dim=1)).any()):
            raise MotifGeometryAdapterError("every row needs at least one active atom")
        if bool((~motif_mask.any(dim=1)).any()):
            raise MotifGeometryAdapterError("every row needs at least one active motif")

        active_atom_bad = atom_mask & (
            (atom_to_motif < 0) | (atom_to_motif >= motif_width)
        )
        if bool(active_atom_bad.any()):
            raise MotifGeometryAdapterError("active atom maps outside the motif domain")
        if bool(((~atom_mask) & (atom_to_motif != -1)).any()):
            raise MotifGeometryAdapterError("padded atom must map to -1")
        safe_atom_groups = atom_to_motif.clamp_min(0).to(torch.long)
        owned_motif_is_active = motif_mask.gather(1, safe_atom_groups)
        if bool((atom_mask & ~owned_motif_is_active).any()):
            raise MotifGeometryAdapterError("active atom maps to a padded motif")

        active_carrier_bad = motif_mask & (
            (motif_to_carrier < 0) | (motif_to_carrier >= token_width)
        )
        if bool(active_carrier_bad.any()):
            raise MotifGeometryAdapterError("active motif carrier is outside token domain")
        if bool(((~motif_mask) & (motif_to_carrier != -1)).any()):
            raise MotifGeometryAdapterError("padded motif carrier must be -1")
        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        carrier_visible = attention_mask.gather(1, safe_carriers)
        if bool((motif_mask & ~carrier_visible).any()):
            raise MotifGeometryAdapterError("active motif carrier points to token padding")
        for row in range(batch_size):
            active = motif_to_carrier[row][motif_mask[row]]
            if active.unique().numel() != active.numel():
                raise MotifGeometryAdapterError("active motifs require distinct carriers")

        ownership_count = torch.zeros(
            batch_size,
            motif_width,
            dtype=torch.long,
            device=token_hidden.device,
        )
        ownership_count.scatter_add_(
            1,
            safe_atom_groups,
            atom_mask.to(torch.long),
        )
        if bool((motif_mask & (ownership_count == 0)).any()):
            raise MotifGeometryAdapterError("every active motif needs an owned atom")
        return batch_size, token_width, atom_width, motif_width

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
            raise MotifGeometryAdapterError("E3FP IDs and atom mask must share one device")
        if atom_is_attachment is None:
            atom_is_attachment = torch.zeros_like(atom_mask)
        if atom_is_attachment.shape != atom_mask.shape or atom_is_attachment.dtype != torch.bool:
            raise MotifGeometryAdapterError("atom_is_attachment must be bool [B,A]")
        if atom_is_attachment.device != atom_mask.device:
            raise MotifGeometryAdapterError("atom roles must share the adapter device")
        bad_id = (e3fp_input_ids < -1) | (e3fp_input_ids > self.mask_token_id)
        if bool(bad_id.any()):
            raise MotifGeometryAdapterError("E3FP input ID is outside the state domain")
        if bool(((~atom_mask).unsqueeze(-1) & (e3fp_input_ids != -1)).any()):
            raise MotifGeometryAdapterError("padded atom E3FP rows must contain only -1")

        # L0 duplicates identity and L3 was retained only as a diagnostic in
        # the frozen scientific protocol.  Slicing before embedding makes
        # their non-influence executable rather than conventional.
        consumed_ids = e3fp_input_ids[..., list(self.consumed_levels)]
        shell_valid = (consumed_ids >= 0) & atom_mask.unsqueeze(-1)
        normalized = consumed_ids.masked_fill(consumed_ids < 0, self.padding_token_id)
        level_ids = torch.tensor(
            self.consumed_levels,
            device=e3fp_input_ids.device,
            dtype=torch.long,
        ).view(1, 1, 2)
        shell_hidden = self.state_embedding(normalized.to(torch.long))
        shell_hidden = shell_hidden + self.level_embedding(level_ids)
        shell_hidden = shell_hidden * shell_valid.unsqueeze(-1).to(shell_hidden.dtype)
        role_hidden = self.atom_role_embedding(atom_is_attachment.to(torch.long))
        role_hidden = role_hidden * atom_mask.unsqueeze(-1).to(role_hidden.dtype)
        atom_input = torch.cat(
            (shell_hidden.flatten(start_dim=2), shell_valid.to(shell_hidden.dtype), role_hidden),
            dim=-1,
        )
        atom_memory = self.atom_encoder(atom_input)
        return atom_memory * atom_mask.unsqueeze(-1).to(atom_memory.dtype)

    def _pool_identity_queries(
        self,
        input_embeddings: Tensor,
        motif_mask: Tensor,
        identity_span_bounds: Tensor,
    ) -> Tensor:
        batch_size, token_width, _ = input_embeddings.shape
        motif_width = motif_mask.shape[1]
        if identity_span_bounds.shape != (batch_size, motif_width, 2):
            raise MotifGeometryAdapterError("identity_span_bounds must be [B,M,2]")
        self._require_integer(identity_span_bounds, "identity_span_bounds")
        if identity_span_bounds.device != input_embeddings.device:
            raise MotifGeometryAdapterError("identity spans must share the adapter device")
        starts = identity_span_bounds[..., 0]
        stops = identity_span_bounds[..., 1]
        invalid_active = motif_mask & (
            (starts < 0)
            | (stops <= starts)
            | (stops > token_width)
            | ((stops - starts) > self.max_identity_span_length)
        )
        if bool(invalid_active.any()):
            raise MotifGeometryAdapterError("active identity span is empty, out of range, or too long")
        if bool(((~motif_mask) & ((starts != -1) | (stops != -1))).any()):
            raise MotifGeometryAdapterError("padded motif identity span must be (-1,-1)")

        positions = torch.arange(token_width, device=input_embeddings.device).view(1, 1, -1)
        span_membership = (
            motif_mask.unsqueeze(-1)
            & (positions >= starts.unsqueeze(-1))
            & (positions < stops.unsqueeze(-1))
        )
        relative_positions = (positions - starts.unsqueeze(-1)).clamp(
            min=0,
            max=self.max_identity_span_length - 1,
        )
        token_scores = self.identity_token_score(input_embeddings).squeeze(-1).unsqueeze(1)
        position_scores = self.identity_position_score(relative_positions.to(torch.long)).squeeze(-1)
        scores = token_scores + position_scores
        scores = scores.masked_fill(~span_membership, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(input_embeddings.dtype)
        weights = weights * span_membership.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.einsum("bmt,bth->bmh", weights, input_embeddings)
        query = self.identity_query(self.identity_norm(pooled))
        return query * motif_mask.unsqueeze(-1).to(query.dtype)

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
        atom_is_attachment: Tensor | None = None,
    ) -> MotifGeometryEncoding:
        """Build atom memory, perform one owned-atom attention, and write carriers."""

        batch_size, token_width, atom_width, motif_width = self._validate_common(
            token_hidden=input_embeddings,
            attention_mask=attention_mask,
            atom_mask=atom_mask,
            atom_to_motif=atom_to_motif,
            motif_mask=motif_mask,
            motif_to_carrier=motif_to_carrier,
        )
        atom_memory = self._encode_atom_memory(
            e3fp_input_ids,
            atom_mask,
            atom_is_attachment,
        )
        motif_query = self._pool_identity_queries(
            input_embeddings,
            motif_mask,
            identity_span_bounds,
        )

        keys = self.atom_key(atom_memory)
        values = self.atom_value(atom_memory)
        scores = torch.matmul(motif_query, keys.transpose(1, 2)) / math.sqrt(self.hidden_size)
        motif_ids = torch.arange(motif_width, device=input_embeddings.device).view(1, -1, 1)
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
        motif_context = self.motif_context_norm(motif_query + attended)
        motif_context = motif_context * motif_mask.unsqueeze(-1).to(motif_context.dtype)
        delta = self.geometry_output(motif_context) * self.geometry_residual_scale

        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        offsets = torch.arange(batch_size, device=input_embeddings.device).unsqueeze(1) * token_width
        flat_carriers = (safe_carriers + offsets)[motif_mask]
        flat_delta = delta[motif_mask]
        flat_updates = input_embeddings.new_zeros((batch_size * token_width, self.hidden_size))
        flat_updates.index_add_(0, flat_carriers, flat_delta)
        fused = input_embeddings + flat_updates.view(batch_size, token_width, self.hidden_size)
        return MotifGeometryEncoding(
            fused_embeddings=fused,
            atom_memory=atom_memory,
            pre_t5_motif_context=motif_context,
            cross_attention_weights=weights,
        )

    def decode_state(
        self,
        atom_memory: Tensor,
        encoder_hidden: Tensor,
        *,
        attention_mask: Tensor,
        atom_mask: Tensor,
        atom_to_motif: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
    ) -> Tensor:
        """Return level-1/2 logits using the post-T5 owner-carrier context.

        The returned shape is ``[B,A,2,N]`` and level axis ``0,1`` corresponds
        exactly to external E3FP levels ``1,2``.
        """

        batch_size, _token_width, atom_width, _motif_width = self._validate_common(
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
            raise MotifGeometryAdapterError("atom_memory must share the encoder device and be floating point")

        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        motif_hidden = encoder_hidden.gather(
            1,
            safe_carriers.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        motif_hidden = motif_hidden * motif_mask.unsqueeze(-1).to(motif_hidden.dtype)
        safe_atom_groups = atom_to_motif.clamp_min(0).to(torch.long)
        atom_owner_context = motif_hidden.gather(
            1,
            safe_atom_groups.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        atom_owner_context = atom_owner_context * atom_mask.unsqueeze(-1).to(
            atom_owner_context.dtype
        )
        level_hidden = self.state_level_embedding(
            torch.arange(2, device=encoder_hidden.device)
        ).view(1, 1, 2, self.state_embedding_dim)
        decoder_input = torch.cat(
            (
                atom_memory.unsqueeze(2).expand(-1, -1, 2, -1),
                atom_owner_context.unsqueeze(2).expand(-1, -1, 2, -1),
                level_hidden.expand(batch_size, atom_width, -1, -1),
            ),
            dim=-1,
        )
        logits = self.state_decoder(decoder_input)
        return logits * atom_mask.unsqueeze(-1).unsqueeze(-1).to(logits.dtype)

    def forward(self, input_embeddings: Tensor, **kwargs: Tensor) -> MotifGeometryEncoding:
        """Alias :meth:`encode`; state decoding remains an explicit post-T5 call."""

        return self.encode(input_embeddings, **kwargs)


__all__ = [
    "MotifGeometryAdapterError",
    "MotifGeometryAdapterV1",
    "MotifGeometryEncoding",
]
