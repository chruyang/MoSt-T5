"""Same-identity, cross-record motif/state matching for 3D-MotifT5 V3.

The auxiliary head is discarded at inference.  Positives are one motif's own
state summary; negatives must have the same exact motif identity but come from
another molecule.  Identity alone therefore cannot solve the task.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MATCHING_ID = "most-t5-p2/motif-state-matching/v3-same-identity-cross-record"


class MotifStateMatchingError(ValueError):
    """The carrier/state/identity domains are inconsistent."""


@dataclass(frozen=True)
class MotifStateMatchingOutput:
    loss: Tensor
    eligible_anchors: int
    correct_anchors: int
    eligible_identity_groups: int
    tie_aware_top1_credit_sum: float = 0.0
    positive_probability_sum: float = 0.0
    uniform_loss_sum: float = 0.0

    @property
    def accuracy(self) -> float | None:
        if self.eligible_anchors == 0:
            return None
        return self.tie_aware_top1_credit_sum / self.eligible_anchors

    @property
    def mean_positive_probability(self) -> float | None:
        if self.eligible_anchors == 0:
            return None
        return self.positive_probability_sum / self.eligible_anchors

    @property
    def uniform_loss(self) -> float | None:
        if self.eligible_anchors == 0:
            return None
        return self.uniform_loss_sum / self.eligible_anchors


class MotifStateMatchingHeadV3(nn.Module):
    """Contrast post-T5 motif carriers against atom-derived motif states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        projection_dim: int = 128,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
            or isinstance(projection_dim, bool)
            or not isinstance(projection_dim, int)
            or projection_dim <= 0
            or isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or float(temperature) <= 0.0
        ):
            raise MotifStateMatchingError("matching-head dimensions are invalid")
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.temperature = float(temperature)
        self.query_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.state_projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(
        self,
        *,
        encoder_hidden: Tensor,
        motif_state: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
        exact_identity_sha256: Sequence[Sequence[str]],
    ) -> MotifStateMatchingOutput:
        if (
            encoder_hidden.ndim != 3
            or motif_state.ndim != 3
            or encoder_hidden.shape[0] != motif_state.shape[0]
            or encoder_hidden.shape[2] != self.hidden_size
            or motif_state.shape[2] != self.hidden_size
        ):
            raise MotifStateMatchingError(
                "encoder carriers and motif states must be [B,T,H]/[B,M,H]"
            )
        batch_size, motif_width = motif_state.shape[:2]
        if motif_mask.shape != (batch_size, motif_width) or motif_mask.dtype != torch.bool:
            raise MotifStateMatchingError("motif_mask must be bool [B,M]")
        if motif_to_carrier.shape != motif_mask.shape:
            raise MotifStateMatchingError("motif_to_carrier must be [B,M]")
        if motif_to_carrier.dtype == torch.bool or motif_to_carrier.is_floating_point():
            raise MotifStateMatchingError("motif carriers must be integer indices")
        if len(exact_identity_sha256) != batch_size:
            raise MotifStateMatchingError("identity rows differ from batch size")

        active: list[tuple[int, int, str]] = []
        for batch_index, identities in enumerate(exact_identity_sha256):
            count = int(motif_mask[batch_index].sum().item())
            if len(identities) != count:
                raise MotifStateMatchingError(
                    "identity count differs from the active motif domain"
                )
            for motif_index, identity in enumerate(identities):
                if not isinstance(identity, str) or len(identity) != 64:
                    raise MotifStateMatchingError("motif identity is not SHA-256")
                active.append((batch_index, motif_index, identity))
        if not active:
            raise MotifStateMatchingError("matching batch has no active motifs")

        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        if bool((safe_carriers[motif_mask] >= encoder_hidden.shape[1]).any()):
            raise MotifStateMatchingError("motif carrier lies outside token width")
        carriers = encoder_hidden.gather(
            1,
            safe_carriers.unsqueeze(-1).expand(-1, -1, self.hidden_size),
        )
        queries = F.normalize(self.query_projection(carriers), dim=-1)
        states = F.normalize(self.state_projection(motif_state), dim=-1)

        by_identity: dict[str, list[tuple[int, int]]] = {}
        for batch_index, motif_index, identity in active:
            by_identity.setdefault(identity, []).append((batch_index, motif_index))

        losses: list[Tensor] = []
        correct_rows: list[Tensor] = []
        tie_credit_rows: list[Tensor] = []
        positive_probability_rows: list[Tensor] = []
        uniform_loss = 0.0
        eligible_groups = 0
        for members in by_identity.values():
            if len({batch_index for batch_index, _ in members}) < 2:
                continue
            eligible_groups += 1
            batch_indices = torch.as_tensor(
                [batch_index for batch_index, _ in members],
                dtype=torch.long,
                device=queries.device,
            )
            motif_indices = torch.as_tensor(
                [motif_index for _, motif_index in members],
                dtype=torch.long,
                device=queries.device,
            )
            group_queries = queries[batch_indices, motif_indices]
            group_states = states[batch_indices, motif_indices]
            logits = torch.matmul(group_queries, group_states.transpose(0, 1))
            logits = logits / self.temperature
            size = len(members)
            diagonal = torch.eye(size, dtype=torch.bool, device=queries.device)
            candidates = batch_indices[:, None] != batch_indices[None, :]
            candidates = candidates | diagonal
            masked_logits = logits.masked_fill(~candidates, float("-inf"))
            targets = torch.arange(size, dtype=torch.long, device=queries.device)
            losses.append(F.cross_entropy(masked_logits, targets, reduction="none"))

            detached = masked_logits.detach().float()
            probabilities = torch.softmax(detached, dim=1)
            positive_probability_rows.append(probabilities.diagonal())
            maxima = detached.max(dim=1, keepdim=True).values
            tied = detached == maxima
            positive_is_tied = tied.diagonal()
            tie_credit_rows.append(
                positive_is_tied.to(torch.float32)
                / tied.sum(dim=1).clamp_min(1).to(torch.float32)
            )
            # Preserve the historical raw top-1 count: the old per-anchor
            # candidate vector placed its positive first, so a maximum tie was
            # credited as correct.  Scientific reports use tie-aware credit.
            correct_rows.append(positive_is_tied)
            per_record_counts: dict[int, int] = {}
            for batch_index, _ in members:
                per_record_counts[batch_index] = per_record_counts.get(batch_index, 0) + 1
            for batch_index, _ in members:
                uniform_loss += math.log(1 + size - per_record_counts[batch_index])

        if not losses:
            zero = (queries.sum() + states.sum()) * 0.0
            return MotifStateMatchingOutput(zero, 0, 0, 0)
        loss_rows = torch.cat(losses)
        correct = int(torch.cat(correct_rows).sum().cpu().item())
        tie_aware_credit = float(torch.cat(tie_credit_rows).sum().cpu().item())
        positive_probability = float(
            torch.cat(positive_probability_rows).sum().cpu().item()
        )
        return MotifStateMatchingOutput(
            loss=loss_rows.mean(),
            eligible_anchors=int(loss_rows.numel()),
            correct_anchors=correct,
            eligible_identity_groups=eligible_groups,
            tie_aware_top1_credit_sum=tie_aware_credit,
            positive_probability_sum=positive_probability,
            uniform_loss_sum=uniform_loss,
        )


__all__ = [
    "MATCHING_ID",
    "MotifStateMatchingError",
    "MotifStateMatchingHeadV3",
    "MotifStateMatchingOutput",
]
