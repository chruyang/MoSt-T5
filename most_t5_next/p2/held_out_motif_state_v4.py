"""Leakage-free held-out atom-state objective for 3D-MotifT5 V4.

For every eligible motif, exactly one atom has its consumed L1/L2 state
replaced by the established E3FP mask token.  At least one other atom in the
same motif must retain a real L1/L2 state.  The post-T5 motif carrier is then
matched against the detached original memory of the held-out atom, using only
same-identity, cross-record candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .motif_state_matching_v3 import MotifStateMatchingOutput


HELD_OUT_MATCHING_ID = (
    "most-t5-p2/held-out-motif-atom-state/v4-one-atom-visible-peer-detached-target"
)


class HeldOutMotifStateError(ValueError):
    """The held-out atom/state/identity domains are inconsistent."""


@dataclass(frozen=True)
class HeldOutMotifStatePlan:
    corrupted_e3fp_input_ids: Tensor
    target_atom_indices: Tensor
    target_local_positions: Tensor
    target_motif_mask: Tensor
    visible_peer_counts: Tensor

    @property
    def selected_targets(self) -> int:
        return int(self.target_motif_mask.sum().item())

    def to(self, device: object) -> "HeldOutMotifStatePlan":
        return HeldOutMotifStatePlan(
            corrupted_e3fp_input_ids=self.corrupted_e3fp_input_ids.to(device),
            target_atom_indices=self.target_atom_indices.to(device),
            target_local_positions=self.target_local_positions.to(device),
            target_motif_mask=self.target_motif_mask.to(device),
            visible_peer_counts=self.visible_peer_counts.to(device),
        )


def _choice_index(
    *,
    seed: int,
    epoch: int,
    identity: str,
    width: int,
) -> int:
    payload = (
        f"{seed}\0{epoch}\0{identity}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % width


def build_held_out_motif_state_plan(
    *,
    e3fp_input_ids: Tensor,
    atom_mask: Tensor,
    atom_to_motif: Tensor,
    atom_local_positions: Tensor,
    motif_mask: Tensor,
    record_ids: Sequence[str],
    exact_identity_sha256: Sequence[Sequence[str]],
    mask_token_id: int,
    seed: int,
    epoch: int,
) -> HeldOutMotifStatePlan:
    """Select one target atom per eligible motif without exposing its L1/L2."""

    if (
        e3fp_input_ids.ndim != 3
        or e3fp_input_ids.shape[-1] != 4
        or atom_mask.shape != e3fp_input_ids.shape[:2]
        or atom_to_motif.shape != atom_mask.shape
        or atom_local_positions.shape != atom_mask.shape
        or motif_mask.ndim != 2
        or motif_mask.shape[0] != atom_mask.shape[0]
        or atom_mask.dtype != torch.bool
        or motif_mask.dtype != torch.bool
    ):
        raise HeldOutMotifStateError("held-out tensors have incompatible shapes")
    if (
        e3fp_input_ids.is_floating_point()
        or atom_to_motif.is_floating_point()
        or atom_local_positions.is_floating_point()
    ):
        raise HeldOutMotifStateError("state IDs and ownership must be integers")
    if len(record_ids) != atom_mask.shape[0] or len(exact_identity_sha256) != atom_mask.shape[0]:
        raise HeldOutMotifStateError("record/identity rows differ from batch size")
    if (
        isinstance(mask_token_id, bool)
        or not isinstance(mask_token_id, int)
        or mask_token_id <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise HeldOutMotifStateError("held-out seed/epoch/mask token is invalid")
    if bool(((~atom_mask) & (atom_to_motif != -1)).any()):
        raise HeldOutMotifStateError("padded atoms must have owner -1")

    batch_size, _atom_width = atom_mask.shape
    motif_width = motif_mask.shape[1]
    corrupted = e3fp_input_ids.clone()
    targets = torch.full(
        (batch_size, motif_width),
        -1,
        dtype=torch.long,
        device=e3fp_input_ids.device,
    )
    target_positions = torch.full_like(targets, -1)
    selected = torch.zeros_like(motif_mask)
    visible_peers = torch.zeros_like(targets)
    real_joint = (
        atom_mask
        & (e3fp_input_ids[..., 1] >= 0)
        & (e3fp_input_ids[..., 1] < mask_token_id)
        & (e3fp_input_ids[..., 2] >= 0)
        & (e3fp_input_ids[..., 2] < mask_token_id)
    )

    for row in range(batch_size):
        active_count = int(motif_mask[row].sum().item())
        if active_count and not bool(motif_mask[row, :active_count].all()):
            raise HeldOutMotifStateError("active motifs must occupy a dense prefix")
        identities = tuple(exact_identity_sha256[row])
        if len(identities) != active_count:
            raise HeldOutMotifStateError("identity count differs from active motifs")
        if not isinstance(record_ids[row], str) or not record_ids[row]:
            raise HeldOutMotifStateError("record IDs must be non-empty text")
        for motif_index, identity in enumerate(identities):
            if not isinstance(identity, str) or len(identity) != 64:
                raise HeldOutMotifStateError("motif identity is not SHA-256")
            candidates = torch.nonzero(
                real_joint[row] & (atom_to_motif[row] == motif_index),
                as_tuple=False,
            ).flatten()
            if candidates.numel() < 2:
                continue
            local_positions = atom_local_positions[row, candidates]
            if bool((local_positions < 0).any()) or int(local_positions.unique().numel()) != int(candidates.numel()):
                raise HeldOutMotifStateError(
                    "eligible motif atoms need distinct canonical local positions"
                )
            candidates = candidates[torch.argsort(local_positions)]
            offset = _choice_index(
                seed=seed,
                epoch=epoch,
                identity=identity,
                width=int(candidates.numel()),
            )
            atom_index = int(candidates[offset].item())
            targets[row, motif_index] = atom_index
            target_positions[row, motif_index] = int(
                atom_local_positions[row, atom_index].item()
            )
            selected[row, motif_index] = True
            visible_peers[row, motif_index] = int(candidates.numel()) - 1
            corrupted[row, atom_index, 1] = mask_token_id
            corrupted[row, atom_index, 2] = mask_token_id

    if not bool(selected.any()):
        raise HeldOutMotifStateError("batch has no motif with a visible state peer")
    if bool((visible_peers[selected] < 1).any()):
        raise HeldOutMotifStateError("held-out motif lacks a visible state peer")
    return HeldOutMotifStatePlan(
        corrupted, targets, target_positions, selected, visible_peers
    )


class HeldOutAtomStateMatchingHeadV4(nn.Module):
    """Match post-T5 carriers to detached held-out atom memories."""

    def __init__(
        self,
        *,
        hidden_size: int,
        atom_memory_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if (
            min(hidden_size, atom_memory_dim, projection_dim) <= 0
            or float(temperature) <= 0.0
        ):
            raise HeldOutMotifStateError("held-out head dimensions are invalid")
        self.hidden_size = int(hidden_size)
        self.atom_memory_dim = int(atom_memory_dim)
        self.projection_dim = int(projection_dim)
        self.temperature = float(temperature)
        self.query_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.target_projection = nn.Linear(atom_memory_dim, projection_dim, bias=False)

    def forward(
        self,
        *,
        encoder_hidden: Tensor,
        original_atom_memory: Tensor,
        motif_mask: Tensor,
        motif_to_carrier: Tensor,
        target_atom_indices: Tensor,
        target_local_positions: Tensor,
        target_motif_mask: Tensor,
        exact_identity_sha256: Sequence[Sequence[str]],
    ) -> MotifStateMatchingOutput:
        if (
            encoder_hidden.ndim != 3
            or original_atom_memory.ndim != 3
            or encoder_hidden.shape[0] != original_atom_memory.shape[0]
            or encoder_hidden.shape[2] != self.hidden_size
            or original_atom_memory.shape[2] != self.atom_memory_dim
            or motif_mask.ndim != 2
            or motif_to_carrier.shape != motif_mask.shape
            or target_atom_indices.shape != motif_mask.shape
            or target_local_positions.shape != motif_mask.shape
            or target_motif_mask.shape != motif_mask.shape
            or motif_mask.dtype != torch.bool
            or target_motif_mask.dtype != torch.bool
        ):
            raise HeldOutMotifStateError("held-out matching tensors are inconsistent")
        if bool((target_motif_mask & ~motif_mask).any()):
            raise HeldOutMotifStateError("held-out targets must be active motifs")
        if bool(((~target_motif_mask) & (target_atom_indices != -1)).any()):
            raise HeldOutMotifStateError("non-target motifs must use atom index -1")
        if bool(((~target_motif_mask) & (target_local_positions != -1)).any()):
            raise HeldOutMotifStateError("non-target motifs must use local position -1")
        if bool((target_local_positions[target_motif_mask] < 0).any()):
            raise HeldOutMotifStateError("target local positions must be non-negative")
        if len(exact_identity_sha256) != motif_mask.shape[0]:
            raise HeldOutMotifStateError("identity rows differ from batch size")

        safe_carriers = motif_to_carrier.clamp_min(0).to(torch.long)
        safe_targets = target_atom_indices.clamp_min(0).to(torch.long)
        if bool((safe_carriers[motif_mask] >= encoder_hidden.shape[1]).any()):
            raise HeldOutMotifStateError("motif carrier lies outside token width")
        if bool((safe_targets[target_motif_mask] >= original_atom_memory.shape[1]).any()):
            raise HeldOutMotifStateError("held-out atom lies outside atom width")
        carriers = encoder_hidden.gather(
            1, safe_carriers.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        atom_targets = original_atom_memory.detach().gather(
            1, safe_targets.unsqueeze(-1).expand(-1, -1, self.atom_memory_dim)
        )
        queries = F.normalize(self.query_projection(carriers), dim=-1)
        targets = F.normalize(self.target_projection(atom_targets), dim=-1)

        active: list[tuple[int, int, str, int]] = []
        for row, identities in enumerate(exact_identity_sha256):
            active_count = int(motif_mask[row].sum().item())
            if len(identities) != active_count:
                raise HeldOutMotifStateError("identity count differs from active motifs")
            for motif_index, identity in enumerate(identities):
                if target_motif_mask[row, motif_index]:
                    active.append((
                        row,
                        motif_index,
                        identity,
                        int(target_local_positions[row, motif_index].item()),
                    ))

        by_identity: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for row, motif_index, identity, local_position in active:
            by_identity.setdefault((identity, local_position), []).append(
                (row, motif_index)
            )

        losses: list[Tensor] = []
        correct_rows: list[Tensor] = []
        tie_rows: list[Tensor] = []
        probability_rows: list[Tensor] = []
        uniform_loss = 0.0
        eligible_groups = 0
        for members in by_identity.values():
            if len({row for row, _ in members}) < 2:
                continue
            eligible_groups += 1
            rows = torch.as_tensor(
                [row for row, _ in members], device=queries.device, dtype=torch.long
            )
            motifs = torch.as_tensor(
                [motif for _, motif in members], device=queries.device, dtype=torch.long
            )
            logits = torch.matmul(
                queries[rows, motifs], targets[rows, motifs].transpose(0, 1)
            ) / self.temperature
            size = len(members)
            diagonal = torch.eye(size, dtype=torch.bool, device=queries.device)
            candidates = (rows[:, None] != rows[None, :]) | diagonal
            masked_logits = logits.masked_fill(~candidates, float("-inf"))
            labels = torch.arange(size, device=queries.device)
            losses.append(F.cross_entropy(masked_logits, labels, reduction="none"))

            detached = masked_logits.detach().float()
            probabilities = torch.softmax(detached, dim=1)
            probability_rows.append(probabilities.diagonal())
            maxima = detached.max(dim=1, keepdim=True).values
            tied = detached == maxima
            positive_tied = tied.diagonal()
            correct_rows.append(positive_tied)
            tie_rows.append(
                positive_tied.float() / tied.sum(dim=1).clamp_min(1).float()
            )
            per_record: dict[int, int] = {}
            for row, _ in members:
                per_record[row] = per_record.get(row, 0) + 1
            for row, _ in members:
                uniform_loss += math.log(1 + size - per_record[row])

        if not losses:
            zero = (queries.sum() + targets.sum()) * 0.0
            return MotifStateMatchingOutput(zero, 0, 0, 0)
        loss_rows = torch.cat(losses)
        return MotifStateMatchingOutput(
            loss=loss_rows.mean(),
            eligible_anchors=int(loss_rows.numel()),
            correct_anchors=int(torch.cat(correct_rows).sum().cpu().item()),
            eligible_identity_groups=eligible_groups,
            tie_aware_top1_credit_sum=float(torch.cat(tie_rows).sum().cpu().item()),
            positive_probability_sum=float(
                torch.cat(probability_rows).sum().cpu().item()
            ),
            uniform_loss_sum=uniform_loss,
        )


def eligible_held_out_anchor_count(
    target_motif_mask: Tensor,
    target_local_positions: Tensor,
    exact_identity_sha256: Sequence[Sequence[str]],
) -> int:
    """Count anchors whose microbatch has a cross-record identity peer."""

    if target_motif_mask.ndim != 2 or target_motif_mask.dtype != torch.bool:
        raise HeldOutMotifStateError("target_motif_mask must be bool [B,M]")
    if target_local_positions.shape != target_motif_mask.shape:
        raise HeldOutMotifStateError("target local positions differ from motif mask")
    if len(exact_identity_sha256) != target_motif_mask.shape[0]:
        raise HeldOutMotifStateError("identity rows differ from batch size")
    grouped: dict[tuple[str, int], list[int]] = {}
    for row, identities in enumerate(exact_identity_sha256):
        if len(identities) > target_motif_mask.shape[1]:
            raise HeldOutMotifStateError("identity row exceeds motif width")
        for motif_index, identity in enumerate(identities):
            if target_motif_mask[row, motif_index]:
                position = int(target_local_positions[row, motif_index].item())
                grouped.setdefault((identity, position), []).append(row)
    return sum(
        len(rows)
        for rows in grouped.values()
        if len(set(rows)) >= 2
    )


__all__ = [
    "HELD_OUT_MATCHING_ID",
    "HeldOutAtomStateMatchingHeadV4",
    "HeldOutMotifStateError",
    "HeldOutMotifStatePlan",
    "build_held_out_motif_state_plan",
    "eligible_held_out_anchor_count",
]
