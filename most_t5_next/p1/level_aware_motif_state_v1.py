"""Level-aware permutation-invariant E3FP state encoder for the G1 gate.

This module is deliberately independent of T5.  It first asks whether a small
motif-set encoder can reconstruct masked categorical E3FP shell states.  Only
after that mechanism gate passes should its motif representation be bridged to
the frozen language backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


STATE_ENCODER_VERSION = "most-t5-p1/level-aware-motif-state/v1"


class MotifStateContractError(ValueError):
    pass


@dataclass(frozen=True)
class MaskedE3FPStateBatch:
    corrupted_ids: Tensor
    target_ids: Tensor
    target_mask: Tensor


@dataclass(frozen=True)
class MotifStateOutput:
    logits: Tensor
    atom_hidden: Tensor
    group_hidden: Tensor
    atom_group_context: Tensor
    atom_weights: Tensor


def build_masked_e3fp_state_batch(
    e3fp_ids: Tensor,
    atom_valid: Tensor,
    *,
    mask_token_id: int,
    probability: float,
    seed: int,
    first_geometry_level: int = 1,
    target_levels: Optional[Tuple[int, ...]] = None,
) -> MaskedE3FPStateBatch:
    """Mask populated geometry shell slots with at least one target per record."""

    if e3fp_ids.ndim != 3 or e3fp_ids.shape[-1] != 4:
        raise MotifStateContractError("e3fp_ids must have shape [batch, atoms, 4]")
    if atom_valid.shape != e3fp_ids.shape[:2] or atom_valid.dtype != torch.bool:
        raise MotifStateContractError("atom_valid must be bool [batch, atoms]")
    if not 0.0 < float(probability) <= 1.0:
        raise MotifStateContractError("probability must be in (0, 1]")
    if not 0 <= int(first_geometry_level) < 4:
        raise MotifStateContractError("first_geometry_level must be in [0, 3]")

    if target_levels is None:
        normalized_target_levels = tuple(range(int(first_geometry_level), 4))
    else:
        normalized_target_levels = tuple(int(level) for level in target_levels)
        if (
            not normalized_target_levels
            or len(set(normalized_target_levels)) != len(normalized_target_levels)
            or any(
                level < int(first_geometry_level) or level > 3
                for level in normalized_target_levels
            )
        ):
            raise MotifStateContractError(
                "target_levels must be unique levels within the permitted geometry range"
            )

    eligible = (e3fp_ids >= 0) & atom_valid.unsqueeze(-1)
    permitted = torch.zeros(4, dtype=torch.bool, device=e3fp_ids.device)
    permitted[list(normalized_target_levels)] = True
    eligible &= permitted.view(1, 1, 4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    draws = torch.rand(e3fp_ids.shape, generator=generator, device="cpu")
    target_mask = (draws.to(e3fp_ids.device) < float(probability)) & eligible
    for batch_index in range(e3fp_ids.shape[0]):
        if bool(eligible[batch_index].any()) and not bool(target_mask[batch_index].any()):
            first = eligible[batch_index].nonzero(as_tuple=False)[0]
            target_mask[batch_index, int(first[0]), int(first[1])] = True
    corrupted = e3fp_ids.clone()
    corrupted[target_mask] = int(mask_token_id)
    return MaskedE3FPStateBatch(
        corrupted_ids=corrupted,
        target_ids=e3fp_ids.clone(),
        target_mask=target_mask,
    )


class LevelAwareMotifStateEncoder(nn.Module):
    """Encode fixed shell order within atoms, then pool atoms as a set."""

    def __init__(
        self,
        *,
        num_e3fp_embeddings: int = 4096,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        pooling: str = "gated",
    ) -> None:
        super().__init__()
        if int(num_e3fp_embeddings) <= 1:
            raise MotifStateContractError("num_e3fp_embeddings must exceed one")
        if pooling not in ("deep_sets", "gated"):
            raise MotifStateContractError("pooling must be deep_sets or gated")
        self.num_e3fp_embeddings = int(num_e3fp_embeddings)
        self.padding_token_id = self.num_e3fp_embeddings
        self.mask_token_id = self.num_e3fp_embeddings + 1
        self.pooling = str(pooling)
        self.state_embedding = nn.Embedding(
            self.num_e3fp_embeddings + 2,
            int(embedding_dim),
            padding_idx=self.padding_token_id,
        )
        self.level_embedding = nn.Embedding(4, int(embedding_dim))
        self.atom_role_embedding = nn.Embedding(2, int(embedding_dim))
        self.atom_phi = nn.Sequential(
            nn.Linear(5 * int(embedding_dim) + 4, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.atom_score = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1, bias=False),
        )
        self.group_rho = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2 + int(embedding_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_e3fp_embeddings),
        )

    def _validate_inputs(
        self, e3fp_ids: Tensor, atom_valid: Tensor, atom_to_group: Tensor
    ) -> None:
        if e3fp_ids.ndim != 3 or e3fp_ids.shape[-1] != 4:
            raise MotifStateContractError("e3fp_ids must be [batch, atoms, 4]")
        if atom_valid.shape != e3fp_ids.shape[:2] or atom_valid.dtype != torch.bool:
            raise MotifStateContractError("atom_valid must be bool [batch, atoms]")
        if atom_to_group.shape != e3fp_ids.shape[:2]:
            raise MotifStateContractError("atom_to_group must be [batch, atoms]")
        valid_ids = e3fp_ids[e3fp_ids >= 0]
        if valid_ids.numel() and (
            int(valid_ids.min()) < 0 or int(valid_ids.max()) > self.mask_token_id
        ):
            raise MotifStateContractError("E3FP token ID is outside the encoder domain")
        if bool((atom_to_group[atom_valid] < 0).any()):
            raise MotifStateContractError("valid atoms require non-negative group IDs")

    def forward(
        self,
        e3fp_ids: Tensor,
        atom_valid: Tensor,
        atom_to_group: Tensor,
        *,
        num_groups: Optional[int] = None,
        atom_is_attachment: Optional[Tensor] = None,
    ) -> MotifStateOutput:
        self._validate_inputs(e3fp_ids, atom_valid, atom_to_group)
        batch_size, atom_count, _ = e3fp_ids.shape
        if atom_is_attachment is None:
            atom_is_attachment = torch.zeros_like(atom_valid)
        if atom_is_attachment.shape != atom_valid.shape or atom_is_attachment.dtype != torch.bool:
            raise MotifStateContractError(
                "atom_is_attachment must be bool [batch, atoms]"
            )
        if num_groups is None:
            present = atom_to_group[atom_valid]
            num_groups = 0 if present.numel() == 0 else int(present.max()) + 1
        if int(num_groups) <= 0:
            raise MotifStateContractError("num_groups must be positive")

        shell_valid = (e3fp_ids >= 0) & atom_valid.unsqueeze(-1)
        normalized_ids = e3fp_ids.masked_fill(e3fp_ids < 0, self.padding_token_id)
        level_ids = torch.arange(4, device=e3fp_ids.device).view(1, 1, 4)
        shell_hidden = self.state_embedding(normalized_ids) + self.level_embedding(level_ids)
        shell_hidden = shell_hidden * shell_valid.unsqueeze(-1).to(shell_hidden.dtype)
        role_hidden = self.atom_role_embedding(atom_is_attachment.to(torch.long))
        role_hidden = role_hidden * atom_valid.unsqueeze(-1).to(role_hidden.dtype)
        atom_input = torch.cat(
            [
                shell_hidden.flatten(start_dim=2),
                shell_valid.to(shell_hidden.dtype),
                role_hidden,
            ],
            dim=-1,
        )
        atom_hidden = self.atom_phi(atom_input)
        atom_hidden = atom_hidden * atom_valid.unsqueeze(-1).to(atom_hidden.dtype)

        total_groups = batch_size * int(num_groups)
        offsets = (
            torch.arange(batch_size, device=e3fp_ids.device).unsqueeze(1) * int(num_groups)
        )
        global_groups = atom_to_group.clamp(min=0) + offsets
        flat_valid = atom_valid.flatten()
        flat_groups = global_groups.flatten()[flat_valid]
        flat_hidden = atom_hidden.flatten(0, 1)[flat_valid]
        # Keep segment reductions in FP32 under BF16 autocast.  PyTorch promotes
        # exp/multiply outputs, while index_add_ requires exact dtype equality.
        aggregate_hidden = flat_hidden.float()
        counts = torch.zeros(total_groups, dtype=torch.float32, device=e3fp_ids.device)
        counts.index_add_(0, flat_groups, torch.ones_like(flat_groups, dtype=torch.float32))
        if self.pooling == "gated":
            scores = self.atom_score(flat_hidden).squeeze(-1).float()
            maxima = scores.new_full((total_groups,), -torch.inf)
            maxima.scatter_reduce_(
                0, flat_groups, scores, reduce="amax", include_self=True
            )
            exponentials = torch.exp(scores - maxima[flat_groups])
            denominators = scores.new_zeros(total_groups)
            denominators.index_add_(0, flat_groups, exponentials)
            flat_weights = exponentials / denominators[flat_groups]
        else:
            flat_weights = counts[flat_groups].reciprocal()
        pooled = torch.zeros(
            total_groups,
            atom_hidden.shape[-1],
            dtype=torch.float32,
            device=e3fp_ids.device,
        )
        pooled.index_add_(0, flat_groups, flat_weights.unsqueeze(-1) * aggregate_hidden)
        group_hidden = self.group_rho(pooled.to(atom_hidden.dtype)).view(
            batch_size, int(num_groups), atom_hidden.shape[-1]
        )
        group_present = (counts > 0).view(batch_size, int(num_groups), 1)
        group_hidden = group_hidden * group_present.to(group_hidden.dtype)
        atom_weights = torch.zeros(
            batch_size * atom_count, dtype=torch.float32, device=e3fp_ids.device
        )
        atom_weights[flat_valid] = flat_weights
        atom_weights = atom_weights.view(batch_size, atom_count)

        safe_groups = atom_to_group.clamp(min=0, max=int(num_groups) - 1)
        gather_index = safe_groups.unsqueeze(-1).expand(-1, -1, group_hidden.shape[-1])
        atom_group_context = torch.gather(group_hidden, 1, gather_index)
        atom_group_context = atom_group_context * atom_valid.unsqueeze(-1).to(
            atom_group_context.dtype
        )
        decoder_input = torch.cat(
            [
                atom_hidden.unsqueeze(2).expand(-1, -1, 4, -1),
                atom_group_context.unsqueeze(2).expand(-1, -1, 4, -1),
                self.level_embedding(level_ids).expand(batch_size, atom_count, -1, -1),
            ],
            dim=-1,
        )
        logits = self.state_decoder(decoder_input)
        return MotifStateOutput(
            logits=logits,
            atom_hidden=atom_hidden,
            group_hidden=group_hidden,
            atom_group_context=atom_group_context,
            atom_weights=atom_weights,
        )


def masked_state_ce(
    logits: Tensor, target_ids: Tensor, target_mask: Tensor
) -> Tuple[Tensor, Dict[int, Dict[str, Tensor]]]:
    """Token-weighted categorical loss plus level-wise sufficient statistics."""

    if logits.shape[:-1] != target_ids.shape or target_ids.shape != target_mask.shape:
        raise MotifStateContractError("logits, targets and target_mask shapes disagree")
    if target_mask.dtype != torch.bool or not bool(target_mask.any()):
        raise MotifStateContractError("target_mask must select at least one slot")
    selected_logits = logits[target_mask]
    selected_targets = target_ids[target_mask]
    loss = F.cross_entropy(selected_logits, selected_targets, reduction="mean")
    metrics: Dict[int, Dict[str, Tensor]] = {}
    predicted = logits.argmax(dim=-1)
    for level in range(target_ids.shape[-1]):
        level_mask = target_mask[..., level]
        count = level_mask.sum()
        if int(count) == 0:
            continue
        level_losses = F.cross_entropy(
            logits[..., level, :][level_mask], target_ids[..., level][level_mask], reduction="sum"
        )
        correct = (predicted[..., level][level_mask] == target_ids[..., level][level_mask]).sum()
        metrics[level] = {"nll_sum": level_losses, "correct": correct, "count": count}
    return loss, metrics
