"""Production-record collator for the factorized grammar/state views.

The historical PF-1 collator remains unchanged.  This module retains the
motif/span ownership arrays that the state-memory adapter needs and makes the
three views explicit instead of combining independent masks in one example.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol, Sequence

import torch
from torch import Tensor

from most_t5_next.p1.level_aware_motif_state_v1 import (
    build_masked_e3fp_state_batch,
)
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    collate_production_motif_record,
)


VIEW_COLLATOR_ID = "most-t5-p2/factorized-motif-view-collator/v1"
VIEW_IDS = ("grammar", "state", "cross_view")


class FactorizedViewCollatorError(ValueError):
    """A production record cannot form the requested factorized view."""


class AtomStateProvider(Protocol):
    """Read one already-aligned four-slot categorical atom-state matrix."""

    state_kind: str

    def get(self, record_id: str) -> Sequence[Sequence[int]]: ...


@dataclass(frozen=True)
class FactorizedMotifViewBatch:
    record_ids: tuple[str, ...]
    objective_mode: str
    state_kind: str
    e3fp_mask_token_id: int
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor | None
    e3fp_input_ids: Tensor
    atom_mask: Tensor
    atom_to_motif: Tensor
    motif_mask: Tensor
    motif_to_carrier: Tensor
    identity_span_bounds: Tensor
    atom_is_attachment: Tensor
    state_target_ids: Tensor | None
    state_target_mask: Tensor | None
    state_corruption_mask: Tensor | None

    def model_inputs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "objective_mode": self.objective_mode,
            "e3fp_mask_token_id": self.e3fp_mask_token_id,
            "e3fp_input_ids": self.e3fp_input_ids,
            "atom_mask": self.atom_mask,
            "atom_to_motif": self.atom_to_motif,
            "motif_mask": self.motif_mask,
            "motif_to_carrier": self.motif_to_carrier,
            "identity_span_bounds": self.identity_span_bounds,
            "atom_is_attachment": self.atom_is_attachment,
        }
        for key in (
            "labels",
            "state_target_ids",
            "state_target_mask",
            "state_corruption_mask",
        ):
            value = getattr(self, key)
            if value is not None:
                values[key] = value
        return values


def _record_seed(seed: int, epoch: int, record_id: str) -> int:
    payload = f"{int(seed)}\n{int(epoch)}\n{record_id}\nstate".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _pad_rows(
    rows: Sequence[Sequence[int]],
    *,
    width: int,
    pad_value: int,
) -> list[list[int]]:
    return [list(row) + [pad_value] * (width - len(row)) for row in rows]


def collate_factorized_motif_view(
    records: Sequence[ProductionMotifRecord],
    *,
    tokenizer: ProductionTokenizerRuntime,
    objective_mode: str,
    seed: int,
    epoch: int,
    identity_mask_probability: float = 0.15,
    state_mask_probability: float = 0.15,
    state_masking_strategy: str = "motif_atom_row",
    num_e3fp_embeddings: int = 4096,
    atom_state_provider: AtomStateProvider | None = None,
    device: object | None = None,
) -> FactorizedMotifViewBatch:
    """Collate one explicit grammar, state, or later cross-view batch.

    Grammar masks GraphPorts identity and hides every populated E3FP shell for
    atoms owned by each selected motif.  State keeps the original identity and
    selects at most one atom row per sampled motif, avoiding indistinguishable
    same-role masked atoms.  Cross-view masks identity while preserving aligned
    state; it is intentionally a separate, later diagnostic.
    """

    rows = tuple(records)
    if not rows:
        raise FactorizedViewCollatorError("records cannot be empty")
    if objective_mode not in VIEW_IDS:
        raise FactorizedViewCollatorError(
            "objective_mode must be grammar, state or cross_view"
        )
    if any(not isinstance(row, ProductionMotifRecord) for row in rows):
        raise FactorizedViewCollatorError(
            "every row must be a validated ProductionMotifRecord"
        )
    if isinstance(num_e3fp_embeddings, bool) or int(num_e3fp_embeddings) <= 1:
        raise FactorizedViewCollatorError(
            "num_e3fp_embeddings must exceed one"
        )
    if state_masking_strategy != "motif_atom_row":
        raise FactorizedViewCollatorError(
            "formal state masking must select at most one atom row per motif"
        )
    mask_token_id = int(num_e3fp_embeddings) + 1
    state_kind = "e3fp" if atom_state_provider is None else str(atom_state_provider.state_kind)
    if not state_kind:
        raise FactorizedViewCollatorError("atom state provider kind must be nonempty")

    input_rows: list[tuple[int, ...]] = []
    label_rows: list[tuple[int, ...]] = []
    spans_by_row: list[tuple[tuple[int, int], ...]] = []
    carriers_by_row: list[tuple[int, ...]] = []
    e3fp_by_row: list[Tensor] = []
    state_targets: list[Tensor] = []
    state_target_masks: list[Tensor] = []
    state_corruption_masks: list[Tensor] = []

    for record in rows:
        if (
            record.tokenizer_contract_sha256
            != tokenizer.tokenizer_contract_sha256
            or record.tokenizer_snapshot_sha256
            != tokenizer.tokenizer_snapshot_sha256
        ):
            raise FactorizedViewCollatorError(
                "production record and tokenizer binding differ"
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < tokenizer.vocab_size
            for token_id in record.input_ids
        ):
            raise FactorizedViewCollatorError(
                "production input ID is outside the tokenizer vocabulary"
            )
        if tokenizer.pad_token_id in record.input_ids:
            raise FactorizedViewCollatorError(
                "unpadded production input cannot contain the pad token"
            )
        source_state = (
            record.full_e3fp_ids
            if atom_state_provider is None
            else atom_state_provider.get(record.record_id)
        )
        try:
            source_rows = tuple(tuple(row) for row in source_state)
        except TypeError as exc:
            raise FactorizedViewCollatorError(
                "atom state provider must return rectangular integer rows"
            ) from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for row in source_rows
            for value in row
        ):
            raise FactorizedViewCollatorError(
                "atom state provider values must be discrete integer IDs"
            )
        base_e3fp = torch.as_tensor(source_rows, dtype=torch.long)
        atom_valid = torch.as_tensor(record.atom_valid_mask, dtype=torch.bool)
        atom_to_motif = torch.as_tensor(
            record.atom_to_logical_motif,
            dtype=torch.long,
        )
        if base_e3fp.ndim != 2 or base_e3fp.shape[1] != 4:
            raise FactorizedViewCollatorError(
                "production E3FP rows must have four levels"
            )
        if base_e3fp.shape[0] != len(record.full_e3fp_ids):
            raise FactorizedViewCollatorError(
                "atom state provider rows differ from the frozen model atom axis"
            )
        populated_state = base_e3fp[base_e3fp >= 0]
        if populated_state.numel() and int(populated_state.max()) >= int(
            num_e3fp_embeddings
        ):
            raise FactorizedViewCollatorError(
                "atom state provider ID lies outside the declared state domain"
            )
        if atom_valid.shape != base_e3fp.shape[:1] or atom_to_motif.shape != atom_valid.shape:
            raise FactorizedViewCollatorError(
                "production atom arrays disagree"
            )

        if objective_mode == "state":
            input_rows.append(record.input_ids)
            spans_by_row.append(
                tuple((span.start, span.stop) for span in record.identity_spans)
            )
            carriers_by_row.append(record.logical_to_carrier)
            masked = build_masked_e3fp_state_batch(
                base_e3fp.unsqueeze(0),
                atom_valid.unsqueeze(0),
                mask_token_id=mask_token_id,
                probability=float(state_mask_probability),
                seed=_record_seed(seed, epoch, record.record_id),
                target_levels=(1, 2),
                masking_strategy=state_masking_strategy,
                atom_to_group=atom_to_motif.unsqueeze(0),
            )
            e3fp_by_row.append(masked.corrupted_ids.squeeze(0))
            state_targets.append(masked.target_ids.squeeze(0))
            state_target_masks.append(masked.target_mask.squeeze(0))
            state_corruption_masks.append(masked.corruption_mask.squeeze(0))
            continue

        example = collate_production_motif_record(
            record,
            tokenizer=tokenizer,
            seed=seed,
            epoch=epoch,
            mask_probability=float(identity_mask_probability),
        )
        input_rows.append(example.input_ids)
        label_rows.append(example.labels)
        spans_by_row.append(
            tuple((span.start, span.stop) for span in example.identity_input_spans)
        )
        carriers_by_row.append(example.logical_to_carrier)
        view_e3fp = base_e3fp.clone()
        if objective_mode == "grammar":
            selected = torch.as_tensor(
                example.identity_recovery_mask,
                dtype=torch.bool,
            )
            owned_by_selected = selected[atom_to_motif]
            # Grammar must not recover a masked identity through any retained
            # recursive E3FP shell.  L0/L3 are ignored by adapter-v1, but the
            # view contract hides the complete populated state block so that
            # this remains true for later descriptor providers as well.
            for level in range(4):
                populated = view_e3fp[:, level] >= 0
                view_e3fp[owned_by_selected & populated, level] = mask_token_id
        e3fp_by_row.append(view_e3fp)

    input_width = max(len(row) for row in input_rows)
    atom_width = max(len(row.atom_valid_mask) for row in rows)
    motif_width = max(len(row.identity_spans) for row in rows)
    input_tensor = torch.as_tensor(
        _pad_rows(input_rows, width=input_width, pad_value=tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.as_tensor(
        [
            [1] * len(row) + [0] * (input_width - len(row))
            for row in input_rows
        ],
        dtype=torch.long,
        device=device,
    )
    labels: Tensor | None = None
    if objective_mode != "state":
        label_width = max(len(row) for row in label_rows)
        labels = torch.as_tensor(
            _pad_rows(label_rows, width=label_width, pad_value=-100),
            dtype=torch.long,
            device=device,
        )

    e3fp_input = torch.full(
        (len(rows), atom_width, 4),
        -1,
        dtype=torch.long,
        device=device,
    )
    atom_mask = torch.zeros((len(rows), atom_width), dtype=torch.bool, device=device)
    atom_to_motif = torch.full(
        (len(rows), atom_width),
        -1,
        dtype=torch.long,
        device=device,
    )
    atom_is_attachment = torch.zeros(
        (len(rows), atom_width),
        dtype=torch.bool,
        device=device,
    )
    motif_mask = torch.zeros((len(rows), motif_width), dtype=torch.bool, device=device)
    motif_to_carrier = torch.full(
        (len(rows), motif_width),
        -1,
        dtype=torch.long,
        device=device,
    )
    span_bounds = torch.full(
        (len(rows), motif_width, 2),
        -1,
        dtype=torch.long,
        device=device,
    )
    for batch_index, record in enumerate(rows):
        atoms = len(record.atom_valid_mask)
        motifs = len(record.identity_spans)
        e3fp_input[batch_index, :atoms] = e3fp_by_row[batch_index].to(device=device)
        atom_mask[batch_index, :atoms] = torch.as_tensor(
            record.atom_valid_mask,
            dtype=torch.bool,
            device=device,
        )
        atom_to_motif[batch_index, :atoms] = torch.as_tensor(
            record.atom_to_logical_motif,
            dtype=torch.long,
            device=device,
        )
        if record.atom_is_attachment:
            atom_is_attachment[batch_index, :atoms] = torch.as_tensor(
                record.atom_is_attachment,
                dtype=torch.bool,
                device=device,
            )
        motif_mask[batch_index, :motifs] = True
        motif_to_carrier[batch_index, :motifs] = torch.as_tensor(
            carriers_by_row[batch_index],
            dtype=torch.long,
            device=device,
        )
        span_bounds[batch_index, :motifs] = torch.as_tensor(
            spans_by_row[batch_index],
            dtype=torch.long,
            device=device,
        )

    target_ids: Tensor | None = None
    target_mask: Tensor | None = None
    corruption_mask: Tensor | None = None
    if objective_mode == "state":
        target_ids = torch.full_like(e3fp_input, -1)
        target_mask = torch.zeros_like(e3fp_input, dtype=torch.bool)
        corruption_mask = torch.zeros_like(e3fp_input, dtype=torch.bool)
        for batch_index, record in enumerate(rows):
            atoms = len(record.atom_valid_mask)
            target_ids[batch_index, :atoms] = state_targets[batch_index].to(device=device)
            target_mask[batch_index, :atoms] = state_target_masks[batch_index].to(device=device)
            corruption_mask[batch_index, :atoms] = state_corruption_masks[batch_index].to(device=device)

    return FactorizedMotifViewBatch(
        record_ids=tuple(row.record_id for row in rows),
        objective_mode=objective_mode,
        state_kind=state_kind,
        e3fp_mask_token_id=mask_token_id,
        input_ids=input_tensor,
        attention_mask=attention,
        labels=labels,
        e3fp_input_ids=e3fp_input,
        atom_mask=atom_mask,
        atom_to_motif=atom_to_motif,
        motif_mask=motif_mask,
        motif_to_carrier=motif_to_carrier,
        identity_span_bounds=span_bounds,
        atom_is_attachment=atom_is_attachment,
        state_target_ids=target_ids,
        state_target_mask=target_mask,
        state_corruption_mask=corruption_mask,
    )


__all__ = [
    "VIEW_COLLATOR_ID",
    "VIEW_IDS",
    "AtomStateProvider",
    "FactorizedMotifViewBatch",
    "FactorizedViewCollatorError",
    "collate_factorized_motif_view",
]
