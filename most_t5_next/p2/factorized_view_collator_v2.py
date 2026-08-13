"""V2 factorized collator with GraphPorts canonical atom addresses.

All identity/state corruption remains owned by the validated V1 collator.  V2
adds exactly one non-geometric tensor: for every model-atom row, the zero-based
canonical-local atom ID inside its logical motif.  The mapping is read from the
already published GraphPorts donor-planning sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
from torch import Tensor

from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)
from most_t5_next.r1.adapter.graphports_donor_atom_map_sidecar_v1 import (
    iter_release_rows,
)

from .factorized_view_collator_v1 import (
    AtomStateProvider,
    FactorizedMotifViewBatch,
    FactorizedViewCollatorError,
    collate_factorized_motif_view,
)


VIEW_COLLATOR_ID = "most-t5-p2/factorized-motif-view-collator/v2-canonical-address"


class CanonicalAtomAddressProvider(Protocol):
    """Return canonical-local positions on one record's model-atom axis."""

    def get(self, record: ProductionMotifRecord) -> Sequence[int]: ...


@dataclass(frozen=True)
class _AddressEntry:
    storage_key: str
    owners: tuple[int, ...]
    positions: tuple[int, ...]


class GraphPortsCanonicalAtomAddressProvider:
    """In-memory read-only index over a validated donor-map JSONL artifact."""

    def __init__(
        self,
        path: Path,
        *,
        required_record_ids: Sequence[str] | None = None,
    ) -> None:
        required = None
        if required_record_ids is not None:
            required = set(required_record_ids)
            if not required or len(required) != len(tuple(required_record_ids)):
                raise FactorizedViewCollatorError(
                    "required canonical-address record IDs must be unique and nonempty"
                )
        remaining = set(required) if required is not None else None
        entries: dict[str, _AddressEntry] = {}
        for row in iter_release_rows(Path(path)):
            record_id = str(row["member_id"])
            if required is not None and record_id not in required:
                continue
            if record_id in entries:
                raise FactorizedViewCollatorError(
                    "canonical atom-address sidecar repeats a record"
                )
            sidecar = row["overlay_planning_sidecar"]
            maps = sidecar["canonical_local_atom_to_model_atom"]  # type: ignore[index]
            atom_count = sum(len(atom_map) for atom_map in maps)
            positions = [-1] * atom_count
            owners = [-1] * atom_count
            for motif_id, atom_map in enumerate(maps):
                for local_id, model_atom in atom_map:
                    positions[int(model_atom)] = int(local_id) - 1
                    owners[int(model_atom)] = motif_id
            if -1 in positions or -1 in owners:
                raise FactorizedViewCollatorError(
                    "canonical atom-address sidecar does not cover the model axis"
                )
            entries[record_id] = _AddressEntry(
                storage_key=str(row["storage_key"]),
                owners=tuple(owners),
                positions=tuple(positions),
            )
            if remaining is not None:
                remaining.discard(record_id)
                if not remaining:
                    break
        if required is not None and set(entries) != required:
            raise FactorizedViewCollatorError(
                "canonical atom-address sidecar lacks required records"
            )
        if not entries:
            raise FactorizedViewCollatorError(
                "canonical atom-address sidecar is empty"
            )
        self._entries = entries

    @property
    def record_count(self) -> int:
        return len(self._entries)

    def get(self, record: ProductionMotifRecord) -> tuple[int, ...]:
        entry = self._entries.get(record.record_id)
        if entry is None:
            raise FactorizedViewCollatorError(
                "record is absent from the canonical atom-address sidecar"
            )
        if entry.storage_key != record.storage_key:
            raise FactorizedViewCollatorError(
                "record and canonical atom-address storage keys differ"
            )
        if entry.owners != tuple(record.atom_to_logical_motif):
            raise FactorizedViewCollatorError(
                "record motif ownership differs from its canonical atom address"
            )
        return entry.positions


@dataclass(frozen=True)
class FactorizedMotifViewBatchV2:
    """A V1 view plus the canonical-local address tensor required by V2."""

    base: FactorizedMotifViewBatch
    atom_local_positions: Tensor

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def model_inputs(self) -> dict[str, object]:
        values = self.base.model_inputs()
        values["atom_local_positions"] = self.atom_local_positions
        return values


def collate_factorized_motif_view_v2(
    records: Sequence[ProductionMotifRecord],
    *,
    tokenizer: ProductionTokenizerRuntime,
    objective_mode: str,
    seed: int,
    epoch: int,
    atom_address_provider: CanonicalAtomAddressProvider,
    identity_mask_probability: float = 0.15,
    state_mask_probability: float = 0.15,
    state_masking_strategy: str = "motif_atom_row",
    num_e3fp_embeddings: int = 4096,
    atom_state_provider: AtomStateProvider | None = None,
    device: object | None = None,
) -> FactorizedMotifViewBatchV2:
    """Build a V1-compatible view and attach canonical atom addresses."""

    rows = tuple(records)
    if not rows:
        raise FactorizedViewCollatorError("records cannot be empty")
    base = collate_factorized_motif_view(
        rows,
        tokenizer=tokenizer,
        objective_mode=objective_mode,
        seed=seed,
        epoch=epoch,
        identity_mask_probability=identity_mask_probability,
        state_mask_probability=state_mask_probability,
        state_masking_strategy=state_masking_strategy,
        num_e3fp_embeddings=num_e3fp_embeddings,
        atom_state_provider=atom_state_provider,
        device=device,
    )
    positions = torch.full_like(base.atom_to_motif, -1)
    for batch_index, record in enumerate(rows):
        try:
            row = tuple(atom_address_provider.get(record))
        except FactorizedViewCollatorError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise FactorizedViewCollatorError(
                "canonical atom-address provider rejected a record"
            ) from exc
        if len(row) != len(record.atom_valid_mask) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in row
        ):
            raise FactorizedViewCollatorError(
                "canonical atom-address row is malformed"
            )
        positions[batch_index, : len(row)] = torch.as_tensor(
            row,
            dtype=torch.long,
            device=positions.device,
        )
    return FactorizedMotifViewBatchV2(
        base=base,
        atom_local_positions=positions,
    )


__all__ = [
    "VIEW_COLLATOR_ID",
    "CanonicalAtomAddressProvider",
    "FactorizedMotifViewBatchV2",
    "GraphPortsCanonicalAtomAddressProvider",
    "collate_factorized_motif_view_v2",
]
