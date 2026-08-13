"""Streaming support census and S-stage membership for factorized state views.

PF-1 and PF-10 use the same paired-reader boundary; only their published
membership differs.  This module therefore consumes the reader protocol rather
than either release directory directly.  It performs one ordered pass to count
state support and freezes an ordered subsequence for the S-stage.  No decoded
record is retained by the census.

An atom is jointly targetable when it is valid and has populated E3FP levels 1
and 2.  A motif enters the S-stage only when it owns at least two such atoms,
which prevents a nominal motif-state stage from collapsing to singleton-only
support.  A record enters only when at least one of its motifs passes that gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence

from most_t5_next.p1.production_bridge import ProductionMotifRecord


FACTORISED_SUPPORT_CENSUS_ID = "most-t5-p2/factorized-support-census/v1"
S_STAGE_MIN_ELIGIBLE_ATOMS = 2


class FactorizedSupportCensusError(ValueError):
    """A paired reader or motif record cannot support a stable S membership."""


class FactorizedPairedReader(Protocol):
    """Shared ordered reader boundary implemented by PF-1 and PF-10 releases."""

    train_member_count: int
    dev_member_count: int

    def iter_train_epoch(
        self, *, epoch: int, batch_size: int
    ) -> Iterator[Sequence[Any]]: ...

    def iter_dev(self, *, batch_size: int) -> Iterator[Sequence[Any]]: ...


@dataclass(frozen=True)
class EligibleMotifSupport:
    """One S-stage motif and its exact jointly targetable atom coordinates."""

    motif_id: int
    identity_span_length: int
    level1_atom_count: int
    level2_atom_count: int
    eligible_atom_indices: tuple[int, ...]

    @property
    def eligible_atom_count(self) -> int:
        return len(self.eligible_atom_indices)


@dataclass(frozen=True)
class StateEligibleMembership:
    """One ordered reader member retained by the S-stage support gate."""

    record_id: str
    storage_key: str
    split_index: int
    eligible_motifs: tuple[EligibleMotifSupport, ...]

    @property
    def eligible_motif_ids(self) -> tuple[int, ...]:
        return tuple(row.motif_id for row in self.eligible_motifs)


@dataclass(frozen=True)
class FactorizedSupportCensus:
    """Sufficient streaming statistics plus the stable eligible subsequence."""

    schema_id: str
    split: str
    total_records: int
    total_motifs: int
    max_identity_span_length: int
    level1_atoms_per_motif_histogram: tuple[tuple[int, int], ...]
    level2_atoms_per_motif_histogram: tuple[tuple[int, int], ...]
    jointly_eligible_atoms_per_motif_histogram: tuple[tuple[int, int], ...]
    state_targetable_records: int
    state_targetable_motifs: int
    state_eligible_records: int
    state_eligible_motifs: int
    state_eligible_atoms: int
    minimum_eligible_atoms_per_motif: int
    state_eligible_membership: tuple[StateEligibleMembership, ...]


@dataclass(frozen=True)
class EligibleFactorizedRecord:
    """A decoded paired row accompanied by its frozen eligible motif subset."""

    paired_record: Any
    membership: StateEligibleMembership


def _ordered_batches(
    reader: FactorizedPairedReader,
    *,
    split: str,
    batch_size: int,
) -> tuple[Iterator[Sequence[Any]], int]:
    if split not in ("train", "dev"):
        raise FactorizedSupportCensusError("split must be train or dev")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FactorizedSupportCensusError("batch_size must be a positive integer")
    if split == "train":
        return reader.iter_train_epoch(epoch=0, batch_size=batch_size), int(
            reader.train_member_count
        )
    return reader.iter_dev(batch_size=batch_size), int(reader.dev_member_count)


def _motif_record(loaded: Any) -> ProductionMotifRecord:
    record = getattr(loaded, "motif_record", None)
    if not isinstance(record, ProductionMotifRecord):
        raise FactorizedSupportCensusError(
            "paired reader rows must expose one ProductionMotifRecord"
        )
    return record


def _increment(histogram: dict[int, int], value: int) -> None:
    histogram[int(value)] = histogram.get(int(value), 0) + 1


def _frozen_histogram(histogram: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(histogram.items()))


def census_factorized_support(
    reader: FactorizedPairedReader,
    *,
    split: str,
    batch_size: int = 1024,
) -> FactorizedSupportCensus:
    """Scan one paired split once and freeze its S-stage eligible subsequence."""

    batches, declared_records = _ordered_batches(
        reader,
        split=split,
        batch_size=batch_size,
    )
    total_records = 0
    total_motifs = 0
    max_identity_span_length = 0
    level1_histogram: dict[int, int] = {}
    level2_histogram: dict[int, int] = {}
    joint_histogram: dict[int, int] = {}
    targetable_records = 0
    targetable_motifs = 0
    eligible_motifs_total = 0
    eligible_atoms_total = 0
    membership: list[StateEligibleMembership] = []

    for batch in batches:
        rows = tuple(batch)
        if not rows:
            raise FactorizedSupportCensusError("paired reader yielded an empty batch")
        for loaded in rows:
            record = _motif_record(loaded)
            motif_count = len(record.identity_spans)
            if motif_count <= 0:
                raise FactorizedSupportCensusError("motif record has no logical motif")
            if not (
                len(record.full_e3fp_ids)
                == len(record.atom_valid_mask)
                == len(record.atom_to_logical_motif)
            ):
                raise FactorizedSupportCensusError("motif atom arrays disagree")

            spans = tuple(span.stop - span.start for span in record.identity_spans)
            if any(length <= 0 for length in spans):
                raise FactorizedSupportCensusError("identity span must be nonempty")
            max_identity_span_length = max(max_identity_span_length, max(spans))
            total_motifs += motif_count

            level1_atoms: list[list[int]] = [[] for _ in range(motif_count)]
            level2_atoms: list[list[int]] = [[] for _ in range(motif_count)]
            joint_atoms: list[list[int]] = [[] for _ in range(motif_count)]
            for atom_index, (levels, valid, motif_id) in enumerate(
                zip(
                    record.full_e3fp_ids,
                    record.atom_valid_mask,
                    record.atom_to_logical_motif,
                )
            ):
                if not valid:
                    continue
                if len(levels) != 4:
                    raise FactorizedSupportCensusError(
                        "E3FP atom rows must contain four levels"
                    )
                motif_id = int(motif_id)
                if not 0 <= motif_id < motif_count:
                    raise FactorizedSupportCensusError(
                        "valid atom maps outside the logical motif domain"
                    )
                level1_present = int(levels[1]) >= 0
                level2_present = int(levels[2]) >= 0
                if level1_present:
                    level1_atoms[motif_id].append(atom_index)
                if level2_present:
                    level2_atoms[motif_id].append(atom_index)
                if level1_present and level2_present:
                    joint_atoms[motif_id].append(atom_index)

            record_targetable = False
            record_eligible: list[EligibleMotifSupport] = []
            for motif_id in range(motif_count):
                level1_count = len(level1_atoms[motif_id])
                level2_count = len(level2_atoms[motif_id])
                joint_count = len(joint_atoms[motif_id])
                _increment(level1_histogram, level1_count)
                _increment(level2_histogram, level2_count)
                _increment(joint_histogram, joint_count)
                if joint_count >= 1:
                    targetable_motifs += 1
                    record_targetable = True
                if joint_count >= S_STAGE_MIN_ELIGIBLE_ATOMS:
                    support = EligibleMotifSupport(
                        motif_id=motif_id,
                        identity_span_length=spans[motif_id],
                        level1_atom_count=level1_count,
                        level2_atom_count=level2_count,
                        eligible_atom_indices=tuple(joint_atoms[motif_id]),
                    )
                    record_eligible.append(support)
                    eligible_motifs_total += 1
                    eligible_atoms_total += joint_count
            if record_targetable:
                targetable_records += 1
            if record_eligible:
                membership.append(
                    StateEligibleMembership(
                        record_id=record.record_id,
                        storage_key=record.storage_key,
                        split_index=total_records,
                        eligible_motifs=tuple(record_eligible),
                    )
                )
            total_records += 1

    if total_records != declared_records:
        raise FactorizedSupportCensusError(
            "paired reader count differs from its declared split membership"
        )
    return FactorizedSupportCensus(
        schema_id=FACTORISED_SUPPORT_CENSUS_ID,
        split=split,
        total_records=total_records,
        total_motifs=total_motifs,
        max_identity_span_length=max_identity_span_length,
        level1_atoms_per_motif_histogram=_frozen_histogram(level1_histogram),
        level2_atoms_per_motif_histogram=_frozen_histogram(level2_histogram),
        jointly_eligible_atoms_per_motif_histogram=_frozen_histogram(
            joint_histogram
        ),
        state_targetable_records=targetable_records,
        state_targetable_motifs=targetable_motifs,
        state_eligible_records=len(membership),
        state_eligible_motifs=eligible_motifs_total,
        state_eligible_atoms=eligible_atoms_total,
        minimum_eligible_atoms_per_motif=S_STAGE_MIN_ELIGIBLE_ATOMS,
        state_eligible_membership=tuple(membership),
    )


def iter_state_eligible_batches(
    reader: FactorizedPairedReader,
    census: FactorizedSupportCensus,
    *,
    batch_size: int,
) -> Iterator[tuple[EligibleFactorizedRecord, ...]]:
    """Replay and filter a split in its frozen membership order without shuffle."""

    if not isinstance(census, FactorizedSupportCensus):
        raise FactorizedSupportCensusError("census must be FactorizedSupportCensus")
    batches, declared_records = _ordered_batches(
        reader,
        split=census.split,
        batch_size=batch_size,
    )
    if declared_records != census.total_records:
        raise FactorizedSupportCensusError(
            "reader membership count changed after the support census"
        )
    expected = iter(census.state_eligible_membership)
    pending = next(expected, None)
    output: list[EligibleFactorizedRecord] = []
    split_index = 0
    for batch in batches:
        rows = tuple(batch)
        if not rows:
            raise FactorizedSupportCensusError("paired reader yielded an empty batch")
        for loaded in rows:
            if pending is not None and split_index == pending.split_index:
                record = _motif_record(loaded)
                if not (
                    record.record_id == pending.record_id
                    and record.storage_key == pending.storage_key
                ):
                    raise FactorizedSupportCensusError(
                        "paired reader order changed after the support census"
                    )
                output.append(
                    EligibleFactorizedRecord(
                        paired_record=loaded,
                        membership=pending,
                    )
                )
                pending = next(expected, None)
                if len(output) == batch_size:
                    yield tuple(output)
                    output.clear()
            split_index += 1
    if split_index != census.total_records or pending is not None:
        raise FactorizedSupportCensusError(
            "eligible membership was not exhausted by the paired reader"
        )
    if output:
        yield tuple(output)


__all__ = [
    "FACTORISED_SUPPORT_CENSUS_ID",
    "S_STAGE_MIN_ELIGIBLE_ATOMS",
    "EligibleFactorizedRecord",
    "EligibleMotifSupport",
    "FactorizedPairedReader",
    "FactorizedSupportCensus",
    "FactorizedSupportCensusError",
    "StateEligibleMembership",
    "census_factorized_support",
    "iter_state_eligible_batches",
]
