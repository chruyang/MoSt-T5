"""Deterministic motif-matched state corruption for the PF-10 dev gate.

This module is deliberately an in-memory planning boundary.  It consumes the
already-persisted motif training documents plus an explicit GraphPorts
canonical-local-atom planning sidecar, matches motif occurrences only across
different molecules, and can materialize a replacement ``[atom, 4]`` state
matrix without changing recipient topology or CE fields.  It never guesses an
atom correspondence from molecule-local model/SDF row indices.  A later LMDB
publisher can wrap these pure functions; no production release is rewritten.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


DONOR_PLAN_ID = "most-t5-p2/matched-motif-state-donor-plan/v1"
DONOR_ATOM_MAP_SIDECAR_SCHEMA = (
    "most-t5-p2/graphports-donor-atom-map-sidecar/v1"
)


class MatchedMotifDonorError(ValueError):
    """A motif document or donor assignment violates the fixed interface."""


@dataclass(frozen=True, order=True)
class MotifDonorKey:
    identity_sha256: str
    atom_count: int
    port_degree_by_local_atom: tuple[int, ...]
    incident_bond_types: tuple[str, ...]
    neighbor_signature: tuple[tuple[int, str, str, int], ...] = ()


@dataclass(frozen=True)
class MotifOccurrence:
    record_id: str
    motif_index: int
    model_atom_indices_by_local_id: tuple[int, ...]
    atom_port_degrees: tuple[int, ...]
    key: MotifDonorKey


@dataclass(frozen=True)
class MotifDonorAssignment:
    recipient_record_id: str
    recipient_motif_index: int
    donor_record_id: str
    donor_motif_index: int
    recipient_atom_indices: tuple[int, ...]
    donor_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class MatchedMotifDonorPlan:
    assignments: tuple[MotifDonorAssignment, ...]
    excluded_occurrences: tuple[tuple[str, int], ...]
    coverage: Mapping[str, int | float]
    strict_neighbor_match: bool


@dataclass(frozen=True)
class MatchedStateOverlay:
    state_by_record_id: Mapping[str, tuple[tuple[int, int, int, int], ...]]
    changed_motifs_by_record_id: Mapping[str, tuple[int, ...]]
    changed_state_slot_count: int


def build_graphports_donor_atom_map_sidecar(graph_encoding: object) -> dict[str, object]:
    """Serialize ``MotifRecord.source_atom_map`` for overlay planning only.

    GraphPorts v1 freezes canonical, one-based motif-local atom IDs while the
    second coordinate is the projected molecule/model atom row.  This map is
    necessary for a cross-molecule donor; equal motif identity hashes alone do
    not establish an atom isomorphism.
    """

    try:
        motifs = tuple(graph_encoding.motifs)  # type: ignore[attr-defined]
        format_version = str(graph_encoding.format_version)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise MatchedMotifDonorError("GraphPorts encoding fields are malformed") from exc
    if not motifs or tuple(int(motif.motif_id) for motif in motifs) != tuple(  # type: ignore[attr-defined]
        range(len(motifs))
    ):
        raise MatchedMotifDonorError(
            "GraphPorts motifs must be in contiguous frozen-logical order"
        )
    rows: list[list[list[int]]] = []
    for motif in motifs:
        try:
            source_map = tuple(
                (int(local_id), int(model_atom))
                for local_id, model_atom in motif.source_atom_map  # type: ignore[attr-defined]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise MatchedMotifDonorError(
                "GraphPorts motif source_atom_map is malformed"
            ) from exc
        if tuple(local_id for local_id, _model_atom in source_map) != tuple(
            range(1, len(source_map) + 1)
        ):
            raise MatchedMotifDonorError(
                "GraphPorts canonical local atom IDs must be contiguous"
            )
        if len({model_atom for _local_id, model_atom in source_map}) != len(
            source_map
        ):
            raise MatchedMotifDonorError(
                "GraphPorts source_atom_map repeats a model atom"
            )
        rows.append([[local_id, model_atom] for local_id, model_atom in source_map])
    return {
        "schema_version": DONOR_ATOM_MAP_SIDECAR_SCHEMA,
        "source_codec_format_version": format_version,
        "canonical_local_atom_to_model_atom": rows,
    }


def _document_fields(document: Mapping[str, object]):
    try:
        record_id = str(document["member"]["member_id"])  # type: ignore[index]
        motif = document["logical_motif_domain"]  # type: ignore[index]
        atom = document["atom_domain"]  # type: ignore[index]
        identities = tuple(str(value) for value in motif["exact_identity_sha256"])  # type: ignore[index]
        atom_groups = tuple(tuple(int(value) for value in row) for row in motif["motif_atom_indices"])  # type: ignore[index]
        slot_atoms = tuple(tuple(int(value) for value in row) for row in motif["motif_slot_atom_indices"])  # type: ignore[index]
        cross_bonds = tuple(motif["cross_motif_bonds"])  # type: ignore[index]
        state_rows = tuple(tuple(int(value) for value in row) for row in atom["full_e3fp_ids"])  # type: ignore[index]
        planning = document["overlay_planning_sidecar"]  # type: ignore[index]
        planning_schema = str(planning["schema_version"])  # type: ignore[index]
        raw_local_maps = tuple(planning["canonical_local_atom_to_model_atom"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise MatchedMotifDonorError("motif donor document fields are malformed") from exc
    if not record_id or not identities or not (
        len(identities) == len(atom_groups) == len(slot_atoms)
    ):
        raise MatchedMotifDonorError("motif donor document dimensions disagree")
    if not state_rows or any(len(row) != 4 for row in state_rows):
        raise MatchedMotifDonorError("state rows must have width four")
    flattened = [value for group in atom_groups for value in group]
    if sorted(flattened) != list(range(len(state_rows))):
        raise MatchedMotifDonorError("motif groups must partition the state row axis")
    if planning_schema != DONOR_ATOM_MAP_SIDECAR_SCHEMA:
        raise MatchedMotifDonorError("donor atom-map sidecar schema is unsupported")
    if len(raw_local_maps) != len(atom_groups):
        raise MatchedMotifDonorError("donor atom-map sidecar motif count disagrees")
    local_maps: list[tuple[tuple[int, int], ...]] = []
    for motif_index, (raw_map, atoms) in enumerate(
        zip(raw_local_maps, atom_groups)
    ):
        try:
            pairs = tuple(
                (int(pair[0]), int(pair[1]))  # type: ignore[index]
                for pair in raw_map
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise MatchedMotifDonorError(
                "donor atom-map sidecar row is malformed"
            ) from exc
        expected_local_ids = tuple(range(1, len(atoms) + 1))
        if tuple(local_id for local_id, _model_atom in pairs) != expected_local_ids:
            raise MatchedMotifDonorError(
                "canonical local atom IDs must be contiguous and ordered"
            )
        model_atoms = tuple(model_atom for _local_id, model_atom in pairs)
        if len(set(model_atoms)) != len(model_atoms) or set(model_atoms) != set(atoms):
            raise MatchedMotifDonorError(
                f"donor atom-map sidecar motif {motif_index} disagrees with its atom group"
            )
        local_maps.append(pairs)
    return (
        record_id,
        identities,
        atom_groups,
        slot_atoms,
        cross_bonds,
        state_rows,
        tuple(local_maps),
    )


def extract_motif_occurrences(
    document: Mapping[str, object], *, strict_neighbors: bool = False
) -> tuple[MotifOccurrence, ...]:
    """Extract identity, atom-count and port-pattern donor signatures."""

    (
        record_id,
        identities,
        atom_groups,
        slot_atoms,
        cross_bonds,
        _state,
        local_maps,
    ) = (
        _document_fields(document)
    )
    local_id_by_model_atom = tuple(
        {model_atom: local_id for local_id, model_atom in row}
        for row in local_maps
    )
    degree_by_motif: list[dict[int, int]] = []
    for motif_index, atoms in enumerate(atom_groups):
        counts = Counter(slot_atoms[motif_index])
        if any(atom_index not in atoms for atom_index in counts):
            raise MatchedMotifDonorError("motif port references an atom outside its group")
        degree_by_motif.append(
            {atom_index: int(counts.get(atom_index, 0)) for atom_index in atoms}
        )

    incident: list[list[str]] = [[] for _ in identities]
    neighbor_rows: list[list[tuple[int, str, str, int]]] = [
        [] for _ in identities
    ]
    for edge in cross_bonds:
        if not isinstance(edge, Mapping):
            raise MatchedMotifDonorError("cross-motif bond must be a mapping")
        try:
            left = edge["left"]
            right = edge["right"]
            bond_type = str(edge["bond_type"])
            left_motif = int(left["logical_motif_index"])  # type: ignore[index]
            right_motif = int(right["logical_motif_index"])  # type: ignore[index]
            left_atom = int(left["atom_index"])  # type: ignore[index]
            right_atom = int(right["atom_index"])  # type: ignore[index]
            left_degree = degree_by_motif[left_motif][left_atom]
            right_degree = degree_by_motif[right_motif][right_atom]
            left_local_id = local_id_by_model_atom[left_motif][left_atom]
            right_local_id = local_id_by_model_atom[right_motif][right_atom]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise MatchedMotifDonorError("cross-motif bond endpoint is malformed") from exc
        incident[left_motif].append(bond_type)
        incident[right_motif].append(bond_type)
        neighbor_rows[left_motif].append(
            (left_local_id, bond_type, identities[right_motif], right_local_id)
        )
        neighbor_rows[right_motif].append(
            (right_local_id, bond_type, identities[left_motif], left_local_id)
        )

    occurrences: list[MotifOccurrence] = []
    for motif_index, atoms in enumerate(atom_groups):
        model_atoms_by_local_id = tuple(
            model_atom for _local_id, model_atom in local_maps[motif_index]
        )
        degrees = tuple(
            degree_by_motif[motif_index][model_atom]
            for model_atom in model_atoms_by_local_id
        )
        key = MotifDonorKey(
            identity_sha256=identities[motif_index],
            atom_count=len(atoms),
            port_degree_by_local_atom=degrees,
            incident_bond_types=tuple(sorted(incident[motif_index])),
            neighbor_signature=(
                tuple(sorted(neighbor_rows[motif_index]))
                if strict_neighbors
                else ()
            ),
        )
        occurrences.append(
            MotifOccurrence(
                record_id=record_id,
                motif_index=motif_index,
                model_atom_indices_by_local_id=model_atoms_by_local_id,
                atom_port_degrees=degrees,
                key=key,
            )
        )
    return tuple(occurrences)


def build_matched_motif_donor_plan(
    documents: Sequence[Mapping[str, object]], *, strict_neighbors: bool = False
) -> MatchedMotifDonorPlan:
    """Match every eligible motif to the same signature in another molecule."""

    if not documents:
        raise MatchedMotifDonorError("donor planning requires at least one document")
    occurrences = tuple(
        occurrence
        for document in documents
        for occurrence in extract_motif_occurrences(
            document, strict_neighbors=strict_neighbors
        )
    )
    record_ids = [str(document["member"]["member_id"]) for document in documents]  # type: ignore[index]
    if len(record_ids) != len(set(record_ids)):
        raise MatchedMotifDonorError("donor documents repeat a record_id")
    by_key_record: dict[
        MotifDonorKey, dict[str, list[MotifOccurrence]]
    ] = defaultdict(lambda: defaultdict(list))
    for occurrence in occurrences:
        by_key_record[occurrence.key][occurrence.record_id].append(occurrence)

    assignments: list[MotifDonorAssignment] = []
    excluded: list[tuple[str, int]] = []
    donor_reuse: Counter[tuple[str, int]] = Counter()
    eligible_by_record: Counter[str] = Counter()
    total_by_record: Counter[str] = Counter(o.record_id for o in occurrences)
    eligible_atoms = 0
    for key in sorted(by_key_record):
        record_map = by_key_record[key]
        ordered_records = sorted(record_map)
        if len(ordered_records) < 2:
            excluded.extend(
                (occurrence.record_id, occurrence.motif_index)
                for occurrence in record_map[ordered_records[0]]
            )
            continue
        for record_position, recipient_record in enumerate(ordered_records):
            donor_record = ordered_records[(record_position + 1) % len(ordered_records)]
            recipients = sorted(
                record_map[recipient_record], key=lambda row: row.motif_index
            )
            donors = sorted(record_map[donor_record], key=lambda row: row.motif_index)
            for occurrence_index, recipient in enumerate(recipients):
                donor = donors[occurrence_index % len(donors)]
                recipient_order = recipient.model_atom_indices_by_local_id
                donor_order = donor.model_atom_indices_by_local_id
                if len(recipient_order) != len(donor_order):
                    raise MatchedMotifDonorError("matched donor atom counts differ")
                assignments.append(
                    MotifDonorAssignment(
                        recipient_record_id=recipient.record_id,
                        recipient_motif_index=recipient.motif_index,
                        donor_record_id=donor.record_id,
                        donor_motif_index=donor.motif_index,
                        recipient_atom_indices=recipient_order,
                        donor_atom_indices=donor_order,
                    )
                )
                donor_reuse[(donor.record_id, donor.motif_index)] += 1
                eligible_by_record[recipient.record_id] += 1
                eligible_atoms += len(recipient_order)

    total_atoms = sum(
        len(occurrence.model_atom_indices_by_local_id)
        for occurrence in occurrences
    )
    molecules_any = sum(eligible_by_record[record_id] > 0 for record_id in record_ids)
    molecules_all = sum(
        eligible_by_record[record_id] == total_by_record[record_id]
        for record_id in record_ids
    )
    eligible_motifs = len(assignments)
    coverage: dict[str, int | float] = {
        "total_records": len(record_ids),
        "total_motif_occurrences": len(occurrences),
        "eligible_motif_occurrences": eligible_motifs,
        "excluded_motif_occurrences": len(excluded),
        "motif_occurrence_coverage": eligible_motifs / len(occurrences),
        "total_atom_rows": total_atoms,
        "eligible_atom_rows": eligible_atoms,
        "atom_row_coverage": eligible_atoms / total_atoms,
        "records_with_any_eligible_motif": molecules_any,
        "records_with_all_motifs_eligible": molecules_all,
        "max_donor_reuse": max(donor_reuse.values(), default=0),
    }
    return MatchedMotifDonorPlan(
        assignments=tuple(
            sorted(
                assignments,
                key=lambda row: (
                    row.recipient_record_id,
                    row.recipient_motif_index,
                ),
            )
        ),
        excluded_occurrences=tuple(sorted(excluded)),
        coverage=coverage,
        strict_neighbor_match=bool(strict_neighbors),
    )


def materialize_matched_state_overlay(
    documents: Sequence[Mapping[str, object]], plan: MatchedMotifDonorPlan
) -> MatchedStateOverlay:
    """Copy donor state rows onto recipients while leaving documents untouched."""

    parsed = {}
    for document in documents:
        (
            record_id,
            _ids,
            _groups,
            _slots,
            _bonds,
            state_rows,
            _local_maps,
        ) = _document_fields(document)
        if record_id in parsed:
            raise MatchedMotifDonorError("overlay documents repeat a record_id")
        parsed[record_id] = state_rows
    mutable = {record_id: [list(row) for row in rows] for record_id, rows in parsed.items()}
    changed_motifs: dict[str, set[int]] = defaultdict(set)
    changed_slots = 0
    for assignment in plan.assignments:
        try:
            recipient_rows = mutable[assignment.recipient_record_id]
            donor_rows = parsed[assignment.donor_record_id]
        except KeyError as exc:
            raise MatchedMotifDonorError("donor plan references an absent record") from exc
        if assignment.recipient_record_id == assignment.donor_record_id:
            raise MatchedMotifDonorError("donor plan contains a self-molecule donor")
        for recipient_atom, donor_atom in zip(
            assignment.recipient_atom_indices, assignment.donor_atom_indices
        ):
            previous = tuple(recipient_rows[recipient_atom])
            replacement = tuple(donor_rows[donor_atom])
            changed_slots += sum(left != right for left, right in zip(previous, replacement))
            recipient_rows[recipient_atom] = list(replacement)
        changed_motifs[assignment.recipient_record_id].add(
            assignment.recipient_motif_index
        )
    return MatchedStateOverlay(
        state_by_record_id={
            record_id: tuple(tuple(row) for row in rows)  # type: ignore[misc]
            for record_id, rows in mutable.items()
        },
        changed_motifs_by_record_id={
            record_id: tuple(sorted(motifs))
            for record_id, motifs in changed_motifs.items()
        },
        changed_state_slot_count=changed_slots,
    )


__all__ = [
    "DONOR_ATOM_MAP_SIDECAR_SCHEMA",
    "DONOR_PLAN_ID",
    "MatchedMotifDonorError",
    "MatchedMotifDonorPlan",
    "MatchedStateOverlay",
    "MotifDonorAssignment",
    "MotifDonorKey",
    "MotifOccurrence",
    "build_graphports_donor_atom_map_sidecar",
    "build_matched_motif_donor_plan",
    "extract_motif_occurrences",
    "materialize_matched_state_overlay",
]
