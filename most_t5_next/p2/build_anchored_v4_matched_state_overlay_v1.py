"""Build one anchored-identity matched-state overlay from the tensor cache.

The cache already owns every deterministic training field needed by this
diagnostic.  Donors are matched across molecules by the anchored motif
identity, atom count, canonical-local endpoint-degree pattern, and are copied
with the persisted canonical-local atom correspondence.  Both E3FP and Morgan
states use the same donor assignment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .pf10_training_tensor_cache_v1 import IndexedPF10TrainingTensorCache


SCHEMA_VERSION = "most-t5-p2/anchored-v4-matched-state-overlay/v1"
ROW_SCHEMA = "most-t5-p2/anchored-v4-matched-state-row/v1"
ROWS_NAME = "matched_state_rows.jsonl"
MANIFEST_NAME = "manifest.json"
STATE_KINDS = ("e3fp", "morgan")


class AnchoredV4MatchedStateOverlayError(RuntimeError):
    """The anchored matched-state overlay contract is inconsistent."""


@dataclass(frozen=True)
class _Occurrence:
    record_id: str
    motif_index: int
    atom_indices_by_local_id: tuple[int, ...]
    key: tuple[str, int, tuple[int, ...]]


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _occurrences(cache: object, record: object) -> tuple[_Occurrence, ...]:
    record_id = str(record.record_id)
    owners = tuple(int(value) for value in record.atom_to_logical_motif)
    local_positions = tuple(int(value) for value in cache.atom_local_positions(record))
    if len(owners) != len(local_positions) or len(owners) != len(record.full_e3fp_ids):
        raise AnchoredV4MatchedStateOverlayError("cache atom axes disagree")
    endpoint_degrees = Counter(
        int(atom_index)
        for atom_index in record.connection_token_to_atom
        if int(atom_index) >= 0
    )
    rows = []
    for motif_index, identity in enumerate(record.exact_identity_sha256):
        atoms = tuple(index for index, owner in enumerate(owners) if owner == motif_index)
        if not atoms:
            raise AnchoredV4MatchedStateOverlayError("one motif owns no atoms")
        ordered = tuple(sorted(atoms, key=lambda index: local_positions[index]))
        positions = tuple(local_positions[index] for index in ordered)
        if len(set(positions)) != len(positions):
            raise AnchoredV4MatchedStateOverlayError(
                "canonical-local atom positions repeat within one motif"
            )
        degrees = tuple(int(endpoint_degrees[index]) for index in ordered)
        rows.append(
            _Occurrence(
                record_id=record_id,
                motif_index=motif_index,
                atom_indices_by_local_id=ordered,
                key=(str(identity), len(ordered), degrees),
            )
        )
    if len(rows) != len(record.exact_identity_sha256):
        raise AnchoredV4MatchedStateOverlayError("motif occurrence count differs")
    return tuple(rows)


def _plan(
    cache: object,
    records: Sequence[object],
) -> tuple[dict[tuple[str, int], _Occurrence], dict[str, int | float]]:
    by_key_record: dict[
        tuple[str, int, tuple[int, ...]], dict[str, list[_Occurrence]]
    ] = defaultdict(lambda: defaultdict(list))
    all_occurrences = []
    for record in records:
        for occurrence in _occurrences(cache, record):
            all_occurrences.append(occurrence)
            by_key_record[occurrence.key][occurrence.record_id].append(occurrence)

    donor_by_recipient: dict[tuple[str, int], _Occurrence] = {}
    excluded = 0
    donor_reuse: Counter[tuple[str, int]] = Counter()
    eligible_atoms = 0
    records_with_any: set[str] = set()
    eligible_by_record: Counter[str] = Counter()
    total_by_record: Counter[str] = Counter(
        occurrence.record_id for occurrence in all_occurrences
    )
    for key in sorted(by_key_record):
        record_map = by_key_record[key]
        record_ids = sorted(record_map)
        if len(record_ids) < 2:
            excluded += sum(len(rows) for rows in record_map.values())
            continue
        for position, recipient_id in enumerate(record_ids):
            donor_id = record_ids[(position + 1) % len(record_ids)]
            recipients = sorted(record_map[recipient_id], key=lambda row: row.motif_index)
            donors = sorted(record_map[donor_id], key=lambda row: row.motif_index)
            for occurrence_index, recipient in enumerate(recipients):
                donor = donors[occurrence_index % len(donors)]
                if donor.record_id == recipient.record_id:
                    raise AnchoredV4MatchedStateOverlayError("donor cannot be self-molecule")
                donor_by_recipient[(recipient.record_id, recipient.motif_index)] = donor
                donor_reuse[(donor.record_id, donor.motif_index)] += 1
                eligible_atoms += len(recipient.atom_indices_by_local_id)
                eligible_by_record[recipient.record_id] += 1
                records_with_any.add(recipient.record_id)

    total_atoms = sum(len(row.atom_indices_by_local_id) for row in all_occurrences)
    eligible_motifs = len(donor_by_recipient)
    total_motifs = len(all_occurrences)
    record_ids = tuple(str(record.record_id) for record in records)
    coverage: dict[str, int | float] = {
        "total_records": len(records),
        "total_motif_occurrences": total_motifs,
        "eligible_motif_occurrences": eligible_motifs,
        "excluded_motif_occurrences": excluded,
        "motif_occurrence_coverage": eligible_motifs / total_motifs,
        "total_atom_rows": total_atoms,
        "eligible_atom_rows": eligible_atoms,
        "atom_row_coverage": eligible_atoms / total_atoms,
        "records_with_any_eligible_motif": len(records_with_any),
        "records_with_all_motifs_eligible": sum(
            eligible_by_record[record_id] == total_by_record[record_id]
            for record_id in record_ids
        ),
        "max_donor_reuse": max(donor_reuse.values(), default=0),
    }
    return donor_by_recipient, coverage


def build_anchored_v4_matched_state_overlay(
    *, cache_root: Path, output_dir: Path
) -> dict[str, object]:
    cache_root = Path(cache_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise AnchoredV4MatchedStateOverlayError("output or staging path already exists")
    cache = IndexedPF10TrainingTensorCache(cache_root)
    dev_indices = cache.split_indices("dev")
    records = tuple(cache[index] for index in dev_indices)
    if not records or len({record.record_id for record in records}) != len(records):
        raise AnchoredV4MatchedStateOverlayError("dev record IDs are empty or repeated")
    donors, coverage = _plan(cache, records)
    by_id = {record.record_id: record for record in records}
    index_by_id = {record.record_id: record.cache_index for record in records}

    staging.mkdir(parents=True, exist_ok=False)
    changed_slots = {kind: 0 for kind in STATE_KINDS}
    changed_records = 0
    with (staging / ROWS_NAME).open("w", encoding="utf-8", newline="\n") as handle:
        for split_index, record in enumerate(records):
            e3fp = [list(row) for row in record.full_e3fp_ids]
            morgan = [
                list(row)
                for row in cache.morgan_state(
                    record.record_id, cache_index=record.cache_index
                )
            ]
            changed_motifs = []
            for motif_index in range(len(record.exact_identity_sha256)):
                donor = donors.get((record.record_id, motif_index))
                if donor is None:
                    continue
                donor_record = by_id[donor.record_id]
                donor_e3fp = donor_record.full_e3fp_ids
                donor_morgan = cache.morgan_state(
                    donor.record_id, cache_index=index_by_id[donor.record_id]
                )
                recipient = _occurrences(cache, record)[motif_index]
                for recipient_atom, donor_atom in zip(
                    recipient.atom_indices_by_local_id,
                    donor.atom_indices_by_local_id,
                ):
                    for kind, target, source in (
                        ("e3fp", e3fp, donor_e3fp),
                        ("morgan", morgan, donor_morgan),
                    ):
                        changed_slots[kind] += sum(
                            int(left != right)
                            for left, right in zip(target[recipient_atom], source[donor_atom])
                        )
                        target[recipient_atom] = list(source[donor_atom])
                changed_motifs.append(motif_index)
            changed_records += int(bool(changed_motifs))
            handle.write(
                _json_text(
                    {
                        "schema_version": ROW_SCHEMA,
                        "split_index": split_index,
                        "record_id": record.record_id,
                        "changed_motif_indices": changed_motifs,
                        "e3fp_state_ids": e3fp,
                        "morgan_state_ids": morgan,
                    }
                )
                + "\n"
            )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "anchored_pf1_dev_matched_state_diagnostic",
        "source_cache": str(cache_root),
        "rows_file": ROWS_NAME,
        "state_kinds": list(STATE_KINDS),
        "matching_signature": [
            "anchored_exact_motif_identity",
            "atom_count",
            "canonical_local_endpoint_degree_pattern",
        ],
        "counts": {
            "dev_records": len(records),
            "published_rows": len(records),
            "records_with_any_matched_motif": changed_records,
            "changed_state_slots": changed_slots,
        },
        "coverage": coverage,
        "semantics": {
            "cross_molecule_donors_only": True,
            "canonical_local_atom_correspondence": True,
            "same_donor_assignment_for_e3fp_and_morgan": True,
            "unmatched_motifs_keep_aligned_state": True,
            "training_cache_unchanged": True,
            "dev_diagnostic_only": True,
        },
    }
    (staging / MANIFEST_NAME).write_text(
        _json_text(manifest) + "\n", encoding="utf-8"
    )
    staging.rename(output_dir)
    return manifest


class AnchoredV4MatchedStateProvider:
    def __init__(self, root: Path, *, state_kind: str) -> None:
        if state_kind not in STATE_KINDS:
            raise AnchoredV4MatchedStateOverlayError("state kind is unsupported")
        root = Path(root).expanduser().resolve()
        try:
            manifest = json.loads((root / MANIFEST_NAME).read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise AnchoredV4MatchedStateOverlayError("overlay manifest is unreadable") from exc
        if not isinstance(manifest, Mapping) or not (
            manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("status") == "pass"
        ):
            raise AnchoredV4MatchedStateOverlayError("overlay manifest is not passed")
        field = f"{state_kind}_state_ids"
        rows: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
        with (root / str(manifest["rows_file"])).open(encoding="utf-8") as handle:
            for split_index, line in enumerate(handle):
                row = json.loads(line)
                if not isinstance(row, Mapping) or not (
                    row.get("schema_version") == ROW_SCHEMA
                    and row.get("split_index") == split_index
                    and isinstance(row.get("record_id"), str)
                    and row["record_id"] not in rows
                ):
                    raise AnchoredV4MatchedStateOverlayError("overlay row order is invalid")
                values = tuple(tuple(int(value) for value in state) for state in row[field])
                if not values or any(len(state) != 4 for state in values):
                    raise AnchoredV4MatchedStateOverlayError("overlay state shape differs")
                rows[str(row["record_id"])] = values
        if len(rows) != int(manifest["counts"]["published_rows"]):  # type: ignore[index]
            raise AnchoredV4MatchedStateOverlayError("overlay row count differs")
        self.state_kind = f"anchored_matched_{state_kind}"
        self.manifest = dict(manifest)
        self._rows = rows

    def get(self, record_id: str) -> tuple[tuple[int, int, int, int], ...]:
        try:
            return self._rows[record_id]
        except KeyError as exc:
            raise AnchoredV4MatchedStateOverlayError("record is absent from overlay") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_anchored_v4_matched_state_overlay(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
    )
    print(_json_text({"status": report["status"], "coverage": report["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnchoredV4MatchedStateOverlayError",
    "AnchoredV4MatchedStateProvider",
    "SCHEMA_VERSION",
    "build_anchored_v4_matched_state_overlay",
]
