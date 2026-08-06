#!/usr/bin/env python3
"""Derive the frozen HIV MoleculeNet scaffold split used by MoSt-T5.

The 3D-MolT5 release does not publish a traceable HIV dataset/split artifact.
This builder therefore starts from DeepChem's authoritative MoleculeNet
``HIV.csv`` object and derives a transparent 8:1:1 Bemis-Murcko split.  It is
*not* an official 3D-MolT5, KPGT, or DeepChem membership release.

The assignment mirrors ``deepchem.splits.ScaffoldSplitter`` in DeepChem 2.8.0:

1. compute a non-chiral Bemis-Murcko scaffold for every source member;
2. sort scaffold groups by ``(group_size, first_source_member_index)`` in
   descending order;
3. greedily place each complete group before the 0.8 and 0.9 row cutoffs.

DeepChem's ``seed`` argument is unused by this algorithm.  This implementation
performs no random operation and records ``seed = null``.  Unlike DeepChem's
general splitter, invalid SMILES are rejected rather than silently omitted.

Production CLI use requires the known SHA-256 and revision of the official
source object.  The Python API accepts an explicit ``SourceBinding`` so small
fixture files can test the protocol without downloading the dataset locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
except ImportError as exc:  # pragma: no cover - exercised in a broken environment.
    raise RuntimeError(
        "build_hiv_murcko_split.py requires RDKit for canonicalization and scaffolds"
    ) from exc


PROTOCOL_ID = "HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1"
DATASET_ID = "HIV-MoleculeNet-DeepChem"

DEEPCHEM_VERSION = "2.8.0"
DEEPCHEM_COMMIT = "d5b293934d427062f52e2d92c1569d53d10418f9"
DEEPCHEM_HIV_LOADER_URL = (
    "https://github.com/deepchem/deepchem/blob/"
    + DEEPCHEM_COMMIT
    + "/deepchem/molnet/load_function/hiv_datasets.py"
)
DEEPCHEM_SCAFFOLD_SPLITTER_URL = (
    "https://github.com/deepchem/deepchem/blob/"
    + DEEPCHEM_COMMIT
    + "/deepchem/splits/splitters.py#L1360-L1482"
)
OFFICIAL_SOURCE_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
)
OFFICIAL_SOURCE_TRANSPORT_ALIAS = (
    "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/HIV.csv"
)
OFFICIAL_SOURCE_REVISION = (
    "deepchem-HIV.csv-etag-9ad10c88f82f1dac7eb5c52b668c30a7"
)
OFFICIAL_SOURCE_SHA256 = (
    "9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22"
)
OFFICIAL_SOURCE_MD5 = "9ad10c88f82f1dac7eb5c52b668c30a7"
OFFICIAL_SOURCE_BYTES = 2_193_844
OFFICIAL_SOURCE_LAST_MODIFIED = "2020-07-10T06:45:51Z"
OFFICIAL_SOURCE_MEMBER_COUNT = 41_127

EXPECTED_COLUMNS = ("smiles", "activity", "HIV_active")
ACTIVITY_TO_LABEL = {"CI": 0, "CM": 1, "CA": 1}
SPLIT_NAMES = ("train", "validation", "test")

SOURCE_MANIFEST_FILENAME = "source_manifest.json"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
MEMBER_MANIFEST_FILENAME = "member_manifest.jsonl"
PROTECTED_ROWS_FILENAME = "protected_union_identity_rows.jsonl"
PROTECTED_MANIFEST_FILENAME = "protected_union_manifest.json"

SOURCE_MANIFEST_SCHEMA = "most-t5-r1/hiv-authoritative-source-manifest/v1"
SPLIT_MANIFEST_SCHEMA = "most-t5-r1/hiv-murcko-derived-split-manifest/v1"
MEMBER_ROW_SCHEMA = "most-t5-r1/hiv-derived-split-member/v1"
PROTECTED_ROW_SCHEMA = "most-t5-r1/protected-connectivity-identity-row/v1"
PROTECTED_MANIFEST_SCHEMA = "most-t5-r1/protected-connectivity-union-manifest/v1"


class HivSplitProtocolError(ValueError):
    """Raised when source bytes or members violate the frozen protocol."""


@dataclass(frozen=True)
class SourceBinding:
    """An explicit immutable claim about one input CSV artifact."""

    revision: str
    expected_sha256: str
    source_url: str
    expected_bytes: int | None = None
    expected_md5: str | None = None
    expected_member_count: int | None = None
    last_modified_utc: str | None = None
    authority: str = "DeepChem MoleculeNet"


OFFICIAL_SOURCE_BINDING = SourceBinding(
    revision=OFFICIAL_SOURCE_REVISION,
    expected_sha256=OFFICIAL_SOURCE_SHA256,
    source_url=OFFICIAL_SOURCE_URL,
    expected_bytes=OFFICIAL_SOURCE_BYTES,
    expected_md5=OFFICIAL_SOURCE_MD5,
    expected_member_count=OFFICIAL_SOURCE_MEMBER_COUNT,
    last_modified_utc=OFFICIAL_SOURCE_LAST_MODIFIED,
)


@dataclass(frozen=True)
class SourceObservation:
    bytes: int
    sha256: str
    md5: str


@dataclass(frozen=True)
class Member:
    source_member_index: int
    source_csv_line_number: int
    raw_smiles: str
    activity: str
    label: int
    canonical_isomeric_smiles: str
    canonical_connectivity_smiles: str
    scaffold_smiles: str

    @property
    def member_id(self) -> str:
        return "hiv-moleculenet-source-member:" + str(self.source_member_index)

    @property
    def connectivity_identity_sha256(self) -> str:
        return sha256_text(self.canonical_connectivity_smiles)

    @property
    def stereo_identity_sha256(self) -> str:
        return sha256_text(self.canonical_isomeric_smiles)

    @property
    def scaffold_sha256(self) -> str:
        return sha256_text(self.scaffold_smiles)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def observe_file(path: Path) -> SourceObservation:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(block)
            sha256.update(block)
            md5.update(block)
    return SourceObservation(
        bytes=byte_count,
        sha256=sha256.hexdigest(),
        md5=md5.hexdigest(),
    )


def _require_hex_digest(value: str, length: int, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HivSplitProtocolError(label + " must be lowercase hexadecimal")


def validate_source_binding(binding: SourceBinding) -> None:
    if not binding.revision.strip():
        raise HivSplitProtocolError("source revision must be explicit and non-empty")
    if not binding.source_url.startswith("https://"):
        raise HivSplitProtocolError("source URL must use HTTPS")
    _require_hex_digest(binding.expected_sha256, 64, "expected source SHA-256")
    if binding.expected_md5 is not None:
        _require_hex_digest(binding.expected_md5, 32, "expected source MD5")
    if binding.expected_bytes is not None and binding.expected_bytes <= 0:
        raise HivSplitProtocolError("expected source byte count must be positive")
    if binding.expected_member_count is not None and binding.expected_member_count <= 0:
        raise HivSplitProtocolError("expected member count must be positive")


def bind_source(path: Path, binding: SourceBinding) -> SourceObservation:
    validate_source_binding(binding)
    if not path.is_file() or path.is_symlink():
        raise HivSplitProtocolError("source CSV must be one regular non-symlink file")
    observation = observe_file(path)
    if observation.sha256 != binding.expected_sha256:
        raise HivSplitProtocolError("source CSV SHA-256 differs from explicit binding")
    if binding.expected_bytes is not None and observation.bytes != binding.expected_bytes:
        raise HivSplitProtocolError("source CSV byte count differs from explicit binding")
    if binding.expected_md5 is not None and observation.md5 != binding.expected_md5:
        raise HivSplitProtocolError("source CSV MD5 differs from explicit binding")
    return observation


def _strict_field(row: Mapping[str | None, str | list[str] | None], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise HivSplitProtocolError(label + " field " + key + " must be non-empty")
    if value != value.strip():
        raise HivSplitProtocolError(label + " field " + key + " has surrounding whitespace")
    return value


def canonical_forms(smiles: str, label: str) -> tuple[str, str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise HivSplitProtocolError(label + " contains an RDKit-invalid SMILES")
    isomeric = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True, kekuleSmiles=False
    )
    connectivity = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=False, kekuleSmiles=False
    )
    scaffold = MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    if not isinstance(scaffold, str):
        raise RuntimeError("RDKit returned a non-string Bemis-Murcko scaffold")
    return isomeric, connectivity, scaffold


def read_members(path: Path, binding: SourceBinding) -> list[Member]:
    members: list[Member] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise HivSplitProtocolError(
                "source CSV header must be exactly " + ",".join(EXPECTED_COLUMNS)
            )
        for member_index, row in enumerate(reader):
            label_prefix = "source member " + str(member_index)
            if None in row:
                raise HivSplitProtocolError(label_prefix + " has extra CSV fields")
            if set(row) != set(EXPECTED_COLUMNS):
                raise HivSplitProtocolError(label_prefix + " violates the exact CSV schema")
            smiles = _strict_field(row, "smiles", label_prefix)
            activity = _strict_field(row, "activity", label_prefix)
            raw_label = _strict_field(row, "HIV_active", label_prefix)
            if activity not in ACTIVITY_TO_LABEL:
                raise HivSplitProtocolError(label_prefix + " has an unknown activity code")
            if raw_label not in ("0", "1"):
                raise HivSplitProtocolError(label_prefix + " HIV_active must be exactly 0 or 1")
            numeric_label = int(raw_label)
            if ACTIVITY_TO_LABEL[activity] != numeric_label:
                raise HivSplitProtocolError(
                    label_prefix + " activity and HIV_active labels are inconsistent"
                )
            isomeric, connectivity, scaffold = canonical_forms(smiles, label_prefix)
            members.append(
                Member(
                    source_member_index=member_index,
                    source_csv_line_number=reader.line_num,
                    raw_smiles=smiles,
                    activity=activity,
                    label=numeric_label,
                    canonical_isomeric_smiles=isomeric,
                    canonical_connectivity_smiles=connectivity,
                    scaffold_smiles=scaffold,
                )
            )
    if not members:
        raise HivSplitProtocolError("source CSV contains no data members")
    if binding.expected_member_count is not None and len(members) != binding.expected_member_count:
        raise HivSplitProtocolError("source member count differs from explicit binding")
    return members


def assign_scaffold_splits(members: Sequence[Member]) -> tuple[dict[int, str], list[dict[str, object]]]:
    """Return DeepChem-2.8.0-equivalent deterministic scaffold assignments."""
    scaffold_members: dict[str, list[int]] = defaultdict(list)
    for member in members:
        scaffold_members[member.scaffold_smiles].append(member.source_member_index)
    for indices in scaffold_members.values():
        indices.sort()
    ordered = sorted(
        scaffold_members.items(),
        key=lambda item: (len(item[1]), item[1][0]),
        reverse=True,
    )
    train_cutoff = 0.8 * len(members)
    validation_cutoff = 0.9 * len(members)
    split_indices: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    group_rows: list[dict[str, object]] = []
    for rank, (scaffold, indices) in enumerate(ordered):
        if len(split_indices["train"]) + len(indices) > train_cutoff:
            if (
                len(split_indices["train"])
                + len(split_indices["validation"])
                + len(indices)
                > validation_cutoff
            ):
                split = "test"
            else:
                split = "validation"
        else:
            split = "train"
        split_indices[split].extend(indices)
        group_rows.append(
            {
                "scaffold_order_rank": rank,
                "scaffold_smiles": scaffold,
                "scaffold_sha256": sha256_text(scaffold),
                "member_count": len(indices),
                "first_source_member_index": indices[0],
                "assigned_split": split,
            }
        )
    assignments = {
        source_member_index: split
        for split, indices in split_indices.items()
        for source_member_index in indices
    }
    if len(assignments) != len(members) or set(assignments) != set(range(len(members))):
        raise RuntimeError("internal split assignment did not cover each member exactly once")
    return assignments, group_rows


def _class_counts(members: Iterable[Member]) -> dict[str, int]:
    counts = Counter(member.label for member in members)
    return {"negative_0": counts[0], "positive_1": counts[1]}


def validate_assignments(
    members: Sequence[Member], assignments: Mapping[int, str]
) -> dict[str, object]:
    split_members = {
        split: [member for member in members if assignments[member.source_member_index] == split]
        for split in SPLIT_NAMES
    }
    if any(not values for values in split_members.values()):
        raise HivSplitProtocolError("every derived split must be non-empty")
    class_counts = {split: _class_counts(values) for split, values in split_members.items()}
    for split, counts in class_counts.items():
        if counts["negative_0"] <= 0 or counts["positive_1"] <= 0:
            raise HivSplitProtocolError(
                split + " lacks one binary class, so AUROC would not be computable"
            )
    scaffolds = {
        split: {member.scaffold_smiles for member in values}
        for split, values in split_members.items()
    }
    if (
        scaffolds["train"] & scaffolds["validation"]
        or scaffolds["train"] & scaffolds["test"]
        or scaffolds["validation"] & scaffolds["test"]
    ):
        raise RuntimeError("Bemis-Murcko scaffold leakage detected across splits")
    return {
        "member_counts": {split: len(split_members[split]) for split in SPLIT_NAMES},
        "scaffold_counts": {split: len(scaffolds[split]) for split in SPLIT_NAMES},
        "class_counts": class_counts,
        "invariants": {
            "all_source_members_assigned_exactly_once": True,
            "bemis_murcko_scaffolds_split_disjoint": True,
            "both_binary_classes_present_in_each_split": True,
            "binary_auroc_computable_in_each_split": True,
        },
    }


def _artifact(payload: bytes, path: str, row_count: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def _json_payload(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl_payload(rows: Iterable[object]) -> tuple[bytes, int]:
    payload_parts: list[bytes] = []
    count = 0
    for row in rows:
        payload_parts.append(canonical_json_bytes(row) + b"\n")
        count += 1
    return b"".join(payload_parts), count


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_hiv_murcko_split(
    source_csv: Path | str,
    output_dir: Path | str,
    *,
    source_binding: SourceBinding,
) -> dict[str, object]:
    source_path = Path(source_csv)
    output_path = Path(output_dir)
    if output_path.exists():
        raise HivSplitProtocolError("output directory must not already exist")
    observation = bind_source(source_path, source_binding)
    members = read_members(source_path, source_binding)
    assignments, scaffold_group_rows = assign_scaffold_splits(members)
    split_facts = validate_assignments(members, assignments)

    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "source_status": "authoritative_raw_moleculenet_dataset",
        "split_status": "no_source_split_consumed; membership_is_derived_locally",
        "source_authority": source_binding.authority,
        "source_url": source_binding.source_url,
        "source_transport_alias": (
            OFFICIAL_SOURCE_TRANSPORT_ALIAS
            if source_binding == OFFICIAL_SOURCE_BINDING
            else None
        ),
        "source_revision": source_binding.revision,
        "source_file": {
            "name": source_path.name,
            "bytes": observation.bytes,
            "sha256": observation.sha256,
            "md5": observation.md5,
            "member_count": len(members),
        },
        "source_object_metadata": {
            "last_modified_utc": source_binding.last_modified_utc,
            "etag_md5": source_binding.expected_md5,
        },
        "deepchem_anchor": {
            "version": DEEPCHEM_VERSION,
            "commit": DEEPCHEM_COMMIT,
            "hiv_loader": DEEPCHEM_HIV_LOADER_URL,
            "scaffold_splitter": DEEPCHEM_SCAFFOLD_SPLITTER_URL,
        },
        "csv_contract": {
            "columns_exact_order": list(EXPECTED_COLUMNS),
            "blank_physical_lines": "ignored_by_RFC4180_csv_reader",
            "activity_to_binary_label": ACTIVITY_TO_LABEL,
            "invalid_smiles_policy": "fail_without_emitting_artifacts",
            "duplicate_source_members_policy": "retain_each_source_row_with_unique_source_index",
        },
    }
    source_payload = _json_payload(source_manifest)
    source_artifact = _artifact(source_payload, SOURCE_MANIFEST_FILENAME)

    member_rows = []
    for member in members:
        member_rows.append(
            {
                "schema_version": MEMBER_ROW_SCHEMA,
                "dataset_id": DATASET_ID,
                "protocol_id": PROTOCOL_ID,
                "member_id": member.member_id,
                "source_member_index": member.source_member_index,
                "source_csv_line_number": member.source_csv_line_number,
                "raw_smiles": member.raw_smiles,
                "canonical_isomeric_smiles": member.canonical_isomeric_smiles,
                "canonical_connectivity_smiles": member.canonical_connectivity_smiles,
                "stereo_identity_sha256": member.stereo_identity_sha256,
                "connectivity_identity_sha256": member.connectivity_identity_sha256,
                "activity": member.activity,
                "HIV_active": member.label,
                "bemis_murcko_scaffold_smiles": member.scaffold_smiles,
                "bemis_murcko_scaffold_sha256": member.scaffold_sha256,
                "assigned_split": assignments[member.source_member_index],
            }
        )
    member_payload, member_count = _jsonl_payload(member_rows)
    member_artifact = _artifact(
        member_payload, MEMBER_MANIFEST_FILENAME, row_count=member_count
    )

    protected_by_connectivity: dict[str, dict[str, object]] = {}
    for member in members:
        split = assignments[member.source_member_index]
        if split == "train":
            continue
        key = member.connectivity_identity_sha256
        entry = protected_by_connectivity.setdefault(
            key,
            {
                "canonical_connectivity_smiles": member.canonical_connectivity_smiles,
                "protected_splits": set(),
                "source_member_ids": [],
            },
        )
        entry["protected_splits"].add(split)  # type: ignore[union-attr]
        entry["source_member_ids"].append(member.member_id)  # type: ignore[union-attr]
    protected_rows = []
    for connectivity_sha256 in sorted(protected_by_connectivity):
        entry = protected_by_connectivity[connectivity_sha256]
        protected_rows.append(
            {
                "schema_version": PROTECTED_ROW_SCHEMA,
                "dataset_id": DATASET_ID,
                "protocol_id": PROTOCOL_ID,
                "connectivity_identity_sha256": connectivity_sha256,
                "canonical_connectivity_smiles": entry[
                    "canonical_connectivity_smiles"
                ],
                "protected_splits": sorted(entry["protected_splits"]),
                "source_member_count": len(entry["source_member_ids"]),
                "source_member_ids": sorted(entry["source_member_ids"]),
            }
        )
    protected_payload, protected_count = _jsonl_payload(protected_rows)
    protected_rows_artifact = _artifact(
        protected_payload, PROTECTED_ROWS_FILENAME, row_count=protected_count
    )
    protected_manifest = {
        "schema_version": PROTECTED_MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "role": "downstream_validation_and_test_connectivity_exclusion_union",
        "hard_exclusion_key": "connectivity_identity_sha256",
        "protected_splits": ["validation", "test"],
        "training_split_is_not_protected_by_this_manifest": True,
        "identity_canonicalization": {
            "library": "RDKit",
            "version": rdBase.rdkitVersion,
            "function": "Chem.MolToSmiles",
            "parameters": {
                "canonical": True,
                "isomericSmiles": False,
                "kekuleSmiles": False,
            },
        },
        "identity_rows": protected_rows_artifact,
    }
    protected_manifest_payload = _json_payload(protected_manifest)
    protected_manifest_artifact = _artifact(
        protected_manifest_payload, PROTECTED_MANIFEST_FILENAME
    )

    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "split_origin": "locally_derived_from_authoritative_unsplit_csv",
        "official_exact_split_reproduction": False,
        "must_not_be_reported_as": [
            "3D-MolT5 official HIV split",
            "KPGT HIV split",
            "DeepChem released split membership",
        ],
        "source_manifest": source_artifact,
        "algorithm": {
            "reference": "DeepChem 2.8.0 ScaffoldSplitter assignment semantics",
            "reference_url": DEEPCHEM_SCAFFOLD_SPLITTER_URL,
            "scaffold": {
                "definition": "Bemis-Murcko rings and linkers",
                "rdkit_function": "MurckoScaffoldSmiles",
                "includeChirality": False,
                "acyclic_scaffold": "empty_string_group_retained",
            },
            "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "group_order": {
                "key": ["group_size", "first_source_member_index"],
                "direction": "descending_for_both_fields",
                "tie_handling": "larger_first_source_member_index_first_when_group_sizes_tie",
            },
            "assignment": (
                "for each ordered complete scaffold group: assign train unless it would "
                "exceed 0.8*N; otherwise assign validation unless cumulative train+validation "
                "would exceed 0.9*N; otherwise assign test"
            ),
            "randomness": {
                "uses_randomness": False,
                "seed": None,
                "seed_semantics": "not_applicable; no random operation is executed",
            },
            "invalid_smiles_policy": "fail; never silently skip a source member",
        },
        "canonicalization": {
            "library": "RDKit",
            "version": rdBase.rdkitVersion,
            "molecule_identity": {
                "strict": {
                    "canonical": True,
                    "isomericSmiles": True,
                    "kekuleSmiles": False,
                },
                "protected_union_connectivity": {
                    "canonical": True,
                    "isomericSmiles": False,
                    "kekuleSmiles": False,
                },
            },
        },
        "counts": split_facts,
        "scaffold_groups_in_assignment_order": scaffold_group_rows,
        "artifacts": {
            "member_manifest": member_artifact,
            "protected_union_identity_rows": protected_rows_artifact,
            "protected_union_manifest": protected_manifest_artifact,
        },
    }
    split_payload = _json_payload(split_manifest)

    output_path.mkdir(parents=True, exist_ok=False)
    _write_new(output_path / SOURCE_MANIFEST_FILENAME, source_payload)
    _write_new(output_path / MEMBER_MANIFEST_FILENAME, member_payload)
    _write_new(output_path / PROTECTED_ROWS_FILENAME, protected_payload)
    _write_new(output_path / PROTECTED_MANIFEST_FILENAME, protected_manifest_payload)
    _write_new(output_path / SPLIT_MANIFEST_FILENAME, split_payload)
    return split_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _require_official_cli_binding(arguments: argparse.Namespace) -> None:
    if arguments.source_sha256 != OFFICIAL_SOURCE_SHA256:
        raise HivSplitProtocolError(
            "production CLI source SHA-256 must equal the frozen official HIV.csv SHA-256"
        )
    if arguments.source_revision != OFFICIAL_SOURCE_REVISION:
        raise HivSplitProtocolError(
            "production CLI source revision must equal the frozen official HIV.csv revision"
        )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    _require_official_cli_binding(arguments)
    result = build_hiv_murcko_split(
        arguments.source_csv,
        arguments.output_dir,
        source_binding=OFFICIAL_SOURCE_BINDING,
    )
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HivSplitProtocolError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
