#!/usr/bin/env python3
"""Expose frozen QM9 and HIV evaluation members through one identity interface.

The task-specific builders remain the scientific source of split membership.
This adapter performs no chemistry and no resplitting: it only projects their
already-frozen validation/test identities into the repository's existing
``identity-collection-manifest/v1`` schema so one pretraining decontamination
path can consume QM9, HIV, and the KPGT collections.

Input file digests are recorded as provenance observations.  They are not
caller-supplied admission gates; scientific admission comes from the input
schemas, protocol IDs, counts, split membership, and identity consistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence, Tuple

from . import prove_membership_identity_overlap_v1 as proof


QM9_ROW_SCHEMA = "most-t5-r1/qm9-clean-split-member/v2"
QM9_SUMMARY_SCHEMA = "most-t5-r1/qm9-clean-split-summary/v2"
HIV_ROW_SCHEMA = "most-t5-r1/hiv-derived-split-member/v2"
HIV_SPLIT_SCHEMA = "most-t5-r1/hiv-murcko-derived-split-manifest/v2"
COLLECTION_SCHEMA = "most-t5-r1/identity-collection-manifest/v1"
IDENTITY_ROW_SCHEMA = "most-t5-r1/molecule-identity-row/v1"
SUMMARY_SCHEMA = "most-t5-r1/qm9-hiv-identity-collection-adapter-summary/v2"

QM9_PROTOCOL_ID = "qm9-3dmolt5-connectivity-group-110k10k-rest-s42-v2"
HIV_PROTOCOL_ID = "HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v2"
QM9_DATASET_ID = "3dmolt5-e3fp-mol-instructions-qm9-clean-view"
HIV_DATASET_ID = "HIV-MoleculeNet-DeepChem"
PROTECTED_SPLITS = ("validation", "test")
ALL_SPLITS = ("train", "validation", "test")
IDENTITY_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "pcqm4mv2_identity_normalization_contract.json"
)


class IdentityAdapterError(ValueError):
    """Raised when a frozen task output cannot be projected without ambiguity."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_observation(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            byte_count += len(block)
            digest.update(block)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise IdentityAdapterError(label + " is not a file: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise IdentityAdapterError(label + " must contain one JSON object")
    return value


def iter_jsonl(path: Path, label: str) -> Iterator[Dict[str, object]]:
    if not path.is_file():
        raise IdentityAdapterError(label + " is not a file: " + str(path))
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise IdentityAdapterError(
                    "blank line in {} at {}".format(label, line_number)
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise IdentityAdapterError(
                    "non-object row in {} at {}".format(label, line_number)
                )
            yield value


def require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IdentityAdapterError(label + " must be a lowercase 64-hex identity")
    return value


def write_bytes_new(path: Path, payload: bytes) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def identity_spec_id() -> str:
    if not IDENTITY_CONTRACT.is_file():
        raise IdentityAdapterError(
            "shared identity normalization contract is missing: "
            + str(IDENTITY_CONTRACT)
        )
    return str(file_observation(IDENTITY_CONTRACT)["sha256"])


def key_lf_digest(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["member_id"]).encode("utf-8") + b"\n")
    return digest.hexdigest()


def write_collection(
    output_dir: Path,
    *,
    dataset_slug: str,
    dataset_id: str,
    release_id: str,
    task_family: str,
    split: str,
    source_namespace: str,
    source_manifest_path: Path,
    members: Mapping[str, Tuple[str, str]],
    spec_id: str,
) -> Dict[str, object]:
    collection_id = "{}-{}-identity-v2".format(dataset_slug, split)
    rows = [
        {
            "schema_version": IDENTITY_ROW_SCHEMA,
            "collection_id": collection_id,
            "member_id": member_id,
            "connectivity_identity_sha256": identities[0],
            "stereo_identity_sha256": identities[1],
            "conformer_identity_sha256": None,
        }
        for member_id, identities in members.items()
    ]
    rows.sort(key=lambda row: str(row["member_id"]).encode("utf-8"))
    rows_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    relative_dir = Path("collections") / dataset_slug / split
    rows_path = output_dir / relative_dir / "molecule_identity_rows.jsonl"
    rows_observation = write_bytes_new(rows_path, rows_payload)
    rows_declaration = {
        "path": "molecule_identity_rows.jsonl",
        "bytes": rows_observation["bytes"],
        "sha256": rows_observation["sha256"],
        "row_count": len(rows),
        "key_lf_sha256": key_lf_digest(rows),
    }
    source_observation = file_observation(source_manifest_path)
    extractor_observation = file_observation(Path(__file__).resolve())
    manifest = {
        "schema_version": COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "dataset_id": dataset_id,
        "release_id": release_id,
        "phase": "downstream",
        "split": split,
        "role": "downstream_" + split,
        "task_family": task_family,
        "identity_specs": {
            "connectivity_identity_spec_sha256": spec_id,
            "stereo_identity_spec_sha256": spec_id,
            "conformer_identity": {"status": "unavailable", "spec_sha256": None},
            "text_identity": {
                "status": "unavailable",
                "exact_spec_sha256": None,
                "normalized_spec_sha256": None,
            },
        },
        "molecule_rows": rows_declaration,
        "text_pair_rows": None,
        "provenance": {
            "source_identity_namespace": source_namespace,
            "source_release_manifest_sha256": source_observation["sha256"],
            "extractor_sha256": extractor_observation["sha256"],
            "excluded_source_metadata_keys": [],
        },
    }
    proof.validate_collection_manifest(manifest)
    manifest_path = output_dir / relative_dir / "collection_manifest.json"
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    manifest_observation = write_bytes_new(manifest_path, manifest_payload)
    return {
        "collection_id": collection_id,
        "dataset_id": dataset_id,
        "split": split,
        "role": "downstream_" + split,
        "member_count": len(rows),
        "manifest": {
            "relative_path": (relative_dir / "collection_manifest.json").as_posix(),
            **manifest_observation,
        },
        "molecule_rows": {
            "relative_path": (relative_dir / "molecule_identity_rows.jsonl").as_posix(),
            **rows_observation,
        },
    }


def collect_qm9(
    split_manifest_path: Path,
    summary_path: Path,
    *,
    expected_identity_spec_id: str,
) -> Tuple[
    Dict[str, Dict[str, Tuple[str, str]]],
    Dict[str, int],
    Dict[str, int],
]:
    summary = load_json(summary_path, "QM9 split summary")
    if summary.get("schema_version") != QM9_SUMMARY_SCHEMA:
        raise IdentityAdapterError("QM9 summary schema is not the frozen clean split")
    if summary.get("dataset_id") != QM9_DATASET_ID:
        raise IdentityAdapterError("QM9 dataset ID differs")
    if summary.get("split_protocol_id") != QM9_PROTOCOL_ID:
        raise IdentityAdapterError("QM9 split protocol ID differs")
    if summary.get("identity_normalization_contract_sha256") != expected_identity_spec_id:
        raise IdentityAdapterError("QM9 identity-normalization contract differs")
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise IdentityAdapterError("QM9 summary counts are missing")
    expected_groups = counts.get("output_groups")
    expected_rows = counts.get("output_rows")
    if not isinstance(expected_groups, dict) or not isinstance(expected_rows, dict):
        raise IdentityAdapterError("QM9 output group/row counts are missing")

    members = {split: {} for split in PROTECTED_SPLITS}  # type: ignore[var-annotated]
    row_counts = {split: 0 for split in ALL_SPLITS}
    group_connectivity = {split: {} for split in ALL_SPLITS}  # type: ignore[var-annotated]
    group_split: Dict[str, str] = {}
    for row in iter_jsonl(split_manifest_path, "QM9 split manifest"):
        if row.get("schema_version") != QM9_ROW_SCHEMA:
            raise IdentityAdapterError("QM9 split row schema differs")
        split = row.get("assigned_split")
        if split not in ALL_SPLITS:
            raise IdentityAdapterError("QM9 row has an unknown assigned split")
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise IdentityAdapterError("QM9 row has no group_id")
        connectivity = require_digest(
            row.get("canonical_connectivity_smiles_sha256"),
            "QM9 connectivity identity",
        )
        expected_group_id = "qm9-canonical-connectivity-smiles-sha256:" + connectivity
        if group_id != expected_group_id:
            raise IdentityAdapterError(
                "QM9 group_id is not derived from its connectivity identity"
            )
        stereo = require_digest(
            row.get("strict_canonical_isomeric_smiles_sha256"),
            "QM9 stereo identity",
        )
        row_counts[split] += 1
        previous_split = group_split.setdefault(group_id, split)
        if previous_split != split:
            raise IdentityAdapterError("one QM9 connectivity group crosses splits")
        previous_connectivity = group_connectivity[split].setdefault(
            group_id, connectivity
        )
        if previous_connectivity != connectivity:
            raise IdentityAdapterError(
                "one QM9 connectivity group maps to conflicting identities"
            )
        if split in PROTECTED_SPLITS:
            member_id = group_id + ":stereo:" + stereo
            previous = members[split].setdefault(member_id, (connectivity, stereo))
            if previous != (connectivity, stereo):
                raise IdentityAdapterError("one QM9 stereo-state member maps inconsistently")
    for split in ALL_SPLITS:
        if row_counts[split] != expected_rows.get(split):
            raise IdentityAdapterError("QM9 {} row count differs from summary".format(split))
        if len(group_connectivity[split]) != expected_groups.get(split):
            raise IdentityAdapterError(
                "QM9 {} connectivity-group count differs from summary".format(split)
            )
    return (
        members,
        {split: row_counts[split] for split in PROTECTED_SPLITS},
        {split: len(group_connectivity[split]) for split in PROTECTED_SPLITS},
    )


def collect_hiv(
    member_manifest_path: Path,
    split_manifest_path: Path,
    *,
    expected_identity_spec_id: str,
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    split_manifest = load_json(split_manifest_path, "HIV split manifest")
    if split_manifest.get("schema_version") != HIV_SPLIT_SCHEMA:
        raise IdentityAdapterError("HIV split schema differs")
    if split_manifest.get("dataset_id") != HIV_DATASET_ID:
        raise IdentityAdapterError("HIV dataset ID differs")
    if split_manifest.get("protocol_id") != HIV_PROTOCOL_ID:
        raise IdentityAdapterError("HIV protocol ID differs")
    canonicalization = split_manifest.get("canonicalization")
    if not isinstance(canonicalization, dict):
        raise IdentityAdapterError("HIV canonicalization declaration is missing")
    if canonicalization.get("identity_normalization_contract_sha256") != expected_identity_spec_id:
        raise IdentityAdapterError("HIV identity-normalization contract differs")
    counts = split_manifest.get("counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("member_counts"), dict):
        raise IdentityAdapterError("HIV member counts are missing")
    expected = counts["member_counts"]

    members = {split: {} for split in PROTECTED_SPLITS}  # type: ignore[var-annotated]
    row_counts = {split: 0 for split in ALL_SPLITS}
    member_split: Dict[str, str] = {}
    for row in iter_jsonl(member_manifest_path, "HIV member manifest"):
        if row.get("schema_version") != HIV_ROW_SCHEMA:
            raise IdentityAdapterError("HIV member row schema differs")
        if row.get("protocol_id") != HIV_PROTOCOL_ID:
            raise IdentityAdapterError("HIV member row protocol ID differs")
        if row.get("dataset_id") != HIV_DATASET_ID:
            raise IdentityAdapterError("HIV member row dataset ID differs")
        split = row.get("assigned_split")
        if split not in ALL_SPLITS:
            raise IdentityAdapterError("HIV row has an unknown assigned split")
        member_id = row.get("member_id")
        if not isinstance(member_id, str) or not member_id:
            raise IdentityAdapterError("HIV row has no member_id")
        identities = (
            require_digest(row.get("connectivity_identity_sha256"), "HIV connectivity identity"),
            require_digest(row.get("stereo_identity_sha256"), "HIV stereo identity"),
        )
        if member_id in member_split:
            raise IdentityAdapterError("duplicate HIV member_id across the split manifest")
        member_split[member_id] = split
        row_counts[split] += 1
        if split in PROTECTED_SPLITS:
            members[split][member_id] = identities
    for split in ALL_SPLITS:
        if row_counts[split] != expected.get(split):
            raise IdentityAdapterError(
                "HIV {} member count differs from split manifest".format(split)
            )
    return members


def build_identity_collections(
    *,
    qm9_split_manifest: Path,
    qm9_summary: Path,
    hiv_member_manifest: Path,
    hiv_split_manifest: Path,
    output_dir: Path,
) -> Dict[str, object]:
    if output_dir.exists():
        raise IdentityAdapterError("output directory must not already exist")
    qm9_source_manifest = qm9_split_manifest.parent / "source_manifest.json"
    hiv_source_manifest = hiv_member_manifest.parent / "source_manifest.json"
    if not qm9_source_manifest.is_file() or not hiv_source_manifest.is_file():
        raise IdentityAdapterError("task source manifest is missing beside an input")
    spec_id = identity_spec_id()
    qm9_members, qm9_row_counts, qm9_group_counts = collect_qm9(
        qm9_split_manifest,
        qm9_summary,
        expected_identity_spec_id=spec_id,
    )
    hiv_members = collect_hiv(
        hiv_member_manifest,
        hiv_split_manifest,
        expected_identity_spec_id=spec_id,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    collections = []
    for split in PROTECTED_SPLITS:
        collections.append(
            write_collection(
                output_dir,
                dataset_slug="qm9",
                dataset_id=QM9_DATASET_ID,
                release_id=QM9_PROTOCOL_ID,
                task_family="qm9_homo_lumo_gap_property_prediction",
                split=split,
                source_namespace="qm9_clean_connectivity_group_and_stereo_state",
                source_manifest_path=qm9_source_manifest,
                members=qm9_members[split],
                spec_id=spec_id,
            )
        )
        collections.append(
            write_collection(
                output_dir,
                dataset_slug="hiv",
                dataset_id=HIV_DATASET_ID,
                release_id=HIV_PROTOCOL_ID,
                task_family="moleculenet_hiv_classification",
                split=split,
                source_namespace="deepchem_hiv_csv_source_member_index",
                source_manifest_path=hiv_source_manifest,
                members=hiv_members[split],
                spec_id=spec_id,
            )
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_boundary": (
            "identity_projection_only; no chemistry, resplitting, or membership change"
        ),
        "input_digest_role": "provenance_observation_not_admission_gate",
        "identity_spec_id": spec_id,
        "qm9_instruction_rows_scanned": qm9_row_counts,
        "qm9_connectivity_groups_scanned": qm9_group_counts,
        "collections": collections,
    }
    write_bytes_new(
        output_dir / "summary.json", canonical_json_bytes(summary) + b"\n"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qm9-split-manifest", required=True, type=Path)
    parser.add_argument("--qm9-summary", required=True, type=Path)
    parser.add_argument("--hiv-member-manifest", required=True, type=Path)
    parser.add_argument("--hiv-split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    summary = build_identity_collections(
        qm9_split_manifest=arguments.qm9_split_manifest,
        qm9_summary=arguments.qm9_summary,
        hiv_member_manifest=arguments.hiv_member_manifest,
        hiv_split_manifest=arguments.hiv_split_manifest,
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "collection_count": len(summary["collections"]),
                "output_dir": str(arguments.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (IdentityAdapterError, OSError, ValueError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
