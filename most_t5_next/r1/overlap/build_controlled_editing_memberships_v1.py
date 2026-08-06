#!/usr/bin/env python3
"""Freeze the Controlled Motif Editing validation and sealed-test members.

The compatibility test is the 200-source-molecule list published in the
MoleculeSTM dataset repository.  It is never development data.  A separate
400-molecule validation membership is sampled from the complete ZINC250K CSV
at the same repository revision after canonical-connectivity deduplication and
exclusion of every test connectivity.

Production canonicalization is frozen to RDKit 2024.03.5.  Selection is fully
replayable: candidates are ordered by their stable source-row key, shuffled by
the explicitly implemented SplitMix64/Fisher-Yates algorithm with seed 42,
and the first 400 are retained.  File digests are provenance observations,
not scientific admission gates; admission rests on the declared repository
revision, source schemas and populations, complete RDKit parsing, and the
membership invariants recorded below.

This builder emits molecule membership only.  The twelve FineMolTex prompt
IDs are recorded as a namespace, but no molecule-prompt Cartesian product or
supervised molecule-text pair is invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from rdkit import Chem, rdBase
except ImportError as exc:  # pragma: no cover - broken environment only.
    raise RuntimeError(
        "build_controlled_editing_memberships_v1.py requires RDKit"
    ) from exc

from most_t5_next.r1.overlap import shared_identity_normalization_v1 as identity_normalization

from . import prove_membership_identity_overlap_v1 as proof


MOLECULESTM_REPOSITORY_ID = "chao1224/MoleculeSTM"
MOLECULESTM_REPOSITORY_URL = "https://huggingface.co/datasets/chao1224/MoleculeSTM"
MOLECULESTM_REVISION = "ff2de71fa6bb0533d5e740db6d88a0442a0d38e8"
TEST_ORIGIN_PATH = "Editing_data/single_multi_property_SMILES.txt"
ZINC_ORIGIN_PATH = "ZINC250K_data/raw/250k_rndm_zinc_drugs_clean_3.csv"

PRODUCTION_RDKIT_VERSION = "2024.03.5"
EXPECTED_TEST_SOURCE_COUNT = 200
EXPECTED_ZINC_SOURCE_COUNT = 249_455
VALIDATION_MEMBER_COUNT = 400
SELECTION_SEED = 42
ZINC_COLUMNS = ("smiles", "logP", "qed", "SAS")
PROMPT_IDS = (101, 102, 103, 104, 105, 106, 205, 206, 501, 502, 503, 504)

DATASET_ID = "Controlled-Motif-Editing"
RELEASE_ID = "moleculestm-ff2de71-controlled-editing-membership-v2"
TASK_FAMILY = "controlled_motif_editing"
COLLECTION_SCHEMA = "most-t5-r1/identity-collection-manifest/v1"
IDENTITY_ROW_SCHEMA = "most-t5-r1/molecule-identity-row/v1"
SOURCE_MANIFEST_SCHEMA = "most-t5-r1/controlled-editing-source-manifest/v1"
SUMMARY_SCHEMA = "most-t5-r1/controlled-editing-membership-summary/v1"

SOURCE_MANIFEST_FILENAME = "source_manifest.json"
SUMMARY_FILENAME = "summary.json"
IDENTITY_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "pcqm4mv2_identity_normalization_contract.json"
)


class ControlledEditingProtocolError(ValueError):
    """Raised when a source or derived membership violates the protocol."""


@dataclass(frozen=True)
class SourceBinding:
    """Explicit provenance and semantic population claims for one build."""

    repository_id: str
    repository_url: str
    revision: str
    expected_test_source_count: int
    expected_zinc_source_count: int
    validation_member_count: int
    selection_seed: int = SELECTION_SEED
    authority: str = "MoleculeSTM dataset repository"


PRODUCTION_SOURCE_BINDING = SourceBinding(
    repository_id=MOLECULESTM_REPOSITORY_ID,
    repository_url=MOLECULESTM_REPOSITORY_URL,
    revision=MOLECULESTM_REVISION,
    expected_test_source_count=EXPECTED_TEST_SOURCE_COUNT,
    expected_zinc_source_count=EXPECTED_ZINC_SOURCE_COUNT,
    validation_member_count=VALIDATION_MEMBER_COUNT,
)


@dataclass(frozen=True)
class MoleculeMember:
    member_id: str
    source_member_index: int
    source_line_number: int
    raw_smiles: str
    canonical_isomeric_smiles: str
    canonical_connectivity_smiles: str

    @property
    def connectivity_identity_sha256(self) -> str:
        return sha256_text(self.canonical_connectivity_smiles)

    @property
    def stereo_identity_sha256(self) -> str:
        return sha256_text(self.canonical_isomeric_smiles)


class SplitMix64:
    """Small version-stable pseudorandom generator for membership replay."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (value ^ (value >> 31)) & self.MASK

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


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


def observe_file(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            byte_count += len(block)
            digest.update(block)
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def canonical_forms(smiles: str) -> Optional[Tuple[str, str]]:
    try:
        forms = identity_normalization.canonical_forms_from_smiles(smiles)
    except identity_normalization.IdentityNormalizationError:
        return None
    return forms.strict_isomeric_smiles, forms.connectivity_smiles


def _is_production_binding(binding: SourceBinding) -> bool:
    return (
        binding.repository_id == MOLECULESTM_REPOSITORY_ID
        and binding.repository_url == MOLECULESTM_REPOSITORY_URL
        and binding.revision == MOLECULESTM_REVISION
        and binding.expected_test_source_count == EXPECTED_TEST_SOURCE_COUNT
        and binding.expected_zinc_source_count == EXPECTED_ZINC_SOURCE_COUNT
        and binding.validation_member_count == VALIDATION_MEMBER_COUNT
        and binding.selection_seed == SELECTION_SEED
    )


def validate_binding(binding: SourceBinding) -> None:
    if not binding.repository_id.strip() or not binding.revision.strip():
        raise ControlledEditingProtocolError(
            "repository identity and revision must be explicit"
        )
    if not binding.repository_url.startswith("https://"):
        raise ControlledEditingProtocolError("repository URL must use HTTPS")
    if binding.expected_test_source_count <= 0:
        raise ControlledEditingProtocolError("test source count must be positive")
    if binding.expected_zinc_source_count <= 0:
        raise ControlledEditingProtocolError("ZINC source count must be positive")
    if binding.validation_member_count <= 0:
        raise ControlledEditingProtocolError("validation member count must be positive")
    if binding.validation_member_count > binding.expected_zinc_source_count:
        raise ControlledEditingProtocolError(
            "validation size cannot exceed the declared ZINC population"
        )
    if _is_production_binding(binding) and rdBase.rdkitVersion != PRODUCTION_RDKIT_VERSION:
        raise ControlledEditingProtocolError(
            "production controlled-editing canonicalization requires RDKit "
            + PRODUCTION_RDKIT_VERSION
            + "; observed "
            + rdBase.rdkitVersion
        )


def _require_source_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ControlledEditingProtocolError(
            label + " must be one regular non-symlink file"
        )


def read_sealed_test(path: Path, binding: SourceBinding) -> List[MoleculeMember]:
    members: List[MoleculeMember] = []
    seen_raw = set()
    with path.open("r", encoding="utf-8") as handle:
        physical_lines = list(handle)
    if len(physical_lines) != binding.expected_test_source_count:
        raise ControlledEditingProtocolError(
            "sealed test physical-line count differs from the frozen source"
        )
    for source_index, physical_line in enumerate(physical_lines):
        smiles = physical_line.strip()
        if not smiles:
            raise ControlledEditingProtocolError("sealed test contains an empty source row")
        if smiles in seen_raw:
            raise ControlledEditingProtocolError(
                "sealed test source SMILES rows must be unique"
            )
        seen_raw.add(smiles)
        forms = canonical_forms(smiles)
        if forms is None:
            raise ControlledEditingProtocolError(
                "sealed test source row {} is not RDKit-parseable".format(source_index)
            )
        isomeric, connectivity = forms
        members.append(
            MoleculeMember(
                member_id="controlled-editing-sealed-test-source-row:{:06d}".format(
                    source_index
                ),
                source_member_index=source_index,
                source_line_number=source_index + 1,
                raw_smiles=smiles,
                canonical_isomeric_smiles=isomeric,
                canonical_connectivity_smiles=connectivity,
            )
        )
    return members


def _strict_smiles_cell(
    row: Mapping[Optional[str], object], source_index: int
) -> str:
    if None in row:
        raise ControlledEditingProtocolError(
            "ZINC source row {} has extra CSV fields".format(source_index)
        )
    raw_value = row.get("smiles")
    if not isinstance(raw_value, str):
        raise ControlledEditingProtocolError(
            "ZINC source row {} has no string SMILES cell".format(source_index)
        )
    smiles = raw_value.strip()
    if not smiles:
        raise ControlledEditingProtocolError(
            "ZINC source row {} has an empty SMILES cell".format(source_index)
        )
    return smiles


def read_zinc_population(path: Path, binding: SourceBinding) -> List[MoleculeMember]:
    members: List[MoleculeMember] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ZINC_COLUMNS:
            raise ControlledEditingProtocolError(
                "ZINC CSV header must be exactly " + ",".join(ZINC_COLUMNS)
            )
        for source_index, row in enumerate(reader):
            if set(row) != set(ZINC_COLUMNS):
                raise ControlledEditingProtocolError(
                    "ZINC source row {} violates the exact CSV schema".format(
                        source_index
                    )
                )
            smiles = _strict_smiles_cell(row, source_index)
            forms = canonical_forms(smiles)
            if forms is None:
                raise ControlledEditingProtocolError(
                    "ZINC source row {} is not RDKit-parseable".format(source_index)
                )
            isomeric, connectivity = forms
            members.append(
                MoleculeMember(
                    member_id="zinc250k-source-row:{:06d}".format(source_index),
                    source_member_index=source_index,
                    source_line_number=reader.line_num,
                    raw_smiles=smiles,
                    canonical_isomeric_smiles=isomeric,
                    canonical_connectivity_smiles=connectivity,
                )
            )
    if len(members) != binding.expected_zinc_source_count:
        raise ControlledEditingProtocolError(
            "ZINC source population differs from the frozen full population"
        )
    return members


def _shuffle_splitmix64(values: List[MoleculeMember], seed: int) -> None:
    generator = SplitMix64(seed)
    for index in range(len(values) - 1, 0, -1):
        swap_index = generator.randbelow(index + 1)
        values[index], values[swap_index] = values[swap_index], values[index]


def select_validation_members(
    zinc_members: Sequence[MoleculeMember],
    test_members: Sequence[MoleculeMember],
    *,
    validation_member_count: int,
    seed: int,
) -> Tuple[List[MoleculeMember], Dict[str, int]]:
    test_connectivity = {
        member.connectivity_identity_sha256 for member in test_members
    }
    ordered = sorted(zinc_members, key=lambda member: member.member_id.encode("utf-8"))
    representatives: Dict[str, MoleculeMember] = {}
    for member in ordered:
        representatives.setdefault(member.connectivity_identity_sha256, member)
    candidates = [
        member
        for connectivity, member in representatives.items()
        if connectivity not in test_connectivity
    ]
    candidates.sort(key=lambda member: member.member_id.encode("utf-8"))
    if len(candidates) < validation_member_count:
        raise ControlledEditingProtocolError(
            "too few connectivity-disjoint ZINC candidates for validation"
        )
    shuffled = list(candidates)
    _shuffle_splitmix64(shuffled, seed)
    selected = shuffled[:validation_member_count]
    selected.sort(key=lambda member: member.member_id.encode("utf-8"))

    selected_connectivity = {
        member.connectivity_identity_sha256 for member in selected
    }
    if len(selected_connectivity) != len(selected):
        raise RuntimeError("validation connectivity deduplication failed")
    if selected_connectivity & test_connectivity:
        raise RuntimeError("validation/test connectivity overlap survived exclusion")
    return selected, {
        "zinc_source_members_scanned": len(zinc_members),
        "zinc_unique_connectivity_members": len(representatives),
        "zinc_connectivity_duplicates_removed": len(zinc_members) - len(representatives),
        "zinc_unique_connectivities_excluded_by_test": (
            len(representatives) - len(candidates)
        ),
        "eligible_validation_candidates": len(candidates),
        "selected_validation_members": len(selected),
    }


def _key_lf_digest(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["member_id"]).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _artifact(payload: bytes, path: str, row_count: Optional[int] = None) -> Dict[str, object]:
    result: Dict[str, object] = {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _identity_spec_id() -> str:
    if not IDENTITY_CONTRACT.is_file():
        raise ControlledEditingProtocolError(
            "shared identity normalization contract is missing: "
            + str(IDENTITY_CONTRACT)
        )
    return str(observe_file(IDENTITY_CONTRACT)["sha256"])


def _collection_payloads(
    *,
    split: str,
    members: Sequence[MoleculeMember],
    source_manifest_sha256: str,
    identity_spec_id: str,
    extractor_sha256: str,
) -> Tuple[bytes, bytes, Dict[str, object]]:
    collection_id = "controlled-motif-editing-{}-identity-v2".format(split)
    rows = [
        {
            "schema_version": IDENTITY_ROW_SCHEMA,
            "collection_id": collection_id,
            "member_id": member.member_id,
            "connectivity_identity_sha256": member.connectivity_identity_sha256,
            "stereo_identity_sha256": member.stereo_identity_sha256,
            "conformer_identity_sha256": None,
        }
        for member in members
    ]
    rows.sort(key=lambda row: str(row["member_id"]).encode("utf-8"))
    rows_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    rows_artifact = _artifact(
        rows_payload, "molecule_identity_rows.jsonl", row_count=len(rows)
    )
    rows_artifact["key_lf_sha256"] = _key_lf_digest(rows)
    manifest = {
        "schema_version": COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "phase": "downstream",
        "split": split,
        "role": "downstream_" + split,
        "task_family": TASK_FAMILY,
        "identity_specs": {
            "connectivity_identity_spec_sha256": identity_spec_id,
            "stereo_identity_spec_sha256": identity_spec_id,
            "conformer_identity": {"status": "unavailable", "spec_sha256": None},
            "text_identity": {
                "status": "unavailable",
                "exact_spec_sha256": None,
                "normalized_spec_sha256": None,
            },
        },
        "molecule_rows": rows_artifact,
        "text_pair_rows": None,
        "provenance": {
            "source_identity_namespace": (
                "moleculestm_editing_source_row"
                if split == "test"
                else "moleculestm_zinc250k_source_row"
            ),
            "source_release_manifest_sha256": source_manifest_sha256,
            "extractor_sha256": extractor_sha256,
            "excluded_source_metadata_keys": (
                [] if split == "test" else ["logP", "qed", "SAS"]
            ),
        },
    }
    proof.validate_collection_manifest(manifest)
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    return rows_payload, manifest_payload, manifest


def build_controlled_editing_memberships(
    sealed_test_smiles: Path,
    zinc_csv: Path,
    output_dir: Path,
    *,
    source_binding: SourceBinding,
) -> Dict[str, object]:
    if output_dir.exists():
        raise ControlledEditingProtocolError("output directory must not already exist")
    validate_binding(source_binding)
    _require_source_file(sealed_test_smiles, "sealed test source")
    _require_source_file(zinc_csv, "ZINC source")

    test_observation = observe_file(sealed_test_smiles)
    zinc_observation = observe_file(zinc_csv)
    test_members = read_sealed_test(sealed_test_smiles, source_binding)
    zinc_members = read_zinc_population(zinc_csv, source_binding)
    validation_members, census = select_validation_members(
        zinc_members,
        test_members,
        validation_member_count=source_binding.validation_member_count,
        seed=source_binding.selection_seed,
    )

    test_connectivity = {
        member.connectivity_identity_sha256 for member in test_members
    }
    validation_connectivity = {
        member.connectivity_identity_sha256 for member in validation_members
    }
    test_member_ids = {member.member_id for member in test_members}
    validation_member_ids = {member.member_id for member in validation_members}
    if len(test_member_ids) != len(test_members):
        raise RuntimeError("sealed test member IDs are not unique")
    if len(validation_member_ids) != len(validation_members):
        raise RuntimeError("validation member IDs are not unique")
    if test_connectivity & validation_connectivity:
        raise RuntimeError("validation and sealed test are not connectivity-disjoint")

    source_manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "dataset_id": DATASET_ID,
        "repository": {
            "authority": source_binding.authority,
            "repository_id": source_binding.repository_id,
            "url": source_binding.repository_url,
            "revision": source_binding.revision,
        },
        "source_files": {
            "sealed_test": {
                "origin_relative_path": TEST_ORIGIN_PATH,
                "format": "one_SMILES_per_nonempty_line",
                "semantic_population_count": len(test_members),
                "file_integrity_observation": test_observation,
            },
            "zinc250k": {
                "origin_relative_path": ZINC_ORIGIN_PATH,
                "format": "CSV",
                "columns_exact_order": list(ZINC_COLUMNS),
                "smiles_column": "smiles",
                "smiles_cell_normalization": "strip_surrounding_whitespace_including_embedded_terminal_newline",
                "semantic_population_count": len(zinc_members),
                "file_integrity_observation": zinc_observation,
            },
        },
        "canonicalization": {
            "library": "RDKit",
            "observed_version": rdBase.rdkitVersion,
            "production_required_version": PRODUCTION_RDKIT_VERSION,
            "function": "Chem.MolToSmiles",
            "connectivity_parameters": {
                "canonical": True,
                "isomericSmiles": False,
                "kekuleSmiles": False,
            },
            "stereo_parameters": {
                "canonical": True,
                "isomericSmiles": True,
                "kekuleSmiles": False,
            },
        },
        "admission_criteria": {
            "hard": [
                "declared_MoleculeSTM_repository_and_revision",
                "sealed_test_exactly_200_nonempty_200_raw_unique_parseable_source_rows",
                "ZINC_exact_columns_and_complete_249455_row_population",
                "all_source_SMILES_parseable_under_RDKit_2024_03_5",
                "validation_size_400_after_connectivity_deduplication_and_test_exclusion",
                "validation_test_connectivity_disjointness_and_unique_member_assignment",
            ],
            "not_hard": ["file_sha256", "file_byte_count"],
        },
        "file_integrity_observation_role": "provenance_only_not_scientific_admission_gate",
    }
    source_payload = canonical_json_bytes(source_manifest) + b"\n"
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    identity_spec_id = _identity_spec_id()
    extractor_sha256 = str(observe_file(Path(__file__).resolve())["sha256"])

    collection_payloads = {}
    collection_summaries = []
    for split, members in (
        ("validation", validation_members),
        ("test", test_members),
    ):
        rows_payload, manifest_payload, manifest = _collection_payloads(
            split=split,
            members=members,
            source_manifest_sha256=source_sha256,
            identity_spec_id=identity_spec_id,
            extractor_sha256=extractor_sha256,
        )
        relative_dir = Path("collections") / split
        collection_payloads[split] = (relative_dir, rows_payload, manifest_payload)
        collection_summaries.append(
            {
                "collection_id": manifest["collection_id"],
                "split": split,
                "role": manifest["role"],
                "member_count": len(members),
                "manifest_relative_path": (
                    relative_dir / "collection_manifest.json"
                ).as_posix(),
            }
        )

    worst_case_standard_error = (0.25 / source_binding.validation_member_count) ** 0.5
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "dataset_id": DATASET_ID,
        "release_id": RELEASE_ID,
        "source_manifest": _artifact(source_payload, SOURCE_MANIFEST_FILENAME),
        "prompt_namespace": {
            "source": "FineMolTex published implementation",
            "prompt_ids": list(PROMPT_IDS),
            "prompt_count": len(PROMPT_IDS),
            "pairing_status": "namespace_only_no_cartesian_or_explicit_molecule_prompt_pairs_assumed",
        },
        "membership_protocol": {
            "supervised_train": None,
            "sealed_test_source_members": len(test_members),
            "validation_members": len(validation_members),
            "validation_size_rationale": {
                "estimand": "binomial_proportion",
                "worst_case_p": 0.5,
                "normal_approximation_z": 1.96,
                "standard_error": worst_case_standard_error,
                "approximate_95_percent_half_width_percentage_points": (
                    100.0 * 1.96 * worst_case_standard_error
                ),
                "interpretation": "approximately_plus_or_minus_5_percentage_points",
            },
            "candidate_order": "ascending_UTF8_bytes_of_zero_padded_source_row_member_id",
            "connectivity_deduplication": "retain_first_member_in_candidate_order_per_connectivity",
            "test_exclusion_key": "connectivity_identity_sha256",
            "rng": {
                "algorithm": "SplitMix64_v1_plus_descending_Fisher_Yates",
                "seed": source_binding.selection_seed,
                "integer_sampling": "64_bit_rejection_sampling_then_modulo",
                "selection": "first_k_after_shuffle_then_sort_selected_members_by_source_key",
            },
        },
        "census": {
            "sealed_test_source_rows": len(test_members),
            "sealed_test_unique_raw_smiles": len(
                {member.raw_smiles for member in test_members}
            ),
            "sealed_test_unique_connectivities": len(test_connectivity),
            **census,
        },
        "invariants": {
            "sealed_test_all_nonempty_unique_raw_and_parseable": True,
            "zinc_full_population_scanned_and_parseable": True,
            "validation_connectivity_unique": True,
            "validation_test_connectivity_disjoint": True,
            "every_output_member_assigned_exactly_once": True,
            "sealed_test_used_only_for_connectivity_exclusion_from_validation": True,
            "sealed_test_not_used_for_model_or_hyperparameter_selection": True,
            "no_supervised_train_or_text_pairs_materialized": True,
            "output_selection_deterministic_given_declared_sources_revision_and_rdkit": True,
        },
        "identity_spec_id": identity_spec_id,
        "collections": collection_summaries,
        "digest_role": "artifact_and_provenance_observation_not_scientific_admission_gate",
    }
    summary_payload = canonical_json_bytes(summary) + b"\n"

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_new(output_dir / SOURCE_MANIFEST_FILENAME, source_payload)
    for relative_dir, rows_payload, manifest_payload in collection_payloads.values():
        _write_new(output_dir / relative_dir / "molecule_identity_rows.jsonl", rows_payload)
        _write_new(output_dir / relative_dir / "collection_manifest.json", manifest_payload)
    _write_new(output_dir / SUMMARY_FILENAME, summary_payload)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-test-smiles", required=True, type=Path)
    parser.add_argument("--zinc-csv", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_args(argv)
    if arguments.source_revision != MOLECULESTM_REVISION:
        raise ControlledEditingProtocolError(
            "production source revision must equal the frozen MoleculeSTM revision"
        )
    summary = build_controlled_editing_memberships(
        arguments.sealed_test_smiles,
        arguments.zinc_csv,
        arguments.output_dir,
        source_binding=PRODUCTION_SOURCE_BINDING,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "validation_members": summary["membership_protocol"][
                    "validation_members"
                ],
                "test_members": summary["membership_protocol"][
                    "sealed_test_source_members"
                ],
                "output_dir": str(arguments.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ControlledEditingProtocolError, OSError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
