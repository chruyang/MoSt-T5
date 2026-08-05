#!/usr/bin/env python3
"""Build a bounded, geometry-only PCQM4Mv2 P1 sidecar smoke.

This R1 adapter is intentionally *not* a training-data builder.  It streams a
deterministic prefix of the frozen OGB train-3D SDF on the remote host, checks
the official train-split/CSV identity association, and writes only a bounded
pre-tokenizer geometry sidecar with an explicit membership/reject partition.

The output is non-admissible by construction: it contains no token IDs, no
tokenizer binding, no dynamic mask plan, and no P1 launcher permission.  It
exists to validate the one-Mol dataflow before any full topology census or
frozen tokenizer is created.

The feature branch is deliberately single-source:

    streamed SDF Mol -> tagged minimal-H projection -> geometry_mol
        -> molecule-native motifs, conformer coordinates, and E3FP

CSV SMILES are parsed only for strict identity comparison.  Their atom order
never enters a coordinate, E3FP, or motif mapping.  The script has no full
mode and refuses more than 1,000 records; it neither extracts the SDF nor
downloads any dataset to the local workstation.

Run this only on the remote host after copying this small harness there::

  python -B build_pcqm_p1_geometry_sidecar.py --mode smoke ...

The corresponding independent reader is
``../gates/validate_p1_pcqm_geometry_sidecar.py``.  A successful smoke is
evidence for the bounded dataflow only, never P1 admission or training.
"""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path


MAX_SMOKE_RECORDS = 1000
MIN_MAP_SIZE_MIB = 64
MAX_MAP_SIZE_MIB = 512
SIDE_CAR_SCHEMA = "most-t5-r1/p1-pcqm-geometry-smoke/v2"
RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-pretokenizer-record/v2"
SIDE_CAR_MODE = "bounded_smoke_only"
IDENTITY_NAMESPACE = "ogb_pcqm4mv2_train_row_index"
SOURCE_ATOM_TAG = "_r1_source_atom_index"
SOURCE_ADDRESS_SCHEMA = "most-t5-r1/pcqm-source-address/v1"
PAYLOAD_INDEX_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-index-row/v2"
PAYLOAD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-payload/v2"
ADAPTER_LOCK_SCHEMA = "most-t5-r1/p1-pcqm-geometry-adapter-lock/v2"
BUILD_REPORT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-build-report/v2"
RELEASE_ROOT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-release-root/v2"

# These are deliberately closed, raw-free diagnostic tokens.  They may be
# emitted to the reject ledger and are also frozen in the v2 record contract.
# RDKit/E3FP exception messages and CSV values never cross this boundary.
PREFLIGHT_DIAGNOSTIC_CODES = frozenset(
    (
        "preflight_sdf_conformer_count_not_one",
        "preflight_sdf_conformer_access_failed",
        "preflight_atom_coordinate_count_mismatch",
        "preflight_nonfinite_coordinates",
        "preflight_zero_source_atoms",
        "preflight_source_atom_tag_missing",
        "preflight_source_atom_tag_not_integer",
        "preflight_source_atom_tag_out_of_range",
        "preflight_source_atom_tag_not_unique",
        "preflight_source_atom_tag_domain_invalid",
        "preflight_source_atom_tag_order_not_preserved",
        "preflight_hydrogen_projection_failed",
        "preflight_zero_model_atoms",
        "preflight_hydrogen_projection_residual_h",
        "preflight_geometry_non_e3fp_atom",
        "preflight_e3fp_shell_radius_invalid",
        "preflight_e3fp_radius_multiplier_invalid",
        "preflight_e3fp_shell_radius_unmappable",
        "preflight_e3fp_shell_center_invalid",
        "preflight_e3fp_shell_center_out_of_range",
        "preflight_e3fp_shell_level_above_requested",
        "preflight_e3fp_duplicate_center_radius_slot",
        "preflight_e3fp_shell_identifier_missing",
        "preflight_e3fp_identifier_fold_failed",
        "preflight_e3fp_identifier_out_of_range",
        "preflight_e3fp_no_shells",
        "preflight_e3fp_shape_invalid",
        "preflight_e3fp_value_range_invalid",
        "preflight_e3fp_level0_missing",
        "preflight_e3fp_all_padding_model_row",
        "preflight_e3fp_generation_failed",
        "preflight_e3fp_empty_fingerprint_result",
        "preflight_e3fp_resolved_config_mismatch",
    )
)
CLOSED_DIAGNOSTIC_CODES = frozenset(
    (
        "sdf_rdkit_none",
        "csv_idx_not_integer",
        "csv_idx_row_index_mismatch",
        "csv_smiles_missing",
        "csv_row_unresolved",
        "official_smiles_parse_failed",
        "official_canonicalization_failed",
        "strict_mismatch_connectivity_match",
        "connectivity_mismatch",
        "feature_coordinates_invalid",
        "linearizer_failed",
    )
) | PREFLIGHT_DIAGNOSTIC_CODES

CLOSED_REJECT_REASON_TO_STAGE = {
    "SDF_PARSE_FAILED": "sdf_parse",
    "SDF_CONFORMER_INVALID": "sdf_parse",
    "NONFINITE_COORDINATES": "sdf_parse",
    "PCQM_PARSE_OR_NORMALIZATION_ERROR": "identity",
    "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH": "identity",
    "PCQM_STEREO_2D3D_DIVERGENCE": "identity",
    "HYDROGEN_PROJECTION_FAILED": "hydrogen_projection",
    "HYDROGEN_PROJECTION_RESIDUAL_H": "hydrogen_projection",
    "SOURCE_ATOM_INDEX_TAG_INVALID": "source_atom_index",
    "ZERO_MODEL_ATOMS": "source_atom_index",
    "MOTIF_LINEARIZATION_FAILED": "topology",
    "MOTIF_MAPPING_INVALID": "topology",
    "E3FP_GENERATION_FAILED": "e3fp",
    "E3FP_SHAPE_OR_RANGE_INVALID": "e3fp",
    "DOWNSTREAM_EVAL_IDENTITY_OVERLAP": "downstream_exclusion",
    "SERIALIZATION_VALIDATION_FAILED": "serialization",
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value):
    return sha256_bytes(canonical_json_bytes(value))


def regular_file(path, label):
    result = Path(path).expanduser()
    if not result.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, result))
    return result.resolve()


def load_json(path, label):
    target = regular_file(path, label)
    try:
        with open(str(target), "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise RuntimeError("cannot parse {}: {}".format(label, type(exc).__name__)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("{} must contain a JSON object".format(label))
    return target, value


def import_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct module spec for {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def require_exact_keys(mapping, expected, label):
    observed = set(mapping.keys())
    expected = set(expected)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            "{} keys differ from the frozen schema (missing={}, extra={})".format(
                label, missing, extra
            )
        )


def write_json_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing sidecar file: {}".format(path))
    with open(str(path), "x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl_line(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def array_descriptor(np, value):
    if not isinstance(value, np.ndarray):
        raise TypeError("native array expected")
    if not value.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")
    return {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "order": "C",
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def logical_projection(np, value):
    """Project native arrays to a stable, raw-free logical hash representation."""
    if isinstance(value, np.ndarray):
        return {"__ndarray__": array_descriptor(np, value)}
    if isinstance(value, dict):
        return {str(key): logical_projection(np, value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [logical_projection(np, item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("unsupported record value for logical hash: {}".format(type(value).__name__))


def logical_record_sha256(np, record):
    return sha256_json(logical_projection(np, record))


def sha256_selected_ordinals(selected_ordinals):
    return sha256_json([int(item) for item in selected_ordinals])


def storage_key(ordinal):
    if ordinal < 0 or ordinal >= 1_000_000_000:
        raise ValueError("ordinal cannot be represented by a 9-digit storage key")
    return "{:09d}".format(int(ordinal))


def member_id(ordinal):
    return "{}:{}".format(IDENTITY_NAMESPACE, int(ordinal))


def source_address_sha256(archive_sha256, sdf_tar_member, ordinal, csv_row):
    if not isinstance(sdf_tar_member, dict):
        raise TypeError("locked SDF tar-member observation must be an object")
    member_name = sdf_tar_member.get("tar_member_name")
    member_sha256 = sdf_tar_member.get("sha256")
    member_bytes = sdf_tar_member.get("uncompressed_bytes")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64 or set(archive_sha256) - _HEX64:
        raise ValueError("source archive SHA-256 is invalid")
    if (
        not isinstance(member_name, str)
        or not member_name
        or not isinstance(member_sha256, str)
        or len(member_sha256) != 64
        or set(member_sha256) - _HEX64
        or sdf_tar_member.get("member_type") != "regular_file"
        or not isinstance(member_bytes, int)
        or isinstance(member_bytes, bool)
        or member_bytes < 0
    ):
        raise ValueError("locked SDF tar-member observation is incomplete")
    require_nonnegative_int(ordinal, "source address SDF ordinal")
    require_nonnegative_int(csv_row, "source address official CSV row")
    return sha256_json(
        {
            "address_schema_version": SOURCE_ADDRESS_SCHEMA,
            "archive_sha256": archive_sha256,
            "identity_namespace": IDENTITY_NAMESPACE,
            "official_csv_row_index": int(csv_row),
            "sdf_record_index": int(ordinal),
            "sdf_tar_member_name": member_name,
            "sdf_tar_member_sha256": member_sha256,
        }
    )


def find_locked_sdf_member(archive, sdf_tar_member):
    """Select only the exact tar member already bound by source integrity."""
    if not isinstance(sdf_tar_member, dict):
        raise TypeError("locked SDF tar-member observation must be an object")
    expected_name = sdf_tar_member.get("tar_member_name")
    expected_bytes = sdf_tar_member.get("uncompressed_bytes")
    if not isinstance(expected_name, str) or not isinstance(expected_bytes, int):
        raise ValueError("locked SDF tar-member observation is incomplete")
    for member in archive:
        if member.name != expected_name:
            continue
        if not member.isfile() or int(member.size) != expected_bytes:
            raise RuntimeError("streamed tar member differs from the locked SDF member specification")
        return member
    raise RuntimeError("streamed archive does not contain the locked SDF member")


def molecule_identity_sha256(Chem, np, mol):
    """Hash source/geometry structure and coordinates without serializing SMILES."""
    conformer_count = int(mol.GetNumConformers())
    coordinates_sha = None
    if conformer_count == 1:
        coordinates = np.ascontiguousarray(
            np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        )
        coordinates_sha = sha256_bytes(coordinates.tobytes(order="C"))
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(
            {
                "atomic_num": int(atom.GetAtomicNum()),
                "aromatic": bool(atom.GetIsAromatic()),
                "formal_charge": int(atom.GetFormalCharge()),
                "isotope": int(atom.GetIsotope()),
                "chiral_tag": str(atom.GetChiralTag()),
            }
        )
    bonds = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin > end:
            begin, end = end, begin
        bonds.append(
            {
                "begin": begin,
                "end": end,
                "bond_type": str(bond.GetBondType()),
                "aromatic": bool(bond.GetIsAromatic()),
                "stereo": str(bond.GetStereo()),
            }
        )
    bonds.sort(key=lambda item: (item["begin"], item["end"], item["bond_type"], item["stereo"]))
    return sha256_json(
        {
            "atom_count": int(mol.GetNumAtoms()),
            "atoms": atoms,
            "bond_count": int(mol.GetNumBonds()),
            "bonds": bonds,
            "conformer_count": conformer_count,
            "coordinates_float64_sha256": coordinates_sha,
        }
    )


def validate_source_contract(source_integrity, source_contract_path, archive_path, data_csv_path, split_dict_path):
    """Verify every PCQM input before any parser or torch deserializer runs."""
    verified_inputs = source_integrity.verify_pcqm_inputs(
        source_contract_path, archive_path, data_csv_path, split_dict_path
    )
    archive = verified_inputs.artifact("train_3d_sdf_archive")
    return verified_inputs, {
        "archive_sha256": archive["sha256"],
        "source_record_count": int(verified_inputs.source_record_count),
        "verified_input_lock": verified_inputs.report(),
    }


def validate_identity_contract(contract):
    if contract.get("schema_version") != "most-t5-r1/pcqm4mv2-identity-normalization-contract/v1":
        raise RuntimeError("identity normalization contract schema is not the expected R1 v1")
    branch = contract.get("single_source_feature_branch")
    if not isinstance(branch, dict) or branch.get("feature_mol") != "the post-projection SDF RDKit Mol for one admitted SDF ordinal":
        raise RuntimeError("identity normalization contract does not lock the required feature_mol branch")


def validate_record_schema(contract, max_records):
    if contract.get("schema_version") != "most-t5-r1/p1-pcqm-geometry-record-schema-contract/v2":
        raise RuntimeError("record schema contract is not the expected R1 geometry v2")
    scope = contract.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError("record schema contract lacks scope")
    if scope.get("sidecar_mode") != "bounded_smoke_only":
        raise RuntimeError("record schema does not permit bounded smoke mode")
    if scope.get("p1_training_admission") is not False or scope.get("p1_training_launcher_permitted") is not False:
        raise RuntimeError("record schema must explicitly forbid P1 admission/launcher")
    if not isinstance(scope.get("maximum_selected_source_records"), int) or max_records > scope["maximum_selected_source_records"]:
        raise RuntimeError("requested smoke size exceeds the frozen record schema bound")
    bindings = contract.get("source_and_adapter_bindings")
    if not isinstance(bindings, dict) or "p1_geometry_payload_format" not in bindings.get("required_source_locks", []):
        raise RuntimeError("record schema does not require the v2 safe payload contract")
    diagnostics = contract.get("closed_reject_diagnostic_codes")
    if not isinstance(diagnostics, list) or set(diagnostics) != CLOSED_DIAGNOSTIC_CODES:
        raise RuntimeError("record schema diagnostic-code set differs from the frozen v2 builder")


def validate_payload_format_contract(contract):
    if contract.get("schema_version") != "most-t5-r1/p1-pcqm-geometry-payload-format-contract/v2":
        raise RuntimeError("payload format contract is not the expected R1 v2")
    if contract.get("payload_schema_version") != "most-t5-r1/p1-pcqm-geometry-sidecar-payload/v2":
        raise RuntimeError("payload format contract does not bind the expected payload schema")
    index = contract.get("payload_index")
    if not isinstance(index, dict) or index.get("schema_version") != PAYLOAD_INDEX_SCHEMA:
        raise RuntimeError("payload format contract does not bind the expected payload-index schema")
    if contract.get("magic_ascii") != "MST5PCQM2\x00":
        raise RuntimeError("payload format contract does not bind the codec magic")
    framing = contract.get("framing")
    if not isinstance(framing, dict) or framing.get("max_header_bytes") != 16 * 1024 * 1024 or framing.get("max_payload_bytes") != 16 * 1024 * 1024:
        raise RuntimeError("payload format contract does not bind the fixed codec size bounds")
    if tuple(contract.get("header_required_fields", ())) != (
        "payload_schema_version", "record", "array_blocks", "logical_record_sha256"
    ):
        raise RuntimeError("payload format contract header field order differs from the codec")
    if tuple(contract.get("array_block_required_fields", ())) != (
        "index", "dtype", "shape", "order", "offset", "nbytes", "sha256"
    ):
        raise RuntimeError("payload format contract block field order differs from the codec")
    if set(contract.get("allowed_dtypes", ())) != {"int32", "float32", "bool"}:
        raise RuntimeError("payload format contract dtype set differs from the codec")
    prohibited = contract.get("prohibited")
    if not isinstance(prohibited, list) or "pickle" not in prohibited:
        raise RuntimeError("payload format contract must explicitly prohibit pickle")


def validate_adapter_lock(lock_path, lock, builder_path, linearizer_path, e3fp_gate_path, identity_gate_path,
                          source_integrity_path, codec_path, source_contract_path, identity_contract_path,
                          record_schema_path, payload_format_contract_path):
    if lock.get("schema_version") != ADAPTER_LOCK_SCHEMA:
        raise RuntimeError("adapter lock schema is not the expected R1 v2")
    entries = lock.get("components")
    if not isinstance(entries, list):
        raise RuntimeError("adapter lock components must be an array")
    by_name = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("adapter lock component is not an object")
        name = entry.get("name")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("adapter lock component is malformed")
        if name in by_name:
            raise RuntimeError("adapter lock component names must be unique")
        by_name[name] = digest
    expected = {
        "build_pcqm_p1_geometry_sidecar.py": (builder_path, "adapter/build_pcqm_p1_geometry_sidecar.py"),
        "mol_linearizer.py": (linearizer_path, "adapter/mol_linearizer.py"),
        "pcqm_e3fp_preflight.py": (e3fp_gate_path, "gates/pcqm_e3fp_preflight.py"),
        "pcqm_identity_smoke.py": (identity_gate_path, "gates/pcqm_identity_smoke.py"),
        "pcqm_source_integrity.py": (source_integrity_path, "adapter/pcqm_source_integrity.py"),
        "sidecar_v2_codec.py": (codec_path, "adapter/sidecar_v2_codec.py"),
    }
    if set(by_name) != set(expected):
        raise RuntimeError("adapter lock component set does not match the required sidecar harness")
    for entry in entries:
        require_exact_keys(entry, ("name", "relative_harness_path", "sha256"), "adapter lock component")
        expected_relative = expected[entry["name"]][1]
        if entry["relative_harness_path"] != expected_relative:
            raise RuntimeError("adapter lock relative harness path differs for {}".format(entry["name"]))
    for name, (file_path, _) in expected.items():
        observed = sha256_file(file_path)
        if observed != by_name[name]:
            raise RuntimeError("adapter lock SHA mismatch for {}".format(name))
    fixed_contracts = lock.get("fixed_contracts")
    if not isinstance(fixed_contracts, list):
        raise RuntimeError("adapter lock fixed contracts must be an array")
    expected_contracts = {
        "pcqm4mv2_source_contract.json": (source_contract_path, "contracts/pcqm4mv2_source_contract.json"),
        "pcqm4mv2_identity_normalization_contract.json": (identity_contract_path, "contracts/pcqm4mv2_identity_normalization_contract.json"),
        "p1_pcqm_geometry_record_schema.json": (record_schema_path, "contracts/p1_pcqm_geometry_record_schema.json"),
        "p1_pcqm_geometry_payload_format_contract.json": (payload_format_contract_path, "contracts/p1_pcqm_geometry_payload_format_contract.json"),
    }
    observed_contracts = {}
    for entry in fixed_contracts:
        if not isinstance(entry, dict):
            raise RuntimeError("adapter lock fixed contract entry is malformed")
        require_exact_keys(entry, ("name", "relative_contract_path", "sha256"), "adapter lock fixed contract")
        name = entry.get("name")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 or name in observed_contracts:
            raise RuntimeError("adapter lock fixed contract entry is malformed")
        observed_contracts[name] = digest
    if set(observed_contracts) != set(expected_contracts):
        raise RuntimeError("adapter lock fixed contract set does not match the v2 harness")
    for name, (path, expected_relative) in expected_contracts.items():
        matching = next(entry for entry in fixed_contracts if entry["name"] == name)
        if matching["relative_contract_path"] != expected_relative:
            raise RuntimeError("adapter lock relative contract path differs for {}".format(name))
        if observed_contracts[name] != sha256_file(path):
            raise RuntimeError("adapter lock fixed contract SHA mismatch for {}".format(name))
    return sha256_file(lock_path)


def select_prefix_companion_rows(source_integrity, verified_inputs, max_records, allow_unsafe):
    split_dict, load_method = source_integrity.load_verified_split_dict(verified_inputs, allow_unsafe)
    if not isinstance(split_dict, dict) or "train" not in split_dict:
        raise RuntimeError("official split_dict has no train split")
    values = split_dict["train"]
    if len(values) != 3_378_606:
        raise RuntimeError("official train split length differs from the locked OGB SDF count")
    rows = [scalar_to_int(values[index]) for index in range(max_records)]
    if len(set(rows)) != len(rows) or min(rows) < 0:
        raise RuntimeError("official train split prefix has invalid companion row indices")
    return rows, {"split_key": "train", "split_entries": int(len(values)), "load_method": load_method}


def scalar_to_int(value):
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def read_selected_csv_smiles(data_csv_path, selected_rows):
    """Read only selected official rows; return transient raw strings in memory."""
    wanted = set(int(item) for item in selected_rows)
    resolved = {}
    malformed_rows = {}
    with gzip.open(str(data_csv_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "idx" not in fieldnames or "smiles" not in fieldnames:
            raise RuntimeError("official data.csv.gz lacks idx/smiles columns")
        for row_index, row in enumerate(reader):
            if row_index not in wanted:
                if len(resolved) + len(malformed_rows) == len(wanted):
                    break
                continue
            try:
                csv_index = int(row.get("idx", ""))
            except (TypeError, ValueError):
                malformed_rows[row_index] = "csv_idx_not_integer"
                continue
            if csv_index != row_index:
                malformed_rows[row_index] = "csv_idx_row_index_mismatch"
                continue
            value = row.get("smiles")
            if not isinstance(value, str) or not value:
                malformed_rows[row_index] = "csv_smiles_missing"
                continue
            resolved[row_index] = value
            if len(resolved) + len(malformed_rows) == len(wanted):
                break
    unresolved = wanted - set(resolved) - set(malformed_rows)
    for row_index in unresolved:
        malformed_rows[row_index] = "csv_row_unresolved"
    return resolved, malformed_rows


def classify_preflight_rejection(reason_code):
    """Map granular gate reasons into the record schema's closed ledger codes."""
    if reason_code in ("SDF_CONFORMER_COUNT_NOT_ONE", "SDF_CONFORMER_ACCESS_FAILED", "ATOM_COORDINATE_COUNT_MISMATCH"):
        return "SDF_CONFORMER_INVALID", "sdf_parse"
    if reason_code == "NONFINITE_COORDINATES":
        return "NONFINITE_COORDINATES", "sdf_parse"
    if reason_code in (
        "ZERO_SOURCE_ATOMS",
        "SOURCE_ATOM_TAG_MISSING",
        "SOURCE_ATOM_TAG_NOT_INTEGER",
        "SOURCE_ATOM_TAG_OUT_OF_RANGE",
        "SOURCE_ATOM_TAG_NOT_UNIQUE",
        "SOURCE_ATOM_TAG_DOMAIN_INVALID",
        "SOURCE_ATOM_TAG_ORDER_NOT_PRESERVED",
        "GEOMETRY_NON_E3FP_ATOM",
    ):
        return "SOURCE_ATOM_INDEX_TAG_INVALID", "source_atom_index"
    if reason_code == "ZERO_MODEL_ATOMS":
        return "ZERO_MODEL_ATOMS", "source_atom_index"
    if reason_code in ("HYDROGEN_PROJECTION_FAILED",):
        return "HYDROGEN_PROJECTION_FAILED", "hydrogen_projection"
    if reason_code == "HYDROGEN_PROJECTION_RESIDUAL_H":
        return "HYDROGEN_PROJECTION_RESIDUAL_H", "hydrogen_projection"
    if reason_code in ("E3FP_GENERATION_FAILED", "E3FP_EMPTY_FINGERPRINT_RESULT"):
        return "E3FP_GENERATION_FAILED", "e3fp"
    if reason_code.startswith("E3FP_"):
        return "E3FP_SHAPE_OR_RANGE_INVALID", "e3fp"
    raise RuntimeError("unmapped expected preflight rejection code: {}".format(reason_code))


def diagnostic_code_for_preflight(reason_code):
    """Translate a frozen preflight reason to a closed ledger token."""
    if not isinstance(reason_code, str):
        raise RuntimeError("preflight reason code is not a string")
    diagnostic_code = "preflight_{}".format(reason_code.lower())
    if diagnostic_code not in PREFLIGHT_DIAGNOSTIC_CODES:
        raise RuntimeError("preflight reason code has no frozen diagnostic token: {}".format(reason_code))
    return diagnostic_code


def motif_arrays(np, linearizer_result, model_atom_count):
    groups = []
    atom_seen = np.full((model_atom_count,), -1, dtype=np.int32)
    for motif_ordinal, group in enumerate(linearizer_result.motif_atom_groups):
        array = np.ascontiguousarray(np.asarray(group, dtype=np.int32))
        if array.ndim != 1 or array.size == 0:
            raise ValueError("empty/non-vector motif group")
        if not np.array_equal(array, np.sort(array)) or len(set(int(item) for item in array.tolist())) != array.size:
            raise ValueError("motif group is not strictly sorted unique")
        if int(array.min()) < 0 or int(array.max()) >= model_atom_count:
            raise ValueError("motif group index out of range")
        if np.any(atom_seen[array] >= 0):
            raise ValueError("motif groups overlap")
        atom_seen[array] = int(motif_ordinal)
        groups.append(array)
    if not groups or np.any(atom_seen < 0):
        raise ValueError("motif groups do not partition model atoms")
    if len(linearizer_result.fragment_sequence) != len(groups):
        raise ValueError("linearizer fragment/group cardinality mismatch")
    return groups


_FORBIDDEN_RAW_FIELD_NAMES = frozenset(
    (
        "raw_smiles", "smiles", "canonical_smiles", "official_smiles", "sdf_smiles",
        "source_smiles", "generated_smiles", "topology_smiles", "motif_fragment_sequence",
        "reconstructed_mol",
    )
)
_HEX64 = frozenset("0123456789abcdef")


def require_sha256(value, label, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX64:
        raise ValueError("{} must be a lowercase SHA-256{}".format(label, " or null" if nullable else ""))


def require_nonnegative_int(value, label, positive=False):
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        raise ValueError("{} has an invalid integer value".format(label))


def forbid_raw_fields(value, label="record"):
    """Enforce the contract's raw-SMILES prohibition recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("{} contains a non-string field name".format(label))
            if key in _FORBIDDEN_RAW_FIELD_NAMES:
                raise ValueError("{} contains forbidden raw field {}".format(label, key))
            forbid_raw_fields(item, "{}.{}".format(label, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            forbid_raw_fields(item, "{}[{}]".format(label, index))


def validate_admitted_record(np, record):
    forbid_raw_fields(record)
    require_exact_keys(
        record,
        ("record_schema_version", "sidecar", "member", "identity", "atom_universe", "topology", "geometry", "array_metadata"),
        "admitted record",
    )
    if record["record_schema_version"] != RECORD_SCHEMA:
        raise ValueError("record schema version mismatch")
    expected_sidecar = {
        "sidecar_id", "sidecar_mode", "selected_ordinal_set_sha256", "source_contract_sha256", "identity_normalization_contract_sha256",
        "adapter_harness_sha256", "record_schema_sha256", "geometry_only_pretokenizer",
        "p1_training_admission", "p1_training_launcher_permitted",
    }
    require_exact_keys(record["sidecar"], expected_sidecar, "record.sidecar")
    if record["sidecar"]["sidecar_mode"] != SIDE_CAR_MODE or record["sidecar"]["geometry_only_pretokenizer"] is not True:
        raise ValueError("record is not marked as a bounded pretokenizer sidecar")
    if record["sidecar"]["p1_training_admission"] is not False or record["sidecar"]["p1_training_launcher_permitted"] is not False:
        raise ValueError("pretokenizer record incorrectly permits training")
    if not isinstance(record["sidecar"]["sidecar_id"], str) or not record["sidecar"]["sidecar_id"]:
        raise ValueError("record sidecar ID is invalid")
    for key in (
        "selected_ordinal_set_sha256", "source_contract_sha256", "identity_normalization_contract_sha256",
        "adapter_harness_sha256", "record_schema_sha256",
    ):
        require_sha256(record["sidecar"][key], "record.sidecar.{}".format(key))
    expected_member = {
        "identity_namespace", "member_id", "sdf_record_index", "official_csv_row_index", "storage_key",
        "source_archive_sha256", "source_address_sha256", "source_mol_identity_sha256",
    }
    require_exact_keys(record["member"], expected_member, "record.member")
    ordinal = record["member"]["sdf_record_index"]
    if record["member"]["identity_namespace"] != IDENTITY_NAMESPACE or record["member"]["member_id"] != member_id(ordinal):
        raise ValueError("record member identity mismatch")
    if record["member"]["storage_key"] != storage_key(ordinal):
        raise ValueError("record storage key mismatch")
    require_nonnegative_int(ordinal, "record.member.sdf_record_index")
    require_nonnegative_int(record["member"]["official_csv_row_index"], "record.member.official_csv_row_index")
    for key in ("source_archive_sha256", "source_address_sha256", "source_mol_identity_sha256"):
        require_sha256(record["member"][key], "record.member.{}".format(key))
    identity = record["identity"]
    require_exact_keys(
        identity,
        (
            "official_identity_status", "sdf_strict_smiles_sha256", "official_strict_smiles_sha256",
            "canonical_connectivity_sha256", "identity_spec_sha256", "rdkit_version",
        ),
        "record.identity",
    )
    if identity["official_identity_status"] != "strict_isomeric_match":
        raise ValueError("admitted record identity status is not strict match")
    if identity["sdf_strict_smiles_sha256"] != identity["official_strict_smiles_sha256"]:
        raise ValueError("admitted record strict identity hashes differ")
    for key in (
        "sdf_strict_smiles_sha256", "official_strict_smiles_sha256", "canonical_connectivity_sha256",
        "identity_spec_sha256",
    ):
        require_sha256(identity[key], "record.identity.{}".format(key))
    if not isinstance(identity["rdkit_version"], str) or not identity["rdkit_version"]:
        raise ValueError("record identity RDKit version is invalid")
    atom_universe = record["atom_universe"]
    require_exact_keys(
        atom_universe,
        (
            "policy_id", "hydrogen_projection_spec_sha256", "source_atom_count", "source_explicit_hydrogen_count",
            "model_atom_count", "model_to_source_atom_index", "geometry_mol_identity_sha256",
        ),
        "record.atom_universe",
    )
    if atom_universe["policy_id"] != "project_explicit_hydrogens_before_e3fp_v1":
        raise ValueError("record atom-universe policy differs from the frozen policy")
    require_sha256(atom_universe["hydrogen_projection_spec_sha256"], "record.atom_universe.hydrogen_projection_spec_sha256")
    require_sha256(atom_universe["geometry_mol_identity_sha256"], "record.atom_universe.geometry_mol_identity_sha256")
    require_nonnegative_int(atom_universe["source_atom_count"], "record.atom_universe.source_atom_count", positive=True)
    require_nonnegative_int(atom_universe["source_explicit_hydrogen_count"], "record.atom_universe.source_explicit_hydrogen_count")
    require_nonnegative_int(atom_universe["model_atom_count"], "record.atom_universe.model_atom_count", positive=True)
    model_count = int(atom_universe["model_atom_count"])
    mapping = atom_universe["model_to_source_atom_index"]
    if not isinstance(mapping, np.ndarray) or mapping.dtype != np.int32 or mapping.shape != (model_count,) or not mapping.flags.c_contiguous:
        raise ValueError("model_to_source_atom_index native array contract failed")
    if model_count <= 0 or np.any(mapping < 0) or np.any(mapping >= int(atom_universe["source_atom_count"])):
        raise ValueError("source map range contract failed")
    if not np.all(mapping[:-1] < mapping[1:]):
        raise ValueError("source map must be strictly ascending")
    topology = record["topology"]
    require_exact_keys(
        topology,
        ("linearizer_spec_sha256", "motif_count", "motif_atom_indices", "motif_atom_indices_sha256"),
        "record.topology",
    )
    require_sha256(topology["linearizer_spec_sha256"], "record.topology.linearizer_spec_sha256")
    require_sha256(topology["motif_atom_indices_sha256"], "record.topology.motif_atom_indices_sha256")
    require_nonnegative_int(topology["motif_count"], "record.topology.motif_count", positive=True)
    groups = topology["motif_atom_indices"]
    if int(topology["motif_count"]) != len(groups):
        raise ValueError("motif count mismatch")
    atom_seen = np.zeros((model_count,), dtype=np.int8)
    for group in groups:
        if not isinstance(group, np.ndarray) or group.dtype != np.int32 or group.ndim != 1 or group.size == 0 or not group.flags.c_contiguous:
            raise ValueError("motif group array contract failed")
        if np.any(group < 0) or np.any(group >= model_count) or not np.all(group[:-1] < group[1:]):
            raise ValueError("motif group values are invalid")
        atom_seen[group] += 1
    if not np.all(atom_seen == 1):
        raise ValueError("motif groups fail the exact atom partition")
    geometry = record["geometry"]
    require_exact_keys(
        geometry,
        (
            "geometry_valid", "geometry_mse_enabled", "geometry_mse_candidate_after_tokenizer_binding",
            "motif_geometry_valid", "coordinates", "coordinates_sha256", "e3fp", "e3fp_shape",
            "e3fp_params_sha256", "e3fp_sha256",
        ),
        "record.geometry",
    )
    coords = geometry["coordinates"]
    e3fp = geometry["e3fp"]
    valid = geometry["motif_geometry_valid"]
    if not isinstance(coords, np.ndarray) or coords.dtype != np.float32 or coords.shape != (model_count, 3) or not coords.flags.c_contiguous:
        raise ValueError("coordinates native array contract failed")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates are non-finite")
    if not isinstance(e3fp, np.ndarray) or e3fp.dtype != np.int32 or e3fp.shape != (model_count, 4) or not e3fp.flags.c_contiguous:
        raise ValueError("E3FP native array contract failed")
    if np.any(e3fp < -1) or np.any(e3fp > 4095) or np.any(np.all(e3fp == -1, axis=1)) or np.any(e3fp[:, 0] == -1):
        raise ValueError("E3FP value/level-0 contract failed")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.shape != (len(groups),) or not valid.flags.c_contiguous or not bool(np.all(valid)):
        raise ValueError("motif geometry validity contract failed")
    if geometry["geometry_valid"] is not True or geometry["geometry_mse_enabled"] is not False or geometry["geometry_mse_candidate_after_tokenizer_binding"] is not True:
        raise ValueError("pretokenizer geometry status contract failed")
    if geometry["coordinates_sha256"] != array_descriptor(np, coords)["sha256"]:
        raise ValueError("coordinates hash mismatch")
    if geometry["e3fp_sha256"] != array_descriptor(np, e3fp)["sha256"]:
        raise ValueError("E3FP hash mismatch")
    if geometry["e3fp_shape"] != [model_count, 4]:
        raise ValueError("E3FP declared shape mismatch")
    require_sha256(geometry["coordinates_sha256"], "record.geometry.coordinates_sha256")
    require_sha256(geometry["e3fp_params_sha256"], "record.geometry.e3fp_params_sha256")
    require_sha256(geometry["e3fp_sha256"], "record.geometry.e3fp_sha256")
    motif_hash = sha256_json([array_descriptor(np, group) for group in groups])
    if topology["motif_atom_indices_sha256"] != motif_hash:
        raise ValueError("motif-group hash mismatch")
    metadata = record["array_metadata"]
    require_exact_keys(
        metadata,
        (
            "coordinates_dtype", "coordinates_shape", "coordinates_order", "e3fp_dtype", "e3fp_shape",
            "e3fp_order", "model_to_source_atom_index_dtype", "motif_atom_indices_dtype",
        ),
        "record.array_metadata",
    )
    expected_metadata = {
        "coordinates_dtype": "float32",
        "coordinates_shape": [model_count, 3],
        "coordinates_order": "C",
        "e3fp_dtype": "int32",
        "e3fp_shape": [model_count, 4],
        "e3fp_order": "C",
        "model_to_source_atom_index_dtype": "int32",
        "motif_atom_indices_dtype": "int32",
    }
    if metadata != expected_metadata:
        raise ValueError("array metadata differs from native arrays")
    forbidden = {
        "tokenizer_binding", "tokenizer_contract_sha256", "id_to_token_sha256", "full_input_ids", "unmasked_input_ids",
        "motif_ordinal_to_unmasked_token_index", "token_geometry_valid_mask", "joint_mask_positions", "geo_only_mask_positions",
        "geometry_input_mask", "geometry_target_mask", "mask_positions", "raw_smiles", "smiles", "canonical_smiles",
    }
    if forbidden.intersection(record):
        raise ValueError("pretokenizer record includes forbidden training/raw fields")


def make_reject(reason_code, stage, source_address, source_mol_identity, geometry_mol_identity, diagnostic_code):
    """Construct a raw-free, semantically unambiguous v2 reject witness."""
    if CLOSED_REJECT_REASON_TO_STAGE.get(reason_code) != stage:
        raise ValueError("reject reason/stage is not one of the frozen schema pairs")
    require_sha256(source_address, "reject.source_address_sha256")
    require_sha256(source_mol_identity, "reject.source_mol_identity_sha256", nullable=True)
    require_sha256(geometry_mol_identity, "reject.geometry_mol_identity_sha256", nullable=True)
    if diagnostic_code not in CLOSED_DIAGNOSTIC_CODES:
        raise ValueError("reject diagnostic_code is invalid")
    return {
        "reason_code": reason_code,
        "stage": stage,
        "source_address_sha256": source_address,
        "source_mol_identity_sha256": source_mol_identity,
        "geometry_mol_identity_sha256": geometry_mol_identity,
        "diagnostic_code": diagnostic_code,
    }


def build_record(Chem, np, preflight, linearizer, e3fp_api, identity_gate, ordinal, csv_row, raw_official_smiles, source_mol,
                 sidecar_values, archive_sha256, source_address, identity_contract_sha256, projection_spec_sha256,
                 linearizer_spec_sha256, official_input_diagnostic=None):
    """Build one record or return one normalized, raw-free v2 reject object."""
    if source_mol is None:
        return None, make_reject("SDF_PARSE_FAILED", "sdf_parse", source_address, None, None, "sdf_rdkit_none")
    source_mol_identity = None
    geometry_identity = None
    try:
        source_mol_identity = molecule_identity_sha256(Chem, np, source_mol)
        preflight.finite_single_conformer(source_mol, "sdf_parse")
        tagged_source, source_atom_count, _ = preflight.tag_source_atoms(Chem, source_mol)
        source_explicit_h_count = int(sum(atom.GetAtomicNum() == 1 for atom in tagged_source.GetAtoms()))
        geometry_mol, model_to_source = preflight.project_hydrogens(Chem, tagged_source, source_atom_count)
        geometry_identity = molecule_identity_sha256(Chem, np, geometry_mol)
        # Both sides execute the same contract-locked projection/canonicalizer.
        # The SDF-derived geometry Mol remains the only feature source.
        feature_forms = identity_gate.canonical_forms(Chem, geometry_mol)
        if official_input_diagnostic is not None:
            return None, make_reject(
                "PCQM_PARSE_OR_NORMALIZATION_ERROR", "identity", source_address, source_mol_identity,
                geometry_identity, official_input_diagnostic,
            )
        try:
            official_mol = Chem.MolFromSmiles(raw_official_smiles)
        except Exception:
            official_mol = None
        if official_mol is None:
            return None, make_reject(
                "PCQM_PARSE_OR_NORMALIZATION_ERROR", "identity", source_address, source_mol_identity,
                geometry_identity, "official_smiles_parse_failed",
            )
        try:
            official_forms = identity_gate.canonical_forms(Chem, official_mol)
        except Exception:
            return None, make_reject(
                "PCQM_PARSE_OR_NORMALIZATION_ERROR", "identity", source_address, source_mol_identity,
                geometry_identity, "official_canonicalization_failed",
            )
        if feature_forms["strict"] != official_forms["strict"]:
            if feature_forms["connectivity"] == official_forms["connectivity"]:
                return None, make_reject(
                    "PCQM_STEREO_2D3D_DIVERGENCE", "identity", source_address, source_mol_identity,
                    geometry_identity, "strict_mismatch_connectivity_match",
                )
            return None, make_reject(
                "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH", "identity", source_address, source_mol_identity,
                geometry_identity, "connectivity_mismatch",
            )
        coordinates = np.ascontiguousarray(
            np.asarray(geometry_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        )
        if coordinates.shape != (geometry_mol.GetNumAtoms(), 3) or not np.all(np.isfinite(coordinates)):
            return None, make_reject(
                "NONFINITE_COORDINATES", "sdf_parse", source_address, source_mol_identity,
                geometry_identity, "feature_coordinates_invalid",
            )
        e3fp, _, resolved_e3fp = preflight.generate_e3fp(np, e3fp_api, geometry_mol, ordinal)
        e3fp = np.ascontiguousarray(np.asarray(e3fp, dtype=np.int32))
        try:
            result = linearizer.linearize_mol(geometry_mol)
            groups = motif_arrays(np, result, int(geometry_mol.GetNumAtoms()))
        except Exception:
            return None, make_reject(
                "MOTIF_LINEARIZATION_FAILED", "topology", source_address, source_mol_identity,
                geometry_identity, "linearizer_failed",
            )
        motif_valid = np.ascontiguousarray(np.ones((len(groups),), dtype=np.bool_))
        model_to_source = np.ascontiguousarray(np.asarray(model_to_source, dtype=np.int32))
        motif_hash = sha256_json([array_descriptor(np, group) for group in groups])
        e3fp_params_sha256 = sha256_json(resolved_e3fp)
        record = {
            "record_schema_version": RECORD_SCHEMA,
            "sidecar": {
                "sidecar_id": sidecar_values["sidecar_id"],
                "sidecar_mode": SIDE_CAR_MODE,
                "selected_ordinal_set_sha256": sidecar_values["selected_ordinal_set_sha256"],
                "source_contract_sha256": sidecar_values["source_contract_sha256"],
                "identity_normalization_contract_sha256": identity_contract_sha256,
                "adapter_harness_sha256": sidecar_values["adapter_harness_sha256"],
                "record_schema_sha256": sidecar_values["record_schema_sha256"],
                "geometry_only_pretokenizer": True,
                "p1_training_admission": False,
                "p1_training_launcher_permitted": False,
            },
            "member": {
                "identity_namespace": IDENTITY_NAMESPACE,
                "member_id": member_id(ordinal),
                "sdf_record_index": int(ordinal),
                "official_csv_row_index": int(csv_row),
                "storage_key": storage_key(ordinal),
                "source_archive_sha256": archive_sha256,
                "source_address_sha256": source_address,
                "source_mol_identity_sha256": source_mol_identity,
            },
            "identity": {
                "official_identity_status": "strict_isomeric_match",
                "sdf_strict_smiles_sha256": sha256_bytes(feature_forms["strict"].encode("utf-8")),
                "official_strict_smiles_sha256": sha256_bytes(official_forms["strict"].encode("utf-8")),
                "canonical_connectivity_sha256": sha256_bytes(feature_forms["connectivity"].encode("utf-8")),
                "identity_spec_sha256": identity_contract_sha256,
                "rdkit_version": Chem.rdBase.rdkitVersion,
            },
            "atom_universe": {
                "policy_id": "project_explicit_hydrogens_before_e3fp_v1",
                "hydrogen_projection_spec_sha256": projection_spec_sha256,
                "source_atom_count": int(source_atom_count),
                "source_explicit_hydrogen_count": int(source_explicit_h_count),
                "model_atom_count": int(geometry_mol.GetNumAtoms()),
                "model_to_source_atom_index": model_to_source,
                "geometry_mol_identity_sha256": geometry_identity,
            },
            "topology": {
                "linearizer_spec_sha256": linearizer_spec_sha256,
                "motif_count": int(len(groups)),
                "motif_atom_indices": groups,
                "motif_atom_indices_sha256": motif_hash,
            },
            "geometry": {
                "geometry_valid": True,
                "geometry_mse_enabled": False,
                "geometry_mse_candidate_after_tokenizer_binding": True,
                "motif_geometry_valid": motif_valid,
                "coordinates": coordinates,
                "coordinates_sha256": array_descriptor(np, coordinates)["sha256"],
                "e3fp": e3fp,
                "e3fp_shape": [int(e3fp.shape[0]), int(e3fp.shape[1])],
                "e3fp_params_sha256": e3fp_params_sha256,
                "e3fp_sha256": array_descriptor(np, e3fp)["sha256"],
            },
            "array_metadata": {
                "coordinates_dtype": "float32",
                "coordinates_shape": [int(coordinates.shape[0]), int(coordinates.shape[1])],
                "coordinates_order": "C",
                "e3fp_dtype": "int32",
                "e3fp_shape": [int(e3fp.shape[0]), int(e3fp.shape[1])],
                "e3fp_order": "C",
                "model_to_source_atom_index_dtype": "int32",
                "motif_atom_indices_dtype": "int32",
            },
        }
        validate_admitted_record(np, record)
        return record, None
    except preflight.RecordRejected as exc:
        reason, stage = classify_preflight_rejection(exc.reason_code)
        return None, make_reject(
            reason, stage, source_address, source_mol_identity, geometry_identity,
            diagnostic_code_for_preflight(exc.reason_code),
        )
    except Exception:
        # A coding/infrastructure failure must not become a conveniently named
        # per-record rejection; stop and preserve the incomplete sidecar for
        # inspection instead of silently manufacturing a ledger entry.
        raise


def build_membership_row(np, sidecar_id, selected_sha, ordinal, csv_row, source_address, record, reject):
    row = {
        "record_schema_version": RECORD_SCHEMA,
        "sidecar_id": sidecar_id,
        "sidecar_mode": SIDE_CAR_MODE,
        "selected_ordinal_set_sha256": selected_sha,
        "member_id": member_id(ordinal),
        "sdf_record_index": int(ordinal),
        "official_csv_row_index": int(csv_row),
        "source_address_sha256": source_address,
        "disposition": None,
        "record_storage_key": None,
        "record_content_sha256": None,
        "reject_reason_code": None,
    }
    if record is not None:
        if record["member"]["source_address_sha256"] != source_address:
            raise ValueError("admitted record source address does not match membership")
        row.update(
            {
                "disposition": "admit",
                "record_storage_key": storage_key(ordinal),
                "record_content_sha256": logical_record_sha256(np, record),
            }
        )
    else:
        if reject["source_address_sha256"] != source_address:
            raise ValueError("rejected record source address does not match membership")
        row.update({"disposition": "reject", "reject_reason_code": reject["reason_code"]})
    return row


def build_reject_row(sidecar_id, selected_sha, ordinal, csv_row, reject):
    reason = reject["reason_code"]
    stage = reject["stage"]
    source_address = reject["source_address_sha256"]
    detail_sha256 = sha256_json(
        {
            "diagnostic_code": reject["diagnostic_code"],
            "reason_code": reason,
            "source_address_sha256": source_address,
            "stage": stage,
        }
    )
    return {
        "record_schema_version": RECORD_SCHEMA,
        "sidecar_id": sidecar_id,
        "sidecar_mode": SIDE_CAR_MODE,
        "selected_ordinal_set_sha256": selected_sha,
        "member_id": member_id(ordinal),
        "sdf_record_index": int(ordinal),
        "official_csv_row_index": int(csv_row),
        "source_address_sha256": source_address,
        "stage": stage,
        "reason_code": reason,
        "action": "exclude_from_geometry_release",
        "geometry_mse_enabled": False,
        "source_mol_identity_sha256": reject["source_mol_identity_sha256"],
        "geometry_mol_identity_sha256": reject["geometry_mol_identity_sha256"],
        "diagnostic_code": reject["diagnostic_code"],
        "detail_sha256": detail_sha256,
    }


MEMBERSHIP_ROW_FIELDS = (
    "record_schema_version", "sidecar_id", "sidecar_mode", "selected_ordinal_set_sha256",
    "member_id", "sdf_record_index", "official_csv_row_index", "source_address_sha256",
    "disposition", "record_storage_key", "record_content_sha256", "reject_reason_code",
)
REJECT_LEDGER_ROW_FIELDS = (
    "record_schema_version", "sidecar_id", "sidecar_mode", "selected_ordinal_set_sha256",
    "member_id", "sdf_record_index", "official_csv_row_index", "source_address_sha256",
    "stage", "reason_code", "action", "geometry_mse_enabled", "source_mol_identity_sha256",
    "geometry_mol_identity_sha256", "diagnostic_code", "detail_sha256",
)
PAYLOAD_INDEX_ROW_FIELDS = (
    "payload_index_schema_version", "record_storage_key", "record_wire_bytes",
    "record_wire_sha256", "record_content_sha256",
)


def validate_membership_row(row, sidecar_id, selected_sha, ordinal, csv_row, source_address):
    forbid_raw_fields(row, "membership row")
    require_exact_keys(row, MEMBERSHIP_ROW_FIELDS, "membership row")
    if row["record_schema_version"] != RECORD_SCHEMA or row["sidecar_id"] != sidecar_id or row["sidecar_mode"] != SIDE_CAR_MODE:
        raise ValueError("membership row sidecar identity mismatch")
    if row["selected_ordinal_set_sha256"] != selected_sha or row["member_id"] != member_id(ordinal):
        raise ValueError("membership row selected/member identity mismatch")
    if row["sdf_record_index"] != int(ordinal) or row["official_csv_row_index"] != int(csv_row):
        raise ValueError("membership row source ordinal binding mismatch")
    if row["source_address_sha256"] != source_address:
        raise ValueError("membership row source address mismatch")
    require_sha256(row["selected_ordinal_set_sha256"], "membership.selected_ordinal_set_sha256")
    require_sha256(row["source_address_sha256"], "membership.source_address_sha256")
    if row["disposition"] == "admit":
        if row["record_storage_key"] != storage_key(ordinal) or row["reject_reason_code"] is not None:
            raise ValueError("admitted membership conditional fields are invalid")
        require_sha256(row["record_content_sha256"], "membership.record_content_sha256")
    elif row["disposition"] == "reject":
        if row["record_storage_key"] is not None or row["record_content_sha256"] is not None:
            raise ValueError("rejected membership has a persisted-record field")
        if row["reject_reason_code"] not in CLOSED_REJECT_REASON_TO_STAGE:
            raise ValueError("rejected membership reason is not closed")
    else:
        raise ValueError("membership disposition is invalid")


def validate_reject_row(row, sidecar_id, selected_sha, ordinal, csv_row, source_address):
    forbid_raw_fields(row, "reject ledger row")
    require_exact_keys(row, REJECT_LEDGER_ROW_FIELDS, "reject ledger row")
    if row["record_schema_version"] != RECORD_SCHEMA or row["sidecar_id"] != sidecar_id or row["sidecar_mode"] != SIDE_CAR_MODE:
        raise ValueError("reject ledger sidecar identity mismatch")
    if row["selected_ordinal_set_sha256"] != selected_sha or row["member_id"] != member_id(ordinal):
        raise ValueError("reject ledger selected/member identity mismatch")
    if row["sdf_record_index"] != int(ordinal) or row["official_csv_row_index"] != int(csv_row):
        raise ValueError("reject ledger source ordinal binding mismatch")
    if row["source_address_sha256"] != source_address:
        raise ValueError("reject ledger source address mismatch")
    for key in ("selected_ordinal_set_sha256", "source_address_sha256", "detail_sha256"):
        require_sha256(row[key], "reject ledger.{}".format(key))
    require_sha256(row["source_mol_identity_sha256"], "reject ledger.source_mol_identity_sha256", nullable=True)
    require_sha256(row["geometry_mol_identity_sha256"], "reject ledger.geometry_mol_identity_sha256", nullable=True)
    reason = row["reason_code"]
    if CLOSED_REJECT_REASON_TO_STAGE.get(reason) != row["stage"]:
        raise ValueError("reject ledger reason/stage is not closed")
    if row["action"] != "exclude_from_geometry_release" or row["geometry_mse_enabled"] is not False:
        raise ValueError("reject ledger action/MSE flags are invalid")
    if row["diagnostic_code"] not in CLOSED_DIAGNOSTIC_CODES:
        raise ValueError("reject ledger diagnostic code is not closed")
    if row["source_mol_identity_sha256"] is None and not (
        reason == "SDF_PARSE_FAILED" and row["diagnostic_code"] == "sdf_rdkit_none"
    ):
        raise ValueError("only an RDKit-null source record may lack its source Mol identity")
    expected_detail = sha256_json(
        {
            "diagnostic_code": row["diagnostic_code"],
            "reason_code": reason,
            "source_address_sha256": source_address,
            "stage": row["stage"],
        }
    )
    if row["detail_sha256"] != expected_detail:
        raise ValueError("reject ledger detail hash mismatch")


def validate_payload_index_row(row, storage_key_value, record_content_sha256, payload):
    require_exact_keys(row, PAYLOAD_INDEX_ROW_FIELDS, "payload-index row")
    if row["payload_index_schema_version"] != PAYLOAD_INDEX_SCHEMA:
        raise ValueError("payload-index schema mismatch")
    if row["record_storage_key"] != storage_key_value or row["record_content_sha256"] != record_content_sha256:
        raise ValueError("payload-index record binding mismatch")
    if row["record_wire_bytes"] != len(payload) or row["record_wire_sha256"] != sha256_bytes(payload):
        raise ValueError("payload-index wire binding mismatch")
    require_sha256(row["record_wire_sha256"], "payload-index.record_wire_sha256")
    require_sha256(row["record_content_sha256"], "payload-index.record_content_sha256")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke",), default="smoke")
    parser.add_argument("--selector", choices=("prefix",), default="prefix")
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--map-size-mib", type=int, default=256)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--split-dict", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--identity-normalization-contract", required=True)
    parser.add_argument("--record-schema", required=True)
    parser.add_argument("--payload-format-contract", required=True)
    parser.add_argument("--adapter-lock", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-unsafe-legacy-torch-load", action="store_true")
    args = parser.parse_args(argv)
    if args.max_records < 1 or args.max_records > MAX_SMOKE_RECORDS:
        parser.error("--max-records must be in [1, {}]".format(MAX_SMOKE_RECORDS))
    if args.map_size_mib < MIN_MAP_SIZE_MIB or args.map_size_mib > MAX_MAP_SIZE_MIB:
        parser.error("--map-size-mib must be in [{}, {}]".format(MIN_MAP_SIZE_MIB, MAX_MAP_SIZE_MIB))
    return args


def run(args):
    archive_path = regular_file(args.archive, "archive")
    data_csv_path = regular_file(args.data_csv, "official data.csv.gz")
    split_dict_path = regular_file(args.split_dict, "official split_dict.pt")
    source_contract_path, source_contract = load_json(args.source_contract, "source contract")
    identity_contract_path, identity_contract = load_json(args.identity_normalization_contract, "identity normalization contract")
    record_schema_path, record_schema = load_json(args.record_schema, "record schema contract")
    payload_format_contract_path, payload_format_contract = load_json(args.payload_format_contract, "payload format contract")
    adapter_lock_path, adapter_lock = load_json(args.adapter_lock, "adapter lock")

    root = Path(__file__).resolve().parents[1]
    source_integrity_path = root / "adapter" / "pcqm_source_integrity.py"
    if not source_integrity_path.is_file():
        raise FileNotFoundError("PCQM source-integrity helper is absent: {}".format(source_integrity_path))
    source_integrity = import_module_from_file(source_integrity_path, "r1_pcqm_source_integrity_shared")

    # This has to precede all SDF/CSV parsing and every torch.load fallback.
    verified_inputs, source_info = validate_source_contract(
        source_integrity, source_contract_path, archive_path, data_csv_path, split_dict_path
    )
    validate_identity_contract(identity_contract)
    validate_record_schema(record_schema, args.max_records)
    validate_payload_format_contract(payload_format_contract)

    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        raise FileExistsError("--output-dir must be a new path: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        import lmdb
        import numpy as np
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("remote smoke requires lmdb, NumPy, and RDKit") from exc

    linearizer_path = root / "adapter" / "mol_linearizer.py"
    codec_path = root / "adapter" / "sidecar_v2_codec.py"
    e3fp_gate_path = root / "gates" / "pcqm_e3fp_preflight.py"
    identity_gate_path = root / "gates" / "pcqm_identity_smoke.py"
    for path, label in ((linearizer_path, "molecule-native linearizer"), (codec_path, "safe sidecar codec"), (e3fp_gate_path, "E3FP preflight gate"), (identity_gate_path, "identity gate")):
        if not path.is_file():
            raise FileNotFoundError("{} is absent beside this harness: {}".format(label, path))
    adapter_harness_sha256 = validate_adapter_lock(
        adapter_lock_path, adapter_lock, Path(__file__).resolve(), linearizer_path, e3fp_gate_path,
        identity_gate_path, source_integrity_path, codec_path, source_contract_path, identity_contract_path,
        record_schema_path, payload_format_contract_path,
    )
    preflight = import_module_from_file(e3fp_gate_path, "r1_pcqm_e3fp_preflight_shared")
    identity_gate = import_module_from_file(identity_gate_path, "r1_pcqm_identity_shared")
    linearizer = import_module_from_file(linearizer_path, "r1_pcqm_mol_linearizer_shared")
    codec = import_module_from_file(codec_path, "r1_pcqm_sidecar_v2_codec_shared")
    import_root, package_root, e3fp_files = preflight.resolve_e3fp_source(args.e3fp_source)
    e3fp_api = preflight.import_locked_e3fp(import_root, package_root)

    selected_ordinals = list(range(args.max_records))
    selected_sha = sha256_selected_ordinals(selected_ordinals)
    companion_rows, split_observed = select_prefix_companion_rows(
        source_integrity, verified_inputs, args.max_records, args.allow_unsafe_legacy_torch_load
    )
    csv_smiles, csv_malformed = read_selected_csv_smiles(data_csv_path, companion_rows)
    sidecar_id = output_dir.name
    sidecar_values = {
        "sidecar_id": sidecar_id,
        "selected_ordinal_set_sha256": selected_sha,
        "source_contract_sha256": sha256_file(source_contract_path),
        "adapter_harness_sha256": adapter_harness_sha256,
        "record_schema_sha256": sha256_file(record_schema_path),
    }
    identity_contract_sha256 = sha256_file(identity_contract_path)
    projection_spec_sha256 = sha256_json(preflight.HYDROGEN_PROJECTION_PROFILE)
    linearizer_spec_sha256 = sha256_file(linearizer_path)

    scope_manifest = {
        "schema_version": SIDE_CAR_SCHEMA,
        "created_utc": utc_now(),
        "release_status": "non_admissible_pre_tokenizer",
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "tokenizer_binding": "absent_and_forbidden",
        "sidecar_id": sidecar_id,
        "selection": {
            "kind": "prefix",
            "selected_ordinals": "[0,{})".format(args.max_records),
            "selected_record_count": int(args.max_records),
            "selected_ordinal_set_sha256": selected_sha,
        },
        "source": {
            "source_contract_sha256": sha256_file(source_contract_path),
            "source_archive_sha256_observed": source_info["archive_sha256"],
            "source_archive_bytes_observed": int(archive_path.stat().st_size),
            "source_record_count": source_info["source_record_count"],
            "sdf_tar_member": source_info["verified_input_lock"]["train_sdf_member"],
            "source_address_schema_version": SOURCE_ADDRESS_SCHEMA,
            "data_csv_sha256": sha256_file(data_csv_path),
            "split_dict_sha256": sha256_file(split_dict_path),
            "verified_input_lock": source_info["verified_input_lock"],
            "split_loading": split_observed,
        },
        "contracts": {
            "identity_normalization_contract_sha256": identity_contract_sha256,
            "record_schema_sha256": sidecar_values["record_schema_sha256"],
            "payload_format_contract_sha256": sha256_file(payload_format_contract_path),
            "payload_schema_version": PAYLOAD_SCHEMA,
            "payload_index_schema_version": PAYLOAD_INDEX_SCHEMA,
            "adapter_lock_sha256": adapter_harness_sha256,
            "hydrogen_projection_spec_sha256": projection_spec_sha256,
            "linearizer_spec_sha256": linearizer_spec_sha256,
        },
        "harness": {
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "e3fp_gate_sha256": sha256_file(e3fp_gate_path),
            "identity_gate_sha256": sha256_file(identity_gate_path),
            "sidecar_codec_sha256": sha256_file(codec_path),
            "e3fp_module_version": e3fp_api["module_version"],
            "e3fp_module_file": str(e3fp_api["module_file"]),
            "e3fp_source_file_sha256": {label: sha256_file(path) for label, path in sorted(e3fp_files.items())},
            "rdkit_version": Chem.rdBase.rdkitVersion,
        },
        "limits": {"map_size_mib": int(args.map_size_mib), "max_records": int(args.max_records)},
        "prohibitions": {
            "full_mode_available": False,
            "sdf_extracted": False,
            "local_data_transfer": False,
            "raw_smiles_serialized": False,
        },
    }
    write_json_new(output_dir / "smoke_scope_manifest.json", scope_manifest)

    records_path = output_dir / "geometry_records.lmdb"
    membership_path = output_dir / "membership.jsonl"
    reject_path = output_dir / "reject_ledger.jsonl"
    payload_index_path = output_dir / "payload_index.jsonl"
    env = lmdb.open(
        str(records_path),
        subdir=True,
        map_size=int(args.map_size_mib) * 1024 * 1024,
        readonly=False,
        lock=True,
        sync=True,
        metasync=True,
        map_async=False,
    )
    summary = Counter()
    e3fp_param_hashes = set()
    try:
        with open(str(membership_path), "x", encoding="utf-8") as membership_handle, open(
            str(reject_path), "x", encoding="utf-8"
        ) as reject_handle, open(str(payload_index_path), "x", encoding="utf-8") as payload_index_handle, env.begin(write=True) as transaction:
            with tarfile.open(str(archive_path), mode="r|gz") as archive:
                source_member = source_info["verified_input_lock"]["train_sdf_member"]
                member = find_locked_sdf_member(archive, source_member)
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError("unable to open SDF tar member")
                try:
                    supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
                    for ordinal, source_mol in enumerate(supplier):
                        if ordinal >= args.max_records:
                            break
                        csv_row = companion_rows[ordinal]
                        source_address = source_address_sha256(
                            source_info["archive_sha256"], source_member, ordinal, csv_row
                        )
                        diagnostic = csv_malformed.get(csv_row)
                        if diagnostic is None and csv_row not in csv_smiles:
                            diagnostic = "csv_row_unresolved"
                        record, reject = build_record(
                            Chem, np, preflight, linearizer, e3fp_api, identity_gate, ordinal, csv_row,
                            csv_smiles.get(csv_row), source_mol, sidecar_values, source_info["archive_sha256"],
                            source_address, identity_contract_sha256, projection_spec_sha256, linearizer_spec_sha256,
                            official_input_diagnostic=diagnostic,
                        )
                        row = build_membership_row(
                            np, sidecar_id, selected_sha, ordinal, csv_row, source_address, record, reject
                        )
                        validate_membership_row(row, sidecar_id, selected_sha, ordinal, csv_row, source_address)
                        if record is not None:
                            payload = codec.encode_record(np, record)
                            if codec.logical_record_sha256(np, record) != row["record_content_sha256"]:
                                raise RuntimeError("safe payload codec logical hash differs from membership hash")
                            if not transaction.put(row["record_storage_key"].encode("ascii"), payload, overwrite=False):
                                raise RuntimeError("duplicate LMDB storage key")
                            write_jsonl_line(membership_handle, row)
                            payload_index_row = {
                                "payload_index_schema_version": PAYLOAD_INDEX_SCHEMA,
                                "record_storage_key": row["record_storage_key"],
                                "record_wire_bytes": int(len(payload)),
                                "record_wire_sha256": sha256_bytes(payload),
                                "record_content_sha256": row["record_content_sha256"],
                            }
                            validate_payload_index_row(
                                payload_index_row, row["record_storage_key"], row["record_content_sha256"], payload
                            )
                            write_jsonl_line(payload_index_handle, payload_index_row)
                            e3fp_param_hashes.add(record["geometry"]["e3fp_params_sha256"])
                            summary["payload_wire_total_bytes"] += int(len(payload))
                            summary["admitted_record_count"] += 1
                        else:
                            reject_row = build_reject_row(sidecar_id, selected_sha, ordinal, csv_row, reject)
                            validate_reject_row(reject_row, sidecar_id, selected_sha, ordinal, csv_row, source_address)
                            write_jsonl_line(membership_handle, row)
                            write_jsonl_line(reject_handle, reject_row)
                            summary["reject_ledger_record_count"] += 1
                            summary["reject_reason:{}".format(reject["reason_code"])] += 1
                        summary["membership_record_count"] += 1
                    if summary["membership_record_count"] != args.max_records:
                        raise RuntimeError("SDF ended before the requested bounded prefix")
                finally:
                    stream.close()
    finally:
        env.close()

    if len(e3fp_param_hashes) > 1:
        raise RuntimeError("resolved E3FP parameters drifted within one bounded sidecar")
    scope_manifest_sha256 = sha256_file(output_dir / "smoke_scope_manifest.json")
    membership_sha256 = sha256_file(membership_path)
    reject_ledger_sha256 = sha256_file(reject_path)
    payload_index_sha256 = sha256_file(payload_index_path)
    release_root_sha256 = sha256_json(
        {
            "release_root_schema_version": RELEASE_ROOT_SCHEMA,
            "scope_manifest_sha256": scope_manifest_sha256,
            "membership_sha256": membership_sha256,
            "reject_ledger_sha256": reject_ledger_sha256,
            "payload_index_sha256": payload_index_sha256,
            "selected_ordinal_count": int(args.max_records),
            "admitted_record_count": int(summary["admitted_record_count"]),
            "reject_ledger_record_count": int(summary["reject_ledger_record_count"]),
            "payload_wire_total_bytes": int(summary["payload_wire_total_bytes"]),
        }
    )
    build_report = {
        "schema_version": BUILD_REPORT_SCHEMA,
        "created_utc": utc_now(),
        "sidecar_id": sidecar_id,
        "sidecar_schema_version": SIDE_CAR_SCHEMA,
        "logical_record_schema_version": RECORD_SCHEMA,
        "release_status": "non_admissible_pre_tokenizer",
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "selection": scope_manifest["selection"],
        "counts": {
            "selected_ordinal_count": int(args.max_records),
            "membership_record_count": int(summary["membership_record_count"]),
            "admitted_record_count": int(summary["admitted_record_count"]),
            "reject_ledger_record_count": int(summary["reject_ledger_record_count"]),
        },
        "reject_reason_counts": {
            key.split(":", 1)[1]: int(value)
            for key, value in sorted(summary.items())
            if key.startswith("reject_reason:")
        },
        "partition_invariant_pass": int(args.max_records)
        == int(summary["admitted_record_count"]) + int(summary["reject_ledger_record_count"]),
        "e3fp_params_sha256_values": sorted(e3fp_param_hashes),
        "artifacts": {
            "scope_manifest_sha256": scope_manifest_sha256,
            "membership_sha256": membership_sha256,
            "reject_ledger_sha256": reject_ledger_sha256,
            "payload_index_sha256": payload_index_sha256,
            "payload_wire_total_bytes": int(summary["payload_wire_total_bytes"]),
            "release_root_schema_version": RELEASE_ROOT_SCHEMA,
            "release_root_sha256": release_root_sha256,
            "payload_schema_version": PAYLOAD_SCHEMA,
            "payload_format_contract_sha256": sha256_file(payload_format_contract_path),
            "lmdb_map_size_mib": int(args.map_size_mib),
        },
        "next_gate": "Run validate_p1_pcqm_geometry_sidecar.py over every bounded record before any full-pass decision.",
        "pass": True,
    }
    build_report_path = output_dir / "build_report.json"
    write_json_new(build_report_path, build_report)
    # This post-report handoff root deliberately does not feed back into the
    # build report: embedding its own SHA would create a self-reference.  It
    # binds the immutable build-report bytes to the four release artifacts and
    # is the object a later external handoff can hash or sign.
    handoff_root = {
        "schema_version": RELEASE_ROOT_SCHEMA,
        "release_status": "non_admissible_pre_tokenizer",
        "logical_release_root_sha256": release_root_sha256,
        "build_report_sha256": sha256_file(build_report_path),
        "artifacts": {
            "scope_manifest_sha256": scope_manifest_sha256,
            "membership_sha256": membership_sha256,
            "reject_ledger_sha256": reject_ledger_sha256,
            "payload_index_sha256": payload_index_sha256,
        },
        "counts": {
            "selected_ordinal_count": int(args.max_records),
            "admitted_record_count": int(summary["admitted_record_count"]),
            "reject_ledger_record_count": int(summary["reject_ledger_record_count"]),
            "payload_wire_total_bytes": int(summary["payload_wire_total_bytes"]),
        },
    }
    write_json_new(output_dir / "release_root.json", handoff_root)
    return build_report


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    try:
        report = run(args)
    except Exception as exc:
        # If the fresh directory exists, leave an error-only diagnostic rather
        # than deleting evidence or pretending an incomplete partition passed.
        if output_dir.is_dir():
            failure_path = output_dir / "build_failure.json"
            if not failure_path.exists():
                try:
                    write_json_new(
                        failure_path,
                        {
                            "schema_version": "most-t5-r1/p1-pcqm-geometry-sidecar-build-failure/v1",
                            "created_utc": utc_now(),
                            "pass": False,
                            "exception_type": type(exc).__name__,
                            "message_class_only": "{}".format(type(exc).__name__),
                        },
                    )
                except Exception:
                    pass
        raise
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "sidecar_id": report["sidecar_id"],
                "selected": report["counts"]["selected_ordinal_count"],
                "admitted": report["counts"]["admitted_record_count"],
                "rejected": report["counts"]["reject_ledger_record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
