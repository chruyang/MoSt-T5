#!/usr/bin/env python3
"""Independently replay PCQM identity and E3FP semantics for a v2 release.

The executable deliberately does not import the producer, its preflight
helpers, its payload codec, or either structural auditor.  It consumes the
already pre-registered v3 semantic-review plan, replays *every* reject from
the original SDF/CSV sources, and replays identity, atom provenance,
coordinates, and E3FP for every admitted sample in that plan.

Only content hashes, categorical observations, and pass/fail checks are
written.  Raw SMILES, SDF blocks, RDKit molecules, and coordinates are never
serialized.  A pass is an engineering-reproduction result, not scientific
validation of E3FP and not P1 admission.
"""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib
import io
import json
import logging
import math
import os
import platform
import struct
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


CONTRACT_SCHEMA = "most-t5-r1/p0-pcqm-independent-semantic-recompute-contract/v1"
REPORT_SCHEMA = "most-t5-r1/p0-pcqm-independent-semantic-recompute-report/v1"
LEDGER_SCHEMA = "most-t5-r1/p0-pcqm-independent-semantic-recompute-ledger/v1"
STAGING_SCHEMA = "most-t5-r1/p0-pcqm-independent-semantic-recompute-staging/v1"
COMPLETION_RECEIPT_SCHEMA = (
    "most-t5-r1/p0-pcqm-independent-semantic-recompute-completion-receipt/v1"
)
COMPLETED_MARKER_SCHEMA = (
    "most-t5-r1/p0-pcqm-independent-semantic-recompute-completed-marker/v1"
)
FULL_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
SHARD_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-shard/v2"
STRUCTURAL_REPORT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-independent-audit-report/v3"
SEMANTIC_PLAN_SCHEMA = "most-t5-r1/p1-pcqm-geometry-semantic-review-plan/v3"
IDENTITY_CONTRACT_SCHEMA = "most-t5-r1/pcqm4mv2-identity-normalization-contract/v1"
PAYLOAD_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-format-contract/v2"
PAYLOAD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-payload/v2"
PRODUCTION_RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-pretokenizer-record/v2"
IDENTITY_NAMESPACE = "ogb_pcqm4mv2_train_row_index"
SOURCE_ADDRESS_SCHEMA = "most-t5-r1/pcqm-source-address/v1"
EXPECTED_SOURCE_RECORDS = 3_378_606
MAGIC = b"MST5PCQM2\x00"
HEADER_LENGTH_BYTES = 4
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BLOCKS = 100000
FP_BITS = 4096
FP_LEVEL = 3
SOURCE_ATOM_TAG = "_r1_semantic_source_atom_index"
HEX64 = frozenset("0123456789abcdef")
REPORT_HASH_FIELD = "report_canonical_payload_sha256"
RECEIPT_HASH_FIELD = "receipt_canonical_payload_sha256"
SEMANTIC_CONTRACT_SHA256 = "05bdd2083d5796b8f6368d0bc9862a170ef04a557e456098682638afa17dd490"
SEMANTIC_CONTRACT_BYTES = 14082
IDENTITY_CONTRACT_SHA256 = "5f9be346294e08bf73d47c089a00be4c2f19d89612b5e4c09d0d7f5f6b23b044"
IDENTITY_CONTRACT_BYTES = 4193
PAYLOAD_CONTRACT_SHA256 = "fe7df29b1a7d358676c0eec6c44c1a9a42cff0f38cf14d2dd03c24c0c8f3003b"
PAYLOAD_CONTRACT_BYTES = 2228
FROZEN_RELEASE_MANIFEST_SHA256 = (
    "4db380c63b00f2a595e3a86f70f434a059a6ca724fe35dee94ae2bdafb7d5a2d"
)
FROZEN_STRUCTURAL_AUDIT_SHA256 = (
    "8875f61c0c2691e1805081ee81228fce2c0a218f92ad3bbc095c5946bf430902"
)
FROZEN_SEMANTIC_PLAN_SHA256 = (
    "822e9a67f73f0f33caa38a8e015365929e87bf946238faa19810827c2dd58781"
)
SHARD_ARTIFACT_PATHS = {
    "geometry_records_lmdb_data": "geometry_records.lmdb/data.mdb",
    "membership": "membership.jsonl",
    "reject_ledger": "reject_ledger.jsonl",
    "payload_index": "payload_index.jsonl",
    "motif_census": "motif_census.jsonl",
}
E3FP_REQUIRED_IMPORTED_MODULES = {
    "e3fp": "__init__.py",
    "e3fp.pipeline": "pipeline.py",
    "e3fp.fingerprint.fprinter": "fingerprint/fprinter.py",
}
STAGED_REPORT_FILE_NAME = "semantic_recompute_report.staged.json"
RESULT_LEDGER_FILE_NAME = "semantic_recompute_results.jsonl"
COMPLETION_RECEIPT_FILE_NAME = "completion_receipt.json"
COMPLETED_MARKER_FILE_NAME = "COMPLETED"
HEADER_FIELDS = frozenset(
    ("payload_schema_version", "record", "array_blocks", "logical_record_sha256")
)
BLOCK_FIELDS = frozenset(("index", "dtype", "shape", "order", "offset", "nbytes", "sha256"))
PLACEHOLDER_FIELDS = frozenset(("__array_block__", "dtype", "shape", "order", "sha256"))
WIRE_DTYPES = {"int32": "<i4", "float32": "<f4", "bool": "|b1"}
HISTORICAL_E3FP_INVOCATION = {
    "bits": FP_BITS,
    "level": FP_LEVEL,
    "rdkit_invariants": True,
    "all_iters": True,
    "exclude_floating": False,
}


class GateError(RuntimeError):
    """Fail-closed gate or input-contract violation."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_json(value):
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path):
    digest = hashlib.sha256()
    byte_count = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            byte_count += len(block)
            digest.update(block)
    return byte_count, digest.hexdigest()


def read_file_snapshot(path, label):
    """Read a critical small file once, binding parsed bytes to its digest."""
    path = require_regular_file(path, label)
    with open(str(path), "rb") as handle:
        raw = handle.read()
    return {
        "path": path,
        "raw": raw,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def snapshot_observation(snapshot):
    return {
        "path": str(snapshot["path"]),
        "bytes": snapshot["bytes"],
        "sha256": snapshot["sha256"],
    }


def parse_json_snapshot(snapshot, label, canonical_required=False):
    return strict_json_bytes(snapshot["raw"], label, canonical_required)


def load_pinned_json_snapshot(path, label, expected_sha256, expected_bytes=None):
    snapshot = read_file_snapshot(path, label)
    require_sha256(expected_sha256, "{} pinned hash".format(label))
    if snapshot["sha256"] != expected_sha256:
        raise GateError("{} differs from the executable byte lock".format(label))
    if expected_bytes is not None and snapshot["bytes"] != expected_bytes:
        raise GateError("{} byte count differs from the executable lock".format(label))
    return snapshot, parse_json_snapshot(snapshot, label)


def require_frozen_evidence_snapshot(path, label, expected_file_name, expected_sha256):
    """Bind a deployment-specific evidence file before any semantic use."""
    snapshot = read_file_snapshot(path, label)
    require_sha256(expected_sha256, "{} frozen hash".format(label))
    if (
        snapshot["path"].name != expected_file_name
        or snapshot["sha256"] != expected_sha256
    ):
        raise GateError("{} differs from the executable evidence lock".format(label))
    return snapshot


def require_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - HEX64:
        raise GateError("{} is not a lowercase SHA-256".format(label))


def require_regular_file(path, label):
    result = Path(path).expanduser()
    if not result.is_file() or result.is_symlink():
        raise GateError("{} is not a regular non-symlink file: {}".format(label, result))
    return result.resolve()


def require_directory(path, label):
    result = Path(path).expanduser()
    if not result.is_dir() or result.is_symlink():
        raise GateError("{} is not a directory: {}".format(label, result))
    return result.resolve()


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GateError("duplicate JSON object key: {}".format(key))
        result[key] = value
    return result


def _reject_nonfinite(token):
    raise GateError("non-finite JSON token is forbidden: {}".format(token))


def strict_json_bytes(raw, label, canonical_required=False):
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    try:
        value = json.loads(
            bytes(raw).decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("{} is not strict UTF-8 JSON".format(label)) from exc
    if canonical_required and canonical_json_bytes(value) != bytes(raw):
        raise GateError("{} is not canonical JSON".format(label))
    return value


def load_json(path, label):
    path = require_regular_file(path, label)
    with open(str(path), "rb") as handle:
        return path, strict_json_bytes(handle.read(), label)


def iter_canonical_jsonl(path, label):
    with open(str(path), "rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                raise GateError("{} line {} lacks LF termination".format(label, line_number))
            body = raw[:-1]
            if not body:
                raise GateError("{} line {} is blank".format(label, line_number))
            yield strict_json_bytes(body, "{} line {}".format(label, line_number), True)


def parse_canonical_jsonl_snapshot(snapshot, label):
    raw = snapshot["raw"]
    if raw and not raw.endswith(b"\n"):
        raise GateError("{} final line lacks LF termination".format(label))
    rows = []
    for line_number, body in enumerate(raw.splitlines(), 1):
        if not body:
            raise GateError("{} line {} is blank".format(label, line_number))
        rows.append(strict_json_bytes(body, "{} line {}".format(label, line_number), True))
    return rows


def require_fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise GateError("{} fields differ from the closed schema".format(label))


def validate_contract(contract):
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise GateError("semantic gate contract schema mismatch")
    prerequisites = contract.get("prerequisites", {})
    expected = {
        "production_release_schema": FULL_MANIFEST_SCHEMA,
        "production_release_status": "complete",
        "structural_audit_report_schema": STRUCTURAL_REPORT_SCHEMA,
        "structural_audit_status": "pass",
        "semantic_plan_schema": SEMANTIC_PLAN_SCHEMA,
        "identity_contract_schema": IDENTITY_CONTRACT_SCHEMA,
        "payload_contract_schema": PAYLOAD_CONTRACT_SCHEMA,
    }
    if prerequisites != expected:
        raise GateError("semantic gate prerequisite block mismatch")
    if contract.get("trust_root") != {
        "external_root": (
            "An externally reviewed exact semantic-gate script SHA-256 recorded in a "
            "separate deployment/execution receipt"
        ),
        "script_pins_contracts_and_frozen_evidence": True,
        "semantic_contract_pins_script": False,
        "recursive_script_contract_hashing_permitted": False,
    }:
        raise GateError("semantic gate trust-root boundary mismatch")
    if contract.get("artifact_locks") != {
        "production_release_manifest": {
            "file_name": "full_release_manifest.json",
            "sha256": FROZEN_RELEASE_MANIFEST_SHA256,
        },
        "destination_structural_audit_report": {
            "file_name": "independent_audit_report.json",
            "sha256": FROZEN_STRUCTURAL_AUDIT_SHA256,
        },
        "semantic_review_plan": {
            "file_name": "semantic_review_plan.jsonl",
            "sha256": FROZEN_SEMANTIC_PLAN_SHA256,
        },
        "identity_contract": {
            "bytes": IDENTITY_CONTRACT_BYTES,
            "sha256": IDENTITY_CONTRACT_SHA256,
        },
        "payload_contract": {
            "bytes": PAYLOAD_CONTRACT_BYTES,
            "sha256": PAYLOAD_CONTRACT_SHA256,
        },
    }:
        raise GateError("semantic gate frozen artifact locks mismatch")
    boundary = contract.get("independence_boundary", {})
    if not (
        boundary.get("standalone_gate_required") is True
        and boundary.get("producer_module_imports_permitted") is False
        and boundary.get("structural_auditor_module_imports_permitted") is False
        and boundary.get("release_mutation_permitted") is False
        and boundary.get("raw_smiles_or_molecule_output_permitted") is False
    ):
        raise GateError("semantic gate independence boundary mismatch")
    selection = contract.get("selection", {})
    if not (
        selection.get("rejects") == "all rows in the v3 semantic plan without exception"
        and selection.get("admitted")
        == "all admitted_payload_sample rows in the same pre-registered v3 semantic plan"
        and selection.get("post_hoc_resampling_permitted") is False
        and selection.get("expected_full_release_reject_count") == 13029
        and selection.get("expected_full_release_admitted_sample_count") == 544
    ):
        raise GateError("semantic gate selection is not frozen")
    reject_contract = contract.get("reject_recompute", {})
    required_reasons = reject_contract.get("required_immediate_reason_codes")
    if required_reasons != [
        "PCQM_STEREO_2D3D_DIVERGENCE",
        "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH",
        "HYDROGEN_PROJECTION_RESIDUAL_H",
    ]:
        raise GateError("semantic gate immediate reject vocabulary mismatch")
    if reject_contract.get("expected_reason_counts") != {
        "PCQM_STEREO_2D3D_DIVERGENCE": 12978,
        "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH": 33,
        "HYDROGEN_PROJECTION_RESIDUAL_H": 18,
    }:
        raise GateError("semantic gate reject-reason census mismatch")
    if reject_contract.get("required_reason_stage_diagnostic") != {
        "PCQM_STEREO_2D3D_DIVERGENCE": ["identity", "strict_mismatch_connectivity_match"],
        "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH": ["identity", "connectivity_mismatch"],
        "HYDROGEN_PROJECTION_RESIDUAL_H": [
            "hydrogen_projection", "preflight_hydrogen_projection_residual_h"
        ],
    }:
        raise GateError("semantic gate reason/stage/diagnostic map mismatch")
    if contract.get("source_replay", {}).get("selected_record_parser") != (
        "one-record in-memory Chem.ForwardSDMolSupplier with sanitize=true, removeHs=false, "
        "strictParsing=true before reproducing the production worker transport"
    ):
        raise GateError("semantic gate selected-record parser mismatch")
    source_replay = contract.get("source_replay", {})
    if not (
        source_replay.get("production_worker_transport")
        == "bytes(source_mol.ToBinary()) in the parent followed by Chem.Mol(mol_binary) in the feature worker"
        and source_replay.get("identity_projection_and_e3fp_source")
        == "the worker-side Chem.Mol reconstructed from the RDKit binary transport"
        and source_replay.get("transport_precision_boundary")
        == (
            "the frozen RDKit binary transport deterministically stores conformer coordinates "
            "at float32 precision; atom, bond, and stereo components remain equal and the "
            "geometry payload is already float32"
        )
    ):
        raise GateError("semantic gate production worker transport mismatch")
    invocation = contract.get("e3fp_recompute", {}).get("invocation")
    if invocation != HISTORICAL_E3FP_INVOCATION:
        raise GateError("semantic gate E3FP invocation mismatch")
    release_rehash = contract.get("full_release_artifact_rehash", {})
    if not (
        release_rehash.get("required_phases")
        == ["before_source_replay", "after_source_replay_before_completion"]
        and release_rehash.get("required_shard_artifact_roles")
        == list(SHARD_ARTIFACT_PATHS)
        and release_rehash.get("required_relative_paths") == SHARD_ARTIFACT_PATHS
        and release_rehash.get("include_every_shard_manifest") is True
        and release_rehash.get("include_global_motif_census") is True
    ):
        raise GateError("semantic gate full-release rehash policy mismatch")
    closure_rehash = contract.get("e3fp_source_closure_rehash", {})
    if not (
        closure_rehash.get("required_phases") == [
            "before_import_and_source",
            "after_import_before_source",
            "after_source_before_output",
            "after_staged_output_before_completion",
            "before_completion_receipt",
            "before_completed_marker",
        ]
        and closure_rehash.get("required_imported_modules")
        == E3FP_REQUIRED_IMPORTED_MODULES
        and closure_rehash.get("all_observed_closures_must_equal_runtime_attestation") is True
        and closure_rehash.get("imported_module_paths_must_resolve_under_pinned_package_root") is True
    ):
        raise GateError("semantic gate E3FP closure rehash policy mismatch")
    completion = contract.get("completion_protocol", {})
    if not (
        completion.get("initial_marker") == "STAGING.json"
        and completion.get("staged_report") == "semantic_recompute_report.staged.json"
        and completion.get("result_ledger") == "semantic_recompute_results.jsonl"
        and completion.get("authoritative_receipt") == "completion_receipt.json"
        and completion.get("final_marker") == "COMPLETED"
        and completion.get("staged_report_may_claim_overall_pass") is False
        and completion.get("consumer_must_require_receipt_and_completed_marker") is True
        and completion.get("publisher_derives_status_from_fixed_staged_report") is True
        and completion.get("fixed_staged_report_and_ledger_paths_required") is True
        and completion.get("staged_report_and_ledger_semantics_must_be_parsed") is True
    ):
        raise GateError("semantic gate completion protocol mismatch")


def validate_identity_contract(contract):
    if contract.get("schema_version") != IDENTITY_CONTRACT_SCHEMA:
        raise GateError("identity contract schema mismatch")
    normalization = contract.get("rdkit_identity_normalization", {})
    projection = normalization.get("projection", {})
    override = projection.get("only_nondefault_override", {})
    if override != {"removeDefiningBondStereo": True}:
        raise GateError("identity contract projection mismatch")


def validate_payload_contract(contract):
    if contract.get("schema_version") != PAYLOAD_CONTRACT_SCHEMA:
        raise GateError("payload contract schema mismatch")
    if not (
        contract.get("payload_schema_version") == PAYLOAD_SCHEMA
        and contract.get("magic_ascii") == MAGIC.decode("ascii")
        and set(contract.get("header_required_fields", [])) == set(HEADER_FIELDS)
        and contract.get("array_block_required_fields")
        == ["index", "dtype", "shape", "order", "offset", "nbytes", "sha256"]
        and set(contract.get("allowed_dtypes", [])) == set(WIRE_DTYPES)
    ):
        raise GateError("payload framing contract mismatch")


def validate_structural_report(report):
    if report.get("schema_version") != STRUCTURAL_REPORT_SCHEMA or report.get("audit_status") != "pass":
        raise GateError("a passing v3 structural report is required")
    claimed = report.get(REPORT_HASH_FIELD)
    require_sha256(claimed, "structural report self hash")
    projection = dict(report)
    projection.pop(REPORT_HASH_FIELD, None)
    if sha256_json(projection) != claimed:
        raise GateError("structural report canonical payload hash mismatch")


def validate_release_manifest(manifest, contract):
    if manifest.get("schema_version") != FULL_MANIFEST_SCHEMA or manifest.get("release_status") != "complete":
        raise GateError("completed production-v2 release manifest required")
    counts = manifest.get("counts", {})
    expected_rejects = contract["selection"]["expected_full_release_reject_count"]
    if not (
        counts.get("source_record_count") == EXPECTED_SOURCE_RECORDS
        and counts.get("membership_record_count") == EXPECTED_SOURCE_RECORDS
        and counts.get("reject_ledger_record_count") == expected_rejects
        and counts.get("admitted_record_count") + expected_rejects == EXPECTED_SOURCE_RECORDS
        and manifest.get("range_no_gap_no_overlap") is True
        and manifest.get("tokenizer_binding") == "absent_and_forbidden"
        and manifest.get("p1_training_admission") is False
    ):
        raise GateError("release count/boundary mismatch")
    configuration = manifest.get("configuration", {})
    if not (
        configuration.get("source_record_count") == EXPECTED_SOURCE_RECORDS
        and configuration.get("selected_record_count") == EXPECTED_SOURCE_RECORDS
        and configuration.get("selected_ordinal_range") == [0, EXPECTED_SOURCE_RECORDS]
        and configuration.get("release_kind") == "full_production"
    ):
        raise GateError("release source selection mismatch")


def validate_plan(plan_rows, plan_sha256, release_manifest_sha256, structural_report, contract):
    if not plan_rows or plan_rows[0].get("document_kind") != "semantic_review_plan_header":
        raise GateError("semantic plan header missing")
    header = plan_rows[0]
    if not (
        header.get("schema_version") == SEMANTIC_PLAN_SCHEMA
        and header.get("release_manifest_sha256") == release_manifest_sha256
        and header.get("semantic_recompute_executed_by_this_gate") is False
        and header.get("all_rejects_included") is True
    ):
        raise GateError("semantic plan header binding mismatch")
    artifact = structural_report.get("semantic_review_plan", {})
    if not (
        artifact.get("relative_path") == "semantic_review_plan.jsonl"
        and artifact.get("sha256") == plan_sha256
    ):
        raise GateError("semantic plan does not match the passing structural report")
    if structural_report.get("release_manifest_sha256") != release_manifest_sha256:
        raise GateError("structural report release binding mismatch")
    admitted = {}
    rejected = {}
    reject_reason_counts = Counter()
    reason_stage_diagnostic = contract["reject_recompute"]["required_reason_stage_diagnostic"]
    for row in plan_rows[1:]:
        kind = row.get("document_kind") if isinstance(row, dict) else None
        ordinal = row.get("sdf_record_index") if isinstance(row, dict) else None
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not (0 <= ordinal < EXPECTED_SOURCE_RECORDS):
            raise GateError("semantic plan contains an invalid ordinal")
        if ordinal in admitted or ordinal in rejected:
            raise GateError("semantic plan contains a duplicate ordinal")
        if kind == "admitted_payload_sample":
            if not (
                isinstance(row.get("shard_index"), int)
                and row.get("record_storage_key") == "{:09d}".format(ordinal)
                and row.get("required_review") == "wire_hash_decode_and_logical_structure"
            ):
                raise GateError("admitted semantic-plan row mismatch")
            admitted[ordinal] = row
        elif kind == "reject_semantic_review":
            reason = row.get("reason_code")
            expected_stage_diagnostic = reason_stage_diagnostic.get(reason)
            if not (
                isinstance(row.get("shard_index"), int)
                and expected_stage_diagnostic is not None
                and row.get("stage") == expected_stage_diagnostic[0]
                and row.get("selection_reason") == "all_rejects_without_exception"
                and row.get("required_review") == "independent_source_and_feature_semantic_recompute"
            ):
                raise GateError("reject semantic-plan row differs from the frozen immediate gate")
            rejected[ordinal] = row
            reject_reason_counts[reason] += 1
        else:
            raise GateError("semantic plan contains an unknown row kind")
    expected_admitted = contract["selection"]["expected_full_release_admitted_sample_count"]
    expected_rejected = contract["selection"]["expected_full_release_reject_count"]
    if len(admitted) != expected_admitted or len(rejected) != expected_rejected:
        raise GateError("semantic plan sample/reject count mismatch")
    if dict(sorted(reject_reason_counts.items())) != contract["reject_recompute"]["expected_reason_counts"]:
        raise GateError("semantic plan reject-reason census differs from the frozen contract")
    if header.get("admitted_sample_count") != len(admitted) or header.get("reject_review_count") != len(rejected):
        raise GateError("semantic plan header counts mismatch")
    return header, admitted, rejected


def _validate_shape(shape, label):
    if not isinstance(shape, list) or len(shape) > 8:
        raise GateError("{} shape is malformed".format(label))
    result = []
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise GateError("{} shape dimension is invalid".format(label))
        result.append(dimension)
    return result


def _array_descriptor(value):
    return {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "order": "C",
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def logical_projection(np, value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": _array_descriptor(value)}
    if isinstance(value, dict):
        return {str(key): logical_projection(np, value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [logical_projection(np, item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise GateError("unsupported logical-record value")


def _rehydrate_arrays(value, arrays, consumed):
    if isinstance(value, dict):
        if "__array_block__" in value:
            require_fields(value, PLACEHOLDER_FIELDS, "array placeholder")
            index = value["__array_block__"]
            if not isinstance(index, int) or isinstance(index, bool) or index not in arrays or index in consumed:
                raise GateError("array placeholder index is invalid or duplicated")
            array = arrays[index]
            descriptor = _array_descriptor(array)
            if not (
                value["dtype"] == descriptor["dtype"]
                and value["shape"] == descriptor["shape"]
                and value["order"] == "C"
                and value["sha256"] == descriptor["sha256"]
            ):
                raise GateError("array placeholder differs from its block")
            consumed.add(index)
            return array
        return {key: _rehydrate_arrays(item, arrays, consumed) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate_arrays(item, arrays, consumed) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise GateError("payload JSON contains an unsupported value")


def decode_payload(np, payload):
    """Decode payload v2 independently from the producer and auditor."""
    if sys.byteorder != "little":
        raise GateError("payload v2 replay requires a little-endian runtime")
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise GateError("payload exceeds the public safety bound")
    prefix = len(MAGIC) + HEADER_LENGTH_BYTES
    if len(payload) < prefix or payload[: len(MAGIC)] != MAGIC:
        raise GateError("payload magic mismatch")
    header_size = struct.unpack(">I", payload[len(MAGIC):prefix])[0]
    if header_size < 2 or header_size > MAX_HEADER_BYTES or prefix + header_size > len(payload):
        raise GateError("payload header length invalid")
    header_raw = payload[prefix:prefix + header_size]
    header = strict_json_bytes(header_raw, "payload header", True)
    require_fields(header, HEADER_FIELDS, "payload header")
    if header["payload_schema_version"] != PAYLOAD_SCHEMA:
        raise GateError("payload schema mismatch")
    require_sha256(header["logical_record_sha256"], "payload logical hash")
    blocks = header["array_blocks"]
    if not isinstance(blocks, list) or len(blocks) > MAX_ARRAY_BLOCKS:
        raise GateError("payload block list malformed")
    raw_blocks = payload[prefix + header_size:]
    arrays = {}
    expected_offset = 0
    for expected_index, block in enumerate(blocks):
        require_fields(block, BLOCK_FIELDS, "array block")
        if block["index"] != expected_index or block["dtype"] not in WIRE_DTYPES or block["order"] != "C":
            raise GateError("payload block index/dtype/order mismatch")
        shape = _validate_shape(block["shape"], "array block")
        offset, nbytes = block["offset"], block["nbytes"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset != expected_offset:
            raise GateError("payload block offset mismatch")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise GateError("payload block size invalid")
        item_count = 1
        for dimension in shape:
            item_count *= dimension
            if item_count > (1 << 62):
                raise GateError("payload shape exceeds safety bound")
        expected_nbytes = item_count * np.dtype(WIRE_DTYPES[block["dtype"]]).itemsize
        if nbytes != expected_nbytes or offset + nbytes > len(raw_blocks):
            raise GateError("payload block size disagrees with shape")
        block_raw = raw_blocks[offset:offset + nbytes]
        require_sha256(block["sha256"], "payload block hash")
        if sha256_bytes(block_raw) != block["sha256"]:
            raise GateError("payload block hash mismatch")
        wire = np.frombuffer(block_raw, dtype=np.dtype(WIRE_DTYPES[block["dtype"]]))
        arrays[expected_index] = np.ascontiguousarray(
            wire.reshape(tuple(shape), order="C").astype(np.dtype(block["dtype"]), copy=True)
        )
        expected_offset += nbytes
    if expected_offset != len(raw_blocks):
        raise GateError("payload has trailing bytes")
    consumed = set()
    record = _rehydrate_arrays(header["record"], arrays, consumed)
    if consumed != set(arrays):
        raise GateError("payload has an unreferenced array block")
    logical_hash = sha256_json(logical_projection(np, record))
    if logical_hash != header["logical_record_sha256"]:
        raise GateError("payload logical hash mismatch")
    return record, logical_hash


def _safe_relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise GateError("E3FP closure path is not a forward-slash relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise GateError("E3FP closure path is not canonical")
    return pure


def resolve_e3fp_package_root(path):
    supplied = require_directory(path, "E3FP source")
    if (supplied / "e3fp" / "pipeline.py").is_file():
        return supplied, supplied / "e3fp"
    if supplied.name == "e3fp" and (supplied / "pipeline.py").is_file():
        return supplied.parent, supplied
    raise GateError("E3FP source must be a 3d_tokenization root or its e3fp package")


def scan_e3fp_closure(package_root, closure, phase, imported_module_paths=None):
    expected_files = closure.get("files")
    exclusion = closure.get("exclusion_policy", {})
    if not isinstance(expected_files, list) or not expected_files:
        raise GateError("runtime attestation lacks E3FP closure files")
    excluded_dirs = set(exclusion.get("directories", []))
    excluded_suffixes = tuple(exclusion.get("file_suffixes", []))
    observed = []
    for current_root, directory_names, file_names in os.walk(
        str(package_root), topdown=True, followlinks=False
    ):
        current = Path(current_root)
        kept = []
        for directory_name in sorted(directory_names):
            candidate = current / directory_name
            if candidate.is_symlink():
                raise GateError("symlink in E3FP closure")
            if directory_name not in excluded_dirs:
                kept.append(directory_name)
        directory_names[:] = kept
        for file_name in sorted(file_names):
            candidate = current / file_name
            if candidate.is_symlink() or not candidate.is_file():
                raise GateError("non-regular path in E3FP closure")
            if candidate.suffix in excluded_suffixes:
                continue
            relative = candidate.relative_to(package_root).as_posix()
            _safe_relative_path(relative)
            size, file_hash = sha256_file(candidate)
            observed.append(
                {"relative_path": relative, "bytes": size, "sha256": file_hash}
            )
    observed.sort(key=lambda item: item["relative_path"])
    for item in expected_files:
        require_fields(item, ("relative_path", "bytes", "sha256"), "E3FP closure row")
        _safe_relative_path(item["relative_path"])
        require_sha256(item["sha256"], "E3FP closure file hash")
    observed_closure_sha = sha256_bytes(canonical_json_bytes(observed) + b"\n")
    if observed != expected_files or observed_closure_sha != closure.get("closure_sha256"):
        raise GateError("live E3FP source closure differs from production attestation")
    modules = None
    if imported_module_paths is not None:
        if set(imported_module_paths) != set(E3FP_REQUIRED_IMPORTED_MODULES):
            raise GateError("actual imported E3FP module set is incomplete")
        modules = {}
        for module_name, expected_relative in E3FP_REQUIRED_IMPORTED_MODULES.items():
            module_path = require_regular_file(
                imported_module_paths[module_name], "imported {} module".format(module_name)
            )
            expected_path = (package_root / Path(expected_relative)).resolve()
            if module_path != expected_path:
                raise GateError(
                    "imported {} module escaped its exact attested path".format(module_name)
                )
            try:
                module_path.relative_to(package_root)
            except ValueError as exc:
                raise GateError("imported E3FP module escaped the package root") from exc
            size, digest = sha256_file(module_path)
            expected_rows = {
                row["relative_path"]: row for row in expected_files
            }
            if expected_relative not in expected_rows or expected_rows[expected_relative] != {
                "relative_path": expected_relative,
                "bytes": size,
                "sha256": digest,
            }:
                raise GateError("imported E3FP module bytes differ from the closure")
            modules[module_name] = {
                "relative_path": expected_relative,
                "bytes": size,
                "sha256": digest,
            }
    return {
        "phase": phase,
        "file_count": len(observed),
        "total_bytes": sum(item["bytes"] for item in observed),
        "closure_sha256": observed_closure_sha,
        "imported_modules": modules,
    }


def require_same_e3fp_closure(observations):
    if not observations:
        raise GateError("E3FP closure observations are absent")
    first = observations[0]
    for observation in observations[1:]:
        for key in ("file_count", "total_bytes", "closure_sha256"):
            if observation.get(key) != first.get(key):
                raise GateError("E3FP source closure changed between gate phases")


def verify_runtime_and_e3fp_closure(runtime_attestation_path, expected_sha256, e3fp_source):
    snapshot = read_file_snapshot(runtime_attestation_path, "production runtime attestation")
    attestation = parse_json_snapshot(snapshot, "production runtime attestation")
    if snapshot["sha256"] != expected_sha256:
        raise GateError("runtime attestation differs from release configuration")
    claimed = attestation.get("attestation_payload_sha256")
    require_sha256(claimed, "runtime attestation payload hash")
    projection = dict(attestation)
    projection.pop("attestation_payload_sha256", None)
    # The frozen runtime-attestation contract hashes canonical JSON *with* a
    # trailing LF, unlike the geometry payload/report canonical projection.
    if sha256_bytes(canonical_json_bytes(projection) + b"\n") != claimed or attestation.get("pass") is not True:
        raise GateError("production runtime attestation is invalid")
    import_root, package_root = resolve_e3fp_package_root(e3fp_source)
    closure = attestation.get("e3fp_source_closure", {})
    first_closure = scan_e3fp_closure(
        package_root, closure, "before_import_and_source"
    )
    return {
        "attestation_path": str(snapshot["path"]),
        "attestation_bytes": snapshot["bytes"],
        "attestation_sha256": snapshot["sha256"],
        "attestation_payload_sha256": claimed,
        "e3fp_package_root": str(package_root),
        "e3fp_file_count": first_closure["file_count"],
        "e3fp_total_bytes": first_closure["total_bytes"],
        "e3fp_closure_sha256": closure["closure_sha256"],
    }, import_root, package_root, closure, first_closure, snapshot


def import_locked_e3fp(import_root, package_root):
    root_text = str(import_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        e3fp = importlib.import_module("e3fp")
        pipeline_module = importlib.import_module("e3fp.pipeline")
        fprinter_module = importlib.import_module("e3fp.fingerprint.fprinter")
        from e3fp.fingerprint.fprinter import signed_to_unsigned_int
        from e3fp.pipeline import fprints_from_mol_verbose
    except ImportError as exc:
        raise GateError("cannot import the attested E3FP source") from exc
    module_paths = {
        "e3fp": str(Path(getattr(e3fp, "__file__", "")).resolve()),
        "e3fp.pipeline": str(Path(getattr(pipeline_module, "__file__", "")).resolve()),
        "e3fp.fingerprint.fprinter": str(
            Path(getattr(fprinter_module, "__file__", "")).resolve()
        ),
    }
    for module_path in module_paths.values():
        try:
            Path(module_path).relative_to(package_root)
        except ValueError as exc:
            raise GateError("ambient E3FP package escaped the attested source root") from exc
    return {
        "fprints_from_mol_verbose": fprints_from_mol_verbose,
        "signed_to_unsigned_int": signed_to_unsigned_int,
        "module_path": module_paths["e3fp"],
        "imported_module_paths": module_paths,
        "module_version": getattr(e3fp, "__version__", None),
    }


def validate_artifact(path, expected, label):
    path = require_regular_file(path, label)
    if not isinstance(expected, dict):
        raise GateError("release lacks {} staged observation".format(label))
    size, digest = sha256_file(path)
    if size != expected.get("bytes") or digest != expected.get("sha256"):
        raise GateError("{} bytes/SHA-256 differ from release staging".format(label))
    return path, {"path": str(path), "bytes": size, "sha256": digest}


def validate_shard_manifest(release_root, top_entry, release_id):
    index = top_entry.get("shard_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise GateError("top-level shard index invalid")
    shard_dir = require_directory(
        release_root / "shard-{:06d}".format(index), "shard directory"
    )
    path = shard_dir / "shard_manifest.json"
    snapshot = read_file_snapshot(path, "shard manifest")
    manifest = parse_json_snapshot(snapshot, "shard manifest")
    if snapshot["sha256"] != top_entry.get("shard_manifest_sha256"):
        raise GateError("shard manifest hash differs from full manifest")
    if not (
        manifest.get("schema_version") == SHARD_MANIFEST_SCHEMA
        and manifest.get("release_status") == "complete"
        and manifest.get("release_id") == release_id
        and manifest.get("shard_index") == index
        and manifest.get("range_start") == top_entry.get("range_start")
        and manifest.get("range_end") == top_entry.get("range_end")
    ):
        raise GateError("shard manifest envelope mismatch")
    return snapshot["path"].parent, manifest, {
        "bytes": snapshot["bytes"], "sha256": snapshot["sha256"]
    }


def _declared_artifact_observation(shard_dir, shard_index, role, expected):
    if not isinstance(expected, dict) or set(expected) != {
        "relative_path", "bytes", "sha256"
    }:
        raise GateError("shard artifact declaration is malformed")
    required_relative = SHARD_ARTIFACT_PATHS[role]
    if expected.get("relative_path") != required_relative:
        raise GateError("shard artifact path differs from the closed role map")
    require_sha256(expected.get("sha256"), "shard artifact hash")
    if not isinstance(expected.get("bytes"), int) or isinstance(expected.get("bytes"), bool):
        raise GateError("shard artifact byte count is invalid")
    path = require_regular_file(shard_dir / Path(required_relative), "shard artifact")
    try:
        path.relative_to(shard_dir)
    except ValueError as exc:
        raise GateError("shard artifact resolved outside its shard directory") from exc
    size, digest = sha256_file(path)
    if size != expected["bytes"] or digest != expected["sha256"]:
        raise GateError(
            "shard {} artifact {} differs from its manifest".format(shard_index, role)
        )
    return {
        "kind": "shard_artifact",
        "shard_index": shard_index,
        "role": role,
        "relative_path": path.relative_to(shard_dir.parent).as_posix(),
        "bytes": size,
        "sha256": digest,
    }


def verify_full_release_artifacts(release_root, manifest, phase):
    """Stream-hash every manifest, shard artifact, and global census."""
    if phase not in {"before_source_replay", "after_source_replay_before_completion"}:
        raise GateError("unknown full-release artifact rehash phase")
    shard_roots = manifest.get("shards")
    if not isinstance(shard_roots, list) or not shard_roots:
        raise GateError("release manifest shard roots are missing")
    ordered = sorted(shard_roots, key=lambda item: item.get("shard_index", -1))
    if [item.get("shard_index") for item in ordered] != list(range(len(ordered))):
        raise GateError("release shard indices are not closed and contiguous")
    rows = []
    for top_entry in ordered:
        shard_index = top_entry["shard_index"]
        shard_dir, shard_manifest, manifest_obs = validate_shard_manifest(
            release_root, top_entry, manifest["release_id"]
        )
        rows.append(
            {
                "kind": "shard_manifest",
                "shard_index": shard_index,
                "relative_path": (shard_dir / "shard_manifest.json")
                .relative_to(release_root)
                .as_posix(),
                "bytes": manifest_obs["bytes"],
                "sha256": manifest_obs["sha256"],
            }
        )
        artifacts = shard_manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(SHARD_ARTIFACT_PATHS):
            raise GateError("shard artifact role set differs from the closed contract")
        for role in SHARD_ARTIFACT_PATHS:
            rows.append(
                _declared_artifact_observation(
                    shard_dir, shard_index, role, artifacts[role]
                )
            )
    global_expected = manifest.get("global_motif_census")
    if not isinstance(global_expected, dict) or set(global_expected) != {
        "relative_path", "bytes", "sha256"
    }:
        raise GateError("global motif census declaration is malformed")
    if global_expected.get("relative_path") != "motif_census.jsonl":
        raise GateError("global motif census path is not canonical")
    require_sha256(global_expected.get("sha256"), "global motif census hash")
    global_path = require_regular_file(
        release_root / "motif_census.jsonl", "global motif census"
    )
    try:
        global_path.relative_to(release_root)
    except ValueError as exc:
        raise GateError("global motif census resolved outside the release root") from exc
    global_size, global_sha = sha256_file(global_path)
    if global_size != global_expected.get("bytes") or global_sha != global_expected.get("sha256"):
        raise GateError("global motif census differs from the full manifest")
    rows.append(
        {
            "kind": "global_artifact",
            "role": "global_motif_census",
            "relative_path": "motif_census.jsonl",
            "bytes": global_size,
            "sha256": global_sha,
        }
    )
    return {
        "phase": phase,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "aggregate_sha256": sha256_json(rows),
        "observations": rows,
    }


def require_same_release_rehash(first, second):
    for key in ("file_count", "total_bytes", "aggregate_sha256", "observations"):
        if first.get(key) != second.get(key):
            raise GateError("full-release artifacts changed across source replay")


def _read_selected_membership(shard_dir, start, end, wanted):
    wanted = set(wanted)
    found = {}
    if not wanted:
        return found
    max_offset = max(wanted) - start
    path = shard_dir / "membership.jsonl"
    with open(str(path), "rb") as handle:
        for offset, raw in enumerate(handle):
            if offset > max_offset:
                break
            ordinal = start + offset
            if ordinal not in wanted:
                continue
            if not raw.endswith(b"\n"):
                raise GateError("selected membership line lacks LF")
            row = strict_json_bytes(raw[:-1], "selected membership", True)
            if row.get("sdf_record_index") != ordinal:
                raise GateError("membership line position/ordinal mismatch")
            found[ordinal] = row
    if set(found) != wanted or any(not (start <= ordinal < end) for ordinal in wanted):
        raise GateError("selected membership rows are incomplete")
    return found


def _read_reject_ledger(shard_dir):
    rows = {}
    for row in iter_canonical_jsonl(shard_dir / "reject_ledger.jsonl", "reject ledger"):
        ordinal = row.get("sdf_record_index")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal in rows:
            raise GateError("reject ledger ordinal invalid or duplicated")
        rows[ordinal] = row
    return rows


def collect_release_expectations(np, release_root, manifest, admitted_plan, reject_plan,
                                 reason_stage_diagnostic):
    """Bind selected plan rows to membership, reject ledgers, and LMDB values."""
    try:
        import lmdb
    except ImportError as exc:
        raise GateError("python-lmdb is required for admitted sample replay") from exc
    by_shard = defaultdict(set)
    for ordinal, row in admitted_plan.items():
        by_shard[row["shard_index"]].add(ordinal)
    for ordinal, row in reject_plan.items():
        by_shard[row["shard_index"]].add(ordinal)
    top_entries = {row["shard_index"]: row for row in manifest.get("shards", [])}
    if set(by_shard) - set(top_entries):
        raise GateError("semantic plan references an unknown shard")
    expectations = {}
    observed_reject_ordinals = set()
    shard_observations = []
    for shard_index in sorted(by_shard):
        shard_dir, shard_manifest, manifest_observation = validate_shard_manifest(
            release_root, top_entries[shard_index], manifest["release_id"]
        )
        start, end = shard_manifest["range_start"], shard_manifest["range_end"]
        selected_membership = _read_selected_membership(shard_dir, start, end, by_shard[shard_index])
        ledger = _read_reject_ledger(shard_dir)
        planned_rejects = {ordinal for ordinal in by_shard[shard_index] if ordinal in reject_plan}
        if set(ledger) != planned_rejects:
            raise GateError("v3 plan does not contain exactly every reject in shard {}".format(shard_index))
        observed_reject_ordinals.update(ledger)
        admitted_ordinals = sorted(ordinal for ordinal in by_shard[shard_index] if ordinal in admitted_plan)
        decoded = {}
        if admitted_ordinals:
            env = lmdb.open(
                str(shard_dir / "geometry_records.lmdb"),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=16,
            )
            try:
                with env.begin(write=False) as transaction:
                    for ordinal in admitted_ordinals:
                        membership = selected_membership[ordinal]
                        key = membership.get("record_storage_key")
                        if key != "{:09d}".format(ordinal):
                            raise GateError("admitted membership storage key mismatch")
                        payload = transaction.get(key.encode("ascii"))
                        if payload is None:
                            raise GateError("admitted sample LMDB value missing")
                        record, logical_hash = decode_payload(np, payload)
                        if logical_hash != membership.get("record_content_sha256"):
                            raise GateError("admitted payload differs from membership logical hash")
                        decoded[ordinal] = {"record": record, "logical_hash": logical_hash}
            finally:
                env.close()
        for ordinal in sorted(by_shard[shard_index]):
            membership = selected_membership[ordinal]
            if ordinal in reject_plan:
                expected = ledger[ordinal]
                plan = reject_plan[ordinal]
                reason = expected.get("reason_code")
                expected_pair = reason_stage_diagnostic.get(reason)
                if not (
                    membership.get("disposition") == "reject"
                    and expected_pair is not None
                    and membership.get("reject_reason_code") == reason
                    and reason == plan.get("reason_code")
                    and expected.get("stage") == plan.get("stage") == expected_pair[0]
                    and expected.get("diagnostic_code") == expected_pair[1]
                    and expected.get("geometry_mse_enabled") is False
                ):
                    raise GateError("reject membership/ledger/plan binding mismatch")
                expectations[ordinal] = {
                    "disposition": "reject",
                    "shard_index": shard_index,
                    "membership": membership,
                    "reject": expected,
                }
            else:
                record = decoded[ordinal]["record"]
                if not (
                    membership.get("disposition") == "admit"
                    and membership.get("reject_reason_code") is None
                    and record.get("record_schema_version") == PRODUCTION_RECORD_SCHEMA
                    and record.get("member", {}).get("sdf_record_index") == ordinal
                    and record.get("member", {}).get("storage_key") == membership.get("record_storage_key")
                ):
                    raise GateError("admitted membership/payload binding mismatch")
                expectations[ordinal] = {
                    "disposition": "admit",
                    "shard_index": shard_index,
                    "membership": membership,
                    "record": record,
                    "logical_hash": decoded[ordinal]["logical_hash"],
                }
        shard_observations.append(
            {
                "shard_index": shard_index,
                "range_start": start,
                "range_end": end,
                "selected_target_count": len(by_shard[shard_index]),
                "selected_admitted_count": len(admitted_ordinals),
                "reject_ledger_count": len(ledger),
                "shard_manifest": manifest_observation,
            }
        )
    if observed_reject_ordinals != set(reject_plan) or set(expectations) != set(admitted_plan) | set(reject_plan):
        raise GateError("release expectations do not close against the semantic plan")
    return expectations, shard_observations


def read_selected_csv(data_csv_path, selected_ordinals):
    wanted = set(selected_ordinals)
    found = {}
    max_ordinal = max(wanted)
    with gzip.open(str(data_csv_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"idx", "smiles"}.issubset(set(reader.fieldnames or [])):
            raise GateError("official data CSV lacks idx/smiles columns")
        for row_index, row in enumerate(reader):
            if row_index > max_ordinal:
                break
            if row_index not in wanted:
                continue
            try:
                csv_index = int(row.get("idx", ""))
            except (TypeError, ValueError) as exc:
                raise GateError("selected official CSV idx is not an integer") from exc
            smiles = row.get("smiles")
            if csv_index != row_index or not isinstance(smiles, str) or not smiles:
                raise GateError("selected official CSV row is invalid")
            found[row_index] = smiles
    if set(found) != wanted:
        raise GateError("official CSV did not resolve every selected ordinal")
    return found


def molecule_identity_components(np, mol):
    """Return raw-free component hashes for the frozen molecule identity."""
    conformer_count = int(mol.GetNumConformers())
    coordinates_sha = None
    if conformer_count == 1:
        coordinates = np.ascontiguousarray(
            np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        )
        coordinates_sha = sha256_bytes(coordinates.tobytes(order="C"))
    atoms = [
        {
            "atomic_num": int(atom.GetAtomicNum()),
            "aromatic": bool(atom.GetIsAromatic()),
            "formal_charge": int(atom.GetFormalCharge()),
            "isotope": int(atom.GetIsotope()),
            "chiral_tag": str(atom.GetChiralTag()),
        }
        for atom in mol.GetAtoms()
    ]
    bonds = []
    for bond in mol.GetBonds():
        begin, end = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
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
    payload = {
        "atom_count": int(mol.GetNumAtoms()),
        "atoms": atoms,
        "bond_count": int(mol.GetNumBonds()),
        "bonds": bonds,
        "conformer_count": conformer_count,
        "coordinates_float64_sha256": coordinates_sha,
    }
    return {
        "atom_count": payload["atom_count"],
        "atoms_sha256": sha256_json(atoms),
        "bond_count": payload["bond_count"],
        "bonds_sha256": sha256_json(bonds),
        "conformer_count": conformer_count,
        "coordinates_float64_sha256": coordinates_sha,
        "molecule_identity_sha256": sha256_json(payload),
    }


def molecule_identity_sha256(np, mol):
    return molecule_identity_components(np, mol)["molecule_identity_sha256"]


def production_worker_ipc_roundtrip(Chem, source_mol):
    """Reproduce the production parent-to-worker RDKit binary transport."""
    if source_mol is None:
        return None
    try:
        mol_binary = bytes(source_mol.ToBinary())
        worker_mol = Chem.Mol(mol_binary)
    except Exception as exc:
        raise GateError("production worker molecule IPC round-trip failed") from exc
    if worker_mol is None:
        raise GateError("production worker molecule IPC round-trip returned None")
    return worker_mol


def worker_ipc_roundtrip_summary(Chem, np, source_mol):
    """Report only hashes and coordinate quantization statistics, never molecule data."""
    worker_mol = production_worker_ipc_roundtrip(Chem, source_mol)
    raw = molecule_identity_components(np, source_mol)
    replayed = molecule_identity_components(np, worker_mol)
    raw_positions = np.asarray(source_mol.GetConformer(0).GetPositions(), dtype=np.float64)
    replayed_positions = np.asarray(
        worker_mol.GetConformer(0).GetPositions(), dtype=np.float64
    )
    if raw_positions.shape != replayed_positions.shape:
        raise GateError("worker IPC round-trip coordinate shape changed")
    delta = np.abs(raw_positions - replayed_positions)
    expected_quantized = raw_positions.astype(np.float32).astype(np.float64)
    return worker_mol, {
        "raw_identity_sha256": raw["molecule_identity_sha256"],
        "worker_identity_sha256": replayed["molecule_identity_sha256"],
        "atom_component_equal": raw["atoms_sha256"] == replayed["atoms_sha256"],
        "bond_component_equal": raw["bonds_sha256"] == replayed["bonds_sha256"],
        "conformer_count_equal": raw["conformer_count"] == replayed["conformer_count"],
        "raw_coordinates_float64_sha256": raw["coordinates_float64_sha256"],
        "worker_coordinates_float64_sha256": replayed["coordinates_float64_sha256"],
        "coordinate_value_count": int(delta.size),
        "coordinate_changed_value_count": int(np.count_nonzero(delta)),
        "coordinates_equal_float32_roundtrip": bool(
            np.array_equal(replayed_positions, expected_quantized)
        ),
        "coordinate_max_abs_delta": float(delta.max()) if delta.size else 0.0,
        "coordinate_mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
    }


def finite_single_conformer(np, mol):
    if int(mol.GetNumConformers()) != 1:
        raise GateError("selected SDF molecule lacks exactly one conformer")
    coordinates = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
    if coordinates.shape != (mol.GetNumAtoms(), 3) or not bool(np.all(np.isfinite(coordinates))):
        raise GateError("selected SDF coordinates are invalid")


def project_hydrogen_candidate(Chem, np, source_mol):
    tagged = Chem.Mol(source_mol)
    source_atom_count = int(tagged.GetNumAtoms())
    if source_atom_count <= 0:
        raise GateError("selected source molecule has zero atoms")
    for index, atom in enumerate(tagged.GetAtoms()):
        atom.SetIntProp(SOURCE_ATOM_TAG, int(index))
    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise GateError("RDKit lacks removeDefiningBondStereo")
    parameters.removeDefiningBondStereo = True
    geometry = Chem.RemoveHs(Chem.Mol(tagged), parameters, sanitize=True)
    Chem.SanitizeMol(geometry)
    Chem.AssignStereochemistry(geometry, cleanIt=True, force=True)
    mapping = []
    for atom in geometry.GetAtoms():
        if not atom.HasProp(SOURCE_ATOM_TAG):
            raise GateError("projected atom lost source tag")
        mapping.append(int(atom.GetIntProp(SOURCE_ATOM_TAG)))
    if not mapping or mapping != sorted(mapping) or len(mapping) != len(set(mapping)):
        raise GateError("projected source tags are empty, duplicated, or reordered")
    if mapping[0] < 0 or mapping[-1] >= source_atom_count:
        raise GateError("projected source tag is out of range")
    finite_single_conformer(np, geometry)
    residual_hydrogen_count = sum(atom.GetAtomicNum() == 1 for atom in geometry.GetAtoms())
    non_e3fp_atom_count = sum(atom.GetAtomicNum() <= 1 for atom in geometry.GetAtoms())
    return (
        geometry,
        np.ascontiguousarray(np.asarray(mapping, dtype=np.int32)),
        int(residual_hydrogen_count),
        int(non_e3fp_atom_count),
    )


def project_geometry_mol(Chem, np, source_mol):
    geometry, mapping, residual_hydrogen_count, non_e3fp_atom_count = project_hydrogen_candidate(
        Chem, np, source_mol
    )
    if residual_hydrogen_count or non_e3fp_atom_count:
        raise GateError("projected geometry molecule retains a non-E3FP atom")
    return geometry, mapping


def normalized_identity_mol(Chem, mol):
    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise GateError("RDKit lacks removeDefiningBondStereo")
    parameters.removeDefiningBondStereo = True
    normalized = Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)
    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
    return normalized


def canonical_forms(Chem, mol):
    normalized = normalized_identity_mol(Chem, mol)
    strict = Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)
    connectivity = Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=False)
    return normalized, {
        "strict_sha256": sha256_bytes(strict.encode("utf-8")),
        "connectivity_sha256": sha256_bytes(connectivity.encode("utf-8")),
    }


def inchi_corroboration(sdf_normalized, official_normalized):
    try:
        from rdkit.Chem import inchi

        sdf_key = inchi.MolToInchiKey(sdf_normalized)
        official_key = inchi.MolToInchiKey(official_normalized)
        if not sdf_key or not official_key:
            return {"available": True, "status": "generation_empty"}
        return {
            "available": True,
            "status": "ok",
            "connectivity_block_equal": sdf_key.split("-", 1)[0] == official_key.split("-", 1)[0],
            "full_key_equal": sdf_key == official_key,
        }
    except Exception as exc:
        return {"available": False, "status": "{}_error".format(type(exc).__name__)}


def shell_level_from_radius(shell, radius_multiplier):
    radius = float(shell.radius)
    multiplier = float(radius_multiplier)
    if not math.isfinite(radius) or radius < 0.0 or not math.isfinite(multiplier) or multiplier <= 0.0:
        raise GateError("E3FP shell radius/multiplier invalid")
    if radius == 0.0:
        return 0
    level = int(round(radius / multiplier))
    tolerance = max(1e-10, abs(multiplier) * 1e-9)
    if level < 0 or not math.isclose(radius, level * multiplier, rel_tol=0.0, abs_tol=tolerance):
        raise GateError("E3FP shell radius is not mappable to a level")
    return level


def build_explicit_shell_matrix(np, fingerprinter, signed_to_unsigned_int, model_atom_count):
    matrix = np.full((model_atom_count, FP_LEVEL + 1), -1, dtype=np.int32)
    slots = set()
    shells_seen = 0
    for shell in fingerprinter.all_shells:
        shells_seen += 1
        center = int(shell.center_atom)
        if center < 0 or center >= model_atom_count:
            raise GateError("E3FP shell center out of range")
        level = shell_level_from_radius(shell, fingerprinter.radius_multiplier)
        if level > FP_LEVEL or (center, level) in slots:
            raise GateError("E3FP shell level is above limit or duplicated")
        slots.add((center, level))
        if getattr(shell, "identifier", None) is None:
            raise GateError("E3FP shell identifier missing")
        folded = int(signed_to_unsigned_int(int(shell.identifier)) % FP_BITS)
        if folded < 0 or folded >= FP_BITS:
            raise GateError("E3FP folded identifier out of range")
        matrix[center, level] = folded
    if shells_seen == 0 or bool(np.any(matrix[:, 0] == -1)) or bool(np.any(np.all(matrix == -1, axis=1))):
        raise GateError("E3FP matrix lacks required shells")
    return np.ascontiguousarray(matrix)


def generate_e3fp(np, e3fp_api, geometry_mol, ordinal):
    geometry_mol.SetProp("_Name", "r1_pcqm_e3fp_preflight_{:06d}".format(ordinal))
    root_logger = logging.getLogger()
    previous = root_logger.level
    try:
        if previous < logging.WARNING:
            root_logger.setLevel(logging.WARNING)
        fprints, fingerprinter = e3fp_api["fprints_from_mol_verbose"](
            geometry_mol, fprint_params=dict(HISTORICAL_E3FP_INVOCATION)
        )
    finally:
        root_logger.setLevel(previous)
    if not fprints:
        raise GateError("E3FP generation returned no fingerprints")
    matrix = build_explicit_shell_matrix(
        np, fingerprinter, e3fp_api["signed_to_unsigned_int"], int(geometry_mol.GetNumAtoms())
    )
    resolved = {
        "bits": int(fingerprinter.bits),
        "level": int(fingerprinter.level),
        "radius_multiplier": float(fingerprinter.radius_multiplier),
        "stereo": bool(fingerprinter.stereo),
        "include_disconnected": bool(fingerprinter.include_disconnected),
        "rdkit_invariants": bool(fingerprinter.rdkit_invariants),
        "exclude_floating": bool(fingerprinter.exclude_floating),
        "remove_duplicate_substructs": bool(fingerprinter.remove_duplicate_substructs),
        "fingerprint_type": getattr(fingerprinter.fp_type, "__name__", str(fingerprinter.fp_type)),
        "all_iters": True,
    }
    if not (
        resolved["bits"] == FP_BITS
        and resolved["level"] == FP_LEVEL
        and resolved["rdkit_invariants"] is True
        and resolved["exclude_floating"] is False
    ):
        raise GateError("resolved E3FP configuration mismatch")
    return matrix, resolved


def source_address_sha256(archive_sha256, locked_member, ordinal):
    return sha256_json(
        {
            "address_schema_version": SOURCE_ADDRESS_SCHEMA,
            "archive_sha256": archive_sha256,
            "identity_namespace": IDENTITY_NAMESPACE,
            "official_csv_row_index": int(ordinal),
            "sdf_record_index": int(ordinal),
            "sdf_tar_member_name": locked_member["tar_member_name"],
            "sdf_tar_member_sha256": locked_member["sha256"],
        }
    )


def classify_identity(sdf_forms, official_forms):
    if sdf_forms["strict_sha256"] == official_forms["strict_sha256"]:
        return "strict_isomeric_match"
    if sdf_forms["connectivity_sha256"] == official_forms["connectivity_sha256"]:
        return "PCQM_STEREO_2D3D_DIVERGENCE"
    return "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH"


def compare_target(Chem, np, e3fp_api, ordinal, source_mol, official_smiles, expected,
                   archive_sha256, locked_member, rdkit_version):
    mismatch = []
    source_identity = molecule_identity_sha256(np, source_mol)
    finite_single_conformer(np, source_mol)
    geometry_mol, mapping, residual_hydrogen_count, non_e3fp_atom_count = project_hydrogen_candidate(
        Chem, np, source_mol
    )
    geometry_identity = molecule_identity_sha256(np, geometry_mol)
    source_address = source_address_sha256(archive_sha256, locked_member, ordinal)
    membership = expected["membership"]
    if membership.get("source_address_sha256") != source_address:
        mismatch.append("source_address_sha256")
    expected_reject = expected.get("reject")
    expected_reason = None if expected_reject is None else expected_reject.get("reason_code")
    if expected_reason == "HYDROGEN_PROJECTION_RESIDUAL_H":
        classification = (
            "HYDROGEN_PROJECTION_RESIDUAL_H"
            if residual_hydrogen_count > 0
            else "hydrogen_projection_without_residual_h"
        )
        if classification != expected_reason:
            mismatch.append("reject_reason_classification")
        if source_identity != expected_reject.get("source_mol_identity_sha256"):
            mismatch.append("source_mol_identity_sha256")
        if expected_reject.get("geometry_mol_identity_sha256") is not None:
            mismatch.append("ledger_geometry_identity_should_be_null_at_terminal_projection_stage")
        if expected_reject.get("diagnostic_code") != "preflight_hydrogen_projection_residual_h":
            mismatch.append("reject_diagnostic_code")
        if expected_reject.get("stage") != "hydrogen_projection":
            mismatch.append("reject_stage")
        if residual_hydrogen_count <= 0 or non_e3fp_atom_count < residual_hydrogen_count:
            mismatch.append("residual_hydrogen_not_reproduced")
        result = {
            "document_kind": "semantic_recompute_result",
            "schema_version": LEDGER_SCHEMA,
            "sdf_record_index": int(ordinal),
            "shard_index": int(expected["shard_index"]),
            "disposition": "reject",
            "recomputed_terminal_classification": classification,
            "recomputed_source_mol_identity_sha256": source_identity,
            "recomputed_geometry_candidate_identity_sha256": geometry_identity,
            "recomputed_sdf_strict_sha256": None,
            "recomputed_official_strict_sha256": None,
            "recomputed_connectivity_sha256": None,
            "recomputed_residual_hydrogen_count": residual_hydrogen_count,
            "recomputed_non_e3fp_atom_count": non_e3fp_atom_count,
            "inchi_corroboration": {"available": False, "status": "not_applicable_terminal_hydrogen_projection_reject"},
            "rdkit_version": rdkit_version,
            "e3fp_recomputed": False,
            "expected_reject_reason_code": expected_reason,
            "expected_reject_diagnostic_code": expected_reject.get("diagnostic_code"),
            "downstream_identity_and_e3fp_status": "not_applicable_first_terminal_hydrogen_projection_reject",
            "mismatch_codes": sorted(set(mismatch)),
        }
        result["status"] = "pass" if not result["mismatch_codes"] else "fail"
        return result

    if residual_hydrogen_count or non_e3fp_atom_count:
        mismatch.append("unexpected_non_e3fp_atom_after_hydrogen_projection")
    sdf_normalized, sdf_forms = canonical_forms(Chem, geometry_mol)
    official_mol = Chem.MolFromSmiles(official_smiles)
    if official_mol is None:
        raise GateError("selected official SMILES did not parse")
    official_normalized, official_forms = canonical_forms(Chem, official_mol)
    classification = classify_identity(sdf_forms, official_forms)
    corroboration = inchi_corroboration(sdf_normalized, official_normalized)
    common = {
        "document_kind": "semantic_recompute_result",
        "schema_version": LEDGER_SCHEMA,
        "sdf_record_index": int(ordinal),
        "shard_index": int(expected["shard_index"]),
        "disposition": expected["disposition"],
        "recomputed_terminal_classification": classification,
        "recomputed_source_mol_identity_sha256": source_identity,
        "recomputed_geometry_candidate_identity_sha256": geometry_identity,
        "recomputed_sdf_strict_sha256": sdf_forms["strict_sha256"],
        "recomputed_official_strict_sha256": official_forms["strict_sha256"],
        "recomputed_connectivity_sha256": sdf_forms["connectivity_sha256"],
        "recomputed_residual_hydrogen_count": residual_hydrogen_count,
        "recomputed_non_e3fp_atom_count": non_e3fp_atom_count,
        "inchi_corroboration": corroboration,
        "rdkit_version": rdkit_version,
        "e3fp_recomputed": expected["disposition"] == "admit",
    }
    if expected["disposition"] == "reject":
        reject = expected["reject"]
        if classification != reject.get("reason_code"):
            mismatch.append("reject_reason_classification")
        expected_diagnostic = {
            "PCQM_STEREO_2D3D_DIVERGENCE": "strict_mismatch_connectivity_match",
            "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH": "connectivity_mismatch",
        }.get(reject.get("reason_code"))
        if reject.get("diagnostic_code") != expected_diagnostic:
            mismatch.append("reject_diagnostic_code")
        if source_identity != reject.get("source_mol_identity_sha256"):
            mismatch.append("source_mol_identity_sha256")
        if geometry_identity != reject.get("geometry_mol_identity_sha256"):
            mismatch.append("geometry_mol_identity_sha256")
        if sdf_forms["strict_sha256"] == official_forms["strict_sha256"]:
            mismatch.append("strict_hashes_should_differ")
        connectivity_equal = sdf_forms["connectivity_sha256"] == official_forms["connectivity_sha256"]
        if reject.get("reason_code") == "PCQM_STEREO_2D3D_DIVERGENCE" and not connectivity_equal:
            mismatch.append("connectivity_hashes_should_match")
        if reject.get("reason_code") == "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH" and connectivity_equal:
            mismatch.append("connectivity_hashes_should_differ")
        common.update(
            {
                "expected_reject_reason_code": reject.get("reason_code"),
                "expected_reject_diagnostic_code": reject.get("diagnostic_code"),
                "downstream_identity_and_e3fp_status": "e3fp_not_applicable_first_terminal_identity_reject",
            }
        )
    else:
        record = expected["record"]
        member = record.get("member", {})
        identity = record.get("identity", {})
        atom_universe = record.get("atom_universe", {})
        geometry = record.get("geometry", {})
        if member.get("source_address_sha256") != source_address:
            mismatch.append("payload_source_address_sha256")
        if member.get("source_mol_identity_sha256") != source_identity:
            mismatch.append("payload_source_mol_identity_sha256")
        if atom_universe.get("geometry_mol_identity_sha256") != geometry_identity:
            mismatch.append("payload_geometry_mol_identity_sha256")
        if identity.get("rdkit_version") != rdkit_version:
            mismatch.append("payload_rdkit_version")
        if identity.get("sdf_strict_smiles_sha256") != sdf_forms["strict_sha256"]:
            mismatch.append("payload_sdf_strict_sha256")
        if identity.get("official_strict_smiles_sha256") != official_forms["strict_sha256"]:
            mismatch.append("payload_official_strict_sha256")
        if identity.get("canonical_connectivity_sha256") != sdf_forms["connectivity_sha256"]:
            mismatch.append("payload_connectivity_sha256")
        expected_mapping = atom_universe.get("model_to_source_atom_index")
        if not isinstance(expected_mapping, np.ndarray) or not np.array_equal(mapping, expected_mapping):
            mismatch.append("model_to_source_atom_index")
        coordinates = np.ascontiguousarray(
            np.asarray(geometry_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        )
        expected_coordinates = geometry.get("coordinates")
        if not isinstance(expected_coordinates, np.ndarray) or not np.array_equal(coordinates, expected_coordinates):
            mismatch.append("coordinates_bytes")
        if sha256_bytes(coordinates.tobytes(order="C")) != geometry.get("coordinates_sha256"):
            mismatch.append("coordinates_sha256")
        e3fp_matrix, resolved = generate_e3fp(np, e3fp_api, geometry_mol, ordinal)
        expected_e3fp = geometry.get("e3fp")
        e3fp_sha = sha256_bytes(e3fp_matrix.tobytes(order="C"))
        params_sha = sha256_json(resolved)
        if not isinstance(expected_e3fp, np.ndarray) or not np.array_equal(e3fp_matrix, expected_e3fp):
            mismatch.append("e3fp_bytes")
        if e3fp_sha != geometry.get("e3fp_sha256"):
            mismatch.append("e3fp_sha256")
        if params_sha != geometry.get("e3fp_params_sha256"):
            mismatch.append("e3fp_params_sha256")
        common.update(
            {
                "record_content_sha256": expected["logical_hash"],
                "recomputed_model_atom_count": int(geometry_mol.GetNumAtoms()),
                "recomputed_model_to_source_sha256": sha256_bytes(mapping.tobytes(order="C")),
                "recomputed_coordinates_sha256": sha256_bytes(coordinates.tobytes(order="C")),
                "recomputed_e3fp_sha256": e3fp_sha,
                "recomputed_e3fp_params_sha256": params_sha,
            }
        )
    common["mismatch_codes"] = sorted(set(mismatch))
    common["status"] = "pass" if not mismatch else "fail"
    return common


def _parse_selected_mol(Chem, block):
    """Parse one selected record through the release's exact supplier API.

    ``MolFromMolBlock`` and ``ForwardSDMolSupplier`` can assign different
    low-level bond stereo enums for unusual atropisomer/chirality records even
    when canonical identity, coordinates, and E3FP are unchanged.  The
    producer contract explicitly fixes ``ForwardSDMolSupplier``.  Wrapping
    only the selected CTAB in an in-memory one-record SDF preserves the fast
    raw delimiter scan while reproducing the declared parser semantics.
    """
    one_record_sdf = bytes(block) + b"$$$$\n"
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(one_record_sdf), sanitize=True, removeHs=False, strictParsing=True
    )
    try:
        mol = next(iter(supplier))
    except StopIteration as exc:
        raise GateError("selected one-record SDF supplier ended without a molecule") from exc
    return mol


def stream_and_recompute(Chem, np, e3fp_api, archive_path, locked_member, official_smiles,
                         expectations, archive_sha256, rdkit_version):
    targets = set(expectations)
    results = {}
    member_observation = None
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = None
        for candidate in archive:
            if candidate.name == locked_member.get("tar_member_name"):
                member = candidate
                break
        if member is None or not member.isfile():
            raise GateError("locked SDF tar member is absent or non-regular")
        if int(member.size) != locked_member.get("uncompressed_bytes"):
            raise GateError("locked SDF member byte size differs")
        stream = archive.extractfile(member)
        if stream is None:
            raise GateError("cannot open locked SDF tar member")
        digest = hashlib.sha256()
        byte_count = 0
        ordinal = 0
        buffer = bytearray() if 0 in targets else None
        record_has_content = False
        try:
            for line in stream:
                digest.update(line)
                byte_count += len(line)
                if line.rstrip(b"\r\n") == b"$$$$":
                    if ordinal in targets:
                        source_mol = _parse_selected_mol(Chem, buffer)
                        if source_mol is None:
                            raise GateError("selected SDF record parsed as RDKit None")
                        source_mol = production_worker_ipc_roundtrip(Chem, source_mol)
                        results[ordinal] = compare_target(
                            Chem, np, e3fp_api, ordinal, source_mol, official_smiles[ordinal],
                            expectations[ordinal], archive_sha256, locked_member, rdkit_version,
                        )
                    ordinal += 1
                    buffer = bytearray() if ordinal in targets else None
                    record_has_content = False
                    continue
                record_has_content = True
                if buffer is not None:
                    buffer.extend(line)
            if record_has_content:
                if ordinal in targets:
                    source_mol = _parse_selected_mol(Chem, buffer)
                    if source_mol is None:
                        raise GateError("selected final SDF record parsed as RDKit None")
                    source_mol = production_worker_ipc_roundtrip(Chem, source_mol)
                    results[ordinal] = compare_target(
                        Chem, np, e3fp_api, ordinal, source_mol, official_smiles[ordinal],
                        expectations[ordinal], archive_sha256, locked_member, rdkit_version,
                    )
                ordinal += 1
        finally:
            stream.close()
        member_observation = {
            "tar_member_name": member.name,
            "member_type": "regular_file",
            "uncompressed_bytes": byte_count,
            "sha256": digest.hexdigest(),
            "sdf_record_count": ordinal,
        }
    expected_observation = {
        "tar_member_name": locked_member.get("tar_member_name"),
        "member_type": locked_member.get("member_type"),
        "uncompressed_bytes": locked_member.get("uncompressed_bytes"),
        "sha256": locked_member.get("sha256"),
        "sdf_record_count": EXPECTED_SOURCE_RECORDS,
    }
    if member_observation != expected_observation:
        raise GateError("streamed SDF member bytes/hash/count differ from the release lock")
    if set(results) != targets:
        raise GateError("SDF stream did not resolve every selected semantic target")
    return [results[ordinal] for ordinal in sorted(results)], member_observation


def runtime_observation(np, rdBase, e3fp_api):
    try:
        import lmdb
        lmdb_version = getattr(lmdb, "__version__", "unknown")
    except Exception:
        lmdb_version = "unknown"
    return {
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_version": str(np.__version__),
        "rdkit_version": str(rdBase.rdkitVersion),
        "python_lmdb_version": str(lmdb_version),
        "e3fp_module_path": e3fp_api["module_path"],
        "e3fp_module_version": e3fp_api["module_version"],
    }


def write_jsonl_new(path, rows):
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_new(path, payload)


def write_json_new(path, value):
    atomic_write_new(path, canonical_json_bytes(value) + b"\n")


def atomic_write_new(path, payload):
    """Publish one new file atomically without overwriting an existing path."""
    path = Path(path)
    if path.exists():
        raise GateError("exclusive output already exists: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    if temporary.exists():
        raise GateError("exclusive temporary output already exists: {}".format(temporary))
    with open(str(temporary), "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(str(temporary), str(path))
    except Exception as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise GateError("atomic exclusive publication failed: {}".format(path)) from exc
    temporary.unlink()


def require_snapshot_unchanged(snapshot, label):
    size, digest = sha256_file(snapshot["path"])
    if size != snapshot["bytes"] or digest != snapshot["sha256"]:
        raise GateError("{} changed after its bound snapshot".format(label))
    return {"path": str(snapshot["path"]), "bytes": size, "sha256": digest}


def report_payload_sha256(report):
    projection = dict(report)
    projection.pop(REPORT_HASH_FIELD, None)
    return sha256_json(projection)


def receipt_payload_sha256(receipt):
    projection = dict(receipt)
    projection.pop(RECEIPT_HASH_FIELD, None)
    return sha256_json(projection)


def validate_staged_report_claims(report):
    if not isinstance(report, dict):
        raise GateError("staged report is not an object")
    if (
        report.get("authoritative_overall_gate_status") is not None
        or report.get("completion_status") != "pending_authoritative_receipt"
        or "gate_status" in report
        or "overall_gate_status" in report
    ):
        raise GateError("staged report made a premature authoritative gate claim")


def parse_canonical_json_snapshot(snapshot, label):
    value = parse_json_snapshot(snapshot, label)
    if snapshot["raw"] != canonical_json_bytes(value) + b"\n":
        raise GateError("{} is not canonical JSON with one LF".format(label))
    return value


def fixed_output_snapshot(output_dir, file_name, label):
    output_dir = require_directory(output_dir, "semantic output directory")
    if not isinstance(file_name, str) or PurePosixPath(file_name).name != file_name:
        raise GateError("{} file name is not a fixed safe relative path".format(label))
    path = require_regular_file(output_dir / file_name, label)
    if path.parent != output_dir or path.name != file_name:
        raise GateError("{} resolved outside the fixed output path".format(label))
    return read_file_snapshot(path, label)


def _validate_external_binding(binding, label, expected_file_name,
                               expected_sha256=None, expected_bytes=None):
    if not isinstance(binding, dict) or set(binding) != {"path", "bytes", "sha256"}:
        raise GateError("{} binding fields are invalid".format(label))
    path_text = binding.get("path")
    if not isinstance(path_text, str):
        raise GateError("{} path is invalid".format(label))
    bound_path = Path(path_text)
    if not bound_path.is_absolute() or bound_path.name != expected_file_name:
        raise GateError("{} path/name is not the fixed external artifact".format(label))
    if not isinstance(binding.get("bytes"), int) or isinstance(binding.get("bytes"), bool):
        raise GateError("{} byte count is invalid".format(label))
    require_sha256(binding.get("sha256"), "{} hash".format(label))
    if expected_sha256 is not None and binding["sha256"] != expected_sha256:
        raise GateError("{} hash differs from the frozen constant".format(label))
    if expected_bytes is not None and binding["bytes"] != expected_bytes:
        raise GateError("{} bytes differ from the frozen constant".format(label))
    return dict(binding)


def _validate_relative_binding(binding, label, fixed_file_name):
    if not isinstance(binding, dict) or set(binding) != {
        "relative_path", "bytes", "sha256"
    }:
        raise GateError("{} binding fields are invalid".format(label))
    if binding.get("relative_path") != fixed_file_name:
        raise GateError("{} path is not the fixed safe relative path".format(label))
    if not isinstance(binding.get("bytes"), int) or isinstance(binding.get("bytes"), bool):
        raise GateError("{} byte count is invalid".format(label))
    require_sha256(binding.get("sha256"), "{} hash".format(label))
    return dict(binding)


def validate_staged_bundle(output_dir, expected_release_id=None,
                           expected_script_binding=None, require_pass=False):
    """Parse and bind the fixed staged report and ledger; bytes alone never pass."""
    output_dir = require_directory(output_dir, "semantic output directory")
    report_snapshot = fixed_output_snapshot(
        output_dir, STAGED_REPORT_FILE_NAME, "staged semantic report"
    )
    report = parse_canonical_json_snapshot(report_snapshot, "staged semantic report")
    if report.get("schema_version") != REPORT_SCHEMA:
        raise GateError("staged semantic report schema mismatch")
    if report.get(REPORT_HASH_FIELD) != report_payload_sha256(report):
        raise GateError("staged semantic report self hash mismatch")
    validate_staged_report_claims(report)
    semantic_status = report.get("semantic_recompute_status")
    if semantic_status not in {"pass", "fail"}:
        raise GateError("staged semantic status is not closed")
    if require_pass and semantic_status != "pass":
        raise GateError("staged semantic report is not passing")
    release_id = report.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise GateError("staged semantic report release ID is invalid")
    if expected_release_id is not None and release_id != expected_release_id:
        raise GateError("staged semantic report release ID mismatch")
    bindings = report.get("bindings")
    if not isinstance(bindings, dict):
        raise GateError("staged semantic report bindings are absent")
    frozen_bindings = {
        "release_manifest": _validate_external_binding(
            bindings.get("release_manifest"), "release manifest",
            "full_release_manifest.json", FROZEN_RELEASE_MANIFEST_SHA256,
        ),
        "structural_audit_report": _validate_external_binding(
            bindings.get("structural_audit_report"), "structural audit report",
            "independent_audit_report.json", FROZEN_STRUCTURAL_AUDIT_SHA256,
        ),
        "semantic_plan": _validate_external_binding(
            bindings.get("semantic_plan"), "semantic plan",
            "semantic_review_plan.jsonl", FROZEN_SEMANTIC_PLAN_SHA256,
        ),
        "semantic_gate_contract": _validate_external_binding(
            bindings.get("semantic_gate_contract"), "semantic gate contract",
            "p0_pcqm_independent_semantic_recompute_contract_v1.json",
            SEMANTIC_CONTRACT_SHA256, SEMANTIC_CONTRACT_BYTES,
        ),
        "identity_contract": _validate_external_binding(
            bindings.get("identity_contract"), "identity contract",
            "pcqm4mv2_identity_normalization_contract.json",
            IDENTITY_CONTRACT_SHA256, IDENTITY_CONTRACT_BYTES,
        ),
        "payload_contract": _validate_external_binding(
            bindings.get("payload_contract"), "payload contract",
            "p1_pcqm_geometry_payload_format_contract.json",
            PAYLOAD_CONTRACT_SHA256, PAYLOAD_CONTRACT_BYTES,
        ),
    }
    script_binding = _validate_external_binding(
        bindings.get("semantic_gate_script"), "semantic gate script",
        "recompute_pcqm_geometry_semantics_v1.py",
    )
    if expected_script_binding is not None and script_binding != expected_script_binding:
        raise GateError("staged report script binding differs from the running executable")
    ledger_binding = _validate_relative_binding(
        bindings.get("result_ledger"), "result ledger", RESULT_LEDGER_FILE_NAME
    )
    ledger_snapshot = fixed_output_snapshot(
        output_dir, RESULT_LEDGER_FILE_NAME, "semantic result ledger"
    )
    if (ledger_snapshot["bytes"], ledger_snapshot["sha256"]) != (
        ledger_binding["bytes"], ledger_binding["sha256"]
    ):
        raise GateError("semantic result ledger bytes differ from the staged report")
    ledger_rows = parse_canonical_jsonl_snapshot(ledger_snapshot, "semantic result ledger")
    if not ledger_rows:
        raise GateError("semantic result ledger is empty")
    header = ledger_rows[0]
    require_fields(
        header,
        (
            "document_kind", "schema_version", "release_id",
            "release_manifest_sha256", "structural_audit_report_sha256",
            "semantic_plan_sha256", "semantic_gate_contract_sha256",
            "semantic_gate_script_sha256", "selected_admitted_count",
            "selected_reject_count", "raw_smiles_or_molecule_output",
        ),
        "semantic result ledger header",
    )
    counts = report.get("counts", {})
    if not (
        header["document_kind"] == "semantic_recompute_result_header"
        and header["schema_version"] == LEDGER_SCHEMA
        and header["release_id"] == release_id
        and header["release_manifest_sha256"] == FROZEN_RELEASE_MANIFEST_SHA256
        and header["structural_audit_report_sha256"] == FROZEN_STRUCTURAL_AUDIT_SHA256
        and header["semantic_plan_sha256"] == FROZEN_SEMANTIC_PLAN_SHA256
        and header["semantic_gate_contract_sha256"] == SEMANTIC_CONTRACT_SHA256
        and header["semantic_gate_script_sha256"] == script_binding["sha256"]
        and header["selected_admitted_count"] == counts.get("selected_admitted")
        and header["selected_reject_count"] == counts.get("selected_reject")
        and header["raw_smiles_or_molecule_output"] is False
    ):
        raise GateError("semantic result ledger header is misbound")
    report_binding = {
        "relative_path": STAGED_REPORT_FILE_NAME,
        "bytes": report_snapshot["bytes"],
        "sha256": report_snapshot["sha256"],
        REPORT_HASH_FIELD: report[REPORT_HASH_FIELD],
    }
    return {
        "release_id": release_id,
        "semantic_status": semantic_status,
        "report": report,
        "report_binding": report_binding,
        "ledger_binding": ledger_binding,
        "script_binding": script_binding,
        "frozen_bindings": frozen_bindings,
    }


def validate_completed_output(output_dir, expected_script_sha256):
    """Consumer-side minimum: parse semantics and require receipt plus marker."""
    require_sha256(expected_script_sha256, "externally pinned semantic gate script hash")
    output_dir = require_directory(output_dir, "completed output directory")
    receipt_snapshot = fixed_output_snapshot(
        output_dir, COMPLETION_RECEIPT_FILE_NAME, "completion receipt"
    )
    marker_snapshot = fixed_output_snapshot(
        output_dir, COMPLETED_MARKER_FILE_NAME, "COMPLETED marker"
    )
    receipt = parse_canonical_json_snapshot(receipt_snapshot, "completion receipt")
    marker = parse_canonical_json_snapshot(marker_snapshot, "COMPLETED marker")
    receipt_script = receipt.get("semantic_gate_script")
    if not isinstance(receipt_script, dict) or receipt_script.get("sha256") != expected_script_sha256:
        raise GateError("completion receipt differs from the external script trust root")
    bundle = validate_staged_bundle(
        output_dir, receipt.get("release_id"), receipt_script, require_pass=True
    )
    if not (
        receipt.get("schema_version") == COMPLETION_RECEIPT_SCHEMA
        and receipt.get("overall_gate_status") == "pass"
        and receipt.get("consumer_requirement")
        == "receipt_and_COMPLETED_marker_required_for_pass"
        and receipt.get(RECEIPT_HASH_FIELD) == receipt_payload_sha256(receipt)
        and receipt.get("staged_report") == bundle["report_binding"]
        and receipt.get("result_ledger") == bundle["ledger_binding"]
        and receipt.get("frozen_bindings") == bundle["frozen_bindings"]
    ):
        raise GateError("completion receipt is not an authoritative passing receipt")
    marker_projection = dict(marker)
    claimed_marker_hash = marker_projection.pop(
        "completed_marker_canonical_payload_sha256", None
    )
    if not (
        marker.get("schema_version") == COMPLETED_MARKER_SCHEMA
        and marker.get("overall_gate_status") == "pass"
        and marker.get("release_id") == bundle["release_id"]
        and claimed_marker_hash == sha256_json(marker_projection)
    ):
        raise GateError("COMPLETED marker is invalid")
    expected_receipt = marker.get("completion_receipt", {})
    if not (
        expected_receipt.get("bytes") == receipt_snapshot["bytes"]
        and expected_receipt.get("sha256") == receipt_snapshot["sha256"]
        and expected_receipt.get(RECEIPT_HASH_FIELD) == receipt.get(RECEIPT_HASH_FIELD)
        and marker.get("staged_report") == receipt.get("staged_report")
        and marker.get("result_ledger") == receipt.get("result_ledger")
        and marker.get("semantic_gate_script") == receipt.get("semantic_gate_script")
        and marker.get("frozen_bindings") == bundle["frozen_bindings"]
    ):
        raise GateError("COMPLETED marker does not bind the receipt and staged outputs")
    final_observation = marker.get("final_pre_marker_observation", {})
    report_rehash = bundle["report"].get("full_release_artifact_rehash", {})
    receipt_checks = receipt.get("final_checks", {})
    require_same_release_rehash(
        report_rehash.get("before_source_replay", {}),
        report_rehash.get("after_source_replay_before_completion", {}),
    )
    require_same_release_rehash(
        report_rehash.get("after_source_replay_before_completion", {}),
        receipt_checks.get("full_release_artifacts_final_rehash", {}),
    )
    if not (
        receipt_checks.get("source_archive_final_rehash")
        == bundle["report"]["bindings"].get("source_archive")
        and receipt_checks.get("official_data_csv_final_rehash")
        == bundle["report"]["bindings"].get("official_data_csv")
        and final_observation.get("script_sha256") == expected_script_sha256
        and final_observation.get("ledger_sha256")
        == bundle["ledger_binding"]["sha256"]
        and final_observation.get("staged_report_sha256")
        == bundle["report_binding"]["sha256"]
    ):
        raise GateError("completion evidence does not close against the staged bundle")
    return {
        "overall_gate_status": "pass",
        "completion_receipt_sha256": receipt_snapshot["sha256"],
        "completed_marker_sha256": marker_snapshot["sha256"],
    }


def publish_completion(output_dir, release_id, script_binding, final_checks,
                       final_verifier):
    """Derive status from the fixed staged bundle, then publish receipt/marker."""
    script_path = require_regular_file(__file__, "running semantic gate script")
    running_script_bytes, running_script_sha = sha256_file(script_path)
    running_script_binding = {
        "path": str(script_path), "bytes": running_script_bytes,
        "sha256": running_script_sha,
    }
    if script_binding != running_script_binding:
        raise GateError("completion script binding differs from the running executable")
    bundle = validate_staged_bundle(
        output_dir, release_id, running_script_binding, require_pass=False
    )
    overall_pass = bundle["semantic_status"] == "pass"
    pre_receipt_observation = final_verifier("before_completion_receipt")
    bound_final_checks = dict(final_checks)
    bound_final_checks["before_completion_receipt"] = pre_receipt_observation
    receipt = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA,
        "created_utc": utc_now(),
        "release_id": release_id,
        "overall_gate_status": "pass" if overall_pass else "fail",
        "staged_report": bundle["report_binding"],
        "result_ledger": bundle["ledger_binding"],
        "semantic_gate_script": running_script_binding,
        "frozen_bindings": bundle["frozen_bindings"],
        "final_checks": bound_final_checks,
        "consumer_requirement": "receipt_and_COMPLETED_marker_required_for_pass",
    }
    receipt[RECEIPT_HASH_FIELD] = receipt_payload_sha256(receipt)
    receipt_path = Path(output_dir) / COMPLETION_RECEIPT_FILE_NAME
    write_json_new(receipt_path, receipt)
    receipt_size, receipt_sha = sha256_file(receipt_path)
    receipt_binding = {
        "relative_path": receipt_path.name,
        "bytes": receipt_size,
        "sha256": receipt_sha,
        RECEIPT_HASH_FIELD: receipt[RECEIPT_HASH_FIELD],
    }
    marker_path = Path(output_dir) / COMPLETED_MARKER_FILE_NAME
    if overall_pass:
        final_observation = final_verifier("before_completed_marker")
        if sha256_file(receipt_path) != (receipt_size, receipt_sha):
            raise GateError("completion receipt changed before marker publication")
        marker = {
            "schema_version": COMPLETED_MARKER_SCHEMA,
            "created_utc": utc_now(),
            "release_id": release_id,
            "overall_gate_status": "pass",
            "completion_receipt": receipt_binding,
            "staged_report": bundle["report_binding"],
            "result_ledger": bundle["ledger_binding"],
            "semantic_gate_script": running_script_binding,
            "frozen_bindings": bundle["frozen_bindings"],
            "final_pre_marker_observation": final_observation,
        }
        marker["completed_marker_canonical_payload_sha256"] = sha256_json(marker)
        write_json_new(marker_path, marker)
    return receipt_path, marker_path if marker_path.is_file() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--identity-contract", required=True)
    parser.add_argument("--payload-contract", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--structural-audit-report", required=True)
    parser.add_argument("--semantic-plan", required=True)
    parser.add_argument("--runtime-attestation", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    script_path = Path(__file__).resolve()
    script_bytes, script_sha = sha256_file(script_path)

    contract_snapshot, contract = load_pinned_json_snapshot(
        args.contract, "semantic recompute contract", SEMANTIC_CONTRACT_SHA256
    )
    identity_snapshot, identity_contract = load_pinned_json_snapshot(
        args.identity_contract, "identity contract", IDENTITY_CONTRACT_SHA256,
        IDENTITY_CONTRACT_BYTES,
    )
    payload_snapshot, payload_contract = load_pinned_json_snapshot(
        args.payload_contract, "payload contract", PAYLOAD_CONTRACT_SHA256,
        PAYLOAD_CONTRACT_BYTES,
    )
    validate_contract(contract)
    validate_identity_contract(identity_contract)
    validate_payload_contract(payload_contract)
    contract_obs = snapshot_observation(contract_snapshot)
    identity_obs = snapshot_observation(identity_snapshot)
    payload_obs = snapshot_observation(payload_snapshot)

    release_root = require_directory(args.release_root, "release root")
    manifest_snapshot = require_frozen_evidence_snapshot(
        release_root / "full_release_manifest.json", "full release manifest",
        "full_release_manifest.json", FROZEN_RELEASE_MANIFEST_SHA256,
    )
    manifest = parse_json_snapshot(manifest_snapshot, "full release manifest")
    validate_release_manifest(manifest, contract)
    manifest_path = manifest_snapshot["path"]
    manifest_bytes, manifest_sha = manifest_snapshot["bytes"], manifest_snapshot["sha256"]
    audit_snapshot = require_frozen_evidence_snapshot(
        args.structural_audit_report, "structural audit report",
        "independent_audit_report.json", FROZEN_STRUCTURAL_AUDIT_SHA256,
    )
    structural_report = parse_json_snapshot(audit_snapshot, "structural audit report")
    validate_structural_report(structural_report)
    audit_path = audit_snapshot["path"]
    audit_bytes, audit_sha = audit_snapshot["bytes"], audit_snapshot["sha256"]
    plan_snapshot = require_frozen_evidence_snapshot(
        args.semantic_plan, "semantic plan", "semantic_review_plan.jsonl",
        FROZEN_SEMANTIC_PLAN_SHA256,
    )
    plan_path = plan_snapshot["path"]
    plan_bytes, plan_sha = plan_snapshot["bytes"], plan_snapshot["sha256"]
    plan_rows = parse_canonical_jsonl_snapshot(plan_snapshot, "semantic plan")
    plan_header, admitted_plan, reject_plan = validate_plan(
        plan_rows, plan_sha, manifest_sha, structural_report, contract
    )
    configuration = manifest["configuration"]
    if structural_report.get("release_id") != manifest.get("release_id") or plan_header.get("release_id") != manifest.get("release_id"):
        raise GateError("release ID differs across manifest, audit, and plan")

    runtime_attestation_expected = configuration.get("runtime_attestation_sha256")
    require_sha256(runtime_attestation_expected, "release runtime attestation hash")
    (
        runtime_lock, e3fp_import_root, e3fp_package_root, e3fp_attested_closure,
        e3fp_closure_before_import, runtime_snapshot,
    ) = verify_runtime_and_e3fp_closure(
        args.runtime_attestation, runtime_attestation_expected, args.e3fp_source
    )
    staged = configuration.get("staged_inputs", {})
    archive_path, archive_obs = validate_artifact(
        args.archive, staged.get("train_3d_sdf_archive"), "train-3D SDF archive"
    )
    data_csv_path, data_csv_obs = validate_artifact(
        args.data_csv, staged.get("companion_data_csv_gz"), "official data CSV"
    )
    locked_member = configuration.get("locked_sdf_member")
    if not isinstance(locked_member, dict):
        raise GateError("release lacks a locked SDF member")
    for key in ("tar_member_name", "member_type", "uncompressed_bytes", "sha256"):
        if key not in locked_member:
            raise GateError("locked SDF member field missing")
    require_sha256(locked_member["sha256"], "locked SDF member hash")

    release_rehash_before = verify_full_release_artifacts(
        release_root, manifest, "before_source_replay"
    )

    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        raise GateError("--output-dir must be new")
    try:
        output_dir.resolve().relative_to(release_root)
        raise GateError("semantic output directory must be outside the release root")
    except ValueError:
        pass
    output_dir.mkdir(parents=True, exist_ok=False)
    staging_path = output_dir / "STAGING.json"
    write_json_new(
        staging_path,
        {
            "schema_version": STAGING_SCHEMA,
            "created_utc": utc_now(),
            "release_id": manifest["release_id"],
            "completion_status": "unfinalized",
            "authoritative_overall_gate_status": None,
            "semantic_gate_script_sha256": script_sha,
            "consumer_warning": "not_trustable_without_completion_receipt_and_COMPLETED",
        },
    )

    try:
        import numpy as np
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise GateError("NumPy and RDKit are required") from exc
    expectations, shard_observations = collect_release_expectations(
        np, release_root, manifest, admitted_plan, reject_plan,
        contract["reject_recompute"]["required_reason_stage_diagnostic"],
    )
    official_smiles = read_selected_csv(data_csv_path, expectations)
    if sha256_file(script_path) != (script_bytes, script_sha):
        raise GateError("semantic gate script changed before source replay")
    e3fp_api = import_locked_e3fp(e3fp_import_root, e3fp_package_root)
    e3fp_closure_after_import = scan_e3fp_closure(
        e3fp_package_root, e3fp_attested_closure, "after_import_before_source",
        e3fp_api["imported_module_paths"],
    )
    require_same_e3fp_closure(
        [e3fp_closure_before_import, e3fp_closure_after_import]
    )
    rows, member_observation = stream_and_recompute(
        Chem, np, e3fp_api, archive_path, locked_member, official_smiles,
        expectations, archive_obs["sha256"], rdBase.rdkitVersion,
    )
    if sha256_file(script_path) != (script_bytes, script_sha):
        raise GateError("semantic gate script changed during source replay")
    e3fp_closure_after_source = scan_e3fp_closure(
        e3fp_package_root, e3fp_attested_closure, "after_source_before_output",
        e3fp_api["imported_module_paths"],
    )
    require_same_e3fp_closure(
        [e3fp_closure_before_import, e3fp_closure_after_import,
         e3fp_closure_after_source]
    )
    release_rehash_after = verify_full_release_artifacts(
        release_root, manifest, "after_source_replay_before_completion"
    )
    require_same_release_rehash(release_rehash_before, release_rehash_after)
    archive_final_path, archive_obs_after = validate_artifact(
        archive_path, staged.get("train_3d_sdf_archive"), "train-3D SDF archive final"
    )
    csv_final_path, data_csv_obs_after = validate_artifact(
        data_csv_path, staged.get("companion_data_csv_gz"), "official data CSV final"
    )
    if archive_final_path != archive_path or archive_obs_after != archive_obs:
        raise GateError("source archive changed across source replay")
    if csv_final_path != data_csv_path or data_csv_obs_after != data_csv_obs:
        raise GateError("official data CSV changed across source replay")

    status_counts = Counter(row["status"] for row in rows)
    disposition_counts = Counter(row["disposition"] for row in rows)
    terminal_counts = Counter(row["recomputed_terminal_classification"] for row in rows)
    mismatch_counts = Counter(code for row in rows for code in row["mismatch_codes"])
    inchi_counts = Counter()
    for row in rows:
        observation = row["inchi_corroboration"]
        token = observation.get("status", "unknown")
        if token == "ok":
            token = "ok_connectivity_{}_strict_{}".format(
                str(observation.get("connectivity_block_equal")).lower(),
                str(observation.get("full_key_equal")).lower(),
            )
        inchi_counts[token] += 1
    overall_pass = (
        status_counts.get("fail", 0) == 0
        and disposition_counts == {"admit": len(admitted_plan), "reject": len(reject_plan)}
        and terminal_counts.get("strict_isomeric_match", 0) == len(admitted_plan)
        and {
            key: terminal_counts.get(key, 0)
            for key in contract["reject_recompute"]["expected_reason_counts"]
        } == contract["reject_recompute"]["expected_reason_counts"]
    )

    ledger_header = {
        "document_kind": "semantic_recompute_result_header",
        "schema_version": LEDGER_SCHEMA,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": manifest_sha,
        "structural_audit_report_sha256": audit_sha,
        "semantic_plan_sha256": plan_sha,
        "semantic_gate_contract_sha256": contract_obs["sha256"],
        "semantic_gate_script_sha256": script_sha,
        "selected_admitted_count": len(admitted_plan),
        "selected_reject_count": len(reject_plan),
        "raw_smiles_or_molecule_output": False,
    }
    ledger_path = output_dir / RESULT_LEDGER_FILE_NAME
    write_jsonl_new(ledger_path, [ledger_header] + rows)
    ledger_bytes, ledger_sha = sha256_file(ledger_path)
    runtime = runtime_observation(np, rdBase, e3fp_api)
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "semantic_recompute_status": "pass" if overall_pass else "fail",
        "completion_status": "pending_authoritative_receipt",
        "authoritative_overall_gate_status": None,
        "gate_class": "all_reject_identity_and_preregistered_admitted_e3fp_exact_recompute_v1",
        "release_id": manifest["release_id"],
        "bindings": {
            "release_manifest": {"path": str(manifest_path), "bytes": manifest_bytes, "sha256": manifest_sha},
            "structural_audit_report": {"path": str(audit_path), "bytes": audit_bytes, "sha256": audit_sha},
            "semantic_plan": {"path": str(plan_path), "bytes": plan_bytes, "sha256": plan_sha},
            "semantic_gate_contract": contract_obs,
            "identity_contract": identity_obs,
            "payload_contract": payload_obs,
            "semantic_gate_script": {"path": str(script_path), "bytes": script_bytes, "sha256": script_sha},
            "runtime_attestation": runtime_lock,
            "source_archive": archive_obs,
            "official_data_csv": data_csv_obs,
            "streamed_sdf_member": member_observation,
            "result_ledger": {"relative_path": ledger_path.name, "bytes": ledger_bytes, "sha256": ledger_sha},
        },
        "full_release_artifact_rehash": {
            "before_source_replay": release_rehash_before,
            "after_source_replay_before_completion": release_rehash_after,
        },
        "e3fp_closure_rehash": [
            e3fp_closure_before_import,
            e3fp_closure_after_import,
            e3fp_closure_after_source,
        ],
        "runtime_observation": runtime,
        "runtime_observation_sha256": sha256_json(runtime),
        "counts": {
            "selected_total": len(rows),
            "selected_admitted": disposition_counts.get("admit", 0),
            "selected_reject": disposition_counts.get("reject", 0),
            "passed_records": status_counts.get("pass", 0),
            "failed_records": status_counts.get("fail", 0),
            "terminal_classification_counts": dict(sorted(terminal_counts.items())),
            "mismatch_code_counts": dict(sorted(mismatch_counts.items())),
            "inchi_corroboration_counts": dict(sorted(inchi_counts.items())),
            "release_shards_touched": len(shard_observations),
        },
        "shard_observations": shard_observations,
        "passed_checks": [
            "v3_plan_exact_selection_consumed",
            "all_release_reject_ledgers_equal_plan",
            "archive_csv_and_sdf_member_content_addressed",
            "all_declared_release_artifacts_rehashed_before_and_after_source_replay",
            "production_parent_to_worker_rdkit_binary_transport_reproduced",
            "all_reject_identity_classifications_and_molecule_hashes_exact",
            "admitted_sample_identity_atom_mapping_coordinates_and_e3fp_exact",
            "standalone_script_and_e3fp_source_closure_stable",
            "raw_molecular_content_not_serialized",
        ] if overall_pass else [],
        "limitations": [
            "The admitted set is a deterministic engineering replay sample, not a statistical scientific-validity sample.",
            "The same content-addressed historical E3FP implementation is replayed through an independent harness; external-implementation equivalence is not claimed.",
            "Identity-rejected members stop at their first terminal identity stage; downstream E3FP characterization belongs to a separate ablation.",
            "InChIKey is reported as non-gating corroboration because its normalization semantics differ from the frozen RDKit-SMILES contract.",
            "No conclusion is made about motif optimality, E3FP scientific superiority, MSE benefit, overlap, tokenizer binding, or P1 admission.",
        ],
        "pass_boundary": contract["pass_boundary"],
        "p1_training_admission": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report[REPORT_HASH_FIELD] = report_payload_sha256(report)
    report_path = output_dir / STAGED_REPORT_FILE_NAME
    write_json_new(report_path, report)
    if sha256_file(script_path) != (script_bytes, script_sha):
        raise GateError("semantic gate script changed after staged output publication")
    report_snapshot = read_file_snapshot(report_path, "written staged semantic report")
    written = parse_canonical_json_snapshot(
        report_snapshot, "written staged semantic report"
    )
    if written.get(REPORT_HASH_FIELD) != report_payload_sha256(written):
        raise GateError("written staged semantic report self hash mismatch")
    validate_staged_report_claims(written)

    e3fp_closure_after_staged = scan_e3fp_closure(
        e3fp_package_root, e3fp_attested_closure,
        "after_staged_output_before_completion", e3fp_api["imported_module_paths"],
    )
    closure_observations = [
        e3fp_closure_before_import,
        e3fp_closure_after_import,
        e3fp_closure_after_source,
        e3fp_closure_after_staged,
    ]
    require_same_e3fp_closure(closure_observations)

    critical_final = {
        "semantic_gate_contract": require_snapshot_unchanged(
            contract_snapshot, "semantic recompute contract"
        ),
        "identity_contract": require_snapshot_unchanged(
            identity_snapshot, "identity contract"
        ),
        "payload_contract": require_snapshot_unchanged(
            payload_snapshot, "payload contract"
        ),
        "release_manifest": require_snapshot_unchanged(
            manifest_snapshot, "full release manifest"
        ),
        "structural_audit_report": require_snapshot_unchanged(
            audit_snapshot, "structural audit report"
        ),
        "semantic_plan": require_snapshot_unchanged(plan_snapshot, "semantic plan"),
        "runtime_attestation": require_snapshot_unchanged(
            runtime_snapshot, "production runtime attestation"
        ),
    }
    script_binding = {"path": str(script_path), "bytes": script_bytes, "sha256": script_sha}

    def final_verifier(phase):
        if sha256_file(script_path) != (script_bytes, script_sha):
            raise GateError("semantic gate script changed during finalization")
        for label, snapshot in (
            ("semantic recompute contract", contract_snapshot),
            ("identity contract", identity_snapshot),
            ("payload contract", payload_snapshot),
            ("full release manifest", manifest_snapshot),
            ("structural audit report", audit_snapshot),
            ("semantic plan", plan_snapshot),
            ("production runtime attestation", runtime_snapshot),
        ):
            require_snapshot_unchanged(snapshot, label)
        if sha256_file(ledger_path) != (ledger_bytes, ledger_sha):
            raise GateError("result ledger changed during finalization")
        if sha256_file(report_path) != (
            report_snapshot["bytes"], report_snapshot["sha256"]
        ):
            raise GateError("staged report changed during finalization")
        closure_observation = scan_e3fp_closure(
            e3fp_package_root, e3fp_attested_closure, phase,
            e3fp_api["imported_module_paths"],
        )
        closure_observations.append(closure_observation)
        require_same_e3fp_closure(closure_observations)
        observation = {
            "script_sha256": script_sha,
            "critical_input_aggregate_sha256": sha256_json(critical_final),
            "e3fp_closure": closure_observation,
            "ledger_sha256": ledger_sha,
            "staged_report_sha256": report_snapshot["sha256"],
        }
        return observation

    final_checks = {
        "critical_inputs_final_rehash": critical_final,
        "source_archive_final_rehash": archive_obs_after,
        "official_data_csv_final_rehash": data_csv_obs_after,
        "full_release_artifacts_final_rehash": release_rehash_after,
        "e3fp_closure_after_staged_output": e3fp_closure_after_staged,
    }
    receipt_path, completed_path = publish_completion(
        output_dir, manifest["release_id"], script_binding,
        final_checks, final_verifier,
    )
    if overall_pass:
        validate_completed_output(output_dir, script_sha)
    elif completed_path is not None or (output_dir / COMPLETED_MARKER_FILE_NAME).exists():
        raise GateError("a failing semantic recomputation published COMPLETED")
    print(
        json.dumps(
            {
                "overall_gate_status": "pass" if overall_pass else "fail",
                "selected_admitted": report["counts"]["selected_admitted"],
                "selected_reject": report["counts"]["selected_reject"],
                "failed_records": report["counts"]["failed_records"],
                "staged_report": str(report_path),
                "completion_receipt": str(receipt_path),
                "completed_marker": str(completed_path) if completed_path else None,
            },
            sort_keys=True,
        )
    )
    return 0 if overall_pass else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(json.dumps({"gate_status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        sys.exit(1)
