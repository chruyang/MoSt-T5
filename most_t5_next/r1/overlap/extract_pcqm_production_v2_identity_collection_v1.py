#!/usr/bin/env python3
"""Extract a hash-only P1 identity collection from PCQM production-v2.

The release is treated as immutable evidence.  Every manifest and declared
artifact is byte/SHA-256 checked, admitted LMDB values are independently
decoded from the public non-executable v2 wire format, and rejects are checked
but never emitted.  This program never writes inside ``--release-root``.
"""

from __future__ import print_function

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sqlite3
import struct
import sys
from collections import Counter
from pathlib import Path


EXTRACTOR_CONTRACT_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-extraction-contract/v1"
CONFIG_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-extraction-config/v1"
RECEIPT_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-extraction-receipt/v1"
SOURCE_LOCK_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-source-lock/v1"
COLLECTION_SCHEMA = "most-t5-r1/identity-collection-manifest/v1"
MOLECULE_ROW_SCHEMA = "most-t5-r1/molecule-identity-row/v1"

FULL_RELEASE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
SHARD_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-shard/v2"
PRODUCTION_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-release-contract/v2"
PAYLOAD_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-format-contract/v2"
IDENTITY_CONTRACT_SCHEMA = "most-t5-r1/pcqm4mv2-identity-normalization-contract/v1"
PAYLOAD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-payload/v2"
PRODUCTION_RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-pretokenizer-record/v2"
PRODUCTION_MODE = "full_sharded_production"
PAYLOAD_INDEX_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-index-row/v2"
IDENTITY_NAMESPACE = "ogb_pcqm4mv2_train_row_index"

MAGIC = b"MST5PCQM2\x00"
HEADER_LENGTH_BYTES = 4
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BLOCKS = 100000
HEX64 = frozenset("0123456789abcdef")
WIRE_DTYPES = {"int32": "<i4", "float32": "<f4", "bool": "|b1"}

CONFIG_FIELDS = frozenset(("schema_version", "extraction_id", "contract_sha256", "collection", "release_lock"))
COLLECTION_CONFIG_FIELDS = frozenset(
    ("collection_id", "dataset_id", "release_id", "phase", "split", "role", "task_family", "source_identity_namespace")
)
RELEASE_LOCK_FIELDS = frozenset(
    (
        "release_manifest_relative_path", "expected_release_manifest_bytes",
        "expected_release_manifest_sha256", "expected_release_id",
        "expected_production_contract_sha256", "expected_payload_contract_sha256",
        "expected_identity_normalization_contract_sha256", "required_rdkit_version",
    )
)
TOP_FIELDS = frozenset(
    (
        "schema_version", "created_utc", "release_status", "release_id",
        "logical_release_root_sha256", "configuration", "counts",
        "global_motif_census", "shards", "range_no_gap_no_overlap",
        "lmdb_merged", "tokenizer_binding", "p1_training_admission",
        "p1_training_launcher_permitted", "next_gate",
    )
)
TOP_CONFIGURATION_FIELDS = frozenset(
    (
        "release_id", "production_contract_sha256", "runtime_attestation_sha256",
        "staged_input_receipt_sha256", "source_contract_sha256", "release_kind",
        "source_record_count", "selected_record_count", "selected_ordinal_range",
        "selected_ordinal_set_sha256", "shard_size", "shard_count", "staged_inputs",
        "locked_sdf_member", "harness", "logical_record_schema_version", "sidecar_mode",
    )
)
TOP_COUNT_FIELDS = frozenset(
    (
        "source_record_count", "membership_record_count", "admitted_record_count",
        "reject_ledger_record_count", "shard_count", "unique_motif_count",
        "motif_occurrence_count",
    )
)
TOP_SHARD_FIELDS = frozenset(("shard_index", "range_start", "range_end", "shard_manifest_sha256"))
ARTIFACT_FIELDS = frozenset(("relative_path", "bytes", "sha256"))
SHARD_FIELDS = frozenset(
    (
        "schema_version", "created_utc", "release_status", "release_id",
        "production_contract_sha256", "shard_index", "range_start", "range_end",
        "selected_record_count", "counts", "reject_reason_counts",
        "e3fp_params_sha256_values", "artifacts", "partition_invariant_pass",
        "lmdb_merged", "p1_training_admission",
    )
)
SHARD_COUNT_FIELDS = frozenset(
    (
        "membership_record_count", "admitted_record_count", "reject_ledger_record_count",
        "payload_index_record_count", "payload_wire_total_bytes",
        "motif_occurrence_count", "unique_motif_count",
    )
)
ARTIFACT_PATHS = {
    "geometry_records_lmdb_data": "geometry_records.lmdb/data.mdb",
    "membership": "membership.jsonl",
    "reject_ledger": "reject_ledger.jsonl",
    "payload_index": "payload_index.jsonl",
    "motif_census": "motif_census.jsonl",
}
MEMBERSHIP_FIELDS = frozenset(
    (
        "record_schema_version", "sidecar_id", "sidecar_mode",
        "selected_ordinal_set_sha256", "member_id", "sdf_record_index",
        "official_csv_row_index", "source_address_sha256", "disposition",
        "record_storage_key", "record_content_sha256", "reject_reason_code",
    )
)
REJECT_FIELDS = frozenset(
    (
        "record_schema_version", "sidecar_id", "sidecar_mode",
        "selected_ordinal_set_sha256", "member_id", "sdf_record_index",
        "official_csv_row_index", "source_address_sha256", "stage",
        "reason_code", "action", "geometry_mse_enabled",
        "source_mol_identity_sha256", "geometry_mol_identity_sha256",
        "diagnostic_code", "detail_sha256",
    )
)
PAYLOAD_INDEX_FIELDS = frozenset(
    ("payload_index_schema_version", "record_storage_key", "record_wire_bytes", "record_wire_sha256", "record_content_sha256")
)
HEADER_FIELDS = frozenset(("payload_schema_version", "record", "array_blocks", "logical_record_sha256"))
BLOCK_FIELDS = frozenset(("index", "dtype", "shape", "order", "offset", "nbytes", "sha256"))
PLACEHOLDER_FIELDS = frozenset(("__array_block__", "dtype", "shape", "order", "sha256"))
RECORD_FIELDS = frozenset(
    ("record_schema_version", "sidecar", "member", "identity", "atom_universe", "topology", "geometry", "array_metadata")
)
SIDECAR_FIELDS = frozenset(
    (
        "sidecar_id", "sidecar_mode", "selected_ordinal_set_sha256",
        "source_contract_sha256", "identity_normalization_contract_sha256",
        "adapter_harness_sha256", "record_schema_sha256",
        "geometry_only_pretokenizer", "p1_training_admission",
        "p1_training_launcher_permitted",
    )
)
MEMBER_FIELDS = frozenset(
    (
        "identity_namespace", "member_id", "sdf_record_index",
        "official_csv_row_index", "storage_key", "source_archive_sha256",
        "source_address_sha256", "source_mol_identity_sha256",
    )
)
IDENTITY_FIELDS = frozenset(
    (
        "official_identity_status", "sdf_strict_smiles_sha256",
        "official_strict_smiles_sha256", "canonical_connectivity_sha256",
        "identity_spec_sha256", "rdkit_version",
    )
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_json(value):
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and not (set(value) - HEX64)


def require_sha256(value, label, nullable=False):
    if value is None and nullable:
        return
    if not is_sha256(value):
        raise ValueError("{} must be a lowercase SHA-256{}".format(label, " or null" if nullable else ""))


def require_string(value, label):
    if not isinstance(value, str) or not value or any(character in value for character in "\x00\r\n\t"):
        raise ValueError("{} must be a non-empty control-free string".format(label))
    return value


def require_exact_fields(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(label))
    actual = frozenset(value)
    if actual != expected:
        raise ValueError("{} fields differ; missing={}, extra={}".format(label, sorted(expected - actual), sorted(actual - expected)))


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: {}".format(key))
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant is forbidden: {}".format(value))


def load_json(path, label):
    with open(str(path), "r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicate_pairs, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(label))
    return value


def strict_json_bytes(raw, label):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs, parse_constant=reject_nonfinite)
    except Exception as exc:
        raise ValueError("{} is not strict UTF-8 JSON: {}".format(label, exc))
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(label))
    return value


def regular_file(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("{} is not a regular non-symlink file: {}".format(label, path))
    return path


def regular_directory(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise NotADirectoryError("{} is not a regular non-symlink directory: {}".format(label, path))
    return path


def record_observation(observations, path, label):
    path = regular_file(path, label).resolve()
    observed = sha256_file(path)
    if path in observations and observations[path] != observed:
        raise RuntimeError("{} changed between observations".format(label))
    observations[path] = observed
    return {"path": str(path), "bytes": observed[0], "sha256": observed[1]}


def verify_observations_unchanged(observations):
    for path, expected in observations.items():
        if sha256_file(path) != expected:
            raise RuntimeError("source evidence changed during extraction: {}".format(path))


def iter_canonical_jsonl(path, label):
    with open(str(path), "rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if raw == b"\n" or not raw.endswith(b"\n"):
                raise ValueError("{} line {} is blank or lacks one LF".format(label, line_number))
            value = strict_json_bytes(raw[:-1], "{} line {}".format(label, line_number))
            if canonical_json_bytes(value) + b"\n" != raw:
                raise ValueError("{} line {} is not canonical JSONL".format(label, line_number))
            yield value


def _next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def member_id(ordinal):
    return "{}:{}".format(IDENTITY_NAMESPACE, int(ordinal))


def storage_key(ordinal):
    return "{:09d}".format(int(ordinal))


def validate_config(config, contract_sha256):
    require_exact_fields(config, CONFIG_FIELDS, "extraction config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("extraction config schema mismatch")
    require_string(config["extraction_id"], "extraction_id")
    require_sha256(config["contract_sha256"], "config contract SHA-256")
    if config["contract_sha256"] != contract_sha256:
        raise ValueError("config does not bind the supplied extraction contract")
    collection = config["collection"]
    require_exact_fields(collection, COLLECTION_CONFIG_FIELDS, "collection")
    for key in COLLECTION_CONFIG_FIELDS:
        require_string(collection[key], "collection.{}".format(key))
    if not (
        collection["dataset_id"] == "pcqm4mv2"
        and collection["phase"] == "p1"
        and collection["split"] == "train"
        and collection["role"] == "p1_structure_train"
        and collection["task_family"] == "none"
        and collection["source_identity_namespace"] == IDENTITY_NAMESPACE
    ):
        raise ValueError("collection is not the closed PCQM P1 structural membership role")
    release = config["release_lock"]
    require_exact_fields(release, RELEASE_LOCK_FIELDS, "release_lock")
    if release["release_manifest_relative_path"] != "full_release_manifest.json":
        raise ValueError("only full_release_manifest.json is accepted")
    if not is_int(release["expected_release_manifest_bytes"]) or release["expected_release_manifest_bytes"] <= 0:
        raise ValueError("expected release-manifest bytes must be positive")
    for key in (
        "expected_release_manifest_sha256", "expected_production_contract_sha256",
        "expected_payload_contract_sha256", "expected_identity_normalization_contract_sha256",
    ):
        require_sha256(release[key], "release_lock.{}".format(key))
    require_string(release["expected_release_id"], "release_lock.expected_release_id")
    require_string(release["required_rdkit_version"], "release_lock.required_rdkit_version")
    if collection["release_id"] != release["expected_release_id"]:
        raise ValueError("collection release_id differs from the locked source release")


def validate_contract_documents(extraction_contract, production_contract, payload_contract, identity_contract):
    if extraction_contract.get("schema_version") != EXTRACTOR_CONTRACT_SCHEMA:
        raise ValueError("identity extraction contract schema mismatch")
    if production_contract.get("schema_version") != PRODUCTION_CONTRACT_SCHEMA:
        raise ValueError("production contract schema mismatch")
    logical = production_contract.get("logical_record")
    if not isinstance(logical, dict) or not (
        logical.get("schema_version") == PRODUCTION_RECORD_SCHEMA
        and logical.get("mode") == PRODUCTION_MODE
        and logical.get("p1_training_admission") is False
    ):
        raise ValueError("production contract logical-record boundary mismatch")
    if payload_contract.get("schema_version") != PAYLOAD_CONTRACT_SCHEMA:
        raise ValueError("payload contract schema mismatch")
    if not (
        payload_contract.get("payload_schema_version") == PAYLOAD_SCHEMA
        and payload_contract.get("magic_ascii") == MAGIC.decode("ascii")
        and set(payload_contract.get("header_required_fields", [])) == set(HEADER_FIELDS)
        and payload_contract.get("array_block_required_fields") == [
            "index", "dtype", "shape", "order", "offset", "nbytes", "sha256"
        ]
        and set(payload_contract.get("allowed_dtypes", [])) == set(WIRE_DTYPES)
    ):
        raise ValueError("payload contract framing/field boundary mismatch")
    framing = payload_contract.get("framing")
    if not isinstance(framing, dict) or not (
        framing.get("max_header_bytes") == MAX_HEADER_BYTES
        and framing.get("max_payload_bytes") == MAX_PAYLOAD_BYTES
    ):
        raise ValueError("payload contract safety bounds mismatch")
    if identity_contract.get("schema_version") != IDENTITY_CONTRACT_SCHEMA:
        raise ValueError("identity normalization contract schema mismatch")


def _validate_shape(shape, label):
    if not isinstance(shape, list) or len(shape) > 8:
        raise RuntimeError("{} shape is malformed".format(label))
    result = []
    for dimension in shape:
        if not is_int(dimension) or dimension < 0:
            raise RuntimeError("{} contains an invalid dimension".format(label))
        result.append(dimension)
    return result


def _array_descriptor(value):
    return {
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "order": "C",
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def _logical_projection(np, value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": _array_descriptor(value)}
    if isinstance(value, dict):
        return {str(key): _logical_projection(np, value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_logical_projection(np, item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError("unsupported value in logical record projection")


def _rehydrate_arrays(value, arrays, consumed):
    if isinstance(value, dict):
        if "__array_block__" in value:
            require_exact_fields(value, PLACEHOLDER_FIELDS, "array placeholder")
            index = value["__array_block__"]
            if not is_int(index) or index not in arrays or index in consumed:
                raise RuntimeError("array placeholder index is invalid or duplicated")
            array = arrays[index]
            descriptor = _array_descriptor(array)
            if not (
                value["dtype"] == descriptor["dtype"]
                and value["shape"] == descriptor["shape"]
                and value["order"] == "C"
                and value["sha256"] == descriptor["sha256"]
            ):
                raise RuntimeError("array placeholder disagrees with its block")
            consumed.add(index)
            return array
        return {key: _rehydrate_arrays(item, arrays, consumed) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate_arrays(item, arrays, consumed) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError("payload JSON contains an unsupported value")


def decode_payload_independently(np, payload):
    if sys.byteorder != "little":
        raise RuntimeError("v2 payload extraction requires a little-endian runtime")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise RuntimeError("LMDB payload is not bytes-like")
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RuntimeError("payload exceeds the public safety bound")
    prefix = len(MAGIC) + HEADER_LENGTH_BYTES
    if len(payload) < prefix or payload[: len(MAGIC)] != MAGIC:
        raise RuntimeError("payload magic/version mismatch")
    header_size = struct.unpack(">I", payload[len(MAGIC):prefix])[0]
    if header_size < 2 or header_size > MAX_HEADER_BYTES or prefix + header_size > len(payload):
        raise RuntimeError("payload header length is invalid")
    header_raw = payload[prefix:prefix + header_size]
    header = strict_json_bytes(header_raw, "payload header")
    if canonical_json_bytes(header) != header_raw:
        raise RuntimeError("payload header is not canonical JSON")
    require_exact_fields(header, HEADER_FIELDS, "payload header")
    if header["payload_schema_version"] != PAYLOAD_SCHEMA or not isinstance(header["record"], dict):
        raise RuntimeError("payload schema/record projection mismatch")
    require_sha256(header["logical_record_sha256"], "payload logical hash")
    blocks = header["array_blocks"]
    if not isinstance(blocks, list) or len(blocks) > MAX_ARRAY_BLOCKS:
        raise RuntimeError("payload block list is malformed")
    raw_blocks = payload[prefix + header_size:]
    arrays = {}
    expected_offset = 0
    for expected_index, block in enumerate(blocks):
        require_exact_fields(block, BLOCK_FIELDS, "array block {}".format(expected_index))
        if block["index"] != expected_index or block["dtype"] not in WIRE_DTYPES or block["order"] != "C":
            raise RuntimeError("array block index/dtype/order is invalid")
        shape = _validate_shape(block["shape"], "array block {}".format(expected_index))
        offset, nbytes = block["offset"], block["nbytes"]
        if not is_int(offset) or offset != expected_offset or not is_int(nbytes) or nbytes < 0:
            raise RuntimeError("array block offset/length is invalid")
        count = 1
        for dimension in shape:
            count *= dimension
            if count > (1 << 62):
                raise RuntimeError("array block shape exceeds the safety bound")
        expected_nbytes = count * np.dtype(WIRE_DTYPES[block["dtype"]]).itemsize
        if nbytes != expected_nbytes or offset + nbytes > len(raw_blocks):
            raise RuntimeError("array block length disagrees with dtype/shape")
        block_raw = raw_blocks[offset:offset + nbytes]
        require_sha256(block["sha256"], "array block SHA-256")
        if sha256_bytes(block_raw) != block["sha256"]:
            raise RuntimeError("array block SHA-256 mismatch")
        wire = np.frombuffer(block_raw, dtype=np.dtype(WIRE_DTYPES[block["dtype"]]))
        arrays[expected_index] = np.ascontiguousarray(
            wire.reshape(tuple(shape), order="C").astype(np.dtype(block["dtype"]), copy=True)
        )
        expected_offset += nbytes
    if expected_offset != len(raw_blocks):
        raise RuntimeError("payload contains trailing or unreferenced bytes")
    consumed = set()
    record = _rehydrate_arrays(header["record"], arrays, consumed)
    if consumed != set(arrays):
        raise RuntimeError("payload contains an unreferenced array block")
    logical_hash = sha256_json(_logical_projection(np, record))
    if logical_hash != header["logical_record_sha256"]:
        raise RuntimeError("LMDB record logical hash mismatch")
    return record, logical_hash


def validate_release_manifest(top, config, contract_hashes):
    require_exact_fields(top, TOP_FIELDS, "full release manifest")
    release_lock = config["release_lock"]
    if not (
        top["schema_version"] == FULL_RELEASE_SCHEMA
        and top["release_status"] == "complete"
        and top["release_id"] == release_lock["expected_release_id"]
        and top["range_no_gap_no_overlap"] is True
        and top["lmdb_merged"] is False
        and top["tokenizer_binding"] == "absent_and_forbidden"
        and top["p1_training_admission"] is False
        and top["p1_training_launcher_permitted"] is False
    ):
        raise ValueError("full release status/identity boundary mismatch")
    require_sha256(top["logical_release_root_sha256"], "logical release root")
    configuration = top["configuration"]
    require_exact_fields(configuration, TOP_CONFIGURATION_FIELDS, "release configuration")
    if not (
        configuration["release_id"] == top["release_id"]
        and configuration["production_contract_sha256"] == contract_hashes["production"]
        and configuration["release_kind"] == "full_production"
        and configuration["logical_record_schema_version"] == PRODUCTION_RECORD_SCHEMA
        and configuration["sidecar_mode"] == PRODUCTION_MODE
    ):
        raise ValueError("release configuration binding mismatch")
    for key in (
        "production_contract_sha256", "runtime_attestation_sha256",
        "staged_input_receipt_sha256", "source_contract_sha256",
        "selected_ordinal_set_sha256",
    ):
        require_sha256(configuration[key], "release configuration {}".format(key))
    harness = configuration["harness"]
    require_exact_fields(harness, frozenset(("components", "bundle_sha256")), "release harness")
    if not isinstance(harness["components"], dict):
        raise ValueError("release harness components must be an object")
    components = harness["components"]
    for key, expected in (
        ("production_contract", contract_hashes["production"]),
        ("payload_contract", contract_hashes["payload"]),
        ("identity_contract", contract_hashes["identity"]),
    ):
        if components.get(key) != expected:
            raise ValueError("release harness {} hash mismatch".format(key))
    require_sha256(harness["bundle_sha256"], "release harness bundle")
    if harness["bundle_sha256"] != sha256_json(components):
        raise ValueError("release harness bundle hash mismatch")
    selected_count = configuration["selected_record_count"]
    source_count = configuration["source_record_count"]
    if not (
        is_int(selected_count) and selected_count > 0
        and source_count == selected_count
        and configuration["selected_ordinal_range"] == [0, selected_count]
        and is_int(configuration["shard_size"]) and configuration["shard_size"] > 0
        and is_int(configuration["shard_count"]) and configuration["shard_count"] > 0
    ):
        raise ValueError("full release selected-range/count invariant failed")
    counts = top["counts"]
    require_exact_fields(counts, TOP_COUNT_FIELDS, "release counts")
    for key in TOP_COUNT_FIELDS:
        if not is_int(counts[key]) or counts[key] < 0:
            raise ValueError("release count {} is invalid".format(key))
    if not (
        counts["source_record_count"] == selected_count
        and counts["membership_record_count"] == selected_count
        and counts["membership_record_count"] == counts["admitted_record_count"] + counts["reject_ledger_record_count"]
        and counts["shard_count"] == configuration["shard_count"]
    ):
        raise ValueError("release count partition invariant failed")
    if not isinstance(top["shards"], list) or len(top["shards"]) != counts["shard_count"]:
        raise ValueError("release shard-root cardinality mismatch")
    return configuration, counts


def validate_release_envelope(release_root, shard_count):
    expected_shards = {"shard-{:06d}".format(index) for index in range(shard_count)}
    allowed_files = {"full_release_manifest.json", "motif_census.jsonl", "production_scope.json", "run_state.json"}
    observed_files, observed_dirs = set(), set()
    for path in release_root.iterdir():
        if path.is_symlink():
            raise RuntimeError("source release contains a symlink: {}".format(path))
        if path.is_file():
            observed_files.add(path.name)
        elif path.is_dir():
            observed_dirs.add(path.name)
        else:
            raise RuntimeError("source release contains an unsupported filesystem entry: {}".format(path))
    if not {"full_release_manifest.json", "motif_census.jsonl"}.issubset(observed_files):
        raise RuntimeError("source release lacks a required top-level artifact")
    if observed_files - allowed_files:
        raise RuntimeError("source release contains an undeclared artifact: {}".format(sorted(observed_files - allowed_files)))
    if observed_dirs != expected_shards:
        raise RuntimeError("source release shard directory envelope mismatch")


def validate_shard_envelope(shard_dir):
    allowed_files = {"shard_manifest.json", "membership.jsonl", "reject_ledger.jsonl", "payload_index.jsonl", "motif_census.jsonl"}
    observed_files, observed_dirs = set(), set()
    for path in shard_dir.iterdir():
        if path.is_symlink():
            raise RuntimeError("shard contains a symlink: {}".format(path))
        if path.is_file():
            observed_files.add(path.name)
        elif path.is_dir():
            observed_dirs.add(path.name)
        else:
            raise RuntimeError("shard contains an unsupported filesystem entry: {}".format(path))
    if observed_files != allowed_files or observed_dirs != {"geometry_records.lmdb"}:
        raise RuntimeError("shard contains an undeclared artifact or lacks a required artifact")
    lmdb_dir = regular_directory(shard_dir / "geometry_records.lmdb", "shard LMDB directory")
    lmdb_files = set()
    for path in lmdb_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("LMDB directory contains an unsupported entry: {}".format(path))
        lmdb_files.add(path.name)
    if "data.mdb" not in lmdb_files or lmdb_files - {"data.mdb", "lock.mdb"}:
        raise RuntimeError("LMDB directory contains an undeclared artifact or lacks data.mdb")
    return lmdb_files


def verify_artifact(shard_dir, role, declaration, observations):
    require_exact_fields(declaration, ARTIFACT_FIELDS, "shard artifact {}".format(role))
    expected_relative = ARTIFACT_PATHS[role]
    if declaration["relative_path"] != expected_relative:
        raise ValueError("shard artifact path mismatch for {}".format(role))
    if not is_int(declaration["bytes"]) or declaration["bytes"] < 0:
        raise ValueError("shard artifact bytes are invalid")
    require_sha256(declaration["sha256"], "shard artifact SHA-256")
    path = regular_file(shard_dir / Path(expected_relative), "shard artifact {}".format(role))
    observed = record_observation(observations, path, "shard artifact {}".format(role))
    if (observed["bytes"], observed["sha256"]) != (declaration["bytes"], declaration["sha256"]):
        raise RuntimeError("shard artifact bytes/SHA-256 mismatch for {}".format(role))
    return path, {"relative_path": expected_relative, "bytes": observed["bytes"], "sha256": observed["sha256"]}


def validate_membership(row, release_id, selected_hash, ordinal):
    require_exact_fields(row, MEMBERSHIP_FIELDS, "membership row")
    if not (
        row["record_schema_version"] == PRODUCTION_RECORD_SCHEMA
        and row["sidecar_id"] == release_id
        and row["sidecar_mode"] == PRODUCTION_MODE
        and row["selected_ordinal_set_sha256"] == selected_hash
        and row["member_id"] == member_id(ordinal)
        and row["sdf_record_index"] == ordinal
        and row["official_csv_row_index"] == ordinal
    ):
        raise RuntimeError("membership row identity/order mismatch at ordinal {}".format(ordinal))
    require_sha256(row["source_address_sha256"], "membership source address")
    if row["disposition"] == "admit":
        if row["record_storage_key"] != storage_key(ordinal) or row["reject_reason_code"] is not None:
            raise RuntimeError("admitted membership conditional fields are invalid")
        require_sha256(row["record_content_sha256"], "membership content hash")
    elif row["disposition"] == "reject":
        if row["record_storage_key"] is not None or row["record_content_sha256"] is not None:
            raise RuntimeError("rejected membership contains admitted fields")
        require_string(row["reject_reason_code"], "membership reject reason")
    else:
        raise RuntimeError("membership disposition is not closed")


def validate_payload_index(row, membership, payload):
    require_exact_fields(row, PAYLOAD_INDEX_FIELDS, "payload-index row")
    if not (
        row["payload_index_schema_version"] == PAYLOAD_INDEX_SCHEMA
        and row["record_storage_key"] == membership["record_storage_key"]
        and row["record_content_sha256"] == membership["record_content_sha256"]
        and row["record_wire_bytes"] == len(payload)
        and row["record_wire_sha256"] == sha256_bytes(payload)
    ):
        raise RuntimeError("payload-index/LMDB/membership binding mismatch")
    require_sha256(row["record_wire_sha256"], "payload-index wire hash")


def validate_reject(row, membership):
    require_exact_fields(row, REJECT_FIELDS, "reject-ledger row")
    common = (
        "record_schema_version", "sidecar_id", "sidecar_mode", "selected_ordinal_set_sha256",
        "member_id", "sdf_record_index", "official_csv_row_index", "source_address_sha256",
    )
    if any(row[key] != membership[key] for key in common):
        raise RuntimeError("reject-ledger/membership binding mismatch")
    if not (
        row["reason_code"] == membership["reject_reason_code"]
        and row["action"] == "exclude_from_geometry_release"
        and row["geometry_mse_enabled"] is False
    ):
        raise RuntimeError("reject-ledger reason/action boundary mismatch")
    for key in ("stage", "reason_code", "diagnostic_code"):
        require_string(row[key], "reject-ledger {}".format(key))
    require_sha256(row["source_mol_identity_sha256"], "reject source identity", nullable=True)
    require_sha256(row["geometry_mol_identity_sha256"], "reject geometry identity", nullable=True)
    require_sha256(row["detail_sha256"], "reject detail hash")
    expected_detail = sha256_json(
        {
            "diagnostic_code": row["diagnostic_code"],
            "reason_code": row["reason_code"],
            "source_address_sha256": row["source_address_sha256"],
            "stage": row["stage"],
        }
    )
    if row["detail_sha256"] != expected_detail:
        raise RuntimeError("reject-ledger detail hash mismatch")


def validate_record_identity(record, membership, configuration, config, contract_hashes):
    require_exact_fields(record, RECORD_FIELDS, "decoded production record")
    if record["record_schema_version"] != PRODUCTION_RECORD_SCHEMA:
        raise RuntimeError("decoded production record schema mismatch")
    for key in ("atom_universe", "topology", "geometry", "array_metadata"):
        if not isinstance(record[key], dict):
            raise RuntimeError("decoded record {} must be an object".format(key))
    sidecar = record["sidecar"]
    require_exact_fields(sidecar, SIDECAR_FIELDS, "decoded record sidecar")
    if not (
        sidecar["sidecar_id"] == configuration["release_id"]
        and sidecar["sidecar_mode"] == PRODUCTION_MODE
        and sidecar["selected_ordinal_set_sha256"] == configuration["selected_ordinal_set_sha256"]
        and sidecar["source_contract_sha256"] == configuration["source_contract_sha256"]
        and sidecar["identity_normalization_contract_sha256"] == contract_hashes["identity"]
        and sidecar["adapter_harness_sha256"] == configuration["harness"]["bundle_sha256"]
        and sidecar["record_schema_sha256"] == contract_hashes["production"]
        and sidecar["geometry_only_pretokenizer"] is True
        and sidecar["p1_training_admission"] is False
        and sidecar["p1_training_launcher_permitted"] is False
    ):
        raise RuntimeError("decoded record sidecar binding/status mismatch")
    member = record["member"]
    require_exact_fields(member, MEMBER_FIELDS, "decoded record member")
    if not (
        member["identity_namespace"] == IDENTITY_NAMESPACE
        and member["member_id"] == membership["member_id"]
        and member["sdf_record_index"] == membership["sdf_record_index"]
        and member["official_csv_row_index"] == membership["official_csv_row_index"]
        and member["storage_key"] == membership["record_storage_key"]
        and member["source_address_sha256"] == membership["source_address_sha256"]
    ):
        raise RuntimeError("decoded record member/membership binding mismatch")
    for key in ("source_archive_sha256", "source_address_sha256", "source_mol_identity_sha256"):
        require_sha256(member[key], "decoded record member {}".format(key))
    identity = record["identity"]
    require_exact_fields(identity, IDENTITY_FIELDS, "decoded record identity")
    if not (
        identity["official_identity_status"] == "strict_isomeric_match"
        and identity["sdf_strict_smiles_sha256"] == identity["official_strict_smiles_sha256"]
        and identity["identity_spec_sha256"] == contract_hashes["identity"]
        and identity["rdkit_version"] == config["release_lock"]["required_rdkit_version"]
    ):
        raise RuntimeError("decoded record identity spec/hash/version mismatch")
    for key in (
        "sdf_strict_smiles_sha256", "official_strict_smiles_sha256",
        "canonical_connectivity_sha256", "identity_spec_sha256",
    ):
        require_sha256(identity[key], "decoded record identity {}".format(key))
    return identity["canonical_connectivity_sha256"], identity["sdf_strict_smiles_sha256"]


def create_sort_database(path):
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE rows (member_id TEXT PRIMARY KEY COLLATE BINARY, row BLOB NOT NULL)")
    return connection


def write_molecule_rows(connection, path):
    file_digest, key_digest = hashlib.sha256(), hashlib.sha256()
    byte_count = row_count = 0
    previous = None
    with open(str(path), "xb") as handle:
        for member, row in connection.execute("SELECT member_id,row FROM rows ORDER BY member_id COLLATE BINARY"):
            encoded = member.encode("utf-8")
            if previous is not None and encoded <= previous:
                raise RuntimeError("SQLite output is not strictly UTF-8-key sorted")
            previous = encoded
            raw = bytes(row) + b"\n"
            handle.write(raw)
            file_digest.update(raw)
            key_digest.update(encoded + b"\n")
            byte_count += len(raw)
            row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "bytes": byte_count,
        "sha256": file_digest.hexdigest(),
        "row_count": row_count,
        "key_lf_sha256": key_digest.hexdigest(),
    }


def write_json_new(path, value):
    path = Path(path)
    with open(str(path), "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def artifact_of(path):
    observed = sha256_file(path)
    return {"path": path.name, "bytes": observed[0], "sha256": observed[1]}


def relative_observations(release_root, observations):
    rows = []
    for path, observed in sorted(observations.items(), key=lambda item: item[0].as_posix()):
        rows.append(
            {
                "relative_path": path.relative_to(release_root).as_posix(),
                "bytes": observed[0],
                "sha256": observed[1],
            }
        )
    return rows


def extract_collection(
    extraction_contract_path, config_path, release_root, production_contract_path,
    payload_contract_path, identity_contract_path, output_dir,
):
    extraction_contract_path = regular_file(extraction_contract_path, "extraction contract").resolve()
    config_path = regular_file(config_path, "extraction config").resolve()
    production_contract_path = regular_file(production_contract_path, "production contract").resolve()
    payload_contract_path = regular_file(payload_contract_path, "payload contract").resolve()
    identity_contract_path = regular_file(identity_contract_path, "identity normalization contract").resolve()
    release_root = regular_directory(Path(release_root).expanduser().resolve(), "release root")
    output_dir = Path(output_dir).expanduser().resolve()
    try:
        if os.path.commonpath((str(release_root), str(output_dir))) == str(release_root):
            raise ValueError("output directory must be outside the immutable release root")
    except ValueError:
        if str(output_dir).startswith(str(release_root)):
            raise

    extractor_path = Path(__file__).resolve()
    fixed_observations = {
        "extraction_contract": sha256_file(extraction_contract_path),
        "config": sha256_file(config_path),
        "production_contract": sha256_file(production_contract_path),
        "payload_contract": sha256_file(payload_contract_path),
        "identity_contract": sha256_file(identity_contract_path),
        "extractor": sha256_file(extractor_path),
    }
    extraction_contract = load_json(extraction_contract_path, "extraction contract")
    config = load_json(config_path, "extraction config")
    production_contract = load_json(production_contract_path, "production contract")
    payload_contract = load_json(payload_contract_path, "payload contract")
    identity_contract = load_json(identity_contract_path, "identity contract")
    validate_contract_documents(extraction_contract, production_contract, payload_contract, identity_contract)
    validate_config(config, fixed_observations["extraction_contract"][1])
    contract_hashes = {
        "production": fixed_observations["production_contract"][1],
        "payload": fixed_observations["payload_contract"][1],
        "identity": fixed_observations["identity_contract"][1],
    }
    release_lock = config["release_lock"]
    for name, key in (
        ("production", "expected_production_contract_sha256"),
        ("payload", "expected_payload_contract_sha256"),
        ("identity", "expected_identity_normalization_contract_sha256"),
    ):
        if contract_hashes[name] != release_lock[key]:
            raise ValueError("supplied {} contract differs from config lock".format(name))

    source_observations = {}
    top_path = regular_file(release_root / "full_release_manifest.json", "full release manifest")
    top_observed = record_observation(source_observations, top_path, "full release manifest")
    if (top_observed["bytes"], top_observed["sha256"]) != (
        release_lock["expected_release_manifest_bytes"], release_lock["expected_release_manifest_sha256"]
    ):
        raise RuntimeError("full release manifest differs from the config bytes/SHA-256 lock")
    top = load_json(top_path, "full release manifest")
    configuration, top_counts = validate_release_manifest(top, config, contract_hashes)
    validate_release_envelope(release_root, configuration["shard_count"])
    for optional in ("production_scope.json", "run_state.json"):
        candidate = release_root / optional
        if candidate.exists():
            record_observation(source_observations, candidate, "optional release control file")

    global_decl = top["global_motif_census"]
    require_exact_fields(global_decl, ARTIFACT_FIELDS, "global motif census")
    if global_decl["relative_path"] != "motif_census.jsonl":
        raise ValueError("global motif census path mismatch")
    require_sha256(global_decl["sha256"], "global motif census SHA-256")
    global_path = release_root / "motif_census.jsonl"
    global_observed = record_observation(source_observations, global_path, "global motif census")
    if (global_observed["bytes"], global_observed["sha256"]) != (global_decl["bytes"], global_decl["sha256"]):
        raise RuntimeError("global motif census bytes/SHA-256 mismatch")

    try:
        import lmdb
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PCQM production-v2 extraction requires python-lmdb and NumPy") from exc

    output_dir.mkdir(parents=True, exist_ok=False)
    sort_path = output_dir / ".pcqm_identity_sort.sqlite3"
    connection = create_sort_database(sort_path)
    observed_counts = Counter()
    reject_reasons = Counter()
    shard_receipts = []
    expected_start = 0
    try:
        for expected_index, root_entry in enumerate(top["shards"]):
            require_exact_fields(root_entry, TOP_SHARD_FIELDS, "release shard root")
            if root_entry["shard_index"] != expected_index:
                raise RuntimeError("release shard indices are not contiguous")
            require_sha256(root_entry["shard_manifest_sha256"], "release shard-manifest SHA-256")
            shard_dir = regular_directory(release_root / "shard-{:06d}".format(expected_index), "release shard")
            lmdb_files = validate_shard_envelope(shard_dir)
            if "lock.mdb" in lmdb_files:
                record_observation(source_observations, shard_dir / "geometry_records.lmdb" / "lock.mdb", "LMDB runtime lock file")
            shard_manifest_path = shard_dir / "shard_manifest.json"
            manifest_observed = record_observation(source_observations, shard_manifest_path, "shard manifest")
            if manifest_observed["sha256"] != root_entry["shard_manifest_sha256"]:
                raise RuntimeError("top-level shard-manifest SHA-256 mismatch")
            manifest = load_json(shard_manifest_path, "shard manifest")
            require_exact_fields(manifest, SHARD_FIELDS, "shard manifest")
            if not (
                manifest["schema_version"] == SHARD_MANIFEST_SCHEMA
                and manifest["release_status"] == "complete"
                and manifest["release_id"] == configuration["release_id"]
                and manifest["production_contract_sha256"] == contract_hashes["production"]
                and manifest["shard_index"] == expected_index
                and manifest["range_start"] == root_entry["range_start"] == expected_start
                and manifest["range_end"] == root_entry["range_end"]
                and manifest["partition_invariant_pass"] is True
                and manifest["lmdb_merged"] is False
                and manifest["p1_training_admission"] is False
            ):
                raise RuntimeError("shard identity/range/status binding mismatch")
            start, end = manifest["range_start"], manifest["range_end"]
            if not is_int(start) or not is_int(end) or end <= start or end > configuration["selected_record_count"]:
                raise RuntimeError("shard range is invalid")
            selected = end - start
            counts = manifest["counts"]
            require_exact_fields(counts, SHARD_COUNT_FIELDS, "shard counts")
            if any(not is_int(counts[key]) or counts[key] < 0 for key in SHARD_COUNT_FIELDS):
                raise RuntimeError("shard count is invalid")
            if not (
                manifest["selected_record_count"] == selected
                and counts["membership_record_count"] == selected
                and counts["admitted_record_count"] + counts["reject_ledger_record_count"] == selected
                and counts["payload_index_record_count"] == counts["admitted_record_count"]
            ):
                raise RuntimeError("shard count/partition invariant failed")
            if not isinstance(manifest["reject_reason_counts"], dict):
                raise RuntimeError("shard reject-reason counts are malformed")
            if not isinstance(manifest["e3fp_params_sha256_values"], list):
                raise RuntimeError("shard E3FP hash list is malformed")
            artifacts = manifest["artifacts"]
            if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_PATHS):
                raise RuntimeError("shard artifact declaration set mismatch")
            paths, artifact_receipt = {}, {}
            for role in sorted(ARTIFACT_PATHS):
                paths[role], artifact_receipt[role] = verify_artifact(
                    shard_dir, role, artifacts[role], source_observations
                )

            membership_iter = iter(iter_canonical_jsonl(paths["membership"], "membership"))
            reject_iter = iter(iter_canonical_jsonl(paths["reject_ledger"], "reject ledger"))
            index_iter = iter(iter_canonical_jsonl(paths["payload_index"], "payload index"))
            env = lmdb.open(
                str(shard_dir / "geometry_records.lmdb"), subdir=True, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=8, create=False,
            )
            local = Counter()
            local_reasons = Counter()
            local_wire_bytes = 0
            try:
                with env.begin(write=False) as transaction:
                    db_iter = iter(transaction.cursor())
                    for ordinal in range(start, end):
                        membership = _next(membership_iter)
                        if membership is None:
                            raise RuntimeError("membership ended before shard range")
                        validate_membership(membership, configuration["release_id"], configuration["selected_ordinal_set_sha256"], ordinal)
                        local["membership"] += 1
                        if membership["disposition"] == "reject":
                            reject = _next(reject_iter)
                            if reject is None:
                                raise RuntimeError("reject ledger ended before rejected membership")
                            validate_reject(reject, membership)
                            local["rejected"] += 1
                            local_reasons[reject["reason_code"]] += 1
                            continue
                        index_row = _next(index_iter)
                        db_item = _next(db_iter)
                        if index_row is None or db_item is None:
                            raise RuntimeError("payload index or LMDB ended before admitted membership")
                        raw_key, raw_payload = db_item
                        raw_key, raw_payload = bytes(raw_key), bytes(raw_payload)
                        if raw_key.startswith(b"__"):
                            raise RuntimeError("undeclared LMDB metadata key is forbidden")
                        expected_key = membership["record_storage_key"].encode("ascii")
                        if raw_key != expected_key:
                            raise RuntimeError("LMDB keys do not equal admitted membership keys")
                        validate_payload_index(index_row, membership, raw_payload)
                        record, logical_hash = decode_payload_independently(np, raw_payload)
                        if logical_hash != membership["record_content_sha256"] or logical_hash != index_row["record_content_sha256"]:
                            raise RuntimeError("LMDB record logical hash differs from membership/payload index")
                        connectivity, stereo = validate_record_identity(
                            record, membership, configuration, config, contract_hashes
                        )
                        molecule_row = {
                            "schema_version": MOLECULE_ROW_SCHEMA,
                            "collection_id": config["collection"]["collection_id"],
                            "member_id": membership["member_id"],
                            "connectivity_identity_sha256": connectivity,
                            "stereo_identity_sha256": stereo,
                            "conformer_identity_sha256": None,
                        }
                        try:
                            connection.execute(
                                "INSERT INTO rows VALUES (?,?)",
                                (membership["member_id"], sqlite3.Binary(canonical_json_bytes(molecule_row))),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise RuntimeError("duplicate admitted member ID") from exc
                        local["admitted"] += 1
                        local["payload_index"] += 1
                        local["lmdb_keys"] += 1
                        local["decoded_payloads"] += 1
                        local_wire_bytes += len(raw_payload)
                    if any(_next(iterator) is not None for iterator in (membership_iter, reject_iter, index_iter, db_iter)):
                        raise RuntimeError("shard streams contain excess rows, keys, or metadata")
            finally:
                env.close()
            if not (
                local["membership"] == counts["membership_record_count"]
                and local["admitted"] == counts["admitted_record_count"]
                and local["rejected"] == counts["reject_ledger_record_count"]
                and local["payload_index"] == counts["payload_index_record_count"]
                and local["lmdb_keys"] == counts["admitted_record_count"]
                and local_wire_bytes == counts["payload_wire_total_bytes"]
                and dict(sorted(local_reasons.items())) == manifest["reject_reason_counts"]
            ):
                raise RuntimeError("observed shard streams disagree with manifest counts")
            connection.commit()
            observed_counts.update(local)
            reject_reasons.update(local_reasons)
            shard_receipts.append(
                {
                    "shard_index": expected_index,
                    "range_start": start,
                    "range_end": end,
                    "shard_manifest_sha256": manifest_observed["sha256"],
                    "artifacts": artifact_receipt,
                    "counts": dict(sorted(local.items())),
                    "reject_reason_counts": dict(sorted(local_reasons.items())),
                }
            )
            expected_start = end

        if expected_start != configuration["selected_record_count"]:
            raise RuntimeError("shard ranges do not cover the selected range")
        if not (
            observed_counts["membership"] == top_counts["membership_record_count"]
            and observed_counts["admitted"] == top_counts["admitted_record_count"]
            and observed_counts["rejected"] == top_counts["reject_ledger_record_count"]
            and observed_counts["decoded_payloads"] == top_counts["admitted_record_count"]
        ):
            raise RuntimeError("observed global streams disagree with release counts")
        if observed_counts["admitted"] <= 0:
            raise RuntimeError("a proof-compatible identity collection cannot be empty")
        logical_root = sha256_json(
            {
                "configuration": configuration,
                "global_motif_census_sha256": global_observed["sha256"],
                "shards": top["shards"],
                "membership_record_count": observed_counts["membership"],
                "admitted_record_count": observed_counts["admitted"],
                "reject_ledger_record_count": observed_counts["rejected"],
            }
        )
        if logical_root != top["logical_release_root_sha256"]:
            raise RuntimeError("top-level logical release root mismatch")
        validate_release_envelope(release_root, configuration["shard_count"])
        verify_observations_unchanged(source_observations)
        for name, path in (
            ("extraction_contract", extraction_contract_path), ("config", config_path),
            ("production_contract", production_contract_path), ("payload_contract", payload_contract_path),
            ("identity_contract", identity_contract_path), ("extractor", extractor_path),
        ):
            if sha256_file(path) != fixed_observations[name]:
                raise RuntimeError("{} bytes changed during extraction".format(name))

        molecule_path = output_dir / "molecule_identity_rows.jsonl"
        molecule_artifact = write_molecule_rows(connection, molecule_path)
        if molecule_artifact["row_count"] != observed_counts["admitted"]:
            raise RuntimeError("emitted molecule row count differs from admitted count")
        source_lock = {
            "schema_version": SOURCE_LOCK_SCHEMA,
            "extraction_id": config["extraction_id"],
            "release_root": str(release_root),
            "release_id": configuration["release_id"],
            "release_manifest": {
                "relative_path": "full_release_manifest.json",
                "bytes": top_observed["bytes"],
                "sha256": top_observed["sha256"],
                "logical_release_root_sha256": logical_root,
            },
            "contract_sha256": contract_hashes,
            "source_files": relative_observations(release_root, source_observations),
            "shards": shard_receipts,
            "excluded_membership_dispositions": ["reject"],
            "excluded_lmdb_metadata_keys": [],
            "permitted_non_evidence_lmdb_runtime_files": ["lock.mdb"],
            "source_open_mode": "readonly_lock_false_create_false",
        }
        source_lock_path = output_dir / "source_lock.json"
        write_json_new(source_lock_path, source_lock)
        resolved_config_path = output_dir / "resolved_config.json"
        write_json_new(resolved_config_path, config)
        collection = config["collection"]
        collection_manifest = {
            "schema_version": COLLECTION_SCHEMA,
            "collection_id": collection["collection_id"],
            "dataset_id": collection["dataset_id"],
            "release_id": collection["release_id"],
            "phase": collection["phase"],
            "split": collection["split"],
            "role": collection["role"],
            "task_family": collection["task_family"],
            "identity_specs": {
                "connectivity_identity_spec_sha256": contract_hashes["identity"],
                "stereo_identity_spec_sha256": contract_hashes["identity"],
                "conformer_identity": {"status": "unavailable", "spec_sha256": None},
                "text_identity": {"status": "unavailable", "exact_spec_sha256": None, "normalized_spec_sha256": None},
            },
            "molecule_rows": molecule_artifact,
            "text_pair_rows": None,
            "provenance": {
                "source_identity_namespace": IDENTITY_NAMESPACE,
                "source_release_manifest_sha256": top_observed["sha256"],
                "extractor_sha256": fixed_observations["extractor"][1],
                "excluded_source_metadata_keys": [],
            },
        }
        collection_manifest_path = output_dir / "collection_manifest.json"
        write_json_new(collection_manifest_path, collection_manifest)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "pass",
            "generated_at_utc": utc_now(),
            "extraction_id": config["extraction_id"],
            "p1_training_admission": False,
            "p2_training_admission": False,
            "counts": {
                "source_membership_rows": observed_counts["membership"],
                "admitted_payloads_independently_decoded": observed_counts["decoded_payloads"],
                "rejected_members_filtered": observed_counts["rejected"],
                "lmdb_member_keys": observed_counts["lmdb_keys"],
                "emitted_molecule_rows": molecule_artifact["row_count"],
                "shard_count": len(shard_receipts),
            },
            "reject_reason_counts": dict(sorted(reject_reasons.items())),
            "artifacts": {
                "source_lock": artifact_of(source_lock_path),
                "resolved_config": artifact_of(resolved_config_path),
                "molecule_rows": molecule_artifact,
                "collection_manifest": artifact_of(collection_manifest_path),
            },
            "provenance": {
                "release_manifest_sha256": top_observed["sha256"],
                "logical_release_root_sha256": logical_root,
                "extraction_contract_sha256": fixed_observations["extraction_contract"][1],
                "config_sha256": fixed_observations["config"][1],
                "production_contract_sha256": contract_hashes["production"],
                "payload_contract_sha256": contract_hashes["payload"],
                "identity_normalization_contract_sha256": contract_hashes["identity"],
                "extractor_sha256": fixed_observations["extractor"][1],
                "required_rdkit_version": release_lock["required_rdkit_version"],
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "lmdb": getattr(lmdb, "__version__", "unknown"),
                "platform": platform.platform(),
            },
            "passed_checks": [
                "closed_release_and_shard_artifact_envelopes",
                "all_manifest_and_artifact_bytes_sha256_before_and_after",
                "membership_reject_payload_index_lmdb_partition_closure",
                "all_admitted_payloads_independently_decoded",
                "all_wire_and_logical_record_hashes",
                "identity_spec_release_member_and_rdkit_bindings",
                "rejects_and_lmdb_metadata_excluded",
                "proof_gate_compatible_hash_only_collection",
            ],
            "policy_boundary": "No P1/P2 overlap policy, downstream split, tokenizer, or training-admission decision is made.",
        }
        receipt["receipt_canonical_payload_sha256"] = sha256_json(receipt)
        write_json_new(output_dir / "extraction_receipt.json", receipt)
        return receipt
    finally:
        connection.close()
        if sort_path.exists():
            sort_path.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-contract", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--production-contract", required=True)
    parser.add_argument("--payload-contract", required=True)
    parser.add_argument("--identity-normalization-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    receipt = extract_collection(
        args.extraction_contract, args.config, args.release_root,
        args.production_contract, args.payload_contract,
        args.identity_normalization_contract, args.output_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
