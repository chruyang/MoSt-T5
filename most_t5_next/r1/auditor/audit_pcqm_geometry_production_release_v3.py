#!/usr/bin/env python3
"""Independently audit a sharded PCQM geometry production release.

This v3 auditor is standalone: its own exact bytes and execution-runtime
observation are content-addressed into both the pre-registered plan and final
report.  The report also carries a canonical payload hash that excludes only
that hash field itself.

The audit intentionally depends only on the published JSON contracts, Python's
standard library, NumPy, and the LMDB reader.  It performs a full, streaming
partition/key/hash audit and decodes only a deterministic, pre-registered
stratified sample of admitted payloads.  Every rejected member is put into the
semantic-review plan.

This gate is read-only with respect to the release.  Its output directory must
be new and outside the release root.  Passing this gate is evidence about the
release envelope and sampled logical records; it is not a tokenizer binding,
an independent molecular-feature recomputation, or P1 training admission.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
from collections import Counter
from pathlib import Path


AUDIT_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-independent-audit-contract/v3"
AUDIT_REPORT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-independent-audit-report/v3"
SEMANTIC_PLAN_SCHEMA = "most-t5-r1/p1-pcqm-geometry-semantic-review-plan/v3"
AUDIT_RUNTIME_SCHEMA = "most-t5-r1/p1-pcqm-geometry-audit-runtime-observation/v3"
REPORT_PAYLOAD_HASH_FIELD = "report_canonical_payload_sha256"
AUDIT_REPORT_FIELDS = frozenset(
    (
        "schema_version", "audit_status", "audit_class", "release_id",
        "release_manifest_sha256", "audit_contract_sha256",
        "production_contract_sha256", "payload_contract_sha256",
        "auditor_script", "audit_runtime_observation",
        "audit_runtime_observation_sha256", "counts", "passed_checks",
        "semantic_review_plan", "limitations", REPORT_PAYLOAD_HASH_FIELD,
    )
)
PRODUCTION_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-release-contract/v2"
PAYLOAD_CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-format-contract/v2"
PAYLOAD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-payload/v2"
PRODUCTION_RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-pretokenizer-record/v2"
PRODUCTION_MODE = "full_sharded_production"
SHARD_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-shard/v2"
FULL_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
BENCHMARK_REPORT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-benchmark/v2"
SCOPE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-scope/v2"
PAYLOAD_INDEX_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-index-row/v2"
IDENTITY_NAMESPACE = "ogb_pcqm4mv2_train_row_index"
MAGIC = b"MST5PCQM2\x00"
HEADER_LENGTH_BYTES = 4
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_ARRAY_BLOCKS = 100000
HEX64 = frozenset("0123456789abcdef")

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
    (
        "payload_index_schema_version", "record_storage_key",
        "record_wire_bytes", "record_wire_sha256", "record_content_sha256",
    )
)
ARTIFACT_PATHS = {
    "geometry_records_lmdb_data": "geometry_records.lmdb/data.mdb",
    "membership": "membership.jsonl",
    "reject_ledger": "reject_ledger.jsonl",
    "payload_index": "payload_index.jsonl",
    "motif_census": "motif_census.jsonl",
}
HEADER_FIELDS = frozenset(
    ("payload_schema_version", "record", "array_blocks", "logical_record_sha256")
)
BLOCK_FIELDS = frozenset(("index", "dtype", "shape", "order", "offset", "nbytes", "sha256"))
PLACEHOLDER_FIELDS = frozenset(("__array_block__", "dtype", "shape", "order", "sha256"))
WIRE_DTYPES = {"int32": "<i4", "float32": "<f4", "bool": "|b1"}
FORBIDDEN_RECORD_FIELDS = frozenset(
    (
        "raw_smiles", "smiles", "canonical_smiles", "official_smiles",
        "sdf_smiles", "source_smiles", "generated_smiles", "topology_smiles",
        "motif_fragment_sequence", "reconstructed_mol", "tokenizer_binding",
        "tokenizer_contract_sha256", "id_to_token_sha256", "full_input_ids",
        "unmasked_input_ids", "motif_ordinal_to_unmasked_token_index",
        "token_geometry_valid_mask", "joint_mask_positions",
        "geo_only_mask_positions", "geometry_input_mask",
        "geometry_target_mask", "mask_positions",
    )
)


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_json(value):
    return sha256_bytes(canonical_json_bytes(value))


def motif_lexeme_sha256(fragment):
    if not isinstance(fragment, str) or not fragment:
        raise RuntimeError("motif fragment must be non-empty text")
    return sha256_bytes(fragment.encode("utf-8"))


def register_motif_binding(counts, lexemes, digest, fragment, count):
    require_sha256(digest, "motif lexeme SHA-256")
    if motif_lexeme_sha256(fragment) != digest:
        raise RuntimeError("motif lexeme digest does not match exact UTF-8 fragment")
    if not is_int(count) or count < 1:
        raise RuntimeError("motif census count must be a positive integer")
    prior = lexemes.get(digest)
    if prior is not None and prior != fragment:
        raise RuntimeError("motif lexeme SHA-256 collision detected")
    lexemes[digest] = fragment
    counts[digest] += count


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def require_sha256(value, label, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or set(value) - HEX64:
        raise RuntimeError("{} must be a lowercase SHA-256".format(label))


def require_exact_fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise RuntimeError("{} fields are not closed: {}".format(label, observed))


def regular_file(path, label):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, path))
    return path.resolve()


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("JSON contains duplicate key: {}".format(key))
        result[key] = value
    return result


def _reject_nonfinite(token):
    raise RuntimeError("JSON contains non-finite token: {}".format(token))


def strict_json_bytes(raw, label):
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_nonfinite,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("{} is not valid UTF-8 JSON".format(label)) from exc
    return value


def load_json(path, label):
    path = regular_file(path, label)
    with open(str(path), "rb") as handle:
        value = strict_json_bytes(handle.read(), label)
    if not isinstance(value, dict):
        raise RuntimeError("{} must contain a JSON object".format(label))
    return path, value


def iter_canonical_jsonl(path, label):
    path = regular_file(path, label)
    with open(str(path), "rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n") or raw == b"\n":
                raise RuntimeError("{} line {} is blank or unterminated".format(label, line_number))
            body = raw[:-1]
            value = strict_json_bytes(body, "{} line {}".format(label, line_number))
            if canonical_json_bytes(value) != body:
                raise RuntimeError("{} line {} is not canonical JSON".format(label, line_number))
            if not isinstance(value, dict):
                raise RuntimeError("{} line {} must be an object".format(label, line_number))
            yield value


def write_json_new(path, value):
    path = Path(path)
    with open(str(path), "x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_plan_new(path, rows):
    path = Path(path)
    with open(str(path), "xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def observe_auditor_script():
    """Content-address the exact standalone script that is executing."""
    unresolved = Path(__file__)
    if unresolved.is_symlink():
        raise RuntimeError("the v3 auditor script must not be reached through a symlink")
    path = regular_file(unresolved, "v3 auditor script")
    return {
        "file_name": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def verify_auditor_script(observation):
    require_exact_fields(observation, ("file_name", "bytes", "sha256"), "auditor script observation")
    current = observe_auditor_script()
    if current != observation:
        raise RuntimeError("v3 auditor script bytes changed during the audit")


def observe_audit_runtime(np):
    """Record, but do not cross-machine-normalize, the audit runtime."""
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("the v3 independent audit requires python-lmdb") from exc
    binding_version = getattr(lmdb, "__version__", None)
    if not isinstance(binding_version, str) or not binding_version:
        try:
            binding_version = importlib.metadata.version("lmdb")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("python-lmdb binding version is unavailable") from exc
    library_version = lmdb.version() if hasattr(lmdb, "version") else None
    if not isinstance(library_version, tuple) or not library_version or not all(is_int(item) for item in library_version):
        raise RuntimeError("python-lmdb does not expose a closed liblmdb version tuple")
    libc_name, libc_version = platform.libc_ver()
    return {
        "schema_version": AUDIT_RUNTIME_SCHEMA,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "machine": platform.machine() or "unknown",
        },
        "libc": {
            "name": libc_name or "unknown",
            "version": libc_version or "unknown",
        },
        "numpy_version": str(np.__version__),
        "python_lmdb_version": binding_version,
        "liblmdb_version": ".".join(str(item) for item in library_version),
        "byteorder": sys.byteorder,
    }


def validate_audit_runtime_observation(observation):
    require_exact_fields(
        observation,
        (
            "schema_version", "python", "platform", "libc", "numpy_version",
            "python_lmdb_version", "liblmdb_version", "byteorder",
        ),
        "audit runtime observation",
    )
    if observation["schema_version"] != AUDIT_RUNTIME_SCHEMA:
        raise RuntimeError("audit runtime observation schema mismatch")
    require_exact_fields(
        observation["python"], ("executable", "implementation", "version"),
        "audit runtime Python observation",
    )
    require_exact_fields(
        observation["platform"], ("system", "release", "machine"),
        "audit runtime platform observation",
    )
    require_exact_fields(observation["libc"], ("name", "version"), "audit runtime libc observation")
    strings = [
        observation["python"]["executable"], observation["python"]["implementation"],
        observation["python"]["version"], observation["platform"]["system"],
        observation["platform"]["release"], observation["platform"]["machine"],
        observation["libc"]["name"], observation["libc"]["version"],
        observation["numpy_version"], observation["python_lmdb_version"],
        observation["liblmdb_version"],
    ]
    if any(not isinstance(item, str) or not item for item in strings):
        raise RuntimeError("audit runtime observation contains a missing version/platform field")
    if observation["byteorder"] != "little":
        raise RuntimeError("the v2 payload wire contract requires a little-endian audit runtime")


def report_canonical_payload_sha256(report):
    if not isinstance(report, dict):
        raise RuntimeError("audit report must be an object")
    projection = {key: value for key, value in report.items() if key != REPORT_PAYLOAD_HASH_FIELD}
    return sha256_json(projection)


def validate_audit_report_document(report):
    if not isinstance(report, dict) or report.get("schema_version") != AUDIT_REPORT_SCHEMA:
        raise RuntimeError("v3 audit report schema mismatch")
    require_exact_fields(report, AUDIT_REPORT_FIELDS, "v3 audit report")
    if not (
        report["audit_status"] == "pass"
        and report["audit_class"]
        == "independent_full_envelope_and_deterministic_sampled_payload_v3"
        and isinstance(report["release_id"], str) and report["release_id"]
    ):
        raise RuntimeError("v3 audit report identity/status mismatch")
    for field in (
        "release_manifest_sha256", "audit_contract_sha256",
        "production_contract_sha256", "payload_contract_sha256",
    ):
        require_sha256(report[field], "audit report {}".format(field))
    require_sha256(report.get(REPORT_PAYLOAD_HASH_FIELD), "audit report canonical payload hash")
    if report[REPORT_PAYLOAD_HASH_FIELD] != report_canonical_payload_sha256(report):
        raise RuntimeError("audit report canonical payload hash mismatch")
    script = report.get("auditor_script")
    require_exact_fields(script, ("file_name", "bytes", "sha256"), "audit report script observation")
    if not isinstance(script["bytes"], int) or isinstance(script["bytes"], bool) or script["bytes"] < 1:
        raise RuntimeError("audit report script byte count is invalid")
    require_sha256(script["sha256"], "audit report script SHA-256")
    runtime = report.get("audit_runtime_observation")
    validate_audit_runtime_observation(runtime)
    require_sha256(report.get("audit_runtime_observation_sha256"), "audit runtime observation hash")
    if report["audit_runtime_observation_sha256"] != sha256_json(runtime):
        raise RuntimeError("audit runtime observation hash mismatch")
    plan = report["semantic_review_plan"]
    require_exact_fields(plan, ("relative_path", "bytes", "sha256"), "semantic review plan artifact")
    if not (
        plan["relative_path"] == "semantic_review_plan.jsonl"
        and isinstance(plan["bytes"], int) and not isinstance(plan["bytes"], bool)
        and plan["bytes"] > 0
    ):
        raise RuntimeError("semantic review plan artifact path/size is invalid")
    require_sha256(plan["sha256"], "semantic review plan artifact SHA-256")


def sha256_ordinal_range(start, end):
    if not is_int(start) or not is_int(end) or start < 0 or end < start:
        raise RuntimeError("invalid ordinal range")
    digest = hashlib.sha256()
    digest.update(b"[")
    for ordinal in range(start, end):
        if ordinal != start:
            digest.update(b",")
        digest.update(str(ordinal).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def storage_key(ordinal):
    if not is_int(ordinal) or ordinal < 0 or ordinal >= 1_000_000_000:
        raise RuntimeError("ordinal cannot be represented as a storage key")
    return "{:09d}".format(ordinal)


def member_id(ordinal):
    return "{}:{}".format(IDENTITY_NAMESPACE, ordinal)


def validate_contracts(audit_contract, production_contract, payload_contract):
    if audit_contract.get("schema_version") != AUDIT_CONTRACT_SCHEMA:
        raise RuntimeError("independent audit contract schema mismatch")
    provenance = audit_contract.get("execution_provenance")
    report_fields = provenance.get("report_top_level_fields") if isinstance(provenance, dict) else None
    if not isinstance(provenance, dict) or not (
        provenance.get("auditor_script_fields") == ["file_name", "bytes", "sha256"]
        and provenance.get("runtime_observation_schema") == AUDIT_RUNTIME_SCHEMA
        and provenance.get("runtime_version_policy")
        == "observe_and_content_address_without_cross_environment_equality"
        and provenance.get("report_payload_hash_field") == REPORT_PAYLOAD_HASH_FIELD
        and provenance.get("report_payload_hash_projection")
        == "canonical JSON of the complete report object after removing only report_canonical_payload_sha256"
        and provenance.get("exclusive_create_required") is True
        and isinstance(report_fields, list)
        and len(report_fields) == len(AUDIT_REPORT_FIELDS)
        and set(report_fields) == set(AUDIT_REPORT_FIELDS)
    ):
        raise RuntimeError("independent audit execution-provenance contract mismatch")
    selection = audit_contract.get("admitted_sample_selection")
    if not isinstance(selection, dict) or not (
        selection.get("algorithm") == "lowest_blake2b256_per_shard_ordinal_band_v1"
        and selection.get("digest_size_bytes") == 32
        and selection.get("ordinal_bands_per_shard") == 4
        and is_int(selection.get("samples_per_nonempty_stratum"))
        and 1 <= selection["samples_per_nonempty_stratum"] <= 16
        and isinstance(selection.get("seed"), str)
        and selection["seed"]
    ):
        raise RuntimeError("independent audit sample-selection contract mismatch")
    if audit_contract.get("reject_selection") != "all_rejects_without_exception":
        raise RuntimeError("independent audit contract must schedule every reject")
    if production_contract.get("schema_version") != PRODUCTION_CONTRACT_SCHEMA:
        raise RuntimeError("production contract schema mismatch")
    source = production_contract.get("source")
    logical = production_contract.get("logical_record")
    if not isinstance(source, dict) or source.get("required_record_count") != 3_378_606:
        raise RuntimeError("production contract source count mismatch")
    if not isinstance(logical, dict) or not (
        logical.get("schema_version") == PRODUCTION_RECORD_SCHEMA
        and logical.get("mode") == PRODUCTION_MODE
        and logical.get("p1_training_admission") is False
    ):
        raise RuntimeError("production contract logical-record boundary mismatch")
    binding = logical.get("motif_lexeme_binding")
    if binding != {
        "digest_algorithm": "sha256",
        "digest_input": "exact_motif_fragment_utf8_without_normalization",
        "ordered_digest_field": "topology.motif_lexeme_sha256",
        "ordered_digest_field_required": True,
        "per_record_fragment_storage_permitted": False,
    }:
        raise RuntimeError("production contract motif-lexeme binding mismatch")
    shard_artifacts = production_contract.get("shard_artifacts")
    if not isinstance(shard_artifacts, dict) or not (
        shard_artifacts.get("motif_census_row_fields") == [
            "motif_lexeme_sha256", "motif_fragment", "count"
        ]
        and shard_artifacts.get("motif_digest_collision_policy") == "fail_closed"
    ):
        raise RuntimeError("production contract motif-census boundary mismatch")
    if payload_contract.get("schema_version") != PAYLOAD_CONTRACT_SCHEMA:
        raise RuntimeError("payload contract schema mismatch")
    if not (
        payload_contract.get("payload_schema_version") == PAYLOAD_SCHEMA
        and payload_contract.get("magic_ascii") == MAGIC.decode("ascii")
        and set(payload_contract.get("header_required_fields", [])) == set(HEADER_FIELDS)
        and payload_contract.get("array_block_required_fields") == [
            "index", "dtype", "shape", "order", "offset", "nbytes", "sha256"
        ]
        and set(payload_contract.get("allowed_dtypes", [])) == set(WIRE_DTYPES)
    ):
        raise RuntimeError("payload contract framing/field boundary mismatch")
    framing = payload_contract.get("framing")
    if not isinstance(framing, dict) or not (
        framing.get("max_header_bytes") == MAX_HEADER_BYTES
        and framing.get("max_payload_bytes") == MAX_PAYLOAD_BYTES
    ):
        raise RuntimeError("payload contract safety bounds mismatch")


def _validate_shape(shape, label):
    if not isinstance(shape, list) or len(shape) > 8:
        raise RuntimeError("{} shape is malformed".format(label))
    result = []
    for dimension in shape:
        if not is_int(dimension) or dimension < 0:
            raise RuntimeError("{} shape contains an invalid dimension".format(label))
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


def decode_payload(np, payload):
    """Decode v2 wire bytes using only the public format rules."""
    if sys.byteorder != "little":
        raise RuntimeError("the v2 native-array contract is only audited on a little-endian runtime")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise RuntimeError("payload must be bytes-like")
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
    if header["payload_schema_version"] != PAYLOAD_SCHEMA:
        raise RuntimeError("payload schema mismatch")
    require_sha256(header["logical_record_sha256"], "payload logical hash")
    if not isinstance(header["record"], dict):
        raise RuntimeError("payload record projection must be an object")
    blocks = header["array_blocks"]
    if not isinstance(blocks, list) or len(blocks) > MAX_ARRAY_BLOCKS:
        raise RuntimeError("payload block list is malformed")
    raw_blocks = payload[prefix + header_size:]
    arrays = {}
    expected_offset = 0
    for expected_index, block in enumerate(blocks):
        require_exact_fields(block, BLOCK_FIELDS, "array block {}".format(expected_index))
        if block["index"] != expected_index:
            raise RuntimeError("array block indices are not contiguous")
        dtype_name = block["dtype"]
        if dtype_name not in WIRE_DTYPES or block["order"] != "C":
            raise RuntimeError("array block dtype/order is unsupported")
        shape = _validate_shape(block["shape"], "array block {}".format(expected_index))
        offset = block["offset"]
        nbytes = block["nbytes"]
        if not is_int(offset) or offset != expected_offset or not is_int(nbytes) or nbytes < 0:
            raise RuntimeError("array block offset/length is invalid")
        item_count = 1
        for dimension in shape:
            item_count *= dimension
            if item_count > (1 << 62):
                raise RuntimeError("array block shape exceeds the safety bound")
        expected_nbytes = item_count * np.dtype(WIRE_DTYPES[dtype_name]).itemsize
        if nbytes != expected_nbytes or offset + nbytes > len(raw_blocks):
            raise RuntimeError("array block length disagrees with dtype/shape")
        block_raw = raw_blocks[offset:offset + nbytes]
        require_sha256(block["sha256"], "array block hash")
        if sha256_bytes(block_raw) != block["sha256"]:
            raise RuntimeError("array block SHA-256 mismatch")
        wire = np.frombuffer(block_raw, dtype=np.dtype(WIRE_DTYPES[dtype_name]))
        array = np.ascontiguousarray(wire.reshape(tuple(shape), order="C").astype(np.dtype(dtype_name), copy=True))
        arrays[expected_index] = array
        expected_offset += nbytes
    if expected_offset != len(raw_blocks):
        raise RuntimeError("payload contains trailing or unreferenced bytes")
    consumed = set()
    record = _rehydrate_arrays(header["record"], arrays, consumed)
    if consumed != set(arrays):
        raise RuntimeError("payload contains an unreferenced array block")
    logical_hash = sha256_json(logical_projection(np, record))
    if logical_hash != header["logical_record_sha256"]:
        raise RuntimeError("payload logical record hash mismatch")
    return record, logical_hash


def _forbid_fields(value, path="record"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_RECORD_FIELDS:
                raise RuntimeError("{} contains forbidden field {}".format(path, key))
            _forbid_fields(item, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _forbid_fields(item, "{}[{}]".format(path, index))


def validate_sampled_record(np, record, membership, shard_manifest, global_motif_lexemes):
    _forbid_fields(record)
    require_exact_fields(
        record,
        (
            "record_schema_version", "sidecar", "member", "identity",
            "atom_universe", "topology", "geometry", "array_metadata",
        ),
        "sampled record",
    )
    if record["record_schema_version"] != PRODUCTION_RECORD_SCHEMA:
        raise RuntimeError("sampled record schema mismatch")
    sidecar = record["sidecar"]
    required_sidecar = (
        "sidecar_id", "sidecar_mode", "selected_ordinal_set_sha256",
        "source_contract_sha256", "identity_normalization_contract_sha256",
        "adapter_harness_sha256", "record_schema_sha256",
        "geometry_only_pretokenizer", "p1_training_admission",
        "p1_training_launcher_permitted",
    )
    require_exact_fields(sidecar, required_sidecar, "record.sidecar")
    if not (
        sidecar["sidecar_id"] == membership["sidecar_id"]
        and sidecar["sidecar_mode"] == PRODUCTION_MODE
        and sidecar["selected_ordinal_set_sha256"] == membership["selected_ordinal_set_sha256"]
        and sidecar["geometry_only_pretokenizer"] is True
        and sidecar["p1_training_admission"] is False
        and sidecar["p1_training_launcher_permitted"] is False
    ):
        raise RuntimeError("sampled record sidecar binding/status mismatch")
    for key in (
        "selected_ordinal_set_sha256", "source_contract_sha256",
        "identity_normalization_contract_sha256", "adapter_harness_sha256",
        "record_schema_sha256",
    ):
        require_sha256(sidecar[key], "record.sidecar.{}".format(key))
    member = record["member"]
    require_exact_fields(
        member,
        (
            "identity_namespace", "member_id", "sdf_record_index",
            "official_csv_row_index", "storage_key", "source_archive_sha256",
            "source_address_sha256", "source_mol_identity_sha256",
        ),
        "record.member",
    )
    ordinal = membership["sdf_record_index"]
    if not (
        member["identity_namespace"] == IDENTITY_NAMESPACE
        and member["member_id"] == membership["member_id"]
        and member["sdf_record_index"] == ordinal
        and member["official_csv_row_index"] == membership["official_csv_row_index"]
        and member["storage_key"] == membership["record_storage_key"]
        and member["source_address_sha256"] == membership["source_address_sha256"]
    ):
        raise RuntimeError("sampled record member binding mismatch")
    for key in ("source_archive_sha256", "source_address_sha256", "source_mol_identity_sha256"):
        require_sha256(member[key], "record.member.{}".format(key))
    identity = record["identity"]
    require_exact_fields(
        identity,
        (
            "official_identity_status", "sdf_strict_smiles_sha256",
            "official_strict_smiles_sha256", "canonical_connectivity_sha256",
            "identity_spec_sha256", "rdkit_version",
        ),
        "record.identity",
    )
    if not (
        identity["official_identity_status"] == "strict_isomeric_match"
        and identity["sdf_strict_smiles_sha256"] == identity["official_strict_smiles_sha256"]
        and isinstance(identity["rdkit_version"], str) and identity["rdkit_version"]
    ):
        raise RuntimeError("sampled record identity invariant failed")
    for key in (
        "sdf_strict_smiles_sha256", "official_strict_smiles_sha256",
        "canonical_connectivity_sha256", "identity_spec_sha256",
    ):
        require_sha256(identity[key], "record.identity.{}".format(key))
    atoms = record["atom_universe"]
    require_exact_fields(
        atoms,
        (
            "policy_id", "hydrogen_projection_spec_sha256", "source_atom_count",
            "source_explicit_hydrogen_count", "model_atom_count",
            "model_to_source_atom_index", "geometry_mol_identity_sha256",
        ),
        "record.atom_universe",
    )
    model_count = atoms["model_atom_count"]
    if not (
        is_int(model_count) and model_count >= 1
        and is_int(atoms["source_atom_count"]) and atoms["source_atom_count"] >= 1
        and is_int(atoms["source_explicit_hydrogen_count"])
        and atoms["source_explicit_hydrogen_count"] >= 0
    ):
        raise RuntimeError("sampled record atom counts are invalid")
    if atoms["policy_id"] != "project_explicit_hydrogens_before_e3fp_v1":
        raise RuntimeError("sampled record hydrogen policy mismatch")
    require_sha256(atoms["hydrogen_projection_spec_sha256"], "hydrogen projection hash")
    require_sha256(atoms["geometry_mol_identity_sha256"], "geometry molecule identity hash")
    mapping = atoms["model_to_source_atom_index"]
    if not (
        isinstance(mapping, np.ndarray) and mapping.dtype == np.int32
        and mapping.shape == (model_count,) and mapping.flags.c_contiguous
        and bool(np.all(mapping >= 0)) and bool(np.all(mapping < atoms["source_atom_count"]))
        and bool(np.all(mapping[:-1] < mapping[1:]))
    ):
        raise RuntimeError("sampled record source-atom mapping invariant failed")
    topology = record["topology"]
    require_exact_fields(
        topology,
        (
            "linearizer_spec_sha256", "motif_count", "motif_atom_indices",
            "motif_atom_indices_sha256", "motif_lexeme_sha256",
        ),
        "record.topology",
    )
    require_sha256(topology["linearizer_spec_sha256"], "linearizer spec hash")
    groups = topology["motif_atom_indices"]
    if not is_int(topology["motif_count"]) or topology["motif_count"] < 1 or len(groups) != topology["motif_count"]:
        raise RuntimeError("sampled record motif count mismatch")
    motif_digests = topology["motif_lexeme_sha256"]
    if not isinstance(motif_digests, list) or len(motif_digests) != topology["motif_count"]:
        raise RuntimeError("sampled record motif digest sequence cardinality mismatch")
    for index, digest in enumerate(motif_digests):
        require_sha256(digest, "sampled motif digest at index {}".format(index))
        if digest not in global_motif_lexemes:
            raise RuntimeError("sampled motif digest is absent from the global motif dictionary")
    seen = np.zeros((model_count,), dtype=np.int8)
    for group in groups:
        if not (
            isinstance(group, np.ndarray) and group.dtype == np.int32
            and group.ndim == 1 and group.size > 0 and group.flags.c_contiguous
            and bool(np.all(group >= 0)) and bool(np.all(group < model_count))
            and bool(np.all(group[:-1] < group[1:]))
        ):
            raise RuntimeError("sampled record motif group invariant failed")
        seen[group] += 1
    if not bool(np.all(seen == 1)):
        raise RuntimeError("sampled record motifs are not an exact atom partition")
    motif_hash = sha256_json([_array_descriptor(group) for group in groups])
    if topology["motif_atom_indices_sha256"] != motif_hash:
        raise RuntimeError("sampled record motif hash mismatch")
    geometry = record["geometry"]
    require_exact_fields(
        geometry,
        (
            "geometry_valid", "geometry_mse_enabled",
            "geometry_mse_candidate_after_tokenizer_binding", "motif_geometry_valid",
            "coordinates", "coordinates_sha256", "e3fp", "e3fp_shape",
            "e3fp_params_sha256", "e3fp_sha256",
        ),
        "record.geometry",
    )
    coordinates = geometry["coordinates"]
    e3fp = geometry["e3fp"]
    motif_valid = geometry["motif_geometry_valid"]
    if not (
        isinstance(coordinates, np.ndarray) and coordinates.dtype == np.float32
        and coordinates.shape == (model_count, 3) and coordinates.flags.c_contiguous
        and bool(np.all(np.isfinite(coordinates)))
    ):
        raise RuntimeError("sampled record coordinate invariant failed")
    if not (
        isinstance(e3fp, np.ndarray) and e3fp.dtype == np.int32
        and e3fp.shape == (model_count, 4) and e3fp.flags.c_contiguous
        and bool(np.all(e3fp >= -1)) and bool(np.all(e3fp <= 4095))
        and not bool(np.any(np.all(e3fp == -1, axis=1)))
        and not bool(np.any(e3fp[:, 0] == -1))
    ):
        raise RuntimeError("sampled record E3FP structural invariant failed")
    if not (
        isinstance(motif_valid, np.ndarray) and motif_valid.dtype == np.bool_
        and motif_valid.shape == (len(groups),) and motif_valid.flags.c_contiguous
        and bool(np.all(motif_valid))
    ):
        raise RuntimeError("sampled record motif-validity invariant failed")
    if not (
        geometry["geometry_valid"] is True
        and geometry["geometry_mse_enabled"] is False
        and geometry["geometry_mse_candidate_after_tokenizer_binding"] is True
        and geometry["coordinates_sha256"] == _array_descriptor(coordinates)["sha256"]
        and geometry["e3fp_sha256"] == _array_descriptor(e3fp)["sha256"]
        and geometry["e3fp_shape"] == [model_count, 4]
    ):
        raise RuntimeError("sampled record geometry hash/status invariant failed")
    require_sha256(geometry["e3fp_params_sha256"], "E3FP parameter hash")
    if geometry["e3fp_params_sha256"] not in shard_manifest["e3fp_params_sha256_values"]:
        raise RuntimeError("sampled record E3FP parameter hash is absent from shard manifest")
    metadata = record["array_metadata"]
    expected_metadata = {
        "coordinates_dtype": "float32", "coordinates_shape": [model_count, 3],
        "coordinates_order": "C", "e3fp_dtype": "int32",
        "e3fp_shape": [model_count, 4], "e3fp_order": "C",
        "model_to_source_atom_index_dtype": "int32", "motif_atom_indices_dtype": "int32",
    }
    if metadata != expected_metadata:
        raise RuntimeError("sampled record array metadata mismatch")


def _next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def _validate_membership(row, release_id, selected_hash, ordinal):
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
        if not (
            row["record_storage_key"] == storage_key(ordinal)
            and row["reject_reason_code"] is None
        ):
            raise RuntimeError("admitted membership conditional fields are invalid")
        require_sha256(row["record_content_sha256"], "membership content hash")
    elif row["disposition"] == "reject":
        if row["record_storage_key"] is not None or row["record_content_sha256"] is not None:
            raise RuntimeError("rejected membership contains admitted-record fields")
        if not isinstance(row["reject_reason_code"], str):
            raise RuntimeError("rejected membership reason is invalid")
    else:
        raise RuntimeError("membership disposition is not closed")


def _validate_payload_index(row, membership):
    require_exact_fields(row, PAYLOAD_INDEX_FIELDS, "payload-index row")
    if not (
        row["payload_index_schema_version"] == PAYLOAD_INDEX_SCHEMA
        and row["record_storage_key"] == membership["record_storage_key"]
        and row["record_content_sha256"] == membership["record_content_sha256"]
        and is_int(row["record_wire_bytes"]) and 0 < row["record_wire_bytes"] <= MAX_PAYLOAD_BYTES
    ):
        raise RuntimeError("payload-index/membership binding mismatch")
    require_sha256(row["record_wire_sha256"], "payload-index wire hash")
    require_sha256(row["record_content_sha256"], "payload-index content hash")


def _validate_reject(row, membership, reason_to_stage, diagnostic_codes):
    require_exact_fields(row, REJECT_FIELDS, "reject-ledger row")
    common = (
        "record_schema_version", "sidecar_id", "sidecar_mode",
        "selected_ordinal_set_sha256", "member_id", "sdf_record_index",
        "official_csv_row_index", "source_address_sha256",
    )
    if any(row[key] != membership[key] for key in common):
        raise RuntimeError("reject-ledger/membership binding mismatch")
    reason = row["reason_code"]
    if not (
        membership["reject_reason_code"] == reason
        and reason_to_stage.get(reason) == row["stage"]
        and row["action"] == "exclude_from_geometry_release"
        and row["geometry_mse_enabled"] is False
        and row["diagnostic_code"] in diagnostic_codes
    ):
        raise RuntimeError("reject-ledger closed reason/stage/action mismatch")
    require_sha256(row["source_mol_identity_sha256"], "reject source identity", nullable=True)
    require_sha256(row["geometry_mol_identity_sha256"], "reject geometry identity", nullable=True)
    if row["source_mol_identity_sha256"] is None and not (
        reason == "SDF_PARSE_FAILED" and row["diagnostic_code"] == "sdf_rdkit_none"
    ):
        raise RuntimeError("only an RDKit-null SDF parse reject may lack source identity")
    expected_detail = sha256_json(
        {
            "diagnostic_code": row["diagnostic_code"],
            "reason_code": reason,
            "source_address_sha256": row["source_address_sha256"],
            "stage": row["stage"],
        }
    )
    if row["detail_sha256"] != expected_detail:
        raise RuntimeError("reject-ledger detail hash mismatch")


def _selector_digest(selection, release_id, shard_index, band, storage_key_value):
    material = canonical_json_bytes(
        {
            "seed": selection["seed"], "release_id": release_id,
            "shard_index": shard_index, "ordinal_band": band,
            "record_storage_key": storage_key_value,
        }
    )
    return hashlib.blake2b(material, digest_size=selection["digest_size_bytes"]).hexdigest()


def _offer_sample(sample_heaps, selection, release_id, shard_index, start, end, membership, payload_index):
    width = end - start
    ordinal = membership["sdf_record_index"]
    bands = selection["ordinal_bands_per_shard"]
    band = min(bands - 1, ((ordinal - start) * bands) // width)
    stratum = "shard-{:06d}:ordinal-band-{}".format(shard_index, band)
    digest = _selector_digest(selection, release_id, shard_index, band, membership["record_storage_key"])
    candidate = {
        "shard_index": shard_index,
        "sdf_record_index": ordinal,
        "record_storage_key": membership["record_storage_key"],
        "stratum": stratum,
        "selector_blake2b256": digest,
        "membership": membership,
        "payload_index": payload_index,
    }
    # Each bucket is contract-bounded to at most 16 entries.  Keeping the
    # smallest explicit (digest, key) ranks avoids any hidden RNG state and
    # implements the published tie-break rule exactly.
    bucket = sample_heaps.setdefault(stratum, [])
    limit = selection["samples_per_nonempty_stratum"]
    rank = (digest, membership["record_storage_key"])
    if len(bucket) < limit:
        bucket.append(candidate)
    else:
        worst_index = max(
            range(len(bucket)),
            key=lambda index: (bucket[index]["selector_blake2b256"], bucket[index]["record_storage_key"]),
        )
        worst_rank = (
            bucket[worst_index]["selector_blake2b256"],
            bucket[worst_index]["record_storage_key"],
        )
        if rank < worst_rank:
            bucket[worst_index] = candidate


def _iter_lmdb_keys(transaction):
    cursor = transaction.cursor()
    return iter(cursor.iternext(keys=True, values=False))


def _validate_motif_census(path, expected_unique, expected_occurrences):
    counts = Counter()
    lexemes = {}
    previous = None
    for row in iter_canonical_jsonl(path, "shard motif census"):
        require_exact_fields(
            row,
            ("motif_lexeme_sha256", "motif_fragment", "count"),
            "motif-census row",
        )
        digest = row["motif_lexeme_sha256"]
        fragment = row["motif_fragment"]
        count = row["count"]
        register_motif_binding(counts, lexemes, digest, fragment, count)
        if previous is not None and digest <= previous:
            raise RuntimeError("motif census is duplicated or not sorted")
        previous = digest
    if len(counts) != expected_unique or sum(counts.values()) != expected_occurrences:
        raise RuntimeError("motif-census counts disagree with shard manifest")
    return counts, lexemes


def _open_real_lmdb(path):
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("the independent audit requires the read-only lmdb package") from exc
    return lmdb.open(
        str(path), subdir=True, readonly=True, lock=False, readahead=False,
        max_readers=32, create=False,
    )


def _artifact_check(shard_dir, manifest):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_PATHS):
        raise RuntimeError("shard artifact set mismatch")
    for role, expected_relative in ARTIFACT_PATHS.items():
        observed = artifacts[role]
        if not isinstance(observed, dict) or set(observed) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("shard artifact observation fields are invalid")
        if observed["relative_path"] != expected_relative:
            raise RuntimeError("shard artifact path mismatch")
        path = regular_file(shard_dir / Path(expected_relative), "shard artifact")
        if observed["bytes"] != path.stat().st_size:
            raise RuntimeError("shard artifact byte count mismatch")
        require_sha256(observed["sha256"], "shard artifact hash")
        if sha256_file(path) != observed["sha256"]:
            raise RuntimeError("shard artifact SHA-256 mismatch")


def _release_metadata(release_root, production_contract_path, payload_contract_path):
    full = release_root / "full_release_manifest.json"
    benchmark = release_root / "benchmark_report.json"
    if full.is_file() == benchmark.is_file():
        raise RuntimeError("release root must contain exactly one complete top-level manifest")
    top_path, top = load_json(full if full.is_file() else benchmark, "release manifest")
    benchmark_mode = benchmark.is_file()
    expected_schema = BENCHMARK_REPORT_SCHEMA if benchmark_mode else FULL_MANIFEST_SCHEMA
    expected_status = "benchmark_non_release" if benchmark_mode else "complete"
    if top.get("schema_version") != expected_schema or top.get("release_status") != expected_status:
        raise RuntimeError("release manifest schema/status mismatch")
    configuration = top.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("release manifest configuration is missing")
    release_id = top.get("release_id")
    if release_id != release_root.name or configuration.get("release_id") != release_id:
        raise RuntimeError("release ID does not match the immutable root name")
    if configuration.get("production_contract_sha256") != sha256_file(production_contract_path):
        raise RuntimeError("release does not bind the supplied production contract")
    harness = configuration.get("harness")
    components = harness.get("components") if isinstance(harness, dict) else None
    if not isinstance(components, dict) or components.get("payload_contract") != sha256_file(payload_contract_path):
        raise RuntimeError("release does not bind the supplied payload contract")
    if harness.get("bundle_sha256") != sha256_json(components):
        raise RuntimeError("release harness bundle hash mismatch")
    selected_count = configuration.get("selected_record_count")
    if not is_int(selected_count) or selected_count < 1:
        raise RuntimeError("release selected count is invalid")
    if configuration.get("selected_ordinal_range") != [0, selected_count]:
        raise RuntimeError("release selected ordinal range is not the contiguous prefix")
    selected_hash = sha256_ordinal_range(0, selected_count)
    if configuration.get("selected_ordinal_set_sha256") != selected_hash:
        raise RuntimeError("release selected ordinal hash mismatch")
    if not benchmark_mode and selected_count != 3_378_606:
        raise RuntimeError("full release does not cover the locked source count")
    if benchmark_mode and selected_count not in (128, 10000):
        raise RuntimeError("benchmark release count is outside the production contract")
    if not (
        configuration.get("source_record_count") == 3_378_606
        and configuration.get("logical_record_schema_version") == PRODUCTION_RECORD_SCHEMA
        and configuration.get("sidecar_mode") == PRODUCTION_MODE
        and configuration.get("release_kind") == (
            "benchmark_non_release" if benchmark_mode else "full_production"
        )
    ):
        raise RuntimeError("release configuration mode/schema/source boundary mismatch")
    scope_path, scope = load_json(release_root / "production_scope.json", "production scope")
    del scope_path
    if scope.get("schema_version") != SCOPE_SCHEMA or scope.get("configuration") != configuration:
        raise RuntimeError("production scope/configuration binding mismatch")
    return top_path, top, configuration, release_id, selected_count, selected_hash, benchmark_mode


def audit_release(release_root, audit_contract_path, production_contract_path,
                  payload_contract_path, output_dir, lmdb_opener=None,
                  runtime_probe=None):
    import numpy as np

    auditor_script = observe_auditor_script()
    runtime_observation = (runtime_probe or observe_audit_runtime)(np)
    # Round-trip through the same closed JSON domain used by the evidence
    # hashes.  A test probe cannot smuggle a non-JSON object into the report.
    runtime_observation = strict_json_bytes(
        canonical_json_bytes(runtime_observation), "audit runtime observation"
    )
    validate_audit_runtime_observation(runtime_observation)
    runtime_observation_sha = sha256_json(runtime_observation)
    release_root = Path(release_root).expanduser().resolve()
    if not release_root.is_dir():
        raise FileNotFoundError("release root is not a directory: {}".format(release_root))
    audit_contract_path, audit_contract = load_json(audit_contract_path, "independent audit contract")
    production_contract_path, production_contract = load_json(production_contract_path, "production contract")
    payload_contract_path, payload_contract = load_json(payload_contract_path, "payload contract")
    validate_contracts(audit_contract, production_contract, payload_contract)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("audit output directory must be new")
    if output_dir == release_root or release_root in output_dir.parents:
        raise RuntimeError("audit output must be outside the immutable release root")
    top_path, top, configuration, release_id, selected_count, selected_hash, benchmark_mode = _release_metadata(
        release_root, production_contract_path, payload_contract_path
    )
    shard_roots = top.get("shards")
    if not isinstance(shard_roots, list) or not shard_roots:
        raise RuntimeError("release manifest shard roots are missing")
    completed_dirs = sorted(
        path.name for path in release_root.iterdir()
        if path.is_dir() and path.name.startswith("shard-") and ".partial-attempt-" not in path.name
    )
    expected_dirs = ["shard-{:06d}".format(index) for index in range(len(shard_roots))]
    if completed_dirs != expected_dirs:
        raise RuntimeError("completed shard directories contain a gap, excess, or malformed name")
    reason_to_stage = audit_contract.get("closed_reject_reason_to_stage")
    diagnostic_codes = set(audit_contract.get("closed_reject_diagnostic_codes", []))
    if not isinstance(reason_to_stage, dict) or not reason_to_stage or not diagnostic_codes:
        raise RuntimeError("audit contract reject vocabulary is absent")
    selection = audit_contract["admitted_sample_selection"]
    open_lmdb = lmdb_opener or _open_real_lmdb
    sample_heaps = {}
    reject_plan = []
    shard_manifests = []
    global_motifs = Counter()
    global_motif_lexemes = {}
    global_e3fp_hashes = set()
    expected_start = 0
    observed_admitted = 0
    observed_rejected = 0
    observed_membership = 0
    observed_payload_index = 0
    for expected_index, root_entry in enumerate(shard_roots):
        require_exact_fields(
            root_entry,
            ("shard_index", "range_start", "range_end", "shard_manifest_sha256"),
            "release shard root",
        )
        if root_entry["shard_index"] != expected_index:
            raise RuntimeError("release shard indices are not contiguous")
        shard_dir = release_root / "shard-{:06d}".format(expected_index)
        manifest_path, manifest = load_json(shard_dir / "shard_manifest.json", "shard manifest")
        if sha256_file(manifest_path) != root_entry["shard_manifest_sha256"]:
            raise RuntimeError("top-level shard-manifest hash mismatch")
        if not (
            manifest.get("schema_version") == SHARD_MANIFEST_SCHEMA
            and manifest.get("release_status") == "complete"
            and manifest.get("release_id") == release_id
            and manifest.get("production_contract_sha256") == configuration["production_contract_sha256"]
            and manifest.get("shard_index") == expected_index
            and manifest.get("range_start") == root_entry["range_start"] == expected_start
            and manifest.get("range_end") == root_entry["range_end"]
        ):
            raise RuntimeError("shard manifest identity/range binding mismatch")
        start = manifest["range_start"]
        end = manifest["range_end"]
        if not is_int(start) or not is_int(end) or end <= start or end > selected_count:
            raise RuntimeError("shard range is invalid")
        selected = end - start
        counts = manifest.get("counts")
        if not isinstance(counts, dict) or not (
            manifest.get("selected_record_count") == selected
            and counts.get("membership_record_count") == selected
            and is_int(counts.get("admitted_record_count"))
            and is_int(counts.get("reject_ledger_record_count"))
            and counts["admitted_record_count"] + counts["reject_ledger_record_count"] == selected
            and counts.get("payload_index_record_count") == counts["admitted_record_count"]
            and manifest.get("partition_invariant_pass") is True
            and manifest.get("lmdb_merged") is False
            and manifest.get("p1_training_admission") is False
        ):
            raise RuntimeError("shard count/partition/status invariant failed")
        _artifact_check(shard_dir, manifest)
        shard_motifs, shard_lexemes = _validate_motif_census(
            shard_dir / "motif_census.jsonl",
            counts.get("unique_motif_count"), counts.get("motif_occurrence_count"),
        )
        for digest, count in shard_motifs.items():
            register_motif_binding(
                global_motifs, global_motif_lexemes,
                digest, shard_lexemes[digest], count,
            )
        e3fp_hashes = manifest.get("e3fp_params_sha256_values")
        if not isinstance(e3fp_hashes, list) or len(e3fp_hashes) > 1 or (
            counts["admitted_record_count"] > 0 and len(e3fp_hashes) != 1
        ):
            raise RuntimeError("shard E3FP parameter hash cardinality is invalid")
        for value in e3fp_hashes:
            require_sha256(value, "shard E3FP parameter hash")
            global_e3fp_hashes.add(value)
        membership_iter = iter_canonical_jsonl(shard_dir / "membership.jsonl", "membership")
        reject_iter = iter(iter_canonical_jsonl(shard_dir / "reject_ledger.jsonl", "reject ledger"))
        index_iter = iter(iter_canonical_jsonl(shard_dir / "payload_index.jsonl", "payload index"))
        next_reject = _next(reject_iter)
        next_index = _next(index_iter)
        env = open_lmdb(shard_dir / "geometry_records.lmdb")
        try:
            with env.begin(write=False) as transaction:
                lmdb_keys = _iter_lmdb_keys(transaction)
                next_key = _next(lmdb_keys)
                local_membership = local_admitted = local_rejected = local_index = 0
                local_wire_bytes = 0
                local_reject_reasons = Counter()
                for ordinal in range(start, end):
                    row = _next(membership_iter)
                    if row is None:
                        raise RuntimeError("membership ended before its shard range")
                    _validate_membership(row, release_id, selected_hash, ordinal)
                    local_membership += 1
                    if row["disposition"] == "admit":
                        if next_index is None:
                            raise RuntimeError("payload index ended before admitted membership")
                        index_row = next_index
                        _validate_payload_index(index_row, row)
                        expected_key = row["record_storage_key"].encode("ascii")
                        if next_key != expected_key:
                            raise RuntimeError("LMDB keys do not equal admitted membership keys")
                        _offer_sample(
                            sample_heaps, selection, release_id, expected_index,
                            start, end, row, index_row,
                        )
                        next_index = _next(index_iter)
                        next_key = _next(lmdb_keys)
                        local_admitted += 1
                        local_index += 1
                        local_wire_bytes += index_row["record_wire_bytes"]
                    else:
                        if next_reject is None:
                            raise RuntimeError("reject ledger ended before rejected membership")
                        _validate_reject(next_reject, row, reason_to_stage, diagnostic_codes)
                        reject_plan.append(
                            {
                                "document_kind": "reject_semantic_review",
                                "shard_index": expected_index,
                                "sdf_record_index": ordinal,
                                "member_id": row["member_id"],
                                "reason_code": next_reject["reason_code"],
                                "stage": next_reject["stage"],
                                "selection_reason": "all_rejects_without_exception",
                                "required_review": "independent_source_and_feature_semantic_recompute",
                            }
                        )
                        next_reject = _next(reject_iter)
                        local_rejected += 1
                        local_reject_reasons[row["reject_reason_code"]] += 1
                if _next(membership_iter) is not None or next_reject is not None or next_index is not None or next_key is not None:
                    raise RuntimeError("membership/reject/index/LMDB closure contains excess rows or keys")
        finally:
            env.close()
        if not (
            local_membership == counts["membership_record_count"]
            and local_admitted == counts["admitted_record_count"]
            and local_rejected == counts["reject_ledger_record_count"]
            and local_index == counts["payload_index_record_count"]
            and local_wire_bytes == counts.get("payload_wire_total_bytes")
            and dict(sorted(local_reject_reasons.items())) == manifest.get("reject_reason_counts")
        ):
            raise RuntimeError("observed shard streams disagree with manifest counts")
        observed_membership += local_membership
        observed_admitted += local_admitted
        observed_rejected += local_rejected
        observed_payload_index += local_index
        shard_manifests.append(manifest)
        expected_start = end
    if expected_start != selected_count:
        raise RuntimeError("shard ranges do not cover the selected range")
    if (observed_admitted > 0 and len(global_e3fp_hashes) != 1) or len(global_e3fp_hashes) > 1:
        raise RuntimeError("E3FP parameter hash drifted across shards")
    top_counts = top.get("counts")
    if not isinstance(top_counts, dict) or not (
        top_counts.get("membership_record_count") == observed_membership == selected_count
        and top_counts.get("admitted_record_count") == observed_admitted
        and top_counts.get("reject_ledger_record_count") == observed_rejected
        and top_counts.get("shard_count") == len(shard_roots)
        and top_counts.get("unique_motif_count") == len(global_motifs)
        and top_counts.get("motif_occurrence_count") == sum(global_motifs.values())
    ):
        raise RuntimeError("top-level counts disagree with audited shard streams")
    global_census = top.get("global_motif_census")
    if not isinstance(global_census, dict) or set(global_census) != {"relative_path", "bytes", "sha256"}:
        raise RuntimeError("global motif-census artifact declaration is malformed")
    if global_census["relative_path"] != "motif_census.jsonl":
        raise RuntimeError("global motif-census relative path mismatch")
    global_path = regular_file(release_root / "motif_census.jsonl", "global motif census")
    if global_path.stat().st_size != global_census["bytes"] or sha256_file(global_path) != global_census["sha256"]:
        raise RuntimeError("global motif-census artifact hash/size mismatch")
    observed_global, observed_global_lexemes = _validate_motif_census(
        global_path, len(global_motifs), sum(global_motifs.values())
    )
    if (
        observed_global != dict(global_motifs)
        or observed_global_lexemes != global_motif_lexemes
    ):
        raise RuntimeError("global motif census does not equal shard aggregation")
    logical_root = sha256_json(
        {
            "configuration": configuration,
            "global_motif_census_sha256": sha256_file(global_path),
            "shards": shard_roots,
            "membership_record_count": observed_membership,
            "admitted_record_count": observed_admitted,
            "reject_ledger_record_count": observed_rejected,
        }
    )
    logical_field = "logical_benchmark_root_sha256" if benchmark_mode else "logical_release_root_sha256"
    if top.get(logical_field) != logical_root:
        raise RuntimeError("top-level logical release root mismatch")
    if not (
        top.get("range_no_gap_no_overlap") is True
        and top.get("lmdb_merged") is False
        and top.get("p1_training_admission") is False
        and top.get("p1_training_launcher_permitted") is False
    ):
        raise RuntimeError("top-level release status flags violate the production boundary")
    admitted_samples = []
    for stratum in sorted(sample_heaps):
        candidates = sample_heaps[stratum]
        admitted_samples.extend(sorted(candidates, key=lambda item: (item["selector_blake2b256"], item["record_storage_key"])))
    audit_contract_sha = sha256_file(audit_contract_path)
    production_contract_sha = sha256_file(production_contract_path)
    payload_contract_sha = sha256_file(payload_contract_path)
    release_manifest_sha = sha256_file(top_path)
    # Freeze the exact executable evidence before pre-registering the sample.
    verify_auditor_script(auditor_script)
    plan_rows = [
        {
            "document_kind": "semantic_review_plan_header",
            "schema_version": SEMANTIC_PLAN_SCHEMA,
            "release_id": release_id,
            "release_manifest_sha256": release_manifest_sha,
            "audit_contract_sha256": audit_contract_sha,
            "production_contract_sha256": production_contract_sha,
            "payload_contract_sha256": payload_contract_sha,
            "auditor_script": auditor_script,
            "audit_runtime_observation": runtime_observation,
            "audit_runtime_observation_sha256": runtime_observation_sha,
            "selection_algorithm": selection["algorithm"],
            "selection_seed": selection["seed"],
            "admitted_sample_count": len(admitted_samples),
            "reject_review_count": len(reject_plan),
            "all_rejects_included": len(reject_plan) == observed_rejected,
            "semantic_recompute_executed_by_this_gate": False,
        }
    ]
    for sample in admitted_samples:
        plan_rows.append(
            {
                "document_kind": "admitted_payload_sample",
                "shard_index": sample["shard_index"],
                "sdf_record_index": sample["sdf_record_index"],
                "record_storage_key": sample["record_storage_key"],
                "stratum": sample["stratum"],
                "selector_blake2b256": sample["selector_blake2b256"],
                "required_review": "wire_hash_decode_and_logical_structure",
            }
        )
    plan_rows.extend(sorted(reject_plan, key=lambda item: item["sdf_record_index"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "semantic_review_plan.jsonl"
    write_plan_new(plan_path, plan_rows)
    observed_plan_rows = list(iter_canonical_jsonl(plan_path, "written semantic review plan"))
    if observed_plan_rows != plan_rows:
        raise RuntimeError("written semantic review plan differs from its pre-registered rows")
    plan_artifact = {
        "relative_path": "semantic_review_plan.jsonl",
        "bytes": int(plan_path.stat().st_size),
        "sha256": sha256_file(plan_path),
    }
    # The selection is now immutable on disk.  Only after pre-registration do
    # we retrieve values, verify their SHA-256 bindings, decode, and inspect.
    samples_by_shard = {}
    for sample in admitted_samples:
        samples_by_shard.setdefault(sample["shard_index"], []).append(sample)
    sampled_wire_bytes = 0
    for shard_index in sorted(samples_by_shard):
        shard_dir = release_root / "shard-{:06d}".format(shard_index)
        env = open_lmdb(shard_dir / "geometry_records.lmdb")
        try:
            with env.begin(write=False) as transaction:
                for sample in sorted(samples_by_shard[shard_index], key=lambda item: item["record_storage_key"]):
                    payload = transaction.get(sample["record_storage_key"].encode("ascii"))
                    if payload is None:
                        raise RuntimeError("pre-registered sampled LMDB value is missing")
                    payload = bytes(payload)
                    index = sample["payload_index"]
                    if len(payload) != index["record_wire_bytes"] or sha256_bytes(payload) != index["record_wire_sha256"]:
                        raise RuntimeError("sampled payload wire hash/length mismatch")
                    record, logical_hash = decode_payload(np, payload)
                    if logical_hash != index["record_content_sha256"]:
                        raise RuntimeError("sampled payload logical hash differs from index/membership")
                    validate_sampled_record(
                        np, record, sample["membership"],
                        shard_manifests[shard_index], global_motif_lexemes,
                    )
                    sampled_wire_bytes += len(payload)
        finally:
            env.close()
    # The script is checked again after every sampled value has been decoded.
    # A mid-run edit therefore cannot acquire a passing v3 report.
    verify_auditor_script(auditor_script)
    if not (
        plan_path.stat().st_size == plan_artifact["bytes"]
        and sha256_file(plan_path) == plan_artifact["sha256"]
    ):
        raise RuntimeError("pre-registered semantic review plan changed during the audit")
    report = {
        "schema_version": AUDIT_REPORT_SCHEMA,
        "audit_status": "pass",
        "audit_class": "independent_full_envelope_and_deterministic_sampled_payload_v3",
        "release_id": release_id,
        "release_manifest_sha256": release_manifest_sha,
        "audit_contract_sha256": audit_contract_sha,
        "production_contract_sha256": production_contract_sha,
        "payload_contract_sha256": payload_contract_sha,
        "auditor_script": auditor_script,
        "audit_runtime_observation": runtime_observation,
        "audit_runtime_observation_sha256": runtime_observation_sha,
        "counts": {
            "selected_record_count": selected_count,
            "membership_record_count": observed_membership,
            "admitted_record_count": observed_admitted,
            "reject_ledger_record_count": observed_rejected,
            "payload_index_record_count": observed_payload_index,
            "lmdb_key_count": observed_admitted,
            "shard_count": len(shard_roots),
            "sampled_admitted_payload_count": len(admitted_samples),
            "sampled_payload_wire_bytes": sampled_wire_bytes,
            "rejects_scheduled_for_semantic_recompute": len(reject_plan),
        },
        "passed_checks": [
            "shard_ranges_no_gap_no_overlap",
            "all_release_shard_and_artifact_hashes",
            "membership_reject_payload_index_partition_closure",
            "all_lmdb_keys_equal_admitted_membership_keys",
            "global_motif_census_equals_shard_aggregation",
            "motif_utf8_content_addresses_and_collision_free_global_dictionary",
            "single_e3fp_parameter_hash_across_admitted_shards",
            "sample_plan_preregistered_before_payload_value_access",
            "sampled_wire_sha256_and_independent_v2_decode",
            "sampled_logical_record_structure_and_hash_binding",
            "sampled_ordered_motif_digests_resolve_in_global_dictionary",
            "all_rejects_scheduled_for_semantic_recompute",
            "auditor_script_bytes_stable_and_content_addressed",
            "audit_runtime_observation_content_addressed_without_cross_machine_equality",
            "report_canonical_payload_self_hash",
        ],
        "semantic_review_plan": plan_artifact,
        "limitations": {
            "independent_rdkit_e3fp_recompute_executed": False,
            "tokenizer_binding_audited": False,
            "downstream_overlap_proof_audited": False,
            "p1_training_admission": False,
        },
    }
    report[REPORT_PAYLOAD_HASH_FIELD] = report_canonical_payload_sha256(report)
    validate_audit_report_document(report)
    report_path = output_dir / "independent_audit_report.json"
    write_json_new(report_path, report)
    _, observed_report = load_json(report_path, "written independent audit report")
    if observed_report != report:
        raise RuntimeError("written audit report differs from the in-memory report")
    validate_audit_report_document(observed_report)
    if sha256_file(plan_path) != plan_artifact["sha256"]:
        raise RuntimeError("semantic review plan changed after report creation")
    verify_auditor_script(auditor_script)
    return observed_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--audit-contract", required=True)
    parser.add_argument("--production-contract", required=True)
    parser.add_argument("--payload-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = audit_release(
        args.release_root,
        args.audit_contract,
        args.production_contract,
        args.payload_contract,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
