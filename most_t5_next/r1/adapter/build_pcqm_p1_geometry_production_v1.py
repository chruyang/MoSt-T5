#!/usr/bin/env python3
"""Build the full PCQM4Mv2 geometry sidecar as immutable LMDB shards.

This is the version-bumped production runner.  It deliberately does not alter
or widen ``build_pcqm_p1_geometry_sidecar.py``: the old builder remains a
bounded smoke harness.  The production runner reuses its one-record feature
path and validators, captures the already-produced motif fragments for a
content-addressed census, and changes the enclosing record/release identity to
a distinct production schema.  Each admitted record retains only the ordered
SHA-256 addresses of its motif lexemes; raw per-record fragments remain
forbidden.  The global digest-to-lexeme dictionary is therefore sufficient for
later frozen-tokenizer binding without repeating molecular linearization.

The data path is:

    verified tar.gz -> one ordered ForwardSDMolSupplier producer
        -> bounded ProcessPoolExecutor workers -> one ordered shard writer

Each completed shard owns its LMDB, membership, reject ledger, payload index,
motif census, and manifest.  LMDB shards are never merged.  A crash preserves
the partial attempt and resume starts again at the last completed shard
boundary in a new attempt directory.

This remains a geometry-only *pretokenizer* release.  Successful completion is
necessary evidence for P1, but never sets P1 training admission to true.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import copy
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import tarfile
import tempfile
from collections import Counter, deque
from pathlib import Path, PurePosixPath


CONTRACT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-release-contract/v2"
STAGED_RECEIPT_SCHEMA = "most-t5-r1/pcqm4mv2-staging-receipt/v1"
PRODUCTION_RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-pretokenizer-record/v2"
PRODUCTION_MODE = "full_sharded_production"
SCOPE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-scope/v2"
RUN_STATE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-run-state/v2"
SHARD_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-shard/v2"
FULL_MANIFEST_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
FAILURE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-failure/v2"
PAYLOAD_INDEX_SCHEMA = "most-t5-r1/p1-pcqm-geometry-payload-index-row/v2"
BENCHMARK_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-benchmark/v2"
EXPECTED_SOURCE_RECORDS = 3_378_606
DEFAULT_SHARD_SIZE = 25_000
MIN_SHARD_SIZE = 20_000
MAX_SHARD_SIZE = 30_000
MIN_MAP_SIZE_MIB = 512
MAX_MAP_SIZE_MIB = 8192
_HEX64 = frozenset("0123456789abcdef")
_SHARD_NAME = re.compile(r"^shard-([0-9]{6})$")
_WORKER = None


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
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX64:
        raise ValueError("{} must be a lowercase SHA-256".format(label))


def motif_lexeme_sha256(fragment):
    """Content address one exact motif lexeme without normalization."""
    if not isinstance(fragment, str) or not fragment:
        raise ValueError("motif fragment must be non-empty text")
    return sha256_bytes(fragment.encode("utf-8"))


def motif_bindings(fragment_sequence):
    """Return ordered ``(digest, exact lexeme)`` pairs for one record."""
    return tuple((motif_lexeme_sha256(fragment), fragment) for fragment in fragment_sequence)


def _register_motif_binding(counts, lexemes, digest, fragment, count=1):
    """Merge one census row while failing closed on malformed hashes/collisions."""
    require_sha256(digest, "motif lexeme SHA-256")
    if motif_lexeme_sha256(fragment) != digest:
        raise RuntimeError("motif lexeme digest does not match its exact UTF-8 fragment")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise RuntimeError("motif census count must be a positive integer")
    prior = lexemes.get(digest)
    if prior is not None and prior != fragment:
        raise RuntimeError("motif lexeme SHA-256 collision detected")
    lexemes[digest] = fragment
    counts[digest] += count


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


def verify_attested_bundle_files(attestation_path, bundle_root, required_paths):
    """Rehash the live bundle closure named by an otherwise structural attestation."""
    _, report = load_json(attestation_path, "CPU runtime attestation")
    root = Path(bundle_root).resolve()
    lock = report.get("bundle_file_lock")
    rows = lock.get("files") if isinstance(lock, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("CPU runtime attestation bundle lock is absent")
    try:
        locked_root = Path(lock["bundle_root"]).resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("CPU runtime attestation bundle root is invalid") from exc
    if locked_root != root:
        raise RuntimeError("CPU runtime attestation was captured for a different bundle root")
    observations = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("CPU runtime attestation bundle row is malformed")
        relative = row["relative_path"]
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise RuntimeError("CPU runtime attestation bundle path is malformed")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise RuntimeError("CPU runtime attestation bundle path is not canonical")
        if relative in observations:
            raise RuntimeError("CPU runtime attestation bundle path is duplicated")
        cursor = root
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RuntimeError("attested bundle path contains a symlink")
        target = regular_file(cursor, "attested live bundle file")
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("attested bundle file escapes the live bundle root") from exc
        require_sha256(row.get("sha256"), "attested bundle file SHA-256")
        if row.get("bytes") != int(target.stat().st_size) or row["sha256"] != sha256_file(target):
            raise RuntimeError("live bundle file differs from CPU runtime attestation")
        observations[relative] = row
    required_relative = set()
    for path in required_paths:
        target = regular_file(path, "required attested production file")
        try:
            required_relative.add(target.relative_to(root).as_posix())
        except ValueError as exc:
            raise RuntimeError("required production file is outside the bundle root") from exc
    missing = sorted(required_relative - set(observations))
    if missing:
        raise RuntimeError("CPU runtime attestation omits required production files: {}".format(",".join(missing)))
    return observations


def _fsync_directory(path):
    """Persist a directory entry update on the Linux production filesystem."""
    if os.name != "posix":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path, value):
    """Atomically publish one complete JSON file without replacing a target.

    The temporary inode lives beside the target, is fully flushed and fsynced,
    and is then hard-linked to the final name.  ``os.link`` is the portable
    stdlib no-replace primitive needed here: if another process has already
    published the target, it raises ``FileExistsError`` and preserves that
    winner.  At most this function's one known temporary file is cleaned up.
    """
    path = Path(path)
    if os.path.lexists(str(path)):
        raise FileExistsError("refusing to overwrite immutable file: {}".format(path))
    if not path.parent.is_dir():
        raise NotADirectoryError("JSON output parent is not a directory: {}".format(path.parent))

    descriptor = None
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".write-json.{}.".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Atomic exclusive publication: unlike os.replace, this cannot
        # overwrite an immutable target created by a racing process.
        os.link(str(temporary_path), str(path), follow_symlinks=False)
        _fsync_directory(path.parent)
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None and os.path.lexists(str(temporary_path)):
            temporary_path.unlink()


def write_jsonl_line(handle, value):
    handle.write(canonical_json_bytes(value).decode("utf-8"))
    handle.write("\n")


def write_state_atomic(root, value):
    """Replace the one explicitly mutable run-state file atomically."""
    root = Path(root)
    temporary = root / ".run_state.{}.tmp".format(os.getpid())
    if temporary.exists():
        raise FileExistsError("stale run-state temporary file exists: {}".format(temporary))
    with open(str(temporary), "x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(root / "run_state.json"))


def sha256_ordinal_range(start, end):
    """Hash canonical JSON ``[start,...,end-1]`` without allocating the list."""
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError("invalid ordinal range")
    digest = hashlib.sha256()
    digest.update(b"[")
    for ordinal in range(start, end):
        if ordinal != start:
            digest.update(b",")
        digest.update(str(ordinal).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def plan_shards(record_count, shard_size):
    if not isinstance(record_count, int) or record_count < 1:
        raise ValueError("record_count must be positive")
    if not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size must be positive")
    result = []
    start = 0
    index = 0
    while start < record_count:
        end = min(record_count, start + shard_size)
        result.append({"shard_index": index, "range_start": start, "range_end": end})
        start = end
        index += 1
    return result


def validate_contiguous_ranges(manifests, expected_count):
    """Reject missing, duplicated, overlapping, reordered, or excess ranges."""
    expected_start = 0
    for expected_index, manifest in enumerate(manifests):
        if manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA:
            raise RuntimeError("completed shard manifest schema mismatch")
        if manifest.get("release_status") != "complete":
            raise RuntimeError("completed shard manifest is not complete")
        if manifest.get("shard_index") != expected_index:
            raise RuntimeError("completed shard indices are not contiguous")
        start = manifest.get("range_start")
        end = manifest.get("range_end")
        if start != expected_start or not isinstance(end, int) or end <= start:
            raise RuntimeError("completed shard ranges contain a gap, overlap, or invalid interval")
        if manifest.get("selected_record_count") != end - start:
            raise RuntimeError("completed shard selected count differs from its range")
        counts = manifest.get("counts")
        if not isinstance(counts, dict):
            raise RuntimeError("completed shard counts are absent")
        selected = counts.get("membership_record_count")
        admitted = counts.get("admitted_record_count")
        rejected = counts.get("reject_ledger_record_count")
        if selected != end - start or selected != admitted + rejected:
            raise RuntimeError("completed shard membership partition does not balance")
        expected_start = end
    if expected_start > expected_count:
        raise RuntimeError("completed shards extend beyond the selected source range")
    return expected_start


def ordered_bounded_map(function, iterable, workers, max_pending, initializer=None, initargs=()):
    """Yield process results in input order with a hard in-flight task bound."""
    if workers < 1 or max_pending < workers:
        raise ValueError("workers must be positive and max_pending must be >= workers")
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        for item in iterable:
            yield function(item)
        return
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=initializer, initargs=tuple(initargs)
    ) as executor:
        pending = deque()
        for item in iterable:
            pending.append(executor.submit(function, item))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _synthetic_transform(item):
    """Top-level, picklable deterministic worker used only by hermetic tests."""
    value = int(item)
    return {"ordinal": value, "bucket": value % 7, "value": value * value}


def validate_production_contract(contract, shard_size):
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("production release contract schema mismatch")
    source = contract.get("source")
    execution = contract.get("execution")
    logical = contract.get("logical_record")
    if not isinstance(source, dict) or source.get("required_record_count") != EXPECTED_SOURCE_RECORDS:
        raise RuntimeError("production contract does not lock the PCQM train-3D count")
    if source.get("verified_input_receipt_schema") != STAGED_RECEIPT_SCHEMA:
        raise RuntimeError("production contract staged-receipt schema mismatch")
    if not isinstance(execution, dict) or execution.get("lmdb_merge_permitted") is not False:
        raise RuntimeError("production contract must prohibit LMDB merging")
    allowed = execution.get("allowed_shard_size")
    if not isinstance(allowed, dict) or not (
        allowed.get("minimum") == MIN_SHARD_SIZE and allowed.get("maximum") == MAX_SHARD_SIZE
    ):
        raise RuntimeError("production contract shard bounds mismatch")
    if shard_size < MIN_SHARD_SIZE or shard_size > MAX_SHARD_SIZE:
        raise RuntimeError("requested shard size is outside the production contract")
    if not isinstance(logical, dict) or logical.get("schema_version") != PRODUCTION_RECORD_SCHEMA:
        raise RuntimeError("production logical-record schema mismatch")
    if logical.get("mode") != PRODUCTION_MODE:
        raise RuntimeError("production logical-record mode mismatch")
    if logical.get("p1_training_admission") is not False:
        raise RuntimeError("pretokenizer production contract cannot admit P1")
    binding = logical.get("motif_lexeme_binding")
    if not isinstance(binding, dict) or binding != {
        "digest_algorithm": "sha256",
        "digest_input": "exact_motif_fragment_utf8_without_normalization",
        "ordered_digest_field": "topology.motif_lexeme_sha256",
        "ordered_digest_field_required": True,
        "per_record_fragment_storage_permitted": False,
    }:
        raise RuntimeError("production contract motif-lexeme binding is not frozen")
    shard_artifacts = contract.get("shard_artifacts")
    if not isinstance(shard_artifacts, dict) or shard_artifacts.get("motif_census_row_fields") != [
        "motif_lexeme_sha256", "motif_fragment", "count"
    ]:
        raise RuntimeError("production contract motif-census row schema mismatch")
    if shard_artifacts.get("motif_digest_collision_policy") != "fail_closed":
        raise RuntimeError("production contract must fail closed on motif digest collision")


def load_official_prefix(data_csv_path, selected_count):
    """Read the contract-locked contiguous official train prefix once."""
    smiles = [None] * selected_count
    diagnostics = {}
    with gzip.open(str(data_csv_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "idx" not in fields or "smiles" not in fields:
            raise RuntimeError("official data.csv.gz lacks idx/smiles columns")
        for row_index, row in enumerate(reader):
            if row_index >= selected_count:
                break
            try:
                csv_index = int(row.get("idx", ""))
            except (TypeError, ValueError):
                diagnostics[row_index] = "csv_idx_not_integer"
                continue
            if csv_index != row_index:
                diagnostics[row_index] = "csv_idx_row_index_mismatch"
                continue
            value = row.get("smiles")
            if not isinstance(value, str) or not value:
                diagnostics[row_index] = "csv_smiles_missing"
                continue
            smiles[row_index] = value
    for ordinal in range(selected_count):
        if smiles[ordinal] is None and ordinal not in diagnostics:
            diagnostics[ordinal] = "csv_row_unresolved"
    return smiles, diagnostics


def import_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct module spec for {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _CapturingLinearizer(object):
    """Capture the one linearizer result already requested by build_record."""

    def __init__(self, module):
        self.module = module
        self.result = None

    def linearize_mol(self, mol):
        self.result = self.module.linearize_mol(mol)
        return self.result


def _production_record_from_legacy(record):
    record["record_schema_version"] = PRODUCTION_RECORD_SCHEMA
    record["sidecar"]["sidecar_mode"] = PRODUCTION_MODE
    return record


def _production_row_from_legacy(row):
    row["record_schema_version"] = PRODUCTION_RECORD_SCHEMA
    row["sidecar_mode"] = PRODUCTION_MODE
    return row


def _validate_production_record(builder, np, record):
    if record.get("record_schema_version") != PRODUCTION_RECORD_SCHEMA:
        raise ValueError("production record schema mismatch")
    sidecar = record.get("sidecar")
    if not isinstance(sidecar, dict) or sidecar.get("sidecar_mode") != PRODUCTION_MODE:
        raise ValueError("production record mode mismatch")
    if sidecar.get("p1_training_admission") is not False:
        raise ValueError("production pretokenizer record must not admit training")
    topology = record.get("topology", {})
    if "motif_fragment_sequence" in topology:
        raise ValueError("per-record motif fragments must remain transient")
    expected_topology_fields = {
        "linearizer_spec_sha256", "motif_count", "motif_atom_indices",
        "motif_atom_indices_sha256", "motif_lexeme_sha256",
    }
    if not isinstance(topology, dict) or set(topology) != expected_topology_fields:
        raise ValueError("production topology fields differ from the fixed schema")
    motif_digests = topology["motif_lexeme_sha256"]
    if not isinstance(motif_digests, list) or len(motif_digests) != int(topology["motif_count"]):
        raise ValueError("motif lexeme digest sequence cardinality mismatch")
    for index, digest in enumerate(motif_digests):
        require_sha256(digest, "motif lexeme digest at index {}".format(index))
    # Exercise the complete, frozen v2 native-array validator without copying
    # large arrays.  Only the versioned envelope fields and the content-address
    # sequence differ.  The latter is removed only for the legacy validation
    # call and restored before any logical hash or wire encoding is computed.
    record["record_schema_version"] = builder.RECORD_SCHEMA
    sidecar["sidecar_mode"] = builder.SIDE_CAR_MODE
    topology.pop("motif_lexeme_sha256")
    try:
        builder.validate_admitted_record(np, record)
    finally:
        topology["motif_lexeme_sha256"] = motif_digests
        record["record_schema_version"] = PRODUCTION_RECORD_SCHEMA
        sidecar["sidecar_mode"] = PRODUCTION_MODE


def _validate_production_membership(builder, row, release_id, selected_sha, ordinal, csv_row, source_address):
    if row.get("record_schema_version") != PRODUCTION_RECORD_SCHEMA or row.get("sidecar_mode") != PRODUCTION_MODE:
        raise ValueError("production membership envelope mismatch")
    row["record_schema_version"] = builder.RECORD_SCHEMA
    row["sidecar_mode"] = builder.SIDE_CAR_MODE
    try:
        builder.validate_membership_row(row, release_id, selected_sha, ordinal, csv_row, source_address)
    finally:
        row["record_schema_version"] = PRODUCTION_RECORD_SCHEMA
        row["sidecar_mode"] = PRODUCTION_MODE


def _validate_production_reject(builder, row, release_id, selected_sha, ordinal, csv_row, source_address):
    if row.get("record_schema_version") != PRODUCTION_RECORD_SCHEMA or row.get("sidecar_mode") != PRODUCTION_MODE:
        raise ValueError("production reject envelope mismatch")
    row["record_schema_version"] = builder.RECORD_SCHEMA
    row["sidecar_mode"] = builder.SIDE_CAR_MODE
    try:
        builder.validate_reject_row(row, release_id, selected_sha, ordinal, csv_row, source_address)
    finally:
        row["record_schema_version"] = PRODUCTION_RECORD_SCHEMA
        row["sidecar_mode"] = PRODUCTION_MODE


def _init_feature_worker(config):
    global _WORKER
    # Fail closed against BLAS/OpenMP oversubscription on high-core CPU hosts.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import lmdb  # noqa: F401 - confirms the runtime before expensive work.
    import numpy as np
    from rdkit import Chem

    token = str(os.getpid())
    builder = import_module_from_file(config["builder_path"], "r1_prod_builder_" + token)
    preflight = import_module_from_file(config["preflight_path"], "r1_prod_preflight_" + token)
    identity = import_module_from_file(config["identity_path"], "r1_prod_identity_" + token)
    linearizer = import_module_from_file(config["linearizer_path"], "r1_prod_linearizer_" + token)
    codec = import_module_from_file(config["codec_path"], "r1_prod_codec_" + token)
    import_root, package_root, _ = preflight.resolve_e3fp_source(config["e3fp_source"])
    e3fp_api = preflight.import_locked_e3fp(import_root, package_root)
    _WORKER = {
        "config": config,
        "builder": builder,
        "preflight": preflight,
        "identity": identity,
        "linearizer": linearizer,
        "codec": codec,
        "e3fp_api": e3fp_api,
        "np": np,
        "Chem": Chem,
    }


def _feature_worker(item):
    if _WORKER is None:
        raise RuntimeError("feature worker was not initialized")
    ordinal, csv_row, raw_official_smiles, diagnostic, mol_binary = item
    state = _WORKER
    builder = state["builder"]
    np = state["np"]
    Chem = state["Chem"]
    config = state["config"]
    source_mol = None if mol_binary is None else Chem.Mol(mol_binary)
    source_address = builder.source_address_sha256(
        config["archive_sha256"], config["sdf_member"], ordinal, csv_row
    )
    capture = _CapturingLinearizer(state["linearizer"])
    record, reject = builder.build_record(
        Chem,
        np,
        state["preflight"],
        capture,
        state["e3fp_api"],
        state["identity"],
        ordinal,
        csv_row,
        raw_official_smiles,
        source_mol,
        config["sidecar_values"],
        config["archive_sha256"],
        source_address,
        config["identity_contract_sha256"],
        config["projection_spec_sha256"],
        config["linearizer_spec_sha256"],
        official_input_diagnostic=diagnostic,
    )
    if record is not None:
        if capture.result is None:
            raise RuntimeError("admitted record has no captured linearizer result")
        fragments = tuple(capture.result.fragment_sequence)
        if len(fragments) != int(record["topology"]["motif_count"]):
            raise RuntimeError("captured motif fragment/group cardinality mismatch")
        if any(not isinstance(fragment, str) or not fragment for fragment in fragments):
            raise RuntimeError("captured motif fragment is empty or non-text")
        bindings = motif_bindings(fragments)
        _production_record_from_legacy(record)
        record["topology"]["motif_lexeme_sha256"] = [digest for digest, _ in bindings]
        _validate_production_record(builder, np, record)
        payload = state["codec"].encode_record(np, record)
        row = builder.build_membership_row(
            np,
            config["release_id"],
            config["selected_sha256"],
            ordinal,
            csv_row,
            source_address,
            record,
            None,
        )
        _production_row_from_legacy(row)
        _validate_production_membership(
            builder, row, config["release_id"], config["selected_sha256"], ordinal, csv_row, source_address
        )
        payload_index = {
            "payload_index_schema_version": PAYLOAD_INDEX_SCHEMA,
            "record_storage_key": row["record_storage_key"],
            "record_wire_bytes": int(len(payload)),
            "record_wire_sha256": sha256_bytes(payload),
            "record_content_sha256": row["record_content_sha256"],
        }
        builder.validate_payload_index_row(
            payload_index, row["record_storage_key"], row["record_content_sha256"], payload
        )
        return {
            "ordinal": int(ordinal),
            "disposition": "admit",
            "membership": row,
            "reject": None,
            "payload": payload,
            "payload_index": payload_index,
            "motif_bindings": bindings,
            "e3fp_params_sha256": record["geometry"]["e3fp_params_sha256"],
        }
    row = builder.build_membership_row(
        np,
        config["release_id"],
        config["selected_sha256"],
        ordinal,
        csv_row,
        source_address,
        None,
        reject,
    )
    _production_row_from_legacy(row)
    _validate_production_membership(
        builder, row, config["release_id"], config["selected_sha256"], ordinal, csv_row, source_address
    )
    reject_row = builder.build_reject_row(
        config["release_id"], config["selected_sha256"], ordinal, csv_row, reject
    )
    _production_row_from_legacy(reject_row)
    _validate_production_reject(
        builder, reject_row, config["release_id"], config["selected_sha256"], ordinal, csv_row, source_address
    )
    return {
        "ordinal": int(ordinal),
        "disposition": "reject",
        "membership": row,
        "reject": reject_row,
        "payload": None,
        "payload_index": None,
        "motif_bindings": (),
        "e3fp_params_sha256": None,
    }


def _source_items(Chem, builder, archive_path, sdf_member, selected_count, start_ordinal,
                  official_smiles, diagnostics):
    observed_count = 0
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = builder.find_locked_sdf_member(archive, sdf_member)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("unable to open the locked SDF tar member")
        try:
            supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
            for ordinal, source_mol in enumerate(supplier):
                if ordinal >= selected_count:
                    break
                observed_count = ordinal + 1
                if ordinal < start_ordinal:
                    continue
                csv_row = int(ordinal)
                mol_binary = None if source_mol is None else bytes(source_mol.ToBinary())
                yield (
                    int(ordinal),
                    csv_row,
                    official_smiles[ordinal],
                    diagnostics.get(ordinal),
                    mol_binary,
                )
        finally:
            stream.close()
    if observed_count != selected_count:
        raise RuntimeError("SDF ended before the selected ordinal range")


def _next_partial_attempt(root, shard_index):
    for attempt in range(1, 1_000_000):
        path = Path(root) / "shard-{:06d}.partial-attempt-{:06d}".format(shard_index, attempt)
        if not path.exists():
            return path, attempt
    raise RuntimeError("partial-attempt namespace exhausted")


class ShardWriter(object):
    """One parent-process LMDB writer for an immutable contiguous shard."""

    def __init__(self, lmdb, root, release_id, contract_sha256, shard_index, range_start,
                 range_end, map_size_mib):
        self.lmdb = lmdb
        self.root = Path(root)
        self.release_id = release_id
        self.contract_sha256 = contract_sha256
        self.shard_index = int(shard_index)
        self.range_start = int(range_start)
        self.range_end = int(range_end)
        self.expected_ordinal = self.range_start
        self.partial_dir, self.attempt = _next_partial_attempt(self.root, self.shard_index)
        self.final_dir = self.root / "shard-{:06d}".format(self.shard_index)
        if self.final_dir.exists():
            raise FileExistsError("completed shard already exists: {}".format(self.final_dir))
        self.partial_dir.mkdir(parents=False, exist_ok=False)
        self.membership_path = self.partial_dir / "membership.jsonl"
        self.reject_path = self.partial_dir / "reject_ledger.jsonl"
        self.payload_index_path = self.partial_dir / "payload_index.jsonl"
        self.records_path = self.partial_dir / "geometry_records.lmdb"
        self.membership_handle = open(str(self.membership_path), "x", encoding="utf-8")
        self.reject_handle = open(str(self.reject_path), "x", encoding="utf-8")
        self.payload_index_handle = open(str(self.payload_index_path), "x", encoding="utf-8")
        self.env = lmdb.open(
            str(self.records_path),
            subdir=True,
            map_size=int(map_size_mib) * 1024 * 1024,
            readonly=False,
            lock=True,
            sync=True,
            metasync=True,
            map_async=False,
        )
        self.transaction = self.env.begin(write=True)
        self.counts = Counter()
        self.motifs = Counter()
        self.motif_lexemes = {}
        self.e3fp_param_hashes = set()
        self.closed = False

    def append(self, result):
        if self.closed:
            raise RuntimeError("cannot append to a closed shard")
        ordinal = result.get("ordinal")
        if ordinal != self.expected_ordinal or ordinal >= self.range_end:
            raise RuntimeError("worker results are missing, duplicated, or out of order")
        write_jsonl_line(self.membership_handle, result["membership"])
        if result["disposition"] == "admit":
            row = result["membership"]
            payload = result["payload"]
            key = row["record_storage_key"].encode("ascii")
            if not self.transaction.put(key, payload, overwrite=False):
                raise RuntimeError("duplicate LMDB storage key")
            write_jsonl_line(self.payload_index_handle, result["payload_index"])
            for digest, fragment in result["motif_bindings"]:
                _register_motif_binding(self.motifs, self.motif_lexemes, digest, fragment)
            self.e3fp_param_hashes.add(result["e3fp_params_sha256"])
            self.counts["admitted_record_count"] += 1
            self.counts["payload_wire_total_bytes"] += int(len(payload))
            self.counts["motif_occurrence_count"] += len(result["motif_bindings"])
        elif result["disposition"] == "reject":
            write_jsonl_line(self.reject_handle, result["reject"])
            self.counts["reject_ledger_record_count"] += 1
            self.counts["reject_reason:" + result["reject"]["reason_code"]] += 1
        else:
            raise RuntimeError("worker disposition is not closed")
        self.counts["membership_record_count"] += 1
        self.expected_ordinal += 1

    def _close_handles(self, commit):
        if self.closed:
            return
        try:
            if self.transaction is not None:
                if commit:
                    self.transaction.commit()
                else:
                    self.transaction.abort()
                self.transaction = None
            for handle in (self.membership_handle, self.reject_handle, self.payload_index_handle):
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
            if commit:
                # python-lmdb's C extension exposes ``force`` positionally on
                # the locked runtime; keyword dispatch raises TypeError.
                self.env.sync(True)
        finally:
            self.env.close()
            self.closed = True

    def abort(self):
        self._close_handles(False)

    def finalize(self):
        if self.expected_ordinal != self.range_end:
            raise RuntimeError("cannot complete a shard before its full range is written")
        selected = self.range_end - self.range_start
        admitted = int(self.counts["admitted_record_count"])
        rejected = int(self.counts["reject_ledger_record_count"])
        if selected != admitted + rejected or selected != int(self.counts["membership_record_count"]):
            raise RuntimeError("shard partition invariant failed")
        if len(self.e3fp_param_hashes) > 1:
            raise RuntimeError("resolved E3FP parameters drifted within one shard")
        self._close_handles(True)
        motif_path = self.partial_dir / "motif_census.jsonl"
        with open(str(motif_path), "x", encoding="utf-8") as handle:
            for digest in sorted(self.motifs):
                write_jsonl_line(
                    handle,
                    {
                        "motif_lexeme_sha256": digest,
                        "motif_fragment": self.motif_lexemes[digest],
                        "count": int(self.motifs[digest]),
                    },
                )
            handle.flush()
            os.fsync(handle.fileno())
        data_mdb = self.records_path / "data.mdb"
        artifacts = {
            "geometry_records_lmdb_data": _artifact(data_mdb, "geometry_records.lmdb/data.mdb"),
            "membership": _artifact(self.membership_path, "membership.jsonl"),
            "reject_ledger": _artifact(self.reject_path, "reject_ledger.jsonl"),
            "payload_index": _artifact(self.payload_index_path, "payload_index.jsonl"),
            "motif_census": _artifact(motif_path, "motif_census.jsonl"),
        }
        manifest = {
            "schema_version": SHARD_MANIFEST_SCHEMA,
            "created_utc": utc_now(),
            "release_status": "complete",
            "release_id": self.release_id,
            "production_contract_sha256": self.contract_sha256,
            "shard_index": self.shard_index,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "selected_record_count": selected,
            "counts": {
                "membership_record_count": selected,
                "admitted_record_count": admitted,
                "reject_ledger_record_count": rejected,
                "payload_index_record_count": admitted,
                "payload_wire_total_bytes": int(self.counts["payload_wire_total_bytes"]),
                "motif_occurrence_count": int(self.counts["motif_occurrence_count"]),
                "unique_motif_count": int(len(self.motifs)),
            },
            "reject_reason_counts": {
                key.split(":", 1)[1]: int(value)
                for key, value in sorted(self.counts.items())
                if key.startswith("reject_reason:")
            },
            "e3fp_params_sha256_values": sorted(self.e3fp_param_hashes),
            "artifacts": artifacts,
            "partition_invariant_pass": True,
            "lmdb_merged": False,
            "p1_training_admission": False,
        }
        write_json_new(self.partial_dir / "shard_manifest.json", manifest)
        os.rename(str(self.partial_dir), str(self.final_dir))
        return manifest


def _artifact(path, relative_path):
    path = regular_file(path, relative_path)
    return {
        "relative_path": relative_path,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_completed_shard(path, release_id, contract_sha256, rehash=True):
    path = Path(path)
    manifest_path, manifest = load_json(path / "shard_manifest.json", "completed shard manifest")
    del manifest_path
    if manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA or manifest.get("release_status") != "complete":
        raise RuntimeError("completed shard status/schema mismatch")
    if manifest.get("release_id") != release_id or manifest.get("production_contract_sha256") != contract_sha256:
        raise RuntimeError("completed shard release/contract binding mismatch")
    match = _SHARD_NAME.match(path.name)
    if match is None or int(match.group(1)) != manifest.get("shard_index"):
        raise RuntimeError("completed shard directory/index mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "geometry_records_lmdb_data", "membership", "reject_ledger", "payload_index", "motif_census"
    }:
        raise RuntimeError("completed shard artifact set mismatch")
    allowed_paths = {
        "geometry_records.lmdb/data.mdb", "membership.jsonl", "reject_ledger.jsonl",
        "payload_index.jsonl", "motif_census.jsonl",
    }
    observed_paths = set()
    for observation in artifacts.values():
        if not isinstance(observation, dict):
            raise RuntimeError("completed shard artifact observation is malformed")
        relative = observation.get("relative_path")
        if relative not in allowed_paths or relative in observed_paths:
            raise RuntimeError("completed shard artifact relative path is invalid or duplicated")
        observed_paths.add(relative)
        target = regular_file(path / Path(relative), "completed shard artifact")
        if int(target.stat().st_size) != observation.get("bytes"):
            raise RuntimeError("completed shard artifact byte count mismatch")
        require_sha256(observation.get("sha256"), "completed shard artifact SHA-256")
        if rehash and sha256_file(target) != observation["sha256"]:
            raise RuntimeError("completed shard artifact SHA-256 mismatch")
    return manifest


def discover_completed_shards(root, release_id, contract_sha256, rehash=True):
    root = Path(root)
    indexed = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _SHARD_NAME.match(path.name)
        if match is None:
            continue
        indexed.append((int(match.group(1)), path))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(len(indexed))):
        raise RuntimeError("completed shard directory indices contain a gap")
    return [
        validate_completed_shard(path, release_id, contract_sha256, rehash=rehash)
        for _, path in indexed
    ]


def aggregate_motif_census(root, manifests):
    census = Counter()
    lexemes = {}
    for manifest in manifests:
        path = Path(root) / "shard-{:06d}".format(manifest["shard_index"]) / "motif_census.jsonl"
        with open(str(path), "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if set(row) != {"motif_lexeme_sha256", "motif_fragment", "count"}:
                    raise RuntimeError("motif-census row fields are not closed")
                digest = row["motif_lexeme_sha256"]
                fragment = row["motif_fragment"]
                count = row["count"]
                _register_motif_binding(census, lexemes, digest, fragment, count)
    return census, lexemes


def _write_global_census_idempotent(path, census, lexemes):
    expected = b"".join(
        canonical_json_bytes(
            {
                "motif_lexeme_sha256": digest,
                "motif_fragment": lexemes[digest],
                "count": int(census[digest]),
            }
        ) + b"\n"
        for digest in sorted(census)
    )
    path = Path(path)
    if path.exists():
        with open(str(path), "rb") as handle:
            observed = handle.read()
        if observed != expected:
            raise RuntimeError("existing global motif census differs from completed shards")
        return
    with open(str(path), "xb") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())


def _scope_configuration(args, receipt_path, observations, source_contract_sha256,
                         sdf_member, contract_sha256, runtime_attestation_path, harness):
    selected_count = args.benchmark_records or EXPECTED_SOURCE_RECORDS
    return {
        "release_id": Path(args.output_dir).expanduser().name,
        "production_contract_sha256": contract_sha256,
        "runtime_attestation_sha256": sha256_file(runtime_attestation_path),
        "staged_input_receipt_sha256": sha256_file(receipt_path),
        "source_contract_sha256": source_contract_sha256,
        "release_kind": "benchmark_non_release" if args.benchmark_records else "full_production",
        "source_record_count": EXPECTED_SOURCE_RECORDS,
        "selected_record_count": selected_count,
        "selected_ordinal_range": [0, selected_count],
        "selected_ordinal_set_sha256": sha256_ordinal_range(0, selected_count),
        "shard_size": int(args.shard_size),
        "shard_count": len(plan_shards(selected_count, args.shard_size)),
        "staged_inputs": observations,
        "locked_sdf_member": sdf_member,
        "harness": harness,
        "logical_record_schema_version": PRODUCTION_RECORD_SCHEMA,
        "sidecar_mode": PRODUCTION_MODE,
    }


def _load_or_create_scope(root, configuration, resume):
    path = Path(root) / "production_scope.json"
    if path.exists():
        if not resume:
            raise FileExistsError("output root already contains a production scope; use --resume explicitly")
        _, observed = load_json(path, "production scope")
        if observed.get("schema_version") != SCOPE_SCHEMA or observed.get("configuration") != configuration:
            raise RuntimeError("resume configuration differs from the immutable production scope")
        return observed
    scope = {
        "schema_version": SCOPE_SCHEMA,
        "created_utc": utc_now(),
        "release_status": (
            "benchmark_non_release"
            if configuration["release_kind"] == "benchmark_non_release"
            else "partial"
        ),
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "configuration": configuration,
    }
    write_json_new(path, scope)
    return scope


def _run_state(configuration, completed, active=None, status="partial"):
    return {
        "schema_version": RUN_STATE_SCHEMA,
        "updated_utc": utc_now(),
        "release_status": status,
        "release_id": configuration["release_id"],
        "completed_shard_count": len(completed),
        "completed_ordinal_end": 0 if not completed else completed[-1]["range_end"],
        "active_partial_attempt": active,
        "p1_training_admission": False,
    }


def _release_summary(root, configuration, manifests, selected_count):
    if validate_contiguous_ranges(manifests, selected_count) != selected_count:
        raise RuntimeError("full release does not cover the complete source range")
    census, lexemes = aggregate_motif_census(root, manifests)
    global_census_path = Path(root) / "motif_census.jsonl"
    _write_global_census_idempotent(global_census_path, census, lexemes)
    admitted = sum(item["counts"]["admitted_record_count"] for item in manifests)
    rejected = sum(item["counts"]["reject_ledger_record_count"] for item in manifests)
    membership = sum(item["counts"]["membership_record_count"] for item in manifests)
    if membership != selected_count or membership != admitted + rejected:
        raise RuntimeError("global membership partition does not balance")
    shard_roots = []
    for manifest in manifests:
        path = Path(root) / "shard-{:06d}".format(manifest["shard_index"]) / "shard_manifest.json"
        shard_roots.append(
            {
                "shard_index": manifest["shard_index"],
                "range_start": manifest["range_start"],
                "range_end": manifest["range_end"],
                "shard_manifest_sha256": sha256_file(path),
            }
        )
    logical_root = sha256_json(
        {
            "configuration": configuration,
            "global_motif_census_sha256": sha256_file(global_census_path),
            "shards": shard_roots,
            "membership_record_count": membership,
            "admitted_record_count": admitted,
            "reject_ledger_record_count": rejected,
        }
    )
    return census, admitted, rejected, membership, shard_roots, logical_root, global_census_path


def _finalize_full_release(root, configuration, manifests):
    census, admitted, rejected, membership, shard_roots, logical_root, global_census_path = _release_summary(
        root, configuration, manifests, EXPECTED_SOURCE_RECORDS
    )
    final = {
        "schema_version": FULL_MANIFEST_SCHEMA,
        "created_utc": utc_now(),
        "release_status": "complete",
        "release_id": configuration["release_id"],
        "logical_release_root_sha256": logical_root,
        "configuration": configuration,
        "counts": {
            "source_record_count": EXPECTED_SOURCE_RECORDS,
            "membership_record_count": membership,
            "admitted_record_count": admitted,
            "reject_ledger_record_count": rejected,
            "shard_count": len(manifests),
            "unique_motif_count": len(census),
            "motif_occurrence_count": int(sum(census.values())),
        },
        "global_motif_census": _artifact(global_census_path, "motif_census.jsonl"),
        "shards": shard_roots,
        "range_no_gap_no_overlap": True,
        "lmdb_merged": False,
        "tokenizer_binding": "absent_and_forbidden",
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "next_gate": "Independent reference audit, overlap proof, and frozen tokenizer binding remain mandatory before P1 admission.",
    }
    final_path = Path(root) / "full_release_manifest.json"
    if final_path.exists():
        _, observed = load_json(final_path, "full release manifest")
        # Timestamps are intentionally not regenerated into an existing release.
        if observed.get("logical_release_root_sha256") != logical_root or observed.get("release_status") != "complete":
            raise RuntimeError("existing full release manifest differs from completed shards")
        return observed
    write_json_new(final_path, final)
    return final


def _finalize_benchmark(root, configuration, manifests):
    selected_count = int(configuration["selected_record_count"])
    census, admitted, rejected, membership, shard_roots, logical_root, global_census_path = _release_summary(
        root, configuration, manifests, selected_count
    )
    report = {
        "schema_version": BENCHMARK_SCHEMA,
        "created_utc": utc_now(),
        "release_status": "benchmark_non_release",
        "release_id": configuration["release_id"],
        "logical_benchmark_root_sha256": logical_root,
        "configuration": configuration,
        "counts": {
            "source_record_count": EXPECTED_SOURCE_RECORDS,
            "selected_record_count": selected_count,
            "membership_record_count": membership,
            "admitted_record_count": admitted,
            "reject_ledger_record_count": rejected,
            "shard_count": len(manifests),
            "unique_motif_count": len(census),
            "motif_occurrence_count": int(sum(census.values())),
        },
        "global_motif_census": _artifact(global_census_path, "motif_census.jsonl"),
        "shards": shard_roots,
        "range_no_gap_no_overlap": True,
        "lmdb_merged": False,
        "full_release_manifest_permitted": False,
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
    }
    path = Path(root) / "benchmark_report.json"
    if path.exists():
        _, observed = load_json(path, "benchmark report")
        if observed.get("logical_benchmark_root_sha256") != logical_root:
            raise RuntimeError("existing benchmark report differs from completed shards")
        return observed
    if (Path(root) / "full_release_manifest.json").exists():
        raise RuntimeError("benchmark output must never contain a full release manifest")
    write_json_new(path, report)
    return report


def configure_cpu_only_environment():
    required = {
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for name, value in required.items():
        observed = os.environ.get(name)
        if observed not in (None, value):
            raise RuntimeError("{} must be {} for the CPU production runner".format(name, value))
        os.environ[name] = value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-contract", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--staging-receipt", required=True)
    parser.add_argument("--runtime-attestation", required=True)
    parser.add_argument("--runtime-attestation-contract", required=True)
    parser.add_argument("--production-contract", required=True)
    parser.add_argument("--identity-normalization-contract", required=True)
    parser.add_argument("--payload-format-contract", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--shard-map-size-mib", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=0, help="0 selects min(80,cpu_count-8)")
    parser.add_argument("--max-pending", type=int, default=0, help="0 selects workers*3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--benchmark-records",
        type=int,
        choices=(128, 10000),
        default=0,
        help="Run the exact production path on a prefix, but force benchmark_non_release status.",
    )
    parser.add_argument("--skip-completed-payload-rehash", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_size < MIN_SHARD_SIZE or args.shard_size > MAX_SHARD_SIZE:
        parser.error("--shard-size must be in [{},{}]".format(MIN_SHARD_SIZE, MAX_SHARD_SIZE))
    if args.shard_map_size_mib < MIN_MAP_SIZE_MIB or args.shard_map_size_mib > MAX_MAP_SIZE_MIB:
        parser.error("--shard-map-size-mib must be in [{},{}]".format(MIN_MAP_SIZE_MIB, MAX_MAP_SIZE_MIB))
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.max_pending < 0:
        parser.error("--max-pending must be >= 0")
    return args


def run(args):
    configure_cpu_only_environment()
    root = Path(__file__).resolve().parents[1]
    runner_path = Path(__file__).resolve()
    builder_path = root / "adapter" / "build_pcqm_p1_geometry_sidecar.py"
    linearizer_path = root / "adapter" / "mol_linearizer.py"
    codec_path = root / "adapter" / "sidecar_v2_codec.py"
    preflight_path = root / "gates" / "pcqm_e3fp_preflight.py"
    identity_path = root / "gates" / "pcqm_identity_smoke.py"
    staging_adapter_path = root / "adapter" / "pcqm_staging_receipt.py"
    runtime_collector_path = root / "gates" / "capture_cpu_runtime_attestation.py"
    runtime_validator_path = root / "gates" / "validate_cpu_runtime_attestation.py"
    required_code = (
        builder_path, linearizer_path, codec_path, preflight_path, identity_path,
        staging_adapter_path, runtime_collector_path, runtime_validator_path,
    )
    for path in required_code:
        regular_file(path, "production harness component")

    staging_contract_path = regular_file(args.staging_contract, "staging receipt contract")
    source_contract_path, source_contract = load_json(args.source_contract, "source contract")
    receipt_path = regular_file(args.staging_receipt, "staging receipt")
    runtime_attestation_path = regular_file(args.runtime_attestation, "CPU runtime attestation")
    runtime_attestation_contract_path = regular_file(
        args.runtime_attestation_contract, "CPU runtime attestation contract"
    )
    contract_path, contract = load_json(args.production_contract, "production release contract")
    identity_contract_path, identity_contract = load_json(
        args.identity_normalization_contract, "identity normalization contract"
    )
    payload_contract_path, payload_contract = load_json(args.payload_format_contract, "payload contract")
    validate_production_contract(contract, args.shard_size)

    staging_adapter = import_module_from_file(staging_adapter_path, "r1_prod_staging_adapter")
    verified_staging = staging_adapter.verify_staging_receipt(
        staging_contract_path, source_contract_path, receipt_path
    )
    staging_report = verified_staging.report()
    observations = staging_report["artifacts"]
    archive_path = regular_file(
        verified_staging.work_path("train_3d_sdf_archive"), "verified staged train-3D archive"
    )
    data_csv_path = regular_file(
        verified_staging.work_path("companion_data_csv_gz"), "verified staged official data.csv.gz"
    )
    # The validator's script-compatible import expects this exact module name.
    import_module_from_file(runtime_collector_path, "capture_cpu_runtime_attestation")
    runtime_validator = import_module_from_file(runtime_validator_path, "r1_prod_runtime_validator")
    runtime_errors = runtime_validator.validate_attestation(
        runtime_attestation_path, runtime_attestation_contract_path
    )
    if runtime_errors:
        raise RuntimeError("CPU runtime attestation validation failed")
    verify_attested_bundle_files(
        runtime_attestation_path,
        root.parents[1],
        (
            runner_path,
            *required_code,
            staging_contract_path,
            source_contract_path,
            runtime_attestation_contract_path,
            contract_path,
            identity_contract_path,
            payload_contract_path,
        ),
    )

    source = source_contract.get("source")
    companion = source_contract.get("official_companion")
    if not isinstance(source, dict) or source.get("official_train_sdf_records") != EXPECTED_SOURCE_RECORDS:
        raise RuntimeError("source contract does not lock the expected train-3D count")
    sdf_member = source.get("train_sdf_member")
    if not isinstance(sdf_member, dict):
        raise RuntimeError("source contract lacks the locked train SDF member")
    invariants = companion.get("validated_invariants") if isinstance(companion, dict) else None
    if not isinstance(invariants, dict) or not (
        invariants.get("csv_idx_is_zero_based_contiguous") is True
        and invariants.get("train_split_is_contiguous_prefix") is True
        and invariants.get("train_split_records") == EXPECTED_SOURCE_RECORDS
        and invariants.get("train_split_min") == 0
        and invariants.get("train_split_max") == EXPECTED_SOURCE_RECORDS - 1
    ):
        raise RuntimeError("source contract does not prove ordinal == official CSV row for the train prefix")

    builder = import_module_from_file(builder_path, "r1_prod_parent_builder")
    preflight = import_module_from_file(preflight_path, "r1_prod_parent_preflight")
    if identity_contract.get("schema_version") != "most-t5-r1/pcqm4mv2-identity-normalization-contract/v1":
        raise RuntimeError("identity normalization contract schema mismatch")
    builder.validate_payload_format_contract(payload_contract)
    import_root, package_root, e3fp_files = preflight.resolve_e3fp_source(args.e3fp_source)
    del import_root, package_root
    harness_components = {
        "production_runner": sha256_file(runner_path),
        "bounded_record_builder": sha256_file(builder_path),
        "molecule_native_linearizer": sha256_file(linearizer_path),
        "safe_payload_codec": sha256_file(codec_path),
        "e3fp_preflight": sha256_file(preflight_path),
        "identity_gate": sha256_file(identity_path),
        "staging_receipt_adapter": sha256_file(staging_adapter_path),
        "runtime_attestation_collector": sha256_file(runtime_collector_path),
        "runtime_attestation_validator": sha256_file(runtime_validator_path),
        "staging_contract": sha256_file(staging_contract_path),
        "source_contract": sha256_file(source_contract_path),
        "runtime_attestation_contract": sha256_file(runtime_attestation_contract_path),
        "production_contract": sha256_file(contract_path),
        "identity_contract": sha256_file(identity_contract_path),
        "payload_contract": sha256_file(payload_contract_path),
        "e3fp_source_files": {name: sha256_file(path) for name, path in sorted(e3fp_files.items())},
    }
    harness = {
        "components": harness_components,
        "bundle_sha256": sha256_json(harness_components),
    }
    contract_sha256 = sha256_file(contract_path)
    configuration = _scope_configuration(
        args,
        receipt_path,
        observations,
        verified_staging.source_contract_sha256,
        sdf_member,
        contract_sha256,
        runtime_attestation_path,
        harness,
    )
    output_root = Path(args.output_dir).expanduser()
    if output_root.exists() and not args.resume:
        raise FileExistsError("--output-dir must be new unless --resume is explicit")
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    _load_or_create_scope(output_root, configuration, args.resume)

    selected_count = int(args.benchmark_records or EXPECTED_SOURCE_RECORDS)
    final_path = output_root / (
        "benchmark_report.json" if args.benchmark_records else "full_release_manifest.json"
    )
    completed = discover_completed_shards(
        output_root,
        configuration["release_id"],
        contract_sha256,
        rehash=not args.skip_completed_payload_rehash,
    )
    completed_end = validate_contiguous_ranges(completed, selected_count)
    if final_path.exists():
        if completed_end != selected_count:
            raise RuntimeError("complete manifest exists without complete shard coverage")
        final = (
            _finalize_benchmark(output_root, configuration, completed)
            if args.benchmark_records
            else _finalize_full_release(output_root, configuration, completed)
        )
        write_state_atomic(
            output_root,
            _run_state(
                configuration,
                completed,
                status="benchmark_non_release" if args.benchmark_records else "complete",
            ),
        )
        return final
    if completed_end == selected_count:
        final = (
            _finalize_benchmark(output_root, configuration, completed)
            if args.benchmark_records
            else _finalize_full_release(output_root, configuration, completed)
        )
        write_state_atomic(
            output_root,
            _run_state(
                configuration,
                completed,
                status="benchmark_non_release" if args.benchmark_records else "complete",
            ),
        )
        return final

    import lmdb
    import numpy as np
    from rdkit import Chem

    official_smiles, diagnostics = load_official_prefix(data_csv_path, selected_count)
    workers = args.workers or min(80, max(1, int(os.cpu_count() or 1) - 8))
    max_pending = args.max_pending or workers * 3
    if max_pending < workers:
        raise RuntimeError("max pending tasks must be at least the worker count")
    selected_sha = configuration["selected_ordinal_set_sha256"]
    sidecar_values = {
        "sidecar_id": configuration["release_id"],
        "selected_ordinal_set_sha256": selected_sha,
        "source_contract_sha256": verified_staging.source_contract_sha256,
        "adapter_harness_sha256": harness["bundle_sha256"],
        "record_schema_sha256": contract_sha256,
    }
    worker_config = {
        "builder_path": str(builder_path),
        "preflight_path": str(preflight_path),
        "identity_path": str(identity_path),
        "linearizer_path": str(linearizer_path),
        "codec_path": str(codec_path),
        "e3fp_source": str(Path(args.e3fp_source).expanduser().resolve()),
        "archive_sha256": verified_staging.artifact("train_3d_sdf_archive")["sha256"],
        "sdf_member": sdf_member,
        "release_id": configuration["release_id"],
        "selected_sha256": selected_sha,
        "sidecar_values": sidecar_values,
        "identity_contract_sha256": sha256_file(identity_contract_path),
        "projection_spec_sha256": sha256_json(preflight.HYDROGEN_PROJECTION_PROFILE),
        "linearizer_spec_sha256": sha256_file(linearizer_path),
    }
    items = _source_items(
        Chem,
        builder,
        archive_path,
        sdf_member,
        selected_count,
        completed_end,
        official_smiles,
        diagnostics,
    )
    results = ordered_bounded_map(
        _feature_worker,
        items,
        workers,
        max_pending,
        initializer=_init_feature_worker,
        initargs=(worker_config,),
    )
    plans = plan_shards(selected_count, args.shard_size)
    current = None
    try:
        for result in results:
            ordinal = result["ordinal"]
            shard_index = ordinal // args.shard_size
            plan = plans[shard_index]
            if current is None:
                if ordinal != plan["range_start"]:
                    raise RuntimeError("resume did not start at a shard boundary")
                current = ShardWriter(
                    lmdb,
                    output_root,
                    configuration["release_id"],
                    contract_sha256,
                    shard_index,
                    plan["range_start"],
                    plan["range_end"],
                    args.shard_map_size_mib,
                )
                write_state_atomic(
                    output_root,
                    _run_state(
                        configuration,
                        completed,
                        active={
                            "shard_index": shard_index,
                            "range_start": plan["range_start"],
                            "range_end": plan["range_end"],
                            "attempt": current.attempt,
                            "directory": current.partial_dir.name,
                        },
                    ),
                )
            if shard_index != current.shard_index:
                completed.append(current.finalize())
                validate_contiguous_ranges(completed, selected_count)
                write_state_atomic(output_root, _run_state(configuration, completed))
                current = ShardWriter(
                    lmdb,
                    output_root,
                    configuration["release_id"],
                    contract_sha256,
                    shard_index,
                    plan["range_start"],
                    plan["range_end"],
                    args.shard_map_size_mib,
                )
            current.append(result)
        if current is not None:
            completed.append(current.finalize())
            current = None
            validate_contiguous_ranges(completed, selected_count)
            write_state_atomic(output_root, _run_state(configuration, completed))
    except Exception:
        if current is not None:
            current.abort()
        raise
    if completed_end == selected_count and not completed:
        completed = discover_completed_shards(
            output_root, configuration["release_id"], contract_sha256, rehash=False
        )
    final = (
        _finalize_benchmark(output_root, configuration, completed)
        if args.benchmark_records
        else _finalize_full_release(output_root, configuration, completed)
    )
    write_state_atomic(
        output_root,
        _run_state(
            configuration,
            completed,
            status="benchmark_non_release" if args.benchmark_records else "complete",
        ),
    )
    return final


def write_failure(root, exc):
    root = Path(root)
    if not root.is_dir():
        return
    failures = root / "failures"
    failures.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = failures / "failure-{}-{}.json".format(stamp, os.getpid())
    write_json_new(
        path,
        {
            "schema_version": FAILURE_SCHEMA,
            "created_utc": utc_now(),
            "pass": False,
            "exception_type": type(exc).__name__,
            "message_class_only": type(exc).__name__,
        },
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        try:
            write_failure(Path(args.output_dir).expanduser(), exc)
        except Exception:
            pass
        raise
    print(
        json.dumps(
            {
                "pass": True,
                "release_status": report["release_status"],
                "release_id": report["release_id"],
                "source_records": report["counts"]["source_record_count"],
                "admitted": report["counts"]["admitted_record_count"],
                "rejected": report["counts"]["reject_ledger_record_count"],
                "shards": report["counts"]["shard_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
