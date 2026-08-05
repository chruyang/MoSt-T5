#!/usr/bin/env python3
"""Parallel, fail-closed extraction of the PCQM production-v2 identity set.

Only shard scheduling is parallel.  Each worker validates one complete shard
and writes canonical admitted identity rows into a new scratch directory.  The
parent process revalidates all source observations, loads every worker result
into one SQLite database, and uses the serial v1 writer for the final global
UTF-8/BINARY ordering.  Neither workers nor the parent write to the immutable
release tree.
"""

from __future__ import print_function

import argparse
import json
import multiprocessing as mp
import os
import platform
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from most_t5_next.r1.overlap import extract_pcqm_production_v2_identity_collection_v1 as serial


PARALLEL_CONTRACT_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-parallel-extraction-contract/v1"
PARALLEL_RECEIPT_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-parallel-extraction-receipt/v1"
SCRATCH_MANIFEST_SCHEMA = "most-t5-r1/pcqm-production-v2-identity-parallel-scratch-manifest/v1"
DEFAULT_PROCESSES = 8
START_METHOD = "spawn"
MOLECULE_ROW_FIELDS = frozenset(
    (
        "schema_version", "collection_id", "member_id",
        "connectivity_identity_sha256", "stereo_identity_sha256",
        "conformer_identity_sha256",
    )
)


def _is_within(candidate, parent):
    candidate, parent = Path(candidate).resolve(), Path(parent).resolve()
    try:
        return os.path.commonpath((str(candidate), str(parent))) == str(parent)
    except ValueError:
        return False


def _require_disjoint_paths(release_root, output_dir, scratch_dir):
    if _is_within(output_dir, release_root):
        raise ValueError("output directory must be outside the immutable release root")
    if _is_within(scratch_dir, release_root) or _is_within(release_root, scratch_dir):
        raise ValueError("scratch directory and immutable release root must be disjoint")
    if _is_within(scratch_dir, output_dir) or _is_within(output_dir, scratch_dir):
        raise ValueError("scratch directory and output directory must be disjoint")
    if output_dir.exists():
        raise FileExistsError("output directory already exists: {}".format(output_dir))
    if scratch_dir.exists():
        raise FileExistsError("scratch directory already exists: {}".format(scratch_dir))


def _validate_parallel_contract_documents(extraction_contract, production_contract, payload_contract, identity_contract):
    if extraction_contract.get("schema_version") != PARALLEL_CONTRACT_SCHEMA:
        raise ValueError("parallel identity extraction contract schema mismatch")
    serial_compatible = dict(extraction_contract)
    serial_compatible["schema_version"] = serial.EXTRACTOR_CONTRACT_SCHEMA
    serial.validate_contract_documents(
        serial_compatible, production_contract, payload_contract, identity_contract
    )


def _serialize_observations(observations):
    return [
        {"path": str(path), "bytes": value[0], "sha256": value[1]}
        for path, value in sorted(observations.items(), key=lambda item: item[0].as_posix())
    ]


def _merge_observations(destination, rows):
    for row in rows:
        serial.require_exact_fields(row, frozenset(("path", "bytes", "sha256")), "worker observation")
        path = Path(row["path"]).resolve()
        value = (row["bytes"], row["sha256"])
        if path in destination and destination[path] != value:
            raise RuntimeError("workers disagree about source observation: {}".format(path))
        destination[path] = value


def _extract_shard_worker(job):
    """Validate and extract one shard; arguments/results contain only primitives."""
    try:
        import lmdb
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("parallel extraction requires python-lmdb and NumPy in every worker") from exc

    release_root = Path(job["release_root"]).resolve()
    scratch_dir = Path(job["scratch_dir"]).resolve()
    expected_index = job["expected_index"]
    root_entry = job["root_entry"]
    expected_start = job["expected_start"]
    configuration = job["configuration"]
    config = job["config"]
    contract_hashes = job["contract_hashes"]
    source_observations = {}

    serial.require_exact_fields(root_entry, serial.TOP_SHARD_FIELDS, "release shard root")
    if root_entry["shard_index"] != expected_index:
        raise RuntimeError("release shard indices are not contiguous")
    serial.require_sha256(root_entry["shard_manifest_sha256"], "release shard-manifest SHA-256")
    shard_dir = serial.regular_directory(
        release_root / "shard-{:06d}".format(expected_index), "release shard"
    )
    lmdb_files = serial.validate_shard_envelope(shard_dir)
    if "lock.mdb" in lmdb_files:
        serial.record_observation(
            source_observations,
            shard_dir / "geometry_records.lmdb" / "lock.mdb",
            "LMDB runtime lock file",
        )
    shard_manifest_path = shard_dir / "shard_manifest.json"
    manifest_observed = serial.record_observation(
        source_observations, shard_manifest_path, "shard manifest"
    )
    if manifest_observed["sha256"] != root_entry["shard_manifest_sha256"]:
        raise RuntimeError("top-level shard-manifest SHA-256 mismatch")
    manifest = serial.load_json(shard_manifest_path, "shard manifest")
    serial.require_exact_fields(manifest, serial.SHARD_FIELDS, "shard manifest")
    if not (
        manifest["schema_version"] == serial.SHARD_MANIFEST_SCHEMA
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
    if not serial.is_int(start) or not serial.is_int(end) or end <= start or end > configuration["selected_record_count"]:
        raise RuntimeError("shard range is invalid")
    selected = end - start
    counts = manifest["counts"]
    serial.require_exact_fields(counts, serial.SHARD_COUNT_FIELDS, "shard counts")
    if any(not serial.is_int(counts[key]) or counts[key] < 0 for key in serial.SHARD_COUNT_FIELDS):
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
    if not isinstance(artifacts, dict) or set(artifacts) != set(serial.ARTIFACT_PATHS):
        raise RuntimeError("shard artifact declaration set mismatch")
    paths, artifact_receipt = {}, {}
    for role in sorted(serial.ARTIFACT_PATHS):
        paths[role], artifact_receipt[role] = serial.verify_artifact(
            shard_dir, role, artifacts[role], source_observations
        )

    membership_iter = iter(serial.iter_canonical_jsonl(paths["membership"], "membership"))
    reject_iter = iter(serial.iter_canonical_jsonl(paths["reject_ledger"], "reject ledger"))
    index_iter = iter(serial.iter_canonical_jsonl(paths["payload_index"], "payload index"))
    env = lmdb.open(
        str(shard_dir / "geometry_records.lmdb"), subdir=True, readonly=True,
        lock=False, readahead=False, meminit=False, max_readers=8, create=False,
    )
    local = Counter()
    local_reasons = Counter()
    local_wire_bytes = 0
    row_count = row_bytes = 0
    row_digest = serial.hashlib.sha256()
    key_digest = serial.hashlib.sha256()
    row_path = scratch_dir / "shard-{:06d}.molecule_identity_rows.jsonl".format(expected_index)
    try:
        with open(str(row_path), "xb") as row_handle:
            with env.begin(write=False) as transaction:
                db_iter = iter(transaction.cursor())
                for ordinal in range(start, end):
                    membership = serial._next(membership_iter)
                    if membership is None:
                        raise RuntimeError("membership ended before shard range")
                    serial.validate_membership(
                        membership, configuration["release_id"],
                        configuration["selected_ordinal_set_sha256"], ordinal,
                    )
                    local["membership"] += 1
                    if membership["disposition"] == "reject":
                        reject = serial._next(reject_iter)
                        if reject is None:
                            raise RuntimeError("reject ledger ended before rejected membership")
                        serial.validate_reject(reject, membership)
                        local["rejected"] += 1
                        local_reasons[reject["reason_code"]] += 1
                        continue
                    index_row = serial._next(index_iter)
                    db_item = serial._next(db_iter)
                    if index_row is None or db_item is None:
                        raise RuntimeError("payload index or LMDB ended before admitted membership")
                    raw_key, raw_payload = db_item
                    raw_key, raw_payload = bytes(raw_key), bytes(raw_payload)
                    if raw_key.startswith(b"__"):
                        raise RuntimeError("undeclared LMDB metadata key is forbidden")
                    expected_key = membership["record_storage_key"].encode("ascii")
                    if raw_key != expected_key:
                        raise RuntimeError("LMDB keys do not equal admitted membership keys")
                    serial.validate_payload_index(index_row, membership, raw_payload)
                    record, logical_hash = serial.decode_payload_independently(np, raw_payload)
                    if logical_hash != membership["record_content_sha256"] or logical_hash != index_row["record_content_sha256"]:
                        raise RuntimeError("LMDB record logical hash differs from membership/payload index")
                    connectivity, stereo = serial.validate_record_identity(
                        record, membership, configuration, config, contract_hashes
                    )
                    molecule_row = {
                        "schema_version": serial.MOLECULE_ROW_SCHEMA,
                        "collection_id": config["collection"]["collection_id"],
                        "member_id": membership["member_id"],
                        "connectivity_identity_sha256": connectivity,
                        "stereo_identity_sha256": stereo,
                        "conformer_identity_sha256": None,
                    }
                    raw_row = serial.canonical_json_bytes(molecule_row) + b"\n"
                    row_handle.write(raw_row)
                    row_digest.update(raw_row)
                    key_digest.update(membership["member_id"].encode("utf-8") + b"\n")
                    row_bytes += len(raw_row)
                    row_count += 1
                    local["admitted"] += 1
                    local["payload_index"] += 1
                    local["lmdb_keys"] += 1
                    local["decoded_payloads"] += 1
                    local_wire_bytes += len(raw_payload)
                if any(serial._next(iterator) is not None for iterator in (membership_iter, reject_iter, index_iter, db_iter)):
                    raise RuntimeError("shard streams contain excess rows, keys, or metadata")
            row_handle.flush()
            os.fsync(row_handle.fileno())
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
        and row_count == counts["admitted_record_count"]
    ):
        raise RuntimeError("observed shard streams disagree with manifest counts")
    if serial.sha256_file(row_path) != (row_bytes, row_digest.hexdigest()):
        raise RuntimeError("worker scratch row artifact changed before return")
    serial.verify_observations_unchanged(source_observations)
    return {
        "shard_index": expected_index,
        "range_start": start,
        "range_end": end,
        "shard_manifest_sha256": manifest_observed["sha256"],
        "artifacts": artifact_receipt,
        "counts": dict(sorted(local.items())),
        "reject_reason_counts": dict(sorted(local_reasons.items())),
        "scratch_rows": {
            "path": str(row_path), "bytes": row_bytes,
            "sha256": row_digest.hexdigest(), "row_count": row_count,
            "key_lf_sha256": key_digest.hexdigest(),
        },
        "source_observations": _serialize_observations(source_observations),
        "lmdb_files": sorted(lmdb_files),
        "runtime": {
            "numpy": np.__version__,
            "lmdb": getattr(lmdb, "__version__", "unknown"),
            "python": sys.version.split()[0],
        },
    }


def _run_shard_jobs(jobs, processes):
    executor = ProcessPoolExecutor(
        max_workers=processes, mp_context=mp.get_context(START_METHOD)
    )
    futures = {executor.submit(_extract_shard_worker, job): job["expected_index"] for job in jobs}
    results = []
    try:
        for future in as_completed(futures):
            shard_index = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError("parallel shard worker {} failed".format(shard_index)) from exc
    finally:
        # ``cancel_futures`` was added after the locked remote Python 3.8
        # runtime. Pending jobs were already cancelled above; a plain waiting
        # shutdown preserves the same fail-closed boundary on every supported
        # runtime.
        executor.shutdown(wait=True)
    return results


def _validate_and_insert_scratch_rows(connection, result, config):
    declaration = result["scratch_rows"]
    row_path = serial.regular_file(declaration["path"], "worker scratch rows").resolve()
    expected = (declaration["bytes"], declaration["sha256"])
    if serial.sha256_file(row_path) != expected:
        raise RuntimeError("worker scratch rows bytes/SHA-256 mismatch")
    observed_count = 0
    for row in serial.iter_canonical_jsonl(row_path, "worker scratch rows"):
        serial.require_exact_fields(row, MOLECULE_ROW_FIELDS, "worker molecule identity row")
        if not (
            row["schema_version"] == serial.MOLECULE_ROW_SCHEMA
            and row["collection_id"] == config["collection"]["collection_id"]
            and row["conformer_identity_sha256"] is None
        ):
            raise RuntimeError("worker molecule identity row boundary mismatch")
        serial.require_string(row["member_id"], "worker molecule member ID")
        if not row["member_id"].startswith(serial.IDENTITY_NAMESPACE + ":"):
            raise RuntimeError("worker molecule identity namespace mismatch")
        serial.require_sha256(row["connectivity_identity_sha256"], "worker connectivity identity")
        serial.require_sha256(row["stereo_identity_sha256"], "worker stereo identity")
        try:
            connection.execute(
                "INSERT INTO rows VALUES (?,?)",
                (row["member_id"], sqlite3.Binary(serial.canonical_json_bytes(row))),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("duplicate admitted member ID") from exc
        observed_count += 1
    if observed_count != declaration["row_count"]:
        raise RuntimeError("worker scratch row count mismatch")
    if serial.sha256_file(row_path) != expected:
        raise RuntimeError("worker scratch rows changed during parent ingestion")
    connection.commit()


def extract_collection_parallel(
    extraction_contract_path, config_path, release_root, production_contract_path,
    payload_contract_path, identity_contract_path, output_dir, scratch_dir,
    processes=DEFAULT_PROCESSES,
):
    if not serial.is_int(processes) or processes <= 0:
        raise ValueError("processes must be a positive integer")
    extraction_contract_path = serial.regular_file(extraction_contract_path, "parallel extraction contract").resolve()
    config_path = serial.regular_file(config_path, "parallel extraction config").resolve()
    production_contract_path = serial.regular_file(production_contract_path, "production contract").resolve()
    payload_contract_path = serial.regular_file(payload_contract_path, "payload contract").resolve()
    identity_contract_path = serial.regular_file(identity_contract_path, "identity normalization contract").resolve()
    release_root = serial.regular_directory(Path(release_root).expanduser().resolve(), "release root")
    output_dir = Path(output_dir).expanduser().resolve()
    scratch_dir = Path(scratch_dir).expanduser().resolve()
    _require_disjoint_paths(release_root, output_dir, scratch_dir)

    extractor_path = Path(__file__).resolve()
    serial_core_path = Path(serial.__file__).resolve()
    fixed_paths = {
        "extraction_contract": extraction_contract_path,
        "config": config_path,
        "production_contract": production_contract_path,
        "payload_contract": payload_contract_path,
        "identity_contract": identity_contract_path,
        "extractor": extractor_path,
        "serial_validation_core": serial_core_path,
    }
    fixed_observations = {name: serial.sha256_file(path) for name, path in fixed_paths.items()}
    extraction_contract = serial.load_json(extraction_contract_path, "parallel extraction contract")
    config = serial.load_json(config_path, "parallel extraction config")
    production_contract = serial.load_json(production_contract_path, "production contract")
    payload_contract = serial.load_json(payload_contract_path, "payload contract")
    identity_contract = serial.load_json(identity_contract_path, "identity contract")
    _validate_parallel_contract_documents(
        extraction_contract, production_contract, payload_contract, identity_contract
    )
    serial.validate_config(config, fixed_observations["extraction_contract"][1])
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
    top_path = serial.regular_file(release_root / "full_release_manifest.json", "full release manifest")
    top_observed = serial.record_observation(source_observations, top_path, "full release manifest")
    if (top_observed["bytes"], top_observed["sha256"]) != (
        release_lock["expected_release_manifest_bytes"], release_lock["expected_release_manifest_sha256"]
    ):
        raise RuntimeError("full release manifest differs from the config bytes/SHA-256 lock")
    top = serial.load_json(top_path, "full release manifest")
    configuration, top_counts = serial.validate_release_manifest(top, config, contract_hashes)
    serial.validate_release_envelope(release_root, configuration["shard_count"])
    initial_optional_presence = {
        name: (release_root / name).exists() for name in ("production_scope.json", "run_state.json")
    }
    for optional in ("production_scope.json", "run_state.json"):
        candidate = release_root / optional
        if candidate.exists():
            serial.record_observation(source_observations, candidate, "optional release control file")

    global_decl = top["global_motif_census"]
    serial.require_exact_fields(global_decl, serial.ARTIFACT_FIELDS, "global motif census")
    if global_decl["relative_path"] != "motif_census.jsonl":
        raise ValueError("global motif census path mismatch")
    serial.require_sha256(global_decl["sha256"], "global motif census SHA-256")
    global_path = release_root / "motif_census.jsonl"
    global_observed = serial.record_observation(source_observations, global_path, "global motif census")
    if (global_observed["bytes"], global_observed["sha256"]) != (global_decl["bytes"], global_decl["sha256"]):
        raise RuntimeError("global motif census bytes/SHA-256 mismatch")

    jobs = []
    expected_start = 0
    for expected_index, root_entry in enumerate(top["shards"]):
        serial.require_exact_fields(root_entry, serial.TOP_SHARD_FIELDS, "release shard root")
        if root_entry["shard_index"] != expected_index:
            raise RuntimeError("release shard indices are not contiguous")
        start, end = root_entry["range_start"], root_entry["range_end"]
        if not serial.is_int(start) or not serial.is_int(end) or start != expected_start or end <= start:
            raise RuntimeError("release shard ranges are incomplete, duplicated, gapped, or overlapping")
        jobs.append(
            {
                "release_root": str(release_root), "scratch_dir": str(scratch_dir),
                "expected_index": expected_index, "root_entry": root_entry,
                "expected_start": expected_start, "configuration": configuration,
                "config": config, "contract_hashes": contract_hashes,
            }
        )
        expected_start = end
    if expected_start != configuration["selected_record_count"] or len(jobs) != configuration["shard_count"]:
        raise RuntimeError("release shard ranges do not exactly cover the selected range")

    scratch_dir.mkdir(parents=True, exist_ok=False)
    results = _run_shard_jobs(jobs, min(processes, len(jobs)))
    indices = [result.get("shard_index") for result in results]
    if len(results) != len(jobs) or len(set(indices)) != len(indices) or set(indices) != set(range(len(jobs))):
        raise RuntimeError("parallel worker result set is incomplete or duplicated")
    results.sort(key=lambda item: item["shard_index"])
    expected_start = 0
    runtime_values = {"numpy": set(), "lmdb": set(), "python": set()}
    for result in results:
        if result["range_start"] != expected_start or result["range_end"] <= result["range_start"]:
            raise RuntimeError("parallel worker result ranges are incomplete, duplicated, gapped, or overlapping")
        expected_start = result["range_end"]
        _merge_observations(source_observations, result["source_observations"])
        for key in runtime_values:
            runtime_values[key].add(result["runtime"][key])
    if expected_start != configuration["selected_record_count"]:
        raise RuntimeError("parallel worker results do not cover the selected range")
    if any(len(values) != 1 for values in runtime_values.values()):
        raise RuntimeError("parallel workers disagree about runtime versions")

    sort_path = scratch_dir / ".pcqm_identity_sort.sqlite3"
    connection = serial.create_sort_database(sort_path)
    try:
        for result in results:
            _validate_and_insert_scratch_rows(connection, result, config)
        observed_counts = Counter()
        reject_reasons = Counter()
        shard_receipts = []
        for result in results:
            observed_counts.update(result["counts"])
            reject_reasons.update(result["reject_reason_counts"])
            shard_receipts.append(
                {
                    "shard_index": result["shard_index"],
                    "range_start": result["range_start"],
                    "range_end": result["range_end"],
                    "shard_manifest_sha256": result["shard_manifest_sha256"],
                    "artifacts": result["artifacts"],
                    "counts": result["counts"],
                    "reject_reason_counts": result["reject_reason_counts"],
                }
            )
        if not (
            observed_counts["membership"] == top_counts["membership_record_count"]
            and observed_counts["admitted"] == top_counts["admitted_record_count"]
            and observed_counts["rejected"] == top_counts["reject_ledger_record_count"]
            and observed_counts["decoded_payloads"] == top_counts["admitted_record_count"]
        ):
            raise RuntimeError("observed global streams disagree with release counts")
        if observed_counts["admitted"] <= 0:
            raise RuntimeError("a proof-compatible identity collection cannot be empty")
        logical_root = serial.sha256_json(
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
        serial.validate_release_envelope(release_root, configuration["shard_count"])
        if {
            name: (release_root / name).exists() for name in ("production_scope.json", "run_state.json")
        } != initial_optional_presence:
            raise RuntimeError("optional release control-file presence changed during extraction")
        for result in results:
            shard_dir = release_root / "shard-{:06d}".format(result["shard_index"])
            if sorted(serial.validate_shard_envelope(shard_dir)) != result["lmdb_files"]:
                raise RuntimeError("shard LMDB runtime-file envelope changed during extraction")
        serial.verify_observations_unchanged(source_observations)
        for name, path in fixed_paths.items():
            if serial.sha256_file(path) != fixed_observations[name]:
                raise RuntimeError("{} bytes changed during extraction".format(name))

        output_dir.mkdir(parents=True, exist_ok=False)
        molecule_path = output_dir / "molecule_identity_rows.jsonl"
        molecule_artifact = serial.write_molecule_rows(connection, molecule_path)
        if molecule_artifact["row_count"] != observed_counts["admitted"]:
            raise RuntimeError("emitted molecule row count differs from admitted count")
        connection.commit()
        connection.close()
        connection = None

        scratch_manifest = {
            "schema_version": SCRATCH_MANIFEST_SCHEMA,
            "extraction_id": config["extraction_id"],
            "parallel_contract_sha256": fixed_observations["extraction_contract"][1],
            "release_manifest_sha256": top_observed["sha256"],
            "processes_requested": processes,
            "worker_processes_used": min(processes, len(jobs)),
            "multiprocessing_start_method": START_METHOD,
            "core_molecule_rows": molecule_artifact,
            "sort_database": serial.artifact_of(sort_path),
            "shard_rows": [
                {
                    "shard_index": result["shard_index"],
                    "path": Path(result["scratch_rows"]["path"]).name,
                    "bytes": result["scratch_rows"]["bytes"],
                    "sha256": result["scratch_rows"]["sha256"],
                    "row_count": result["scratch_rows"]["row_count"],
                    "key_lf_sha256": result["scratch_rows"]["key_lf_sha256"],
                }
                for result in results
            ],
        }
        scratch_manifest_path = scratch_dir / "scratch_manifest.json"
        serial.write_json_new(scratch_manifest_path, scratch_manifest)

        source_lock = {
            "schema_version": serial.SOURCE_LOCK_SCHEMA,
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
            "source_files": serial.relative_observations(release_root, source_observations),
            "shards": shard_receipts,
            "excluded_membership_dispositions": ["reject"],
            "excluded_lmdb_metadata_keys": [],
            "permitted_non_evidence_lmdb_runtime_files": ["lock.mdb"],
            "source_open_mode": "readonly_lock_false_create_false",
        }
        source_lock_path = output_dir / "source_lock.json"
        serial.write_json_new(source_lock_path, source_lock)
        resolved_config_path = output_dir / "resolved_config.json"
        serial.write_json_new(resolved_config_path, config)
        collection = config["collection"]
        collection_manifest = {
            "schema_version": serial.COLLECTION_SCHEMA,
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
                "source_identity_namespace": serial.IDENTITY_NAMESPACE,
                "source_release_manifest_sha256": top_observed["sha256"],
                "extractor_sha256": fixed_observations["extractor"][1],
                "excluded_source_metadata_keys": [],
            },
        }
        collection_manifest_path = output_dir / "collection_manifest.json"
        serial.write_json_new(collection_manifest_path, collection_manifest)
        scratch_manifest_observed = serial.sha256_file(scratch_manifest_path)
        receipt = {
            "schema_version": PARALLEL_RECEIPT_SCHEMA,
            "status": "pass",
            "generated_at_utc": serial.utc_now(),
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
                "source_lock": serial.artifact_of(source_lock_path),
                "resolved_config": serial.artifact_of(resolved_config_path),
                "molecule_rows": molecule_artifact,
                "collection_manifest": serial.artifact_of(collection_manifest_path),
                "scratch_manifest": {
                    "path": str(scratch_manifest_path),
                    "bytes": scratch_manifest_observed[0],
                    "sha256": scratch_manifest_observed[1],
                },
            },
            "parallel_execution": {
                "processes_requested": processes,
                "worker_processes_used": min(processes, len(jobs)),
                "multiprocessing_start_method": START_METHOD,
                "scratch_directory": str(scratch_dir),
                "serial_validation_core_sha256": fixed_observations["serial_validation_core"][1],
                "core_determinism_boundary": "molecule_identity_rows.jsonl bytes, SHA-256, row_count, and key_lf_sha256",
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
                "python": next(iter(runtime_values["python"])),
                "numpy": next(iter(runtime_values["numpy"])),
                "lmdb": next(iter(runtime_values["lmdb"])),
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
                "complete_unique_contiguous_parallel_shard_result_set",
                "parent_revalidation_and_global_binary_sort",
                "proof_gate_compatible_hash_only_collection",
            ],
            "policy_boundary": "No P1/P2 overlap policy, downstream split, tokenizer, or training-admission decision is made.",
        }
        receipt["receipt_canonical_payload_sha256"] = serial.sha256_json(receipt)
        serial.write_json_new(output_dir / "extraction_receipt.json", receipt)
        return receipt
    finally:
        if connection is not None:
            connection.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-contract", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--production-contract", required=True)
    parser.add_argument("--payload-contract", required=True)
    parser.add_argument("--identity-normalization-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    receipt = extract_collection_parallel(
        args.extraction_contract, args.config, args.release_root,
        args.production_contract, args.payload_contract,
        args.identity_normalization_contract, args.output_dir,
        args.scratch_dir, args.processes,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
