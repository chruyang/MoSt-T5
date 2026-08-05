#!/usr/bin/env python3
"""Re-read and deterministically replay a bounded PCQM P1 geometry sidecar.

This is deliberately a *builder-linked deterministic replay*: it imports the
production builder and its safe v2 codec, verifies the persisted wire values,
then replays the same bounded source records.  It proves persistence and
same-harness determinism only.  It is not an independent semantic audit and
must never be presented as P1 admission evidence.
"""

from __future__ import print_function

import argparse
import datetime as dt
import importlib.util
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path


MAX_SMOKE_RECORDS = 1000
SIDE_CAR_SCHEMA = "most-t5-r1/p1-pcqm-geometry-smoke/v2"
RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-pretokenizer-record/v2"
SIDE_CAR_MODE = "bounded_smoke_only"
REPLAY_LOCK_SCHEMA = "most-t5-r1/p1-pcqm-geometry-replay-gate-lock/v2"
VALIDATION_REPORT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-validation-report/v2"
RELEASE_ROOT_SCHEMA = "most-t5-r1/p1-pcqm-geometry-sidecar-release-root/v2"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def import_module_from_file(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct import spec for {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def strict_json_object(text, label):
    def no_duplicate_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise RuntimeError("{} contains a duplicate JSON key: {}".format(label, key))
            result[key] = item
        return result

    def reject_nonfinite(token):
        raise RuntimeError("{} contains a non-finite JSON token: {}".format(label, token))

    try:
        value = json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=reject_nonfinite)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("{} is not valid JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("{} must contain a JSON object".format(label))
    return value


def read_json(path, label):
    target = Path(path).expanduser()
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError("{} is not a non-symlink regular file: {}".format(label, target))
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("cannot read {}".format(label)) from exc
    return target.resolve(), strict_json_object(text, label)


def read_jsonl(path, label):
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError("{} is missing or not a regular file: {}".format(label, target))
    rows = []
    with open(str(target), "r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or line == "\n":
                raise RuntimeError("{} has a non-canonical JSONL line at {}".format(label, line_number))
            body = line[:-1]
            row = strict_json_object(body, "{} row {}".format(label, line_number))
            if canonical_json_bytes(row) != body.encode("utf-8"):
                raise RuntimeError("{} row {} is not canonical JSON".format(label, line_number))
            rows.append(row)
    return rows


def require_exact_keys(mapping, expected, label):
    if not isinstance(mapping, dict):
        raise RuntimeError("{} is not an object".format(label))
    observed = set(mapping)
    expected = set(expected)
    if observed != expected:
        raise RuntimeError(
            "{} fields differ (missing={}, extra={})".format(
                label, sorted(expected - observed), sorted(observed - expected)
            )
        )


def validate_replay_gate_lock(lock_path, lock, validator_path):
    """Bind this builder-linked replay implementation to its external lock."""
    if lock.get("schema_version") != REPLAY_LOCK_SCHEMA:
        raise RuntimeError("replay gate lock schema mismatch")
    components = lock.get("components")
    if not isinstance(components, list) or len(components) != 1:
        raise RuntimeError("replay gate lock must contain exactly one component")
    component = components[0]
    if not isinstance(component, dict):
        raise RuntimeError("replay gate lock component is malformed")
    require_exact_keys(component, ("name", "relative_harness_path", "sha256"), "replay gate lock component")
    if component.get("name") != "validate_p1_pcqm_geometry_sidecar.py" or component.get("relative_harness_path") != "gates/validate_p1_pcqm_geometry_sidecar.py":
        raise RuntimeError("replay gate lock component path/name is malformed")
    expected_sha256 = component.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError("replay gate lock SHA-256 is malformed")
    observed_sha256 = builder_sha256(validator_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("replay gate lock SHA mismatch")
    return builder_sha256(lock_path)


def builder_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_scope(scope, builder, codec, source_contract_path, identity_contract_path, record_schema_path,
                   payload_format_contract_path, adapter_lock_path, archive_path, data_csv_path, split_dict_path,
                   verified_input_lock, split_observed, sidecar_dir):
    require_exact_keys(
        scope,
        (
            "schema_version", "created_utc", "release_status", "p1_training_admission",
            "p1_training_launcher_permitted", "tokenizer_binding", "sidecar_id", "selection", "source",
            "contracts", "harness", "limits", "prohibitions",
        ),
        "scope manifest",
    )
    if scope.get("schema_version") != SIDE_CAR_SCHEMA:
        raise RuntimeError("scope manifest schema mismatch")
    if scope.get("release_status") != "non_admissible_pre_tokenizer":
        raise RuntimeError("scope manifest does not state pre-tokenizer non-admission")
    if scope.get("p1_training_admission") is not False or scope.get("p1_training_launcher_permitted") is not False:
        raise RuntimeError("scope manifest incorrectly permits P1")
    if scope.get("tokenizer_binding") != "absent_and_forbidden" or scope.get("sidecar_id") != sidecar_dir.name:
        raise RuntimeError("scope sidecar identity/binding is invalid")
    selection = scope["selection"]
    require_exact_keys(
        selection, ("kind", "selected_ordinals", "selected_record_count", "selected_ordinal_set_sha256"), "scope selection"
    )
    count = selection.get("selected_record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count > MAX_SMOKE_RECORDS:
        raise RuntimeError("scope selected record count is outside the smoke bound")
    if selection.get("kind") != "prefix" or selection.get("selected_ordinals") != "[0,{})".format(count):
        raise RuntimeError("only the exact deterministic prefix smoke is valid")
    expected_selected_sha = builder.sha256_selected_ordinals(list(range(count)))
    if selection.get("selected_ordinal_set_sha256") != expected_selected_sha:
        raise RuntimeError("scope selected ordinal hash mismatch")

    source = scope["source"]
    require_exact_keys(
        source,
        (
            "source_contract_sha256", "source_archive_sha256_observed", "source_archive_bytes_observed",
            "source_record_count", "sdf_tar_member", "source_address_schema_version", "data_csv_sha256",
            "split_dict_sha256", "verified_input_lock", "split_loading",
        ),
        "scope source",
    )
    if source["source_contract_sha256"] != builder.sha256_file(source_contract_path):
        raise RuntimeError("scope source contract hash mismatch")
    if source["data_csv_sha256"] != builder.sha256_file(data_csv_path) or source["split_dict_sha256"] != builder.sha256_file(split_dict_path):
        raise RuntimeError("scope companion artifact hash mismatch")
    if source["source_archive_bytes_observed"] != int(archive_path.stat().st_size):
        raise RuntimeError("scope archive byte observation mismatch")
    if source["source_record_count"] != 3_378_606:
        raise RuntimeError("scope source record count differs from locked PCQM train count")
    if source["source_archive_sha256_observed"] != verified_input_lock["artifacts"]["train_3d_sdf_archive"]["sha256"]:
        raise RuntimeError("scope archive SHA differs from live source lock")
    if source["sdf_tar_member"] != verified_input_lock["train_sdf_member"]:
        raise RuntimeError("scope locked SDF tar member differs from live source lock")
    if source["source_address_schema_version"] != builder.SOURCE_ADDRESS_SCHEMA:
        raise RuntimeError("scope source-address schema differs from the v2 builder")
    if source["verified_input_lock"] != verified_input_lock or source["split_loading"] != split_observed:
        raise RuntimeError("scope source verification observation differs from this replay")

    contracts = scope["contracts"]
    require_exact_keys(
        contracts,
        (
            "identity_normalization_contract_sha256", "record_schema_sha256", "payload_format_contract_sha256",
            "payload_schema_version", "payload_index_schema_version", "adapter_lock_sha256",
            "hydrogen_projection_spec_sha256", "linearizer_spec_sha256",
        ),
        "scope contracts",
    )
    expected_contract_hashes = {
        "identity_normalization_contract_sha256": builder.sha256_file(identity_contract_path),
        "record_schema_sha256": builder.sha256_file(record_schema_path),
        "payload_format_contract_sha256": builder.sha256_file(payload_format_contract_path),
        "adapter_lock_sha256": builder.sha256_file(adapter_lock_path),
    }
    for key, expected in expected_contract_hashes.items():
        if contracts[key] != expected:
            raise RuntimeError("scope {} differs from the live contract/lock".format(key))
    if contracts["payload_schema_version"] != codec.PAYLOAD_SCHEMA or contracts["payload_index_schema_version"] != builder.PAYLOAD_INDEX_SCHEMA:
        raise RuntimeError("scope payload schema binding differs from the v2 codec")

    harness = scope["harness"]
    require_exact_keys(
        harness,
        (
            "builder_sha256", "e3fp_gate_sha256", "identity_gate_sha256", "sidecar_codec_sha256",
            "e3fp_module_version", "e3fp_module_file", "e3fp_source_file_sha256", "rdkit_version",
        ),
        "scope harness",
    )
    if harness["sidecar_codec_sha256"] != builder.sha256_file(Path(codec.__file__).resolve()):
        raise RuntimeError("scope codec SHA differs from the live v2 codec")
    if not isinstance(harness["e3fp_source_file_sha256"], dict) or not harness["e3fp_source_file_sha256"]:
        raise RuntimeError("scope lacks a concrete E3FP source-file observation")

    limits = scope["limits"]
    require_exact_keys(limits, ("map_size_mib", "max_records"), "scope limits")
    if limits["max_records"] != count or not isinstance(limits["map_size_mib"], int) or limits["map_size_mib"] < 64:
        raise RuntimeError("scope limits are invalid")
    prohibitions = scope["prohibitions"]
    require_exact_keys(prohibitions, ("full_mode_available", "sdf_extracted", "local_data_transfer", "raw_smiles_serialized"), "scope prohibitions")
    if any(value is not False for value in prohibitions.values()):
        raise RuntimeError("scope prohibition flags are not all fail-closed")
    return count, expected_selected_sha


def validate_membership_and_ledger(builder, record_schema, membership_rows, reject_rows, sidecar_id, selected_sha,
                                   count, companion_rows, source_info):
    if tuple(record_schema.get("membership_row", {}).get("required_fields", ())) != builder.MEMBERSHIP_ROW_FIELDS:
        raise RuntimeError("record contract membership row differs from the v2 builder")
    if tuple(record_schema.get("reject_ledger_row", {}).get("required_fields", ())) != builder.REJECT_LEDGER_ROW_FIELDS:
        raise RuntimeError("record contract reject ledger row differs from the v2 builder")
    if len(membership_rows) != count or len(companion_rows) != count:
        raise RuntimeError("membership/companion rows do not cover the selected prefix")
    source_member = source_info["verified_input_lock"]["train_sdf_member"]
    membership_by_ordinal = {}
    admitted_storage = {}
    expected_rejected = set()
    for ordinal, row in enumerate(membership_rows):
        csv_row = int(companion_rows[ordinal])
        source_address = builder.source_address_sha256(source_info["archive_sha256"], source_member, ordinal, csv_row)
        try:
            builder.validate_membership_row(row, sidecar_id, selected_sha, ordinal, csv_row, source_address)
        except Exception as exc:
            raise RuntimeError("membership row {} violates the v2 contract".format(ordinal)) from exc
        if ordinal in membership_by_ordinal:
            raise RuntimeError("duplicate membership ordinal")
        if row["disposition"] == "admit":
            key = row["record_storage_key"]
            if key in admitted_storage:
                raise RuntimeError("duplicate admitted storage key")
            admitted_storage[key] = row
        else:
            expected_rejected.add(ordinal)
        membership_by_ordinal[ordinal] = row
    if set(membership_by_ordinal) != set(range(count)):
        raise RuntimeError("membership ordinals are not the exact selected prefix")
    if len(reject_rows) != len(expected_rejected):
        raise RuntimeError("reject ledger cardinality differs from rejected membership rows")
    reject_by_ordinal = {}
    last_ordinal = -1
    for index, row in enumerate(reject_rows):
        ordinal = row.get("sdf_record_index")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal not in expected_rejected or ordinal in reject_by_ordinal:
            raise RuntimeError("reject ledger does not form a one-to-one rejected subset")
        if ordinal <= last_ordinal:
            raise RuntimeError("reject ledger order is not strictly ascending")
        last_ordinal = ordinal
        csv_row = int(companion_rows[ordinal])
        source_address = builder.source_address_sha256(source_info["archive_sha256"], source_member, ordinal, csv_row)
        try:
            builder.validate_reject_row(row, sidecar_id, selected_sha, ordinal, csv_row, source_address)
        except Exception as exc:
            raise RuntimeError("reject ledger row {} violates the v2 contract".format(index)) from exc
        membership = membership_by_ordinal[ordinal]
        if membership["disposition"] != "reject" or membership["reject_reason_code"] != row["reason_code"]:
            raise RuntimeError("membership and reject ledger reason/disposition disagree")
        reject_by_ordinal[ordinal] = row
    if set(reject_by_ordinal) != expected_rejected:
        raise RuntimeError("reject ledger rejected set differs from membership")
    return membership_by_ordinal, reject_by_ordinal, admitted_storage


def validate_payload_index_rows(builder, payload_index_rows, admitted_storage):
    if len(payload_index_rows) != len(admitted_storage):
        raise RuntimeError("payload index cardinality differs from admitted membership")
    result = {}
    for position, row in enumerate(payload_index_rows):
        builder.forbid_raw_fields(row, "payload-index row")
        require_exact_keys(row, builder.PAYLOAD_INDEX_ROW_FIELDS, "payload-index row {}".format(position))
        if row["payload_index_schema_version"] != builder.PAYLOAD_INDEX_SCHEMA:
            raise RuntimeError("payload-index row schema mismatch")
        key = row["record_storage_key"]
        if key not in admitted_storage or key in result:
            raise RuntimeError("payload-index key does not equal a unique admitted membership key")
        if row["record_content_sha256"] != admitted_storage[key]["record_content_sha256"]:
            raise RuntimeError("payload-index logical hash differs from membership")
        if not isinstance(row["record_wire_bytes"], int) or isinstance(row["record_wire_bytes"], bool) or row["record_wire_bytes"] < 1:
            raise RuntimeError("payload-index wire byte count is invalid")
        try:
            builder.require_sha256(row["record_wire_sha256"], "payload-index.record_wire_sha256")
            builder.require_sha256(row["record_content_sha256"], "payload-index.record_content_sha256")
        except Exception as exc:
            raise RuntimeError("payload-index hash is invalid") from exc
        result[key] = row
    if set(result) != set(admitted_storage):
        raise RuntimeError("payload-index keys do not exactly equal admitted membership")
    return result


def load_and_validate_lmdb(builder, codec, np, lmdb_path, admitted_storage, payload_index_by_key, sidecar_id,
                           selected_sha, scope):
    if not lmdb_path.is_dir() or lmdb_path.is_symlink():
        raise FileNotFoundError("geometry_records.lmdb directory is missing or unsafe")
    import lmdb

    records = {}
    wire_total_bytes = 0
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin(write=False) as transaction:
            cursor = transaction.cursor()
            for raw_key, raw_value in cursor:
                try:
                    key = raw_key.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise RuntimeError("LMDB key is not ASCII") from exc
                if key not in admitted_storage or key in records or key not in payload_index_by_key:
                    raise RuntimeError("LMDB key is not a unique admitted/payload-index key")
                payload_row = payload_index_by_key[key]
                try:
                    builder.validate_payload_index_row(
                        payload_row, key, admitted_storage[key]["record_content_sha256"], raw_value
                    )
                    record, payload_logical_sha = codec.decode_record(np, raw_value)
                except Exception as exc:
                    raise RuntimeError("LMDB v2 payload fails safe decode/index validation for {}".format(key)) from exc
                if payload_logical_sha != payload_row["record_content_sha256"]:
                    raise RuntimeError("LMDB payload logical hash differs from payload index")
                try:
                    builder.validate_admitted_record(np, record)
                except Exception as exc:
                    raise RuntimeError("LMDB decoded logical record violates v2 schema") from exc
                membership = admitted_storage[key]
                member = record["member"]
                if (
                    member["storage_key"] != key
                    or member["member_id"] != membership["member_id"]
                    or member["official_csv_row_index"] != membership["official_csv_row_index"]
                    or member["source_address_sha256"] != membership["source_address_sha256"]
                ):
                    raise RuntimeError("LMDB record/member row binding mismatch")
                if builder.logical_record_sha256(np, record) != membership["record_content_sha256"]:
                    raise RuntimeError("LMDB logical record hash differs from membership")
                sidecar = record["sidecar"]
                if sidecar["sidecar_id"] != sidecar_id or sidecar["selected_ordinal_set_sha256"] != selected_sha:
                    raise RuntimeError("LMDB record sidecar binding mismatch")
                if sidecar["source_contract_sha256"] != scope["source"]["source_contract_sha256"]:
                    raise RuntimeError("LMDB record source contract hash differs from scope")
                if sidecar["record_schema_sha256"] != scope["contracts"]["record_schema_sha256"]:
                    raise RuntimeError("LMDB record schema hash differs from scope")
                if sidecar["adapter_harness_sha256"] != scope["contracts"]["adapter_lock_sha256"]:
                    raise RuntimeError("LMDB adapter hash differs from scope")
                if record["identity"]["identity_spec_sha256"] != scope["contracts"]["identity_normalization_contract_sha256"]:
                    raise RuntimeError("LMDB identity contract hash differs from scope")
                if record["identity"]["sdf_strict_smiles_sha256"] != record["identity"]["official_strict_smiles_sha256"]:
                    raise RuntimeError("admitted record does not have strict identity equality")
                records[key] = record
                wire_total_bytes += len(raw_value)
    finally:
        env.close()
    if set(records) != set(admitted_storage) or set(records) != set(payload_index_by_key):
        raise RuntimeError("LMDB/payload-index/admitted membership keys are not one-to-one")
    return records, wire_total_bytes


def validate_build_report(builder, build_report, build_report_path, release_root_path, scope_path, membership_path,
                          reject_path, payload_index_path, count, admitted_count, rejected_count,
                          payload_wire_total_bytes, payload_format_contract_path):
    builder.forbid_raw_fields(build_report, "build report")
    if build_report.get("schema_version") != builder.BUILD_REPORT_SCHEMA:
        raise RuntimeError("build report schema is not v2")
    if build_report.get("sidecar_schema_version") != SIDE_CAR_SCHEMA or build_report.get("logical_record_schema_version") != RECORD_SCHEMA:
        raise RuntimeError("build report sidecar/record schema binding mismatch")
    if build_report.get("pass") is not True or build_report.get("partition_invariant_pass") is not True:
        raise RuntimeError("build report does not declare a complete passing partition")
    counts = build_report.get("counts")
    if not isinstance(counts, dict) or (
        counts.get("selected_ordinal_count"), counts.get("admitted_record_count"), counts.get("reject_ledger_record_count")
    ) != (count, admitted_count, rejected_count):
        raise RuntimeError("build report counts differ from persisted sidecar")
    artifacts = build_report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("build report lacks artifacts")
    expected_hashes = {
        "scope_manifest_sha256": builder.sha256_file(scope_path),
        "membership_sha256": builder.sha256_file(membership_path),
        "reject_ledger_sha256": builder.sha256_file(reject_path),
        "payload_index_sha256": builder.sha256_file(payload_index_path),
        "payload_format_contract_sha256": builder.sha256_file(payload_format_contract_path),
    }
    for key, expected in expected_hashes.items():
        if artifacts.get(key) != expected:
            raise RuntimeError("build report artifact {} hash mismatch".format(key))
    if artifacts.get("payload_schema_version") != builder.PAYLOAD_SCHEMA:
        raise RuntimeError("build report payload schema mismatch")
    if artifacts.get("payload_wire_total_bytes") != payload_wire_total_bytes:
        raise RuntimeError("build report payload wire total differs from LMDB")
    if artifacts.get("release_root_schema_version") != RELEASE_ROOT_SCHEMA:
        raise RuntimeError("build report release-root schema mismatch")
    expected_root = builder.sha256_json(
        {
            "release_root_schema_version": RELEASE_ROOT_SCHEMA,
            "scope_manifest_sha256": expected_hashes["scope_manifest_sha256"],
            "membership_sha256": expected_hashes["membership_sha256"],
            "reject_ledger_sha256": expected_hashes["reject_ledger_sha256"],
            "payload_index_sha256": expected_hashes["payload_index_sha256"],
            "selected_ordinal_count": count,
            "admitted_record_count": admitted_count,
            "reject_ledger_record_count": rejected_count,
            "payload_wire_total_bytes": payload_wire_total_bytes,
        }
    )
    if artifacts.get("release_root_sha256") != expected_root:
        raise RuntimeError("build report release-root hash mismatch")
    _, handoff_root = read_json(release_root_path, "release root")
    require_exact_keys(
        handoff_root,
        (
            "schema_version", "release_status", "logical_release_root_sha256", "build_report_sha256",
            "artifacts", "counts",
        ),
        "release root",
    )
    if handoff_root["schema_version"] != RELEASE_ROOT_SCHEMA or handoff_root["release_status"] != "non_admissible_pre_tokenizer":
        raise RuntimeError("release root schema/status mismatch")
    if handoff_root["logical_release_root_sha256"] != expected_root:
        raise RuntimeError("release root logical hash mismatch")
    if handoff_root["build_report_sha256"] != builder.sha256_file(build_report_path):
        raise RuntimeError("release root build-report hash mismatch")
    if handoff_root["artifacts"] != {
        "scope_manifest_sha256": expected_hashes["scope_manifest_sha256"],
        "membership_sha256": expected_hashes["membership_sha256"],
        "reject_ledger_sha256": expected_hashes["reject_ledger_sha256"],
        "payload_index_sha256": expected_hashes["payload_index_sha256"],
    }:
        raise RuntimeError("release root artifact map mismatch")
    if handoff_root["counts"] != {
        "selected_ordinal_count": count,
        "admitted_record_count": admitted_count,
        "reject_ledger_record_count": rejected_count,
        "payload_wire_total_bytes": payload_wire_total_bytes,
    }:
        raise RuntimeError("release root counts mismatch")
    return expected_root, builder.sha256_file(release_root_path)


def recompute_all(builder, Chem, np, preflight, linearizer, e3fp_api, identity_gate, archive_path, companion_rows,
                  csv_smiles, csv_malformed, source_info, scope, identity_contract_sha256,
                  projection_spec_sha256, linearizer_spec_sha256, membership_by_ordinal,
                  reject_by_ordinal, records_by_key):
    count = len(companion_rows)
    sidecar_values = {
        "sidecar_id": scope["sidecar_id"],
        "selected_ordinal_set_sha256": scope["selection"]["selected_ordinal_set_sha256"],
        "source_contract_sha256": scope["source"]["source_contract_sha256"],
        "adapter_harness_sha256": scope["contracts"]["adapter_lock_sha256"],
        "record_schema_sha256": scope["contracts"]["record_schema_sha256"],
    }
    source_member = source_info["verified_input_lock"]["train_sdf_member"]
    reason_counts = Counter()
    recomputed_admits = 0
    recomputed_rejects = 0
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = builder.find_locked_sdf_member(archive, source_member)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("cannot reopen locked SDF tar member")
        try:
            supplier = Chem.ForwardSDMolSupplier(stream, sanitize=True, removeHs=False)
            for ordinal, source_mol in enumerate(supplier):
                if ordinal >= count:
                    break
                csv_row = int(companion_rows[ordinal])
                source_address = builder.source_address_sha256(
                    source_info["archive_sha256"], source_member, ordinal, csv_row
                )
                diagnostic = csv_malformed.get(csv_row)
                if diagnostic is None and csv_row not in csv_smiles:
                    diagnostic = "csv_row_unresolved"
                record, reject = builder.build_record(
                    Chem, np, preflight, linearizer, e3fp_api, identity_gate, ordinal, csv_row,
                    csv_smiles.get(csv_row), source_mol, sidecar_values, source_info["archive_sha256"],
                    source_address, identity_contract_sha256, projection_spec_sha256, linearizer_spec_sha256,
                    official_input_diagnostic=diagnostic,
                )
                membership = membership_by_ordinal[ordinal]
                if record is not None:
                    if membership["disposition"] != "admit":
                        raise RuntimeError("recomputed record is admitted but membership says reject")
                    key = builder.storage_key(ordinal)
                    stored = records_by_key.get(key)
                    if stored is None:
                        raise RuntimeError("recomputed admitted record is missing from LMDB")
                    if builder.logical_record_sha256(np, record) != builder.logical_record_sha256(np, stored):
                        raise RuntimeError("recomputed logical record differs from persisted LMDB record")
                    recomputed_admits += 1
                else:
                    if membership["disposition"] != "reject" or membership["reject_reason_code"] != reject["reason_code"]:
                        raise RuntimeError("recomputed rejection differs from membership")
                    expected_ledger = builder.build_reject_row(
                        scope["sidecar_id"], scope["selection"]["selected_ordinal_set_sha256"], ordinal, csv_row, reject
                    )
                    ledger = reject_by_ordinal.get(ordinal)
                    if ledger != expected_ledger:
                        raise RuntimeError("recomputed v2 reject witness differs from reject ledger")
                    reason_counts[reject["reason_code"]] += 1
                    recomputed_rejects += 1
            if recomputed_admits + recomputed_rejects != count:
                raise RuntimeError("SDF ended before every bounded ordinal was recomputed")
        finally:
            stream.close()
    return recomputed_admits, recomputed_rejects, dict(sorted(reason_counts.items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--split-dict", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--identity-normalization-contract", required=True)
    parser.add_argument("--record-schema", required=True)
    parser.add_argument("--payload-format-contract", required=True)
    parser.add_argument("--adapter-lock", required=True)
    parser.add_argument("--replay-lock", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unsafe-legacy-torch-load", action="store_true")
    return parser.parse_args(argv)


def run(args):
    sidecar_dir = Path(args.sidecar_dir).expanduser().resolve()
    if not sidecar_dir.is_dir() or sidecar_dir.is_symlink():
        raise FileNotFoundError("--sidecar-dir is not a non-symlink directory")
    if (sidecar_dir / "build_failure.json").exists():
        raise RuntimeError("sidecar contains build_failure.json and cannot be validated as complete")
    output_path = Path(args.output).expanduser()
    if output_path.exists():
        raise FileExistsError("validation --output must be a new path")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("validation output parent directory does not exist")

    root = Path(__file__).resolve().parents[1]
    builder_path = root / "adapter" / "build_pcqm_p1_geometry_sidecar.py"
    linearizer_path = root / "adapter" / "mol_linearizer.py"
    source_integrity_path = root / "adapter" / "pcqm_source_integrity.py"
    codec_path = root / "adapter" / "sidecar_v2_codec.py"
    preflight_path = root / "gates" / "pcqm_e3fp_preflight.py"
    identity_gate_path = root / "gates" / "pcqm_identity_smoke.py"
    for path in (builder_path, linearizer_path, source_integrity_path, codec_path, preflight_path, identity_gate_path):
        if not path.is_file():
            raise FileNotFoundError("required sidecar harness component missing: {}".format(path))
    builder = import_module_from_file(builder_path, "r1_pcqm_geometry_builder_validator")
    source_integrity = import_module_from_file(source_integrity_path, "r1_pcqm_source_integrity_validator")
    preflight = import_module_from_file(preflight_path, "r1_pcqm_e3fp_preflight_validator")
    identity_gate = import_module_from_file(identity_gate_path, "r1_pcqm_identity_validator")
    linearizer = import_module_from_file(linearizer_path, "r1_pcqm_linearizer_validator")
    codec = import_module_from_file(codec_path, "r1_pcqm_sidecar_v2_codec_validator")

    try:
        import numpy as np
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("remote validation requires NumPy and RDKit") from exc

    archive_path = builder.regular_file(args.archive, "archive")
    data_csv_path = builder.regular_file(args.data_csv, "official data.csv.gz")
    split_dict_path = builder.regular_file(args.split_dict, "official split_dict.pt")
    source_contract_path, _ = builder.load_json(args.source_contract, "source contract")
    identity_contract_path, identity_contract = builder.load_json(args.identity_normalization_contract, "identity normalization contract")
    record_schema_path, record_schema = builder.load_json(args.record_schema, "record schema contract")
    payload_format_contract_path, payload_format_contract = builder.load_json(args.payload_format_contract, "payload format contract")
    adapter_lock_path, adapter_lock = builder.load_json(args.adapter_lock, "adapter lock")
    replay_lock_path, replay_lock = read_json(args.replay_lock, "replay gate lock")

    # The full source chain must pass before opening SDF/CSV or deserializing split_dict.pt.
    verified_inputs, source_info = builder.validate_source_contract(
        source_integrity, source_contract_path, archive_path, data_csv_path, split_dict_path
    )
    builder.validate_identity_contract(identity_contract)
    builder.validate_record_schema(record_schema, MAX_SMOKE_RECORDS)
    builder.validate_payload_format_contract(payload_format_contract)
    observed_adapter_lock_sha = builder.validate_adapter_lock(
        adapter_lock_path, adapter_lock, builder_path, linearizer_path, preflight_path, identity_gate_path,
        source_integrity_path, codec_path, source_contract_path, identity_contract_path, record_schema_path,
        payload_format_contract_path,
    )
    observed_replay_lock_sha = validate_replay_gate_lock(replay_lock_path, replay_lock, Path(__file__).resolve())

    scope_path, scope = read_json(sidecar_dir / "smoke_scope_manifest.json", "scope manifest")
    requested_count = scope.get("selection", {}).get("selected_record_count") if isinstance(scope.get("selection"), dict) else None
    if not isinstance(requested_count, int) or isinstance(requested_count, bool) or requested_count < 1 or requested_count > MAX_SMOKE_RECORDS:
        raise RuntimeError("scope selected record count is invalid before companion data may be opened")
    companion_rows, split_observed = builder.select_prefix_companion_rows(
        source_integrity, verified_inputs, requested_count,
        args.allow_unsafe_legacy_torch_load,
    )
    count, selected_sha = validate_scope(
        scope, builder, codec, source_contract_path, identity_contract_path, record_schema_path,
        payload_format_contract_path, adapter_lock_path, archive_path, data_csv_path, split_dict_path,
        source_info["verified_input_lock"], split_observed, sidecar_dir,
    )
    if len(companion_rows) != count:
        raise RuntimeError("verified companion prefix length differs from scope selection")
    csv_smiles, csv_malformed = builder.read_selected_csv_smiles(data_csv_path, companion_rows)

    membership_path = sidecar_dir / "membership.jsonl"
    reject_path = sidecar_dir / "reject_ledger.jsonl"
    payload_index_path = sidecar_dir / "payload_index.jsonl"
    membership_rows = read_jsonl(membership_path, "membership")
    reject_rows = read_jsonl(reject_path, "reject ledger")
    payload_index_rows = read_jsonl(payload_index_path, "payload index")
    membership_by_ordinal, reject_by_ordinal, admitted_storage = validate_membership_and_ledger(
        builder, record_schema, membership_rows, reject_rows, scope["sidecar_id"], selected_sha, count,
        companion_rows, source_info,
    )
    payload_index_by_key = validate_payload_index_rows(builder, payload_index_rows, admitted_storage)
    records_by_key, payload_wire_total_bytes = load_and_validate_lmdb(
        builder, codec, np, sidecar_dir / "geometry_records.lmdb", admitted_storage, payload_index_by_key,
        scope["sidecar_id"], selected_sha, scope,
    )

    build_path, build_report = read_json(sidecar_dir / "build_report.json", "build report")
    release_root_path = sidecar_dir / "release_root.json"
    release_root_sha256, handoff_root_sha256 = validate_build_report(
        builder, build_report, build_path, release_root_path, scope_path, membership_path, reject_path, payload_index_path, count,
        len(records_by_key), len(reject_by_ordinal), payload_wire_total_bytes, payload_format_contract_path,
    )

    import_root, package_root, e3fp_files = preflight.resolve_e3fp_source(args.e3fp_source)
    e3fp_api = preflight.import_locked_e3fp(import_root, package_root)
    if scope["harness"]["e3fp_module_version"] != e3fp_api["module_version"]:
        raise RuntimeError("scope E3FP module version differs from replay")
    if scope["harness"]["e3fp_module_file"] != str(e3fp_api["module_file"]):
        raise RuntimeError("scope E3FP module path differs from replay")
    observed_e3fp_files = {label: builder.sha256_file(path) for label, path in sorted(e3fp_files.items())}
    if scope["harness"]["e3fp_source_file_sha256"] != observed_e3fp_files:
        raise RuntimeError("scope E3FP source-file hashes differ from replay")
    if scope["harness"]["rdkit_version"] != Chem.rdBase.rdkitVersion:
        raise RuntimeError("scope RDKit version differs from replay")
    recomputed_admits, recomputed_rejects, recomputed_reason_counts = recompute_all(
        builder, Chem, np, preflight, linearizer, e3fp_api, identity_gate, archive_path, companion_rows,
        csv_smiles, csv_malformed, source_info, scope, builder.sha256_file(identity_contract_path),
        builder.sha256_json(preflight.HYDROGEN_PROJECTION_PROFILE), builder.sha256_file(linearizer_path),
        membership_by_ordinal, reject_by_ordinal, records_by_key,
    )
    report = {
        "schema_version": VALIDATION_REPORT_SCHEMA,
        "created_utc": utc_now(),
        "pass": True,
        "sidecar_dir": str(sidecar_dir),
        "sidecar_id": scope["sidecar_id"],
        "sidecar_schema_version": SIDE_CAR_SCHEMA,
        "logical_record_schema_version": RECORD_SCHEMA,
        "release_status": "non_admissible_pre_tokenizer",
        "validation_class": "builder_linked_deterministic_replay",
        "independent_semantic_validation": False,
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "selection": scope["selection"],
        "counts": {
            "selected_ordinal_count": count,
            "admitted_record_count": len(records_by_key),
            "reject_ledger_record_count": len(reject_by_ordinal),
            "recomputed_admitted_record_count": recomputed_admits,
            "recomputed_reject_ledger_record_count": recomputed_rejects,
            "payload_index_record_count": len(payload_index_by_key),
            "payload_wire_total_bytes": payload_wire_total_bytes,
        },
        "partition_invariant_pass": count == len(records_by_key) + len(reject_by_ordinal),
        "recompute_identity": {
            "all_selected_ordinals_recomputed": recomputed_admits + recomputed_rejects == count,
            "split_loading": split_observed,
            "reject_reason_counts": recomputed_reason_counts,
        },
        "validated_artifacts": {
            "scope_manifest_sha256": builder.sha256_file(scope_path),
            "build_report_sha256": builder.sha256_file(build_path),
            "membership_sha256": builder.sha256_file(membership_path),
            "reject_ledger_sha256": builder.sha256_file(reject_path),
            "payload_index_sha256": builder.sha256_file(payload_index_path),
            "release_root_sha256": release_root_sha256,
            "release_root_handoff_sha256": handoff_root_sha256,
            "source_contract_sha256": builder.sha256_file(source_contract_path),
            "record_schema_sha256": builder.sha256_file(record_schema_path),
            "payload_format_contract_sha256": builder.sha256_file(payload_format_contract_path),
            "adapter_lock_sha256": observed_adapter_lock_sha,
            "replay_gate_lock_sha256": observed_replay_lock_sha,
            "replay_gate_sha256": builder_sha256(Path(__file__).resolve()),
            "sidecar_codec_sha256": builder.sha256_file(codec_path),
        },
        "next_gate": "This builder-linked replay proves safe persistence and same-harness determinism only. Run a separately locked independent reference semantic audit before any CPU census, tokenizer freeze, or P1 admission decision.",
    }
    builder.write_json_new(output_path, report)
    return report


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "selected": report["counts"]["selected_ordinal_count"],
                "admitted": report["counts"]["admitted_record_count"],
                "rejected": report["counts"]["reject_ledger_record_count"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
