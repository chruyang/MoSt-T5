#!/usr/bin/env python3
"""Restore the historical paper-scope downstream exclusions for Phase I.

The old final-v4 membership excluded 5,510 otherwise-admitted PCQM records
because their non-stereo connectivity occurred in downstream validation/test
collections.  That policy is no longer the Phase-I corpus policy.  This tool
does not rebuild chemistry or copy payloads: it binds every historical ledger
row back to its immutable production-v2 membership, payload index, and LMDB
wire bytes, then publishes an explicit reversal ledger and a segmented source
manifest.  Phase-II decontamination remains a separate derivation.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


RESTORED_SCHEMA = "most-t5-r1/phase1-restored-downstream-member/v1"
SEGMENT_SCHEMA = "most-t5-r1/phase1-unified-source-segment/v1"
MANIFEST_SCHEMA = "most-t5-r1/phase1-unified-pcqm-membership-manifest/v1"
RESTORED_FILENAME = "restored_downstream_members.jsonl"
SEGMENTS_FILENAME = "source_segments.jsonl"
MANIFEST_FILENAME = "manifest.json"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


class JsonlWriter(object):
    def __init__(self, path):
        self.path = Path(path)
        self.handle = open(str(self.path), "xb")
        self.digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0

    def write(self, row):
        raw = canonical_bytes(row) + b"\n"
        self.handle.write(raw)
        self.digest.update(raw)
        self.rows += 1
        self.bytes += len(raw)

    def finish(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        return {"path": self.path.name, "rows": self.rows, "bytes": self.bytes, "sha256": self.digest.hexdigest()}


def _ordinal(member_id):
    prefix = "ogb_pcqm4mv2_train_row_index:"
    if not isinstance(member_id, str) or not member_id.startswith(prefix):
        raise ValueError("unexpected PCQM member_id: {!r}".format(member_id))
    return int(member_id[len(prefix):])


def _read_historical_ledger(path):
    rows = {}
    with open(str(path), "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            member_id = row.get("member_id")
            ordinal = _ordinal(member_id)
            if ordinal in rows:
                raise ValueError("duplicate historical exclusion ordinal")
            if not row.get("matched_protected_collections"):
                raise ValueError("historical exclusion lacks a protected collection")
            rows[ordinal] = row
    return rows


def _collect_shard_rows(shard_dir, wanted_ordinals):
    wanted_keys = {"{:09d}".format(value): value for value in wanted_ordinals}
    membership = {}
    payload_index = {}
    with open(str(shard_dir / "membership.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("record_storage_key")
            if key in wanted_keys:
                membership[wanted_keys[key]] = row
    with open(str(shard_dir / "payload_index.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("record_storage_key")
            if key in wanted_keys:
                payload_index[wanted_keys[key]] = row
    if set(membership) != set(wanted_ordinals) or set(payload_index) != set(wanted_ordinals):
        raise RuntimeError("historical exclusions are not closed in shard artifacts")
    return membership, payload_index


def _verify_lmdb(shard_dir, ordinals, membership, payload_index):
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("lmdb is required for payload verification") from exc
    env = lmdb.open(str(shard_dir / "geometry_records.lmdb"), subdir=True, readonly=True, lock=False, readahead=False, max_readers=8)
    try:
        with env.begin(write=False) as txn:
            for ordinal in ordinals:
                member = membership[ordinal]
                index = payload_index[ordinal]
                key = "{:09d}".format(ordinal)
                if not (
                    member.get("disposition") == "admit"
                    and member.get("reject_reason_code") is None
                    and member.get("record_storage_key") == key
                    and member.get("record_content_sha256") == index.get("record_content_sha256")
                ):
                    raise RuntimeError("restored member is not an admitted production record")
                raw = txn.get(key.encode("ascii"))
                if raw is None:
                    raise RuntimeError("restored LMDB payload is absent")
                if len(raw) != index.get("record_wire_bytes") or hashlib.sha256(raw).hexdigest() != index.get("record_wire_sha256"):
                    raise RuntimeError("restored LMDB payload disagrees with payload index")
    finally:
        env.close()


def restore_phase1_membership(final_v4_dir, release_root, supplement_dir, output_dir):
    final_v4_dir = Path(final_v4_dir).resolve()
    release_root = Path(release_root).resolve()
    supplement_dir = Path(supplement_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to reuse output directory")

    old_manifest_path = final_v4_dir / "clean_membership_manifest.json"
    old_manifest = load_json(old_manifest_path)
    release_manifest_path = release_root / "full_release_manifest.json"
    release_manifest = load_json(release_manifest_path)
    supplement_manifest_path = supplement_dir / "manifest.json"
    supplement_manifest = load_json(supplement_manifest_path)
    historical = _read_historical_ledger(final_v4_dir / "excluded_member_ledger.jsonl")

    counts = old_manifest.get("counts", {})
    release_counts = release_manifest.get("counts", {})
    supplement_counts = supplement_manifest.get("counts", {})
    if not (
        counts.get("excluded_member_count") == len(historical) == 5510
        and counts.get("permitted_member_count") + len(historical) == counts.get("pretrain_member_count")
        and counts.get("pretrain_member_count") == release_counts.get("admitted_record_count") == 3365577
        and supplement_counts.get("admitted") == supplement_counts.get("selected") == 12978
        and supplement_counts.get("rejected") == 0
        and supplement_manifest.get("identity_policy") == "stereo_free_connectivity_with_sdf_authoritative_state"
    ):
        raise RuntimeError("input count or identity-policy contract drift")

    by_shard = defaultdict(list)
    shard_size = int(release_manifest["configuration"]["shard_size"])
    for ordinal in historical:
        by_shard[ordinal // shard_size].append(ordinal)

    output_dir.mkdir(parents=True, exist_ok=False)
    restored_writer = JsonlWriter(output_dir / RESTORED_FILENAME)
    restored_rows = {}
    for shard_index in sorted(by_shard):
        ordinals = sorted(by_shard[shard_index])
        shard_dir = release_root / "shard-{:06d}".format(shard_index)
        membership, payload_index = _collect_shard_rows(shard_dir, ordinals)
        _verify_lmdb(shard_dir, ordinals, membership, payload_index)
        for ordinal in ordinals:
            old = historical[ordinal]
            member = membership[ordinal]
            index = payload_index[ordinal]
            restored_rows[ordinal] = {
                "schema_version": RESTORED_SCHEMA,
                "member_id": old["member_id"],
                "sdf_record_index": ordinal,
                "connectivity_identity_sha256": old["connectivity_identity_sha256"],
                "historical_exclusion": {
                    "policy": "final-v4 downstream validation/test connectivity union",
                    "matched_protected_collections": old["matched_protected_collections"],
                },
                "restoration": {
                    "scope": "phase_i_syntax_and_geometry_pretraining",
                    "reason": "downstream overlap is not a Phase-I chemistry admission failure",
                    "source_release": release_manifest["release_id"],
                    "shard_index": shard_index,
                    "record_storage_key": member["record_storage_key"],
                    "record_content_sha256": member["record_content_sha256"],
                    "record_wire_bytes": index["record_wire_bytes"],
                    "record_wire_sha256": index["record_wire_sha256"],
                    "payload_verified": True,
                },
            }
    for ordinal in sorted(restored_rows):
        restored_writer.write(restored_rows[ordinal])
    restored_artifact = restored_writer.finish()

    segments_writer = JsonlWriter(output_dir / SEGMENTS_FILENAME)
    segments_writer.write({
        "schema_version": SEGMENT_SCHEMA,
        "segment_index": 0,
        "member_count": release_counts["admitted_record_count"],
        "source_kind": "immutable_sharded_production_release",
        "source_path": str(release_root),
        "identity_policy": "production_v2_strict_isomeric_admission",
        "ordering": "source ordinal among admitted records",
    })
    segments_writer.write({
        "schema_version": SEGMENT_SCHEMA,
        "segment_index": 1,
        "member_count": supplement_counts["admitted"],
        "source_kind": "stereo_recovery_supplement",
        "source_path": str(supplement_dir),
        "identity_policy": supplement_manifest["identity_policy"],
        "ordering": "supplement selection_index",
    })
    segments_artifact = segments_writer.finish()

    source_count = int(release_manifest["configuration"]["source_record_count"])
    unified_count = release_counts["admitted_record_count"] + supplement_counts["admitted"]
    unresolved = source_count - unified_count
    bindings = {}
    for name, path in (
        ("historical_final_v4_manifest", old_manifest_path),
        ("production_release_manifest", release_manifest_path),
        ("stereo_supplement_manifest", supplement_manifest_path),
    ):
        size, digest = sha256_file(path)
        bindings[name] = {"path": str(path), "bytes": size, "sha256": digest}
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "pass",
        "policy": {
            "phase_i": "restore_all_historical_downstream_overlap_exclusions",
            "phase_ii": "derive_separately_with_chebi20_test_only_connectivity_exclusion",
            "historical_final_v4_mutated": False,
            "production_release_mutated": False,
            "payloads_copied": False,
        },
        "counts": {
            "historical_final_v4_permitted": counts["permitted_member_count"],
            "restored_downstream_members": len(historical),
            "strict_production_admitted": release_counts["admitted_record_count"],
            "stereo_recovery_supplement": supplement_counts["admitted"],
            "unified_phase_i_members": unified_count,
            "source_records": source_count,
            "remaining_unresolved_records": unresolved,
        },
        "invariants": {
            "final_v4_plus_restored_equals_strict_production": counts["permitted_member_count"] + len(historical) == release_counts["admitted_record_count"],
            "strict_production_plus_supplement_equals_unified": unified_count == 3378555,
            "all_restored_payloads_verified": restored_artifact["rows"] == 5510,
            "remaining_unresolved_equals_51": unresolved == 51,
        },
        "artifacts": {"restored_downstream_members": restored_artifact, "source_segments": segments_artifact},
        "bindings": bindings,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve())[1],
        },
    }
    manifest["canonical_payload_sha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    with open(str(output_dir / MANIFEST_FILENAME), "xb") as handle:
        handle.write(canonical_bytes(manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v4-dir", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--stereo-supplement-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    manifest = restore_phase1_membership(args.final_v4_dir, args.release_root, args.stereo_supplement_dir, args.output_dir)
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
