from __future__ import print_function

import ast
import json
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from most_t5_next.r1.auditor import audit_pcqm_geometry_production_release as audit


R1_ROOT = Path(__file__).resolve().parents[2]
AUDIT_CONTRACT = R1_ROOT / "contracts" / "p1_pcqm_geometry_independent_audit_contract.json"
PRODUCTION_CONTRACT = R1_ROOT / "contracts" / "p1_pcqm_geometry_production_release_contract.json"
PAYLOAD_CONTRACT = R1_ROOT / "contracts" / "p1_pcqm_geometry_payload_format_contract.json"
HASH = "a" * 64
E3FP_HASH = "e" * 64
MOTIF_FRAGMENT = "CC"
MOTIF_DIGEST = audit.sha256_bytes(MOTIF_FRAGMENT.encode("utf-8"))


def _externalize(value, blocks):
    if isinstance(value, np.ndarray):
        index = len(blocks)
        raw = value.tobytes(order="C")
        descriptor = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "order": "C",
            "sha256": audit.sha256_bytes(raw),
        }
        blocks.append((value, raw, descriptor))
        return {"__array_block__": index, **descriptor}
    if isinstance(value, dict):
        return {key: _externalize(value[key], blocks) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_externalize(item, blocks) for item in value]
    return value


def encode_fixture_record(record):
    blocks = []
    projection = _externalize(record, blocks)
    metadata = []
    offset = 0
    raw_parts = []
    for index, (_, raw, descriptor) in enumerate(blocks):
        metadata.append(
            {
                "index": index,
                "dtype": descriptor["dtype"],
                "shape": descriptor["shape"],
                "order": "C",
                "offset": offset,
                "nbytes": len(raw),
                "sha256": descriptor["sha256"],
            }
        )
        raw_parts.append(raw)
        offset += len(raw)
    logical_hash = audit.sha256_json(audit.logical_projection(np, record))
    header = {
        "payload_schema_version": audit.PAYLOAD_SCHEMA,
        "record": projection,
        "array_blocks": metadata,
        "logical_record_sha256": logical_hash,
    }
    header_raw = audit.canonical_json_bytes(header)
    return audit.MAGIC + struct.pack(">I", len(header_raw)) + header_raw + b"".join(raw_parts), logical_hash


def fixture_record(release_id, selected_hash, ordinal):
    mapping = np.asarray([0, 1], dtype=np.int32)
    group = np.asarray([0, 1], dtype=np.int32)
    coordinates = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    e3fp = np.asarray([[11, -1, -1, -1], [12, 14, -1, -1]], dtype=np.int32)
    motif_valid = np.asarray([True], dtype=np.bool_)
    return {
        "record_schema_version": audit.PRODUCTION_RECORD_SCHEMA,
        "sidecar": {
            "sidecar_id": release_id,
            "sidecar_mode": audit.PRODUCTION_MODE,
            "selected_ordinal_set_sha256": selected_hash,
            "source_contract_sha256": HASH,
            "identity_normalization_contract_sha256": HASH,
            "adapter_harness_sha256": HASH,
            "record_schema_sha256": HASH,
            "geometry_only_pretokenizer": True,
            "p1_training_admission": False,
            "p1_training_launcher_permitted": False,
        },
        "member": {
            "identity_namespace": audit.IDENTITY_NAMESPACE,
            "member_id": audit.member_id(ordinal),
            "sdf_record_index": ordinal,
            "official_csv_row_index": ordinal,
            "storage_key": audit.storage_key(ordinal),
            "source_archive_sha256": HASH,
            "source_address_sha256": HASH,
            "source_mol_identity_sha256": HASH,
        },
        "identity": {
            "official_identity_status": "strict_isomeric_match",
            "sdf_strict_smiles_sha256": HASH,
            "official_strict_smiles_sha256": HASH,
            "canonical_connectivity_sha256": HASH,
            "identity_spec_sha256": HASH,
            "rdkit_version": "fixture",
        },
        "atom_universe": {
            "policy_id": "project_explicit_hydrogens_before_e3fp_v1",
            "hydrogen_projection_spec_sha256": HASH,
            "source_atom_count": 2,
            "source_explicit_hydrogen_count": 0,
            "model_atom_count": 2,
            "model_to_source_atom_index": mapping,
            "geometry_mol_identity_sha256": HASH,
        },
        "topology": {
            "linearizer_spec_sha256": HASH,
            "motif_count": 1,
            "motif_atom_indices": [group],
            "motif_atom_indices_sha256": audit.sha256_json([audit._array_descriptor(group)]),
            "motif_lexeme_sha256": [MOTIF_DIGEST],
        },
        "geometry": {
            "geometry_valid": True,
            "geometry_mse_enabled": False,
            "geometry_mse_candidate_after_tokenizer_binding": True,
            "motif_geometry_valid": motif_valid,
            "coordinates": coordinates,
            "coordinates_sha256": audit.sha256_bytes(coordinates.tobytes(order="C")),
            "e3fp": e3fp,
            "e3fp_shape": [2, 4],
            "e3fp_params_sha256": E3FP_HASH,
            "e3fp_sha256": audit.sha256_bytes(e3fp.tobytes(order="C")),
        },
        "array_metadata": {
            "coordinates_dtype": "float32",
            "coordinates_shape": [2, 3],
            "coordinates_order": "C",
            "e3fp_dtype": "int32",
            "e3fp_shape": [2, 4],
            "e3fp_order": "C",
            "model_to_source_atom_index_dtype": "int32",
            "motif_atom_indices_dtype": "int32",
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_bytes(b"".join(audit.canonical_json_bytes(row) + b"\n" for row in rows))


class FakeCursor(object):
    def __init__(self, values):
        self.values = values

    def iternext(self, keys=True, values=False):
        if not keys or values:
            raise AssertionError("auditor must request keys only during full closure")
        return iter(sorted(self.values))


class FakeTransaction(object):
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return FakeCursor(self.values)

    def get(self, key):
        return self.values.get(key)


class FakeEnvironment(object):
    def __init__(self, values):
        self.values = values

    def begin(self, write=False):
        if write:
            raise AssertionError("independent audit must open read-only transactions")
        return FakeTransaction(self.values)

    def close(self):
        return None


def build_release(root):
    release_id = "fixture-benchmark-128"
    release_root = root / release_id
    shard = release_root / "shard-000000"
    records_dir = shard / "geometry_records.lmdb"
    records_dir.mkdir(parents=True)
    (records_dir / "data.mdb").write_bytes(b"opaque-lmdb-fixture")
    selected_count = 128
    selected_hash = audit.sha256_ordinal_range(0, selected_count)
    reject_ordinal = 17
    memberships = []
    rejects = []
    indices = []
    payloads = {}
    for ordinal in range(selected_count):
        base = {
            "record_schema_version": audit.PRODUCTION_RECORD_SCHEMA,
            "sidecar_id": release_id,
            "sidecar_mode": audit.PRODUCTION_MODE,
            "selected_ordinal_set_sha256": selected_hash,
            "member_id": audit.member_id(ordinal),
            "sdf_record_index": ordinal,
            "official_csv_row_index": ordinal,
            "source_address_sha256": HASH,
        }
        if ordinal == reject_ordinal:
            memberships.append(
                {
                    **base, "disposition": "reject", "record_storage_key": None,
                    "record_content_sha256": None, "reject_reason_code": "SDF_PARSE_FAILED",
                }
            )
            detail = {
                "diagnostic_code": "sdf_rdkit_none",
                "reason_code": "SDF_PARSE_FAILED",
                "source_address_sha256": HASH,
                "stage": "sdf_parse",
            }
            rejects.append(
                {
                    **base, "stage": "sdf_parse", "reason_code": "SDF_PARSE_FAILED",
                    "action": "exclude_from_geometry_release", "geometry_mse_enabled": False,
                    "source_mol_identity_sha256": None,
                    "geometry_mol_identity_sha256": None,
                    "diagnostic_code": "sdf_rdkit_none",
                    "detail_sha256": audit.sha256_json(detail),
                }
            )
            continue
        payload, logical_hash = encode_fixture_record(fixture_record(release_id, selected_hash, ordinal))
        key = audit.storage_key(ordinal)
        memberships.append(
            {
                **base, "disposition": "admit", "record_storage_key": key,
                "record_content_sha256": logical_hash, "reject_reason_code": None,
            }
        )
        indices.append(
            {
                "payload_index_schema_version": audit.PAYLOAD_INDEX_SCHEMA,
                "record_storage_key": key,
                "record_wire_bytes": len(payload),
                "record_wire_sha256": audit.sha256_bytes(payload),
                "record_content_sha256": logical_hash,
            }
        )
        payloads[key.encode("ascii")] = payload
    write_jsonl(shard / "membership.jsonl", memberships)
    write_jsonl(shard / "reject_ledger.jsonl", rejects)
    write_jsonl(shard / "payload_index.jsonl", indices)
    motif_rows = [
        {
            "motif_lexeme_sha256": MOTIF_DIGEST,
            "motif_fragment": MOTIF_FRAGMENT,
            "count": selected_count - 1,
        }
    ]
    write_jsonl(shard / "motif_census.jsonl", motif_rows)
    artifacts = {}
    for role, relative in audit.ARTIFACT_PATHS.items():
        path = shard / Path(relative)
        artifacts[role] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": audit.sha256_file(path),
        }
    production_hash = audit.sha256_file(PRODUCTION_CONTRACT)
    shard_manifest = {
        "schema_version": audit.SHARD_MANIFEST_SCHEMA,
        "created_utc": "2026-08-05T00:00:00+00:00",
        "release_status": "complete",
        "release_id": release_id,
        "production_contract_sha256": production_hash,
        "shard_index": 0,
        "range_start": 0,
        "range_end": selected_count,
        "selected_record_count": selected_count,
        "counts": {
            "membership_record_count": selected_count,
            "admitted_record_count": selected_count - 1,
            "reject_ledger_record_count": 1,
            "payload_index_record_count": selected_count - 1,
            "payload_wire_total_bytes": sum(len(item) for item in payloads.values()),
            "motif_occurrence_count": selected_count - 1,
            "unique_motif_count": 1,
        },
        "reject_reason_counts": {"SDF_PARSE_FAILED": 1},
        "e3fp_params_sha256_values": [E3FP_HASH],
        "artifacts": artifacts,
        "partition_invariant_pass": True,
        "lmdb_merged": False,
        "p1_training_admission": False,
    }
    write_json(shard / "shard_manifest.json", shard_manifest)
    write_jsonl(release_root / "motif_census.jsonl", motif_rows)
    components = {"payload_contract": audit.sha256_file(PAYLOAD_CONTRACT)}
    configuration = {
        "release_id": release_id,
        "production_contract_sha256": production_hash,
        "release_kind": "benchmark_non_release",
        "source_record_count": 3_378_606,
        "selected_record_count": selected_count,
        "selected_ordinal_range": [0, selected_count],
        "selected_ordinal_set_sha256": selected_hash,
        "shard_size": 25000,
        "shard_count": 1,
        "harness": {"components": components, "bundle_sha256": audit.sha256_json(components)},
        "logical_record_schema_version": audit.PRODUCTION_RECORD_SCHEMA,
        "sidecar_mode": audit.PRODUCTION_MODE,
    }
    scope = {
        "schema_version": audit.SCOPE_SCHEMA,
        "created_utc": "2026-08-05T00:00:00+00:00",
        "release_status": "benchmark_non_release",
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "configuration": configuration,
    }
    write_json(release_root / "production_scope.json", scope)
    shard_roots = [
        {
            "shard_index": 0, "range_start": 0, "range_end": selected_count,
            "shard_manifest_sha256": audit.sha256_file(shard / "shard_manifest.json"),
        }
    ]
    global_path = release_root / "motif_census.jsonl"
    logical_root = audit.sha256_json(
        {
            "configuration": configuration,
            "global_motif_census_sha256": audit.sha256_file(global_path),
            "shards": shard_roots,
            "membership_record_count": selected_count,
            "admitted_record_count": selected_count - 1,
            "reject_ledger_record_count": 1,
        }
    )
    report = {
        "schema_version": audit.BENCHMARK_REPORT_SCHEMA,
        "created_utc": "2026-08-05T00:00:00+00:00",
        "release_status": "benchmark_non_release",
        "release_id": release_id,
        "logical_benchmark_root_sha256": logical_root,
        "configuration": configuration,
        "counts": {
            "source_record_count": 3_378_606,
            "selected_record_count": selected_count,
            "membership_record_count": selected_count,
            "admitted_record_count": selected_count - 1,
            "reject_ledger_record_count": 1,
            "shard_count": 1,
            "unique_motif_count": 1,
            "motif_occurrence_count": selected_count - 1,
        },
        "global_motif_census": {
            "relative_path": "motif_census.jsonl",
            "bytes": global_path.stat().st_size,
            "sha256": audit.sha256_file(global_path),
        },
        "shards": shard_roots,
        "range_no_gap_no_overlap": True,
        "lmdb_merged": False,
        "full_release_manifest_permitted": False,
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
    }
    write_json(release_root / "benchmark_report.json", report)
    return release_root, payloads


class IndependentProductionAuditTest(unittest.TestCase):
    def test_complete_streaming_closure_and_preregistered_sample_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, payloads = build_release(root)
            output = root / "audit-output"
            report = audit.audit_release(
                release, AUDIT_CONTRACT, PRODUCTION_CONTRACT, PAYLOAD_CONTRACT,
                output, lmdb_opener=lambda _: FakeEnvironment(payloads),
            )
            self.assertEqual(report["audit_status"], "pass")
            self.assertEqual(report["counts"]["membership_record_count"], 128)
            self.assertEqual(report["counts"]["rejects_scheduled_for_semantic_recompute"], 1)
            self.assertEqual(report["counts"]["sampled_admitted_payload_count"], 4)
            rows = list(audit.iter_canonical_jsonl(output / "semantic_review_plan.jsonl", "plan"))
            self.assertTrue(rows[0]["all_rejects_included"])
            self.assertEqual(sum(row["document_kind"] == "reject_semantic_review" for row in rows), 1)
            self.assertFalse(report["limitations"]["independent_rdkit_e3fp_recompute_executed"])
            self.assertFalse(report["limitations"]["p1_training_admission"])

    def test_missing_lmdb_key_fails_full_closure_before_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, payloads = build_release(root)
            broken = dict(payloads)
            broken.pop(sorted(broken)[30])
            with self.assertRaisesRegex(RuntimeError, "LMDB keys"):
                audit.audit_release(
                    release, AUDIT_CONTRACT, PRODUCTION_CONTRACT, PAYLOAD_CONTRACT,
                    root / "audit-output", lmdb_opener=lambda _: FakeEnvironment(broken),
                )

    def test_sampled_wire_corruption_fails_after_plan_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, payloads = build_release(root)
            broken = {key: value[:-1] + bytes([value[-1] ^ 1]) for key, value in payloads.items()}
            output = root / "audit-output"
            with self.assertRaisesRegex(RuntimeError, "wire hash"):
                audit.audit_release(
                    release, AUDIT_CONTRACT, PRODUCTION_CONTRACT, PAYLOAD_CONTRACT,
                    output, lmdb_opener=lambda _: FakeEnvironment(broken),
                )
            self.assertTrue((output / "semantic_review_plan.jsonl").is_file())
            self.assertFalse((output / "independent_audit_report.json").exists())

    def test_decoder_rejects_noncanonical_header(self):
        record = fixture_record("release", audit.sha256_ordinal_range(0, 1), 0)
        payload, _ = encode_fixture_record(record)
        prefix = len(audit.MAGIC) + audit.HEADER_LENGTH_BYTES
        size = struct.unpack(">I", payload[len(audit.MAGIC):prefix])[0]
        header = json.loads(payload[prefix:prefix + size].decode("utf-8"))
        noncanonical = json.dumps(header, ensure_ascii=False, sort_keys=False).encode("utf-8")
        altered = audit.MAGIC + struct.pack(">I", len(noncanonical)) + noncanonical + payload[prefix + size:]
        with self.assertRaisesRegex(RuntimeError, "canonical JSON"):
            audit.decode_payload(np, altered)

    def test_motif_content_address_and_collision_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            census = Path(temporary) / "motif_census.jsonl"
            write_jsonl(
                census,
                [{"motif_lexeme_sha256": HASH, "motif_fragment": "CC", "count": 1}],
            )
            with self.assertRaisesRegex(RuntimeError, "exact UTF-8"):
                audit._validate_motif_census(census, 1, 1)
        counts = Counter()
        lexemes = {}
        with mock.patch.object(audit, "motif_lexeme_sha256", return_value=HASH):
            audit.register_motif_binding(counts, lexemes, HASH, "C", 1)
            with self.assertRaisesRegex(RuntimeError, "collision"):
                audit.register_motif_binding(counts, lexemes, HASH, "O", 1)

    def test_sampled_motif_digest_must_resolve_in_global_dictionary(self):
        selected_hash = audit.sha256_ordinal_range(0, 1)
        record = fixture_record("release", selected_hash, 0)
        membership = {
            "sidecar_id": "release",
            "selected_ordinal_set_sha256": selected_hash,
            "member_id": audit.member_id(0),
            "sdf_record_index": 0,
            "official_csv_row_index": 0,
            "record_storage_key": audit.storage_key(0),
            "source_address_sha256": HASH,
        }
        manifest = {"e3fp_params_sha256_values": [E3FP_HASH]}
        with self.assertRaisesRegex(RuntimeError, "global motif dictionary"):
            audit.validate_sampled_record(np, record, membership, manifest, {})

    def test_auditor_has_no_project_module_imports(self):
        source = Path(audit.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name.startswith("most_t5_next") for name in imported))


if __name__ == "__main__":
    unittest.main(verbosity=2)
