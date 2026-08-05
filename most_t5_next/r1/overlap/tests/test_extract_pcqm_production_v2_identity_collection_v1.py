import hashlib
import json
import sqlite3
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy  # Keep the real module resident while mock.patch.dict restores sys.modules.

from most_t5_next.r1.overlap import extract_pcqm_production_v2_identity_collection_v1 as extractor
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(canonical(value))


def digest_file(path):
    raw = Path(path).read_bytes()
    return len(raw), digest_bytes(raw)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_bytes(b"".join(canonical(row) + b"\n" for row in rows))


def artifact(path, relative):
    size, sha = digest_file(path)
    return {"relative_path": relative, "bytes": size, "sha256": sha}


class FakeTransaction(object):
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return iter(sorted(self.records.items()))


class FakeEnvironment(object):
    def __init__(self, records):
        self.records = records

    def begin(self, write=False):
        if write:
            raise AssertionError("extractor attempted a write transaction")
        return FakeTransaction(self.records)

    def close(self):
        return None


def fake_lmdb_module(registry):
    module = types.ModuleType("lmdb")
    module.__version__ = "hermetic-fake-1"

    def open_environment(path, **options):
        assert options["readonly"] is True
        assert options["lock"] is False
        assert options["create"] is False
        return FakeEnvironment(registry[str(Path(path).resolve())])

    module.open = open_environment
    return module


class Fixture(object):
    def __init__(self, root, identity_spec_mismatch=False, strict_hash_mismatch=False, logical_hash_mismatch=False):
        self.root = Path(root)
        self.release_root = self.root / "release"
        self.shard_dir = self.release_root / "shard-000000"
        self.lmdb_dir = self.shard_dir / "geometry_records.lmdb"
        self.release_root.mkdir()
        self.shard_dir.mkdir()
        self.lmdb_dir.mkdir()
        self.output = self.root / "output"
        self.contract = Path(extractor.__file__).resolve().parents[1] / "contracts" / "pcqm_production_v2_identity_extraction_contract_v1.json"
        self.production_contract = self.root / "production_contract.json"
        self.payload_contract = self.root / "payload_contract.json"
        self.identity_contract = self.root / "identity_contract.json"
        write_json(
            self.production_contract,
            {
                "schema_version": extractor.PRODUCTION_CONTRACT_SCHEMA,
                "logical_record": {
                    "schema_version": extractor.PRODUCTION_RECORD_SCHEMA,
                    "mode": extractor.PRODUCTION_MODE,
                    "p1_training_admission": False,
                },
            },
        )
        write_json(
            self.payload_contract,
            {
                "schema_version": extractor.PAYLOAD_CONTRACT_SCHEMA,
                "payload_schema_version": extractor.PAYLOAD_SCHEMA,
                "magic_ascii": extractor.MAGIC.decode("ascii"),
                "header_required_fields": sorted(extractor.HEADER_FIELDS),
                "array_block_required_fields": ["index", "dtype", "shape", "order", "offset", "nbytes", "sha256"],
                "allowed_dtypes": sorted(extractor.WIRE_DTYPES),
                "framing": {
                    "max_header_bytes": extractor.MAX_HEADER_BYTES,
                    "max_payload_bytes": extractor.MAX_PAYLOAD_BYTES,
                },
            },
        )
        write_json(self.identity_contract, {"schema_version": extractor.IDENTITY_CONTRACT_SCHEMA, "fixture": True})
        self.production_sha = digest_file(self.production_contract)[1]
        self.payload_sha = digest_file(self.payload_contract)[1]
        self.identity_sha = digest_file(self.identity_contract)[1]
        self.source_contract_sha = digest_bytes(b"source-contract")
        self.selected_sha = digest_bytes(b"ordinals-0-2")
        components = {
            "production_contract": self.production_sha,
            "payload_contract": self.payload_sha,
            "identity_contract": self.identity_sha,
            "fixture_component": digest_bytes(b"fixture-component"),
        }
        self.harness = {"components": components, "bundle_sha256": digest_json(components)}
        self.release_id = "pcqm-production-v2-fixture"
        self.rdkit_version = "2024.03.5"

        strict = digest_bytes(b"strict-0")
        official_strict = digest_bytes(b"different-strict") if strict_hash_mismatch else strict
        identity_spec = digest_bytes(b"wrong-identity-spec") if identity_spec_mismatch else self.identity_sha
        source_address_0 = digest_bytes(b"source-address-0")
        record = {
            "record_schema_version": extractor.PRODUCTION_RECORD_SCHEMA,
            "sidecar": {
                "sidecar_id": self.release_id,
                "sidecar_mode": extractor.PRODUCTION_MODE,
                "selected_ordinal_set_sha256": self.selected_sha,
                "source_contract_sha256": self.source_contract_sha,
                "identity_normalization_contract_sha256": self.identity_sha,
                "adapter_harness_sha256": self.harness["bundle_sha256"],
                "record_schema_sha256": self.production_sha,
                "geometry_only_pretokenizer": True,
                "p1_training_admission": False,
                "p1_training_launcher_permitted": False,
            },
            "member": {
                "identity_namespace": extractor.IDENTITY_NAMESPACE,
                "member_id": extractor.member_id(0),
                "sdf_record_index": 0,
                "official_csv_row_index": 0,
                "storage_key": extractor.storage_key(0),
                "source_archive_sha256": digest_bytes(b"archive"),
                "source_address_sha256": source_address_0,
                "source_mol_identity_sha256": digest_bytes(b"source-mol-0"),
            },
            "identity": {
                "official_identity_status": "strict_isomeric_match",
                "sdf_strict_smiles_sha256": strict,
                "official_strict_smiles_sha256": official_strict,
                "canonical_connectivity_sha256": digest_bytes(b"connectivity-0"),
                "identity_spec_sha256": identity_spec,
                "rdkit_version": self.rdkit_version,
            },
            "atom_universe": {},
            "topology": {},
            "geometry": {},
            "array_metadata": {},
        }
        fixture_array = numpy.asarray([3, 7], dtype=numpy.int32)
        fixture_array_raw = fixture_array.tobytes(order="C")
        fixture_descriptor = {
            "dtype": "int32",
            "shape": [2],
            "order": "C",
            "sha256": digest_bytes(fixture_array_raw),
        }
        record["atom_universe"]["fixture_array"] = {
            "__array_block__": 0,
            "dtype": fixture_descriptor["dtype"],
            "shape": fixture_descriptor["shape"],
            "order": fixture_descriptor["order"],
            "sha256": fixture_descriptor["sha256"],
        }
        logical_projection = json.loads(json.dumps(record))
        logical_projection["atom_universe"]["fixture_array"] = {"__ndarray__": fixture_descriptor}
        logical_hash = digest_json(logical_projection)
        header_logical_hash = digest_bytes(b"incorrect-logical-record") if logical_hash_mismatch else logical_hash
        header = {
            "payload_schema_version": extractor.PAYLOAD_SCHEMA,
            "record": record,
            "array_blocks": [
                {
                    "index": 0,
                    "dtype": "int32",
                    "shape": [2],
                    "order": "C",
                    "offset": 0,
                    "nbytes": len(fixture_array_raw),
                    "sha256": digest_bytes(fixture_array_raw),
                }
            ],
            "logical_record_sha256": header_logical_hash,
        }
        header_raw = canonical(header)
        payload = extractor.MAGIC + struct.pack(">I", len(header_raw)) + header_raw + fixture_array_raw
        membership = [
            {
                "record_schema_version": extractor.PRODUCTION_RECORD_SCHEMA,
                "sidecar_id": self.release_id,
                "sidecar_mode": extractor.PRODUCTION_MODE,
                "selected_ordinal_set_sha256": self.selected_sha,
                "member_id": extractor.member_id(0),
                "sdf_record_index": 0,
                "official_csv_row_index": 0,
                "source_address_sha256": source_address_0,
                "disposition": "admit",
                "record_storage_key": extractor.storage_key(0),
                "record_content_sha256": header_logical_hash,
                "reject_reason_code": None,
            },
            {
                "record_schema_version": extractor.PRODUCTION_RECORD_SCHEMA,
                "sidecar_id": self.release_id,
                "sidecar_mode": extractor.PRODUCTION_MODE,
                "selected_ordinal_set_sha256": self.selected_sha,
                "member_id": extractor.member_id(1),
                "sdf_record_index": 1,
                "official_csv_row_index": 1,
                "source_address_sha256": digest_bytes(b"source-address-1"),
                "disposition": "reject",
                "record_storage_key": None,
                "record_content_sha256": None,
                "reject_reason_code": "PCQM_STEREO_2D3D_DIVERGENCE",
            },
        ]
        reject_detail = {
            "diagnostic_code": "strict_mismatch_connectivity_match",
            "reason_code": "PCQM_STEREO_2D3D_DIVERGENCE",
            "source_address_sha256": membership[1]["source_address_sha256"],
            "stage": "identity",
        }
        rejects = [
            {
                "record_schema_version": extractor.PRODUCTION_RECORD_SCHEMA,
                "sidecar_id": self.release_id,
                "sidecar_mode": extractor.PRODUCTION_MODE,
                "selected_ordinal_set_sha256": self.selected_sha,
                "member_id": extractor.member_id(1),
                "sdf_record_index": 1,
                "official_csv_row_index": 1,
                "source_address_sha256": membership[1]["source_address_sha256"],
                "stage": "identity",
                "reason_code": "PCQM_STEREO_2D3D_DIVERGENCE",
                "action": "exclude_from_geometry_release",
                "geometry_mse_enabled": False,
                "source_mol_identity_sha256": digest_bytes(b"source-mol-1"),
                "geometry_mol_identity_sha256": None,
                "diagnostic_code": "strict_mismatch_connectivity_match",
                "detail_sha256": digest_json(reject_detail),
            }
        ]
        payload_index = [
            {
                "payload_index_schema_version": extractor.PAYLOAD_INDEX_SCHEMA,
                "record_storage_key": extractor.storage_key(0),
                "record_wire_bytes": len(payload),
                "record_wire_sha256": digest_bytes(payload),
                "record_content_sha256": header_logical_hash,
            }
        ]
        write_jsonl(self.shard_dir / "membership.jsonl", membership)
        write_jsonl(self.shard_dir / "reject_ledger.jsonl", rejects)
        write_jsonl(self.shard_dir / "payload_index.jsonl", payload_index)
        write_jsonl(self.shard_dir / "motif_census.jsonl", [])
        write_jsonl(self.release_root / "motif_census.jsonl", [])
        (self.lmdb_dir / "data.mdb").write_bytes(b"hermetic-lmdb-data-envelope")
        (self.lmdb_dir / "lock.mdb").write_bytes(b"hermetic-lmdb-runtime-lock")
        artifacts = {
            "geometry_records_lmdb_data": artifact(self.lmdb_dir / "data.mdb", "geometry_records.lmdb/data.mdb"),
            "membership": artifact(self.shard_dir / "membership.jsonl", "membership.jsonl"),
            "reject_ledger": artifact(self.shard_dir / "reject_ledger.jsonl", "reject_ledger.jsonl"),
            "payload_index": artifact(self.shard_dir / "payload_index.jsonl", "payload_index.jsonl"),
            "motif_census": artifact(self.shard_dir / "motif_census.jsonl", "motif_census.jsonl"),
        }
        shard_manifest = {
            "schema_version": extractor.SHARD_MANIFEST_SCHEMA,
            "created_utc": "2026-08-05T00:00:00+00:00",
            "release_status": "complete",
            "release_id": self.release_id,
            "production_contract_sha256": self.production_sha,
            "shard_index": 0,
            "range_start": 0,
            "range_end": 2,
            "selected_record_count": 2,
            "counts": {
                "membership_record_count": 2,
                "admitted_record_count": 1,
                "reject_ledger_record_count": 1,
                "payload_index_record_count": 1,
                "payload_wire_total_bytes": len(payload),
                "motif_occurrence_count": 0,
                "unique_motif_count": 0,
            },
            "reject_reason_counts": {"PCQM_STEREO_2D3D_DIVERGENCE": 1},
            "e3fp_params_sha256_values": [],
            "artifacts": artifacts,
            "partition_invariant_pass": True,
            "lmdb_merged": False,
            "p1_training_admission": False,
        }
        write_json(self.shard_dir / "shard_manifest.json", shard_manifest)
        shard_manifest_sha = digest_file(self.shard_dir / "shard_manifest.json")[1]
        self.configuration = {
            "release_id": self.release_id,
            "production_contract_sha256": self.production_sha,
            "runtime_attestation_sha256": digest_bytes(b"runtime"),
            "staged_input_receipt_sha256": digest_bytes(b"staging"),
            "source_contract_sha256": self.source_contract_sha,
            "release_kind": "full_production",
            "source_record_count": 2,
            "selected_record_count": 2,
            "selected_ordinal_range": [0, 2],
            "selected_ordinal_set_sha256": self.selected_sha,
            "shard_size": 2,
            "shard_count": 1,
            "staged_inputs": {},
            "locked_sdf_member": {},
            "harness": self.harness,
            "logical_record_schema_version": extractor.PRODUCTION_RECORD_SCHEMA,
            "sidecar_mode": extractor.PRODUCTION_MODE,
        }
        shard_roots = [{"shard_index": 0, "range_start": 0, "range_end": 2, "shard_manifest_sha256": shard_manifest_sha}]
        logical_root = digest_json(
            {
                "configuration": self.configuration,
                "global_motif_census_sha256": digest_file(self.release_root / "motif_census.jsonl")[1],
                "shards": shard_roots,
                "membership_record_count": 2,
                "admitted_record_count": 1,
                "reject_ledger_record_count": 1,
            }
        )
        top = {
            "schema_version": extractor.FULL_RELEASE_SCHEMA,
            "created_utc": "2026-08-05T00:00:00+00:00",
            "release_status": "complete",
            "release_id": self.release_id,
            "logical_release_root_sha256": logical_root,
            "configuration": self.configuration,
            "counts": {
                "source_record_count": 2,
                "membership_record_count": 2,
                "admitted_record_count": 1,
                "reject_ledger_record_count": 1,
                "shard_count": 1,
                "unique_motif_count": 0,
                "motif_occurrence_count": 0,
            },
            "global_motif_census": artifact(self.release_root / "motif_census.jsonl", "motif_census.jsonl"),
            "shards": shard_roots,
            "range_no_gap_no_overlap": True,
            "lmdb_merged": False,
            "tokenizer_binding": "absent_and_forbidden",
            "p1_training_admission": False,
            "p1_training_launcher_permitted": False,
            "next_gate": "fixture",
        }
        write_json(self.release_root / "full_release_manifest.json", top)
        top_bytes, top_sha = digest_file(self.release_root / "full_release_manifest.json")
        self.config = self.root / "config.json"
        write_json(
            self.config,
            {
                "schema_version": extractor.CONFIG_SCHEMA,
                "extraction_id": "fixture-extraction",
                "contract_sha256": digest_file(self.contract)[1],
                "collection": {
                    "collection_id": "pcqm-fixture-p1",
                    "dataset_id": "pcqm4mv2",
                    "release_id": self.release_id,
                    "phase": "p1",
                    "split": "train",
                    "role": "p1_structure_train",
                    "task_family": "none",
                    "source_identity_namespace": extractor.IDENTITY_NAMESPACE,
                },
                "release_lock": {
                    "release_manifest_relative_path": "full_release_manifest.json",
                    "expected_release_manifest_bytes": top_bytes,
                    "expected_release_manifest_sha256": top_sha,
                    "expected_release_id": self.release_id,
                    "expected_production_contract_sha256": self.production_sha,
                    "expected_payload_contract_sha256": self.payload_sha,
                    "expected_identity_normalization_contract_sha256": self.identity_sha,
                    "required_rdkit_version": self.rdkit_version,
                },
            },
        )
        self.registry = {str(self.lmdb_dir.resolve()): {extractor.storage_key(0).encode("ascii"): payload}}

    def run(self):
        fake = fake_lmdb_module(self.registry)
        with mock.patch.dict(sys.modules, {"lmdb": fake}):
            return extractor.extract_collection(
                self.contract, self.config, self.release_root,
                self.production_contract, self.payload_contract,
                self.identity_contract, self.output,
            )


class PcqmProductionV2IdentityExtractorTests(unittest.TestCase):
    def test_happy_path_filters_reject_and_is_proof_gate_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            receipt = fixture.run()
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(receipt["counts"]["source_membership_rows"], 2)
            self.assertEqual(receipt["counts"]["rejected_members_filtered"], 1)
            self.assertEqual(receipt["counts"]["emitted_molecule_rows"], 1)
            self.assertFalse(receipt["p1_training_admission"])
            rows = (fixture.output / "molecule_identity_rows.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertNotIn("smiles", rows[0].lower())
            manifest_path = fixture.output / "collection_manifest.json"
            connection = proof.create_database(":memory:")
            try:
                collection, observation = proof.load_collection(
                    connection, manifest_path, digest_file(manifest_path)[1]
                )
                self.assertEqual(collection["role"], "p1_structure_train")
                self.assertEqual(observation["molecule_rows"]["row_count"], 1)
            finally:
                connection.close()

    def test_undeclared_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            (fixture.shard_dir / "undeclared.bin").write_bytes(b"not in shard manifest")
            with self.assertRaisesRegex(RuntimeError, "undeclared artifact"):
                fixture.run()

    def test_identity_spec_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp, identity_spec_mismatch=True)
            with self.assertRaisesRegex(RuntimeError, "identity spec/hash/version mismatch"):
                fixture.run()

    def test_strict_identity_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp, strict_hash_mismatch=True)
            with self.assertRaisesRegex(RuntimeError, "identity spec/hash/version mismatch"):
                fixture.run()

    def test_lmdb_record_logical_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp, logical_hash_mismatch=True)
            with self.assertRaisesRegex(RuntimeError, "LMDB record logical hash mismatch"):
                fixture.run()

    def test_undeclared_lmdb_metadata_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(temp)
            fixture.registry[str(fixture.lmdb_dir.resolve())][b"__len__"] = b"2"
            with self.assertRaisesRegex(RuntimeError, "excess rows, keys, or metadata"):
                fixture.run()


if __name__ == "__main__":
    unittest.main()
