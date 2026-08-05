import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from most_t5_next.r1.overlap import extract_pcqm_production_v2_identity_collection_parallel_v1 as parallel
from most_t5_next.r1.overlap import extract_pcqm_production_v2_identity_collection_v1 as serial
from most_t5_next.r1.overlap.tests.test_extract_pcqm_production_v2_identity_collection_v1 import (
    Fixture,
    artifact,
    digest_file,
    digest_json,
    fake_lmdb_module,
    write_json,
    write_jsonl,
)


FAKE_LMDB_SOURCE = r'''
import base64
import json
from pathlib import Path

__version__ = "hermetic-disk-fake-1"


class Transaction(object):
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return iter(sorted(self.records.items()))


class Environment(object):
    def __init__(self, path):
        value = json.loads((Path(path) / "data.mdb").read_bytes())
        self.records = {
            key.encode("ascii"): base64.b64decode(payload.encode("ascii"))
            for key, payload in value.items()
        }

    def begin(self, write=False):
        if write:
            raise AssertionError("parallel extractor attempted a write transaction")
        return Transaction(self.records)

    def close(self):
        return None


def open(path, **options):
    assert options["readonly"] is True
    assert options["lock"] is False
    assert options["create"] is False
    return Environment(path)
'''


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_fake_data(path, records):
    value = {
        key.decode("ascii"): base64.b64encode(payload).decode("ascii")
        for key, payload in sorted(records.items())
    }
    Path(path).write_bytes(serial.canonical_json_bytes(value))


def release_snapshot(root):
    rows = {}
    for path in sorted(Path(root).rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            rows[path.relative_to(root).as_posix()] = digest_file(path)
    return rows


def make_shard_manifest(fixture, shard_index, start, end, payload_wire_bytes, reason_counts):
    shard_dir = fixture.release_root / "shard-{:06d}".format(shard_index)
    artifacts = {
        "geometry_records_lmdb_data": artifact(
            shard_dir / "geometry_records.lmdb" / "data.mdb", "geometry_records.lmdb/data.mdb"
        ),
        "membership": artifact(shard_dir / "membership.jsonl", "membership.jsonl"),
        "reject_ledger": artifact(shard_dir / "reject_ledger.jsonl", "reject_ledger.jsonl"),
        "payload_index": artifact(shard_dir / "payload_index.jsonl", "payload_index.jsonl"),
        "motif_census": artifact(shard_dir / "motif_census.jsonl", "motif_census.jsonl"),
    }
    admitted = 1 if payload_wire_bytes else 0
    rejected = (end - start) - admitted
    manifest = {
        "schema_version": serial.SHARD_MANIFEST_SCHEMA,
        "created_utc": "2026-08-05T00:00:00+00:00",
        "release_status": "complete",
        "release_id": fixture.release_id,
        "production_contract_sha256": fixture.production_sha,
        "shard_index": shard_index,
        "range_start": start,
        "range_end": end,
        "selected_record_count": end - start,
        "counts": {
            "membership_record_count": end - start,
            "admitted_record_count": admitted,
            "reject_ledger_record_count": rejected,
            "payload_index_record_count": admitted,
            "payload_wire_total_bytes": payload_wire_bytes,
            "motif_occurrence_count": 0,
            "unique_motif_count": 0,
        },
        "reject_reason_counts": reason_counts,
        "e3fp_params_sha256_values": [],
        "artifacts": artifacts,
        "partition_invariant_pass": True,
        "lmdb_merged": False,
        "p1_training_admission": False,
    }
    write_json(shard_dir / "shard_manifest.json", manifest)
    return {
        "shard_index": shard_index,
        "range_start": start,
        "range_end": end,
        "shard_manifest_sha256": digest_file(shard_dir / "shard_manifest.json")[1],
    }


def make_two_shard_fixture(root):
    fixture = Fixture(root)
    membership = read_jsonl(fixture.shard_dir / "membership.jsonl")
    rejects = read_jsonl(fixture.shard_dir / "reject_ledger.jsonl")
    payload_index = read_jsonl(fixture.shard_dir / "payload_index.jsonl")
    payload = fixture.registry[str(fixture.lmdb_dir.resolve())][serial.storage_key(0).encode("ascii")]

    shard_one = fixture.release_root / "shard-000001"
    shard_one_lmdb = shard_one / "geometry_records.lmdb"
    shard_one.mkdir()
    shard_one_lmdb.mkdir()

    write_jsonl(fixture.shard_dir / "membership.jsonl", [membership[0]])
    write_jsonl(fixture.shard_dir / "reject_ledger.jsonl", [])
    write_jsonl(fixture.shard_dir / "payload_index.jsonl", payload_index)
    write_jsonl(fixture.shard_dir / "motif_census.jsonl", [])
    write_fake_data(
        fixture.lmdb_dir / "data.mdb", {serial.storage_key(0).encode("ascii"): payload}
    )

    write_jsonl(shard_one / "membership.jsonl", [membership[1]])
    write_jsonl(shard_one / "reject_ledger.jsonl", rejects)
    write_jsonl(shard_one / "payload_index.jsonl", [])
    write_jsonl(shard_one / "motif_census.jsonl", [])
    write_fake_data(shard_one_lmdb / "data.mdb", {})
    (shard_one_lmdb / "lock.mdb").write_bytes(b"hermetic-lmdb-runtime-lock")

    shard_roots = [
        make_shard_manifest(fixture, 0, 0, 1, len(payload), {}),
        make_shard_manifest(
            fixture, 1, 1, 2, 0, {"PCQM_STEREO_2D3D_DIVERGENCE": 1}
        ),
    ]
    fixture.configuration["shard_size"] = 1
    fixture.configuration["shard_count"] = 2
    logical_root = digest_json(
        {
            "configuration": fixture.configuration,
            "global_motif_census_sha256": digest_file(fixture.release_root / "motif_census.jsonl")[1],
            "shards": shard_roots,
            "membership_record_count": 2,
            "admitted_record_count": 1,
            "reject_ledger_record_count": 1,
        }
    )
    top = {
        "schema_version": serial.FULL_RELEASE_SCHEMA,
        "created_utc": "2026-08-05T00:00:00+00:00",
        "release_status": "complete",
        "release_id": fixture.release_id,
        "logical_release_root_sha256": logical_root,
        "configuration": fixture.configuration,
        "counts": {
            "source_record_count": 2,
            "membership_record_count": 2,
            "admitted_record_count": 1,
            "reject_ledger_record_count": 1,
            "shard_count": 2,
            "unique_motif_count": 0,
            "motif_occurrence_count": 0,
        },
        "global_motif_census": artifact(
            fixture.release_root / "motif_census.jsonl", "motif_census.jsonl"
        ),
        "shards": shard_roots,
        "range_no_gap_no_overlap": True,
        "lmdb_merged": False,
        "tokenizer_binding": "absent_and_forbidden",
        "p1_training_admission": False,
        "p1_training_launcher_permitted": False,
        "next_gate": "fixture",
    }
    write_json(fixture.release_root / "full_release_manifest.json", top)
    top_bytes, top_sha = digest_file(fixture.release_root / "full_release_manifest.json")

    serial_config = json.loads(fixture.config.read_text(encoding="utf-8"))
    serial_config["release_lock"]["expected_release_manifest_bytes"] = top_bytes
    serial_config["release_lock"]["expected_release_manifest_sha256"] = top_sha
    write_json(fixture.config, serial_config)

    fixture.parallel_contract = (
        Path(parallel.__file__).resolve().parents[1]
        / "contracts"
        / "pcqm_production_v2_identity_parallel_extraction_contract_v1.json"
    )
    fixture.parallel_config = Path(root) / "parallel_config.json"
    parallel_config = json.loads(json.dumps(serial_config))
    parallel_config["extraction_id"] = "fixture-parallel-extraction"
    parallel_config["contract_sha256"] = digest_file(fixture.parallel_contract)[1]
    write_json(fixture.parallel_config, parallel_config)
    fixture.registry = {
        str(fixture.lmdb_dir.resolve()): {serial.storage_key(0).encode("ascii"): payload},
        str(shard_one_lmdb.resolve()): {},
    }
    return fixture


def install_fake_lmdb(root):
    module_dir = Path(root) / "fake_modules"
    module_dir.mkdir()
    (module_dir / "lmdb.py").write_text(FAKE_LMDB_SOURCE, encoding="utf-8")
    sys.path.insert(0, str(module_dir))
    return module_dir


def run_parallel(fixture, output, scratch, processes):
    return parallel.extract_collection_parallel(
        fixture.parallel_contract,
        fixture.parallel_config,
        fixture.release_root,
        fixture.production_contract,
        fixture.payload_contract,
        fixture.identity_contract,
        output,
        scratch,
        processes,
    )


class PcqmProductionV2ParallelIdentityExtractorTests(unittest.TestCase):
    def new_root(self):
        # Parallel scratch is intentionally retained as diagnostic evidence.
        return Path(tempfile.mkdtemp(prefix="most_t5_parallel_identity_test_"))

    def test_serial_one_process_and_two_process_core_are_byte_identical(self):
        root = self.new_root()
        fixture = make_two_shard_fixture(root)
        before = release_snapshot(fixture.release_root)
        module_dir = install_fake_lmdb(root)
        serial_output = root / "serial_output"
        one_output, one_scratch = root / "one_output", root / "one_scratch"
        two_output, two_scratch = root / "two_output", root / "two_scratch"
        try:
            with mock.patch.dict(sys.modules, {"lmdb": fake_lmdb_module(fixture.registry)}):
                serial.extract_collection(
                    fixture.contract, fixture.config, fixture.release_root,
                    fixture.production_contract, fixture.payload_contract,
                    fixture.identity_contract, serial_output,
                )
            one = run_parallel(fixture, one_output, one_scratch, 1)
            two = run_parallel(fixture, two_output, two_scratch, 2)
        finally:
            sys.path.remove(str(module_dir))

        serial_bytes = (serial_output / "molecule_identity_rows.jsonl").read_bytes()
        one_bytes = (one_output / "molecule_identity_rows.jsonl").read_bytes()
        two_bytes = (two_output / "molecule_identity_rows.jsonl").read_bytes()
        self.assertEqual(serial_bytes, one_bytes)
        self.assertEqual(one_bytes, two_bytes)
        self.assertEqual(one["artifacts"]["molecule_rows"], two["artifacts"]["molecule_rows"])
        self.assertEqual(one["counts"]["rejected_members_filtered"], 1)
        self.assertEqual(two["counts"]["emitted_molecule_rows"], 1)
        self.assertTrue((one_scratch / ".pcqm_identity_sort.sqlite3").is_file())
        self.assertTrue((two_scratch / "scratch_manifest.json").is_file())
        self.assertEqual(release_snapshot(fixture.release_root), before)

    def test_worker_exception_fails_without_publishing_output(self):
        root = self.new_root()
        fixture = make_two_shard_fixture(root)
        module_dir = install_fake_lmdb(root)
        output, scratch = root / "failed_output", root / "failed_scratch"
        with open(str(fixture.release_root / "shard-000001" / "membership.jsonl"), "ab") as handle:
            handle.write(b"{}\n")
        try:
            with self.assertRaisesRegex(RuntimeError, "parallel shard worker"):
                run_parallel(fixture, output, scratch, 2)
        finally:
            sys.path.remove(str(module_dir))
        self.assertFalse(output.exists())
        self.assertTrue(scratch.is_dir())

    def test_duplicate_shard_declaration_fails_before_scratch_creation(self):
        root = self.new_root()
        fixture = make_two_shard_fixture(root)
        top_path = fixture.release_root / "full_release_manifest.json"
        top = json.loads(top_path.read_text(encoding="utf-8"))
        top["shards"][1]["shard_index"] = 0
        write_json(top_path, top)
        top_bytes, top_sha = digest_file(top_path)
        config = json.loads(fixture.parallel_config.read_text(encoding="utf-8"))
        config["release_lock"]["expected_release_manifest_bytes"] = top_bytes
        config["release_lock"]["expected_release_manifest_sha256"] = top_sha
        write_json(fixture.parallel_config, config)
        output, scratch = root / "duplicate_output", root / "duplicate_scratch"
        with self.assertRaisesRegex(RuntimeError, "indices are not contiguous"):
            run_parallel(fixture, output, scratch, 2)
        self.assertFalse(output.exists())
        self.assertFalse(scratch.exists())

    def test_scratch_must_be_new_and_disjoint_from_release(self):
        root = self.new_root()
        fixture = make_two_shard_fixture(root)
        existing = root / "existing_scratch"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            run_parallel(fixture, root / "output_existing", existing, 1)
        with self.assertRaisesRegex(ValueError, "scratch directory and immutable release root"):
            run_parallel(
                fixture,
                root / "output_inside",
                fixture.release_root / "forbidden_scratch",
                1,
            )
        self.assertFalse((fixture.release_root / "forbidden_scratch").exists())


if __name__ == "__main__":
    unittest.main()
