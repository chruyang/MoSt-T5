"""Hermetic tests for the versioned full PCQM geometry production runner.

No molecular dataset, E3FP runtime, LMDB package, or remote connection is
required.  These tests protect deterministic scheduling, range/restart
semantics, staged-receipt closure, and immutable shard discovery.
"""

from __future__ import print_function

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from most_t5_next.r1.adapter import build_pcqm_p1_geometry_production_v1 as runner


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "p1_pcqm_geometry_production_release_contract.json"


class GeometryProductionRunnerTest(unittest.TestCase):
    def test_write_json_new_atomically_publishes_complete_bytes(self):
        root = Path(tempfile.mkdtemp()).resolve()
        target = root / "immutable.json"
        value = {"z": [1, 2], "a": "complete"}
        expected = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        original_link = os.link
        observed_at_publish = []

        def checked_link(source, destination, **kwargs):
            self.assertFalse(Path(destination).exists())
            observed_at_publish.append(Path(source).read_bytes())
            return original_link(source, destination, **kwargs)

        try:
            with mock.patch.object(runner.os, "link", side_effect=checked_link):
                runner.write_json_new(target, value)
            self.assertEqual(observed_at_publish, [expected])
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(list(root.glob(".write-json.*.tmp")), [])
            with self.assertRaises(FileExistsError):
                runner.write_json_new(target, {"replacement": True})
            self.assertEqual(target.read_bytes(), expected)
        finally:
            if target.exists():
                target.unlink()
            root.rmdir()

    def test_write_json_new_preserves_racing_winner_and_cleans_its_temp(self):
        root = Path(tempfile.mkdtemp()).resolve()
        target = root / "immutable.json"
        winner = b"racing-winner\n"
        original_link = os.link

        def racing_link(source, destination, **kwargs):
            Path(destination).write_bytes(winner)
            return original_link(source, destination, **kwargs)

        try:
            with mock.patch.object(runner.os, "link", side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    runner.write_json_new(target, {"loser": True})
            self.assertEqual(target.read_bytes(), winner)
            self.assertEqual(list(root.glob(".write-json.*.tmp")), [])
        finally:
            if target.exists():
                target.unlink()
            root.rmdir()

    def test_streamed_ordinal_hash_matches_canonical_json(self):
        for end in (0, 1, 3, 128):
            self.assertEqual(
                runner.sha256_ordinal_range(0, end),
                runner.sha256_json(list(range(end))),
            )

    def test_serial_and_process_pool_results_are_identical_and_ordered(self):
        values = list(range(257))
        serial = list(
            runner.ordered_bounded_map(
                runner._synthetic_transform, values, workers=1, max_pending=1
            )
        )
        parallel = list(
            runner.ordered_bounded_map(
                runner._synthetic_transform, values, workers=2, max_pending=7
            )
        )
        self.assertEqual(serial, parallel)
        self.assertEqual([row["ordinal"] for row in parallel], values)

    def test_planner_and_range_gate_reject_gap_and_overlap(self):
        plans = runner.plan_shards(11, 5)
        manifests = []
        for plan in plans:
            selected = plan["range_end"] - plan["range_start"]
            manifests.append(
                {
                    "schema_version": runner.SHARD_MANIFEST_SCHEMA,
                    "release_status": "complete",
                    "shard_index": plan["shard_index"],
                    "range_start": plan["range_start"],
                    "range_end": plan["range_end"],
                    "selected_record_count": selected,
                    "counts": {
                        "membership_record_count": selected,
                        "admitted_record_count": selected - 1,
                        "reject_ledger_record_count": 1,
                    },
                }
            )
        self.assertEqual(runner.validate_contiguous_ranges(manifests, 11), 11)
        gap = json.loads(json.dumps(manifests))
        gap[1]["range_start"] += 1
        gap[1]["selected_record_count"] -= 1
        gap[1]["counts"]["membership_record_count"] -= 1
        gap[1]["counts"]["admitted_record_count"] -= 1
        with self.assertRaises(RuntimeError):
            runner.validate_contiguous_ranges(gap, 11)
        overlap = json.loads(json.dumps(manifests))
        overlap[1]["range_start"] -= 1
        overlap[1]["selected_record_count"] += 1
        overlap[1]["counts"]["membership_record_count"] += 1
        overlap[1]["counts"]["admitted_record_count"] += 1
        with self.assertRaises(RuntimeError):
            runner.validate_contiguous_ranges(overlap, 11)

    def test_contract_is_versioned_full_and_non_admissible(self):
        with open(str(CONTRACT_PATH), "r", encoding="utf-8") as handle:
            contract = json.load(handle)
        runner.validate_production_contract(contract, runner.DEFAULT_SHARD_SIZE)
        with self.assertRaises(RuntimeError):
            runner.validate_production_contract(contract, 1000)
        self.assertFalse(contract["logical_record"]["p1_training_admission"])
        self.assertFalse(contract["execution"]["lmdb_merge_permitted"])
        self.assertEqual(contract["schema_version"], runner.CONTRACT_SCHEMA)
        self.assertTrue(
            contract["logical_record"]["motif_lexeme_binding"]["ordered_digest_field_required"]
        )

    def test_motif_content_addresses_preserve_order_and_fail_on_collision(self):
        fragments = ("C", "O", "C")
        bindings = runner.motif_bindings(fragments)
        self.assertEqual([fragment for _, fragment in bindings], list(fragments))
        self.assertEqual(bindings[0][0], bindings[2][0])
        self.assertNotEqual(bindings[0][0], bindings[1][0])
        self.assertEqual(bindings[0][0], runner.sha256_bytes(b"C"))

        counts = Counter()
        lexemes = {}
        with mock.patch.object(runner, "motif_lexeme_sha256", return_value="a" * 64):
            runner._register_motif_binding(counts, lexemes, "a" * 64, "C")
            with self.assertRaises(RuntimeError):
                runner._register_motif_binding(counts, lexemes, "a" * 64, "O")

    def test_production_record_keeps_only_ordered_digest_sequence(self):
        class FakeBuilder(object):
            RECORD_SCHEMA = "legacy-record"
            SIDE_CAR_MODE = "legacy-mode"

            @staticmethod
            def validate_admitted_record(_np, observed):
                self.assertEqual(observed["record_schema_version"], "legacy-record")
                self.assertEqual(observed["sidecar"]["sidecar_mode"], "legacy-mode")
                self.assertNotIn("motif_lexeme_sha256", observed["topology"])

        digests = [runner.sha256_bytes(b"C"), runner.sha256_bytes(b"O")]
        record = {
            "record_schema_version": runner.PRODUCTION_RECORD_SCHEMA,
            "sidecar": {"sidecar_mode": runner.PRODUCTION_MODE, "p1_training_admission": False},
            "topology": {
                "linearizer_spec_sha256": "b" * 64,
                "motif_count": 2,
                "motif_atom_indices": [object(), object()],
                "motif_atom_indices_sha256": "c" * 64,
                "motif_lexeme_sha256": list(digests),
            },
        }
        runner._validate_production_record(FakeBuilder(), object(), record)
        self.assertEqual(record["topology"]["motif_lexeme_sha256"], digests)
        self.assertNotIn("motif_fragment_sequence", record["topology"])

    def test_cli_consumes_unified_staging_gate_and_explicit_benchmark(self):
        args = runner.parse_args(
            [
                "--staging-contract", "staging-contract.json",
                "--source-contract", "source-contract.json",
                "--staging-receipt", "staging-receipt.json",
                "--runtime-attestation", "runtime.json",
                "--runtime-attestation-contract", "runtime-contract.json",
                "--production-contract", "production-contract.json",
                "--identity-normalization-contract", "identity.json",
                "--payload-format-contract", "payload.json",
                "--e3fp-source", "e3fp",
                "--output-dir", "out",
                "--benchmark-records", "128",
            ]
        )
        self.assertEqual(args.benchmark_records, 128)
        self.assertFalse(hasattr(args, "train_row_map"))
        self.assertFalse(hasattr(args, "archive"))

    def test_live_bundle_is_rehashed_against_runtime_attestation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = root / "runner.py"
            second = root / "contracts" / "release.json"
            second.parent.mkdir()
            first.write_bytes(b"runner-v2")
            second.write_bytes(b"contract-v2")
            rows = []
            for target in (first, second):
                relative = target.relative_to(root).as_posix()
                payload = target.read_bytes()
                rows.append(
                    {
                        "relative_path": relative,
                        "bytes": len(payload),
                        "sha256": runner.sha256_bytes(payload),
                    }
                )
            report = {
                "bundle_file_lock": {
                    "bundle_root": str(root),
                    "files": sorted(rows, key=lambda item: item["relative_path"]),
                }
            }
            report_path = root / "runtime.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            observed = runner.verify_attested_bundle_files(
                report_path, root, (first, second)
            )
            self.assertEqual(set(observed), {"runner.py", "contracts/release.json"})
            second.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                runner.verify_attested_bundle_files(report_path, root, (first, second))

    def test_completed_shard_discovery_ignores_partial_and_rehashes_artifacts(self):
        release_id = "fixture-release"
        contract_sha = "d" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shard-000001.partial-attempt-000001").mkdir()
            shard = root / "shard-000000"
            (shard / "geometry_records.lmdb").mkdir(parents=True)
            files = {
                "geometry_records_lmdb_data": ("geometry_records.lmdb/data.mdb", b"lmdb"),
                "membership": ("membership.jsonl", b"{}\n"),
                "reject_ledger": ("reject_ledger.jsonl", b""),
                "payload_index": ("payload_index.jsonl", b"{}\n"),
                "motif_census": (
                    "motif_census.jsonl",
                    (
                        '{"count":1,"motif_fragment":"C","motif_lexeme_sha256":"'
                        + runner.sha256_bytes(b"C")
                        + '"}\n'
                    ).encode("utf-8"),
                ),
            }
            artifacts = {}
            for name, (relative, payload) in files.items():
                target = shard / Path(relative)
                target.write_bytes(payload)
                artifacts[name] = {
                    "relative_path": relative,
                    "bytes": len(payload),
                    "sha256": runner.sha256_bytes(payload),
                }
            manifest = {
                "schema_version": runner.SHARD_MANIFEST_SCHEMA,
                "release_status": "complete",
                "release_id": release_id,
                "production_contract_sha256": contract_sha,
                "shard_index": 0,
                "range_start": 0,
                "range_end": 1,
                "selected_record_count": 1,
                "counts": {
                    "membership_record_count": 1,
                    "admitted_record_count": 1,
                    "reject_ledger_record_count": 0,
                },
                "artifacts": artifacts,
            }
            (shard / "shard_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            observed = runner.discover_completed_shards(
                root, release_id, contract_sha, rehash=True
            )
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["range_end"], 1)
            (shard / "membership.jsonl").write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):
                runner.discover_completed_shards(root, release_id, contract_sha, rehash=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
