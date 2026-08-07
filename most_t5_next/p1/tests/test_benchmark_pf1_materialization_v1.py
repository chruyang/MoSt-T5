from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import benchmark_pf1_materialization_v1 as benchmark
from most_t5_next.p1 import freeze_pf1_connectivity_sample_v1 as selection


class PF1MaterializationBenchmarkTests(unittest.TestCase):
    def _row(self, index: int, ordinal: int) -> dict:
        return {
            "schema_version": selection.BENCHMARK_SCHEMA,
            "benchmark_index": index,
            "pf1_selection_index": index + 3,
            "group_order_index": index,
            "member_id": f"{selection.MEMBER_PREFIX}{ordinal}",
            "sdf_record_index": ordinal,
            "connectivity_identity_sha256": f"group-{index}",
            "split": "train" if index == 0 else "dev",
        }

    def test_load_benchmark_membership_preserves_frozen_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            rows = [self._row(0, 5), self._row(1, 19)]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                benchmark.load_benchmark_membership(path), tuple(rows)
            )

    def test_load_rejects_non_dense_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(json.dumps(self._row(1, 5)) + "\n", encoding="utf-8")
            with self.assertRaises(benchmark.PF1MaterializationBenchmarkError):
                benchmark.load_benchmark_membership(path)

    def test_summary_reports_phase_cost_and_same_worker_projection(self):
        results = [
            {
                "benchmark_index": 0,
                "sdf_record_index": 5,
                "atom_count": 10,
                "motif_count": 4,
                "cross_edge_count": 3,
                "selfies_symbol_count": 12,
                "motif_identity_utf8_bytes": 25,
                "slots_populated": 40,
                "duplicate_slots": 8,
                "changed_token_slots": 7,
                "projection_seconds": 0.1,
                "e3fp_seconds": 0.4,
                "surface_seconds": 0.2,
            },
            {
                "benchmark_index": 1,
                "sdf_record_index": 19,
                "atom_count": 12,
                "motif_count": 5,
                "cross_edge_count": 4,
                "selfies_symbol_count": 15,
                "motif_identity_utf8_bytes": 30,
                "slots_populated": 48,
                "duplicate_slots": 9,
                "changed_token_slots": 8,
                "projection_seconds": 0.2,
                "e3fp_seconds": 0.6,
                "surface_seconds": 0.3,
            },
        ]
        rows = [self._row(0, 5), self._row(1, 19)]
        manifest = benchmark.summarize_results(
            results,
            phase_seconds={
                "selection_load": 0.01,
                "release_lmdb_read": 0.05,
                "sdf_scan": 2.0,
                "worker_pool": 1.0,
                "total": 3.1,
            },
            workers=8,
            max_pending=24,
            source_observation={
                "archive_bytes": 100,
                "sdf_records_scanned": 20,
                "maximum_selected_ordinal": 19,
                "rows": rows,
            },
            shard_count=2,
        )
        self.assertEqual(manifest["counts"]["benchmark_members"], 2)
        self.assertEqual(manifest["counts"]["atoms"], 22)
        self.assertEqual(manifest["counts"]["train_members"], 1)
        self.assertEqual(manifest["counts"]["dev_members"], 1)
        self.assertAlmostEqual(
            manifest["timing_seconds"]["worker_cpu_e3fp_sum"], 1.0
        )
        self.assertEqual(
            manifest["throughput"]["projected_pf1_target_members"], 33_600
        )


if __name__ == "__main__":
    unittest.main()
