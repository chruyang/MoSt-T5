from __future__ import annotations

from array import array
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import analyze_task_aware_registry_on_full_cache_v1 as task


class TaskAwareFullCacheAnalysisTest(unittest.TestCase):
    def test_external_registry_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            with (cache / "shard-000000.motif_ids.u32").open("xb") as handle:
                array("I", [0, 0, 1, 2]).tofile(handle)
            with (cache / "shard-000000.offsets.u64").open("xb") as handle:
                array("Q", [0, 2, 4]).tofile(handle)
            metrics = task._evaluate(cache, {0, 1}, 3)
        self.assertEqual(metrics["macro_occurrence_coverage"], 0.75)
        self.assertEqual(metrics["fully_macro_tokenized_molecule_rate"], 0.5)
        self.assertEqual(metrics["molecules_with_at_most_1_fallback_rate"], 1.0)

    def test_balanced_ranking_can_promote_downstream_identity(self):
        ranking = task._balanced_ranking(
            {"A": (0, 100), "B": (1, 10)}, {"A": 1, "C": 100}
        )
        self.assertEqual(ranking[0], "C")


if __name__ == "__main__":
    unittest.main()
