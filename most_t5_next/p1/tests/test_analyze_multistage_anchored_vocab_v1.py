from __future__ import annotations

from array import array
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import analyze_multistage_anchored_vocab_v1 as multi


class MultistageAnchoredVocabTest(unittest.TestCase):
    def _write(self, path: Path, typecode: str, values) -> None:
        with path.open("xb") as handle:
            array(typecode, values).tofile(handle)

    def test_phase2_can_change_equal_stage_ranking(self):
        p1 = {"A": (0, 1000), "B": (1, 100)}
        p2 = {"A": (0, 1), "C": (1, 100)}
        self.assertEqual(multi._pooled_ranking(p1, p2)[0], "A")
        self.assertEqual(multi._equal_stage_ranking(p1, p2)[0], "C")

    def test_compact_cache_metrics_are_exact(self):
        registry = {"A": (0, 3), "B": (1, 1), "C": (2, 1)}
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            self._write(cache / "motif_ids.u32", "I", [0, 0, 1, 2])
            self._write(cache / "offsets.u64", "Q", [0, 2, 3, 4])
            metrics = multi._evaluate_cache(cache, registry, {"A"})
        self.assertEqual(metrics["records"], 3)
        self.assertEqual(metrics["motif_occurrences"], 4)
        self.assertEqual(metrics["macro_occurrence_coverage"], 0.5)
        self.assertEqual(metrics["fully_macro_tokenized_molecule_rate"], 1 / 3)
        self.assertEqual(metrics["molecules_with_at_most_1_fallback_rate"], 1.0)

    def test_chebi_additions_are_ranked_from_train_counts_only(self):
        counts = {"X": 1, "Y": 10, "Z": 10}
        ranked = tuple(
            sorted(counts, key=lambda pure: (-counts[pure], pure.encode("utf-8")))
        )
        self.assertEqual(ranked, ("Y", "Z", "X"))


if __name__ == "__main__":
    unittest.main()
