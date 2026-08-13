from __future__ import annotations

import unittest

from most_t5_next.p1 import analyze_chebi20_task_aware_vocab_v1 as chebi


class ChEBITaskAwareVocabTest(unittest.TestCase):
    def test_whole_molecule_coverage_and_balanced_ranking(self):
        sequences = (("A", "B"), ("A",), ("C",))
        counts = {"A": 2, "B": 1, "C": 1}
        metrics = chebi._metrics(sequences, counts, {"A"})
        self.assertEqual(metrics["macro_occurrence_coverage"], 0.5)
        self.assertEqual(metrics["fully_macro_tokenized_molecule_rate"], 1 / 3)
        self.assertEqual(metrics["molecules_with_at_most_1_fallback_rate"], 1.0)
        ranking = chebi._balanced_ranking(
            {"A": 100, "B": 10}, {"A": 1, "C": 100}
        )
        self.assertEqual(ranking[0], "C")
        self.assertIn("A", ranking[:2])


if __name__ == "__main__":
    unittest.main()
