from __future__ import annotations

import unittest

from most_t5_next.p1.analyze_chebi20_task_aware_vocab_v1 import _metrics


class MultistageChEBISplitCoverageTest(unittest.TestCase):
    def test_validation_metrics_do_not_select_rows(self):
        sequences = (("A", "B"), ("C",))
        counts = {"A": 1, "B": 1, "C": 1}
        selected_from_train = {"A", "B"}
        metrics = _metrics(sequences, counts, selected_from_train)
        self.assertEqual(metrics["macro_occurrence_coverage"], 2 / 3)
        self.assertEqual(metrics["fully_macro_tokenized_molecule_rate"], 1 / 2)
        self.assertEqual(selected_from_train, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
