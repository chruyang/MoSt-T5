from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import analyze_downstream_motif_coverage_v1 as coverage


class DownstreamMotifCoverageTest(unittest.TestCase):
    def test_csv_source_reads_smiles_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "molecules.csv"
            path.write_text("smiles,label\nCC,1\nCO,0\n", encoding="utf-8")
            rows = list(
                coverage._source_smiles(
                    {"path": str(path), "format": "csv", "smiles_field": "smiles"}
                )
            )
            self.assertEqual(rows, [("0", "CC"), ("1", "CO")])

    def test_occurrence_and_whole_molecule_metrics_differ(self):
        sequences = (("A", "B"), ("A",), ("C",))
        counts = {"A": 2, "B": 1, "C": 1}
        metrics = coverage._metrics(sequences, counts, {"A"})
        self.assertEqual(metrics["macro_occurrence_coverage"], 0.5)
        self.assertEqual(metrics["fully_macro_tokenized_molecule_rate"], 1 / 3)
        self.assertEqual(metrics["molecules_with_at_most_1_fallback_rate"], 1.0)
        self.assertEqual(metrics["uncovered_motif_types"], 2)


if __name__ == "__main__":
    unittest.main()
