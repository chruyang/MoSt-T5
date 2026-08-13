from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import analyze_full_anchored_vocab_v1 as full


class FullAnchoredVocabAnalysisTest(unittest.TestCase):
    def _write_array(self, path: Path, typecode: str, values) -> None:
        with path.open("xb") as handle:
            array(typecode, values).tofile(handle)

    def test_whole_molecule_coverage_is_distinct_from_occurrence_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            self._write_array(
                cache / "shard-000000.motif_ids.u32", "I", [0, 0, 0, 1, 2]
            )
            self._write_array(
                cache / "shard-000000.offsets.u64", "Q", [0, 2, 4, 5]
            )
            self._write_array(
                cache / "shard-000000.ordinals.u32", "I", [10, 11, 12]
            )
            self._write_array(
                cache / "shard-000000.anchor_counts.u16", "H", [0, 1, 0]
            )
            report, ranking, counts = full._analyze_cache(
                cache,
                ("[C]", "[N]", "[O]"),
                "1,2",
                hidden_size=768,
                tie_word_embeddings=False,
            )
        self.assertEqual(ranking, [0, 1, 2])
        self.assertEqual(counts, [3, 1, 1])
        one, two = report["budget_rows"]
        self.assertAlmostEqual(one["macro_occurrence_coverage"], 3 / 5)
        self.assertAlmostEqual(one["fully_macro_tokenized_molecule_rate"], 1 / 3)
        self.assertEqual(one["molecules_with_at_most_1_fallback_rate"], 1.0)
        self.assertAlmostEqual(two["fully_macro_tokenized_molecule_rate"], 2 / 3)
        self.assertEqual(two["additional_untied_vocab_parameters"], 2 * 768 * 2)

    def test_budget_clamping_is_unique(self):
        self.assertEqual(full._candidate_budgets("2,5,8", 3), (2, 3))

    def test_exact_lexeme_registry_collapses_traversal_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "census.jsonl"
            fragments = ("<0*>C=N", "N=C<0*>")
            with path.open("x", encoding="utf-8") as handle:
                for fragment in fragments:
                    handle.write(
                        json.dumps(
                            {
                                "motif_lexeme_sha256": hashlib.sha256(
                                    fragment.encode("utf-8")
                                ).hexdigest(),
                                "motif_fragment": fragment,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            mapping, registry = full._load_pure_registry(path)
        self.assertEqual(registry, ("[C=N]",))
        self.assertEqual({value for value in mapping.values()}, {(0, 1)})


if __name__ == "__main__":
    unittest.main()
