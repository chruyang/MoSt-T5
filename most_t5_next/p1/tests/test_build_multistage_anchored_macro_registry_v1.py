import argparse
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import build_multistage_anchored_macro_registry_v1 as subject


class MultistageMacroRegistryTest(unittest.TestCase):
    @staticmethod
    def _registry(path: Path, rows):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for rank, (pure, count) in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "pure_motif": pure,
                            "pure_motif_id": rank,
                            "occurrences": count,
                            "rank": rank,
                        }
                    )
                    + "\n"
                )

    def test_general_base_precedes_all_train_only_additions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p1 = root / "p1.jsonl"
            p2 = root / "p2.jsonl"
            chebi = root / "chebi.jsonl"
            output = root / "out"
            self._registry(p1, [("A", 10), ("B", 1)])
            self._registry(p2, [("B", 10), ("A", 1)])
            with chebi.open("x", encoding="utf-8", newline="\n") as handle:
                for pure, count in (("A", 5), ("C", 1), ("D", 2)):
                    handle.write(
                        json.dumps(
                            {"pure_motif": pure, "train_occurrences": count}
                        )
                        + "\n"
                    )
            manifest = subject.build(
                argparse.Namespace(
                    phase1_registry=str(p1),
                    phase2_registry=str(p2),
                    chebi_train_census=str(chebi),
                    output_dir=str(output),
                    general_base_budget=1,
                )
            )
            rows = [
                json.loads(line)
                for line in (output / subject.REGISTRY_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["rank"] for row in rows], [0, 1, 2])
            self.assertEqual([row["pure_motif"] for row in rows[1:]], ["D", "C"])
            self.assertEqual(rows[0]["selection_role"], "phase1_phase2_equal_stage_base")
            self.assertTrue(
                all(
                    row["selection_role"] == "chebi20_train_all_extension"
                    for row in rows[1:]
                )
            )
            self.assertEqual(manifest["selection"]["total_macro_rows"], 3)
            self.assertEqual(manifest["selection"]["extension_singletons"], 1)


if __name__ == "__main__":
    unittest.main()
