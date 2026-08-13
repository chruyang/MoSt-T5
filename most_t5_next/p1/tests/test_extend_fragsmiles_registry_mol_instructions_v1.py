from __future__ import annotations

import unittest

from most_t5_next.p1.extend_fragsmiles_registry_mol_instructions_v1 import extend_registry


class MolInstructionsRegistryExtensionV1Tests(unittest.TestCase):
    def test_all_train_motifs_are_union_added_without_frequency_filter(self):
        base = [{"rank": 0, "surface_token": "m0", "fragment_identity": "C"}]
        tasks = {
            "reagent": [{"rank": 0, "fragment_identity": "[LiH]", "occurrences": 1}],
            "forward": [{"rank": 0, "fragment_identity": "N", "occurrences": 5}],
            "retro": [
                {"rank": 0, "fragment_identity": "N", "occurrences": 2},
                {"rank": 1, "fragment_identity": "C", "occurrences": 9},
            ],
        }
        rows, report = extend_registry(base_rows=base, task_rows=tasks)
        self.assertEqual([row["fragment_identity"] for row in rows], ["C", "N", "[LiH]"])
        self.assertEqual(rows[1]["mol_instructions_forward_train_occurrences"], 5)
        self.assertEqual(rows[1]["mol_instructions_retro_train_occurrences"], 2)
        self.assertEqual(rows[2]["mol_instructions_reagent_train_occurrences"], 1)
        self.assertEqual(report["counts"]["final_macros"], 3)
        self.assertFalse(report["policy"]["frequency_filter"])


if __name__ == "__main__":
    unittest.main()
