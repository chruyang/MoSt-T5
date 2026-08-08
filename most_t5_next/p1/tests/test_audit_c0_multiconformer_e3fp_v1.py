import unittest

import numpy as np

from most_t5_next.p1 import audit_c0_multiconformer_e3fp_v1 as subject


class C0MulticonformerAuditTests(unittest.TestCase):
    def test_pair_metrics_separate_shell_atom_and_motif_changes(self):
        left = np.asarray(
            [[10, 11, 12, 13], [20, 21, 22, -1], [30, 31, 32, 33]],
            dtype=np.int32,
        )
        right = np.asarray(
            [[10, 11, 99, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
            dtype=np.int32,
        )
        row = subject._pair_metrics(np, left, right, ((0, 1), (2,)), 1.25)
        self.assertEqual(row["changed_atoms_l1_l3"], 2)
        self.assertEqual(row["changed_motifs_l1_l3"], 1)
        self.assertEqual(row["by_level"][0]["changed_rows"], 0)
        self.assertEqual(row["by_level"][2]["changed_rows"], 1)
        self.assertEqual(row["by_level"][3]["changed_populated_union_rows"], 1)
        self.assertFalse(row["exact_same_l1_l3"])

    def test_distribution_summary_is_categorical(self):
        result = subject._distribution_summary(subject.Counter({7: 3, 99: 1}))
        self.assertEqual(result["observations"], 4)
        self.assertEqual(result["unique_ids"], 2)
        self.assertAlmostEqual(result["top1_rate"], 0.75)
        self.assertGreater(result["entropy_nats"], 0.0)

    def test_spearman_handles_ties_without_scipy(self):
        self.assertAlmostEqual(subject._spearman([1, 2, 2, 4], [2, 4, 4, 8]), 1.0)

    def test_summary_closes_rigid_and_pair_counts(self):
        row = {
            "generated_conformers": 2,
            "rigid_transform_exact": True,
            "motif_sizes": [2],
            "token_counts_by_level": [
                {1: 2},
                {2: 2},
                {3: 2},
                {4: 2},
            ],
            "pairs": [
                {
                    "rmsd_angstrom": 1.5,
                    "atom_rows": 2,
                    "changed_atoms_l1_l3": 1,
                    "motifs": 1,
                    "changed_motifs_l1_l3": 1,
                    "exact_same_l1_l3": False,
                    "by_level": [
                        {
                            "level": level,
                            "rows": 2,
                            "changed_rows": int(level == 2),
                            "populated_union_rows": 2,
                            "changed_populated_union_rows": int(level == 2),
                        }
                        for level in range(4)
                    ],
                }
            ],
        }
        summary = subject.summarize_passes([row])
        self.assertEqual(summary["pair_count"], 1)
        self.assertEqual(summary["rigid_transform"]["exact_rate"], 1.0)
        self.assertEqual(summary["atom_l1_l3_change_rate"], 0.5)
        self.assertEqual(summary["motif_l1_l3_change_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
