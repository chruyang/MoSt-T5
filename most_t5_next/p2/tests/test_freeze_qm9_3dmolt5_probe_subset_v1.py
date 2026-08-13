from __future__ import annotations

import unittest

from most_t5_next.p2.freeze_qm9_3dmolt5_probe_subset_v1 import (
    QM9ProbeSubsetError,
    classify_property,
    freeze_rows,
    parse_target,
)


def _row(smiles: str, instruction: str, output: str, fp: int = 1):
    return {
        "smiles": smiles,
        "selfies": "[C]",
        "molecule_fp": [[fp, fp + 1, fp + 2, -1]],
        "instruction": instruction,
        "output": output,
    }


class FreezeQM9ProbeSubsetV1Test(unittest.TestCase):
    def test_property_and_scalar_parsing(self) -> None:
        self.assertEqual(classify_property("What is the HOMO-LUMO gap?"), "gap")
        self.assertEqual(classify_property("Give the LUMO energy"), "lumo")
        self.assertEqual(classify_property("Give the HOMO energy"), "homo")
        self.assertEqual(parse_target("-0.1234."), -0.1234)

    def test_split_is_by_molecule_and_paraphrases_collapse(self) -> None:
        rows = []
        for index in range(6):
            smiles = f"C{'C' * index}N"
            rows.extend([
                _row(smiles, "Give the HOMO energy", "-0.2."),
                _row(smiles, "What is the HOMO energy?", "-0.2."),
                _row(smiles, "Give the LUMO energy", "0.1."),
            ])
        membership, manifest = freeze_rows(
            rows,
            split_counts={"train": 3, "dev": 2, "test": 1},
            seed=7,
        )
        by_smiles = {}
        for row in membership:
            by_smiles.setdefault(row["smiles"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_smiles.values()))
        self.assertEqual(len(membership), 12)
        self.assertEqual(manifest["conflicting_exact_state_property_groups_rejected"], 0)
        homo = [row for row in membership if row["property"] == "homo"]
        self.assertTrue(all(row["instruction_paraphrase_count"] == 2 for row in homo))

    def test_conflicting_exact_state_target_is_rejected(self) -> None:
        rows = [
            _row("C", "Give the HOMO energy", "-0.20."),
            _row("C", "What is the HOMO energy?", "-0.21."),
            _row("N", "Give the HOMO energy", "-0.30.", fp=4),
            _row("O", "Give the HOMO energy", "-0.40.", fp=8),
        ]
        membership, manifest = freeze_rows(
            rows,
            split_counts={"train": 1, "dev": 1, "test": 1},
            seed=11,
        )
        self.assertEqual(manifest["conflicting_exact_state_property_groups_rejected"], 1)
        self.assertEqual(len(membership), 2)

    def test_unknown_instruction_is_rejected(self) -> None:
        with self.assertRaises(QM9ProbeSubsetError):
            classify_property("Predict this property")


if __name__ == "__main__":
    unittest.main()
