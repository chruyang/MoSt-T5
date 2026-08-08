from __future__ import annotations

import importlib.util
import unittest


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

if RDKIT_AVAILABLE:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from most_t5_next.p2.morgan_atom_state_v1 import (
        MorganAtomStateError,
        derive_morgan_atom_state,
    )


@unittest.skipUnless(RDKIT_AVAILABLE, "RDKit is required")
class MorganAtomStateTest(unittest.TestCase):
    def _derive(self, smiles: str, carriers):
        return derive_morgan_atom_state(
            Chem=Chem,
            rdFingerprintGenerator=rdFingerprintGenerator,
            selfies_decoder=lambda value: value,
            selfies=smiles,
            atom_to_carrier=carriers,
        )

    def test_returns_four_coordinate_blind_slots_on_model_atom_axis(self):
        ordinary = self._derive("CCO", (0, 1, 2))
        permuted = self._derive("CCO", (2, 0, 1))
        self.assertEqual(len(ordinary.state_ids), 3)
        self.assertTrue(all(len(row) == 4 for row in ordinary.state_ids))
        # decoded atom 0/1/2 map respectively to model atom 1/2/0.
        self.assertEqual(permuted.state_ids[1], ordinary.state_ids[0])
        self.assertEqual(permuted.state_ids[2], ordinary.state_ids[1])
        self.assertEqual(permuted.state_ids[0], ordinary.state_ids[2])
        self.assertTrue(all(row[0] >= 0 for row in ordinary.state_ids))

    def test_chirality_is_part_of_the_2d_control(self):
        left = self._derive("F[C@H](Cl)Br", (0, 1, 2, 3))
        right = self._derive("F[C@@H](Cl)Br", (0, 1, 2, 3))
        self.assertNotEqual(left.state_ids, right.state_ids)

    def test_rejects_ambiguous_or_wrong_atom_mapping(self):
        with self.assertRaisesRegex(MorganAtomStateError, "unique"):
            self._derive("CC", (0, 0))
        with self.assertRaisesRegex(MorganAtomStateError, "atom count"):
            self._derive("CCO", (0, 1))


if __name__ == "__main__":
    unittest.main()
