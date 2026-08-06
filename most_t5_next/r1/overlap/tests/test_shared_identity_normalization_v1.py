from __future__ import annotations

import unittest

from rdkit import Chem

from most_t5_next.r1.gates import pcqm_identity_smoke
from most_t5_next.r1.overlap import shared_identity_normalization_v1 as identity


class SharedIdentityNormalizationTests(unittest.TestCase):
    def forms(self, smiles: str) -> tuple[str, str]:
        result = identity.canonical_forms_from_smiles(smiles)
        return result.strict_isomeric_smiles, result.connectivity_smiles

    def test_explicit_stereo_defining_h_matches_implicit_form(self):
        self.assertEqual(self.forms("C[C@]([H])(O)F"), self.forms("C[C@H](O)F"))
        self.assertEqual(self.forms("[H]/C=C/F"), self.forms("C=CF"))

    def test_isotopic_hydrogen_retains_chemical_identity(self):
        result = identity.canonical_forms_from_smiles("[2H]C")
        self.assertIn("[2H]", result.strict_isomeric_smiles)
        self.assertEqual(result.molecule.GetNumAtoms(), 2)

    def test_stereo_is_strict_but_not_connectivity_identity(self):
        r_form = self.forms("C[C@H](O)F")
        s_form = self.forms("C[C@@H](O)F")
        self.assertNotEqual(r_form[0], s_form[0])
        self.assertEqual(r_form[1], s_form[1])

        e_form = self.forms("F/C=C/F")
        z_form = self.forms("F/C=C\\F")
        self.assertNotEqual(e_form[0], z_form[0])
        self.assertEqual(e_form[1], z_form[1])

    def test_charge_components_and_atom_order_are_preserved_semantically(self):
        self.assertEqual(self.forms("[Na+].[Cl-]"), self.forms("[Cl-].[Na+]"))
        self.assertNotEqual(self.forms("C[NH3+]")[1], self.forms("CN")[1])
        self.assertEqual(self.forms("OC(F)C"), self.forms("CC(O)F"))

    def test_matches_frozen_pcqm_reference_implementation(self):
        fixtures = (
            "C[C@]([H])(O)F",
            "[H]/C=C/F",
            "[2H]C",
            "F/C=C\\F",
            "[Na+].[Cl-]",
        )
        for smiles in fixtures:
            molecule = Chem.MolFromSmiles(smiles)
            expected = pcqm_identity_smoke.canonical_forms(Chem, molecule)
            observed = identity.canonical_forms_from_molecule(molecule)
            self.assertEqual(observed.strict_isomeric_smiles, expected["strict"])
            self.assertEqual(observed.connectivity_smiles, expected["connectivity"])
            self.assertEqual(observed.molecule.GetNumAtoms(), expected["atom_count"])


if __name__ == "__main__":
    unittest.main()
