from __future__ import annotations

from pathlib import Path
import unittest

from rdkit import Chem

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import (
    compact_stereo_token_universe,
    decode_compact_stereo_surface,
    encode_compact_stereo_surface,
    strict_round_trip,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class CompactFragSmilesStereoCodecTests(unittest.TestCase):
    def test_stereo_token_universe_is_finite_and_complete(self):
        universe = compact_stereo_token_universe()
        self.assertEqual(len(universe), 10)
        self.assertEqual(len(set(universe)), len(universe))
        mol = Chem.MolFromSmiles("N[C@@H](C)/C=C/F")
        surface = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertTrue(
            all(
                token in universe
                for token in surface.tokens
                if token.startswith("<ST:")
            )
        )

    def assert_round_trip(self, smiles: str) -> None:
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        surface = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        restored = decode_compact_stereo_surface(
            surface.tokens, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )

    def test_tetrahedral_r_s_are_two_token_local_records(self):
        for smiles in ("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                surface = strict_round_trip(
                    mol, chemicalgof_root=CHEMICALGOF_ROOT
                )
                self.assertEqual(len(surface.atom_records), 1)
                self.assertEqual(
                    len(surface.tokens) - len(surface.connectivity_record.tokens), 4
                )

    def test_e_z_are_three_token_local_records(self):
        for smiles in ("F/C=C/F", "F/C=C\\F", "C/C=N/O", "C/C=N\\O"):
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                surface = strict_round_trip(
                    mol, chemicalgof_root=CHEMICALGOF_ROOT
                )
                self.assertEqual(len(surface.bond_records), 1)
                self.assertEqual(
                    len(surface.tokens) - len(surface.connectivity_record.tokens), 7
                )

    def test_stereo_defining_explicit_h_round_trip(self):
        self.assert_round_trip("[H]/N=C(\\N)C#N")

    def test_disconnected_component_and_stereo_round_trip(self):
        self.assert_round_trip("N.F/C=C/F")

    def test_atom_renumbering_preserves_surface(self):
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        left = encode_compact_stereo_surface(
            mol, chemicalgof_root=CHEMICALGOF_ROOT
        )
        right = encode_compact_stereo_surface(
            reverse, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertEqual(left.tokens, right.tokens)

    def test_symmetric_charged_cage_has_one_fixed_point_surface(self):
        mol = Chem.MolFromSmiles("[C@@H]12[C@H]3[C@@H]1[N@@H+]1[C@H]3[C@H]21")
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        left = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        right = strict_round_trip(reverse, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(left.tokens, right.tokens)
        restored = decode_compact_stereo_surface(
            left.tokens, chemicalgof_root=CHEMICALGOF_ROOT
        )
        replay = encode_compact_stereo_surface(
            restored, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertEqual(left.tokens, replay.tokens)

    def test_transient_surface_is_excluded_from_cycle_canonicalization(self):
        # QM9 raw index 2044 enters the same deterministic surface cycle from
        # two different transient representations.  Only cycle members may be
        # candidates for the canonical token surface.
        mol = Chem.MolFromSmiles("C[C@]1(O)[C@@H]2C[C@H]1C2")
        surface = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        restored = decode_compact_stereo_surface(
            surface.tokens, chemicalgof_root=CHEMICALGOF_ROOT
        )
        replay = encode_compact_stereo_surface(
            restored, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertEqual(surface.tokens, replay.tokens)

    def test_non_cip_tetrahedral_identity_uses_local_parity_not_fake_r_s(self):
        mol = Chem.MolFromSmiles("C[C@H]1[C@H]2C[C@@H]1C2")
        surface = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertTrue(any(record.label == "X" for record in surface.atom_records))
        self.assertTrue(
            all(record.local_parity in {"CW", "CCW"} for record in surface.atom_records)
        )


if __name__ == "__main__":
    unittest.main()
