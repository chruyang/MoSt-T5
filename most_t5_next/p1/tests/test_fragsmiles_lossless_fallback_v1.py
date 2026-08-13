from __future__ import annotations

from pathlib import Path
import unittest

from rdkit import Chem

from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import (
    LosslessFallbackError,
    decode_lossless_fallback_mol,
    encode_lossless_fallback,
    encode_main_or_fallback,
    fallback_token_universe,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class FragSmilesLosslessFallbackV1Tests(unittest.TestCase):
    def test_shared_smiles_domain_is_complete_and_collision_free(self) -> None:
        universe = fallback_token_universe()
        self.assertEqual(len(universe), 160)
        self.assertEqual(len(set(universe)), len(universe))
        self.assertNotIn("<unk>", universe)
        self.assertTrue(
            all(
                token.startswith("<MOST:SMI:") or token in set("0123456789")
                for token in universe
            )
        )

    def test_stereo_charge_isotope_and_components_round_trip(self) -> None:
        mol = Chem.MolFromSmiles("[13CH3][C@@H](F)Cl.[H+]")
        self.assertIsNotNone(mol)
        surface = encode_lossless_fallback(mol)
        restored = decode_lossless_fallback_mol(surface.tokens)
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )
        self.assertEqual(len(surface.atom_addresses), restored.GetNumAtoms())
        for address in surface.atom_addresses:
            self.assertTrue(
                all(
                    role == "atom_glyph"
                    for role in surface.roles[address.token_start : address.token_stop]
                )
            )
        hydrogen = [
            row
            for row in surface.atom_addresses
            if restored.GetAtomWithIdx(row.smiles_atom_ordinal).GetAtomicNum() == 1
        ]
        self.assertEqual(len(hydrogen), 1)
        self.assertIsNone(hydrogen[0].e3fp_row)
        heavy_rows = sorted(
            row.e3fp_row for row in surface.atom_addresses if row.e3fp_row is not None
        )
        self.assertEqual(heavy_rows, list(range(len(heavy_rows))))

    def test_directional_ring_smiles_cycle_has_one_fallback_surface(self) -> None:
        # RDKit alternates the slash orientation of this ring on a single
        # parse/reserialize step.  The fallback chooses the minimum cycle member
        # rather than pretending that one serialization call is canonical.
        mol = Chem.MolFromSmiles(r"CC1=C(C)/C=C/C2=C(\C=C/1)CCCC2")
        surface = encode_lossless_fallback(mol)
        restored = decode_lossless_fallback_mol(surface.tokens)
        self.assertEqual(encode_lossless_fallback(restored).tokens, surface.tokens)

    def test_atom_renumbering_preserves_tokens_and_row_mapping_tracks_input(self) -> None:
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        left = encode_lossless_fallback(mol)
        right = encode_lossless_fallback(reverse)
        self.assertEqual(left.tokens, right.tokens)
        self.assertEqual(
            sorted(row.e3fp_row for row in left.atom_addresses),
            list(range(mol.GetNumAtoms())),
        )
        self.assertEqual(
            sorted(row.e3fp_row for row in right.atom_addresses),
            list(range(mol.GetNumAtoms())),
        )

    def test_decoder_rejects_tamper_and_noncanonical_payload(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        surface = encode_lossless_fallback(mol)
        with self.assertRaises(LosslessFallbackError):
            decode_lossless_fallback_mol(surface.tokens[:-1] + ("<unk>",))

        # OCC is valid but not RDKit's canonical serialization of ethanol.
        from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
            encode_smiles_glyphs,
            smiles_glyph_token_map,
        )
        mapping = dict(smiles_glyph_token_map())
        noncanonical = tuple(mapping[x] for x in encode_smiles_glyphs("OCC").glyphs)
        with self.assertRaisesRegex(LosslessFallbackError, "not canonical"):
            decode_lossless_fallback_mol(noncanonical)

    def test_router_keeps_normal_compact_path_and_e3fp_partition(self) -> None:
        mol = Chem.MolFromSmiles("CC(=O)NCCc1ccccc1")
        routed = encode_main_or_fallback(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(routed.mode, "compact")
        self.assertEqual(
            sorted(row.e3fp_row for row in routed.compact_atom_addresses),
            list(range(mol.GetNumAtoms())),
        )

    def test_real_pcqm_compact_reject_routes_to_strict_fallback(self) -> None:
        # PCQM full-domain audit source_index=1858.  The compact local codec
        # rejects the defined C=N stereo because RDKit supplies no E/Z CIP
        # label; the universal path must preserve the canonical isomer instead.
        smiles = (
            r"[H]C([H])([H])O/N=C1\[C@@]2([H])C([H])([H])[C@@]3([H])"
            r"C([H])([H])[C@]1([H])C([H])([H])[C@@](F)(C2([H])[H])C3([H])[H]"
        )
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        routed = encode_main_or_fallback(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(routed.mode, "whole_molecule_fallback")
        self.assertEqual(routed.fallback_reason_type, "CompactStereoCodecError")
        restored = decode_lossless_fallback_mol(routed.tokens)
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )

    def test_raw_explicit_h_address_failure_routes_to_fallback(self) -> None:
        # The formal compact path consumes the heavy-atom projection.  If a raw
        # source with stereo-defining explicit H reaches the public router, the
        # lossless path preserves it instead of fabricating an E3FP row.
        smiles = (
            r"[H]/N=C(/S[C@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])[H])"
            r"N([H])/N=C(\[H])c1c([H])c([H])nc([H])c1[H]"
        )
        mol = Chem.MolFromSmiles(smiles)
        routed = encode_main_or_fallback(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        self.assertEqual(routed.mode, "whole_molecule_fallback")
        self.assertEqual(routed.fallback_reason_type, "LosslessFallbackError")
        restored = decode_lossless_fallback_mol(routed.tokens)
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )

    def test_directional_fragment_routes_to_shared_smiles_fallback(self) -> None:
        # Real PubChem downstream source_index=143587.  The compact chemistry
        # round trip succeeds but one raw fragment still contains directional
        # stereo, which is intentionally outside the pure-motif lexer.
        smiles = (
            r"CC\1=CC(=C2C(=C3C=CC=CC3=N2)/C1=C/4\C(=C(C5=C6C=CC=CC6=NC5=C4O)O)C)OC"
        )
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)
        routed = encode_main_or_fallback(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        # The shared vocabulary supports slash/backslash, but ordinary motif
        # identity remains stereo-free.  The whole-molecule route uses those
        # same glyph rows without a byte namespace.
        self.assertEqual(routed.mode, "whole_molecule_fallback")
        self.assertEqual(routed.fallback_reason_type, "SmirkSmilesVocabularyError")
        restored = decode_lossless_fallback_mol(routed.tokens)
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )



if __name__ == "__main__":
    unittest.main()
