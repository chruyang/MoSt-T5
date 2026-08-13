from __future__ import annotations

from pathlib import Path
import unittest

from rdkit import Chem

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import strict_round_trip
from most_t5_next.p1.fragsmiles_macro_fallback_surface_v1 import (
    CONNECTOR_END,
    FRAGMENT_FALLBACK_END,
    FragSmilesModelSurfaceError,
    decode_model_surface_mol,
    decode_model_tokens,
    encode_compact_model_surface,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class FragSmilesMacroFallbackSurfaceV1Tests(unittest.TestCase):
    def test_macro_and_semantic_fragment_fallback_mix_round_trip(self) -> None:
        mol = Chem.MolFromSmiles("CC(=O)NCCc1ccccc1")
        compact = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        macros = [
            {"fragment_smiles": "C", "surface_token": "<MOST:FM:000000>"},
            {
                "fragment_smiles": "c1ccccc1",
                "surface_token": "<MOST:FM:000001>",
            },
        ]
        surface = encode_compact_model_surface(
            mol, compact, macros, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertIn(True, surface.macro_used)
        self.assertIn(False, surface.macro_used)
        self.assertEqual(
            surface.tokens.count(FRAGMENT_FALLBACK_END),
            surface.macro_used.count(False),
        )
        self.assertNotIn("<unk>", surface.tokens)
        self.assertEqual(decode_model_tokens(surface.tokens, macros), compact.tokens)
        restored = decode_model_surface_mol(
            surface.tokens, macros, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )
        self.assertEqual(
            sorted(row.e3fp_row for row in surface.atom_addresses),
            list(range(mol.GetNumAtoms())),
        )
        for row in surface.atom_addresses:
            phrase = surface.fragment_phrases[row.fragment_index]
            self.assertEqual(row.carrier_token_index, phrase.carrier_token_index)

    def test_stereo_records_survive_macro_compression(self) -> None:
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        compact = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        identities = tuple(
            dict.fromkeys(
                fragment.fragment_smiles
                for fragment in compact.connectivity_record.fragments
            )
        )
        macros = [
            {
                "fragment_smiles": identity,
                "surface_token": f"<MOST:FM:{index:06d}>",
            }
            for index, identity in enumerate(identities)
        ]
        surface = encode_compact_model_surface(
            mol, compact, macros, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertTrue(all(surface.macro_used))
        self.assertTrue(any(token.startswith("<ST:A:") for token in surface.tokens))
        restored = decode_model_surface_mol(
            surface.tokens, macros, chemicalgof_root=CHEMICALGOF_ROOT
        )
        self.assertEqual(
            Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        )

    def test_missing_fragment_suffix_and_connector_end_fail_closed(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        compact = strict_round_trip(mol, chemicalgof_root=CHEMICALGOF_ROOT)
        surface = encode_compact_model_surface(
            mol, compact, (), chemicalgof_root=CHEMICALGOF_ROOT
        )
        suffix_index = surface.tokens.index(FRAGMENT_FALLBACK_END)
        with self.assertRaises(FragSmilesModelSurfaceError):
            decode_model_surface_mol(
                surface.tokens[:suffix_index] + surface.tokens[suffix_index + 1 :],
                (),
                chemicalgof_root=CHEMICALGOF_ROOT,
            )

        connector_positions = [
            index for index, token in enumerate(surface.tokens) if token == CONNECTOR_END
        ]
        if connector_positions:
            index = connector_positions[0]
            with self.assertRaises(FragSmilesModelSurfaceError):
                decode_model_tokens(surface.tokens[:index] + surface.tokens[index + 1 :], ())


if __name__ == "__main__":
    unittest.main()
