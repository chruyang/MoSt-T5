from __future__ import annotations

import json
from pathlib import Path
import unittest

from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
    SmirkSmilesVocabularyError,
    decode_smiles_glyphs,
    encode_smiles_glyphs,
    require_stereo_free_fragment,
    smiles_added_token_universe,
    smiles_glyph_token_map,
    smiles_glyph_universe,
)


class SmirkSmilesVocabularyV1Tests(unittest.TestCase):
    def test_pinned_upstream_order_and_collision_free_surface(self) -> None:
        upstream = (
            Path(__file__).resolve().parents[4]
            / "reference_repos"
            / "smirk_official_src"
            / "python"
            / "smirk"
            / "vocab_smiles.json"
        )
        if upstream.is_file():
            vocab = json.loads(upstream.read_text(encoding="utf-8"))
            observed = tuple(
                token
                for token, _token_id in sorted(vocab.items(), key=lambda row: row[1])
                if token != "[UNK]"
            )
            self.assertEqual(smiles_glyph_universe()[:-2], observed)
        mapping = smiles_glyph_token_map()
        self.assertEqual(len(mapping), 160)
        self.assertEqual(len({row[0] for row in mapping}), 160)
        self.assertEqual(len({row[1] for row in mapping}), 160)
        self.assertEqual(mapping[10:20], tuple((digit, digit) for digit in "0123456789"))
        self.assertEqual(len(smiles_added_token_universe()), 150)
        self.assertEqual(mapping[-2:], (("si", "<MOST:SMI:158>"), ("te", "<MOST:SMI:159>")))

    def test_open_smiles_fields_round_trip(self) -> None:
        for smiles in (
            "CC[N+](C)(C)Cc1ccccc1Br",
            "[13CH3][C@@H](F)Cl.[H+]",
            "[Fe@TB3+3]",
            "[CH4:200]",
            "C%12CCCCC%12",
            "F/C=C\\F",
        ):
            encoding = encode_smiles_glyphs(smiles)
            self.assertEqual(decode_smiles_glyphs(encoding.glyphs), smiles)
            self.assertEqual("".join(smiles[a:b] for a, b in encoding.character_spans), smiles)

    def test_rdkit_aromatic_extensions_round_trip(self) -> None:
        for smiles in ("c1cc[siH]cc1", "c1cc[te]c1", "[siH]1[siH][siH][siH][siH][siH]1"):
            encoding = encode_smiles_glyphs(smiles)
            self.assertNotIn("[UNK]", encoding.glyphs)
            self.assertEqual(decode_smiles_glyphs(encoding.glyphs), smiles)

    def test_invalid_or_non_smiles_text_is_rejected(self) -> None:
        for value in ("", "C馃し", "[CH22+2]", "C<0>", "[C"):
            with self.subTest(value=value), self.assertRaises(SmirkSmilesVocabularyError):
                encode_smiles_glyphs(value)

    def test_ordinary_fragment_policy_is_stereo_free(self) -> None:
        self.assertEqual(require_stereo_free_fragment("c1ccccc1").smiles, "c1ccccc1")
        for value in ("F/C=C/F", "[C@@H](N)C"):
            with self.subTest(value=value), self.assertRaises(SmirkSmilesVocabularyError):
                require_stereo_free_fragment(value)


if __name__ == "__main__":
    unittest.main()
