import unittest

from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
    SLOT_TOKEN,
    StereoFreeMotifLexerError,
    byte_fallback_tokens,
    chemical_token_universe,
    decode_byte_fallback,
    decode_pure_motif,
    lex_pure_motif,
    opaque_chemical_token_map,
    validate_anchor_token_outside_lexer,
)


class StereoFreeMotifChemicalLexerV1Test(unittest.TestCase):
    def test_slots_atoms_bonds_and_rings_round_trip(self):
        pure = "[C1()C(=O)N()[C]2[C]()C(=O)NN21]"
        lexed = lex_pure_motif(pure)
        self.assertEqual(decode_pure_motif(lexed.tokens), pure)
        self.assertEqual(lexed.tokens.count(SLOT_TOKEN), 3)
        self.assertNotIn("<unk>", lexed.tokens)

    def test_bracket_atom_is_bounded_and_reversible(self):
        pure = "[[13CH2+][nH][SiH2:17]()]"
        lexed = lex_pure_motif(pure)
        self.assertEqual(decode_pure_motif(lexed.tokens), pure)
        self.assertIn("13", "".join(lexed.tokens))
        self.assertIn("Si", lexed.tokens)
        self.assertIn("atom_class_marker", lexed.roles)

    def test_percent_ring_and_two_letter_atoms(self):
        pure = "[ClC%12(Br)C%12]"
        lexed = lex_pure_motif(pure)
        self.assertEqual(decode_pure_motif(lexed.tokens), pure)
        self.assertIn("Cl", lexed.tokens)
        self.assertIn("Br", lexed.tokens)
        self.assertEqual(lexed.tokens.count("%"), 2)

    def test_pinned_rdkit_aromatic_silicon_and_tellurium_are_bounded(self):
        for pure, expected in (
            ("[c1cc[siH]cc1]", "si"),
            ("[c1cc[te]c1]", "te"),
        ):
            with self.subTest(pure=pure):
                lexed = lex_pure_motif(pure)
                self.assertEqual(decode_pure_motif(lexed.tokens), pure)
                self.assertIn(expected, lexed.tokens)
                self.assertIn(expected, chemical_token_universe())

    def test_stereo_anchor_and_unknown_element_fail_closed(self):
        for pure in ("[F[C@H](Cl)Br]", "[C/C=C\\C]", "[C<0*>]", "[[Xx]C]"):
            with self.subTest(pure=pure):
                with self.assertRaises(StereoFreeMotifLexerError):
                    lex_pure_motif(pure)

    def test_bytes_are_universal_and_strict(self):
        pure = "[[13C-]C()]"
        tokens = byte_fallback_tokens(pure)
        self.assertEqual(decode_byte_fallback(tokens), pure)
        with self.assertRaises(StereoFreeMotifLexerError):
            decode_byte_fallback(("<0xGG>",))

    def test_anchor_namespace_stays_outside_chemical_lexer(self):
        self.assertEqual(validate_anchor_token_outside_lexer("<12*>"), 12)
        with self.assertRaises(StereoFreeMotifLexerError):
            validate_anchor_token_outside_lexer("<012*>")

    def test_finite_universe_covers_emitted_tokens_and_uses_opaque_surface(self):
        universe = chemical_token_universe()
        emitted = lex_pure_motif("[[13CH2+][SiH2:17]C%12()]").tokens
        self.assertTrue(set(emitted).issubset(universe))
        mapping = opaque_chemical_token_map()
        self.assertEqual(tuple(logical for logical, _ in mapping), universe)
        self.assertEqual(len({surface for _, surface in mapping}), len(universe))
        self.assertTrue(all(surface.startswith("<MOST:CHEM:") for _, surface in mapping))


if __name__ == "__main__":
    unittest.main()
