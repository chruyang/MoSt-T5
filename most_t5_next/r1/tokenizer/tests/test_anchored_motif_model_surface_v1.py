from __future__ import annotations

import unittest

from most_t5_next.r1.tokenizer.anchored_motif_model_surface_v1 import (
    AnchoredMotifModelSurfaceError,
    decode_explicit_sequence,
    decode_fallback_prefixed_sequence,
    decode_fallback_suffixed_sequence,
    decode_implicit_with_sidecar,
    encode_frozen_phrases,
    encode_phrases,
    frozen_grammar_contract,
)


class AnchoredMotifModelSurfaceV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.phrases = [
            {"pure_motif": "[C()]", "anchors": [{"anchor_id": 1}, {"anchor_id": 0}]},
            {"pure_motif": "[[NH+]=C]", "anchors": [{"anchor_id": 0}]},
            {"pure_motif": "[O]", "anchors": []},
        ]
        self.macros = [{"pure_motif": "[C()]", "surface_token": "<MOST:MACRO:000000>"}]
        self.expected = (("[C()]", (1, 0)), ("[[NH+]=C]", (0,)), ("[O]", ()))

    def test_single_prefix_is_standalone_reversible(self) -> None:
        encoded = encode_phrases(
            self.phrases, self.macros, boundary_mode="explicit_single_prefix"
        )
        self.assertEqual(decode_explicit_sequence(encoded.tokens, self.macros), self.expected)
        self.assertEqual(encoded.tokens.count("<MOST:MOTIF>"), 3)
        self.assertEqual(encoded.macro_used, (True, False, False))

    def test_implicit_requires_and_replays_sidecar(self) -> None:
        encoded = encode_phrases(
            self.phrases, self.macros, boundary_mode="implicit_sidecar"
        )
        self.assertEqual(
            decode_implicit_with_sidecar(encoded.tokens, encoded.phrase_spans, self.macros),
            self.expected,
        )
        with self.assertRaisesRegex(AnchoredMotifModelSurfaceError, "prefix"):
            decode_explicit_sequence(encoded.tokens, self.macros)

    def test_only_fallback_phrases_need_one_prefix(self) -> None:
        encoded = encode_phrases(
            self.phrases, self.macros, boundary_mode="fallback_single_prefix"
        )
        self.assertEqual(
            decode_fallback_prefixed_sequence(encoded.tokens, self.macros), self.expected
        )
        self.assertEqual(encoded.tokens.count("<MOST:FALLBACK>"), 2)
        self.assertNotIn("<MOST:MOTIF>", encoded.tokens)

    def test_single_fallback_suffix_preserves_old_carrier_position(self) -> None:
        encoded = encode_frozen_phrases(self.phrases, self.macros)
        self.assertEqual(
            decode_fallback_suffixed_sequence(encoded.tokens, self.macros), self.expected
        )
        self.assertEqual(encoded.tokens.count("<MOST:FALLBACK_END>"), 2)
        for used_macro, (_start, stop) in zip(encoded.macro_used, encoded.phrase_spans):
            carrier = encoded.tokens[stop - 1]
            self.assertTrue(
                carrier.startswith("<MOST:MACRO:") if used_macro else carrier == "<MOST:FALLBACK_END>"
            )
        self.assertEqual(
            frozen_grammar_contract()["boundary_mode"], "fallback_single_suffix"
        )

    def test_frozen_decoder_rejects_missing_or_empty_fallback_suffix(self) -> None:
        fallback = encode_frozen_phrases(
            ({"pure_motif": "[O]", "anchors": []},), self.macros
        ).tokens
        with self.assertRaisesRegex(AnchoredMotifModelSurfaceError, "missing its suffix"):
            decode_fallback_suffixed_sequence(fallback[:-1], self.macros)
        with self.assertRaisesRegex(AnchoredMotifModelSurfaceError, "no chemical identity"):
            decode_fallback_suffixed_sequence(("<MOST:FALLBACK_END>",), self.macros)
        with self.assertRaisesRegex(AnchoredMotifModelSurfaceError, "missing its suffix"):
            decode_fallback_suffixed_sequence(
                (*fallback[:-1], "<1*>", "<MOST:FALLBACK_END>"), self.macros
            )

    def test_broken_or_missing_sidecar_is_rejected(self) -> None:
        encoded = encode_phrases(
            self.phrases, self.macros, boundary_mode="implicit_sidecar"
        )
        with self.assertRaisesRegex(AnchoredMotifModelSurfaceError, "cover"):
            decode_implicit_with_sidecar(encoded.tokens, encoded.phrase_spans[:-1], self.macros)


if __name__ == "__main__":
    unittest.main()
