from __future__ import annotations

import unittest

from most_t5_next.r1.tokenizer import derive_clean_p1_motif_census_v1 as census


class CleanP1MotifCensusTests(unittest.TestCase):
    def test_slot_projection_preserves_attachment_position_and_normalizes_ids(self):
        self.assertEqual(
            census.project_slot_template("O=C(<3*>)N<7*>"),
            ("O=C(<*>)N<*>", 2),
        )
        self.assertEqual(
            census.project_slot_template("O=C(<1*>)N<2*>"),
            ("O=C(<*>)N<*>", 2),
        )
        self.assertNotEqual(
            census.project_slot_template("C(<1*>)N")[0],
            census.project_slot_template("CN<1*>")[0],
        )

    def test_malformed_or_noncanonical_anchor_is_rejected(self):
        for fragment in ("C<01*>", "C<x*>", "C<1**>"):
            with self.subTest(fragment=fragment):
                with self.assertRaises(census.CleanMotifCensusError):
                    census.project_slot_template(fragment)

    def test_exact_subtraction_and_slot_aggregation_close(self):
        fragments = ("C(<1*>)N", "C(<9*>)N", "CN<2*>")
        global_census = {
            census.sha256_text(fragment): (fragment, count)
            for fragment, count in zip(fragments, (4, 3, 5))
        }
        removed = {
            census.sha256_text(fragments[0]): 1,
            census.sha256_text(fragments[2]): 5,
        }
        exact, slots, removed_rows = census.derive_rows(global_census, removed)
        self.assertEqual(sum(row["count"] for row in exact), 6)
        self.assertEqual(sum(row["count"] for row in slots), 6)
        self.assertEqual(len(exact), 2)
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["slot_template"], "C(<*>)N")
        self.assertEqual(slots[0]["exact_lexeme_count"], 2)
        self.assertEqual(len(removed_rows), 2)

    def test_subtraction_cannot_exceed_global_count(self):
        fragment = "C<1*>"
        digest = census.sha256_text(fragment)
        with self.assertRaisesRegex(census.CleanMotifCensusError, "negative"):
            census.derive_rows({digest: (fragment, 1)}, {digest: 2})


if __name__ == "__main__":
    unittest.main()
