from __future__ import annotations

import unittest

from most_t5_next.p1.analyze_fragsmiles_macro_locality_v1 import (
    CapPolicy,
    MacroLocalityAnalysisError,
    analyze_registry,
)


def _row(rank: int, identity: str, count: int, role: str = "base") -> dict:
    return {
        "rank": rank,
        "fragment_identity": identity,
        "selection_role": role,
        "phase1_train_occurrences": count,
        "phase2_train_occurrences": 0,
        "chebi20_train_occurrences": 0,
        "uspto50k_train_reaction_component_occurrences": 0,
    }


class MacroLocalityAnalysisV1Tests(unittest.TestCase):
    def test_cap_reports_weighted_coverage_fallback_cost_and_parameters(self):
        report = analyze_registry(
            [_row(0, "C", 10), _row(1, "CCCC", 2, "extension")],
            policies=(CapPolicy("atoms_2", 2, None),),
            hidden_size=8,
        )
        policy = report["policies"][0]
        self.assertEqual(policy["kept_rows"], 1)
        self.assertEqual(policy["removed_rows"], 1)
        self.assertEqual(policy["embedding_parameters"], 8)
        self.assertEqual(policy["parameter_savings_vs_unbounded_tied"], 8)
        self.assertEqual(policy["removed_by_role"], {"extension": 1})
        phase1 = policy["domains"]["phase1"]
        self.assertEqual(phase1["kept_macro_occurrences"], 10)
        self.assertAlmostEqual(phase1["kept_occurrence_rate"], 10 / 12)
        self.assertEqual(phase1["fallback_extra_glyph_tokens"], 6)

    def test_conjunctive_cap_rejects_either_large_dimension(self):
        report = analyze_registry(
            [_row(0, "C", 1), _row(1, "C1CCCCC1", 1)],
            policies=(CapPolicy("both", 10, 4),),
        )
        self.assertEqual(report["policies"][0]["kept_rows"], 1)

    def test_invalid_dense_rank_is_rejected(self):
        with self.assertRaises(MacroLocalityAnalysisError):
            analyze_registry([_row(3, "C", 1)])


if __name__ == "__main__":
    unittest.main()
