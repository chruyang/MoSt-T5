from __future__ import annotations

import unittest

from most_t5_next.p1 import (
    ATOM_ALIGNED_E3FP,
    ATOM_SELFIES_IDENTITY,
    BASE_T5_INPUT_KEYS,
    GEOMETRY_INPUT_KEYS,
    HYBRID_MOTIF_IDENTITY,
    MOTIF_MEAN_E3FP,
    NO_GEOMETRY,
    FourGridContractError,
    P1_CONDITION_SPECS,
    get_p1_condition_spec,
)


class FourGridConfigurationTest(unittest.TestCase):
    def test_grid_is_the_exact_two_by_two_comparison(self):
        self.assertEqual(set(P1_CONDITION_SPECS), {"A0", "A1", "M0", "M1"})
        a0, a1 = get_p1_condition_spec("A0"), get_p1_condition_spec("A1")
        m0, m1 = get_p1_condition_spec("M0"), get_p1_condition_spec("M1")

        self.assertEqual(a0.identity_representation, ATOM_SELFIES_IDENTITY)
        self.assertEqual(a1.identity_representation, ATOM_SELFIES_IDENTITY)
        self.assertEqual(m0.identity_representation, HYBRID_MOTIF_IDENTITY)
        self.assertEqual(m1.identity_representation, HYBRID_MOTIF_IDENTITY)
        self.assertEqual(a0.geometry_condition, NO_GEOMETRY)
        self.assertEqual(m0.geometry_condition, NO_GEOMETRY)
        self.assertEqual(a1.geometry_condition, ATOM_ALIGNED_E3FP)
        self.assertEqual(m1.geometry_condition, MOTIF_MEAN_E3FP)

    def test_geometry_is_an_explicit_wrapper_side_interface(self):
        for condition_id in ("A0", "A1", "M0", "M1"):
            self.assertEqual(
                get_p1_condition_spec(condition_id).t5_input_keys,
                BASE_T5_INPUT_KEYS,
            )
        self.assertEqual(get_p1_condition_spec("A0").geometry_input_keys, ())
        self.assertEqual(get_p1_condition_spec("M0").geometry_input_keys, ())
        self.assertEqual(
            get_p1_condition_spec("A1").wrapper_input_keys,
            BASE_T5_INPUT_KEYS + GEOMETRY_INPUT_KEYS,
        )
        self.assertEqual(
            get_p1_condition_spec("M1").wrapper_input_keys,
            BASE_T5_INPUT_KEYS + GEOMETRY_INPUT_KEYS,
        )
        self.assertEqual(
            get_p1_condition_spec("A1").atom_to_carrier_cardinality,
            "one_atom_per_carrier",
        )
        self.assertEqual(
            get_p1_condition_spec("M1").atom_to_carrier_cardinality,
            "many_atoms_per_motif_carrier",
        )

    def test_aliases_do_not_silently_change_a_grid_cell(self):
        for invalid in ("a0", "C1-G", "M1+teacher", ""):
            with self.subTest(invalid=invalid), self.assertRaises(FourGridContractError):
                get_p1_condition_spec(invalid)


if __name__ == "__main__":
    unittest.main()
