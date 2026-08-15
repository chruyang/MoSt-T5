from collections import Counter
import unittest

from most_t5_next.data.motif_corruption import (
    MotifCorruptionError,
    build_motif_units,
    geometry_visibility,
    select_motif_units,
)


class MotifCorruptionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.units = build_motif_units(
            ((0, 2), (5, 8)),
            {0: ((10, 12),), 1: ((13, 14), (16, 18))},
            (0, 1, 1, 1),
            sequence_length=18,
        )

    def test_fragment_and_owned_endpoints_form_one_compound_unit(self) -> None:
        self.assertEqual(self.units[0].spans, ((0, 2), (10, 12)))
        self.assertEqual(self.units[1].spans, ((5, 8), (13, 14), (16, 18)))
        self.assertEqual([unit.heavy_atom_count for unit in self.units], [1, 3])

    def test_first_draw_is_uniform_over_heavy_atoms(self) -> None:
        counts = Counter(
            select_motif_units(self.units, noise_density=0.1, seed=seed)[0].fragment_id
            for seed in range(4000)
        )
        fraction = counts[1] / sum(counts.values())
        self.assertGreater(fraction, 0.71)
        self.assertLess(fraction, 0.79)

    def test_selected_motif_closes_its_carrier_and_endpoint_geometry(self) -> None:
        fragments, endpoints = geometry_visibility(
            2, (0, 1, 0), (0,), enabled=True
        )
        self.assertEqual(fragments, (False, True))
        self.assertEqual(endpoints, (False, True, False))

    def test_zero_heavy_atom_fragment_fails_the_compact_contract(self) -> None:
        with self.assertRaisesRegex(MotifCorruptionError, "no retained heavy atom"):
            build_motif_units(
                ((0, 2), (3, 5)),
                {},
                (1, 1),
                sequence_length=5,
            )

    def test_unowned_endpoint_fails_before_model_injection(self) -> None:
        with self.assertRaisesRegex(MotifCorruptionError, "valid fragment owner"):
            geometry_visibility(2, (-1, 1), (), enabled=True)


if __name__ == "__main__":
    unittest.main()
