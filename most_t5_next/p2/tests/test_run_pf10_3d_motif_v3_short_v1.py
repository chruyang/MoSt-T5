from __future__ import annotations

import unittest

from most_t5_next.p2.run_pf10_3d_motif_v3_short_v1 import (
    PROTOCOL,
    ThreeDMotifV3ShortScreenError,
    VIEW_CYCLE,
    view_for_update,
)


class ThreeDMotifV3ShortScreenTest(unittest.TestCase):
    def test_geometry_cells_share_the_frozen_view_cycle(self) -> None:
        expected = VIEW_CYCLE * 2
        for cell in ("B2D", "F3D"):
            self.assertEqual(
                tuple(view_for_update(cell, update) for update in range(1, 9)),
                expected,
            )
        self.assertEqual(
            tuple(view_for_update("B0", update) for update in range(1, 9)),
            ("m_only",) * 8,
        )

    def test_screen_keeps_effective_batch_128_and_exact_view_counts(self) -> None:
        self.assertEqual(PROTOCOL.effective_batch_size, 128)
        counts = {
            view: sum(
                view_for_update("F3D", update) == view
                for update in range(1, PROTOCOL.total_updates + 1)
            )
            for view in ("m_only", "m_plus_g", "g_only")
        }
        self.assertEqual(
            counts,
            {"m_only": 250, "m_plus_g": 500, "g_only": 250},
        )

    def test_invalid_schedule_arguments_are_rejected(self) -> None:
        with self.assertRaises(ThreeDMotifV3ShortScreenError):
            view_for_update("unknown", 1)
        with self.assertRaises(ThreeDMotifV3ShortScreenError):
            view_for_update("F3D", 0)


if __name__ == "__main__":
    unittest.main()
