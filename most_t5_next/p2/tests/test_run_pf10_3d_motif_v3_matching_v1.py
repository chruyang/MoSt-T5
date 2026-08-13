from __future__ import annotations

import hashlib
import unittest

from most_t5_next.p2.run_pf10_3d_motif_v3_matching_v1 import (
    GEOMETRY_FRACTION,
    MATCHING_LOSS_WEIGHT,
    PROTOCOL,
    ThreeDMotifV3MatchingError,
    _eligible_anchor_count,
    view_for_update,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ThreeDMotifV3MatchingRunnerTest(unittest.TestCase):
    def test_protocol_is_paired_and_lower_amplitude(self) -> None:
        self.assertEqual(PROTOCOL.effective_batch_size, 128)
        self.assertEqual(PROTOCOL.total_updates, 1000)
        self.assertEqual(PROTOCOL.base_learning_rate, 1.0e-4)
        self.assertEqual(GEOMETRY_FRACTION, 0.15)
        self.assertEqual(MATCHING_LOSS_WEIGHT, 0.25)
        counts = {
            view: sum(
                view_for_update(update) == view
                for update in range(1, PROTOCOL.total_updates + 1)
            )
            for view in ("m_only", "m_plus_g", "g_only")
        }
        self.assertEqual(counts, {"m_only": 250, "m_plus_g": 500, "g_only": 250})

    def test_matching_coverage_requires_another_record(self) -> None:
        shared = _digest("shared")
        identities = (
            (shared, _digest("same-row"), _digest("same-row")),
            (shared, _digest("unique")),
        )
        self.assertEqual(_eligible_anchor_count(identities), 2)

    def test_invalid_update_is_rejected(self) -> None:
        with self.assertRaises(ThreeDMotifV3MatchingError):
            view_for_update(0)


if __name__ == "__main__":
    unittest.main()
