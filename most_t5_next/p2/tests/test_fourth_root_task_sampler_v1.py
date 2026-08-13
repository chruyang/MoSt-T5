from __future__ import annotations

import json
import unittest

from most_t5_next.p2.fourth_root_task_sampler_v1 import (
    FourthRootTaskSamplerError,
    FourthRootTaskSamplerV1,
    fourth_root_sampling_rows,
)


POPULATIONS = {"caption": 10_000, "computed": 1_000_000, "descriptive": 100_000}


class FourthRootTaskSamplerV1Tests(unittest.TestCase):
    def test_probabilities_are_exactly_normalized_fourth_roots(self) -> None:
        rows = fourth_root_sampling_rows({"small": 1, "large": 16})
        self.assertAlmostEqual(rows[0].raw_weight, 2.0)
        self.assertAlmostEqual(rows[1].raw_weight, 1.0)
        self.assertAlmostEqual(sum(row.probability for row in rows), 1.0)
        self.assertAlmostEqual(rows[0].probability, 2.0 / 3.0)

    def test_exact_resume_preserves_task_and_cursor_sequence(self) -> None:
        uninterrupted = FourthRootTaskSamplerV1(
            POPULATIONS, seed=20260813, examples_per_microbatch=32
        )
        prefix = uninterrupted.draw_many(137)
        state = uninterrupted.state_dict()
        suffix = uninterrupted.draw_many(500)

        resumed = FourthRootTaskSamplerV1(
            POPULATIONS, seed=20260813, examples_per_microbatch=32
        )
        # Exercise the JSON-compatible list form used by manifests.
        json_state = json.loads(json.dumps(state))
        resumed.load_state_dict(json_state)
        self.assertEqual(resumed.draw_many(500), suffix)
        self.assertEqual(resumed.state_dict(), uninterrupted.state_dict())
        self.assertEqual(prefix[0].draw_index, 0)

    def test_realized_long_run_mixture_tracks_probabilities(self) -> None:
        sampler = FourthRootTaskSamplerV1(
            POPULATIONS, seed=7, examples_per_microbatch=16
        )
        sampler.draw_many(50_000)
        report = sampler.report()
        counts = report["selection_counts"]
        expected = {row.task_id: row.probability for row in sampler.rows}
        for task_id, probability in expected.items():
            realized = counts[task_id] / 50_000
            self.assertLess(abs(realized - probability), 0.01)

    def test_cursor_and_pass_counts_include_microbatch_boundary_wrap(self) -> None:
        sampler = FourthRootTaskSamplerV1(
            {"only": 10}, seed=0, examples_per_microbatch=6
        )
        first, second = sampler.draw_many(2)
        self.assertEqual((first.task_cursor, first.task_pass_index), (0, 0))
        self.assertEqual((second.task_cursor, second.task_pass_index), (6, 0))
        report = sampler.report()
        self.assertEqual(report["task_cursors"]["only"], 2)
        self.assertEqual(report["completed_passes"]["only"], 1)

    def test_changed_contract_or_corrupt_cursor_is_rejected(self) -> None:
        sampler = FourthRootTaskSamplerV1(
            POPULATIONS, seed=1, examples_per_microbatch=8
        )
        sampler.draw_many(10)
        state = sampler.state_dict()
        changed = FourthRootTaskSamplerV1(
            POPULATIONS, seed=1, examples_per_microbatch=16
        )
        with self.assertRaises(FourthRootTaskSamplerError):
            changed.load_state_dict(state)
        bad = dict(state)
        bad["task_cursors"] = dict(state["task_cursors"])
        bad["task_cursors"]["caption"] += 1
        with self.assertRaises(FourthRootTaskSamplerError):
            sampler.load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()
