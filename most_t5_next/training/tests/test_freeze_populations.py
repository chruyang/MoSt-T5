from __future__ import annotations

import unittest

from most_t5_next.training.freeze_populations import required_max_epoch


class PopulationEpochTest(unittest.TestCase):
    def test_epoch_budget_uses_optimizer_task_updates(self) -> None:
        self.assertEqual(
            required_max_epoch(
                100, task_updates=10, micro_batch_size=8, accumulation_steps=2
            ),
            1,
        )

    def test_epoch_budget_is_independent_of_microbatch_partition(self) -> None:
        expected = required_max_epoch(
            101, task_updates=10, micro_batch_size=96, accumulation_steps=1
        )
        self.assertEqual(expected, 9)
        self.assertEqual(
            required_max_epoch(
                101, task_updates=10, micro_batch_size=48, accumulation_steps=2
            ),
            expected,
        )
        self.assertEqual(
            required_max_epoch(
                101, task_updates=10, micro_batch_size=32, accumulation_steps=3
            ),
            expected,
        )

    def test_replicated_task_ranks_extend_the_exposure_scan(self) -> None:
        per_rank_updates = 10
        task_replicas = 2
        self.assertEqual(
            required_max_epoch(
                100,
                task_updates=per_rank_updates * task_replicas,
                micro_batch_size=96,
                accumulation_steps=1,
            ),
            19,
        )


if __name__ == "__main__":
    unittest.main()
