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


if __name__ == "__main__":
    unittest.main()
