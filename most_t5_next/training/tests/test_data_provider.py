from __future__ import annotations

import unittest

from most_t5_next.training.data_provider import CurriculumBatchSampler


class CurriculumBatchSamplerTest(unittest.TestCase):
    def test_phase_one_is_task_homogeneous_at_the_update_boundary(self) -> None:
        sampler = CurriculumBatchSampler(
            phase=1,
            total_updates=4,
            populations={"M": range(8), "MG": range(8)},
            micro_batch_size=2,
            accumulation_steps=2,
            seed=42,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 8)
        signatures = [
            (batch[0].task, batch[0].update, batch[0].microbatch)
            for batch in batches
        ]
        self.assertEqual(
            signatures,
            [
                ("M", 0, 0),
                ("M", 0, 1),
                ("MG", 1, 0),
                ("MG", 1, 1),
                ("M", 2, 0),
                ("M", 2, 1),
                ("MG", 3, 0),
                ("MG", 3, 1),
            ],
        )
        self.assertTrue(all(len(batch) == 2 for batch in batches))

    def test_sampler_resume_skips_completed_updates_deterministically(self) -> None:
        kwargs = dict(
            phase=2,
            total_updates=8,
            populations={task: range(16) for task in ("SYN", "TXT", "CAP", "T2M")},
            micro_batch_size=2,
            accumulation_steps=1,
            seed=42,
        )
        full = list(CurriculumBatchSampler(**kwargs))
        resumed = list(CurriculumBatchSampler(**kwargs, start_update=4))
        self.assertEqual(resumed, full[4:])


if __name__ == "__main__":
    unittest.main()
