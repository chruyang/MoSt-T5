from __future__ import annotations

import unittest
from unittest.mock import patch

from most_t5_next.training.data_provider import (
    CurriculumBatchSampler,
    CurriculumCollator,
    CurriculumIndex,
    TaggedSample,
)


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

    def test_optimizer_batches_are_independent_of_physical_partition(self) -> None:
        populations = {
            task: range(101) for task in ("SYN", "TXT", "CAP", "T2M")
        }

        def logical_batches(
            micro_batch_size: int, accumulation_steps: int
        ) -> list[tuple[tuple[int, int], ...]]:
            sampler = CurriculumBatchSampler(
                phase=2,
                total_updates=8,
                populations=populations,
                micro_batch_size=micro_batch_size,
                accumulation_steps=accumulation_steps,
                effective_batch_size=96,
                seed=42,
            )
            microbatches = list(sampler)
            self.assertTrue(
                all(len(batch) == micro_batch_size for batch in microbatches)
            )
            return [
                tuple(
                    (row.source_index, row.epoch)
                    for batch in microbatches[
                        update * accumulation_steps : (update + 1)
                        * accumulation_steps
                    ]
                    for row in batch
                )
                for update in range(8)
            ]

        unsplit = logical_batches(96, 1)
        split_in_two = logical_batches(48, 2)
        split_in_three = logical_batches(32, 3)
        self.assertEqual(split_in_two, unsplit)
        self.assertEqual(split_in_three, unsplit)
        self.assertTrue(all(len(batch) == 96 for batch in unsplit))
        self.assertEqual({epoch for _, epoch in unsplit[4]}, {0, 1})
        first_syn_epoch = [
            source_index
            for source_index, epoch in (*unsplit[0], *unsplit[4])
            if epoch == 0
        ]
        self.assertEqual(len(first_syn_epoch), 101)
        self.assertEqual(set(first_syn_epoch), set(range(101)))

    def test_resume_preserves_logical_batches_across_epoch_wrap(self) -> None:
        kwargs = dict(
            phase=2,
            total_updates=8,
            populations={
                task: range(101) for task in ("SYN", "TXT", "CAP", "T2M")
            },
            micro_batch_size=32,
            accumulation_steps=3,
            effective_batch_size=96,
            seed=42,
        )
        full = list(CurriculumBatchSampler(**kwargs))
        resumed = list(CurriculumBatchSampler(**kwargs, start_update=4))
        self.assertEqual(resumed, full[12:])

    @patch(
        "most_t5_next.training.data_provider.collate_phase2_cap_samples",
        return_value={},
    )
    def test_collator_accepts_one_microbatch_spanning_epochs(self, mocked) -> None:
        rows = [
            TaggedSample(CurriculumIndex("CAP", 100, 0, 4, 0), object()),
            TaggedSample(CurriculumIndex("CAP", 7, 1, 4, 0), object()),
        ]
        batch = CurriculumCollator(0, (1,), 2, 42)(rows)
        self.assertEqual(batch["curriculum_task"], "CAP")
        self.assertEqual(batch["curriculum_update"], 4)
        self.assertEqual(batch["curriculum_microbatch"], 0)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
