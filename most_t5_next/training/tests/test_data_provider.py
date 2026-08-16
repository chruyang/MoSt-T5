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

    def test_tasks_can_use_distinct_physical_partitions_of_one_logical_batch(self) -> None:
        sampler = CurriculumBatchSampler(
            phase=2,
            total_updates=4,
            populations={task: range(200) for task in ("SYN", "TXT", "CAP", "T2M")},
            micro_batch_size=96,
            accumulation_steps=1,
            effective_batch_size=96,
            task_partitions={
                "SYN": (96, 1),
                "TXT": (48, 2),
                "CAP": (48, 2),
                "T2M": (48, 2),
            },
            seed=42,
        )
        batches = list(sampler)
        self.assertEqual([len(batch) for batch in batches], [96, 48, 48, 48, 48, 48, 48])
        by_update: dict[int, list[CurriculumIndex]] = {}
        for batch in batches:
            by_update.setdefault(batch[0].update, []).extend(batch)
        self.assertEqual({update: len(rows) for update, rows in by_update.items()}, {0: 96, 1: 96, 2: 96, 3: 96})
        self.assertEqual(sampler.partition_for_task("SYN"), (96, 1))
        self.assertEqual(sampler.partition_for_task("T2M"), (48, 2))

    def test_fixed_task_replicas_draw_disjoint_logical_batches(self) -> None:
        common = {
            "phase": 1,
            "total_updates": 2,
            "populations": {"M": range(1000)},
            "micro_batch_size": 4,
            "accumulation_steps": 1,
            "effective_batch_size": 4,
            "fixed_task": "M",
            "task_replicas": 2,
            "seed": 42,
        }
        rank_zero = list(
            CurriculumBatchSampler(**common, task_replica_index=0)
        )
        rank_one = list(
            CurriculumBatchSampler(**common, task_replica_index=1)
        )
        self.assertTrue(all(row.task == "M" for batch in rank_zero for row in batch))
        self.assertTrue(all(row.task == "M" for batch in rank_one for row in batch))
        self.assertEqual([batch[0].update for batch in rank_zero], [0, 1])
        self.assertEqual([batch[0].update for batch in rank_one], [0, 1])
        for update in range(2):
            left = {row.source_index for row in rank_zero[update]}
            right = {row.source_index for row in rank_one[update]}
            self.assertFalse(left & right)

        flattened = [
            row.source_index
            for update in range(2)
            for batch in (rank_zero[update], rank_one[update])
            for row in batch
        ]
        reference = list(
            CurriculumBatchSampler(
                phase=1,
                total_updates=4,
                populations={"M": range(1000)},
                micro_batch_size=4,
                accumulation_steps=1,
                effective_batch_size=4,
                fixed_task="M",
                seed=42,
            )
        )
        self.assertEqual(
            flattened,
            [row.source_index for batch in reference for row in batch],
        )

    def test_fixed_task_rejects_invalid_replica_coordinates(self) -> None:
        with self.assertRaisesRegex(Exception, "replica coordinates"):
            CurriculumBatchSampler(
                phase=2,
                total_updates=4,
                populations={"SYN": range(10)},
                micro_batch_size=2,
                accumulation_steps=1,
                fixed_task="SYN",
                task_replica_index=2,
                task_replicas=2,
                seed=42,
            )

    def test_fixed_task_does_not_require_a_complete_legacy_task_cycle(self) -> None:
        sampler = CurriculumBatchSampler(
            phase=2,
            total_updates=1,
            populations={"TXT": range(10)},
            micro_batch_size=2,
            accumulation_steps=1,
            fixed_task="TXT",
            seed=42,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 1)
        self.assertTrue(all(row.task == "TXT" for row in batches[0]))

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
