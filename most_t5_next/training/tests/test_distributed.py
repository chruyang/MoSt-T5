from pathlib import Path
import unittest

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.training.distributed import (
    DistributedLayoutError,
    rank_task_assignment,
    task_batch_partitions,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pretrain.yaml"


class DistributedLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_pretraining_config(CONFIG_PATH)

    def test_phase_one_assigns_two_disjoint_replicas_per_task(self) -> None:
        assignments = [
            rank_task_assignment(self.config, phase=1, rank=rank, world_size=4)
            for rank in range(4)
        ]
        self.assertEqual([row.task for row in assignments], ["M", "M", "MG", "MG"])
        self.assertEqual(
            [row.task_replica_index for row in assignments], [0, 1, 0, 1]
        )
        self.assertEqual([row.task_replicas for row in assignments], [2, 2, 2, 2])

    def test_phase_two_assigns_one_task_and_logical_batch_per_rank(self) -> None:
        assignments = [
            rank_task_assignment(self.config, phase=2, rank=rank, world_size=4)
            for rank in range(4)
        ]
        self.assertEqual(
            [row.task for row in assignments], ["SYN", "TXT", "CAP", "T2M"]
        )
        self.assertEqual(task_batch_partitions(self.config, phase=2), {
            "SYN": (48, 2),
            "TXT": (48, 2),
            "CAP": (32, 3),
            "T2M": (32, 3),
        })

    def test_world_size_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(DistributedLayoutError, "world_size=4"):
            rank_task_assignment(self.config, phase=2, rank=0, world_size=2)


if __name__ == "__main__":
    unittest.main()
