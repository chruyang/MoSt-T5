from __future__ import annotations

from pathlib import Path
import unittest

from most_t5_next.configuration import load_pretraining_config
from scripts.pretrain import _execution_config


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pretrain.yaml"


class PretrainingLauncherTest(unittest.TestCase):
    def test_formal_execution_keeps_the_frozen_config(self) -> None:
        formal = load_pretraining_config(CONFIG_PATH, require_launch_values=True)
        resolved, mode = _execution_config(
            formal,
            smoke_updates_per_phase=None,
            smoke_checkpoint_every_updates=2,
        )
        self.assertEqual(mode, "formal")
        self.assertEqual(resolved, formal)
        self.assertIsNot(resolved, formal)

    def test_smoke_execution_is_explicit_and_does_not_mutate_formal(self) -> None:
        formal = load_pretraining_config(CONFIG_PATH, require_launch_values=True)
        resolved, mode = _execution_config(
            formal,
            smoke_updates_per_phase=4,
            smoke_checkpoint_every_updates=2,
        )
        self.assertEqual(mode, "execution_smoke")
        self.assertEqual(resolved["curriculum"]["phase_one"]["total_updates"], 4)
        self.assertEqual(resolved["curriculum"]["phase_two"]["total_updates"], 4)
        self.assertEqual(resolved["optimization"]["phase_one"]["warmup_updates"], 1)
        self.assertEqual(resolved["optimization"]["phase_two"]["warmup_updates"], 1)
        self.assertEqual(resolved["monitoring"]["checkpoint_every_updates"], 2)
        self.assertEqual(formal["curriculum"]["phase_one"]["total_updates"], 100_000)
        self.assertEqual(formal["optimization"]["phase_one"]["warmup_updates"], 10_000)
        self.assertEqual(formal["monitoring"]["checkpoint_every_updates"], 10_000)

    def test_smoke_execution_rejects_invalid_budgets(self) -> None:
        formal = load_pretraining_config(CONFIG_PATH, require_launch_values=True)
        for updates, interval in ((1, 1), (4, 0), (4, 5)):
            with self.subTest(updates=updates, interval=interval):
                with self.assertRaises(ValueError):
                    _execution_config(
                        formal,
                        smoke_updates_per_phase=updates,
                        smoke_checkpoint_every_updates=interval,
                    )


if __name__ == "__main__":
    unittest.main()
