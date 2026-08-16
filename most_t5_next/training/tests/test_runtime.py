from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.training.runtime import (
    TrainingRuntimeConfig,
    autocast_context,
    optimization_from_config,
    runtime_from_config,
    seed_everything,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pretrain.yaml"


class RuntimeTest(unittest.TestCase):
    def test_runtime_consumes_open_hardware_parameters(self) -> None:
        config = load_pretraining_config(CONFIG_PATH)
        runtime = runtime_from_config(config)
        self.assertEqual(runtime.seed, 42)
        self.assertEqual(runtime.precision, "bf16")
        self.assertEqual(runtime.effective_batch_size, 96)
        self.assertEqual(runtime.num_workers, 8)

    def test_phase_optimization_consumes_launch_values(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["curriculum"]["phase_one"]["total_updates"] = 20_000
        config["optimization"]["phase_one"]["base_learning_rate"] = 1.0e-3
        config["optimization"]["warmup_start_factor"] = 0.5
        config["optimization"]["final_learning_rate"] = 1.0e-5
        optimization = optimization_from_config(config, "phase_one")
        self.assertEqual(optimization.total_updates, 20_000)
        self.assertEqual(optimization.warmup_updates, 10_000)
        self.assertEqual(optimization.base_learning_rate, 1.0e-3)

    def test_seed_everything_is_reproducible(self) -> None:
        seed_everything(42)
        first = (random.random(), float(np.random.rand()), float(torch.rand(())))
        seed_everything(42)
        second = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(first, second)

    def test_precision_values_are_explicit(self) -> None:
        for precision in ("fp32", "bf16", "fp16"):
            TrainingRuntimeConfig(precision=precision)
            with autocast_context(precision, torch.device("cpu")):
                torch.ones(1).add_(1)
        with self.assertRaisesRegex(ValueError, "precision"):
            TrainingRuntimeConfig(precision="tf32")

    def test_single_process_loading_is_a_valid_public_setting(self) -> None:
        self.assertEqual(TrainingRuntimeConfig(num_workers=0).num_workers, 0)


if __name__ == "__main__":
    unittest.main()
