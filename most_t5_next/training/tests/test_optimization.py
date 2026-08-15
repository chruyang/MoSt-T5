from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from most_t5_next.training.optimization import (
    CosineSchedule,
    OptimizationConfig,
    build_optimizer_and_schedule,
)
from most_t5_next.training.phase_boundary import (
    load_phase_one_weights,
    save_phase_one_weights,
)


class OptimizationTest(unittest.TestCase):
    def test_reference_optimizer_and_cosine_schedule_update_parameters(self) -> None:
        model = nn.Linear(2, 1)
        config = OptimizationConfig(
            total_updates=4,
            warmup_updates=2,
            base_learning_rate=1.0e-3,
            warmup_start_factor=0.5,
            final_learning_rate=1.0e-5,
        )
        optimizer, schedule = build_optimizer_and_schedule(model, config)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5.0e-4)
        before = model.weight.detach().clone()
        model(torch.ones((2, 2))).sum().backward()
        optimizer.step()
        schedule.step()
        self.assertFalse(torch.equal(before, model.weight))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 7.5e-4)

    def test_schedule_matches_3d_molt5_pytorch_composition(self) -> None:
        config = OptimizationConfig(
            total_updates=8,
            warmup_updates=3,
            base_learning_rate=2.0e-2,
            warmup_start_factor=0.5,
            final_learning_rate=1.0e-5,
        )
        ours_optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=2.0e-2)
        ours = CosineSchedule(ours_optimizer, config)
        reference_optimizer = torch.optim.SGD(
            nn.Linear(1, 1).parameters(), lr=2.0e-2
        )
        reference = SequentialLR(
            reference_optimizer,
            schedulers=[
                LinearLR(
                    reference_optimizer,
                    start_factor=0.5,
                    end_factor=1.0,
                    total_iters=3,
                    last_epoch=-1,
                ),
                CosineAnnealingLR(
                    reference_optimizer,
                    T_max=5,
                    eta_min=1.0e-5,
                ),
            ],
            milestones=[3],
        )
        ours_rates = [ours_optimizer.param_groups[0]["lr"]]
        reference_rates = [reference_optimizer.param_groups[0]["lr"]]
        for _ in range(config.total_updates):
            ours_optimizer.step()
            ours.step()
            reference_optimizer.step()
            reference.step()
            ours_rates.append(ours_optimizer.param_groups[0]["lr"])
            reference_rates.append(reference_optimizer.param_groups[0]["lr"])
        self.assertEqual(ours.completed_updates, config.total_updates)
        self.assertEqual(len(ours_rates), len(reference_rates))
        for observed, expected in zip(ours_rates, reference_rates):
            self.assertAlmostEqual(observed, expected)

    def test_phase_boundary_contains_model_weights_but_no_optimizer_state(self) -> None:
        source = nn.Linear(2, 1)
        target = nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase-one.pt"
            save_phase_one_weights(path, source, metadata={"updates": 8})
            payload = torch.load(path, map_location="cpu")
            self.assertNotIn("optimizer_state_dict", payload)
            metadata = load_phase_one_weights(path, target)
        self.assertEqual(metadata, {"updates": 8})
        torch.testing.assert_close(source.weight, target.weight)
        torch.testing.assert_close(source.bias, target.bias)


if __name__ == "__main__":
    unittest.main()
