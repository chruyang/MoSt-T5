"""Focused CPU tests for the frozen PF-1 optimizer and schedule."""

from __future__ import annotations

import math
import unittest

import torch

from most_t5_next.p1.pf1_optimization import (
    AdamWScale,
    FROZEN_PF1_PROTOCOL,
    PF1LearningRateSchedule,
    PF1OptimizationProtocol,
    clip_pf1_gradients,
    learning_rate_for_update,
)


class PF1OptimizationTest(unittest.TestCase):
    def test_frozen_protocol_has_the_preregistered_budget(self) -> None:
        protocol = FROZEN_PF1_PROTOCOL
        self.assertEqual(protocol.base_learning_rate, 1.0e-3)
        self.assertEqual(protocol.warmup_updates, 100)
        self.assertEqual(protocol.total_updates, 1000)
        self.assertEqual(protocol.final_learning_rate, 1.0e-5)
        self.assertEqual(protocol.gradient_clip_norm, 1.0)
        self.assertEqual(protocol.weight_decay, 0.0)
        self.assertEqual(protocol.micro_batch_size, 32)
        self.assertEqual(protocol.gradient_accumulation_steps, 4)
        self.assertEqual(protocol.effective_batch_size, 128)

    def test_warmup_and_cosine_endpoints(self) -> None:
        protocol = FROZEN_PF1_PROTOCOL
        self.assertAlmostEqual(
            learning_rate_for_update(1),
            protocol.base_learning_rate * protocol.warmup_start_factor,
        )
        self.assertAlmostEqual(
            learning_rate_for_update(protocol.warmup_updates),
            protocol.base_learning_rate,
        )
        self.assertAlmostEqual(
            learning_rate_for_update(protocol.total_updates),
            protocol.final_learning_rate,
        )
        self.assertGreater(
            learning_rate_for_update(500),
            learning_rate_for_update(900),
        )

    def test_schedule_state_round_trip_points_to_next_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = AdamWScale([parameter], lr=1.0e-3)
        schedule = PF1LearningRateSchedule(optimizer)
        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"], learning_rate_for_update(1)
        )
        for _ in range(500):
            schedule.step()
        saved = schedule.state_dict()

        other_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        other_optimizer = AdamWScale([other_parameter], lr=1.0e-3)
        restored = PF1LearningRateSchedule(other_optimizer)
        restored.load_state_dict(saved)
        self.assertEqual(restored.completed_updates, 500)
        self.assertAlmostEqual(
            other_optimizer.param_groups[0]["lr"], learning_rate_for_update(501)
        )

    def test_adamwscale_uses_parameter_rms_and_finite_state(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([2.0, 2.0]))
        optimizer = AdamWScale(
            [parameter],
            lr=1.0e-3,
            betas=(0.9, 0.999),
            eps=1.0e-6,
            weight_decay=0.0,
        )
        parameter.grad = torch.ones_like(parameter)
        before = parameter.detach().clone()
        optimizer.step()
        self.assertTrue(torch.all(parameter < before))
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertEqual(optimizer.state[parameter]["step"], 1)

    def test_global_gradient_clip_reports_preclip_norm(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        protocol = PF1OptimizationProtocol(
            gradient_clip_norm=1.0,
            total_updates=2,
            warmup_updates=1,
        )
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 10.0)
        preclip = clip_pf1_gradients(model, protocol)
        postclip = math.sqrt(
            sum(
                float(torch.linalg.vector_norm(parameter.grad).item()) ** 2
                for parameter in model.parameters()
            )
        )
        self.assertGreater(preclip, 1.0)
        self.assertLessEqual(postclip, 1.0 + 1.0e-6)


if __name__ == "__main__":
    unittest.main()
