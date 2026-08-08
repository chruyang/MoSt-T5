from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
        ZeroInitGatedE3FPCarrierFusion,
    )
    from most_t5_next.p2.validate_pf2_gated_fusion_gpu_smoke_v1 import (
        FGateSmokeError,
        SMOKE_BATCH_SIZE,
        _require_same_state,
        _require_zero_gate_gradient_boundary,
        build_parser,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PF2GatedFusionGPUSmokeContractTest(unittest.TestCase):
    def test_cli_has_no_training_budget_or_optimizer_controls(self):
        args = build_parser().parse_args([
            "--paired-release", "paired",
            "--base-model-snapshot", "model",
            "--base-tokenizer-snapshot", "tokenizer",
            "--union-init-dir", "init",
            "--output", "report.json",
            "--geometry-fusion-seed", "7",
        ])
        self.assertEqual(SMOKE_BATCH_SIZE, 2)
        self.assertFalse(hasattr(args, "optimizer"))
        self.assertFalse(hasattr(args, "updates"))

    def test_state_parity_rejects_one_changed_parameter(self):
        left = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        _require_same_state(left, {key: value.clone() for key, value in left.items()})
        with self.assertRaisesRegex(FGateSmokeError, "initialization"):
            _require_same_state(left, {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])})

    def test_first_backward_requires_gate_only_before_it_opens(self):
        module = ZeroInitGatedE3FPCarrierFusion(
            num_e3fp_embeddings=8,
            hidden_size=2,
        )
        module.geometry_gate_logit.grad = torch.tensor([0.25])
        module.shared_embedding.weight.grad = torch.zeros_like(
            module.shared_embedding.weight
        )
        report = _require_zero_gate_gradient_boundary(
            type("Wrapper", (), {"geometry_fusion": module})()
        )
        self.assertEqual(report["e3fp_table_gradient_l1"], 0.0)

        module.shared_embedding.weight.grad[1, 0] = 1.0
        with self.assertRaisesRegex(FGateSmokeError, "before.*opened"):
            _require_zero_gate_gradient_boundary(
                type("Wrapper", (), {"geometry_fusion": module})()
            )


if __name__ == "__main__":
    unittest.main()
