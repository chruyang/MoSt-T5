from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from most_t5_next.p1.pf1_optimization import G_CODEC_PF1_PROTOCOL
from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
    ZeroInitGatedE3FPCarrierFusion,
)
from most_t5_next.p2.run_pf2_frozen_topology_adapter_v1 import (
    _load_source_manifest,
    build_frozen_adapter_optimizer,
)


class FrozenTopologyAdapterTest(unittest.TestCase):
    def test_optimizer_freezes_backbone_and_keeps_only_two_adapter_surfaces(self) -> None:
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(3, 3)
        model.geometry_fusion = ZeroInitGatedE3FPCarrierFusion(
            num_e3fp_embeddings=16,
            hidden_size=3,
        )
        optimizer = build_frozen_adapter_optimizer(model, G_CODEC_PF1_PROTOCOL)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(model.geometry_fusion.shared_embedding.weight.requires_grad)
        self.assertTrue(model.geometry_fusion.geometry_gate_logit.requires_grad)
        self.assertEqual(sum(len(group["params"]) for group in optimizer.param_groups), 2)

    def test_source_manifest_requires_one_completed_m0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "conditions": [
                            {
                                "condition": "M0",
                                "evaluations": [
                                    {
                                        "update": 1000,
                                        "token_weighted_nll": 1.5,
                                        "masked_token_accuracy": 0.6,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            _payload, baseline = _load_source_manifest(path)
            self.assertEqual(
                baseline,
                {"token_weighted_nll": 1.5, "masked_token_accuracy": 0.6},
            )


if __name__ == "__main__":
    unittest.main()
