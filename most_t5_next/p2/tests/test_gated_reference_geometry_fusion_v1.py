from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from most_t5_next.p1.four_grid_t5_wrapper import FourGridT5Wrapper
    from most_t5_next.p1.shared_geometry_fusion import GeometryTensorSidecar
    from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
        ZeroInitGatedE3FPCarrierFusion,
    )


@dataclass
class _Config:
    vocab_size: int
    d_model: int


if TORCH_AVAILABLE:

    class _TinyT5(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Config(vocab_size=11, d_model=2)
            self.shared = nn.Embedding(11, 2)
            self.lm_head = nn.Linear(2, 11, bias=False)

        def get_input_embeddings(self):
            return self.shared

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, **kwargs):
            return kwargs


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ZeroInitGatedGeometryFusionTest(unittest.TestCase):
    def setUp(self):
        self.module = ZeroInitGatedE3FPCarrierFusion(
            num_e3fp_embeddings=8,
            hidden_size=2,
        )
        with torch.no_grad():
            self.module.shared_embedding.weight.zero_()
            values = torch.arange(1, 9, dtype=torch.float32)
            self.module.shared_embedding.weight[1:, 0] = values
            self.module.shared_embedding.weight[1:, 1] = -values

    @staticmethod
    def _geometry(ids, carriers):
        atom_count = len(ids[0])
        return GeometryTensorSidecar(
            e3fp_ids=torch.tensor(ids, dtype=torch.long),
            e3fp_atom_mask=torch.ones((1, atom_count), dtype=torch.bool),
            e3fp_atom_to_token=torch.tensor(carriers, dtype=torch.long),
        )

    def test_zero_init_is_bitwise_identity_for_carriers_and_noncarriers(self):
        inputs = torch.tensor(
            [[[2.0, 4.0], [10.0, 20.0], [6.0, 8.0]]],
            dtype=torch.float32,
        )
        geometry = self._geometry([[[0, 2, -1, -1]]], [[1]])
        output = self.module(
            inputs,
            geometry,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
        self.assertTrue(torch.equal(output, inputs))
        self.assertEqual(float(self.module.effective_geometry_gate.item()), 0.0)

    def test_open_gate_uses_reference_reduction_only_at_carriers(self):
        with torch.no_grad():
            self.module.geometry_gate_logit.fill_(torch.atanh(torch.tensor(0.5)))
        inputs = torch.tensor(
            [[[2.0, 4.0], [10.0, 20.0], [6.0, 8.0]]],
            dtype=torch.float32,
        )
        geometry = self._geometry([[[0, 2, -1, -1]]], [[1]])
        output = self.module(
            inputs,
            geometry,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
        # The reference geometry is [1,-1]; gate 0.5 adds [0.5,-0.5].
        self.assertTrue(torch.equal(output[0, 1], torch.tensor([10.5, 19.5])))
        self.assertTrue(torch.equal(output[0, 0], inputs[0, 0]))
        self.assertTrue(torch.equal(output[0, 2], inputs[0, 2]))

    def test_first_backward_opens_gate_before_updating_geometry_table(self):
        inputs = torch.zeros((1, 2, 2), dtype=torch.float32, requires_grad=True)
        geometry = self._geometry([[[0, -1, -1, -1]]], [[1]])
        output = self.module(
            inputs,
            geometry,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )
        output[0, 1, 0].backward()
        self.assertNotEqual(float(self.module.geometry_gate_logit.grad.item()), 0.0)
        self.assertTrue(
            torch.equal(
                self.module.shared_embedding.weight.grad,
                torch.zeros_like(self.module.shared_embedding.weight.grad),
            )
        )

        self.module.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.module.geometry_gate_logit.fill_(0.2)
        self.module(
            inputs.detach(),
            geometry,
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )[0, 1, 0].backward()
        self.assertGreater(
            float(self.module.shared_embedding.weight.grad.abs().sum().item()),
            0.0,
        )

    def test_all_cells_share_one_parameter_schema(self):
        models = {
            cell: FourGridT5Wrapper(
                _TinyT5(),
                condition_id=cell,
                num_e3fp_embeddings=8,
                geometry_fusion_factory=ZeroInitGatedE3FPCarrierFusion,
            )
            for cell in ("A0", "A1", "M0", "M1")
        }
        reference = {
            key: tuple(value.shape)
            for key, value in models["M0"].state_dict().items()
        }
        self.assertEqual(
            reference["geometry_fusion.geometry_gate_logit"],
            (1,),
        )
        self.assertIn("geometry_fusion.shared_embedding.weight", reference)
        for model in models.values():
            self.assertEqual(
                {key: tuple(value.shape) for key, value in model.state_dict().items()},
                reference,
            )


if __name__ == "__main__":
    unittest.main()
