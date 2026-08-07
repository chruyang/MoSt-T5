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
    from most_t5_next.p2.reference_geometry_fusion_v1 import (
        ReferenceE3FPCarrierFusion,
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
class ReferenceGeometryFusionTest(unittest.TestCase):
    def setUp(self):
        self.module = ReferenceE3FPCarrierFusion(
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

    def test_fixed_four_shell_mean_and_balanced_carrier(self):
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
        # Shared rows 1 and 3 sum to [4,-4], then divide by four slots.
        self.assertTrue(
            torch.equal(output[0, 1], torch.tensor([5.5, 9.5]))
        )
        self.assertTrue(torch.equal(output[0, 0], inputs[0, 0]))
        self.assertTrue(torch.equal(output[0, 2], inputs[0, 2]))

    def test_atoms_on_one_motif_carrier_are_mean_pooled(self):
        inputs = torch.zeros((1, 3, 2), dtype=torch.float32)
        geometry = self._geometry(
            [[[0, -1, -1, -1], [4, -1, -1, -1]]],
            [[1, 1]],
        )
        output = self.module(
            inputs,
            geometry,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )
        # Atom states are 1/4 and 5/4; motif mean is 3/4; fusion adds half.
        self.assertTrue(
            torch.equal(output[0, 1], torch.tensor([0.375, -0.375]))
        )

    def test_one_table_and_shell_exchangeability_match_reference(self):
        first = self._geometry([[[1, 5, -1, -1]]], [[1]])
        swapped = self._geometry([[[5, 1, -1, -1]]], [[1]])
        inputs = torch.zeros((1, 3, 2), dtype=torch.float32)
        mask = torch.ones((1, 3), dtype=torch.long)
        self.assertTrue(
            torch.equal(
                self.module(inputs, first, attention_mask=mask),
                self.module(inputs, swapped, attention_mask=mask),
            )
        )
        self.assertEqual(len(tuple(self.module.parameters())), 1)
        self.assertEqual(sum(p.numel() for p in self.module.parameters()), 18)

    def test_wrapper_factory_preserves_one_schema_across_cells(self):
        models = {
            cell: FourGridT5Wrapper(
                _TinyT5(),
                condition_id=cell,
                num_e3fp_embeddings=8,
                geometry_fusion_factory=ReferenceE3FPCarrierFusion,
            )
            for cell in ("A0", "A1", "M0", "M1")
        }
        reference = {
            key: tuple(value.shape)
            for key, value in models["A0"].state_dict().items()
        }
        self.assertIn("geometry_fusion.shared_embedding.weight", reference)
        self.assertFalse(any("level_embeddings" in key for key in reference))
        for model in models.values():
            self.assertEqual(
                {key: tuple(value.shape) for key, value in model.state_dict().items()},
                reference,
            )


if __name__ == "__main__":
    unittest.main()
