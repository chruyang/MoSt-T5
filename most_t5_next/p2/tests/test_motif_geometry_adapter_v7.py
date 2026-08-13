from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class MotifGeometryAdapterV7Test(unittest.TestCase):
    def _adapter(self, cls):
        torch.manual_seed(17)
        return cls(
            num_e3fp_embeddings=32,
            hidden_size=16,
            state_embedding_dim=8,
            atom_memory_dim=12,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
        )

    def test_initial_function_is_exactly_the_fixed_four_mean(self):
        from most_t5_next.p2.motif_geometry_adapter_v5 import MotifGeometryAdapterV5
        from most_t5_next.p2.motif_geometry_adapter_v7 import MotifGeometryAdapterV7

        fixed = self._adapter(MotifGeometryAdapterV5)
        linear = self._adapter(MotifGeometryAdapterV7)
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, -1]]])
        mask = torch.tensor([[True, True]])
        roles = torch.tensor([[False, True]])
        self.assertTrue(
            torch.allclose(
                fixed.encode_atom_memory(ids, mask, roles),
                linear.encode_atom_memory(ids, mask, roles),
                atol=1.0e-7,
                rtol=1.0e-6,
            )
        )

    def test_only_one_bias_free_linear_is_added(self):
        from most_t5_next.p2.motif_geometry_adapter_v7 import MotifGeometryAdapterV7

        adapter = self._adapter(MotifGeometryAdapterV7)
        names = dict(adapter.named_parameters())
        self.assertEqual(
            tuple(adapter.l0_high_projection.weight.shape),
            (adapter.atom_memory_dim, 2 * adapter.atom_memory_dim),
        )
        self.assertIsNone(adapter.l0_high_projection.bias)
        self.assertNotIn("level_embedding.weight", names)
        self.assertNotIn("atom_role_embedding.weight", names)
        self.assertFalse(any(name.startswith("atom_encoder.") for name in names))

    def test_projection_receives_finite_nonzero_gradient(self):
        from most_t5_next.p2.motif_geometry_adapter_v7 import MotifGeometryAdapterV7

        adapter = self._adapter(MotifGeometryAdapterV7)
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, -1]]])
        output = adapter.encode_atom_memory(
            ids, torch.tensor([[True, True]]), torch.tensor([[False, True]])
        )
        output.square().sum().backward()
        gradient = adapter.l0_high_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
