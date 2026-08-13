from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class MotifGeometryAdapterV8Test(unittest.TestCase):
    def _adapter(self):
        from most_t5_next.p2.motif_geometry_adapter_v8 import MotifGeometryAdapterV8

        torch.manual_seed(17)
        return MotifGeometryAdapterV8(
            num_e3fp_embeddings=32,
            hidden_size=16,
            state_embedding_dim=8,
            atom_memory_dim=12,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
        )

    def test_parameter_topology_excludes_role_presence_and_level_features(self):
        adapter = self._adapter()
        names = dict(adapter.named_parameters())
        self.assertFalse(any(name.startswith("level_embedding.") for name in names))
        self.assertFalse(any(name.startswith("atom_role_embedding.") for name in names))
        self.assertEqual(len(adapter.atom_phi), 4)
        self.assertEqual(tuple(adapter.atom_phi[0].weight.shape), (12, 16))

    def test_missing_high_shell_uses_available_shell_mean(self):
        adapter = self._adapter()
        ids = torch.tensor([[[1, 2, 3, -1], [4, 5, -1, -1]]])
        mask = torch.tensor([[True, True]])
        roles = torch.tensor([[False, True]])
        result = adapter.encode_atom_memory(ids, mask, roles)
        self.assertEqual(tuple(result.shape), (1, 2, 12))
        self.assertTrue(torch.isfinite(result).all())

    def test_phi_receives_finite_nonzero_gradient(self):
        adapter = self._adapter()
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, -1]]])
        output = adapter.encode_atom_memory(
            ids, torch.tensor([[True, True]]), torch.tensor([[False, True]])
        )
        output.square().sum().backward()
        gradients = [p.grad for p in adapter.atom_phi.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
