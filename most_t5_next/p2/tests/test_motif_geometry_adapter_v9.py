from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class MotifGeometryAdapterV9Test(unittest.TestCase):
    def _adapter(self):
        from most_t5_next.p2.motif_geometry_adapter_v9 import MotifGeometryAdapterV9

        torch.manual_seed(17)
        return MotifGeometryAdapterV9(
            num_e3fp_embeddings=32,
            hidden_size=16,
            state_embedding_dim=8,
            atom_memory_dim=12,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
        )

    def test_only_level_context_is_restored(self):
        adapter = self._adapter()
        names = dict(adapter.named_parameters())
        self.assertIn("level_embedding.weight", names)
        self.assertFalse(any(name.startswith("atom_role_embedding.") for name in names))
        self.assertEqual(tuple(adapter.level_embedding.weight.shape), (4, 8))

    def test_level_embedding_changes_atom_state_and_receives_gradient(self):
        adapter = self._adapter()
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, -1]]])
        output = adapter.encode_atom_memory(
            ids, torch.tensor([[True, True]]), torch.tensor([[False, True]])
        )
        output.square().sum().backward()
        gradient = adapter.level_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
