from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class MotifGeometryAdapterV6Test(unittest.TestCase):
    def _adapter(self, mode):
        from most_t5_next.p2.motif_geometry_adapter_v6 import MotifGeometryAdapterV6

        torch.manual_seed(17)
        return MotifGeometryAdapterV6(
            num_e3fp_embeddings=32,
            hidden_size=16,
            state_embedding_dim=8,
            atom_memory_dim=12,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
            shell_reducer_mode=mode,
        )

    def test_adaptive_initial_function_matches_fixed_four(self):
        fixed = self._adapter("fixed_four_mean")
        adaptive = self._adapter("adaptive_l0_high")
        with torch.no_grad():
            adaptive.shared_e3fp_embedding.weight.copy_(
                fixed.shared_e3fp_embedding.weight
            )
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, -1]]])
        mask = torch.tensor([[True, True]])
        roles = torch.tensor([[False, True]])
        expected = fixed.encode_atom_memory(ids, mask, roles)
        actual = adaptive.encode_atom_memory(ids, mask, roles)
        self.assertTrue(torch.allclose(expected, actual, atol=1.0e-7, rtol=1.0e-6))
        self.assertAlmostEqual(float(adaptive.l0_weight()), 0.25, places=7)

    def test_only_adaptive_mode_backpropagates_to_global_mix(self):
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, -1]]])
        mask = torch.tensor([[True, True]])
        roles = torch.tensor([[False, True]])
        adaptive = self._adapter("adaptive_l0_high")
        adaptive.encode_atom_memory(ids, mask, roles).square().sum().backward()
        self.assertIsNotNone(adaptive.l0_mix_logit.grad)
        self.assertTrue(torch.isfinite(adaptive.l0_mix_logit.grad))
        self.assertNotEqual(float(adaptive.l0_mix_logit.grad), 0.0)
        fixed = self._adapter("fixed_four_mean")
        fixed.encode_atom_memory(ids, mask, roles).square().sum().backward()
        self.assertIsNone(fixed.l0_mix_logit.grad)

    def test_missing_high_shell_is_zero_with_fixed_denominator(self):
        adaptive = self._adapter("adaptive_l0_high")
        with torch.no_grad():
            adaptive.shared_e3fp_embedding.weight.fill_(1.0)
        ids = torch.tensor([[[1, 2, -1, -1]]])
        memory = adaptive.encode_atom_memory(
            ids, torch.tensor([[True]]), torch.tensor([[False]])
        )
        expected = torch.full_like(memory, 0.25 + 0.75 / 3.0)
        self.assertTrue(torch.allclose(memory, expected))

    def test_checkpoint_rejects_reducer_mode_drift(self):
        source = self._adapter("fixed_four_mean")
        target = self._adapter("adaptive_l0_high")
        with self.assertRaisesRegex(RuntimeError, "shell-reducer contract differs"):
            target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    unittest.main()
