from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local lightweight environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class MotifGeometryAdapterV4Test(unittest.TestCase):
    def _adapter(self, mode):
        from most_t5_next.p2.motif_geometry_adapter_v4 import (
            MotifGeometryAdapterV4,
        )

        torch.manual_seed(7)
        return MotifGeometryAdapterV4(
            num_e3fp_embeddings=32,
            hidden_size=16,
            state_embedding_dim=8,
            atom_memory_dim=12,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
            shell_fusion_mode=mode,
        )

    def _inputs(self):
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=torch.long)
        mask = torch.tensor([[True, True]])
        roles = torch.tensor([[False, True]])
        return ids, mask, roles

    def test_all_candidates_have_identical_parameter_topology(self):
        from most_t5_next.p2.motif_geometry_adapter_v4 import SHELL_FUSION_MODES

        topologies = []
        for mode in SHELL_FUSION_MODES:
            adapter = self._adapter(mode)
            topologies.append(
                tuple((name, tuple(value.shape)) for name, value in adapter.named_parameters())
            )
        self.assertTrue(all(value == topologies[0] for value in topologies[1:]))

    def test_l12_ignores_l0_and_l3_but_l0_mode_consumes_l0(self):
        ids, mask, roles = self._inputs()
        changed_l0 = ids.clone()
        changed_l0[..., 0] += 9
        changed_l3 = ids.clone()
        changed_l3[..., 3] += 9
        l12 = self._adapter("l12_mean")
        baseline = l12.encode_atom_memory(ids, mask, roles)
        self.assertTrue(torch.equal(baseline, l12.encode_atom_memory(changed_l0, mask, roles)))
        self.assertTrue(torch.equal(baseline, l12.encode_atom_memory(changed_l3, mask, roles)))
        l0 = self._adapter("l0_l12_mean")
        self.assertFalse(
            torch.equal(
                l0.encode_atom_memory(ids, mask, roles),
                l0.encode_atom_memory(changed_l0, mask, roles),
            )
        )

    def test_shell_attention_consumes_l3_and_backpropagates(self):
        ids, mask, roles = self._inputs()
        changed = ids.clone()
        changed[..., 3] += 9
        adapter = self._adapter("l0_shell_attention_l123")
        baseline = adapter.encode_atom_memory(ids, mask, roles)
        alternative = adapter.encode_atom_memory(changed, mask, roles)
        self.assertFalse(torch.equal(baseline, alternative))
        baseline.sum().backward()
        self.assertIsNotNone(adapter.shell_attention_score.weight.grad)
        self.assertTrue(torch.isfinite(adapter.shell_attention_score.weight.grad).all())

    def test_l0_l123_keeps_l0_separate_and_consumes_l3(self):
        ids, mask, roles = self._inputs()
        changed_l0 = ids.clone()
        changed_l0[..., 0] += 9
        changed_l3 = ids.clone()
        changed_l3[..., 3] += 9
        adapter = self._adapter("l0_l123_mean")
        baseline = adapter.encode_atom_memory(ids, mask, roles)
        self.assertFalse(
            torch.equal(baseline, adapter.encode_atom_memory(changed_l0, mask, roles))
        )
        self.assertFalse(
            torch.equal(baseline, adapter.encode_atom_memory(changed_l3, mask, roles))
        )

    def test_checkpoint_refuses_shell_mode_drift(self):
        source = self._adapter("l12_mean")
        target = self._adapter("l0_l12_mean")
        with self.assertRaisesRegex(RuntimeError, "shell-fusion mode differs"):
            target.load_state_dict(source.state_dict())

    def test_padding_is_zero_for_every_mode(self):
        from most_t5_next.p2.motif_geometry_adapter_v4 import SHELL_FUSION_MODES

        ids, mask, roles = self._inputs()
        ids = torch.cat((ids, torch.full((1, 1, 4), -1, dtype=torch.long)), dim=1)
        mask = torch.tensor([[True, True, False]])
        roles = torch.tensor([[False, True, False]])
        for mode in SHELL_FUSION_MODES:
            memory = self._adapter(mode).encode_atom_memory(ids, mask, roles)
            self.assertTrue(torch.equal(memory[:, 2], torch.zeros_like(memory[:, 2])))


if __name__ == "__main__":
    unittest.main()
