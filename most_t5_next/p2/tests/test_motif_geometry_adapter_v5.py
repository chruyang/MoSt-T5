from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight local environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class MotifGeometryAdapterV5Test(unittest.TestCase):
    def _adapter(self):
        from most_t5_next.p2.motif_geometry_adapter_v5 import (
            MotifGeometryAdapterV5,
        )

        torch.manual_seed(13)
        return MotifGeometryAdapterV5(
            num_e3fp_embeddings=32,
            hidden_size=8,
            state_embedding_dim=4,
            atom_memory_dim=3,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
        )

    def test_fixed_four_slot_mean_and_missing_shell_zero(self) -> None:
        adapter = self._adapter()
        with torch.no_grad():
            adapter.shared_e3fp_embedding.weight.zero_()
            adapter.shared_e3fp_embedding.weight[1] = torch.tensor([4.0, 0.0, 0.0])
            adapter.shared_e3fp_embedding.weight[2] = torch.tensor([0.0, 8.0, 0.0])
            adapter.shared_e3fp_embedding.weight[3] = torch.tensor([0.0, 0.0, 12.0])
        ids = torch.tensor([[[1, 2, 3, -1]]], dtype=torch.long)
        memory = adapter.encode_atom_memory(
            ids,
            torch.tensor([[True]]),
            torch.tensor([[False]]),
        )
        # Missing L3 contributes zero and the divisor remains exactly four.
        self.assertTrue(
            torch.equal(memory, torch.tensor([[[1.0, 2.0, 3.0]]]))
        )

    def test_attachment_role_is_routing_only(self) -> None:
        adapter = self._adapter()
        ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=torch.long)
        mask = torch.tensor([[True, True]])
        left = adapter.encode_atom_memory(ids, mask, torch.tensor([[False, True]]))
        right = adapter.encode_atom_memory(ids, mask, torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(left, right))

    def test_parameter_topology_has_no_level_role_or_atom_mlp(self) -> None:
        names = tuple(name for name, _ in self._adapter().named_parameters())
        self.assertIn("shared_e3fp_embedding.weight", names)
        self.assertFalse(any(name.startswith("state_embedding.") for name in names))
        self.assertFalse(any(name.startswith("level_embedding.") for name in names))
        self.assertFalse(any(name.startswith("atom_role_embedding.") for name in names))
        self.assertFalse(any(name.startswith("atom_encoder.") for name in names))

    def test_l0_and_l3_are_both_consumed(self) -> None:
        adapter = self._adapter()
        ids = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long)
        mask = torch.tensor([[True]])
        role = torch.tensor([[False]])
        baseline = adapter.encode_atom_memory(ids, mask, role)
        l0 = ids.clone()
        l0[..., 0] = 9
        l3 = ids.clone()
        l3[..., 3] = 10
        self.assertFalse(torch.equal(baseline, adapter.encode_atom_memory(l0, mask, role)))
        self.assertFalse(torch.equal(baseline, adapter.encode_atom_memory(l3, mask, role)))

    def test_padded_atom_is_zero_and_checkpoint_binds_variant(self) -> None:
        source = self._adapter()
        ids = torch.tensor([[[1, 2, 3, 4], [-1, -1, -1, -1]]], dtype=torch.long)
        memory = source.encode_atom_memory(
            ids,
            torch.tensor([[True, False]]),
            torch.tensor([[False, False]]),
        )
        self.assertTrue(torch.equal(memory[:, 1], torch.zeros_like(memory[:, 1])))
        state = source.state_dict()
        state["_extra_state"] = {"atom_encoder_variant": "different"}
        with self.assertRaisesRegex(RuntimeError, "atom-encoder variant differs"):
            self._adapter().load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
