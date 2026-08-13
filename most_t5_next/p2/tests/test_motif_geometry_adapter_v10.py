from __future__ import annotations

import unittest

import torch

from most_t5_next.p2.e3fp_atom_embedding_v1 import (
    LEVEL_SPECIFIC_FIXED4,
    L0_STATE_FIXED4,
    REFERENCE_SHARED_FIXED4,
)
from most_t5_next.p2.motif_geometry_adapter_v10 import MotifGeometryAdapterV10


class MotifGeometryAdapterV10Test(unittest.TestCase):
    def _adapter(self, variant: str) -> MotifGeometryAdapterV10:
        return MotifGeometryAdapterV10(
            num_e3fp_embeddings=16,
            hidden_size=8,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=8,
            geometry_fraction=0.5,
            parameter_tying=variant,
        )

    def test_update_zero_atom_memory_is_exact_for_all_three_arms(self) -> None:
        torch.manual_seed(8)
        reference = self._adapter(REFERENCE_SHARED_FIXED4)
        shared = reference.e3fp_atom_embedding.shared_embedding.weight.detach()
        ids = torch.tensor([[[0, 1, 2, -1], [3, 4, 5, 6]]])
        mask = torch.tensor([[True, True]])
        role = torch.tensor([[False, True]])
        expected = reference.encode_atom_memory(ids, mask, role)
        for variant in (L0_STATE_FIXED4, LEVEL_SPECIFIC_FIXED4):
            candidate = self._adapter(variant)
            candidate.e3fp_atom_embedding.initialize_tied_tables_from_shared(shared)
            torch.testing.assert_close(
                candidate.encode_atom_memory(ids, mask, role),
                expected,
                rtol=0,
                atol=0,
            )

    def test_parameter_topology_changes_only_the_table_tying(self) -> None:
        counts = {}
        for variant in (
            REFERENCE_SHARED_FIXED4,
            L0_STATE_FIXED4,
            LEVEL_SPECIFIC_FIXED4,
        ):
            adapter = self._adapter(variant)
            counts[variant] = adapter.e3fp_atom_embedding.parameter_count()
            names = tuple(name for name, _ in adapter.named_parameters())
            self.assertFalse(any(name.startswith("state_embedding.") for name in names))
            self.assertFalse(any(name.startswith("level_embedding.") for name in names))
            self.assertFalse(any(name.startswith("atom_role_embedding.") for name in names))
            self.assertFalse(any(name.startswith("atom_encoder.") for name in names))
        self.assertEqual(counts[REFERENCE_SHARED_FIXED4], 17 * 8)
        self.assertEqual(counts[L0_STATE_FIXED4], 2 * 17 * 8)
        self.assertEqual(counts[LEVEL_SPECIFIC_FIXED4], 4 * 17 * 8)

    def test_attachment_role_remains_routing_metadata_not_atom_input(self) -> None:
        adapter = self._adapter(L0_STATE_FIXED4)
        ids = torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]]])
        mask = torch.tensor([[True, True]])
        left = adapter.encode_atom_memory(ids, mask, torch.tensor([[False, True]]))
        right = adapter.encode_atom_memory(ids, mask, torch.tensor([[True, False]]))
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_complete_carrier_endpoint_route_is_update_zero_equal(self) -> None:
        torch.manual_seed(23)
        input_embeddings = torch.randn(1, 6, 8)
        common = {
            "attention_mask": torch.ones((1, 6), dtype=torch.bool),
            "e3fp_input_ids": torch.tensor(
                [[[0, 1, 2, -1], [3, 4, 5, 6], [7, 8, 9, 10]]]
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_to_motif": torch.tensor([[0, 0, 1]]),
            "motif_mask": torch.tensor([[True, True]]),
            "motif_to_carrier": torch.tensor([[0, 3]]),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "endpoint_token_to_atom": torch.tensor([[-1, -1, 1, -1, -1, 2]]),
            "atom_is_attachment": torch.tensor([[False, True, True]]),
        }
        torch.manual_seed(31)
        reference = self._adapter(REFERENCE_SHARED_FIXED4)
        shared = reference.e3fp_atom_embedding.shared_embedding.weight.detach()
        expected = reference.encode(input_embeddings, **common)
        for variant in (L0_STATE_FIXED4, LEVEL_SPECIFIC_FIXED4):
            torch.manual_seed(31)
            candidate = self._adapter(variant)
            candidate.e3fp_atom_embedding.initialize_tied_tables_from_shared(shared)
            actual = candidate.encode(input_embeddings, **common)
            torch.testing.assert_close(
                actual.fused_embeddings,
                expected.fused_embeddings,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual.cross_attention_weights,
                expected.cross_attention_weights,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
