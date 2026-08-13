from __future__ import annotations

import unittest

import torch

from most_t5_next.p2.motif_geometry_adapter_v1 import MotifGeometryAdapterError
from most_t5_next.p2.motif_geometry_adapter_v3 import MotifGeometryAdapterV3


class MotifGeometryAdapterV3Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(307)
        self.adapter = MotifGeometryAdapterV3(
            num_e3fp_embeddings=16,
            hidden_size=8,
            state_embedding_dim=4,
            atom_memory_dim=6,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            geometry_fraction=0.5,
        )
        self.values = {
            "input_embeddings": torch.randn(1, 6, 8),
            "attention_mask": torch.ones((1, 6), dtype=torch.bool),
            "e3fp_input_ids": torch.tensor(
                [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_to_motif": torch.tensor([[0, 0, 1]]),
            "motif_mask": torch.tensor([[True, True]]),
            "motif_to_carrier": torch.tensor([[0, 3]]),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "endpoint_token_to_atom": torch.tensor([[-1, -1, 1, -1, -1, 2]]),
            "atom_is_attachment": torch.tensor([[False, True, True]]),
        }

    def encode(self, **overrides):
        values = dict(self.values)
        values.update(overrides)
        return self.adapter.encode(**values)

    def test_only_carriers_and_endpoints_change_without_adding_tokens(self) -> None:
        encoded = self.encode()
        self.assertEqual(encoded.fused_embeddings.shape, self.values["input_embeddings"].shape)
        torch.testing.assert_close(
            encoded.fused_embeddings[0, [1, 4]],
            self.values["input_embeddings"][0, [1, 4]],
        )
        for position in (0, 2, 3, 5):
            self.assertFalse(
                torch.equal(
                    encoded.fused_embeddings[0, position],
                    self.values["input_embeddings"][0, position],
                )
            )
        self.assertFalse(
            any("gate" in name for name, _ in self.adapter.named_parameters())
        )

    def test_endpoint_reads_its_exact_attachment_atom(self) -> None:
        original = self.encode()
        changed_ids = self.values["e3fp_input_ids"].clone()
        changed_ids[0, 1, 1] = 13
        changed = self.encode(e3fp_input_ids=changed_ids)
        self.assertFalse(
            torch.equal(
                original.fused_embeddings[0, 2],
                changed.fused_embeddings[0, 2],
            )
        )
        torch.testing.assert_close(
            original.fused_embeddings[0, 5],
            changed.fused_embeddings[0, 5],
        )

    def test_owned_motif_geometry_is_isolated(self) -> None:
        original = self.encode()
        changed_ids = self.values["e3fp_input_ids"].clone()
        changed_ids[0, 2, 1] = 14
        changed = self.encode(e3fp_input_ids=changed_ids)
        torch.testing.assert_close(
            original.fused_embeddings[0, [0, 2]],
            changed.fused_embeddings[0, [0, 2]],
        )
        self.assertFalse(
            torch.equal(original.fused_embeddings[0, 3], changed.fused_embeddings[0, 3])
        )
        self.assertFalse(
            torch.equal(original.fused_embeddings[0, 5], changed.fused_embeddings[0, 5])
        )

    def test_zero_geometry_is_the_exact_t5_embedding_baseline(self) -> None:
        zero = self.encode(state_memory_mode="zero")
        self.assertTrue(
            torch.equal(zero.fused_embeddings, self.values["input_embeddings"])
        )
        self.assertEqual(int(torch.count_nonzero(zero.atom_memory)), 0)

    def test_public_atom_memory_matches_the_encoded_memory(self) -> None:
        direct = self.adapter.encode_atom_memory(
            self.values["e3fp_input_ids"],
            self.values["atom_mask"],
            self.values["atom_is_attachment"],
        )
        encoded = self.encode(geometry_component_mode="carrier_only")
        torch.testing.assert_close(direct, encoded.atom_memory)

    def test_component_modes_isolate_carrier_and_endpoint_writes(self) -> None:
        baseline = self.values["input_embeddings"]
        carrier = self.encode(geometry_component_mode="carrier_only")
        endpoint = self.encode(geometry_component_mode="endpoint_only")
        zero = self.encode(geometry_component_mode="zero")
        changed_carrier = {
            index
            for index in range(baseline.shape[1])
            if not torch.equal(carrier.fused_embeddings[0, index], baseline[0, index])
        }
        changed_endpoint = {
            index
            for index in range(baseline.shape[1])
            if not torch.equal(endpoint.fused_embeddings[0, index], baseline[0, index])
        }
        self.assertEqual(changed_carrier, {0, 3})
        self.assertEqual(changed_endpoint, {2, 5})
        self.assertTrue(torch.equal(zero.fused_embeddings, baseline))
        # The matching target remains the same atom-derived motif state; only
        # its route into T5 is ablated.
        torch.testing.assert_close(
            carrier.pre_t5_motif_context,
            endpoint.pre_t5_motif_context,
        )

    def test_unknown_component_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(MotifGeometryAdapterError, "component"):
            self.encode(geometry_component_mode="portish")

    def test_nonattachment_endpoint_is_rejected(self) -> None:
        roles = self.values["atom_is_attachment"].clone()
        roles[0, 1] = False
        with self.assertRaisesRegex(MotifGeometryAdapterError, "attachment"):
            self.encode(atom_is_attachment=roles)


if __name__ == "__main__":
    unittest.main()
