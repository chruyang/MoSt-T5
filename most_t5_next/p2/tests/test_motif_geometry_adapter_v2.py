from __future__ import annotations

import unittest

import torch

from most_t5_next.p2.motif_geometry_adapter_v1 import MotifGeometryAdapterError
from most_t5_next.p2.motif_geometry_adapter_v2 import MotifGeometryAdapterV2


class MotifGeometryAdapterV2Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(107)
        self.adapter = MotifGeometryAdapterV2(
            num_e3fp_embeddings=16,
            hidden_size=8,
            state_embedding_dim=4,
            atom_memory_dim=6,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            initial_geometry_gate=0.1,
        )
        self.input_embeddings = torch.randn(1, 6, 8)
        self.attention_mask = torch.ones((1, 6), dtype=torch.bool)
        self.e3fp_ids = torch.tensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
            dtype=torch.long,
        )
        self.atom_mask = torch.tensor([[True, True, True]])
        self.atom_to_motif = torch.tensor([[0, 0, 1]], dtype=torch.long)
        self.atom_local_positions = torch.tensor([[0, 1, 0]], dtype=torch.long)
        self.motif_mask = torch.tensor([[True, True]])
        self.motif_to_carrier = torch.tensor([[0, 3]], dtype=torch.long)
        self.identity_span_bounds = torch.tensor([[[0, 2], [3, 5]]])
        self.atom_is_attachment = torch.tensor([[False, True, False]])

    def encode(self, **overrides):
        values = {
            "input_embeddings": self.input_embeddings,
            "attention_mask": self.attention_mask,
            "e3fp_input_ids": self.e3fp_ids,
            "atom_mask": self.atom_mask,
            "atom_to_motif": self.atom_to_motif,
            "motif_mask": self.motif_mask,
            "motif_to_carrier": self.motif_to_carrier,
            "identity_span_bounds": self.identity_span_bounds,
            "atom_is_attachment": self.atom_is_attachment,
        }
        values.update(overrides)
        return self.adapter.encode(**values)

    def decode(self, atom_memory, encoder_hidden):
        return self.adapter.decode_state(
            atom_memory,
            encoder_hidden,
            attention_mask=self.attention_mask,
            atom_mask=self.atom_mask,
            atom_to_motif=self.atom_to_motif,
            atom_local_positions=self.atom_local_positions,
            motif_mask=self.motif_mask,
            motif_to_carrier=self.motif_to_carrier,
        )

    def test_channel_gate_is_nonzero_bounded_and_changes_only_carriers(self) -> None:
        encoded = self.encode()
        gate = self.adapter.geometry_gate_values()
        torch.testing.assert_close(gate, torch.full((8,), 0.1))
        self.assertFalse(torch.equal(encoded.fused_embeddings, self.input_embeddings))
        torch.testing.assert_close(
            encoded.fused_embeddings[0, [1, 2, 4, 5]],
            self.input_embeddings[0, [1, 2, 4, 5]],
        )
        self.assertFalse(
            torch.equal(encoded.fused_embeddings[0, 0], self.input_embeddings[0, 0])
        )
        self.assertFalse(
            torch.equal(encoded.fused_embeddings[0, 3], self.input_embeddings[0, 3])
        )
        self.assertNotIn("geometry_residual_scale", dict(self.adapter.named_parameters()))

    def test_state_decoder_has_no_direct_atom_memory_bypass(self) -> None:
        encoded = self.encode()
        encoder_hidden = torch.randn(1, 6, 8)
        original = self.decode(encoded.atom_memory, encoder_hidden)
        replacement = torch.randn_like(encoded.atom_memory) * 100.0
        changed = self.decode(replacement, encoder_hidden)
        self.assertTrue(torch.equal(original, changed))

    def test_state_decoder_uses_owner_carrier_and_atom_address(self) -> None:
        encoded = self.encode()
        encoder_hidden = torch.randn(1, 6, 8)
        logits = self.decode(encoded.atom_memory, encoder_hidden)
        self.assertEqual(tuple(logits.shape), (1, 3, 2, 16))
        self.assertFalse(torch.equal(logits[0, 0], logits[0, 1]))

        changed_hidden = encoder_hidden.clone()
        changed_hidden[0, 0] += 3.0
        changed = self.decode(encoded.atom_memory, changed_hidden)
        self.assertFalse(torch.equal(changed[0, :2], logits[0, :2]))
        torch.testing.assert_close(changed[0, 2], logits[0, 2])

    def test_state_gradient_reaches_geometry_only_through_fused_carrier(self) -> None:
        encoded = self.encode()
        projection = torch.nn.Linear(8, 8, bias=False)
        encoder_hidden = projection(encoded.fused_embeddings)
        logits = self.decode(encoded.atom_memory, encoder_hidden)
        logits[..., 0].sum().backward()
        atom_grad = self.adapter.atom_encoder[0].weight.grad
        gate_grad = self.adapter.geometry_output.gate_logits.grad
        self.assertIsNotNone(atom_grad)
        self.assertIsNotNone(gate_grad)
        self.assertGreater(float(atom_grad.abs().sum()), 0.0)
        self.assertGreater(float(gate_grad.abs().sum()), 0.0)

    def test_zero_mode_is_an_exact_geometry_ablation(self) -> None:
        zero = self.encode(state_memory_mode="zero")
        torch.testing.assert_close(zero.fused_embeddings, self.input_embeddings)
        self.assertEqual(int(torch.count_nonzero(zero.atom_memory)), 0)

    def test_motif_atom_capacity_is_explicit(self) -> None:
        small = MotifGeometryAdapterV2(
            num_e3fp_embeddings=16,
            hidden_size=8,
            state_embedding_dim=4,
            atom_memory_dim=6,
            max_identity_span_length=8,
            max_atoms_per_motif=1,
        )
        encoded = self.encode()
        with self.assertRaisesRegex(MotifGeometryAdapterError, "max_atoms_per_motif"):
            small.decode_state(
                encoded.atom_memory,
                torch.randn(1, 6, 8),
                attention_mask=self.attention_mask,
                atom_mask=self.atom_mask,
                atom_to_motif=self.atom_to_motif,
                atom_local_positions=self.atom_local_positions,
                motif_mask=self.motif_mask,
                motif_to_carrier=self.motif_to_carrier,
            )

    def test_noncanonical_or_model_axis_addresses_are_rejected(self) -> None:
        encoded = self.encode()
        with self.assertRaisesRegex(MotifGeometryAdapterError, "contiguous"):
            self.adapter.decode_state(
                encoded.atom_memory,
                torch.randn(1, 6, 8),
                attention_mask=self.attention_mask,
                atom_mask=self.atom_mask,
                atom_to_motif=self.atom_to_motif,
                atom_local_positions=torch.tensor([[1, 2, 0]]),
                motif_mask=self.motif_mask,
                motif_to_carrier=self.motif_to_carrier,
            )


if __name__ == "__main__":
    unittest.main()
