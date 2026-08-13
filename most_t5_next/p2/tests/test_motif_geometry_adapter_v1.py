from __future__ import annotations

import unittest

import torch

from most_t5_next.p2.motif_geometry_adapter_v1 import (
    MotifGeometryAdapterError,
    MotifGeometryAdapterV1,
)


class MotifGeometryAdapterV1Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.adapter = MotifGeometryAdapterV1(
            num_e3fp_embeddings=16,
            hidden_size=8,
            state_embedding_dim=4,
            atom_memory_dim=6,
            max_identity_span_length=8,
        )
        self.input_embeddings = torch.randn(1, 6, 8)
        self.attention_mask = torch.tensor(
            [[True, True, True, True, True, True]], dtype=torch.bool
        )
        self.e3fp_ids = torch.tensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
            dtype=torch.long,
        )
        self.atom_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        self.atom_to_motif = torch.tensor([[0, 0, 1]], dtype=torch.long)
        self.motif_mask = torch.tensor([[True, True]], dtype=torch.bool)
        self.motif_to_carrier = torch.tensor([[0, 3]], dtype=torch.long)
        self.identity_span_bounds = torch.tensor(
            [[[0, 2], [3, 4]]], dtype=torch.long
        )
        self.atom_is_attachment = torch.tensor(
            [[False, True, False]], dtype=torch.bool
        )

    def encode(self, **overrides: torch.Tensor):
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

    def decode(self, atom_memory: torch.Tensor, encoder_hidden: torch.Tensor):
        return self.adapter.decode_state(
            atom_memory,
            encoder_hidden,
            attention_mask=self.attention_mask,
            atom_mask=self.atom_mask,
            atom_to_motif=self.atom_to_motif,
            motif_mask=self.motif_mask,
            motif_to_carrier=self.motif_to_carrier,
        )

    def test_shapes_owned_attention_and_zero_start(self) -> None:
        output = self.encode()
        self.assertEqual(output.fused_embeddings.shape, (1, 6, 8))
        self.assertEqual(output.atom_memory.shape, (1, 3, 6))
        self.assertEqual(output.pre_t5_motif_context.shape, (1, 2, 8))
        self.assertEqual(output.cross_attention_weights.shape, (1, 2, 3))
        torch.testing.assert_close(output.fused_embeddings, self.input_embeddings)
        torch.testing.assert_close(
            output.cross_attention_weights[0, 0, 2], torch.tensor(0.0)
        )
        torch.testing.assert_close(
            output.cross_attention_weights[0, 1, :2], torch.zeros(2)
        )
        torch.testing.assert_close(
            output.cross_attention_weights.sum(dim=-1), torch.ones(1, 2)
        )

    def test_l0_and_l3_cannot_change_encoding_or_state_logits(self) -> None:
        original = self.encode()
        changed_ids = self.e3fp_ids.clone()
        changed_ids[..., 0] = (changed_ids[..., 0] + 7) % 16
        changed_ids[..., 3] = (changed_ids[..., 3] + 5) % 16
        changed = self.encode(e3fp_input_ids=changed_ids)
        torch.testing.assert_close(changed.atom_memory, original.atom_memory)
        torch.testing.assert_close(
            changed.pre_t5_motif_context,
            original.pre_t5_motif_context,
        )
        encoder_hidden = torch.randn(1, 6, 8)
        torch.testing.assert_close(
            self.decode(changed.atom_memory, encoder_hidden),
            self.decode(original.atom_memory, encoder_hidden),
        )

    def test_l1_changes_atom_memory_but_only_owned_motif_context(self) -> None:
        original = self.encode()
        changed_ids = self.e3fp_ids.clone()
        changed_ids[0, 2, 1] = 15
        changed = self.encode(e3fp_input_ids=changed_ids)
        self.assertFalse(torch.equal(changed.atom_memory[0, 2], original.atom_memory[0, 2]))
        torch.testing.assert_close(
            changed.pre_t5_motif_context[0, 0],
            original.pre_t5_motif_context[0, 0],
        )
        self.assertFalse(
            torch.equal(
                changed.pre_t5_motif_context[0, 1],
                original.pre_t5_motif_context[0, 1],
            )
        )

    def test_full_identity_span_and_sentinel_span_share_one_interface(self) -> None:
        original = self.encode()
        changed_embeddings = self.input_embeddings.clone()
        changed_embeddings[0, 1] += 3.0
        full_span_changed = self.encode(input_embeddings=changed_embeddings)
        self.assertFalse(
            torch.equal(
                full_span_changed.pre_t5_motif_context[0, 0],
                original.pre_t5_motif_context[0, 0],
            )
        )

        sentinel_bounds = self.identity_span_bounds.clone()
        sentinel_bounds[0, 0] = torch.tensor([0, 1])
        sentinel_original = self.encode(identity_span_bounds=sentinel_bounds)
        sentinel_changed = self.encode(
            input_embeddings=changed_embeddings,
            identity_span_bounds=sentinel_bounds,
        )
        torch.testing.assert_close(
            sentinel_changed.pre_t5_motif_context[0, 0],
            sentinel_original.pre_t5_motif_context[0, 0],
        )

    def test_state_decoder_reads_post_t5_owner_carrier(self) -> None:
        encoded = self.encode()
        encoder_hidden = torch.randn(1, 6, 8)
        original_logits = self.decode(encoded.atom_memory, encoder_hidden)
        self.assertEqual(original_logits.shape, (1, 3, 2, 16))

        changed_hidden = encoder_hidden.clone()
        changed_hidden[0, 0] += 4.0
        changed_logits = self.decode(encoded.atom_memory, changed_hidden)
        self.assertFalse(torch.equal(changed_logits[0, :2], original_logits[0, :2]))
        torch.testing.assert_close(changed_logits[0, 2], original_logits[0, 2])

    def test_state_loss_backpropagates_to_t5_carrier_and_atom_encoder(self) -> None:
        encoded = self.encode()
        encoder_hidden = torch.randn(1, 6, 8, requires_grad=True)
        logits = self.decode(encoded.atom_memory, encoder_hidden)
        logits[:, :, :, 0].sum().backward()
        self.assertGreater(float(encoder_hidden.grad[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(encoder_hidden.grad[0, 3].abs().sum()), 0.0)
        self.assertEqual(float(encoder_hidden.grad[0, 1].abs().sum()), 0.0)
        self.assertIsNotNone(self.adapter.atom_encoder[0].weight.grad)
        self.assertGreater(float(self.adapter.atom_encoder[0].weight.grad.abs().sum()), 0.0)

    def test_autocast_keeps_residual_stream_dtype(self) -> None:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            encoded = self.encode()
        self.assertEqual(encoded.fused_embeddings.dtype, self.input_embeddings.dtype)

    def test_zero_memory_mode_removes_the_complete_geometry_path(self) -> None:
        aligned = self.encode()
        zero = self.encode(state_memory_mode="zero")
        torch.testing.assert_close(zero.fused_embeddings, self.input_embeddings)
        self.assertEqual(int(torch.count_nonzero(zero.atom_memory)), 0)
        self.assertEqual(int(torch.count_nonzero(zero.pre_t5_motif_context)), 0)
        self.assertEqual(int(torch.count_nonzero(zero.cross_attention_weights)), 0)
        self.assertEqual(aligned.fused_embeddings.shape, zero.fused_embeddings.shape)
        with self.assertRaisesRegex(MotifGeometryAdapterError, "state_memory_mode"):
            self.encode(state_memory_mode="padding_like")

    def test_invalid_span_and_atom_owner_are_rejected(self) -> None:
        bad_spans = self.identity_span_bounds.clone()
        bad_spans[0, 0] = torch.tensor([2, 2])
        with self.assertRaisesRegex(MotifGeometryAdapterError, "identity span"):
            self.encode(identity_span_bounds=bad_spans)

        bad_owner = self.atom_to_motif.clone()
        bad_owner[0, 2] = 2
        with self.assertRaisesRegex(MotifGeometryAdapterError, "outside the motif"):
            self.encode(atom_to_motif=bad_owner)


if __name__ == "__main__":
    unittest.main()
