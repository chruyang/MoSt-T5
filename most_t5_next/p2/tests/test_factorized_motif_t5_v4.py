from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class FactorizedMotifT5V4Test(unittest.TestCase):
    def setUp(self) -> None:
        from most_t5_next.p2.factorized_motif_t5_v4 import FactorizedMotifT5V4
        from most_t5_next.p2.tests.test_factorized_motif_t5_v3 import _TinyT5

        torch.manual_seed(509)
        self.model = FactorizedMotifT5V4(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            shell_fusion_mode="l0_shell_attention_l123",
        )
        self.common = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor(
                [[[7, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]]]
            ),
            "atom_mask": torch.tensor([[True, True, True]]),
            "atom_to_motif": torch.tensor([[0, 0, 1]]),
            "atom_local_positions": torch.tensor([[0, 1, 0]]),
            "motif_mask": torch.tensor([[True, True]]),
            "motif_to_carrier": torch.tensor([[1, 4]]),
            "identity_span_bounds": torch.tensor([[[0, 2], [3, 5]]]),
            "endpoint_token_to_atom": torch.tensor([[0, -1, -1, 2, -1, -1]]),
            "atom_is_attachment": torch.tensor([[True, False, True]]),
        }

    def test_anchored_phrase_end_carriers_and_endpoints_reach_t5(self) -> None:
        labels = torch.tensor([[3, 4, 5, 6, 7, 8]])
        output = self.model(
            **self.common,
            labels=labels,
            objective_mode="grammar",
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(self.model.adapter.shell_attention_score.weight.grad)
        self.assertTrue(
            torch.isfinite(self.model.adapter.shell_attention_score.weight.grad).all()
        )

    def test_zero_is_exact_raw_t5_embedding_path(self) -> None:
        labels = torch.tensor([[3, 4, 5, 6, 7, 8]])
        output = self.model(
            **self.common,
            labels=labels,
            objective_mode="grammar",
            state_memory_mode="zero",
        )
        baseline = self.model.get_input_embeddings()(self.common["input_ids"])
        self.assertTrue(torch.equal(output.adapter_encoding.fused_embeddings, baseline))

    def test_factorisation_and_shell_mode_are_explicit(self) -> None:
        from most_t5_next.p2.factorized_motif_t5_v4 import FACTORISATION_ID

        self.assertEqual(self.model.factorisation_id, FACTORISATION_ID)
        self.assertEqual(
            self.model.adapter.shell_fusion_mode,
            "l0_shell_attention_l123",
        )


if __name__ == "__main__":
    unittest.main()
