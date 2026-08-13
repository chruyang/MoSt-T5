from __future__ import annotations

import unittest

import torch

from most_t5_next.p2.e3fp_atom_embedding_v1 import (
    E3FPAtomEmbeddingError,
    E3FPAtomEmbeddingV1,
    L0_STATE_FIXED4,
    LEVEL_SPECIFIC_FIXED4,
    REFERENCE_SHARED_FIXED4,
)


class E3FPAtomEmbeddingV1Tests(unittest.TestCase):
    def test_reference_matches_literal_shift_lookup_and_fixed_four_mean(self) -> None:
        module = E3FPAtomEmbeddingV1(fp_bits=8, embedding_dim=3)
        with torch.no_grad():
            module.shared_embedding.weight.copy_(
                torch.arange(27, dtype=torch.float32).reshape(9, 3)
            )
            module._zero_padding_rows()
        ids = torch.tensor([[[0, 1, -1, -1], [2, 3, 4, 5]]])
        mask = torch.tensor([[True, True]])
        actual = module(ids, mask)
        expected = module.shared_embedding(ids + 1).mean(dim=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual[0, 0],
            (module.shared_embedding.weight[1] + module.shared_embedding.weight[2]) / 4,
            rtol=0,
            atol=0,
        )

    def test_all_parameter_tying_arms_are_exactly_equal_at_update_zero(self) -> None:
        torch.manual_seed(7)
        reference = E3FPAtomEmbeddingV1(
            fp_bits=16,
            embedding_dim=5,
            variant=REFERENCE_SHARED_FIXED4,
        )
        ids = torch.tensor(
            [[[0, 4, 8, -1], [1, 5, -1, -1]], [[2, 6, 10, 14], [-1] * 4]]
        )
        mask = torch.tensor([[True, True], [True, False]])
        for variant in (L0_STATE_FIXED4, LEVEL_SPECIFIC_FIXED4):
            with self.subTest(variant=variant):
                candidate = E3FPAtomEmbeddingV1(
                    fp_bits=16,
                    embedding_dim=5,
                    variant=variant,
                )
                candidate.initialize_tied_tables_from_shared(
                    reference.shared_embedding.weight
                )
                torch.testing.assert_close(
                    reference(ids, mask), candidate(ids, mask), rtol=0, atol=0
                )

    def test_parameter_inventory_matches_4097_by_768_contract(self) -> None:
        shared = E3FPAtomEmbeddingV1(
            fp_bits=4096, embedding_dim=768, variant=REFERENCE_SHARED_FIXED4
        )
        per_level = E3FPAtomEmbeddingV1(
            fp_bits=4096, embedding_dim=768, variant=LEVEL_SPECIFIC_FIXED4
        )
        l0_state = E3FPAtomEmbeddingV1(
            fp_bits=4096, embedding_dim=768, variant=L0_STATE_FIXED4
        )
        self.assertEqual(shared.parameter_count(), 4097 * 768)
        self.assertEqual(l0_state.parameter_count(), 2 * 4097 * 768)
        self.assertEqual(per_level.parameter_count(), 4 * 4097 * 768)
        self.assertEqual(
            per_level.parameter_count() - shared.parameter_count(),
            3 * 4097 * 768,
        )

    def test_l0_state_arm_routes_l0_and_higher_shells_to_the_intended_tables(self) -> None:
        module = E3FPAtomEmbeddingV1(
            fp_bits=8, embedding_dim=1, variant=L0_STATE_FIXED4
        )
        with torch.no_grad():
            module.l0_embedding.weight.fill_(4.0)
            module.state_embedding.weight.fill_(8.0)
            module._zero_padding_rows()
        output = module(
            torch.tensor([[[0, 1, 2, -1]]]), torch.tensor([[True]])
        )
        # Fixed-four denominator: (L0=4 + L1=8 + L2=8 + L3-missing=0) / 4.
        torch.testing.assert_close(output, torch.tensor([[[5.0]]]), rtol=0, atol=0)

    def test_padding_is_zero_and_padded_atoms_are_zero(self) -> None:
        module = E3FPAtomEmbeddingV1(fp_bits=8, embedding_dim=4)
        self.assertTrue(torch.equal(module.shared_embedding.weight[0], torch.zeros(4)))
        output = module(
            torch.tensor([[[0, -1, -1, -1], [-1, -1, -1, -1]]]),
            torch.tensor([[True, False]]),
        )
        self.assertTrue(torch.equal(output[0, 1], torch.zeros(4)))

    def test_invalid_domains_fail_closed(self) -> None:
        module = E3FPAtomEmbeddingV1(fp_bits=8, embedding_dim=4)
        with self.assertRaises(E3FPAtomEmbeddingError):
            module(torch.tensor([[[8, -1, -1, -1]]]), torch.tensor([[True]]))
        with self.assertRaises(E3FPAtomEmbeddingError):
            module(torch.tensor([[[0, -1, -1, -1]]]), torch.tensor([[False]]))


if __name__ == "__main__":
    unittest.main()
