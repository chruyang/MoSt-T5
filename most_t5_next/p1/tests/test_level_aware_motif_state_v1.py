from __future__ import annotations

import unittest

import torch

from most_t5_next.p1.level_aware_motif_state_v1 import (
    LevelAwareMotifStateEncoder,
    MotifStateContractError,
    build_masked_e3fp_state_batch,
    masked_state_ce,
)


class LevelAwareMotifStateTests(unittest.TestCase):
    def _fixture(self):
        ids = torch.tensor(
            [
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                    [30, 31, 32, -1],
                    [-1, -1, -1, -1],
                ]
            ],
            dtype=torch.long,
        )
        valid = torch.tensor([[True, True, True, False]])
        groups = torch.tensor([[0, 0, 1, -1]], dtype=torch.long)
        return ids, valid, groups

    def test_mask_is_deterministic_and_excludes_level_zero(self):
        ids, valid, _ = self._fixture()
        left = build_masked_e3fp_state_batch(
            ids, valid, mask_token_id=4097, probability=0.15, seed=7
        )
        right = build_masked_e3fp_state_batch(
            ids, valid, mask_token_id=4097, probability=0.15, seed=7
        )
        self.assertTrue(torch.equal(left.target_mask, right.target_mask))
        self.assertFalse(bool(left.target_mask[..., 0].any()))
        self.assertTrue(bool(left.target_mask.any()))
        self.assertTrue(torch.equal(left.target_ids, ids))

    def test_mask_can_exclude_high_entropy_outer_shell(self):
        ids, valid, _ = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids,
            valid,
            mask_token_id=4097,
            probability=1.0,
            seed=7,
            target_levels=(1, 2),
        )
        self.assertTrue(bool(masked.target_mask[..., 1].any()))
        self.assertTrue(bool(masked.target_mask[..., 2].any()))
        self.assertFalse(bool(masked.target_mask[..., 0].any()))
        self.assertFalse(bool(masked.target_mask[..., 3].any()))

    def test_both_poolers_preserve_shapes_and_normalize_members(self):
        ids, valid, groups = self._fixture()
        for pooling in ("deep_sets", "gated"):
            model = LevelAwareMotifStateEncoder(
                embedding_dim=8, hidden_dim=16, pooling=pooling
            )
            output = model(ids, valid, groups, num_groups=2)
            self.assertEqual(tuple(output.logits.shape), (1, 4, 4, 4096))
            self.assertEqual(tuple(output.group_hidden.shape), (1, 2, 16))
            self.assertAlmostEqual(float(output.atom_weights[0, :2].sum()), 1.0, places=6)
            self.assertAlmostEqual(float(output.atom_weights[0, 2]), 1.0, places=6)
            self.assertEqual(float(output.atom_weights[0, 3]), 0.0)

    def test_atom_permutation_keeps_group_context_and_unpermuted_logits(self):
        torch.manual_seed(11)
        ids, valid, groups = self._fixture()
        model = LevelAwareMotifStateEncoder(embedding_dim=8, hidden_dim=16, pooling="gated")
        model.eval()
        original = model(ids, valid, groups, num_groups=2)
        permutation = torch.tensor([1, 0, 2, 3])
        inverse = torch.argsort(permutation)
        permuted = model(
            ids[:, permutation], valid[:, permutation], groups[:, permutation], num_groups=2
        )
        self.assertTrue(torch.allclose(original.group_hidden, permuted.group_hidden, atol=1e-6))
        self.assertTrue(
            torch.allclose(original.logits, permuted.logits[:, inverse], atol=1e-6)
        )

    def test_masked_ce_reports_levelwise_sufficient_statistics(self):
        ids, valid, groups = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids, valid, mask_token_id=4097, probability=1.0, seed=3
        )
        model = LevelAwareMotifStateEncoder(embedding_dim=8, hidden_dim=16)
        output = model(masked.corrupted_ids, valid, groups, num_groups=2)
        loss, metrics = masked_state_ce(
            output.logits, masked.target_ids, masked.target_mask
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {1, 2, 3})
        self.assertEqual(sum(int(value["count"]) for value in metrics.values()), 8)

    def test_rejects_invalid_group_domain(self):
        ids, valid, groups = self._fixture()
        groups[0, 0] = -1
        model = LevelAwareMotifStateEncoder(embedding_dim=8, hidden_dim=16)
        with self.assertRaises(MotifStateContractError):
            model(ids, valid, groups, num_groups=2)

    def test_cpu_bfloat16_autocast_segment_reductions(self):
        ids, valid, groups = self._fixture()
        for pooling in ("deep_sets", "gated"):
            model = LevelAwareMotifStateEncoder(
                embedding_dim=8, hidden_dim=16, pooling=pooling
            )
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = model(ids, valid, groups, num_groups=2)
            self.assertTrue(torch.isfinite(output.logits.float()).all())
            self.assertEqual(output.atom_weights.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
