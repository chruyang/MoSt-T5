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
        self.assertTrue(torch.equal(left.corruption_mask, left.target_mask))
        self.assertTrue(torch.equal(left.corruption_mask, right.corruption_mask))

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

    def test_suffix_level_one_target_hides_all_higher_shells(self):
        ids, valid, _ = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids,
            valid,
            mask_token_id=4097,
            probability=1.0,
            seed=7,
            target_levels=(1,),
            masking_strategy="suffix",
        )
        self.assertTrue(bool(masked.target_mask[0, :3, 1].all()))
        self.assertFalse(bool(masked.target_mask[0, 3, 1]))
        self.assertFalse(bool(masked.target_mask[..., (0, 2, 3)].any()))
        self.assertFalse(bool(masked.corruption_mask[..., 0].any()))
        self.assertTrue(bool(masked.corruption_mask[0, :3, 1].all()))
        self.assertTrue(bool(masked.corruption_mask[0, :3, 2].all()))
        self.assertTrue(bool(masked.corruption_mask[0, :2, 3].all()))
        self.assertFalse(bool(masked.corruption_mask[0, 2, 3]))
        self.assertTrue(
            bool(
                (
                    masked.corrupted_ids[masked.corruption_mask]
                    == 4097
                ).all()
            )
        )

    def test_suffix_level_two_target_keeps_lower_shell_and_hides_outer_shell(self):
        ids, valid, _ = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids,
            valid,
            mask_token_id=4097,
            probability=1.0,
            seed=9,
            target_levels=(2,),
            masking_strategy="suffix",
        )
        self.assertTrue(torch.equal(masked.corrupted_ids[..., :2], ids[..., :2]))
        self.assertTrue(bool(masked.corruption_mask[0, :3, 2].all()))
        self.assertTrue(bool(masked.corruption_mask[0, :2, 3].all()))
        self.assertFalse(bool(masked.corruption_mask[..., 1].any()))
        self.assertTrue(torch.equal(masked.target_mask[..., 2], valid))
        self.assertFalse(bool(masked.target_mask[..., 3].any()))

    def test_atom_row_hides_level_zero_but_only_scores_requested_levels(self):
        ids, valid, _ = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids,
            valid,
            mask_token_id=4097,
            probability=1.0,
            seed=13,
            target_levels=(1, 2),
            masking_strategy="atom_row",
        )
        populated = (ids >= 0) & valid.unsqueeze(-1)
        self.assertTrue(torch.equal(masked.corruption_mask, populated))
        self.assertFalse(bool(masked.target_mask[..., 0].any()))
        self.assertFalse(bool(masked.target_mask[..., 3].any()))
        self.assertTrue(bool(masked.target_mask[0, :3, 1:3].all()))

    def test_motif_block_hides_all_atoms_in_selected_motif(self):
        ids, valid, groups = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids,
            valid,
            mask_token_id=4097,
            probability=1.0,
            seed=17,
            target_levels=(1, 2),
            masking_strategy="motif_block",
            atom_to_group=groups,
        )
        populated = (ids >= 0) & valid.unsqueeze(-1)
        self.assertTrue(torch.equal(masked.corruption_mask, populated))
        self.assertTrue(bool(masked.target_mask[0, :3, 1:3].all()))
        self.assertFalse(bool(masked.target_mask[..., (0, 3)].any()))

    def test_motif_atom_row_selects_at_most_one_atom_per_group(self):
        ids, valid, groups = self._fixture()
        masked = build_masked_e3fp_state_batch(
            ids,
            valid,
            mask_token_id=4097,
            probability=1.0,
            seed=19,
            target_levels=(1, 2),
            masking_strategy="motif_atom_row",
            atom_to_group=groups,
        )
        selected_atoms = masked.corruption_mask.any(dim=-1)[0]
        for group_id in (0, 1):
            self.assertLessEqual(
                int(selected_atoms[groups[0] == group_id].sum()),
                1,
            )
        self.assertEqual(int(selected_atoms.sum()), 1)
        self.assertFalse(bool(selected_atoms[2]))
        self.assertTrue(
            torch.equal(
                masked.corruption_mask,
                selected_atoms.view(1, -1, 1) & ((ids >= 0) & valid.unsqueeze(-1)),
            )
        )

    def test_motif_block_low_probability_selects_whole_group_and_is_deterministic(self):
        ids, valid, groups = self._fixture()
        kwargs = dict(
            mask_token_id=4097,
            probability=0.01,
            seed=23,
            target_levels=(1,),
            masking_strategy="motif_block",
            atom_to_group=groups,
        )
        left = build_masked_e3fp_state_batch(ids, valid, **kwargs)
        right = build_masked_e3fp_state_batch(ids, valid, **kwargs)
        self.assertTrue(torch.equal(left.target_mask, right.target_mask))
        self.assertTrue(torch.equal(left.corruption_mask, right.corruption_mask))
        selected_atoms = left.corruption_mask.any(dim=-1)[0]
        self.assertEqual(bool(selected_atoms[0]), bool(selected_atoms[1]))
        self.assertFalse(bool(selected_atoms[3]))
        self.assertTrue(bool(left.target_mask.any()))

    def test_motif_block_requires_valid_group_mapping(self):
        ids, valid, _ = self._fixture()
        with self.assertRaisesRegex(MotifStateContractError, "atom_to_group"):
            build_masked_e3fp_state_batch(
                ids,
                valid,
                mask_token_id=4097,
                probability=0.5,
                seed=3,
                masking_strategy="motif_block",
            )

    def test_formal_nonempty_fallback_is_not_fixed_to_first_candidate(self):
        ids, valid, _ = self._fixture()
        selected_atoms = set()
        for seed in range(24):
            masked = build_masked_e3fp_state_batch(
                ids,
                valid,
                mask_token_id=4097,
                probability=1.0e-12,
                seed=seed,
                target_levels=(1,),
                masking_strategy="suffix",
            )
            coordinate = masked.target_mask[0].nonzero(as_tuple=False)[0]
            selected_atoms.add(int(coordinate[0]))
        self.assertGreater(len(selected_atoms), 1)

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
