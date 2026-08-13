from __future__ import annotations

import hashlib
import unittest

import torch

from most_t5_next.p2.held_out_motif_state_v4 import (
    HeldOutAtomStateMatchingHeadV4,
    HeldOutMotifStateError,
    build_held_out_motif_state_plan,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class HeldOutMotifStateV4Test(unittest.TestCase):
    def setUp(self) -> None:
        self.states = torch.tensor([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]],
        ])
        self.atom_mask = torch.ones((2, 3), dtype=torch.bool)
        self.atom_to_motif = torch.tensor([[0, 0, 1], [0, 0, 1]])
        self.atom_local_positions = torch.tensor([[1, 0, 0], [0, 1, 0]])
        self.motif_mask = torch.ones((2, 2), dtype=torch.bool)
        self.identities = (
            (_digest("shared"), _digest("singleton")),
            (_digest("shared"), _digest("singleton")),
        )

    def test_one_atom_is_hidden_and_one_peer_remains_visible(self) -> None:
        plan = build_held_out_motif_state_plan(
            e3fp_input_ids=self.states,
            atom_mask=self.atom_mask,
            atom_to_motif=self.atom_to_motif,
            atom_local_positions=self.atom_local_positions,
            motif_mask=self.motif_mask,
            record_ids=("left", "right"),
            exact_identity_sha256=self.identities,
            mask_token_id=17,
            seed=41,
            epoch=3,
        )
        self.assertEqual(plan.selected_targets, 2)
        self.assertTrue(torch.equal(plan.target_motif_mask[:, 0], torch.tensor([True, True])))
        self.assertTrue(torch.equal(plan.target_motif_mask[:, 1], torch.tensor([False, False])))
        self.assertTrue(torch.equal(plan.visible_peer_counts[:, 0], torch.tensor([1, 1])))
        for row in range(2):
            target = int(plan.target_atom_indices[row, 0])
            self.assertIn(target, (0, 1))
            self.assertEqual(plan.corrupted_e3fp_input_ids[row, target, 1:].tolist()[:2], [17, 17])
            peer = 1 - target
            self.assertTrue(torch.equal(plan.corrupted_e3fp_input_ids[row, peer], self.states[row, peer]))
        again = build_held_out_motif_state_plan(
            e3fp_input_ids=self.states,
            atom_mask=self.atom_mask,
            atom_to_motif=self.atom_to_motif,
            atom_local_positions=self.atom_local_positions,
            motif_mask=self.motif_mask,
            record_ids=("left", "right"),
            exact_identity_sha256=self.identities,
            mask_token_id=17,
            seed=41,
            epoch=3,
        )
        self.assertTrue(torch.equal(plan.target_atom_indices, again.target_atom_indices))

    def test_singleton_motifs_cannot_be_targets(self) -> None:
        with self.assertRaisesRegex(HeldOutMotifStateError, "no motif"):
            build_held_out_motif_state_plan(
                e3fp_input_ids=self.states[:, :1],
                atom_mask=self.atom_mask[:, :1],
                atom_to_motif=torch.zeros((2, 1), dtype=torch.long),
                atom_local_positions=torch.zeros((2, 1), dtype=torch.long),
                motif_mask=torch.ones((2, 1), dtype=torch.bool),
                record_ids=("left", "right"),
                exact_identity_sha256=((_digest("shared"),), (_digest("shared"),)),
                mask_token_id=17,
                seed=1,
                epoch=0,
            )

    def test_matching_detaches_original_atom_memory(self) -> None:
        torch.manual_seed(47)
        head = HeldOutAtomStateMatchingHeadV4(
            hidden_size=8,
            atom_memory_dim=6,
            projection_dim=4,
            temperature=0.1,
        )
        encoder = torch.randn((2, 4, 8), requires_grad=True)
        atom_memory = torch.randn((2, 2, 6), requires_grad=True)
        output = head(
            encoder_hidden=encoder,
            original_atom_memory=atom_memory,
            motif_mask=torch.ones((2, 1), dtype=torch.bool),
            motif_to_carrier=torch.tensor([[1], [1]]),
            target_atom_indices=torch.tensor([[0], [1]]),
            target_local_positions=torch.tensor([[0], [0]]),
            target_motif_mask=torch.ones((2, 1), dtype=torch.bool),
            exact_identity_sha256=((_digest("shared"),), (_digest("shared"),)),
        )
        self.assertEqual(output.eligible_anchors, 2)
        output.loss.backward()
        self.assertIsNone(atom_memory.grad)
        self.assertGreater(float(encoder.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.target_projection.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
