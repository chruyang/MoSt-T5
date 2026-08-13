from __future__ import annotations

import hashlib
import unittest

import torch

from most_t5_next.p2.factorized_motif_t5_v3 import FactorizedMotifT5V3
from most_t5_next.p2.held_out_motif_state_v4 import HeldOutAtomStateMatchingHeadV4
from most_t5_next.p2.pf10_training_tensor_cache_v1 import CachedV3Batch
from most_t5_next.p2.run_pf10_3d_motif_v4_heldout_v1 import (
    build_plan,
    forward_held_out_matching,
    freeze_t5_for_held_out,
)
from most_t5_next.p2.tests.test_factorized_motif_t5_v3 import _TinyT5


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class V4HeldOutRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(601)
        self.model = FactorizedMotifT5V3(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            geometry_fraction=0.5,
        )
        self.head = HeldOutAtomStateMatchingHeadV4(
            hidden_size=8,
            atom_memory_dim=8,
            projection_dim=4,
            temperature=0.1,
        )
        values = {
            "input_ids": torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]]),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor([
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[6, 7, 8, 9], [2, 3, 4, 5]],
            ]),
            "atom_mask": torch.ones((2, 2), dtype=torch.bool),
            "atom_to_motif": torch.zeros((2, 2), dtype=torch.long),
            # Model-atom order differs, while canonical local positions agree.
            "atom_local_positions": torch.tensor([[1, 0], [0, 1]]),
            "motif_mask": torch.ones((2, 1), dtype=torch.bool),
            "motif_to_carrier": torch.zeros((2, 1), dtype=torch.long),
            "identity_span_bounds": torch.tensor([[[0, 2]], [[0, 2]]]),
            "endpoint_token_to_atom": torch.full((2, 4), -1, dtype=torch.long),
            "atom_is_attachment": torch.zeros((2, 2), dtype=torch.bool),
            "labels": torch.full((2, 4), -100, dtype=torch.long),
        }
        self.batch = CachedV3Batch(
            view_id="m_plus_g",
            epoch=0,
            record_ids=("left", "right"),
            exact_identity_sha256=((_digest("shared"),), (_digest("shared"),)),
            inputs=values,
            labels=values["labels"],
        )

    def test_target_is_same_canonical_atom_despite_model_order(self) -> None:
        plan = build_plan(self.batch, seed=607)
        self.assertTrue(
            torch.equal(
                plan.target_local_positions,
                plan.target_local_positions[:1].expand_as(plan.target_local_positions),
            )
        )
        self.assertNotEqual(
            int(plan.target_atom_indices[0, 0]),
            int(plan.target_atom_indices[1, 0]),
        )
        self.assertTrue((plan.visible_peer_counts == 1).all())

    def test_frozen_t5_and_detached_target_still_train_adapter_and_head(self) -> None:
        freeze_t5_for_held_out(self.model, self.head)
        plan = build_plan(self.batch, seed=607)
        output = forward_held_out_matching(
            self.model,
            self.head,
            self.batch,
            plan,
            component_mode="carrier_only",
        )
        self.assertEqual(output.eligible_anchors, 2)
        output.loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.t5.parameters()))
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in self.model.adapter.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in self.head.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_zero_component_cannot_distinguish_same_identity_same_atom_rows(self) -> None:
        freeze_t5_for_held_out(self.model, self.head)
        plan = build_plan(self.batch, seed=607)
        output = forward_held_out_matching(
            self.model,
            self.head,
            self.batch,
            plan,
            component_mode="zero",
        )
        self.assertGreaterEqual(float(output.loss), output.uniform_loss)
        self.assertAlmostEqual(output.accuracy, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
