from __future__ import annotations

import hashlib
import unittest

import torch

from most_t5_next.p2.factorized_motif_t5_v3 import FactorizedMotifT5V3
from most_t5_next.p2.motif_state_matching_v3 import MotifStateMatchingHeadV3
from most_t5_next.p2.pf10_training_tensor_cache_v1 import CachedV3Batch
from most_t5_next.p2.run_pf10_3d_motif_v3_matching_only_v1 import (
    V3MatchingOnlyError,
    forward_matching_only,
    freeze_t5_for_matching_only,
)
from most_t5_next.p2.tests.test_factorized_motif_t5_v3 import _TinyT5


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class V3MatchingOnlyRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(503)
        self.model = FactorizedMotifT5V3(
            _TinyT5(),
            num_e3fp_embeddings=16,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            geometry_fraction=0.15,
        )
        self.head = MotifStateMatchingHeadV3(
            hidden_size=8,
            projection_dim=4,
            temperature=0.1,
        )
        values = {
            "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]]),
            "attention_mask": torch.ones((2, 6), dtype=torch.long),
            "e3fp_mask_token_id": 17,
            "e3fp_input_ids": torch.tensor([
                [[7, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]],
                [[3, 4, 5, 6], [4, 5, 6, 7], [9, 10, 11, 12]],
            ]),
            "atom_mask": torch.ones((2, 3), dtype=torch.bool),
            "atom_to_motif": torch.tensor([[0, 0, 1], [0, 0, 1]]),
            "atom_local_positions": torch.tensor([[0, 1, 0], [0, 1, 0]]),
            "motif_mask": torch.ones((2, 2), dtype=torch.bool),
            "motif_to_carrier": torch.tensor([[0, 3], [0, 3]]),
            "identity_span_bounds": torch.tensor([
                [[0, 2], [3, 5]],
                [[0, 2], [3, 5]],
            ]),
            "endpoint_token_to_atom": torch.tensor([
                [-1, -1, 1, -1, -1, 2],
                [-1, -1, 1, -1, -1, 2],
            ]),
            "atom_is_attachment": torch.tensor([
                [False, True, True],
                [False, True, True],
            ]),
            "labels": torch.full((2, 6), -100, dtype=torch.long),
        }
        self.batch = CachedV3Batch(
            view_id="m_plus_g",
            epoch=0,
            record_ids=("left", "right"),
            exact_identity_sha256=(
                (_digest("first"), _digest("second")),
                (_digest("first"), _digest("second")),
            ),
            inputs=values,
            labels=values["labels"],
        )

    def test_frozen_t5_transmits_gradients_without_receiving_them(self) -> None:
        freeze_t5_for_matching_only(self.model)
        output = forward_matching_only(
            self.model,
            self.head,
            self.batch,
            component_mode="both",
        )
        self.assertEqual(output.eligible_anchors, 4)
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

    def test_zero_component_is_the_exact_uniform_matching_baseline(self) -> None:
        freeze_t5_for_matching_only(self.model)
        output = forward_matching_only(
            self.model,
            self.head,
            self.batch,
            component_mode="zero",
        )
        self.assertAlmostEqual(float(output.loss), output.uniform_loss, places=6)
        self.assertAlmostEqual(output.accuracy, 0.5, places=6)
        self.assertAlmostEqual(output.mean_positive_probability, 0.5, places=6)

    def test_unknown_component_mode_fails_before_training(self) -> None:
        with self.assertRaisesRegex(V3MatchingOnlyError, "component"):
            forward_matching_only(
                self.model,
                self.head,
                self.batch,
                component_mode="everything",
            )


if __name__ == "__main__":
    unittest.main()
