from __future__ import annotations

import hashlib
import unittest

import torch

from most_t5_next.p2.motif_state_matching_v3 import (
    MotifStateMatchingError,
    MotifStateMatchingHeadV3,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class MotifStateMatchingV3Test(unittest.TestCase):
    def _head(self) -> MotifStateMatchingHeadV3:
        head = MotifStateMatchingHeadV3(
            hidden_size=2,
            projection_dim=2,
            temperature=0.05,
        )
        with torch.no_grad():
            head.query_projection.weight.copy_(torch.eye(2))
            head.state_projection.weight.copy_(torch.eye(2))
        return head

    def test_same_identity_cross_record_retrieval_has_gradients(self) -> None:
        head = self._head()
        hidden = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            requires_grad=True,
        )
        state = hidden.detach().clone().requires_grad_(True)
        output = head(
            encoder_hidden=hidden,
            motif_state=state,
            motif_mask=torch.ones((2, 2), dtype=torch.bool),
            motif_to_carrier=torch.tensor([[0, 1], [0, 1]]),
            exact_identity_sha256=(
                (_digest("shared"), _digest("left-only")),
                (_digest("shared"), _digest("right-only")),
            ),
        )
        self.assertEqual(output.eligible_identity_groups, 1)
        self.assertEqual(output.eligible_anchors, 2)
        self.assertEqual(output.correct_anchors, 2)
        self.assertEqual(output.accuracy, 1.0)
        self.assertGreater(output.mean_positive_probability, 0.99)
        self.assertAlmostEqual(output.uniform_loss, torch.log(torch.tensor(2.0)).item())
        output.loss.backward()
        self.assertGreater(float(hidden.grad.abs().sum()), 0.0)
        self.assertGreater(float(state.grad.abs().sum()), 0.0)

    def test_same_record_duplicate_is_not_a_negative(self) -> None:
        head = self._head()
        output = head(
            encoder_hidden=torch.ones((1, 2, 2)),
            motif_state=torch.ones((1, 2, 2)),
            motif_mask=torch.ones((1, 2), dtype=torch.bool),
            motif_to_carrier=torch.tensor([[0, 1]]),
            exact_identity_sha256=((_digest("same"), _digest("same")),),
        )
        self.assertEqual(output.eligible_anchors, 0)
        self.assertEqual(float(output.loss), 0.0)

    def test_uniform_logits_receive_chance_credit_not_fake_perfect_accuracy(self) -> None:
        head = self._head()
        with torch.no_grad():
            head.query_projection.weight.zero_()
            head.state_projection.weight.zero_()
        output = head(
            encoder_hidden=torch.ones((2, 1, 2)),
            motif_state=torch.ones((2, 1, 2)),
            motif_mask=torch.ones((2, 1), dtype=torch.bool),
            motif_to_carrier=torch.zeros((2, 1), dtype=torch.long),
            exact_identity_sha256=((_digest("same"),), (_digest("same"),)),
        )
        self.assertEqual(output.correct_anchors, 2)  # legacy argmax tie behaviour
        self.assertEqual(output.accuracy, 0.5)
        self.assertEqual(output.mean_positive_probability, 0.5)
        self.assertAlmostEqual(float(output.loss), output.uniform_loss, places=6)

    def test_identity_count_must_match_active_motifs(self) -> None:
        head = self._head()
        with self.assertRaisesRegex(MotifStateMatchingError, "identity count"):
            head(
                encoder_hidden=torch.ones((1, 2, 2)),
                motif_state=torch.ones((1, 2, 2)),
                motif_mask=torch.ones((1, 2), dtype=torch.bool),
                motif_to_carrier=torch.tensor([[0, 1]]),
                exact_identity_sha256=((_digest("only-one"),),),
            )


if __name__ == "__main__":
    unittest.main()
