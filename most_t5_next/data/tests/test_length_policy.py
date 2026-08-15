from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from most_t5_next.data.length_policy import (
    LengthPolicy,
    write_length_action_ledger,
)


class LengthPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = LengthPolicy()

    def test_text_denoising_uses_the_derived_568_512_114_contract(self) -> None:
        decision = self.policy.decide(
            record_id="block:1", task="TXT", input_length=568, target_length=114
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.action, "online_span_corruption_568_to_512_114")

    def test_t2m_truncates_only_the_text_input(self) -> None:
        decision = self.policy.decide(
            record_id="cid:1", task="T2M", input_length=800, target_length=200
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.action, "truncate_text_input_right")
        structural = self.policy.decide(
            record_id="cid:2", task="T2M", input_length=200, target_length=600
        )
        self.assertFalse(structural.admitted)
        self.assertEqual(structural.action, "exclude_structural_target_view")

    def test_cap_truncates_only_the_text_target(self) -> None:
        decision = self.policy.decide(
            record_id="cid:1", task="CAP", input_length=300, target_length=800
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.action, "truncate_text_target_right_keep_eos")
        structural = self.policy.decide(
            record_id="cid:2", task="CAP", input_length=600, target_length=20
        )
        self.assertFalse(structural.admitted)

    def test_molecular_denoising_never_uses_ordinary_truncation(self) -> None:
        for task in ("M", "MG", "SYN"):
            decision = self.policy.decide(
                record_id="mol:1", task=task, input_length=513, target_length=20
            )
            self.assertFalse(decision.admitted)
            self.assertEqual(decision.action, "exclude_structural_task_view")

            long_target = self.policy.decide(
                record_id="mol:2", task=task, input_length=512, target_length=115
            )
            self.assertFalse(long_target.admitted)
            self.assertEqual(long_target.action, "exclude_structural_task_view")

    def test_ledger_records_both_admitted_and_excluded_views(self) -> None:
        decisions = (
            self.policy.decide(
                record_id="a", task="CAP", input_length=20, target_length=30
            ),
            self.policy.decide(
                record_id="b", task="CAP", input_length=600, target_length=30
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            receipt = write_length_action_ledger(
                decisions, Path(temporary) / "ledger.jsonl"
            )
            self.assertEqual(receipt["records"], 2)
            self.assertEqual(receipt["admitted"], 1)
            self.assertEqual(receipt["excluded"], 1)


if __name__ == "__main__":
    unittest.main()
