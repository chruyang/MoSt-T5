from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from most_t5_next.p1 import (
    P1ArtifactBindings,
    P1MemberRef,
    RuntimeBridgeError,
    SyntheticCEFirstCollator,
    materialize_training_record,
    pad_ce_first_batch,
)
from most_t5_next.p1.tests.test_ce_collator import make_fixture
from most_t5_next.r1.gates import validate_p1_logical_motif_vnext as validator


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RuntimeBridgeTest(unittest.TestCase):
    def setUp(self):
        (
            self.codec,
            self.token_table,
            self.sentinels,
            self.eos_token_id,
            self.build,
        ) = make_fixture()
        self.record = self.build((0, 1, 2, 3))
        self.collator = SyntheticCEFirstCollator(
            codec=self.codec,
            token_to_id=self.token_table,
            sentinel_token_ids=self.sentinels,
            eos_token_id=self.eos_token_id,
            seed=19,
            mask_probability=0.5,
        )
        self.example = self.collator.collate_record(self.record, epoch=3)
        self.bindings = P1ArtifactBindings(
            release_id="synthetic-release-candidate",
            data_release_manifest_sha256=digest("release"),
            geometry_record_schema_sha256=digest("geometry-schema"),
            geometry_record_content_sha256=digest("geometry-record"),
            membership_manifest_sha256=digest("membership"),
            tokenizer_contract_sha256=digest("tokenizer-contract"),
            tokenizer_snapshot_sha256=self.record.token_table_sha256,
            identity_codec_sha256=digest("identity-codec"),
            connection_codec_sha256=digest("connection-codec"),
        )
        self.member = P1MemberRef(self.record.record_id, "fixture/000000")

    def materialize(self):
        return materialize_training_record(
            record=self.record,
            example=self.example,
            bindings=self.bindings,
            member=self.member,
            codec=self.codec,
            token_to_id=self.token_table,
            sentinel_token_ids=self.sentinels,
            eos_token_id=self.eos_token_id,
        )

    def test_bound_record_to_contract_to_validator(self):
        document = self.materialize()
        report = validator.validate_training_record(document)
        self.assertTrue(report["pass"], report["errors"])
        self.assertEqual(document["token_domain"]["input_ids"], list(self.record.input_ids))
        self.assertNotEqual(document["token_domain"]["input_ids"], list(self.example.input_ids))
        self.assertEqual(
            document["masks"]["identity_recovery_mask"],
            list(self.example.identity_recovery_mask),
        )
        self.assertNotIn("labels", document)
        self.assertNotIn("identity_sentinel", document["token_domain"]["token_role"])

    def test_member_and_tokenizer_binding_fail_closed(self):
        with self.assertRaises(RuntimeBridgeError):
            materialize_training_record(
                record=self.record,
                example=self.example,
                bindings=self.bindings,
                member=P1MemberRef("wrong-member", "fixture/000000"),
                codec=self.codec,
                token_to_id=self.token_table,
                sentinel_token_ids=self.sentinels,
                eos_token_id=self.eos_token_id,
            )
        with self.assertRaises(RuntimeBridgeError):
            materialize_training_record(
                record=self.record,
                example=self.example,
                bindings=replace(
                    self.bindings,
                    tokenizer_snapshot_sha256=digest("different-tokenizer"),
                ),
                member=self.member,
                codec=self.codec,
                token_to_id=self.token_table,
                sentinel_token_ids=self.sentinels,
                eos_token_id=self.eos_token_id,
            )

    def test_batch_padding_and_strict_model_allowlist(self):
        second_record = replace(self.record, record_id="synthetic:four-motif-chain:1")
        second_record.validate(self.codec, self.token_table)
        second = self.collator.collate_record(second_record, epoch=4)
        batch = pad_ce_first_batch(
            (self.example, second),
            pad_token_id=self.eos_token_id + 100,
        )
        model_inputs = batch.model_inputs()
        self.assertEqual(set(model_inputs), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(len({len(row) for row in batch.input_ids}), 1)
        self.assertEqual(len({len(row) for row in batch.labels}), 1)
        for row, length in zip(batch.attention_mask, batch.input_lengths):
            self.assertEqual(sum(row), length)
            self.assertTrue(all(row[:length]))
            self.assertFalse(any(row[length:]))
        for row, length in zip(batch.labels, batch.target_lengths):
            self.assertTrue(all(value == -100 for value in row[length:]))

    def test_batch_rejects_duplicate_members_and_existing_padding(self):
        with self.assertRaises(RuntimeBridgeError):
            pad_ce_first_batch(
                (self.example, self.example),
                pad_token_id=self.eos_token_id + 100,
            )
        with self.assertRaises(RuntimeBridgeError):
            pad_ce_first_batch(
                (self.example,),
                pad_token_id=self.example.input_ids[0],
            )


if __name__ == "__main__":
    unittest.main()
