from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import unittest

from most_t5_next.p1 import (
    P1ArtifactBindings,
    P1MemberRef,
    ProductionBridgeError,
    ProductionTokenizerRuntime,
    SyntheticCEFirstCollator,
    collate_production_batch,
    collate_production_motif_record,
    collate_production_training_record,
    load_production_motif_record,
    materialize_training_record,
)
from most_t5_next.p1.tests.test_ce_collator import make_fixture
from most_t5_next.r1.gates import validate_p1_logical_motif_vnext as validator


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProductionBridgeTest(unittest.TestCase):
    def setUp(self):
        codec, token_table, sentinels, eos_token_id, build = make_fixture()
        self.codec = codec
        self.token_table = token_table
        self.sentinels = sentinels
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id + 100
        self.record = build((0, 1, 2, 3))
        collator = SyntheticCEFirstCollator(
            codec=codec,
            token_to_id=token_table,
            sentinel_token_ids=sentinels,
            eos_token_id=eos_token_id,
            seed=19,
            mask_probability=0.5,
        )
        self.synthetic_example = collator.collate_record(self.record, epoch=3)
        bindings = P1ArtifactBindings(
            release_id="production-bridge-fixture",
            data_release_manifest_sha256=digest("release"),
            geometry_record_schema_sha256=digest("geometry-schema"),
            geometry_record_content_sha256=digest("geometry-record"),
            membership_manifest_sha256=digest("membership"),
            tokenizer_contract_sha256=digest("tokenizer-contract"),
            tokenizer_snapshot_sha256=self.record.token_table_sha256,
            identity_codec_sha256=digest("identity-codec"),
            connection_codec_sha256=digest("connection-codec"),
        )
        self.document = materialize_training_record(
            record=self.record,
            example=self.synthetic_example,
            bindings=bindings,
            member=P1MemberRef(self.record.record_id, "fixture/000000"),
            codec=codec,
            token_to_id=token_table,
            sentinel_token_ids=sentinels,
            eos_token_id=eos_token_id,
        )
        self.tokenizer = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=digest("tokenizer-contract"),
            tokenizer_snapshot_sha256=self.record.token_table_sha256,
            vocab_size=self.pad_token_id + 1,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            sentinel_token_ids=self.sentinels,
        )
        self.production_record = load_production_motif_record(self.document)

    def test_validated_production_record_reaches_the_same_t5_ce_boundary(self):
        example = collate_production_training_record(
            self.document,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(example.input_ids, self.synthetic_example.input_ids)
        self.assertEqual(example.labels, self.synthetic_example.labels)
        self.assertEqual(
            example.identity_recovery_mask,
            self.synthetic_example.identity_recovery_mask,
        )
        self.assertEqual(example.full_e3fp_ids, self.record.full_e3fp_ids)
        self.assertEqual(
            example.model_to_source_atom_index,
            self.record.model_to_source_atom_index,
        )
        for atom_index, motif_id in enumerate(example.atom_to_logical_motif):
            self.assertEqual(
                example.atom_to_carrier[atom_index],
                example.logical_to_carrier[motif_id],
            )

    def test_m0_and_m1_share_ce_but_only_m1_exposes_geometry(self):
        m0 = collate_production_batch(
            (self.production_record,),
            condition_id="M0",
            tokenizer=self.tokenizer,
            seed=19,
            epoch=3,
            mask_probability=0.5,
        )
        m1 = collate_production_batch(
            (self.production_record,),
            condition_id="M1",
            tokenizer=self.tokenizer,
            seed=19,
            epoch=3,
            mask_probability=0.5,
        )
        self.assertEqual(m0.ce_batch, m1.ce_batch)
        self.assertIsNone(m0.geometry)
        self.assertEqual(set(m0.t5_inputs()), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(m0.geometry_inputs(), {})
        self.assertIsNotNone(m1.geometry)
        self.assertEqual(
            set(m1.geometry_inputs()),
            {"e3fp_ids", "e3fp_atom_mask", "e3fp_atom_to_token"},
        )
        self.assertEqual(m1.geometry.record_ids, m1.ce_batch.record_ids)
        self.assertEqual(
            m1.geometry.model_to_source_atom_index,
            (self.production_record.model_to_source_atom_index,),
        )

    def test_atom_baselines_are_not_fabricated_from_motif_records(self):
        for condition_id in ("A0", "A1"):
            with self.subTest(condition_id=condition_id), self.assertRaises(
                ProductionBridgeError
            ):
                collate_production_batch(
                    (self.production_record,),
                    condition_id=condition_id,
                    tokenizer=self.tokenizer,
                    seed=19,
                    epoch=3,
                    mask_probability=0.5,
                )

    def test_mutated_production_mapping_fails_at_the_existing_contract_gate(self):
        broken = copy.deepcopy(self.document)
        carrier = broken["logical_motif_domain"]["logical_to_carrier"][0]
        broken["token_domain"]["token_to_logical_motif"][carrier] = -1
        with self.assertRaisesRegex(ProductionBridgeError, "failed validation"):
            collate_production_training_record(
                broken,
                tokenizer=self.tokenizer,
            )

    def test_runtime_special_ids_are_bound_to_the_declared_tokenizer(self):
        with self.assertRaisesRegex(ProductionBridgeError, "snapshot hash differs"):
            collate_production_training_record(
                self.document,
                tokenizer=replace(
                    self.tokenizer,
                    tokenizer_snapshot_sha256=digest("other-tokenizer-snapshot"),
                ),
            )
        with self.assertRaises(ProductionBridgeError):
            ProductionTokenizerRuntime(
                tokenizer_contract_sha256=digest("tokenizer-contract"),
                tokenizer_snapshot_sha256=self.record.token_table_sha256,
                vocab_size=max(self.sentinels),
                pad_token_id=0,
                eos_token_id=self.eos_token_id,
                sentinel_token_ids=self.sentinels,
            )

    def test_once_validated_record_is_reused_for_epoch_keyed_masks(self):
        first = collate_production_motif_record(
            self.production_record,
            tokenizer=self.tokenizer,
            seed=19,
            epoch=3,
            mask_probability=0.5,
        )
        replay = collate_production_motif_record(
            self.production_record,
            tokenizer=self.tokenizer,
            seed=19,
            epoch=3,
            mask_probability=0.5,
        )
        next_epoch = collate_production_motif_record(
            self.production_record,
            tokenizer=self.tokenizer,
            seed=19,
            epoch=4,
            mask_probability=0.5,
        )
        self.assertEqual(first, replay)
        self.assertNotEqual(first.mask_decision_sha256, next_epoch.mask_decision_sha256)

    def test_noncontiguous_connection_indices_remain_visible(self):
        document = self._noncontiguous_connection_record()
        report = validator.validate_training_record(document)
        self.assertTrue(report["pass"], report["errors"])
        example = collate_production_training_record(
            document,
            tokenizer=ProductionTokenizerRuntime(
                tokenizer_contract_sha256=digest("tokenizer-contract"),
                tokenizer_snapshot_sha256=digest("tokenizer-snapshot"),
                vocab_size=201,
                pad_token_id=200,
                eos_token_id=99,
                sentinel_token_ids=(100, 101),
            ),
        )
        self.assertEqual(example.input_ids, (0, 100, 20, 12, 21, 1))
        self.assertEqual(example.connection_input_indices, ((2, 4),))
        self.assertEqual(tuple(example.input_ids[index] for index in (2, 4)), (20, 21))
        self.assertEqual(example.labels, (100, 10, 11, 101, 99))

    def _noncontiguous_connection_record(self):
        seed, epoch, probability = 1, 0, 1.0
        member_id = "pcqm4mv2:noncontiguous-fixture"
        decision_sha = validator._mask_decision_sha256(
            seed, epoch, member_id, "identity_recovery_ce", probability, [0]
        )
        return {
            "schema_version": validator.RECORD_SCHEMA,
            "document_kind": validator.RECORD_KIND,
            "training_profile": validator.CE_PROFILE,
            "bindings": {
                "release_id": "fixture-release",
                "data_release_manifest_sha256": digest("release"),
                "geometry_record_schema_sha256": digest("geometry-schema"),
                "geometry_record_content_sha256": digest("geometry-record"),
                "membership_manifest_sha256": digest("membership"),
                "tokenizer_contract_sha256": digest("tokenizer-contract"),
                "tokenizer_snapshot_sha256": digest("tokenizer-snapshot"),
                "identity_codec_sha256": digest("identity-codec"),
                "connection_codec_sha256": digest("connection-codec"),
            },
            "member": {"member_id": member_id, "storage_key": "000000001"},
            "dimensions": {
                "token_count": 7,
                "logical_motif_count": 1,
                "atom_count": 1,
                "source_atom_count": 1,
                "e3fp_level_count": 4,
            },
            "token_domain": {
                "input_ids": [0, 10, 11, 20, 12, 21, 1],
                "attention_mask": [True] * 7,
                "token_to_logical_motif": [-1, 0, 0, 0, -1, 0, -1],
                "token_role": [
                    "boundary",
                    "identity",
                    "identity",
                    "connection",
                    "boundary",
                    "connection",
                    "boundary",
                ],
            },
            "logical_motif_domain": {
                "identity_spans": [[1, 3]],
                "connection_token_indices": [[3, 5]],
                "logical_to_carrier": [1],
                "exact_identity_sha256": [digest("motif")],
                "motif_geometry_valid": [True],
                "motif_atom_indices": [[0]],
                "motif_slot_atom_indices": [[]],
                "slot_count": [0],
                "cross_motif_bonds": [],
            },
            "atom_domain": {
                "atom_to_logical_motif": [0],
                "model_to_source_atom_index": [0],
                "atom_valid_mask": [True],
                "atom_is_attachment": [False],
                "full_e3fp_ids": [[10, 11, 12, 13]],
            },
            "masks": {"identity_recovery_mask": [True]},
            "mask_decision": {
                "objective": "identity_recovery_ce",
                "seed": seed,
                "epoch": epoch,
                "mask_probability": probability,
                "selected_logical_motif_indices": [0],
                "decision_sha256": decision_sha,
            },
        }


if __name__ == "__main__":
    unittest.main()
