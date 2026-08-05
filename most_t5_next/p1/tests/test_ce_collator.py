from __future__ import annotations

from dataclasses import replace
import unittest

from most_t5_next.p1 import (
    CollatorContractError,
    ConnectionEndpoint,
    CrossMotifConnection,
    HybridMotifCodec,
    LogicalMoleculeSchema,
    LogicalMotif,
    LogicalMotifIdentity,
    SyntheticCEFirstCollator,
    build_bound_record,
    build_synthetic_token_table,
)


def make_fixture():
    identities = tuple(
        LogicalMotifIdentity(
            (
                f"ATOM:{element}",
                "ATOM:C",
                "BOND:SINGLE:0-1",
                "STEREO:NONE",
            ),
            slots,
        )
        for element, slots in (
            ("N", (1,)),
            ("O", (0, 1)),
            ("S", (0, 1)),
            ("F", (0,)),
        )
    )
    motifs = tuple(
        LogicalMotif(motif_id, identity, (motif_id * 2, motif_id * 2 + 1), True)
        for motif_id, identity in enumerate(identities)
    )
    edges = (
        CrossMotifConnection.canonical(
            0,
            ConnectionEndpoint(0, 0, 1),
            ConnectionEndpoint(1, 0, 2),
            "single",
        ),
        CrossMotifConnection.canonical(
            1,
            ConnectionEndpoint(1, 1, 3),
            ConnectionEndpoint(2, 0, 4),
            "single",
        ),
        CrossMotifConnection.canonical(
            2,
            ConnectionEndpoint(2, 1, 5),
            ConnectionEndpoint(3, 0, 6),
            "single",
        ),
    )
    schema = LogicalMoleculeSchema(motifs, edges)
    codec = HybridMotifCodec(identities)
    token_table = build_synthetic_token_table(codec, schema)
    e3fp = tuple(
        (atom * 4, atom * 4 + 1, atom * 4 + 2, atom * 4 + 3)
        for atom in range(schema.atom_count)
    )

    def build(forced=()):
        return build_bound_record(
            record_id="synthetic:four-motif-chain:0",
            schema=schema,
            codec=codec,
            token_to_id=token_table,
            full_e3fp_ids=e3fp,
            source_atom_count=schema.atom_count,
            model_to_source_atom_index=tuple(range(schema.atom_count)),
            force_fallback_motif_ids=forced,
        )

    sentinel_start = max(token_table.values()) + 1
    sentinels = tuple(range(sentinel_start, sentinel_start + len(motifs) + 1))
    eos_token_id = sentinels[-1] + 1
    return codec, token_table, sentinels, eos_token_id, build


class SyntheticCEFirstCollatorTest(unittest.TestCase):
    def setUp(self):
        (
            self.codec,
            self.token_table,
            self.sentinels,
            self.eos_token_id,
            self.build,
        ) = make_fixture()

    def collator(self, probability=0.5, seed=73):
        return SyntheticCEFirstCollator(
            codec=self.codec,
            token_to_id=self.token_table,
            sentinel_token_ids=self.sentinels,
            eos_token_id=self.eos_token_id,
            seed=seed,
            mask_probability=probability,
        )

    def test_same_key_is_stateless_and_deterministic(self):
        record = self.build()
        collator = self.collator()
        first = collator.collate_record(record, epoch=4)
        second = collator.collate_record(record, epoch=4)
        fresh_instance = self.collator().collate_record(record, epoch=4)

        self.assertEqual(first, second)
        self.assertEqual(second, fresh_instance)

    def test_selected_identity_is_one_sentinel_and_whole_target(self):
        record = self.build((0, 1, 2, 3))
        example = self.collator(probability=1.0).collate_record(record, epoch=0)

        self.assertEqual(example.identity_recovery_mask, (True, True, True, True))
        for target in example.masked_identity_targets:
            original = record.identity_spans[target.logical_motif_id]
            self.assertEqual(
                target.original_input_ids,
                record.input_ids[original.start : original.stop],
            )
            self.assertEqual(target.corrupted_span.stop - target.corrupted_span.start, 1)
            self.assertEqual(
                example.input_ids[target.corrupted_span.start : target.corrupted_span.stop],
                (target.sentinel_id,),
            )
        self.assertTrue(example.labels)
        self.assertEqual(example.labels[-2], self.sentinels[4])
        self.assertEqual(example.labels[-1], self.eos_token_id)
        self.assertFalse(example.state_prediction_enabled)

    def test_macro_and_fallback_have_same_logical_mask_and_visibility(self):
        macro = self.build()
        fallback = self.build((0, 1, 2, 3))
        collator = self.collator(probability=0.5)
        macro_example = collator.collate_record(macro, epoch=7)
        fallback_example = collator.collate_record(fallback, epoch=7)

        self.assertEqual(
            macro_example.identity_recovery_mask,
            fallback_example.identity_recovery_mask,
        )
        self.assertEqual(macro_example.full_e3fp_ids, macro.full_e3fp_ids)
        self.assertEqual(fallback_example.full_e3fp_ids, fallback.full_e3fp_ids)
        self.assertTrue(all(macro_example.connection_span_visible))
        self.assertTrue(all(fallback_example.connection_span_visible))
        self.assertFalse(any(macro_example.geometry_corruption_mask))
        self.assertFalse(any(fallback_example.geometry_corruption_mask))

    def test_epoch_participates_in_stateless_mask_key(self):
        record = self.build()
        collator = self.collator(probability=0.5, seed=0)
        epoch_zero = collator.collate_record(record, epoch=0).identity_recovery_mask
        epoch_one = collator.collate_record(record, epoch=1).identity_recovery_mask
        self.assertNotEqual(epoch_zero, epoch_one)

    def test_at_least_one_motif_and_nonempty_target_gate(self):
        record = self.build()
        example = self.collator(probability=1e-12).collate_record(record, epoch=0)
        self.assertEqual(sum(example.identity_recovery_mask), 1)
        self.assertGreater(len(example.labels), 1)

    def test_state_prediction_objective_is_rejected(self):
        with self.assertRaises(CollatorContractError):
            SyntheticCEFirstCollator(
                codec=self.codec,
                token_to_id=self.token_table,
                sentinel_token_ids=self.sentinels,
                eos_token_id=self.eos_token_id,
                seed=1,
                objective="state_prediction_c3",
            )

    def test_mutated_eos_boundary_and_mask_hash_fail_closed(self):
        record = self.build((0, 1, 2, 3))
        example = self.collator(probability=0.5).collate_record(record, epoch=3)

        bad_labels = replace(
            example, labels=example.labels[:-1] + (self.eos_token_id + 99,)
        )
        bad_boundary = replace(
            example, input_ids=(example.input_ids[0] + 99,) + example.input_ids[1:]
        )
        bad_hash = replace(example, mask_decision_sha256="0" * 64)
        for mutated in (bad_labels, bad_boundary, bad_hash):
            with self.subTest(mutated=mutated):
                with self.assertRaises(CollatorContractError):
                    mutated.validate_against(
                        record, self.sentinels, self.eos_token_id
                    )

    def test_selected_count_plus_one_sentinel_gate(self):
        record = self.build()
        collator = SyntheticCEFirstCollator(
            codec=self.codec,
            token_to_id=self.token_table,
            sentinel_token_ids=self.sentinels[:4],
            eos_token_id=self.eos_token_id,
            seed=73,
            mask_probability=1.0,
        )
        with self.assertRaises(CollatorContractError):
            collator.collate_record(record, epoch=0)


if __name__ == "__main__":
    unittest.main()
