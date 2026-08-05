from __future__ import annotations

from dataclasses import replace
import unittest

from most_t5_next.p1 import (
    BoundRecordInvariantError,
    CodecContractError,
    ConnectionEndpoint,
    CrossMotifConnection,
    HybridMotifCodec,
    LogicalMoleculeSchema,
    LogicalMotif,
    LogicalMotifIdentity,
    build_bound_record,
    build_synthetic_token_table,
)


def fixture_schema():
    carbonyl = LogicalMotifIdentity(
        ("ATOM:C", "ATOM:O", "BOND:DOUBLE:0-1", "STEREO:NONE"),
        (0,),
    )
    amine = LogicalMotifIdentity(
        ("ATOM:N", "ATOM:C", "BOND:SINGLE:0-1", "STEREO:NONE"),
        (0,),
    )
    motifs = (
        LogicalMotif(0, carbonyl, (0, 1), True),
        LogicalMotif(1, amine, (2, 3), True),
    )
    connection = CrossMotifConnection.canonical(
        0,
        ConnectionEndpoint(0, 0, 0),
        ConnectionEndpoint(1, 0, 2),
        "single",
    )
    return LogicalMoleculeSchema(motifs, (connection,)), carbonyl, amine


class HybridCodecTest(unittest.TestCase):
    def test_macro_and_forced_fallback_round_trip_to_same_identity(self):
        schema, carbonyl, amine = fixture_schema()
        codec = HybridMotifCodec((amine, carbonyl))
        reversed_codec = HybridMotifCodec((carbonyl, amine))

        macro = codec.verify_round_trip(carbonyl)
        fallback = codec.verify_round_trip(carbonyl, force_fallback=True)

        self.assertEqual(macro.mode, "macro")
        self.assertEqual(fallback.mode, "fallback")
        self.assertEqual(macro.carrier_offset, fallback.carrier_offset)
        self.assertEqual(macro.exact_identity_digest, fallback.exact_identity_digest)
        self.assertEqual(codec.decode(macro.tokens), codec.decode(fallback.tokens))
        self.assertNotIn("<unk>", macro.tokens + fallback.tokens)
        self.assertEqual(codec.macro_tokens, reversed_codec.macro_tokens)
        self.assertEqual(
            build_synthetic_token_table(codec, schema),
            build_synthetic_token_table(reversed_codec, schema),
        )

    def test_fallback_boundary_damage_fails_closed(self):
        _, carbonyl, _ = fixture_schema()
        codec = HybridMotifCodec(())
        fallback = codec.encode(carbonyl)
        with self.assertRaises(CodecContractError):
            codec.decode(fallback.tokens[:-1])


class ConnectionSchemaTest(unittest.TestCase):
    def test_every_declared_slot_must_be_connected_once(self):
        identity = LogicalMotifIdentity(("ATOM:C",), (0,))
        with self.assertRaises(CodecContractError):
            LogicalMoleculeSchema((LogicalMotif(0, identity, (0,)),), ())

    def test_endpoint_atom_must_match_declared_slot_position(self):
        schema, _, _ = fixture_schema()
        bad_connection = CrossMotifConnection.canonical(
            0,
            ConnectionEndpoint(0, 0, 1),
            ConnectionEndpoint(1, 0, 2),
            "single",
        )
        with self.assertRaises(CodecContractError):
            LogicalMoleculeSchema(schema.motifs, (bad_connection,))


class BoundRecordTest(unittest.TestCase):
    def setUp(self):
        self.schema, carbonyl, amine = fixture_schema()
        self.codec = HybridMotifCodec((carbonyl, amine))
        self.token_table = build_synthetic_token_table(self.codec, self.schema)
        self.e3fp = (
            (11, 12, 13, 14),
            (21, 22, 23, 24),
            (31, 32, 33, 34),
            (41, 42, 43, 44),
        )

    def build(self, forced=()):
        return build_bound_record(
            record_id="synthetic:amide-like:0",
            schema=self.schema,
            codec=self.codec,
            token_to_id=self.token_table,
            full_e3fp_ids=self.e3fp,
            source_atom_count=6,
            model_to_source_atom_index=(0, 1, 3, 5),
            force_fallback_motif_ids=forced,
        )

    def test_macro_and_fallback_preserve_logical_and_atom_domains(self):
        macro_record = self.build()
        fallback_record = self.build((0, 1))

        self.assertNotEqual(macro_record.surface_tokens, fallback_record.surface_tokens)
        self.assertEqual(macro_record.exact_identity_digest, fallback_record.exact_identity_digest)
        self.assertEqual(macro_record.motif_atom_indices, fallback_record.motif_atom_indices)
        self.assertEqual(
            macro_record.atom_to_logical_motif,
            fallback_record.atom_to_logical_motif,
        )
        self.assertEqual(
            macro_record.cross_motif_connections,
            fallback_record.cross_motif_connections,
        )
        self.assertEqual(macro_record.full_e3fp_ids, fallback_record.full_e3fp_ids)

        for record in (macro_record, fallback_record):
            record.validate(self.codec, self.token_table)
            self.assertEqual(len(set(record.logical_to_carrier)), 2)
            for motif_id, carrier in enumerate(record.logical_to_carrier):
                self.assertEqual(carrier, record.identity_spans[motif_id].start)
                self.assertEqual(record.token_to_logical_motif[carrier], motif_id)
                self.assertEqual(record.token_role[carrier], "identity")

    def test_carrier_and_atom_mapping_mutations_fail_closed(self):
        record = self.build((0,))
        bad_carrier = replace(
            record,
            logical_to_carrier=(record.logical_to_carrier[0] + 1, record.logical_to_carrier[1]),
        )
        with self.assertRaises(BoundRecordInvariantError):
            bad_carrier.validate(self.codec, self.token_table)

        bad_atom_map = replace(record, atom_to_logical_motif=(0, 0, 0, 1))
        with self.assertRaises(BoundRecordInvariantError):
            bad_atom_map.validate(self.codec, self.token_table)

    def test_source_mapping_and_attachment_mutations_fail_closed(self):
        record = self.build()
        self.assertEqual(record.model_to_source_atom_index, (0, 1, 3, 5))
        self.assertEqual(record.atom_is_attachment, (True, False, True, False))

        for bad_mapping in ((0, 3, 1, 5), (0, 1, 3, 6)):
            with self.subTest(mapping=bad_mapping):
                with self.assertRaises(BoundRecordInvariantError):
                    replace(record, model_to_source_atom_index=bad_mapping).validate(
                        self.codec, self.token_table
                    )

        with self.assertRaises(BoundRecordInvariantError):
            replace(record, atom_is_attachment=(False, False, True, False)).validate(
                self.codec, self.token_table
            )

    def test_narrow_p1_geometry_policy_rejects_all_minus_one_row(self):
        record = self.build()
        mutated_e3fp = list(record.full_e3fp_ids)
        mutated_e3fp[1] = (-1, -1, -1, -1)
        with self.assertRaises(BoundRecordInvariantError):
            replace(record, full_e3fp_ids=tuple(mutated_e3fp)).validate(
                self.codec, self.token_table
            )

    def test_lowercase_bond_closed_set_and_canonical_edge_order(self):
        schema, _, _ = fixture_schema()
        with self.assertRaises(CodecContractError):
            CrossMotifConnection.canonical(
                0,
                schema.connections[0].endpoint_a,
                schema.connections[0].endpoint_b,
                "SINGLE",
            )

    def test_connection_surface_mutation_fails_closed(self):
        record = self.build()
        connection_position = record.connection_spans[0].start + 2
        mutated_tokens = list(record.surface_tokens)
        mutated_tokens[connection_position] = "<BOND:single>"
        mutated_ids = tuple(self.token_table[token] for token in mutated_tokens)
        mutated = replace(
            record,
            surface_tokens=tuple(mutated_tokens),
            input_ids=mutated_ids,
        )
        with self.assertRaises(BoundRecordInvariantError):
            mutated.validate(self.codec, self.token_table)


if __name__ == "__main__":
    unittest.main()
