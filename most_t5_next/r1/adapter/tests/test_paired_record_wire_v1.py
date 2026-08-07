from __future__ import annotations

import copy
import hashlib
import json
import unittest

from most_t5_next.r1.adapter import paired_record_wire_v1 as subject
from most_t5_next.r1.adapter.tests.test_production_paired_identity_records_v1 import (
    _build,
    _mol,
    _tokenizer,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fixture_pair():
    mol = _mol("CC(O)F")
    groups = tuple((atom_id,) for atom_id in range(mol.GetNumAtoms()))
    return _build(mol, groups, tokenizer=_tokenizer(mol))


class PairedRecordWireV1Test(unittest.TestCase):
    def test_canonical_bytes_round_trip_without_graph_payload(self) -> None:
        pair = _fixture_pair()
        payload = subject.encode_paired_training_record(
            pair,
            schedule_index=7,
            sdf_record_index=41,
        )
        loaded = subject.decode_paired_training_record(payload)

        self.assertEqual(loaded.schedule_index, 7)
        self.assertEqual(loaded.sdf_record_index, 41)
        self.assertEqual(loaded.atom_record, pair.atom_record)
        self.assertEqual(loaded.motif_record, pair.motif_record)
        self.assertEqual(loaded.receipt, pair.receipt)
        self.assertEqual(
            loaded.surface_summary.motif_identity_token_counts,
            tuple(len(surface.tokens) for surface in pair.motif_identity_surfaces),
        )
        self.assertNotIn(b"graph_encoding", payload)
        self.assertNotIn(b"motif_identity_surfaces", payload)
        self.assertEqual(payload, _canonical_bytes(json.loads(payload)))

    def test_a_semantic_tamper_is_caught_by_original_artifact_formula(self) -> None:
        pair = _fixture_pair()
        document = subject.paired_record_to_document(
            pair,
            schedule_index=0,
            sdf_record_index=41,
        )
        document["atom_document"]["input_ids"][1] += 1

        with self.assertRaisesRegex(
            subject.PairedRecordWireError,
            "A record artifact hash",
        ):
            subject.decode_paired_training_record(_canonical_bytes(document))

    def test_m_topology_tamper_reenters_vnext_validator(self) -> None:
        pair = _fixture_pair()
        document = subject.paired_record_to_document(
            pair,
            schedule_index=0,
            sdf_record_index=41,
        )
        connection_rows = document["motif_training_document"][
            "logical_motif_domain"
        ]["connection_token_indices"]
        first_nonempty = next(row for row in connection_rows if row)
        first_nonempty[0] = 0

        with self.assertRaisesRegex(
            subject.PairedRecordWireError,
            "M document failed production vNext validation",
        ):
            subject.decode_paired_training_record(_canonical_bytes(document))

    def test_receipt_pair_parity_tamper_is_rejected(self) -> None:
        pair = _fixture_pair()
        document = subject.paired_record_to_document(
            pair,
            schedule_index=0,
            sdf_record_index=41,
        )
        document["receipt"]["effective_inherited_overlay_content_sha256"] = hashlib.sha256(
            b"different-effective-overlay-row"
        ).hexdigest()

        with self.assertRaisesRegex(
            subject.PairedRecordWireError,
            "A/M/receipt member, tokenizer, source or effective geometry parity failed",
        ):
            subject.decode_paired_training_record(_canonical_bytes(document))

    def test_surface_connection_count_tamper_disagrees_with_m_endpoints(self) -> None:
        pair = _fixture_pair()
        document = subject.paired_record_to_document(
            pair,
            schedule_index=0,
            sdf_record_index=41,
        )
        document["surface_summary"]["cross_motif_connection_count"] += 1

        with self.assertRaisesRegex(
            subject.PairedRecordWireError,
            "cross-motif connection count disagrees with M endpoint markers",
        ):
            subject.decode_paired_training_record(_canonical_bytes(document))

    def test_extra_fields_and_nonfinite_json_are_rejected(self) -> None:
        pair = _fixture_pair()
        baseline = subject.paired_record_to_document(
            pair,
            schedule_index=0,
            sdf_record_index=41,
        )
        mutations = []
        envelope_extra = copy.deepcopy(baseline)
        envelope_extra["unexpected"] = 1
        mutations.append(envelope_extra)
        atom_extra = copy.deepcopy(baseline)
        atom_extra["atom_document"]["unexpected"] = 1
        mutations.append(atom_extra)
        summary_extra = copy.deepcopy(baseline)
        summary_extra["surface_summary"]["unexpected"] = 1
        mutations.append(summary_extra)

        for mutated in mutations:
            with self.subTest(keys=sorted(mutated)):
                with self.assertRaises(subject.PairedRecordWireError):
                    subject.decode_paired_training_record(_canonical_bytes(mutated))

        with self.assertRaisesRegex(subject.PairedRecordWireError, "non-finite"):
            subject.decode_paired_training_record(b'{"schema_version":NaN}')


if __name__ == "__main__":
    unittest.main()
