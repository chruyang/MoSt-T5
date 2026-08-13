from __future__ import annotations

from types import SimpleNamespace
import unittest

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import ProductionMotifRecord
from most_t5_next.p2.factorized_support_census_v1 import (
    FACTORISED_SUPPORT_CENSUS_ID,
    FactorizedSupportCensusError,
    census_factorized_support,
    iter_state_eligible_batches,
)


def _record(
    record_id: str,
    *,
    spans: tuple[tuple[int, int], ...],
    e3fp: tuple[tuple[int, int, int, int], ...],
    atom_to_motif: tuple[int, ...],
) -> object:
    motif_count = len(spans)
    token_count = max(stop for _, stop in spans)
    motif = ProductionMotifRecord(
        record_artifact_sha256="a" * 64,
        record_id=record_id,
        storage_key="key/" + record_id,
        release_id="fixture",
        geometry_record_content_sha256="b" * 64,
        tokenizer_contract_sha256="c" * 64,
        tokenizer_snapshot_sha256="d" * 64,
        input_ids=tuple(range(token_count)),
        token_to_logical_motif=tuple(0 for _ in range(token_count)),
        token_role=tuple("identity" for _ in range(token_count)),
        identity_spans=tuple(Span(start, stop) for start, stop in spans),
        connection_token_indices=tuple(() for _ in range(motif_count)),
        logical_to_carrier=tuple(start for start, _ in spans),
        exact_identity_sha256=tuple("e" * 64 for _ in range(motif_count)),
        source_atom_count=len(e3fp),
        full_e3fp_ids=e3fp,
        atom_valid_mask=tuple(True for _ in e3fp),
        model_to_source_atom_index=tuple(range(len(e3fp))),
        atom_to_logical_motif=atom_to_motif,
        atom_is_attachment=tuple(False for _ in e3fp),
    )
    return SimpleNamespace(motif_record=motif)


class _Reader:
    def __init__(self, train, dev=()):
        self.train = tuple(train)
        self.dev = tuple(dev)
        self.train_member_count = len(self.train)
        self.dev_member_count = len(self.dev)
        self.train_epochs = []

    @staticmethod
    def _batches(rows, batch_size):
        for start in range(0, len(rows), batch_size):
            yield rows[start : start + batch_size]

    def iter_train_epoch(self, *, epoch, batch_size):
        self.train_epochs.append(epoch)
        yield from self._batches(self.train, batch_size)

    def iter_dev(self, *, batch_size):
        yield from self._batches(self.dev, batch_size)


class FactorizedSupportCensusTest(unittest.TestCase):
    def setUp(self):
        self.a = _record(
            "a",
            spans=((0, 2), (2, 5)),
            e3fp=(
                (0, 11, 12, 13),
                (0, 21, 22, -1),
                (0, 31, -1, -1),
            ),
            atom_to_motif=(0, 0, 1),
        )
        self.b = _record(
            "b",
            spans=((0, 4),),
            e3fp=((0, 41, 42, -1), (0, 51, -1, -1)),
            atom_to_motif=(0, 0),
        )
        self.c = _record(
            "c",
            spans=((0, 5), (5, 6)),
            e3fp=(
                (0, 61, 62, -1),
                (0, 71, 72, -1),
                (0, 81, 82, -1),
                (0, 91, 92, -1),
                (0, 101, -1, -1),
            ),
            atom_to_motif=(0, 0, 1, 1, 1),
        )

    def test_streaming_census_freezes_only_two_atom_motif_support(self):
        reader = _Reader((self.a, self.b, self.c))
        census = census_factorized_support(reader, split="train", batch_size=2)

        self.assertEqual(census.schema_id, FACTORISED_SUPPORT_CENSUS_ID)
        self.assertEqual(census.total_records, 3)
        self.assertEqual(census.total_motifs, 5)
        self.assertEqual(census.max_identity_span_length, 5)
        self.assertEqual(
            census.level1_atoms_per_motif_histogram,
            ((1, 1), (2, 3), (3, 1)),
        )
        self.assertEqual(
            census.level2_atoms_per_motif_histogram,
            ((0, 1), (1, 1), (2, 3)),
        )
        self.assertEqual(
            census.jointly_eligible_atoms_per_motif_histogram,
            ((0, 1), (1, 1), (2, 3)),
        )
        self.assertEqual(census.state_targetable_records, 3)
        self.assertEqual(census.state_targetable_motifs, 4)
        self.assertEqual(census.state_eligible_records, 2)
        self.assertEqual(census.state_eligible_motifs, 3)
        self.assertEqual(census.state_eligible_atoms, 6)
        self.assertEqual(reader.train_epochs, [0])

        membership = census.state_eligible_membership
        self.assertEqual(tuple(row.record_id for row in membership), ("a", "c"))
        self.assertEqual(tuple(row.split_index for row in membership), (0, 2))
        self.assertEqual(membership[0].eligible_motif_ids, (0,))
        self.assertEqual(membership[1].eligible_motif_ids, (0, 1))
        self.assertEqual(
            membership[1].eligible_motifs[1].eligible_atom_indices,
            (2, 3),
        )

    def test_eligible_sampler_replays_the_stable_reader_subsequence(self):
        reader = _Reader((self.a, self.b, self.c))
        census = census_factorized_support(reader, split="train", batch_size=3)
        batches = tuple(
            iter_state_eligible_batches(reader, census, batch_size=1)
        )
        self.assertEqual(tuple(len(batch) for batch in batches), (1, 1))
        sampled = tuple(row for batch in batches for row in batch)
        self.assertEqual(
            tuple(row.paired_record.motif_record.record_id for row in sampled),
            ("a", "c"),
        )
        self.assertEqual(
            tuple(row.membership.eligible_motif_ids for row in sampled),
            ((0,), (0, 1)),
        )
        self.assertEqual(reader.train_epochs, [0, 0])

    def test_sampler_rejects_reader_order_drift(self):
        source = _Reader((self.a, self.b, self.c))
        census = census_factorized_support(source, split="train", batch_size=2)
        reordered = _Reader((self.c, self.b, self.a))
        with self.assertRaisesRegex(
            FactorizedSupportCensusError,
            "order changed",
        ):
            tuple(iter_state_eligible_batches(reordered, census, batch_size=2))

    def test_dev_split_uses_fixed_dev_order_and_indices(self):
        reader = _Reader((self.a,), (self.b, self.c))
        census = census_factorized_support(reader, split="dev", batch_size=1)
        self.assertEqual(census.total_records, 2)
        self.assertEqual(census.state_eligible_records, 1)
        self.assertEqual(census.state_eligible_membership[0].record_id, "c")
        self.assertEqual(census.state_eligible_membership[0].split_index, 1)
        sampled = tuple(
            row
            for batch in iter_state_eligible_batches(reader, census, batch_size=2)
            for row in batch
        )
        self.assertEqual(
            tuple(row.paired_record.motif_record.record_id for row in sampled),
            ("c",),
        )
        self.assertEqual(reader.train_epochs, [])


if __name__ == "__main__":
    unittest.main()
