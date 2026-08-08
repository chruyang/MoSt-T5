"""Hermetic contracts for the complete streaming PF-1 collator gate."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace
import unittest

from most_t5_next.p1.atom_production_bridge import (
    ATOM_IDENTITY_ROLE,
    ProductionAtomSelfiesRecord,
    collate_production_atom_batch,
)
from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    collate_production_batch,
)
from most_t5_next.p1 import validate_pf1_full_collator_gate_v1 as subject
from most_t5_next.r1.adapter.paired_record_wire_v1 import (
    LoadedPairedTrainingRecord,
    PairedSurfaceSummary,
)
from most_t5_next.r1.adapter.production_paired_identity_records_v1 import (
    ProductionPairReceipt,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


TOKENIZER = ProductionTokenizerRuntime(
    tokenizer_contract_sha256=digest("contract"),
    tokenizer_snapshot_sha256=digest("snapshot"),
    vocab_size=1000,
    pad_token_id=999,
    eos_token_id=998,
    sentinel_token_ids=(900, 901, 902),
)


def paired_row(schedule_index: int) -> LoadedPairedTrainingRecord:
    record_id = "pf1-fixture:{:03d}".format(schedule_index)
    storage_key = "{:09d}".format(schedule_index)
    release_id = "pf1-fixture-release"
    geometry = digest("effective-geometry")
    e3fp = ((10, 11, -1, -1), (20, 21, -1, -1))
    atom = ProductionAtomSelfiesRecord(
        record_artifact_sha256=digest("atom-artifact"),
        record_id=record_id,
        storage_key=storage_key,
        release_id=release_id,
        geometry_record_content_sha256=geometry,
        union_tokenizer_contract_sha256=TOKENIZER.tokenizer_contract_sha256,
        union_tokenizer_snapshot_sha256=TOKENIZER.tokenizer_snapshot_sha256,
        selfies="[C][O]",
        input_ids=(1, 10, 11, 12, 20, 2),
        token_to_atom=(-1, 0, 0, -1, 1, -1),
        token_role=(
            "boundary",
            ATOM_IDENTITY_ROLE,
            ATOM_IDENTITY_ROLE,
            "branch",
            ATOM_IDENTITY_ROLE,
            "boundary",
        ),
        atom_identity_spans=(Span(1, 3), Span(4, 5)),
        atom_to_carrier=(1, 4),
        source_atom_count=2,
        model_to_source_atom_index=(0, 1),
        full_e3fp_ids=e3fp,
        atom_valid_mask=(True, True),
    )
    motif = ProductionMotifRecord(
        record_artifact_sha256=digest("motif-artifact"),
        record_id=record_id,
        storage_key=storage_key,
        release_id=release_id,
        geometry_record_content_sha256=geometry,
        tokenizer_contract_sha256=TOKENIZER.tokenizer_contract_sha256,
        tokenizer_snapshot_sha256=TOKENIZER.tokenizer_snapshot_sha256,
        input_ids=(1, 30, 31, 2),
        token_to_logical_motif=(-1, 0, 0, -1),
        token_role=("boundary", "identity", "identity", "boundary"),
        identity_spans=(Span(1, 3),),
        connection_token_indices=((),),
        logical_to_carrier=(1,),
        exact_identity_sha256=(digest("motif-identity"),),
        source_atom_count=2,
        full_e3fp_ids=e3fp,
        atom_valid_mask=(True, True),
        model_to_source_atom_index=(0, 1),
        atom_to_logical_motif=(0, 0),
        atom_is_attachment=(False, True),
    )
    receipt = ProductionPairReceipt(
        member_id=record_id,
        storage_key=storage_key,
        release_id=release_id,
        base_geometry_record_content_sha256=digest("base-geometry"),
        effective_inherited_overlay_content_sha256=geometry,
        strict_isomeric_identity="CO",
    )
    summary = PairedSurfaceSummary(
        atom_input_token_count=len(atom.input_ids),
        motif_input_token_count=len(motif.input_ids),
        motif_identity_modes=("fallback",),
        motif_identity_token_counts=(2,),
        graph_token_count=0,
        cross_motif_connection_count=0,
    )
    return LoadedPairedTrainingRecord(
        schedule_index=schedule_index,
        sdf_record_index=100 + schedule_index,
        atom_record=atom,
        motif_record=motif,
        receipt=receipt,
        surface_summary=summary,
    )


class StreamingReader:
    def __init__(self, train, dev, *, advertised_train: int | None = None) -> None:
        self.train = tuple(train)
        self.dev = tuple(dev)
        self.train_member_count = (
            len(self.train) if advertised_train is None else advertised_train
        )
        self.dev_member_count = len(self.dev)
        self.train_calls = 0
        self.dev_calls = 0

    def iter_train_epoch(self, *, epoch: int, batch_size: int):
        self.train_calls += 1
        assert epoch == subject.TRAIN_GATE_EPOCH
        for start in range(0, len(self.train), batch_size):
            yield self.train[start : start + batch_size]

    def iter_dev(self, *, batch_size: int):
        self.dev_calls += 1
        for start in range(0, len(self.dev), batch_size):
            yield self.dev[start : start + batch_size]


def condition_collator(
    records, *, condition_id, tokenizer_runtime, seed, epoch
):
    rows = tuple(records)
    if condition_id.startswith("A"):
        return collate_production_atom_batch(
            tuple(row.atom_record for row in rows),
            condition_id=condition_id,
            tokenizer=tokenizer_runtime,
            seed=seed,
            epoch=epoch,
        )
    return collate_production_batch(
        tuple(row.motif_record for row in rows),
        condition_id=condition_id,
        tokenizer=tokenizer_runtime,
        seed=seed,
        epoch=epoch,
    )


def nested_shape(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child = nested_shape(value[0])
    assert all(nested_shape(item) == child for item in value)
    return (len(value),) + child


class FakeTensor:
    def __init__(self, value, dtype) -> None:
        self.shape = nested_shape(value)
        self.dtype = dtype


class FakeTorch:
    long = "int64"
    bool = "bool"

    @staticmethod
    def as_tensor(value, *, dtype, device=None):
        del device
        return FakeTensor(value, dtype)


class PF1FullCollatorGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = tuple(paired_row(index) for index in range(4))
        self.dev = tuple(paired_row(10 + index) for index in range(3))

    def validate(self, reader):
        return subject.validate_full_collator_gate(
            reader=reader,
            tokenizer_runtime=TOKENIZER,
            batch_size=2,
            torch_module=FakeTorch,
            condition_collator=condition_collator,
        )

    def test_streams_each_split_once_and_reports_complete_four_grid_gate(self) -> None:
        reader = StreamingReader(self.train, self.dev)
        report = self.validate(reader)

        self.assertEqual((reader.train_calls, reader.dev_calls), (1, 1))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["members"], {
            "train": 4,
            "dev": 3,
            "total": 7,
            "rejected": 0,
        })
        self.assertEqual(report["splits"]["train"]["batch_count"], 2)
        self.assertEqual(report["splits"]["dev"]["batch_count"], 2)
        for condition_id in subject.CONDITION_ORDER:
            condition = report["conditions"][condition_id]
            self.assertEqual(condition["member_count"]["total"], 7)
            self.assertEqual(condition["batch_count"]["total"], 4)
            self.assertEqual(
                condition["selected_mask_coverage"]["total"]["member_coverage"],
                1.0,
            )
            self.assertEqual(
                condition["tensor_contract"]["verified_batch_count"], 4
            )
            self.assertEqual(
                condition["tensor_contract"]["dtypes"]["attention_mask"],
                "int64",
            )
        self.assertEqual(
            report["conditions"]["A0"]["selected_mask_coverage"]["total"][
                "eligible_mask_units"
            ],
            14,
        )
        self.assertEqual(
            report["conditions"]["M1"]["selected_mask_coverage"]["total"][
                "eligible_atoms"
            ],
            14,
        )
        self.assertEqual(
            report["conditions"]["A1"]["tensor_contract"]["ranks"],
            {
                "input_ids": 2,
                "attention_mask": 2,
                "labels": 2,
                "e3fp_ids": 3,
                "e3fp_atom_mask": 2,
                "e3fp_atom_to_token": 2,
            },
        )
        self.assertEqual(
            report["conditions"]["A1"]["tensor_contract"]["dtypes"][
                "e3fp_atom_mask"
            ],
            "bool",
        )
        self.assertEqual(
            report["conditions"]["A1"]["maxima"],
            {
                "batch_members": 2,
                "input_tokens": 6,
                "target_tokens": 5,
                "atom_rows": 2,
                "selected_mask_units_per_member": 1,
                "sentinel_tokens_per_target": 2,
            },
        )
        self.assertEqual(report["rejects"], {"member_count": 0, "batch_count": 0})
        self.assertFalse(report["schedule"]["record_sampling"])
        self.assertFalse(report["schedule"]["sequence_truncation"])
        self.assertEqual(report["schedule"]["maximum_decoded_records_resident"], 2)

    def test_advertised_membership_cannot_be_silently_truncated(self) -> None:
        reader = StreamingReader(self.train, self.dev, advertised_train=5)
        with self.assertRaisesRegex(subject.PF1FullCollatorGateError, "ended at"):
            self.validate(reader)

    def test_raw_A_M_geometry_difference_rejects_before_collation(self) -> None:
        original = self.train[0]
        broken_motif = replace(
            original.motif_record,
            full_e3fp_ids=((10, 11, -1, -1), (20, 22, -1, -1)),
        )
        broken = SimpleNamespace(
            schedule_index=original.schedule_index,
            sdf_record_index=original.sdf_record_index,
            atom_record=original.atom_record,
            motif_record=broken_motif,
            receipt=original.receipt,
            surface_summary=original.surface_summary,
        )
        reader = StreamingReader((broken, *self.train[1:]), self.dev)
        with self.assertRaisesRegex(
            subject.PF1FullCollatorGateError, "geometry/source mapping differs"
        ):
            self.validate(reader)

    def test_cli_deliberately_has_no_partial_record_option(self) -> None:
        parser = subject.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--paired-release",
                    "release",
                    "--base-tokenizer-snapshot",
                    "base",
                    "--output-report",
                    "report.json",
                    "--max-records",
                    "1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
