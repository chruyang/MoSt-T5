from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock
import unittest
import uuid

import torch

import most_t5_next.p2.run_pf10_factorized_state_v1 as state_runner
from most_t5_next.p2.run_pf10_factorized_state_v1 import (
    PF10StateTrainingError,
    _EligibleReader,
    run_state_cell,
)
from most_t5_next.p2.tests import test_run_pf10_factorized_smoke_v1 as smoke_fixture


@dataclass(frozen=True)
class _Loaded:
    motif_record: object


class _SourceReader:
    def __init__(self, train, dev) -> None:
        self.train = tuple(_Loaded(row) for row in train)
        self.dev = tuple(_Loaded(row) for row in dev)
        self.train_member_count = len(self.train)
        self.dev_member_count = len(self.dev)

    def iter_selected_split_indices(self, *, split, split_indices, batch_size):
        source = self.train if split == "train" else self.dev
        rows = tuple(source[index] for index in split_indices)
        for start in range(0, len(rows), batch_size):
            yield rows[start : start + batch_size]


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for cell in (root / "B2D", root):
        for update in (1, 2):
            checkpoint = cell / f"step-{update:04d}"
            file = checkpoint / "training_state.pt"
            if file.is_file():
                file.unlink()
            if checkpoint.is_dir():
                checkpoint.rmdir()
        for filename in ("state_training_manifest.json", "state_pair_manifest.json"):
            file = cell / filename
            if file.is_file():
                file.unlink()
    if (root / "B2D").is_dir():
        (root / "B2D").rmdir()
    if root.is_dir():
        root.rmdir()


class PF10FormalStateRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        smoke_fixture.PF10FactorizedSmokeRunnerTest.setUpClass()
        cls.records = smoke_fixture.PF10FactorizedSmokeRunnerTest.records[:6]
        cls.tokenizer = smoke_fixture.PF10FactorizedSmokeRunnerTest.tokenizer

    def test_eligible_reader_keeps_order_and_short_tail(self) -> None:
        source = _SourceReader(self.records, self.records[:4])
        reader = _EligibleReader(
            source, train_indices=(0, 2, 3, 5), dev_indices=(1, 3)
        )
        batches = tuple(reader.iter_train_epoch(epoch=7, batch_size=3))
        self.assertEqual([len(batch) for batch in batches], [3, 1])
        self.assertEqual(
            [row.motif_record.record_id for batch in batches for row in batch],
            [self.records[index].record_id for index in (0, 2, 3, 5)],
        )
        with self.assertRaises(PF10StateTrainingError):
            _EligibleReader(source, train_indices=(2, 1), dev_indices=(0,))

    def test_two_update_b2d_stage_freezes_t5_and_writes_resume_state(self) -> None:
        source = _SourceReader(self.records, self.records[:4])
        reader = _EligibleReader(
            source,
            train_indices=tuple(range(6)),
            dev_indices=tuple(range(4)),
        )
        torch.manual_seed(17)
        model = smoke_fixture.PF10FactorizedSmokeRunnerTest.factorized_from_state()
        t5_before = {
            name: value.detach().clone() for name, value in model.t5.state_dict().items()
        }
        adapter_before = {
            name: value.detach().clone()
            for name, value in model.adapter.state_dict().items()
        }
        protocol = replace(
            state_runner.S_PROTOCOL,
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        root = Path.cwd() / ("pf10_state_test_" + uuid.uuid4().hex)
        try:
            with mock.patch.object(state_runner, "S_PROTOCOL", protocol), mock.patch.object(
                state_runner, "EVALUATION_UPDATES", (0, 1, 2)
            ), mock.patch.object(state_runner, "CHECKPOINT_UPDATES", (1, 2)):
                report = run_state_cell(
                    cell="B2D",
                    reader=reader,
                    tokenizer=self.tokenizer,
                    model=model,
                    output_dir=root / "B2D",
                    provider=smoke_fixture._B2DProvider(),
                    device=torch.device("cpu"),
                    use_bf16=False,
                )
            self.assertEqual(report["optimizer_updates"], 2)
            self.assertEqual([row["update"] for row in report["evaluations"]], [0, 1, 2])
            self.assertTrue((root / "B2D" / "step-0002" / "training_state.pt").is_file())
            for name, value in model.t5.state_dict().items():
                torch.testing.assert_close(value, t5_before[name])
            self.assertTrue(
                any(
                    not torch.equal(value, adapter_before[name])
                    for name, value in model.adapter.state_dict().items()
                )
            )
        finally:
            _remove_tree(root)


if __name__ == "__main__":
    unittest.main()
