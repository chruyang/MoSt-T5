from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import mock
import unittest
import uuid

import torch

import most_t5_next.p2.run_pf10_factorized_grammar_v1 as grammar_runner
import most_t5_next.p2.run_pf10_factorized_state_v1 as state_runner
from most_t5_next.p2.run_pf10_factorized_grammar_v1 import run_grammar_cell
from most_t5_next.p2.run_pf10_factorized_state_v1 import _EligibleReader, run_state_cell
from most_t5_next.p2.tests import test_run_pf10_factorized_smoke_v1 as smoke_fixture
from most_t5_next.p2.tests.test_run_pf10_factorized_state_v1 import _Loaded


class _FullReader:
    def __init__(self, records) -> None:
        self.train = tuple(_Loaded(row) for row in records)
        self.dev = tuple(_Loaded(row) for row in records[:4])
        self.train_member_count = len(self.train)
        self.dev_member_count = len(self.dev)

    @staticmethod
    def _batches(rows, batch_size):
        for start in range(0, len(rows), batch_size):
            yield rows[start : start + batch_size]

    def iter_train_epoch(self, *, epoch, batch_size):
        if epoch < 0:
            raise ValueError
        yield from self._batches(self.train, batch_size)

    def iter_dev(self, *, batch_size):
        yield from self._batches(self.dev, batch_size)

    def iter_selected_split_indices(self, *, split, split_indices, batch_size):
        source = self.train if split == "train" else self.dev
        rows = tuple(source[index] for index in split_indices)
        yield from self._batches(rows, batch_size)


def _cleanup(root: Path) -> None:
    for stage in (root / "S", root / "G"):
        for name in ("step-0001", "step-0002", "step-00001", "step-00002"):
            directory = stage / name
            file = directory / "training_state.pt"
            if file.is_file():
                file.unlink()
            if directory.is_dir():
                directory.rmdir()
        for filename in ("state_training_manifest.json", "grammar_training_manifest.json"):
            file = stage / filename
            if file.is_file():
                file.unlink()
    for stage in (root / "G", root / "S"):
        if stage.is_dir():
            stage.rmdir()
    if root.is_dir():
        root.rmdir()


class PF10FormalGrammarRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        smoke_fixture.PF10FactorizedSmokeRunnerTest.setUpClass()
        cls.records = smoke_fixture.PF10FactorizedSmokeRunnerTest.records[:6]
        cls.tokenizer = smoke_fixture.PF10FactorizedSmokeRunnerTest.tokenizer

    def test_f3d_loads_formal_s_and_reports_zero_and_shuffle(self) -> None:
        reader = _FullReader(self.records)
        eligible = _EligibleReader(
            reader,
            train_indices=tuple(range(6)),
            dev_indices=tuple(range(4)),
        )
        s_protocol = replace(
            state_runner.S_PROTOCOL,
            total_updates=1,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        g_protocol = replace(
            grammar_runner.G_PROTOCOL,
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        torch.manual_seed(23)
        initial = smoke_fixture.PF10FactorizedSmokeRunnerTest.factorized_from_state()
        initial_state = {
            name: value.detach().clone() for name, value in initial.state_dict().items()
        }
        root = Path.cwd() / ("pf10_grammar_test_" + uuid.uuid4().hex)
        try:
            s_model = smoke_fixture.PF10FactorizedSmokeRunnerTest.factorized_from_state(
                initial_state
            )
            with mock.patch.object(state_runner, "S_PROTOCOL", s_protocol), mock.patch.object(
                state_runner, "EVALUATION_UPDATES", (0, 1)
            ), mock.patch.object(state_runner, "CHECKPOINT_UPDATES", (1,)):
                run_state_cell(
                    cell="F3D",
                    reader=eligible,
                    tokenizer=self.tokenizer,
                    model=s_model,
                    output_dir=root / "S",
                    provider=None,
                    device=torch.device("cpu"),
                    use_bf16=False,
                )

            g_model = smoke_fixture.PF10FactorizedSmokeRunnerTest.factorized_from_state(
                initial_state
            )
            with mock.patch.object(grammar_runner, "S_PROTOCOL", s_protocol), mock.patch.object(
                grammar_runner, "G_PROTOCOL", g_protocol
            ), mock.patch.object(grammar_runner, "EVALUATION_UPDATES", (0, 1, 2)), mock.patch.object(
                grammar_runner, "CHECKPOINT_UPDATES", (1, 2)
            ):
                report = run_grammar_cell(
                    cell="F3D",
                    reader=reader,
                    tokenizer=self.tokenizer,
                    model=g_model,
                    output_dir=root / "G",
                    provider=None,
                    device=torch.device("cpu"),
                    use_bf16=False,
                    s_checkpoint=root / "S" / "step-0001",
                    shuffle_provider=smoke_fixture._B2DProvider(),
                )
            self.assertEqual(report["optimizer_updates"], 2)
            self.assertEqual([row["update"] for row in report["evaluations"]], [0, 1, 2])
            diagnostic = report["f3d_state_diagnostics"]
            self.assertIsNotNone(diagnostic)
            self.assertEqual(diagnostic["zero"]["state_memory_mode"], "zero")
            self.assertEqual(diagnostic["matched_shuffle"]["state_kind"], "mock_morgan_r3_4096")
            self.assertEqual(len(report["checkpoints"]), 2)
            self.assertTrue((root / "G" / "step-00002" / "training_state.pt").is_file())
        finally:
            _cleanup(root)


if __name__ == "__main__":
    unittest.main()
