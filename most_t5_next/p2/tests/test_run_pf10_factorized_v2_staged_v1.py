from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock
import unittest
import uuid

import torch

import most_t5_next.p2.run_pf10_factorized_v2_staged_v1 as runner
from most_t5_next.p2.factorized_motif_t5_v2 import FactorizedMotifT5V2
from most_t5_next.p2.run_pf10_factorized_state_v1 import _EligibleReader
from most_t5_next.p2.tests import test_run_pf10_factorized_smoke_v1 as fixture


@dataclass(frozen=True)
class _Loaded:
    motif_record: object


class _Reader:
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


class _Addresses:
    record_count = 6

    def get(self, record):
        return tuple(range(len(record.atom_valid_mask)))


class _MatchedShuffle:
    state_kind = "matched_motif_e3fp"

    def __init__(self, records) -> None:
        self._records = {record.record_id: record for record in records}

    def get(self, record_id):
        record = self._records[record_id]
        return tuple(
            tuple(-1 if value < 0 else (value + 97) % 4096 for value in row)
            for row in record.full_e3fp_ids
        )

    def changed_motif_indices(self, record_id):
        return tuple(range(len(self._records[record_id].identity_spans)))


class _AliasedDecoder(torch.nn.Module):
    def __init__(self, embedding, projection) -> None:
        super().__init__()
        self.embed_tokens = embedding
        self.projection = projection

    def forward(self, hidden):
        return self.projection(hidden)


def _cleanup(root: Path) -> None:
    for stage in (root / "S", root / "S-resume", root / "B"):
        for update in (1, 2):
            checkpoint = stage / f"step-{update:05d}"
            payload = checkpoint / "training_state.pt"
            if payload.is_file():
                payload.unlink()
            if checkpoint.is_dir():
                checkpoint.rmdir()
        for filename in ("s_training_manifest.json", "b_training_manifest.json"):
            path = stage / filename
            if path.is_file():
                path.unlink()
    for stage in (root / "B", root / "S-resume", root / "S"):
        if stage.is_dir():
            stage.rmdir()
    if root.is_dir():
        root.rmdir()


class PF10V2StagedRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture.PF10FactorizedSmokeRunnerTest.setUpClass()
        cls.records = fixture.PF10FactorizedSmokeRunnerTest.records[:6]
        cls.tokenizer = fixture.PF10FactorizedSmokeRunnerTest.tokenizer
        cls.reader = _Reader(cls.records)
        cls.eligible = _EligibleReader(
            cls.reader,
            train_indices=tuple(range(6)),
            dev_indices=tuple(range(4)),
        )
        cls.addresses = _Addresses()

    @staticmethod
    def model(state=None):
        t5 = fixture._TinyT5()
        t5.decoder = _AliasedDecoder(t5.shared, t5.decoder)
        model = FactorizedMotifT5V2(
            t5,
            num_e3fp_embeddings=4096,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
        )
        if state is not None:
            model.load_state_dict(state, strict=True)
        return model

    def test_s_resume_is_exact_and_bridge_freezes_carrier_path(self) -> None:
        s_protocol = replace(
            runner.S_PROTOCOL,
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        b_protocol = replace(
            runner.B_PROTOCOL,
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        torch.manual_seed(71)
        initial = self.model()
        initial_state = {
            name: value.detach().clone() for name, value in initial.state_dict().items()
        }
        root = Path.cwd() / ("pf10_v2_staged_test_" + uuid.uuid4().hex)
        patches = (
            mock.patch.object(runner, "S_PROTOCOL", s_protocol),
            mock.patch.object(runner, "B_PROTOCOL", b_protocol),
            mock.patch.object(runner, "S_EVALUATIONS", (0, 1, 2)),
            mock.patch.object(runner, "S_CHECKPOINTS", (1, 2)),
            mock.patch.object(runner, "B_EVALUATIONS", (0, 1, 2)),
            mock.patch.object(runner, "B_CHECKPOINTS", (1, 2)),
        )
        try:
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                s_model = self.model(initial_state)
                s_report = runner.run_stage(
                    stage="S", cell="F3D", full_reader=self.reader,
                    eligible_reader=self.eligible, tokenizer=self.tokenizer,
                    model=s_model, address_provider=self.addresses,
                    output_dir=root / "S", state_provider=None,
                    device=torch.device("cpu"), use_bf16=False,
                )
                self.assertEqual(s_report["trainable_modules"], ["adapter"])
                self.assertIn("state_zero_minus_aligned", s_report["evaluations"][-1])
                self.assertNotIn("state_zero", s_report["evaluations"][1])

                resumed = self.model(initial_state)
                resumed_report = runner.run_stage(
                    stage="S", cell="F3D", full_reader=self.reader,
                    eligible_reader=self.eligible, tokenizer=self.tokenizer,
                    model=resumed, address_provider=self.addresses,
                    output_dir=root / "S-resume", state_provider=None,
                    device=torch.device("cpu"), use_bf16=False,
                    resume_checkpoint=root / "S" / "step-00001",
                )
                self.assertEqual(resumed_report["optimizer_updates"], 2)
                for name, value in s_model.state_dict().items():
                    torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)

                before_bridge = {
                    name: value.detach().clone() for name, value in s_model.state_dict().items()
                }
                bridge = self.model(initial_state)
                b_report = runner.run_stage(
                    stage="B", cell="F3D", full_reader=self.reader,
                    eligible_reader=self.eligible, tokenizer=self.tokenizer,
                    model=bridge, address_provider=self.addresses,
                    output_dir=root / "B", state_provider=None,
                    device=torch.device("cpu"), use_bf16=False,
                    s_checkpoint=root / "S" / "step-00002",
                )
                self.assertEqual(
                    b_report["trainable_modules"],
                    ["t5.decoder_nonembedding", "t5.lm_head"],
                )
                self.assertIn("identity_zero_minus_aligned", b_report["evaluations"][-1])
                self.assertNotIn("identity_zero", b_report["evaluations"][1])
                for name, value in bridge.state_dict().items():
                    if (
                        name.startswith("adapter.")
                        or name.startswith("t5.encoder.")
                        or name == "t5.shared.weight"
                        or name == "t5.decoder.embed_tokens.weight"
                    ):
                        torch.testing.assert_close(value, before_bridge[name], rtol=0, atol=0)
                self.assertTrue(any(
                    not torch.equal(value, before_bridge[name])
                    for name, value in bridge.state_dict().items()
                    if name.startswith("t5.decoder.")
                ))
        finally:
            _cleanup(root)

    def test_formal_protocol_and_both_stage_cli_are_explicit(self) -> None:
        self.assertEqual(runner.S_PROTOCOL.micro_batch_size, 64)
        self.assertEqual(runner.S_PROTOCOL.gradient_accumulation_steps, 2)
        self.assertEqual(runner.S_PROTOCOL.total_updates, 2500)
        self.assertEqual(runner.B_PROTOCOL.micro_batch_size, 64)
        self.assertEqual(runner.B_PROTOCOL.gradient_accumulation_steps, 2)
        self.assertEqual(runner.B_PROTOCOL.total_updates, 10000)
        parsed = runner._parser().parse_args([
            "--stage", "both", "--cell", "F3D",
            "--paired-release", "paired", "--support-census", "support",
            "--shuffle-overlay", "matched", "--base-model-snapshot", "base",
            "--base-tokenizer-snapshot", "base", "--union-init-dir", "init",
            "--output-dir", "output",
        ])
        self.assertEqual(parsed.stage, "both")

    def test_counterfactual_bridge_uses_changed_masked_identity_tokens(self) -> None:
        s_protocol = replace(
            runner.S_PROTOCOL,
            total_updates=1,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        counterfactual_protocol = replace(
            runner.COUNTERFACTUAL_B_PROTOCOL,
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        torch.manual_seed(83)
        model = self.model()
        root = Path.cwd() / ("pf10_v2_staged_test_" + uuid.uuid4().hex)
        shuffle = _MatchedShuffle(self.records)
        patches = (
            mock.patch.object(runner, "S_PROTOCOL", s_protocol),
            mock.patch.object(runner, "S_EVALUATIONS", (0, 1)),
            mock.patch.object(runner, "S_CHECKPOINTS", (1,)),
            mock.patch.object(
                runner, "COUNTERFACTUAL_B_PROTOCOL", counterfactual_protocol
            ),
            mock.patch.object(
                runner, "COUNTERFACTUAL_B_EVALUATIONS", (0, 1, 2)
            ),
            mock.patch.object(
                runner, "COUNTERFACTUAL_B_CHECKPOINTS", (1, 2)
            ),
        )
        try:
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                runner.run_stage(
                    stage="S", cell="F3D", full_reader=self.reader,
                    eligible_reader=self.eligible, tokenizer=self.tokenizer,
                    model=model, address_provider=self.addresses,
                    output_dir=root / "S", state_provider=None,
                    device=torch.device("cpu"), use_bf16=False,
                )
                report = runner.run_stage(
                    stage="B", cell="F3D", full_reader=self.reader,
                    eligible_reader=self.eligible, tokenizer=self.tokenizer,
                    model=model, address_provider=self.addresses,
                    output_dir=root / "B", state_provider=None,
                    device=torch.device("cpu"), use_bf16=False,
                    s_checkpoint=root / "S" / "step-00001",
                    shuffle_provider=shuffle,
                    train_shuffle_provider=shuffle,
                    counterfactual_weight=1.0,
                    counterfactual_margin=0.05,
                )
            contract = report["counterfactual_bridge"]
            self.assertTrue(contract["enabled"])
            self.assertGreater(contract["pair_tokens"], 0)
            self.assertEqual(contract["unique_members_per_update"], 4)
            self.assertEqual(contract["paired_forward_rows_per_update"], 8)
            self.assertEqual(
                report["objective"],
                "cross_view_identity_with_matched_ranking",
            )
            self.assertIn(
                "identity_shuffle_minus_aligned", report["evaluations"][-1]
            )
        finally:
            _cleanup(root)


if __name__ == "__main__":
    unittest.main()
