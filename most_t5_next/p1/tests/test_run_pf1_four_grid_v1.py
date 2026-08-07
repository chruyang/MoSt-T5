"""CPU mock tests for the thin PF-1 four-grid runner."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from most_t5_next.p1.experiment_grid import GeometryBatchSidecar, P1ConditionBatch
from most_t5_next.p1.pf1_optimization import PF1OptimizationProtocol
from most_t5_next.p1.runtime_bridge import PaddedCEBatch
from most_t5_next.p1 import run_pf1_four_grid_v1 as subject


class FakeReader:
    train_member_count = 4
    dev_member_count = 2

    def iter_train_epoch(self, *, epoch: int, batch_size: int):
        self.last_train_batch_size = batch_size
        offset = epoch * 4
        yield tuple(range(offset, offset + batch_size))
        yield tuple(range(offset + batch_size, offset + 2 * batch_size))

    def iter_dev(self, *, batch_size: int):
        self.last_dev_batch_size = batch_size
        yield (20, 21)


@dataclass
class FakeCEBatch:
    record_ids: tuple[str, ...]
    input_lengths: tuple[int, ...]
    target_lengths: tuple[int, ...]
    values: tuple[int, ...]


@dataclass
class FakeConditionBatch:
    condition_id: str
    ce_batch: FakeCEBatch
    geometry: object | None = None


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.projection = torch.nn.Linear(1, 2)

    def forward(self, *, values, labels, use_cache, return_dict):
        del use_cache, return_dict
        logits = self.projection(values).unsqueeze(1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2), labels.reshape(-1)
        )
        return SimpleNamespace(loss=loss, logits=logits)


class _FakeCudaRuntime:
    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self.available = available
        self.bf16 = bf16

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16


class _FakeTorchRuntime:
    def __init__(self, *, available: bool = True, bf16: bool = True) -> None:
        self.cuda = _FakeCudaRuntime(available=available, bf16=bf16)

    @staticmethod
    def device(device_type: str, index: int) -> SimpleNamespace:
        return SimpleNamespace(type=device_type, index=index)


def _parse_cli(output_dir: Path, *extra: str):
    return subject.build_parser().parse_args(
        [
            "--paired-release",
            str(output_dir.parent / "paired"),
            "--base-model-snapshot",
            str(output_dir.parent / "base-model"),
            "--base-tokenizer-snapshot",
            str(output_dir.parent / "base-tokenizer"),
            "--union-init-dir",
            str(output_dir.parent / "union-init"),
            "--output-dir",
            str(output_dir),
            "--geometry-fusion-seed",
            "20260808",
            *extra,
        ]
    )


def _geometry_record(index: int, atom_count: int, *, base: int = 0):
    return SimpleNamespace(
        schedule_index=index,
        atom_record=SimpleNamespace(
            record_id=f"member-{index}",
            full_e3fp_ids=tuple(
                (base + index * 10 + atom, atom + 1, atom + 2, atom + 3)
                for atom in range(atom_count)
            ),
        ),
    )


def _geometry_batch(records, *, condition_id: str = "A1") -> P1ConditionBatch:
    rows = tuple(records)
    input_lengths = tuple(3 for _row in rows)
    target_lengths = tuple(2 if row.schedule_index == 3 else 1 for row in rows)
    max_target = max(target_lengths)
    ce_batch = PaddedCEBatch(
        record_ids=tuple(row.atom_record.record_id for row in rows),
        input_ids=tuple((7, 8, 9) for _row in rows),
        attention_mask=tuple((True, True, True) for _row in rows),
        labels=tuple(
            (0,) * length + (-100,) * (max_target - length)
            for length in target_lengths
        ),
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )
    atom_lengths = tuple(len(row.atom_record.full_e3fp_ids) for row in rows)
    atom_width = max(atom_lengths)
    level_count = 4
    geometry = GeometryBatchSidecar(
        record_ids=ce_batch.record_ids,
        e3fp_ids=tuple(
            row.atom_record.full_e3fp_ids
            + ((-1,) * level_count,) * (atom_width - atom_count)
            for row, atom_count in zip(rows, atom_lengths)
        ),
        e3fp_atom_mask=tuple(
            (True,) * atom_count + (False,) * (atom_width - atom_count)
            for atom_count in atom_lengths
        ),
        e3fp_atom_to_token=tuple(
            tuple(range(atom_count)) + (-1,) * (atom_width - atom_count)
            for atom_count in atom_lengths
        ),
        model_to_source_atom_index=tuple(
            tuple(range(atom_count)) + (-1,) * (atom_width - atom_count)
            for atom_count in atom_lengths
        ),
        atom_lengths=atom_lengths,
        e3fp_level_count=level_count,
        token_width=3,
    )
    return P1ConditionBatch(
        condition_id=condition_id,
        ce_batch=ce_batch,
        geometry=geometry,
    )


class PF1GeometrySensitivityTest(unittest.TestCase):
    def test_derangement_is_stable_bijective_same_count_and_reports_singleton(self):
        records = tuple(
            _geometry_record(index, atom_count)
            for index, atom_count in enumerate((2, 2, 2, 3, 3, 4))
        )
        first = subject.build_pf1_geometry_derangement(records, seed=71)
        second = subject.build_pf1_geometry_derangement(records, seed=71)

        self.assertEqual(first, second)
        self.assertEqual(first.excluded_singleton_indices, (5,))
        self.assertEqual(sorted(first.donor_indices), list(first.eligible_indices))
        for recipient, donor in zip(
            first.eligible_indices, first.donor_indices
        ):
            self.assertNotEqual(recipient, donor)
            self.assertEqual(
                len(records[recipient].atom_record.full_e3fp_ids),
                len(records[donor].atom_record.full_e3fp_ids),
            )

        with self.assertRaisesRegex(subject.PF1TrainingError, "no derangeable"):
            subject.build_pf1_geometry_derangement(
                (_geometry_record(0, 2), _geometry_record(1, 3))
            )

    def test_donor_replacement_preserves_recipient_fields_and_rejects_tampering(self):
        recipients = (_geometry_record(0, 2), _geometry_record(1, 1))
        donors = (
            _geometry_record(10, 2, base=100),
            _geometry_record(11, 1, base=100),
        )
        aligned = _geometry_batch(recipients)
        shuffled = subject._replace_with_donor_e3fp(aligned, donors)

        self.assertIs(shuffled.ce_batch, aligned.ce_batch)
        self.assertIsNotNone(aligned.geometry)
        self.assertIsNotNone(shuffled.geometry)
        for field in (
            "record_ids",
            "e3fp_atom_mask",
            "e3fp_atom_to_token",
            "model_to_source_atom_index",
            "atom_lengths",
            "token_width",
        ):
            self.assertEqual(
                getattr(shuffled.geometry, field),
                getattr(aligned.geometry, field),
            )
        self.assertEqual(
            shuffled.geometry.e3fp_ids[0], donors[0].atom_record.full_e3fp_ids
        )
        self.assertEqual(
            shuffled.geometry.e3fp_ids[1][0],
            donors[1].atom_record.full_e3fp_ids[0],
        )
        self.assertEqual(shuffled.geometry.e3fp_ids[1][1], (-1, -1, -1, -1))

        tampered = (donors[0], _geometry_record(12, 2, base=100))
        with self.assertRaisesRegex(subject.PF1TrainingError, "atom count differs"):
            subject._replace_with_donor_e3fp(aligned, tampered)

    def test_token_weighted_delta_uses_paired_seed_and_shared_recipient_tensors(self):
        records = tuple(_geometry_record(index, 2) for index in range(4))

        class Reader:
            dev_member_count = len(records)

            def iter_dev(self, *, batch_size: int):
                for start in range(0, len(records), batch_size):
                    yield records[start : start + batch_size]

        def encode(batch, *, device):
            assert batch.geometry is not None
            return {
                "input_ids": torch.tensor(batch.ce_batch.input_ids, device=device),
                "attention_mask": torch.tensor(
                    batch.ce_batch.attention_mask, device=device
                ),
                "labels": torch.tensor(batch.ce_batch.labels, device=device),
                "e3fp_ids": torch.tensor(batch.geometry.e3fp_ids, device=device),
                "e3fp_atom_mask": torch.tensor(
                    batch.geometry.e3fp_atom_mask, device=device
                ),
                "e3fp_atom_to_token": torch.tensor(
                    batch.geometry.e3fp_atom_to_token, device=device
                ),
            }

        class PairedLossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.losses = iter((1.0, 3.0, 2.0, 6.0))
                self.calls = []

            def forward(
                self,
                *,
                input_ids,
                attention_mask,
                labels,
                e3fp_ids,
                e3fp_atom_mask,
                e3fp_atom_to_token,
                use_cache,
                return_dict,
            ):
                del input_ids, use_cache, return_dict
                self.calls.append(
                    {
                        "random": float(torch.rand(()).item()),
                        "labels_id": id(labels),
                        "mask_id": id(attention_mask),
                        "atom_mask_id": id(e3fp_atom_mask),
                        "carrier_id": id(e3fp_atom_to_token),
                        "e3fp": e3fp_ids.detach().clone(),
                    }
                )
                logits = torch.zeros(
                    labels.shape[0], labels.shape[1], 2, device=labels.device
                )
                return SimpleNamespace(
                    loss=torch.tensor(next(self.losses), device=labels.device),
                    logits=logits,
                )

        protocol = PF1OptimizationProtocol(
            total_updates=1,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        )
        model = PairedLossModel()
        torch.manual_seed(811)
        rng_before = torch.random.get_rng_state().clone()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    subject,
                    "collate_pf1_condition",
                    side_effect=lambda rows, **kwargs: _geometry_batch(
                        rows, condition_id=kwargs["condition_id"]
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    subject, "to_four_grid_batch_encoding", side_effect=encode
                )
            )
            stack.enter_context(
                patch.object(
                    subject,
                    "select_four_grid_forward_inputs",
                    side_effect=lambda encoded: encoded,
                )
            )
            report = subject.evaluate_pf1_geometry_sensitivity(
                model,
                condition_id="A1",
                reader=Reader(),
                tokenizer_runtime=object(),
                device=torch.device("cpu"),
                use_bf16=False,
                protocol=protocol,
                torch_module=torch,
            )

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(report["dev_members"], 4)
        self.assertEqual(report["eligible_members"], 4)
        self.assertEqual(report["excluded_singletons"], [])
        self.assertAlmostEqual(report["aligned_nll"], 1.6)
        self.assertAlmostEqual(report["shuffled_nll"], 4.8)
        self.assertAlmostEqual(report["delta_nll"], 3.2)
        self.assertTrue(report["no_self_pairing"])
        self.assertEqual(report["atom_count_parity_pairs"], 4)
        self.assertEqual(report["atom_count_mismatches"], 0)
        self.assertEqual(len(model.calls), 4)
        for aligned, shuffled in zip(model.calls[::2], model.calls[1::2]):
            self.assertEqual(aligned["random"], shuffled["random"])
            for field in (
                "labels_id",
                "mask_id",
                "atom_mask_id",
                "carrier_id",
            ):
                self.assertEqual(aligned[field], shuffled[field])
            self.assertFalse(torch.equal(aligned["e3fp"], shuffled["e3fp"]))


class PF1CommandLineTest(unittest.TestCase):
    def test_parser_binds_only_runtime_inputs_and_keeps_protocol_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _parse_cli(Path(temporary) / "output")

        self.assertEqual(args.geometry_fusion_seed, 20260808)
        self.assertEqual(args.num_e3fp_embeddings, 4096)
        self.assertIsNone(args.condition_id)
        self.assertIsNone(args.resume_condition)
        self.assertIsNone(args.resume_checkpoint)
        for forbidden in (
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "total_updates",
        ):
            self.assertFalse(hasattr(args, forbidden), forbidden)

    def test_run_loads_reader_and_verified_runtime_without_vocab_expansion(self) -> None:
        calls: dict[str, object] = {}
        reader = object()
        runtime = SimpleNamespace(vocab_size=34666)

        def reader_factory(path: Path):
            calls["reader_path"] = path
            return reader

        def tokenizer_loader(**kwargs):
            calls["tokenizer"] = kwargs
            return SimpleNamespace(runtime=runtime)

        def executor(**kwargs):
            calls["executor"] = kwargs
            return {"status": "pass"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            args = _parse_cli(output, "--num-e3fp-embeddings", "8192")
            report = subject.run(
                args,
                torch_module=_FakeTorchRuntime(),
                reader_factory=reader_factory,
                tokenizer_loader=tokenizer_loader,
                executor=executor,
            )

            paired = (root / "paired").resolve()
            self.assertEqual(report, {"status": "pass"})
            self.assertEqual(calls["reader_path"], paired)
            self.assertEqual(
                calls["tokenizer"],
                {
                    "base_snapshot": (root / "base-tokenizer").resolve(),
                    "output_dir": paired / subject.TOKENIZER_DIRECTORY,
                },
            )
            bound = calls["executor"]
            self.assertIs(bound["reader"], reader)
            self.assertIs(bound["tokenizer_runtime"], runtime)
            self.assertEqual(bound["expected_vocab_size"], 34666)
            self.assertEqual(bound["num_e3fp_embeddings"], 8192)
            self.assertEqual(bound["output_dir"], output.resolve())
            self.assertEqual(bound["device"].type, "cuda")
            self.assertEqual(bound["device"].index, 0)
            self.assertTrue(bound["use_bf16"])
            self.assertEqual(bound["resume_checkpoints"], {})
            self.assertEqual(bound["condition_ids"], subject.CONDITION_ORDER)
            self.assertNotIn("protocol", bound)
            self.assertFalse(output.exists())
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_cuda_and_bf16_gates_run_before_release_loading(self) -> None:
        def forbidden(*_args, **_kwargs):
            self.fail("release loading must not start before the CUDA/BF16 gate")

        with tempfile.TemporaryDirectory() as temporary:
            args = _parse_cli(Path(temporary) / "output")
            for runtime, message in (
                (_FakeTorchRuntime(available=False), "requires one CUDA GPU"),
                (_FakeTorchRuntime(bf16=False), "requires CUDA BF16 support"),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(subject.PF1TrainingError, message):
                        subject.run(
                            args,
                            torch_module=runtime,
                            reader_factory=forbidden,
                            tokenizer_loader=forbidden,
                            executor=forbidden,
                        )

    def test_existing_output_is_rejected_before_release_loading(self) -> None:
        def forbidden(*_args, **_kwargs):
            self.fail("release loading must not start for an existing output path")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            with self.assertRaisesRegex(subject.PF1TrainingError, "must be a new path"):
                subject.run(
                    _parse_cli(output),
                    torch_module=_FakeTorchRuntime(),
                    reader_factory=forbidden,
                    tokenizer_loader=forbidden,
                    executor=forbidden,
                )

    def test_one_resume_pair_maps_to_exactly_one_condition(self) -> None:
        captured: dict[str, object] = {}

        def executor(**kwargs):
            captured.update(kwargs)
            return {"status": "pass"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            args = _parse_cli(
                root / "output",
                "--resume-condition",
                "M1",
                "--resume-checkpoint",
                str(checkpoint),
            )
            subject.run(
                args,
                torch_module=_FakeTorchRuntime(),
                reader_factory=lambda _path: object(),
                tokenizer_loader=lambda **_kwargs: SimpleNamespace(
                    runtime=SimpleNamespace(vocab_size=41)
                ),
                executor=executor,
            )
            self.assertEqual(
                captured["resume_checkpoints"], {"M1": checkpoint.resolve()}
            )

            incomplete = _parse_cli(
                root / "other-output", "--resume-condition", "A0"
            )
            with self.assertRaisesRegex(subject.PF1TrainingError, "provided together"):
                subject.run(
                    incomplete,
                    torch_module=_FakeTorchRuntime(),
                    reader_factory=lambda _path: object(),
                    tokenizer_loader=lambda **_kwargs: SimpleNamespace(
                        runtime=SimpleNamespace(vocab_size=41)
                    ),
                    executor=executor,
                )

    def test_single_condition_cli_and_resume_must_name_the_same_cell(self) -> None:
        captured: dict[str, object] = {}

        def executor(**kwargs):
            captured.update(kwargs)
            return {"status": "pass"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject.run(
                _parse_cli(root / "m1", "--condition-id", "M1"),
                torch_module=_FakeTorchRuntime(),
                reader_factory=lambda _path: object(),
                tokenizer_loader=lambda **_kwargs: SimpleNamespace(
                    runtime=SimpleNamespace(vocab_size=41)
                ),
                executor=executor,
            )
            self.assertEqual(captured["condition_ids"], ("M1",))

            mismatched = _parse_cli(
                root / "bad",
                "--condition-id",
                "A1",
                "--resume-condition",
                "M1",
                "--resume-checkpoint",
                str(root / "checkpoint"),
            )
            with self.assertRaisesRegex(subject.PF1TrainingError, "must equal"):
                subject.run(
                    mismatched,
                    torch_module=_FakeTorchRuntime(),
                    reader_factory=lambda _path: object(),
                    tokenizer_loader=lambda **_kwargs: SimpleNamespace(
                        runtime=SimpleNamespace(vocab_size=41)
                    ),
                    executor=executor,
                )

    def test_main_serializes_expected_failure_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            argv = _parse_cli(Path(temporary) / "output")
            arguments = [
                "--paired-release",
                argv.paired_release,
                "--base-model-snapshot",
                argv.base_model_snapshot,
                "--base-tokenizer-snapshot",
                argv.base_tokenizer_snapshot,
                "--union-init-dir",
                argv.union_init_dir,
                "--output-dir",
                argv.output_dir,
                "--geometry-fusion-seed",
                str(argv.geometry_fusion_seed),
            ]
            stream = io.StringIO()
            with patch.object(
                subject, "run", side_effect=subject.PF1TrainingError("expected")
            ), redirect_stdout(stream):
                return_code = subject.main(arguments)

        self.assertEqual(return_code, 2)
        self.assertEqual(
            json.loads(stream.getvalue()), {"status": "fail", "error": "expected"}
        )


class PF1RunnerTest(unittest.TestCase):
    def test_frozen_checkpoint_nodes_remain_500_and_1000(self) -> None:
        self.assertEqual(subject.CHECKPOINT_UPDATES, (500, 1000))

    def test_four_cells_share_schedule_and_write_only_two_checkpoints(self) -> None:
        reader = FakeReader()
        collate_calls: list[tuple[str, int, int, tuple[int, ...]]] = []
        checkpoint_calls: list[tuple[str, int]] = []
        checkpoint_diagnostics: list[tuple[str, int, object]] = []
        geometry_diagnostic_calls: list[str] = []

        def collate(records, *, condition_id, tokenizer_runtime, seed, epoch):
            del tokenizer_runtime
            values = tuple(int(value) for value in records)
            collate_calls.append((condition_id, seed, epoch, values))
            return FakeConditionBatch(
                condition_id=condition_id,
                ce_batch=FakeCEBatch(
                    record_ids=tuple(str(value) for value in values),
                    input_lengths=(1,) * len(values),
                    target_lengths=(1,) * len(values),
                    values=values,
                ),
                geometry=object() if condition_id.endswith("1") else None,
            )

        def encode(batch, *, device):
            values = torch.tensor(
                [[[float(value % 3)]] for value in batch.ce_batch.values],
                dtype=torch.float32,
                device=device,
            ).reshape(len(batch.ce_batch.values), 1)
            labels = torch.tensor(
                [[value % 2] for value in batch.ce_batch.values],
                dtype=torch.long,
                device=device,
            )
            return {"values": values, "labels": labels}

        def checkpoint_writer(**kwargs):
            checkpoint_calls.append((kwargs["condition_id"], kwargs["update"]))
            checkpoint_diagnostics.append(
                (
                    kwargs["condition_id"],
                    kwargs["update"],
                    kwargs["training_progress"]["geometry_sensitivity"],
                )
            )
            return f'{kwargs["condition_id"]}/step-{kwargs["update"]}'

        def geometry_diagnostic(_model, **kwargs):
            geometry_diagnostic_calls.append(kwargs["condition_id"])
            return {
                "update": 2,
                "eligible_members": 2,
                "delta_nll": 0.25,
            }

        protocol = PF1OptimizationProtocol(
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pf1"
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(subject, "collate_pf1_condition", side_effect=collate)
                )
                stack.enter_context(
                    patch.object(
                        subject, "to_four_grid_batch_encoding", side_effect=encode
                    )
                )
                stack.enter_context(
                    patch.object(
                        subject,
                        "select_four_grid_forward_inputs",
                        side_effect=lambda encoded: encoded,
                    )
                )
                stack.enter_context(
                    patch.object(subject, "EVALUATION_UPDATES", (0, 1, 2))
                )
                stack.enter_context(
                    patch.object(subject, "CHECKPOINT_UPDATES", (1, 2))
                )
                stack.enter_context(
                    patch.object(subject, "GEOMETRY_PERTURBATION_UPDATE", 2)
                )
                stack.enter_context(
                    patch.object(
                        subject,
                        "evaluate_pf1_geometry_sensitivity",
                        side_effect=geometry_diagnostic,
                    )
                )
                report = subject.execute_pf1_four_grid(
                    reader=reader,
                    tokenizer_runtime=object(),
                    base_model_snapshot=Path("base-model"),
                    base_tokenizer_snapshot=Path("base-tokenizer"),
                    union_tokenizer_dir=Path("union-tokenizer"),
                    union_init_dir=Path("union-init"),
                    geometry_fusion_seed=7,
                    num_e3fp_embeddings=4096,
                    expected_vocab_size=19,
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    use_bf16=False,
                    torch_module=torch,
                    wrapper_loader=lambda **_kwargs: FakeModel(19),
                    checkpoint_writer=checkpoint_writer,
                    protocol=protocol,
                )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["optimization"]["effective_batch_size"], 4)
            self.assertEqual(report["precision"], "test_or_debug_precision")
            self.assertEqual(
                [row["condition"] for row in report["conditions"]],
                list(subject.CONDITION_ORDER),
            )
            for row in report["conditions"]:
                self.assertEqual(row["optimizer_updates"], 2)
                self.assertEqual(row["members_seen"], 8)
                self.assertEqual(
                    [value["update"] for value in row["evaluations"]],
                    [0, 1, 2],
                )
                self.assertEqual(len(row["checkpoints"]), 2)
                self.assertTrue(0.0 <= row["clip_rate"] <= 1.0)
                self.assertGreater(row["encoder_tokens_per_second"], 0.0)
            self.assertEqual(
                checkpoint_calls,
                [
                    (condition, update)
                    for condition in subject.CONDITION_ORDER
                    for update in (1, 2)
                ],
            )

            train_calls = [call for call in collate_calls if call[1] == 0]
            by_condition = {
                condition: [(epoch, values) for cell, _seed, epoch, values in train_calls if cell == condition]
                for condition in subject.CONDITION_ORDER
            }
            self.assertEqual(by_condition["A0"], by_condition["A1"])
            self.assertEqual(by_condition["M0"], by_condition["M1"])
            self.assertEqual(geometry_diagnostic_calls, ["A1", "M1"])
            final_checkpoint_diagnostics = {
                condition: diagnostic
                for condition, update, diagnostic in checkpoint_diagnostics
                if update == 2
            }
            self.assertIsNone(final_checkpoint_diagnostics["A0"])
            self.assertIsNone(final_checkpoint_diagnostics["M0"])
            self.assertEqual(
                final_checkpoint_diagnostics["A1"]["delta_nll"], 0.25
            )
            self.assertEqual(
                final_checkpoint_diagnostics["M1"]["delta_nll"], 0.25
            )
            diagnostics = {
                row["condition"]: row["final_e3fp_shuffle_diagnostic"]
                for row in report["conditions"]
            }
            self.assertIsNone(diagnostics["A0"])
            self.assertIsNone(diagnostics["M0"])
            self.assertEqual(diagnostics["A1"]["delta_nll"], 0.25)
            self.assertEqual(diagnostics["M1"]["delta_nll"], 0.25)
            self.assertTrue((output_dir / "pf1_training_manifest.json").is_file())
            manifest = json.loads(
                (output_dir / "pf1_training_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["interpretation"]["architecture_superiority_claim"])

    def test_single_condition_execution_writes_only_condition_manifest(self) -> None:
        trained: list[str] = []

        def fake_train(**kwargs):
            trained.append(kwargs["condition_id"])
            return {
                "condition": kwargs["condition_id"],
                "optimizer_updates": 2,
            }

        protocol = PF1OptimizationProtocol(
            total_updates=2,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "m1"
            with patch.object(subject, "_train_one_condition", side_effect=fake_train):
                report = subject.execute_pf1_four_grid(
                    reader=FakeReader(),
                    tokenizer_runtime=object(),
                    base_model_snapshot=Path("base-model"),
                    base_tokenizer_snapshot=Path("base-tokenizer"),
                    union_tokenizer_dir=Path("union-tokenizer"),
                    union_init_dir=Path("union-init"),
                    geometry_fusion_seed=7,
                    num_e3fp_embeddings=4096,
                    expected_vocab_size=19,
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    use_bf16=False,
                    torch_module=torch,
                    wrapper_loader=lambda **_kwargs: FakeModel(19),
                    protocol=protocol,
                    condition_ids=("M1",),
                )

            self.assertEqual(trained, ["M1"])
            self.assertEqual(report["execution"]["requested_conditions"], ["M1"])
            self.assertFalse(report["execution"]["complete_four_grid"])
            self.assertTrue((output_dir / subject.CONDITION_MANIFEST_NAME).is_file())
            self.assertFalse((output_dir / subject.FOUR_GRID_MANIFEST_NAME).exists())

    def test_execution_rejects_an_ambiguous_partial_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "partial"
            with self.assertRaisesRegex(subject.PF1TrainingError, "full grid"):
                subject.execute_pf1_four_grid(
                    reader=FakeReader(),
                    tokenizer_runtime=object(),
                    base_model_snapshot=Path("base-model"),
                    base_tokenizer_snapshot=Path("base-tokenizer"),
                    union_tokenizer_dir=Path("union-tokenizer"),
                    union_init_dir=Path("union-init"),
                    geometry_fusion_seed=7,
                    num_e3fp_embeddings=4096,
                    expected_vocab_size=19,
                    output_dir=output_dir,
                    device=torch.device("cpu"),
                    use_bf16=False,
                    torch_module=torch,
                    condition_ids=("A0", "A1"),
                )
            self.assertFalse(output_dir.exists())

    def test_train_reader_rejects_partial_microbatch(self) -> None:
        class PartialReader(FakeReader):
            def iter_train_epoch(self, *, epoch: int, batch_size: int):
                del epoch, batch_size
                yield (1,)

        cursor = subject._TrainCursor(PartialReader(), 2)
        with self.assertRaisesRegex(subject.PF1TrainingError, "full frozen"):
            cursor.next()

    def test_train_cursor_round_trip_restores_the_exact_next_batch(self) -> None:
        reader = FakeReader()
        uninterrupted = subject._TrainCursor(reader, 2)
        uninterrupted.next()
        uninterrupted.next()
        uninterrupted.next()
        saved = uninterrupted.state_dict()

        restored = subject._TrainCursor(reader, 2)
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict(), saved)
        self.assertEqual(restored.next(), uninterrupted.next())
        self.assertEqual(restored.state_dict(), uninterrupted.state_dict())

    def test_ordered_prefetch_matches_synchronous_batch_and_parameter_trajectory(
        self,
    ) -> None:
        class DropoutModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(vocab_size=19)
                self.dropout = torch.nn.Dropout(p=0.35)
                self.projection = torch.nn.Linear(1, 2)

            def forward(self, *, values, labels, use_cache, return_dict):
                del use_cache, return_dict
                logits = self.projection(self.dropout(values)).unsqueeze(1)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 2), labels.reshape(-1)
                )
                return SimpleNamespace(loss=loss, logits=logits)

        protocol = PF1OptimizationProtocol(
            total_updates=6,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
        )

        def run_trajectory(prefetch_depth: int):
            reader = FakeReader()
            collate_trace: list[tuple[int, tuple[int, ...]]] = []

            def collate(records, *, condition_id, tokenizer_runtime, seed, epoch):
                del condition_id, tokenizer_runtime, seed
                values = tuple(int(value) for value in records)
                collate_trace.append((epoch, values))
                return FakeConditionBatch(
                    condition_id="A0",
                    ce_batch=FakeCEBatch(
                        record_ids=tuple(str(value) for value in values),
                        input_lengths=(1,) * len(values),
                        target_lengths=(1,) * len(values),
                        values=values,
                    ),
                )

            def encode(batch, *, device):
                return {
                    "values": torch.tensor(
                        [[float(value % 3)] for value in batch.ce_batch.values],
                        dtype=torch.float32,
                        device=device,
                    ),
                    "labels": torch.tensor(
                        [[value % 2] for value in batch.ce_batch.values],
                        dtype=torch.long,
                        device=device,
                    ),
                }

            torch.manual_seed(7183)
            model = DropoutModel()
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(subject, "collate_pf1_condition", side_effect=collate)
                )
                stack.enter_context(
                    patch.object(
                        subject, "to_four_grid_batch_encoding", side_effect=encode
                    )
                )
                stack.enter_context(
                    patch.object(
                        subject,
                        "select_four_grid_forward_inputs",
                        side_effect=lambda encoded: encoded,
                    )
                )
                stack.enter_context(
                    patch.object(
                        subject,
                        "evaluate_pf1_condition",
                        return_value={"members": reader.dev_member_count},
                    )
                )
                stack.enter_context(patch.object(subject, "EVALUATION_UPDATES", ()))
                stack.enter_context(patch.object(subject, "CHECKPOINT_UPDATES", ()))
                report = subject._train_one_condition(
                    condition_id="A0",
                    reader=reader,
                    tokenizer_runtime=object(),
                    model=model,
                    device=torch.device("cpu"),
                    use_bf16=False,
                    output_dir=Path("unused"),
                    protocol=protocol,
                    torch_module=torch,
                    checkpoint_writer=lambda **_kwargs: "unused",
                    train_prefetch_depth=prefetch_depth,
                )
            return (
                collate_trace,
                {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                },
                torch.random.get_rng_state().clone(),
                report["final_data_cursor"],
                report["train_token_weighted_nll"],
            )

        synchronous = run_trajectory(0)
        prefetched = run_trajectory(2)
        self.assertEqual(prefetched[0], synchronous[0])
        self.assertEqual(prefetched[3:], synchronous[3:])
        self.assertTrue(torch.equal(prefetched[2], synchronous[2]))
        self.assertEqual(set(prefetched[1]), set(synchronous[1]))
        for name, expected in synchronous[1].items():
            self.assertTrue(torch.equal(prefetched[1][name], expected), name)

    def test_step_500_uses_committed_prefetch_cursor_for_exact_resume(self) -> None:
        reader = FakeReader()
        producer_cursor = subject._TrainCursor(reader, 2)

        def collate(records, **_kwargs):
            return tuple(records)

        with patch.object(subject, "collate_pf1_condition", side_effect=collate):
            with subject._OrderedTrainPrefetch(
                cursor=producer_cursor,
                total_updates=502,
                depth=2,
                gradient_accumulation_steps=1,
                condition_id="A0",
                tokenizer_runtime=object(),
                data_lock=threading.Lock(),
            ) as prefetched:
                prepared = None
                for _ in range(500):
                    prepared = next(prefetched)
                self.assertIsNotNone(prepared)
                committed_step_500 = dict(prepared.committed_cursor_state)

        reference = subject._TrainCursor(reader, 2)
        for _ in range(500):
            reference.next()
        restored = subject._TrainCursor(reader, 2)
        restored.load_state_dict(committed_step_500)
        self.assertEqual(restored.state_dict(), reference.state_dict())
        self.assertEqual(restored.next(), reference.next())

    def test_prefetch_propagates_producer_error_and_joins_thread(self) -> None:
        def fail_collate(_records, **_kwargs):
            raise RuntimeError("expected producer failure")

        cursor = subject._TrainCursor(FakeReader(), 2)
        with patch.object(
            subject, "collate_pf1_condition", side_effect=fail_collate
        ):
            prefetch = subject._OrderedTrainPrefetch(
                cursor=cursor,
                total_updates=3,
                depth=2,
                gradient_accumulation_steps=1,
                condition_id="A0",
                tokenizer_runtime=object(),
                data_lock=threading.Lock(),
            )
            with self.assertRaisesRegex(RuntimeError, "expected producer failure"):
                with prefetch:
                    next(prefetch)
        self.assertTrue(prefetch.closed)
        self.assertFalse(
            any(
                thread.name.startswith("pf1-ordered-prefetch")
                for thread in threading.enumerate()
            )
        )

    def test_step_500_checkpoint_restores_the_exact_next_update(self) -> None:
        protocol = PF1OptimizationProtocol(
            total_updates=501,
            warmup_updates=1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        )
        reader = FakeReader()

        def build_training_objects():
            model = torch.nn.Linear(1, 1)
            optimizer = subject.build_pf1_optimizer(model, protocol)
            scheduler = subject.PF1LearningRateSchedule(optimizer, protocol)
            cursor = subject._TrainCursor(reader, protocol.micro_batch_size)
            return model, optimizer, scheduler, cursor

        def one_update(model, optimizer, scheduler, cursor):
            epoch, records = cursor.next()
            random_value = torch.rand(())
            values = torch.tensor(
                [[float(value % 3)] for value in records], dtype=torch.float32
            )
            targets = torch.tensor(
                [[float(value % 2)] for value in records], dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(values + random_value), targets
            )
            loss.backward()
            subject.clip_pf1_gradients(model, protocol)
            optimizer.step()
            scheduler.step()
            return epoch, tuple(records), random_value.detach().clone()

        torch.manual_seed(923)
        model, optimizer, scheduler, cursor = build_training_objects()
        for _ in range(500):
            one_update(model, optimizer, scheduler, cursor)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            checkpoint_path = Path(
                subject.write_pf1_checkpoint(
                    output_dir=output_dir,
                    condition_id="A0",
                    update=500,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cursor_state=cursor.state_dict(),
                    torch_module=torch,
                    training_progress={"sentinel": 500},
                )
            )
            self.assertEqual(
                sorted(path.name for path in checkpoint_path.iterdir()),
                ["training_state.pt"],
            )

            reference_batch = one_update(model, optimizer, scheduler, cursor)
            reference_model = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            reference_optimizer = optimizer.state_dict()

            restored_model, restored_optimizer, restored_scheduler, restored_cursor = (
                build_training_objects()
            )
            restored = subject.load_pf1_checkpoint(
                checkpoint_dir=checkpoint_path,
                condition_id="A0",
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                cursor=restored_cursor,
                torch_module=torch,
            )
            self.assertEqual(restored["completed_updates"], 500)
            self.assertEqual(restored["training_progress"], {"sentinel": 500})

            resumed_batch = one_update(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                restored_cursor,
            )
            self.assertEqual(resumed_batch[0:2], reference_batch[0:2])
            self.assertTrue(torch.equal(resumed_batch[2], reference_batch[2]))
            for name, value in reference_model.items():
                self.assertTrue(
                    torch.equal(value, restored_model.state_dict()[name]), name
                )
            self.assertEqual(
                restored_scheduler.state_dict(), scheduler.state_dict()
            )
            self.assertEqual(
                restored_cursor.state_dict(), cursor.state_dict()
            )

            restored_optimizer_state = restored_optimizer.state_dict()
            self.assertEqual(
                restored_optimizer_state["param_groups"],
                reference_optimizer["param_groups"],
            )
            for parameter_id, reference_slots in reference_optimizer["state"].items():
                restored_slots = restored_optimizer_state["state"][parameter_id]
                self.assertEqual(set(restored_slots), set(reference_slots))
                for key, reference_value in reference_slots.items():
                    restored_value = restored_slots[key]
                    if isinstance(reference_value, torch.Tensor):
                        self.assertTrue(
                            torch.equal(reference_value, restored_value),
                            f"optimizer slot {parameter_id}:{key}",
                        )
                    else:
                        self.assertEqual(reference_value, restored_value)


if __name__ == "__main__":
    unittest.main()
