from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from most_t5_next.p1.pf1_optimization import PF1_SCREEN_PROTOCOL_ID
    from most_t5_next.p2.run_pf2_gated_fusion_v1 import (
        FUSION_CONTRACT,
        build_f_gate_optimizer,
    )
    from most_t5_next.p2.run_pf2_t3mi_v1 import (
        REPORT_SCHEMA,
        T3MI_CHECKPOINT_CONTRACT_NAME,
        T3MI_MANIFEST_NAME,
        T3MI_MASK_PROBABILITY,
        T3MI_OBJECTIVE_CONTRACT,
        T3MI_PROTOCOL,
        T3MI_PROTOCOL_ID,
        build_parser,
        execute_pf2_t3mi,
        validate_t3mi_checkpoint_contract,
        write_t3mi_checkpoint,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PF2T3MIRunnerTest(unittest.TestCase):
    @staticmethod
    def _required_cli():
        return [
            "--paired-release", "paired",
            "--base-model-snapshot", "model",
            "--base-tokenizer-snapshot", "tokenizer",
            "--union-init-dir", "init",
            "--output-dir", "out",
            "--geometry-fusion-seed", "7",
        ]

    @staticmethod
    def _write_contract(checkpoint: Path, condition: str, update: int, logit: float):
        checkpoint.mkdir(parents=True)
        payload = {
            "schema_version": "most-t5-p2/t3mi-checkpoint-contract/v1",
            "condition_id": condition,
            "completed_updates": update,
            "objective_contract": T3MI_OBJECTIVE_CONTRACT,
            "fusion_contract": FUSION_CONTRACT,
            "optimization_protocol": asdict(T3MI_PROTOCOL),
            "optimization_protocol_id": T3MI_PROTOCOL_ID,
            "gate_state": {
                "logit": logit,
                "effective_tanh_gate": math.tanh(logit),
            },
        }
        (checkpoint / T3MI_CHECKPOINT_CONTRACT_NAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_cli_freezes_one_motif_condition_and_32x4_protocol(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(self._required_cli())
        args = build_parser().parse_args(
            self._required_cli() + ["--condition-id", "M1"]
        )
        self.assertEqual(args.protocol_id, PF1_SCREEN_PROTOCOL_ID)
        self.assertEqual(T3MI_PROTOCOL.micro_batch_size, 32)
        self.assertEqual(T3MI_PROTOCOL.gradient_accumulation_steps, 4)
        self.assertEqual(T3MI_MASK_PROBABILITY, 1.0)

    def test_executor_changes_only_the_all_identity_view(self):
        captured = {}

        def fake_engine(**kwargs):
            captured.update(kwargs)
            output = Path(kwargs["output_dir"])
            checkpoints = []
            for update, logit in ((500, 0.1), (1000, 0.2)):
                checkpoint = output / "M1" / f"step-{update:04d}"
                self._write_contract(checkpoint, "M1", update, logit)
                checkpoints.append(str(checkpoint))
            return {
                "schema_version": "pf1-engine/v1",
                "status": "pass",
                "data": {
                    "train_members": 10,
                    "dev_members": 2,
                    "mask_probability": 1.0,
                },
                "optimization": {"protocol_id": "old"},
                "interpretation": {},
                "conditions": [{
                    "condition": "M1",
                    "checkpoints": checkpoints,
                }],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            release_root = Path(temp_dir) / "paired"
            release_root.mkdir()
            paired_manifest = {
                "schema_version": "most-t5-p1/pf1-paired-release/v1",
                "counts": {"train_members": 10, "dev_members": 2},
            }
            (release_root / "manifest.json").write_text(
                json.dumps(paired_manifest), encoding="utf-8"
            )
            report = execute_pf2_t3mi(
                engine=fake_engine,
                output_dir=output_dir,
                condition_ids=("M1",),
                protocol=T3MI_PROTOCOL,
                reader=SimpleNamespace(
                    release_root=release_root,
                    manifest=paired_manifest,
                ),
            )
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(report["objective_contract"], T3MI_OBJECTIVE_CONTRACT)
            self.assertEqual(report["optimization"]["protocol_id"], T3MI_PROTOCOL_ID)
            self.assertEqual(captured["mask_probability"], 1.0)
            self.assertIs(captured["protocol"], T3MI_PROTOCOL)
            self.assertIs(captured["checkpoint_writer"], write_t3mi_checkpoint)
            self.assertIs(captured["optimizer_builder"], build_f_gate_optimizer)
            saved = json.loads(
                (output_dir / T3MI_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(saved, report)

    def test_executor_rejects_scope_protocol_or_non_t3mi_engine(self):
        def should_not_run(**kwargs):
            raise AssertionError("invalid contract reached the engine")

        for conditions in (("A1",), ("M0", "M1"), ()):
            with self.subTest(conditions=conditions):
                with self.assertRaisesRegex(RuntimeError, "exactly M0 or M1"):
                    execute_pf2_t3mi(
                        engine=should_not_run,
                        output_dir=Path("unused"),
                        condition_ids=conditions,
                        protocol=T3MI_PROTOCOL,
                    )
        with self.assertRaisesRegex(RuntimeError, "32x4"):
            execute_pf2_t3mi(
                engine=should_not_run,
                output_dir=Path("unused"),
                condition_ids=("M1",),
                protocol=None,
            )

        def wrong_mask(**_kwargs):
            return {
                "status": "pass",
                "data": {"mask_probability": 0.15},
                "conditions": [],
            }

        with self.assertRaisesRegex(RuntimeError, "all-identity"):
            execute_pf2_t3mi(
                engine=wrong_mask,
                output_dir=Path("unused"),
                condition_ids=("M1",),
                protocol=T3MI_PROTOCOL,
            )

    def test_checkpoint_contract_binds_objective_and_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "step-0500"
            self._write_contract(checkpoint, "M1", 500, 0.1)
            loaded = validate_t3mi_checkpoint_contract(
                checkpoint, condition_id="M1"
            )
            self.assertEqual(loaded["objective_contract"], T3MI_OBJECTIVE_CONTRACT)
            payload = json.loads(
                (checkpoint / T3MI_CHECKPOINT_CONTRACT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            payload["objective_contract"]["mask_probability"] = 0.15
            (checkpoint / T3MI_CHECKPOINT_CONTRACT_NAME).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "static contract"):
                validate_t3mi_checkpoint_contract(checkpoint, condition_id="M1")


if __name__ == "__main__":
    unittest.main()
