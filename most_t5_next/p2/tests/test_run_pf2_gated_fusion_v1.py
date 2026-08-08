from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from most_t5_next.p1.pf1_optimization import G_CODEC_PROTOCOL_ID
    from most_t5_next.p2.gated_reference_geometry_fusion_v1 import (
        FUSION_ID,
        ZeroInitGatedE3FPCarrierFusion,
    )
    from most_t5_next.p2.run_pf2_gated_fusion_v1 import (
        FUSION_CONTRACT,
        F_GATE_CHECKPOINT_CONTRACT_NAME,
        F_GATE_MANIFEST_NAME,
        F_GATE_PROTOCOL,
        F_GATE_PROTOCOL_ID,
        REPORT_SCHEMA,
        build_parser,
        execute_pf2_gated_fusion,
        validate_f_gate_checkpoint_contract,
        write_f_gate_checkpoint,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PF2GatedFusionRunnerTest(unittest.TestCase):
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
            "schema_version": "most-t5-p2/gated-fusion-checkpoint-contract/v1",
            "fusion_contract": FUSION_CONTRACT,
            "condition_id": condition,
            "completed_updates": update,
            "optimization_protocol": asdict(F_GATE_PROTOCOL),
            "optimization_protocol_id": F_GATE_PROTOCOL_ID,
            "gate_state": {
                "logit": logit,
                "effective_tanh_gate": __import__("math").tanh(logit),
            },
        }
        (checkpoint / F_GATE_CHECKPOINT_CONTRACT_NAME).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_cli_freezes_one_motif_condition_and_64x2_protocol(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(self._required_cli())
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                self._required_cli() + ["--condition-id", "A1"]
            )
        args = build_parser().parse_args(
            self._required_cli() + ["--condition-id", "M1"]
        )
        self.assertEqual(args.condition_id, "M1")
        self.assertEqual(args.protocol_id, G_CODEC_PROTOCOL_ID)
        self.assertEqual(F_GATE_PROTOCOL.micro_batch_size, 64)
        self.assertEqual(F_GATE_PROTOCOL.gradient_accumulation_steps, 2)

    def test_report_binds_only_the_gated_fusion_change(self):
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
                "data": {"train_members": 10, "dev_members": 2},
                "optimization": {"protocol_id": "old"},
                "interpretation": {"architecture_superiority_claim": False},
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
                json.dumps(paired_manifest),
                encoding="utf-8",
            )
            report = execute_pf2_gated_fusion(
                engine=fake_engine,
                output_dir=output_dir,
                condition_ids=("M1",),
                protocol=F_GATE_PROTOCOL,
                reader=SimpleNamespace(
                    release_root=release_root,
                    manifest=paired_manifest,
                ),
            )
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(report["fusion_contract"]["fusion_id"], FUSION_ID)
            self.assertEqual(
                report["optimization"]["protocol_id"],
                F_GATE_PROTOCOL_ID,
            )
            self.assertEqual(
                report["conditions"][0]["final_geometry_gate"]["update"],
                1000,
            )
            self.assertIs(
                captured["wrapper_loader"],
                __import__(
                    "most_t5_next.p2.gated_reference_geometry_fusion_v1",
                    fromlist=["load_verified_gated_four_grid_wrapper"],
                ).load_verified_gated_four_grid_wrapper,
            )
            self.assertIs(captured["protocol"], F_GATE_PROTOCOL)
            self.assertIs(captured["checkpoint_writer"], write_f_gate_checkpoint)
            self.assertEqual(
                report["data"]["paired_release_root"],
                str(release_root.resolve()),
            )
            saved = json.loads(
                (output_dir / F_GATE_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(saved, report)

    def test_executor_rejects_wrong_scope_or_protocol(self):
        def should_not_run(**kwargs):
            raise AssertionError("invalid contract reached the engine")

        for conditions in (("A1",), ("M0", "M1"), ()):
            with self.subTest(conditions=conditions):
                with self.assertRaisesRegex(RuntimeError, "one motif condition"):
                    execute_pf2_gated_fusion(
                        engine=should_not_run,
                        output_dir=Path("unused"),
                        condition_ids=conditions,
                        protocol=F_GATE_PROTOCOL,
                    )
        with self.assertRaisesRegex(RuntimeError, "64x2"):
            execute_pf2_gated_fusion(
                engine=should_not_run,
                output_dir=Path("unused"),
                condition_ids=("M1",),
                protocol=None,
            )

    def test_checkpoint_contract_records_and_validates_gate_state(self):
        module = ZeroInitGatedE3FPCarrierFusion(
            num_e3fp_embeddings=8,
            hidden_size=2,
        )
        module.geometry_gate_logit.detach().zero_()
        model = SimpleNamespace(geometry_fusion=module)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "step-0500"
            checkpoint.mkdir()
            with mock.patch(
                "most_t5_next.p2.run_pf2_gated_fusion_v1.write_pf1_checkpoint",
                return_value=str(checkpoint),
            ):
                returned = write_f_gate_checkpoint(
                    condition_id="M1",
                    update=500,
                    model=model,
                )
            self.assertEqual(Path(returned), checkpoint)
            loaded = validate_f_gate_checkpoint_contract(
                checkpoint,
                condition_id="M1",
            )
            self.assertEqual(loaded["gate_state"]["logit"], 0.0)

            path = checkpoint / F_GATE_CHECKPOINT_CONTRACT_NAME
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["gate_state"]["effective_tanh_gate"] = 0.5
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "gate state"):
                validate_f_gate_checkpoint_contract(checkpoint, condition_id="M1")


if __name__ == "__main__":
    unittest.main()
