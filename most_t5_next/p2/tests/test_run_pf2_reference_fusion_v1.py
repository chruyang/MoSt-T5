from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from most_t5_next.p2.reference_geometry_fusion_v1 import FUSION_ID
    from most_t5_next.p2.run_pf2_reference_fusion_v1 import (
        PF2_MANIFEST_NAME,
        PF2_CHECKPOINT_CONTRACT_NAME,
        PF2_PROTOCOL,
        REPORT_SCHEMA,
        build_parser,
        execute_pf2_reference_fusion,
        validate_pf2_checkpoint_contract,
        write_pf2_checkpoint,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PF2ReferenceFusionRunnerTest(unittest.TestCase):
    def test_cli_requires_exactly_one_motif_condition(self):
        required = [
            "--paired-release", "paired",
            "--base-model-snapshot", "model",
            "--base-tokenizer-snapshot", "tokenizer",
            "--union-init-dir", "init",
            "--output-dir", "out",
            "--geometry-fusion-seed", "7",
        ]
        with self.assertRaises(SystemExit):
            build_parser().parse_args(required)
        with self.assertRaises(SystemExit):
            build_parser().parse_args(required + ["--condition-id", "A1"])
        args = build_parser().parse_args(required + ["--condition-id", "M1"])
        self.assertEqual(args.condition_id, "M1")

    def test_report_binds_the_only_scientific_change(self):
        captured = {}

        def fake_engine(**kwargs):
            captured.update(kwargs)
            Path(kwargs["output_dir"]).mkdir()
            return {
                "schema_version": "legacy-engine/v1",
                "status": "pass",
                "interpretation": {"architecture_superiority_claim": False},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            report = execute_pf2_reference_fusion(
                engine=fake_engine,
                output_dir=output_dir,
                condition_ids=("M1",),
            )
            self.assertEqual(report["schema_version"], REPORT_SCHEMA)
            self.assertEqual(report["fusion_contract"]["fusion_id"], FUSION_ID)
            self.assertIs(
                captured["wrapper_loader"],
                __import__(
                    "most_t5_next.p2.reference_geometry_fusion_v1",
                    fromlist=["load_verified_reference_four_grid_wrapper"],
                ).load_verified_reference_four_grid_wrapper,
            )
            self.assertEqual(PF2_PROTOCOL.micro_batch_size, 64)
            self.assertEqual(PF2_PROTOCOL.gradient_accumulation_steps, 2)
            self.assertEqual(PF2_PROTOCOL.effective_batch_size, 128)
            self.assertIs(captured["protocol"], PF2_PROTOCOL)
            self.assertIs(captured["checkpoint_writer"], write_pf2_checkpoint)
            saved = json.loads(
                (output_dir / PF2_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(saved, report)
            self.assertFalse(saved["fusion_contract"]["teacher"])
            self.assertFalse(saved["fusion_contract"]["auxiliary_loss"])

    def test_executor_rejects_atom_or_multi_condition_scope(self):
        def should_not_run(**kwargs):
            raise AssertionError("invalid scope reached the engine")

        for conditions in (("A1",), ("M0", "M1"), ()):
            with self.subTest(conditions=conditions):
                with self.assertRaisesRegex(RuntimeError, "one motif condition"):
                    execute_pf2_reference_fusion(
                        engine=should_not_run,
                        output_dir=Path("unused"),
                        condition_ids=conditions,
                    )

    def test_checkpoint_contract_binds_fusion_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "step-0500"
            checkpoint.mkdir()
            with mock.patch(
                "most_t5_next.p2.run_pf2_reference_fusion_v1.write_pf1_checkpoint",
                return_value=str(checkpoint),
            ):
                returned = write_pf2_checkpoint(condition_id="M1", update=500)
            self.assertEqual(Path(returned), checkpoint)
            loaded = validate_pf2_checkpoint_contract(
                checkpoint,
                condition_id="M1",
            )
            self.assertEqual(loaded["completed_updates"], 500)

            path = checkpoint / PF2_CHECKPOINT_CONTRACT_NAME
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["fusion_contract"]["carrier_geometry_coefficient"] = 0.25
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "contract differs"):
                validate_pf2_checkpoint_contract(checkpoint, condition_id="M1")


if __name__ == "__main__":
    unittest.main()
