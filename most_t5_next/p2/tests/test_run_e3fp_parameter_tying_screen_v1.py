from __future__ import annotations

import argparse
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from most_t5_next.p2.e3fp_atom_embedding_v1 import (
    LEVEL_SPECIFIC_FIXED4,
    L0_STATE_FIXED4,
    REFERENCE_SHARED_FIXED4,
)
from most_t5_next.p2.run_e3fp_parameter_tying_screen_v1 import (
    CANDIDATES,
    PF10_PROTOCOL,
    cell_contract,
    validate_pf10_cache_boundary,
)


class E3FPParameterTyingScreenTest(unittest.TestCase):
    def test_cache_boundary_rejects_pf1_and_accepts_exact_pf10(self) -> None:
        root = Path("tmp") / f"e3fp_cache_boundary_{uuid4().hex}"
        root.mkdir(parents=True)
        manifest = root / "manifest.json"
        try:
            payload = {
                "status": "pass",
                "counts": {
                    "records": 33600,
                    "train_records": 30240,
                    "dev_records": 3360,
                },
                "source": {"derived_representation": {
                    "schema_version": "most-t5-p2/anchored-training-tensor-cache-build/v1"
                }},
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "336,006"):
                validate_pf10_cache_boundary(root)
            payload["counts"] = {
                "records": 336006,
                "train_records": 302406,
                "dev_records": 33600,
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(validate_pf10_cache_boundary(root)["records"], 336006)
        finally:
            if manifest.is_file():
                manifest.unlink()
            if root.is_dir():
                root.rmdir()

    def test_matrix_candidates_and_pf10_budget_are_frozen(self) -> None:
        self.assertEqual(
            CANDIDATES,
            (REFERENCE_SHARED_FIXED4, L0_STATE_FIXED4, LEVEL_SPECIFIC_FIXED4),
        )
        self.assertEqual(PF10_PROTOCOL.total_updates, 10000)
        self.assertEqual(
            PF10_PROTOCOL.micro_batch_size * PF10_PROTOCOL.gradient_accumulation_steps,
            128,
        )
        for candidate in CANDIDATES:
            self.assertEqual(cell_contract("B2D", candidate)["state_kind"], "coordinate_blind_morgan")
            self.assertEqual(cell_contract("F3D", candidate)["state_kind"], "e3fp")

    def test_source_control_boundary_requires_matching_clean_checkout(self) -> None:
        import most_t5_next.p2.run_e3fp_parameter_tying_screen_v1 as module

        expected = "a" * 40
        clean = type("Completed", (), {"stdout": expected + "\n"})()
        status = type("Completed", (), {"stdout": ""})()
        with patch.object(module.subprocess, "run", side_effect=(clean, status)) as mocked:
            report = module.validate_source_control_boundary(expected)
        self.assertEqual(report["git_commit"], expected)
        self.assertEqual(mocked.call_args_list[1].args[0][-1], "--untracked-files=normal")

        dirty = type("Completed", (), {"stdout": "?? stray.py\n"})()
        with patch.object(module.subprocess, "run", side_effect=(clean, dirty)):
            with self.assertRaisesRegex(Exception, "not clean"):
                module.validate_source_control_boundary(expected)

    def test_bridge_forces_768_and_carrier_only_training(self) -> None:
        import most_t5_next.p2.run_e3fp_parameter_tying_screen_v1 as module

        root = Path("tmp") / f"e3fp_tying_{uuid4().hex}"
        output = root / "cell"
        root.mkdir(parents=True)
        try:
            args = argparse.Namespace(
                cell="F3D",
                parameter_tying=L0_STATE_FIXED4,
                cache_root=root / "cache",
                output_dir=output,
                code_commit="a" * 40,
            )

            args.cache_root.mkdir()
            (args.cache_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "counts": {
                            "records": 336006,
                            "train_records": 302406,
                            "dev_records": 33600,
                        },
                        "source": {"derived_representation": {
                            "schema_version": "most-t5-p2/anchored-training-tensor-cache-build/v1"
                        }},
                    }
                ),
                encoding="utf-8",
            )

            def fake_shared(bridge, **kwargs):
                self.assertEqual(bridge.shell_fusion_mode, "not_applicable")
                self.assertEqual(kwargs["training_component_mode"], "carrier_only")
                self.assertEqual(kwargs["optimization_protocol"].total_updates, 10000)
                observed = {}

                def fake_loader(**values):
                    observed.update(values)
                    return object()

                with patch.object(module, "load_deterministic_factorized_model_v10", fake_loader):
                    kwargs["model_loader"](
                        shell_fusion_mode="not_applicable",
                        atom_memory_dim=128,
                    )
                self.assertEqual(observed["atom_memory_dim"], 768)
                self.assertEqual(observed["parameter_tying"], L0_STATE_FIXED4)
                output.mkdir()
                return {"status": "pass", "cell": {"cell": "F3D"}}

            source_control = {
                "git_commit": "a" * 40,
                "tracked_worktree_clean": True,
                "architecture_contract_id": module.ARCHITECTURE_CONTRACT_ID,
            }
            with patch.object(module, "validate_source_control_boundary", return_value=source_control):
                with patch.object(module, "run_shared_screen", fake_shared):
                    report = module.run(args)
            self.assertTrue(report["decision_contract"]["endpoint_injection_selection_deferred"])
            self.assertTrue(report["decision_contract"]["three_dimensional_probe_required_after_training"])
            self.assertEqual(report["source_control"], source_control)
        finally:
            cache_manifest = root / "cache" / "manifest.json"
            if cache_manifest.is_file():
                cache_manifest.unlink()
            if (root / "cache").is_dir():
                (root / "cache").rmdir()
            manifest = output / module.MANIFEST_FILENAME
            if manifest.is_file():
                manifest.unlink()
            if output.is_dir():
                output.rmdir()
            if root.is_dir():
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
