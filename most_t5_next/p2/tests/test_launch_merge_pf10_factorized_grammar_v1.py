from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock
import unittest
import uuid

from most_t5_next.p2 import launch_pf10_factorized_grammar_matrix_v1 as launcher
from most_t5_next.p2 import merge_pf10_factorized_grammar_v1 as merger
from most_t5_next.p2.run_pf10_factorized_grammar_v1 import SCHEMA_VERSION as CELL_SCHEMA


class _Process:
    def __init__(self, command, **kwargs) -> None:
        self.command = command
        self.environment = kwargs["env"]

    @staticmethod
    def poll() -> int:
        return 0


def _cell(cell: str, nll: float, *, diagnostics=None) -> dict[str, object]:
    return {
        "schema_version": CELL_SCHEMA,
        "status": "pass",
        "cell": cell,
        "protocol": {
            "updates": 10000,
            "micro_batch_size": 32,
            "gradient_accumulation_steps": 4,
        },
        "data_contract": {"paired_release": "same", "dev_members": 33600},
        "evaluations": [
            {"update": 0, "token_weighted_nll": nll + 1, "masked_token_accuracy": 0.0},
            {"update": 10000, "token_weighted_nll": nll, "masked_token_accuracy": 0.5},
        ],
        "f3d_state_diagnostics": diagnostics,
    }


class PF10GrammarLaunchMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / ("pf10_launch_merge_test_" + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self) -> None:
        output = self.root / "launch"
        for cell in launcher.CELLS:
            path = output / f"{cell}.log"
            if path.is_file():
                path.unlink()
        launch_manifest = output / "launcher_manifest.json"
        if launch_manifest.is_file():
            launch_manifest.unlink()
        if output.is_dir():
            output.rmdir()
        for name in ("B0.json", "B2D.json", "F3D.json", "merged.json"):
            path = self.root / name
            if path.is_file():
                path.unlink()
        self.root.rmdir()

    def test_two_gpus_queue_three_cells_without_changing_cell_commands(self) -> None:
        args = argparse.Namespace(
            cells="B0,B2D,F3D",
            gpu_ids="0,1",
            paired_release=Path("paired"),
            morgan_overlay=Path("morgan"),
            shuffle_overlay=Path("shuffle"),
            s_stage_root=Path("state"),
            base_model_snapshot=Path("model"),
            base_tokenizer_snapshot=Path("tokenizer"),
            union_init_dir=Path("init"),
            output_root=self.root / "launch",
            cache_workers=4,
            cache_max_pending=16,
        )
        created = []

        def factory(command, **kwargs):
            process = _Process(command, **kwargs)
            created.append(process)
            return process

        with mock.patch.object(launcher.subprocess, "Popen", side_effect=factory):
            report = launcher.launch_matrix(args)

        self.assertEqual(report["status"], "pass")
        self.assertEqual([row["cell"] for row in report["cell_results"]], list(launcher.CELLS))
        self.assertEqual([row["gpu_id"] for row in report["cell_results"]], ["0", "1", "0"])
        self.assertEqual([row.environment["CUDA_VISIBLE_DEVICES"] for row in created], ["0", "1", "0"])
        self.assertTrue(
            all(
                any("run_pf10_factorized_grammar_v1" in item for item in row.command)
                for row in created
            )
        )

    def test_merge_reports_directional_screen_without_claiming_final_training(self) -> None:
        diagnostics = {
            "zero_minus_aligned_delta_nll": 0.3,
            "shuffle_minus_aligned_delta_nll": 0.2,
        }
        documents = {
            "B0": _cell("B0", 1.2),
            "B2D": _cell("B2D", 1.1),
            "F3D": _cell("F3D", 0.9, diagnostics=diagnostics),
        }
        for cell, document in documents.items():
            (self.root / f"{cell}.json").write_text(json.dumps(document), encoding="utf-8")

        report = merger.merge_grammar_matrix(
            b0_manifest=self.root / "B0.json",
            b2d_manifest=self.root / "B2D.json",
            f3d_manifest=self.root / "F3D.json",
            output=self.root / "merged.json",
        )

        self.assertTrue(report["all_directional_gates_pass"])
        self.assertEqual(report["scope"], "pf10_one_seed_causal_screen")
        self.assertIn("not final pretraining", report["interpretation_boundary"])


if __name__ == "__main__":
    unittest.main()
