from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from most_t5_next.p2.run_e3fp_parameter_tying_screen_v1 import (
    CANDIDATES,
    MANIFEST_FILENAME,
)
from most_t5_next.p2.launch_e3fp_parameter_tying_screen_v1 import (
    cell_command,
    merge_completed_matrix,
    pair_command,
)


class E3FPParameterTyingLauncherTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            base_model_snapshot=Path("base"),
            base_tokenizer_snapshot=Path("tokenizer"),
            anchored_tokenizer_dir=Path("anchored"),
            semantic_plan_sha256="a" * 64,
            union_init_dir=Path("init"),
            cache_root=Path("cache"),
            output_root=Path("out"),
            code_commit="a" * 40,
            workers_per_cell=8,
            prefetch_factor=5,
            matched_b2d_overlay=Path("matched-b2d"),
            matched_f3d_overlay=Path("matched-f3d"),
            save_final_checkpoints=False,
        )

    def test_commands_bind_workers_candidate_and_gpu_pair_mode(self) -> None:
        args = self._args()
        command = cell_command(args, cell="F3D", candidate=CANDIDATES[1])
        self.assertEqual(command[command.index("--parameter-tying") + 1], CANDIDATES[1])
        self.assertEqual(command[command.index("--num-workers") + 1], "8")
        self.assertEqual(command[command.index("--prefetch-factor") + 1], "5")
        self.assertEqual(command[command.index("--code-commit") + 1], "a" * 40)
        self.assertEqual(command[command.index("--matched-overlay") + 1], "matched-f3d")
        self.assertNotIn("--save-final-checkpoint", command)
        pair = pair_command(args, candidate=CANDIDATES[2], gpu_id=2)
        self.assertEqual(pair[pair.index("--pair-candidate") + 1], CANDIDATES[2])
        self.assertEqual(pair[pair.index("--pair-gpu-id") + 1], "2")
        self.assertEqual(pair[pair.index("--matched-b2d-overlay") + 1], "matched-b2d")
        self.assertEqual(pair[pair.index("--matched-f3d-overlay") + 1], "matched-f3d")

    def test_merge_requires_update_zero_equality_within_b2d_and_f3d(self) -> None:
        root = Path("tmp") / f"e3fp_merge_{uuid4().hex}"
        created = []
        try:
            for candidate in CANDIDATES:
                for cell in ("B2D", "F3D"):
                    directory = root / f"{candidate}-{cell}"
                    directory.mkdir(parents=True)
                    created.append(directory)
                    payload = {
                        "status": "pass",
                        "source_control": {
                            "git_commit": "a" * 40,
                            "tracked_worktree_clean": True,
                            "architecture_contract_id": "test-architecture",
                        },
                        "evaluations": [
                            {
                                "update": 0,
                                "conditions": [{"cell": cell, "token_weighted_nll": 4.0}],
                            }
                        ],
                    }
                    (directory / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
            report = merge_completed_matrix(root)
            self.assertEqual(report["cell_count"], 6)
            self.assertTrue(report["update_zero_evaluation_equal_within_state_kind"])
            bad = root / f"{CANDIDATES[-1]}-F3D" / MANIFEST_FILENAME
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload["evaluations"][0]["conditions"][0]["token_weighted_nll"] = 5.0
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "update-zero"):
                merge_completed_matrix(root)
        finally:
            for directory in reversed(created):
                manifest = directory / MANIFEST_FILENAME
                if manifest.is_file():
                    manifest.unlink()
                if directory.is_dir():
                    directory.rmdir()
            if root.is_dir():
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
