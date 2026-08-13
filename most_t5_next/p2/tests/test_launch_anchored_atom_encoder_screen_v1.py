from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from most_t5_next.p2.launch_anchored_atom_encoder_screen_v1 import (
    MATRIX,
    cell_command,
)


class AnchoredAtomEncoderLauncherTest(unittest.TestCase):
    def test_matrix_is_three_strict_b2d_f3d_pairs(self) -> None:
        self.assertEqual(len(MATRIX), 6)
        candidates = {row[1] for row in MATRIX}
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertEqual(
                {row[0] for row in MATRIX if row[1] == candidate},
                {"B2D", "F3D"},
            )

    def test_command_binds_candidate_workers_and_overlay(self) -> None:
        args = SimpleNamespace(
            base_model_snapshot=Path("base"),
            base_tokenizer_snapshot=Path("tokenizer"),
            anchored_tokenizer_dir=Path("anchored"),
            semantic_plan_sha256="a" * 64,
            union_init_dir=Path("init"),
            cache_root=Path("cache"),
            output_root=Path("out"),
            num_workers=0,
            prefetch_factor=4,
            matched_overlay=Path("matched"),
        )
        command = cell_command(
            args,
            "F3D",
            "l0_high_level_aware_phi",
            "F3D-level",
        )
        self.assertIn("most_t5_next.p2.run_anchored_atom_encoder_screen_v1", command)
        self.assertEqual(
            command[command.index("--atom-encoder-candidate") + 1],
            "l0_high_level_aware_phi",
        )
        self.assertEqual(command[command.index("--num-workers") + 1], "0")
        self.assertEqual(command[command.index("--prefetch-factor") + 1], "4")
        self.assertEqual(command[command.index("--matched-overlay") + 1], "matched")


if __name__ == "__main__":
    unittest.main()
