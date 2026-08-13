from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from most_t5_next.p2.launch_anchored_v4_confirmation_v1 import MATRIX, cell_command


class AnchoredV4ConfirmationLauncherTest(unittest.TestCase):
    def test_matrix_and_command_are_minimal_and_checkpointed(self) -> None:
        self.assertEqual(
            MATRIX,
            (
                ("B2D", "l0_l12_mean", "B2D"),
                ("F3D", "l0_l12_mean", "F3D-l0_l12_mean"),
                ("F3D", "l0_l123_mean", "F3D-l0_l123_mean"),
            ),
        )
        args = argparse.Namespace(
            base_model_snapshot=Path("base-model"),
            base_tokenizer_snapshot=Path("base-tokenizer"),
            anchored_tokenizer_dir=Path("tokenizer"),
            semantic_plan_sha256="a" * 64,
            union_init_dir=Path("init"),
            cache_root=Path("cache"),
            matched_overlay=Path("matched"),
            output_root=Path("out"),
            num_workers=0,
        )
        command = cell_command(args, "F3D", "l0_l123_mean", "F3D-l0_l123_mean")
        self.assertIn("--matched-overlay", command)
        self.assertIn("--save-final-checkpoint", command)
        self.assertEqual(command[command.index("--shell-fusion-mode") + 1], "l0_l123_mean")


if __name__ == "__main__":
    unittest.main()
