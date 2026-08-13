from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from most_t5_next.p2.launch_anchored_v4_shell_screen_v1 import MATRIX, cell_command


class AnchoredV4ShellLauncherTest(unittest.TestCase):
    def test_matrix_is_one_baseline_one_2d_and_four_3d_modes(self) -> None:
        self.assertEqual(len(MATRIX), 6)
        self.assertEqual([row[0] for row in MATRIX].count("B0"), 1)
        self.assertEqual([row[0] for row in MATRIX].count("B2D"), 1)
        self.assertEqual([row[0] for row in MATRIX].count("F3D"), 4)

    def test_command_names_independent_output(self) -> None:
        args = SimpleNamespace(
            base_model_snapshot=Path("base"),
            base_tokenizer_snapshot=Path("tokenizer"),
            anchored_tokenizer_dir=Path("anchored"),
            semantic_plan_sha256="a" * 64,
            union_init_dir=Path("init"),
            cache_root=Path("cache"),
            output_root=Path("out"),
            num_workers=0,
        )
        command = cell_command(args, "F3D", "l12_mean", "F3D-l12_mean")
        self.assertIn("most_t5_next.p2.run_anchored_v4_shell_screen_v1", command)
        self.assertEqual(command[command.index("--output-dir") + 1], str(Path("out/F3D-l12_mean")))


if __name__ == "__main__":
    unittest.main()
