from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from most_t5_next.p2.launch_qm9_standard_anchored_probe_v1 import (
    CELLS,
    PROPERTY_NAMES,
    cell_command,
)


class LaunchQM9StandardAnchoredProbeV1Test(unittest.TestCase):
    def test_matrix_is_minimal_and_command_binds_overlay(self) -> None:
        self.assertEqual(
            CELLS,
            (
                ("B0", "B0", "l0_l123_mean"),
                ("B2D", "B2D", "l0_l12_mean"),
                ("F3D-l0_l123_mean", "F3D", "l0_l123_mean"),
            ),
        )
        args = argparse.Namespace(
            base_model_snapshot=Path("base-model"),
            base_tokenizer_snapshot=Path("base-tokenizer"),
            anchored_tokenizer_dir=Path("anchored-tokenizer"),
            semantic_plan_sha256="a" * 64,
            union_init_dir=Path("union-init"),
            cache_root=Path("cache"),
            target_overlay_dir=Path("overlay"),
            output_root=Path("output"),
            epochs=30,
            micro_batch_size=256,
            gradient_accumulation_steps=1,
            learning_rate=3.0e-4,
            warmup_updates=100,
            num_workers=8,
            prefetch_factor=4,
        )
        command = cell_command(
            args,
            name="F3D-l0_l123_mean",
            cell="F3D",
            shell="l0_l123_mean",
        )
        self.assertEqual(
            command[command.index("--target-overlay-dir") + 1], "overlay"
        )
        property_start = command.index("--property-names") + 1
        self.assertEqual(tuple(command[property_start : property_start + 5]), PROPERTY_NAMES)
        self.assertEqual(command[command.index("--num-workers") + 1], "8")


if __name__ == "__main__":
    unittest.main()

