"""Contract tests for the dedicated PF-1 GPU phase profiler."""

from __future__ import annotations

import argparse
import unittest

from most_t5_next.p1 import profile_pf1_gpu_pipeline_v1 as subject


class _NoCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def is_bf16_supported() -> bool:
        return False


class _FakeTorch:
    cuda = _NoCuda()


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        paired_release="release",
        base_model_snapshot="model",
        base_tokenizer_snapshot="tokenizer",
        union_init_dir="init",
        output_report="report.json",
        geometry_fusion_seed=1,
        num_e3fp_embeddings=4096,
        condition_id="M0",
        warmup_updates=3,
        profile_updates=10,
    )


class PF1GPUPipelineProfileTest(unittest.TestCase):
    def test_distribution_reports_fixed_population(self) -> None:
        summary = subject._distribution((1.0, 2.0, 3.0, 4.0))
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["max"], 4.0)

    def test_negative_timing_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            subject.PF1GPUPipelineProfileError,
            "finite and non-negative",
        ):
            subject._distribution((0.1, -0.1))

    def test_cli_exposes_diagnostics_but_not_training_hyperparameters(self) -> None:
        destinations = {action.dest for action in subject.build_parser()._actions}
        self.assertIn("warmup_updates", destinations)
        self.assertIn("profile_updates", destinations)
        self.assertIn("condition_id", destinations)
        self.assertNotIn("learning_rate", destinations)
        self.assertNotIn("micro_batch_size", destinations)
        self.assertNotIn("gradient_accumulation_steps", destinations)

    def test_run_requires_cuda_before_loading_any_artifact(self) -> None:
        with self.assertRaisesRegex(
            subject.PF1GPUPipelineProfileError,
            "BF16 CUDA",
        ):
            subject.run(_args(), torch_module=_FakeTorch())


if __name__ == "__main__":
    unittest.main()
