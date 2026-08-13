from __future__ import annotations

import argparse
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local fixture")
class AnchoredAtomEncoderScreenTest(unittest.TestCase):
    def test_three_candidates_isolate_level_without_role_or_presence(self) -> None:
        from most_t5_next.p2.factorized_model_init_v5 import (
            factorized_initialization_contract_v5,
        )
        from most_t5_next.p2.factorized_model_init_v8 import (
            factorized_initialization_contract_v8,
        )
        from most_t5_next.p2.factorized_model_init_v9 import (
            factorized_initialization_contract_v9,
        )

        kwargs = dict(
            semantic_plan_sha256="a" * 64,
            adapter_seed=7,
            num_e3fp_embeddings=4096,
            state_level2_weight=0.25,
            state_embedding_dim=64,
            atom_memory_dim=128,
            max_identity_span_length=128,
            max_atoms_per_motif=128,
            geometry_fraction=0.5,
        )
        reference = factorized_initialization_contract_v5(**kwargs)
        split = factorized_initialization_contract_v8(**kwargs)
        level = factorized_initialization_contract_v9(**kwargs)
        self.assertFalse(reference["level_embedding"])
        self.assertFalse(split["level_embedding"])
        self.assertTrue(level["level_embedding"])
        for contract in (reference, split, level):
            self.assertFalse(contract["attachment_role_is_learned_atom_input"])
        self.assertFalse(split["presence_feature"])
        self.assertFalse(level["presence_feature"])

    def test_cell_contract_requires_one_b2d_or_f3d_pair(self) -> None:
        from most_t5_next.p2.run_anchored_atom_encoder_screen_v1 import (
            AnchoredAtomEncoderScreenError,
            cell_contract,
        )

        candidate = "l0_high_level_aware_phi"
        self.assertEqual(cell_contract("B2D", candidate)["state_kind"], "coordinate_blind_morgan")
        self.assertEqual(cell_contract("F3D", candidate)["state_kind"], "e3fp")
        self.assertEqual(cell_contract("F3D", candidate)["factor_under_test"], "level_embedding")
        with self.assertRaisesRegex(AnchoredAtomEncoderScreenError, "cell"):
            cell_contract("B0", candidate)
        with self.assertRaisesRegex(AnchoredAtomEncoderScreenError, "candidate"):
            cell_contract("F3D", "unknown")

    def test_generic_bridge_strips_only_the_legacy_shell_argument(self) -> None:
        import most_t5_next.p2.run_anchored_atom_encoder_screen_v1 as module

        observed = {}

        def target(**kwargs):
            observed.update(kwargs)
            return "ok"

        wrapped = module._without_legacy_shell_argument(target)
        self.assertEqual(wrapped(shell_fusion_mode="not_applicable", value=3), "ok")
        self.assertEqual(observed, {"value": 3})
        with self.assertRaisesRegex(module.AnchoredAtomEncoderScreenError, "legacy"):
            wrapped(shell_fusion_mode="l0_l123_mean")

    def test_run_records_elimination_only_boundary(self) -> None:
        import most_t5_next.p2.run_anchored_atom_encoder_screen_v1 as module

        root = Path("tmp") / f"atom_encoder_screen_test_{uuid4().hex}"
        output = root / "cell"
        root.mkdir(parents=True)
        try:
            args = argparse.Namespace(
                cell="F3D",
                atom_encoder_candidate="l0_high_minimal_phi",
                output_dir=output,
            )

            def fake_shared(bridge_args, **kwargs):
                self.assertEqual(bridge_args.shell_fusion_mode, "not_applicable")
                self.assertEqual(
                    kwargs["contract_builder"]("F3D", "not_applicable")[
                        "atom_encoder_candidate"
                    ],
                    "l0_high_minimal_phi",
                )
                output.mkdir()
                return {"status": "pass", "cell": {"cell": "F3D"}}

            with patch.object(module, "run_shared_screen", fake_shared):
                report = module.run(args)
            self.assertTrue(report["selection_boundary"]["one_percent_is_elimination_only"])
            self.assertTrue(report["selection_boundary"]["ten_percent_is_required_for_architecture_freeze"])
            self.assertTrue((output / module.MANIFEST_FILENAME).is_file())
        finally:
            manifest = output / module.MANIFEST_FILENAME
            if manifest.is_file():
                manifest.unlink()
            if output.is_dir():
                output.rmdir()
            if root.is_dir():
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
