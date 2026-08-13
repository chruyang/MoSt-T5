from __future__ import annotations

import unittest

import torch

from most_t5_next.p2.factorized_motif_t5_v2 import FactorizedMotifT5V2
from most_t5_next.p2.run_pf10_factorized_v2_smoke_v1 import (
    B_OBJECTIVE_SCHEDULE,
    PF10FactorizedV2SmokeError,
    run_pf10_factorized_v2_smoke_stage,
)
from most_t5_next.p2.tests.test_run_pf10_factorized_smoke_v1 import (
    PF10FactorizedSmokeRunnerTest,
    _B2DProvider,
    _TinyT5,
    _workspace_test_directory,
)


class _AddressProvider:
    def get(self, record):
        return (0, 1)


class PF10FactorizedV2SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        PF10FactorizedSmokeRunnerTest.setUpClass()
        cls.records = PF10FactorizedSmokeRunnerTest.records
        cls.tokenizer = PF10FactorizedSmokeRunnerTest.tokenizer
        cls.addresses = _AddressProvider()

    @staticmethod
    def model(initial_state=None):
        model = FactorizedMotifT5V2(
            _TinyT5(),
            num_e3fp_embeddings=4096,
            state_embedding_dim=4,
            atom_memory_dim=8,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
        )
        if initial_state is not None:
            model.load_state_dict(initial_state, strict=True)
        return model

    def test_f3d_s_then_frozen_encoder_bridge_stage(self) -> None:
        torch.manual_seed(311)
        initial = self.model()
        initial_state = {
            name: value.detach().clone() for name, value in initial.state_dict().items()
        }
        with _workspace_test_directory() as directory:
            s_model = self.model(initial_state)
            s_report = run_pf10_factorized_v2_smoke_stage(
                cell="F3D",
                stage="S",
                records=self.records,
                tokenizer=self.tokenizer,
                model=s_model,
                atom_address_provider=self.addresses,
                output_dir=directory / "S",
            )
            self.assertEqual(s_report["objective_schedule"], ["state"] * 3)
            self.assertIn("v2-carrier-only", s_report["factorisation_id"])
            self.assertTrue(s_report["state_causal_diagnostic"]["same_targets"])

            b_model = self.model(initial_state)
            b_report = run_pf10_factorized_v2_smoke_stage(
                cell="F3D",
                stage="B",
                records=self.records,
                tokenizer=self.tokenizer,
                model=b_model,
                atom_address_provider=self.addresses,
                output_dir=directory / "B",
                s_checkpoint=directory / "S" / "s_stage_checkpoint.pt",
            )
            self.assertEqual(
                b_report["objective_schedule"],
                list(B_OBJECTIVE_SCHEDULE),
            )
            self.assertEqual(b_report["optimizer_updates"], 4)
            self.assertGreater(b_report["geometry_gate"]["mean"], 0.0)
            self.assertIn(
                "zero_minus_aligned",
                b_report["state_causal_diagnostic"],
            )
            self.assertIn(
                "zero_minus_aligned",
                b_report["identity_causal_diagnostic"],
            )
            s_state = s_model.state_dict()
            b_state = b_model.state_dict()
            for name in s_state:
                if name.startswith("adapter.") or name.startswith("t5.encoder."):
                    self.assertTrue(torch.equal(s_state[name], b_state[name]), name)
            self.assertTrue(any(
                not torch.equal(s_state[name], b_state[name])
                for name in s_state
                if name.startswith("t5.decoder.")
            ))

    def test_b2d_uses_the_same_v2_state_path(self) -> None:
        with _workspace_test_directory() as directory:
            report = run_pf10_factorized_v2_smoke_stage(
                cell="B2D",
                stage="S",
                records=self.records,
                tokenizer=self.tokenizer,
                model=self.model(),
                atom_address_provider=self.addresses,
                atom_state_provider=_B2DProvider(),
                output_dir=directory / "S",
            )
            self.assertEqual(report["state_kind"], "mock_morgan_r3_4096")

    def test_bridge_cannot_skip_or_cross_the_v2_s_boundary(self) -> None:
        with _workspace_test_directory() as directory:
            with self.assertRaisesRegex(PF10FactorizedV2SmokeError, "requires an S"):
                run_pf10_factorized_v2_smoke_stage(
                    cell="F3D",
                    stage="B",
                    records=self.records,
                    tokenizer=self.tokenizer,
                    model=self.model(),
                    atom_address_provider=self.addresses,
                    output_dir=directory / "B",
                )


if __name__ == "__main__":
    unittest.main()
