from __future__ import annotations

import copy
import unittest

from most_t5_next.p2.run_pf2_reference_fusion_v1 import (
    FUSION_CONTRACT,
    REPORT_SCHEMA as CONDITION_SCHEMA,
)
from most_t5_next.p2.merge_pf2_reference_fusion_v1 import (
    PF2MergeError,
    merge_pf2_reference_fusion,
)


def _manifest(condition: str, *, nll: float, accuracy: float, delta=None):
    return {
        "schema_version": CONDITION_SCHEMA,
        "status": "pass",
        "fusion_contract": FUSION_CONTRACT,
        "data": {"members": 10},
        "optimization": {"micro_batch_size": 63, "accumulation": 2},
        "precision": "bf16_autocast",
        "evaluation_updates": [0, 1000],
        "checkpoint_updates": [500, 1000],
        "execution": {
            "requested_conditions": [condition],
            "forward_seed": 1,
            "geometry_fusion_seed": 2,
            "num_e3fp_embeddings": 4096,
            "expected_vocab_size": 100,
            "base_model_snapshot": "model",
            "base_tokenizer_snapshot": "tokenizer",
            "union_tokenizer_dir": "union",
            "union_init_dir": "init",
        },
        "conditions": [{
            "condition": condition,
            "optimizer_updates": 1000,
            "members_seen": 128000,
            "train_encoder_nonpadding_tokens": 200,
            "train_supervised_target_tokens": 100,
            "final_data_cursor": {"next_epoch": 4, "next_batch_in_epoch": 1},
            "evaluations": [{
                "update": 1000,
                "members": 10,
                "encoder_nonpadding_tokens": 20,
                "supervised_target_tokens": 10,
                "token_weighted_nll": nll,
                "masked_token_accuracy": accuracy,
            }],
            "final_e3fp_shuffle_diagnostic": (
                None if delta is None else {"update": 1000, "delta_nll": delta}
            ),
        }],
    }


def _initial(delta: float = 0.18):
    return {
        "status": "pass",
        "fusion_id": FUSION_CONTRACT["fusion_id"],
        "optimizer_updates": 0,
        "practical_final_gate": max(0.01, 0.1 * delta),
        "initial_sensitivity": {"update": 0, "delta_nll": delta},
    }


class PF2ReferenceFusionMergeTest(unittest.TestCase):
    def test_passes_only_when_all_three_practical_gates_pass(self):
        report = merge_pf2_reference_fusion(
            m0_manifest=_manifest("M0", nll=1.0, accuracy=0.70),
            m1_manifest=_manifest("M1", nll=1.01, accuracy=0.695, delta=0.03),
            initial_sensitivity=_initial(),
        )
        self.assertTrue(report["scientific_gate_pass"])
        self.assertEqual(
            report["decision"],
            "retain_f_ref_and_proceed_to_3d_sensitive_probe",
        )

    def test_routes_non_degraded_but_insensitive_fusion_to_t3mi(self):
        report = merge_pf2_reference_fusion(
            m0_manifest=_manifest("M0", nll=1.0, accuracy=0.70),
            m1_manifest=_manifest("M1", nll=1.01, accuracy=0.695, delta=0.001),
            initial_sensitivity=_initial(),
        )
        self.assertFalse(report["scientific_gate_pass"])
        self.assertEqual(report["decision"], "enter_pf2b_t3mi")

    def test_routes_degraded_fusion_to_independent_gate(self):
        report = merge_pf2_reference_fusion(
            m0_manifest=_manifest("M0", nll=1.0, accuracy=0.70),
            m1_manifest=_manifest("M1", nll=1.04, accuracy=0.65, delta=0.03),
            initial_sensitivity=_initial(),
        )
        self.assertEqual(report["decision"], "test_independent_f_gate_before_pf2b")

    def test_rejects_any_paired_contract_drift(self):
        m0 = _manifest("M0", nll=1.0, accuracy=0.70)
        m1 = _manifest("M1", nll=1.0, accuracy=0.70, delta=0.03)
        drifted = copy.deepcopy(m1)
        drifted["optimization"]["micro_batch_size"] = 32
        with self.assertRaisesRegex(PF2MergeError, "optimization"):
            merge_pf2_reference_fusion(
                m0_manifest=m0,
                m1_manifest=drifted,
                initial_sensitivity=_initial(),
            )


if __name__ == "__main__":
    unittest.main()
