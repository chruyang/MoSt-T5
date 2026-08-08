from __future__ import annotations

import copy
import math
import unittest

from most_t5_next.p2.merge_pf2_gated_fusion_v1 import (
    FGateMergeError,
    merge_pf2_gated_fusion,
)
from most_t5_next.p2.run_pf2_gated_fusion_v1 import (
    FUSION_CONTRACT,
    F_GATE_PROTOCOL_ID,
    REPORT_SCHEMA as CONDITION_SCHEMA,
)


def _trajectory(condition: str, final_logit: float):
    values = (0.0, 0.0) if condition == "M0" else (final_logit / 2, final_logit)
    return [
        {
            "update": update,
            "logit": logit,
            "effective_tanh_gate": math.tanh(logit),
        }
        for update, logit in zip((500, 1000), values)
    ]


def _manifest(condition: str, *, nll: float, accuracy: float, delta=None, logit=0.2):
    trajectory = _trajectory(condition, logit)
    return {
        "schema_version": CONDITION_SCHEMA,
        "status": "pass",
        "fusion_contract": FUSION_CONTRACT,
        "data": {"members": 10},
        "optimization": {
            "protocol_id": F_GATE_PROTOCOL_ID,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 2,
        },
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
            "members_seen": 127872,
            "train_encoder_nonpadding_tokens": 200,
            "train_supervised_target_tokens": 100,
            "final_data_cursor": {"next_epoch": 4, "next_batch_in_epoch": 108},
            "geometry_gate_trajectory": trajectory,
            "final_geometry_gate": trajectory[-1],
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


class PF2GatedFusionMergeTest(unittest.TestCase):
    def test_retains_gate_only_when_quality_and_sensitivity_pass(self):
        report = merge_pf2_gated_fusion(
            m0_manifest=_manifest("M0", nll=1.0, accuracy=0.70),
            m1_manifest=_manifest(
                "M1", nll=1.01, accuracy=0.695, delta=0.03
            ),
        )
        self.assertTrue(report["scientific_gate_pass"])
        self.assertEqual(
            report["decision"],
            "retain_f_gate_and_proceed_to_3d_sensitive_probe",
        )

    def test_routes_safe_but_insensitive_gate_to_t3mi(self):
        report = merge_pf2_gated_fusion(
            m0_manifest=_manifest("M0", nll=1.0, accuracy=0.70),
            m1_manifest=_manifest(
                "M1", nll=1.01, accuracy=0.695, delta=0.001
            ),
        )
        self.assertFalse(report["scientific_gate_pass"])
        self.assertEqual(report["decision"], "enter_pf2b_t3mi")

    def test_stops_scale_up_when_gate_still_degrades_ce(self):
        report = merge_pf2_gated_fusion(
            m0_manifest=_manifest("M0", nll=1.0, accuracy=0.70),
            m1_manifest=_manifest("M1", nll=1.04, accuracy=0.65, delta=0.03),
        )
        self.assertEqual(
            report["decision"],
            "stop_formal_training_and_revisit_geometry_interface",
        )

    def test_rejects_pair_drift_or_an_active_m0_gate(self):
        m0 = _manifest("M0", nll=1.0, accuracy=0.70)
        m1 = _manifest("M1", nll=1.0, accuracy=0.70, delta=0.03)
        drifted = copy.deepcopy(m1)
        drifted["optimization"]["micro_batch_size"] = 32
        with self.assertRaisesRegex(FGateMergeError, "optimization"):
            merge_pf2_gated_fusion(m0_manifest=m0, m1_manifest=drifted)

        active_m0 = copy.deepcopy(m0)
        active_m0["conditions"][0]["geometry_gate_trajectory"][1]["logit"] = 0.1
        active_m0["conditions"][0]["geometry_gate_trajectory"][1][
            "effective_tanh_gate"
        ] = math.tanh(0.1)
        active_m0["conditions"][0]["final_geometry_gate"] = copy.deepcopy(
            active_m0["conditions"][0]["geometry_gate_trajectory"][1]
        )
        with self.assertRaisesRegex(FGateMergeError, "M0 geometry gate"):
            merge_pf2_gated_fusion(m0_manifest=active_m0, m1_manifest=m1)


if __name__ == "__main__":
    unittest.main()
