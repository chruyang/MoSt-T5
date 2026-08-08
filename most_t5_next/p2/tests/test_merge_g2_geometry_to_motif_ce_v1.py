import copy
import unittest

from most_t5_next.p2.g1_deep_sets_geometry_fusion_v1 import FUSION_ID
from most_t5_next.p2.merge_g2_geometry_to_motif_ce_v1 import (
    G2MergeError,
    merge_g2,
)
from most_t5_next.p2.run_g2_geometry_to_motif_ce_v1 import (
    G2_OBJECTIVE_CONTRACT,
    G2_PROTOCOL_ID,
    REPORT_SCHEMA,
)


def _manifest(condition, nll, accuracy, sensitivity=None):
    row = {
        "condition": condition,
        "optimizer_updates": 1000,
        "members_seen": 128000,
        "train_encoder_nonpadding_tokens": 100,
        "train_supervised_target_tokens": 200,
        "final_data_cursor": {"next_epoch": 4, "next_batch_in_epoch": 108},
        "evaluations": [
            {
                "update": 1000,
                "token_weighted_nll": nll,
                "masked_token_accuracy": accuracy,
            }
        ],
        "final_e3fp_shuffle_diagnostic": sensitivity,
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "objective_contract": G2_OBJECTIVE_CONTRACT,
        "fusion_id": FUSION_ID,
        "data": {"mask_probability": 1.0},
        "optimization": {"protocol_id": G2_PROTOCOL_ID},
        "precision": "bf16_autocast",
        "evaluation_updates": [0, 250, 500, 750, 1000],
        "checkpoint_updates": [500, 1000],
        "conditions": [row],
    }


class G2MergeTest(unittest.TestCase):
    def test_paired_gate_accepts_useful_and_sensitive_geometry(self):
        control = _manifest("M0", 2.0, 0.50)
        geometry = _manifest(
            "M1", 1.8, 0.54, {"update": 1000, "delta_nll": 0.2}
        )
        report = merge_g2(control_manifest=control, geometry_manifest=geometry)
        self.assertTrue(report["scientific_gate_pass"])
        self.assertEqual(
            report["decision"], "accept_frozen_g1b_bridge_for_stage1_pretraining"
        )

    def test_pair_contract_mismatch_is_rejected(self):
        control = _manifest("M0", 2.0, 0.50)
        geometry = _manifest(
            "M1", 1.8, 0.54, {"update": 1000, "delta_nll": 0.2}
        )
        geometry = copy.deepcopy(geometry)
        geometry["data"]["mask_probability"] = 0.15
        with self.assertRaisesRegex(G2MergeError, "data"):
            merge_g2(control_manifest=control, geometry_manifest=geometry)


if __name__ == "__main__":
    unittest.main()
