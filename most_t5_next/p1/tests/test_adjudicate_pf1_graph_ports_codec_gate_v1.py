"""Tests for the preregistered GraphPorts codec adjudicator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import adjudicate_pf1_graph_ports_codec_gate_v1 as subject
from most_t5_next.p1.run_pf1_four_grid_v1 import REPORT_SCHEMA
from most_t5_next.p1.validate_pf1_graph_ports_codec_pair_v1 import (
    REPORT_SCHEMA as PAIR_REPORT_SCHEMA,
)


def _report(*, nll: float, accuracy: float, encoder_tokens: int, speed: float, memory: int):
    evaluations = []
    for update, scale in ((0, 50.0), (250, 5.0), (500, 1.5), (750, 1.02), (1000, 1.0)):
        evaluations.append(
            {
                "update": update,
                "members": 3360,
                "supervised_target_tokens": 23065,
                "encoder_nonpadding_tokens": encoder_tokens,
                "token_weighted_nll": nll * scale,
                "masked_token_accuracy": accuracy if update == 1000 else accuracy - 0.002,
            }
        )
    condition = {
        "condition": "M0",
        "optimizer_updates": 1000,
        "members_seen": 127872,
        "nominal_effective_batch_size": 128,
        "short_microbatches": 4,
        "min_microbatch_members": 32,
        "max_microbatch_members": 64,
        "mean_microbatch_members": 63.936,
        "min_members_per_update": 96,
        "max_members_per_update": 128,
        "mean_members_per_update": 127.872,
        "train_supervised_target_tokens": 830000,
        "final_data_cursor": {"next_epoch": 4, "next_batch_in_epoch": 108},
        "last_update_learning_rate": 1e-5,
        "members_per_second": speed,
        "peak_gpu_memory_bytes": memory,
        "evaluations": evaluations,
        "final_e3fp_shuffle_diagnostic": None,
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "pf1_one_percent_failure_screen_only",
        "interpretation": {"architecture_superiority_claim": False},
        "comparison_contract": {"same_update_budget": True},
        "data": {"train_members": 30240, "dev_members": 3360},
        "optimization": {"effective_batch_size": 128},
        "precision": "bf16_autocast",
        "evaluation_updates": [0, 250, 500, 750, 1000],
        "checkpoint_updates": [500, 1000],
        "execution": {
            "requested_conditions": ["M0"],
            "complete_four_grid": False,
            "forward_seed": 20260807,
            "union_tokenizer_dir": "/codec-specific-copy",
            "union_init_dir": "/same-init",
        },
        "conditions": [condition],
    }


def _pair_report():
    return {
        "schema_version": PAIR_REPORT_SCHEMA,
        "status": "pass",
        "decision_boundary": {"eligible_for_paired_m0_codec_screen": True},
    }


class GraphPortsCodecAdjudicationTest(unittest.TestCase):
    def _run(self, source, target):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        paths = {}
        for name, payload in (("pair", _pair_report()), ("source", source), ("target", target)):
            path = root / (name + ".json")
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths[name] = path
        return subject.adjudicate(
            pair_report=paths["pair"],
            source_manifest=paths["source"],
            target_manifest=paths["target"],
            output_report=root / "decision.json",
        )

    def test_promotes_shorter_candidate_inside_all_quality_and_cost_gates(self):
        source = _report(nll=1.0, accuracy=0.80, encoder_tokens=165287, speed=100, memory=20_000)
        target = _report(nll=1.01, accuracy=0.795, encoder_tokens=103322, speed=99, memory=20_500)
        result = self._run(source, target)
        self.assertEqual(result["decision"], "promote_graphports_v2")
        self.assertTrue(all(result["promotion_gates"].values()))

    def test_retains_v1_after_severe_quality_loss(self):
        source = _report(nll=1.0, accuracy=0.80, encoder_tokens=165287, speed=100, memory=20_000)
        target = _report(nll=1.08, accuracy=0.75, encoder_tokens=103322, speed=110, memory=19_000)
        result = self._run(source, target)
        self.assertEqual(result["decision"], "retain_graphports_v1")

    def test_rejects_target_exposure_mismatch(self):
        source = _report(nll=1.0, accuracy=0.80, encoder_tokens=165287, speed=100, memory=20_000)
        target = copy.deepcopy(source)
        target["conditions"][0]["train_supervised_target_tokens"] += 1
        with self.assertRaisesRegex(
            subject.PF1GraphPortsAdjudicationError,
            "exposure",
        ):
            self._run(source, target)


if __name__ == "__main__":
    unittest.main()
