"""Tests for independent PF-1 condition manifest merging."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import merge_pf1_condition_manifests_v1 as subject
from most_t5_next.p1.run_pf1_four_grid_v1 import CONDITION_ORDER, REPORT_SCHEMA


def report_for(condition_id: str) -> dict[str, object]:
    condition_index = CONDITION_ORDER.index(condition_id)
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "pf1_one_percent_failure_screen_only",
        "interpretation": {"architecture_superiority_claim": False},
        "comparison_contract": {"same_update_budget": True},
        "data": {
            "train_members": 30240,
            "dev_members": 3360,
            "validated_record_cache": {
                "enabled": True,
                "entries": 33600,
                "expected_entries": 33600,
                "complete": True,
                "warmup_workers": 4,
            },
        },
        "optimization": {"effective_batch_size": 128},
        "precision": "bf16_autocast",
        "evaluation_updates": [0, 250, 500, 750, 1000],
        "checkpoint_updates": [500, 1000],
        "resumed_condition": None,
        "execution": {
            "requested_conditions": [condition_id],
            "complete_four_grid": False,
            "parallelizable_one_condition_per_process": True,
            "forward_seed": 20260807,
            "geometry_fusion_seed": 20260808,
            "num_e3fp_embeddings": 4096,
            "expected_vocab_size": 34666,
            "base_model_snapshot": "/model",
            "base_tokenizer_snapshot": "/tokenizer",
            "union_tokenizer_dir": "/union",
            "union_init_dir": "/init",
        },
        "conditions": [
            {
                "condition": condition_id,
                "optimizer_updates": 1000,
                "input_pipeline_telemetry": {
                    "validated_record_cache": {
                        "hits": 100000 + condition_index,
                        "warmup_seconds": 20.0 + condition_index,
                    }
                },
            }
        ],
    }


class MergePF1ConditionManifestsTest(unittest.TestCase):
    def write_reports(self, root: Path):
        paths = []
        for condition_id in CONDITION_ORDER:
            path = root / (condition_id + ".json")
            path.write_text(
                json.dumps(report_for(condition_id), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_merges_four_exact_conditions_in_frozen_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_reports(root)
            merged = subject.merge_pf1_condition_manifests(
                condition_manifests=tuple(reversed(paths)),
                output_dir=root / "merged",
            )
            self.assertEqual(
                [row["condition"] for row in merged["conditions"]],
                list(CONDITION_ORDER),
            )
            self.assertTrue(merged["execution"]["complete_four_grid"])
            self.assertTrue(
                merged["execution"]["merged_from_independent_condition_processes"]
            )
            self.assertEqual(
                [
                    row["input_pipeline_telemetry"]["validated_record_cache"][
                        "hits"
                    ]
                    for row in merged["conditions"]
                ],
                [100000 + index for index in range(len(CONDITION_ORDER))],
            )
            self.assertEqual(
                [
                    row["input_pipeline_telemetry"]["validated_record_cache"][
                        "warmup_seconds"
                    ]
                    for row in merged["conditions"]
                ],
                [20.0 + index for index in range(len(CONDITION_ORDER))],
            )
            self.assertTrue((root / "merged" / "pf1_training_manifest.json").is_file())

    def test_rejects_duplicate_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_reports(root)
            paths[-1].write_text(paths[0].read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(subject.PF1TrainingError, "duplicated"):
                subject.merge_pf1_condition_manifests(
                    condition_manifests=paths,
                    output_dir=root / "merged",
                )

    def test_rejects_shared_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_reports(root)
            altered = copy.deepcopy(report_for("M1"))
            altered["optimization"]["effective_batch_size"] = 64
            paths[-1].write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(subject.PF1TrainingError, "shared field optimization"):
                subject.merge_pf1_condition_manifests(
                    condition_manifests=paths,
                    output_dir=root / "merged",
                )


if __name__ == "__main__":
    unittest.main()
