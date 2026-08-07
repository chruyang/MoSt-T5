from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from most_t5_next.p1 import freeze_pf1_connectivity_sample_v1 as selection


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class PF1ConnectivitySelectionTests(unittest.TestCase):
    def test_closest_prefix_never_splits_a_group(self):
        groups = {"g0": 3, "g1": 4, "g2": 10}
        self.assertEqual(
            selection.closest_complete_prefix(tuple(groups), groups, 6),
            ("g0", "g1"),
        )
        self.assertEqual(
            selection.closest_complete_prefix(tuple(groups), groups, 4),
            ("g0",),
        )

    def test_pcg64_plan_is_deterministic_and_group_disjoint(self):
        sizes = {f"g{index}": 1 + (index % 3) for index in range(30)}
        first = selection.freeze_group_plan(
            sizes,
            seed=20260807,
            target_members=31,
            dev_fraction=0.10,
            benchmark_target_members=11,
            np=np,
        )
        second = selection.freeze_group_plan(
            sizes,
            seed=20260807,
            target_members=31,
            dev_fraction=0.10,
            benchmark_target_members=11,
            np=np,
        )
        self.assertEqual(first, second)
        train = set(first["train_groups"])
        dev = set(first["dev_groups"])
        self.assertFalse(train & dev)
        self.assertEqual(train | dev, set(first["selection_order"]))
        self.assertEqual(
            tuple(first["selection_order"][: len(first["benchmark_groups"])]),
            first["benchmark_groups"],
        )

    def test_support_filter_refills_from_unchanged_complete_group_order(self):
        sizes = {f"g{index}": 1 for index in range(30)}
        baseline = selection.freeze_group_plan(
            sizes,
            seed=20260807,
            target_members=10,
            dev_fraction=0.20,
            benchmark_target_members=4,
            np=np,
        )
        excluded = {
            baseline["selection_order"][2],
            baseline["selection_order"][7],
        }
        candidate = selection.freeze_group_plan(
            sizes,
            seed=20260807,
            target_members=10,
            dev_fraction=0.20,
            benchmark_target_members=4,
            ineligible_group_ids=excluded,
            np=np,
        )
        unchanged_order = tuple(baseline["complete_group_order"])
        expected = tuple(group for group in unchanged_order if group not in excluded)[:10]
        self.assertEqual(candidate["complete_group_order"], unchanged_order)
        self.assertEqual(candidate["selection_order"], expected)
        self.assertFalse(excluded & set(candidate["selection_order"]))
        self.assertEqual(
            set(candidate["train_groups"]) | set(candidate["dev_groups"]),
            set(candidate["selection_order"]),
        )
        self.assertFalse(
            set(candidate["train_groups"]) & set(candidate["dev_groups"])
        )

    def test_run_stream_joins_member_only_permitted_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permitted_path = root / "permitted.jsonl"
            identity_path = root / "identities.jsonl"
            output = root / "out"
            permitted_ids = [
                f"{selection.MEMBER_PREFIX}{index}" for index in range(12)
            ]
            _write_jsonl(
                permitted_path,
                (
                    {
                        "schema_version": selection.PERMITTED_SCHEMA,
                        "member_id": member_id,
                    }
                    for member_id in permitted_ids
                ),
            )
            group_by_ordinal = {
                0: "a",
                1: "a",
                2: "b",
                3: "c",
                4: "c",
                5: "d",
                6: "e",
                7: "f",
                8: "f",
                9: "g",
                10: "h",
                11: "i",
                12: "excluded",
            }
            _write_jsonl(
                identity_path,
                (
                    {
                        "schema_version": selection.IDENTITY_SCHEMA,
                        "member_id": f"{selection.MEMBER_PREFIX}{ordinal}",
                        "connectivity_identity_sha256": group_id,
                    }
                    for ordinal, group_id in group_by_ordinal.items()
                ),
            )
            args = argparse.Namespace(
                permitted_membership=str(permitted_path),
                identity_rows=str(identity_path),
                output_dir=str(output),
                seed=7,
                expected_permitted_members=12,
                target_members=8,
                dev_fraction=0.25,
                benchmark_target_members=4,
            )
            manifest = selection.run(args)
            membership = [
                json.loads(line)
                for line in (output / "membership.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            benchmark = [
                json.loads(line)
                for line in (output / "benchmark_membership.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(membership), manifest["selection"]["selected_member_count"])
            self.assertEqual(len(benchmark), manifest["benchmark"]["selected_member_count"])
            self.assertNotIn(f"{selection.MEMBER_PREFIX}12", {r["member_id"] for r in membership})
            group_splits = {}
            for row in membership:
                group_splits.setdefault(row["connectivity_identity_sha256"], set()).add(
                    row["split"]
                )
            self.assertTrue(all(len(values) == 1 for values in group_splits.values()))
            benchmark_groups = {row["connectivity_identity_sha256"] for row in benchmark}
            for group_id in benchmark_groups:
                self.assertEqual(
                    sum(r["connectivity_identity_sha256"] == group_id for r in benchmark),
                    sum(r["connectivity_identity_sha256"] == group_id for r in membership),
                )

    def test_support_aware_run_is_candidate_and_reports_group_complete_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            permitted_path = root / "permitted.jsonl"
            identity_path = root / "identities.jsonl"
            baseline_output = root / "baseline"
            candidate_output = root / "candidate"
            support_path = root / "ineligible.jsonl"
            permitted_ids = [
                f"{selection.MEMBER_PREFIX}{index}" for index in range(16)
            ]
            group_by_ordinal = {
                ordinal: f"g{ordinal // 2}" for ordinal in range(16)
            }
            _write_jsonl(
                permitted_path,
                (
                    {
                        "schema_version": selection.PERMITTED_SCHEMA,
                        "member_id": member_id,
                    }
                    for member_id in permitted_ids
                ),
            )
            _write_jsonl(
                identity_path,
                (
                    {
                        "schema_version": selection.IDENTITY_SCHEMA,
                        "member_id": f"{selection.MEMBER_PREFIX}{ordinal}",
                        "connectivity_identity_sha256": group_id,
                    }
                    for ordinal, group_id in group_by_ordinal.items()
                ),
            )
            common = dict(
                permitted_membership=str(permitted_path),
                identity_rows=str(identity_path),
                seed=11,
                expected_permitted_members=16,
                target_members=8,
                dev_fraction=0.25,
                benchmark_target_members=4,
            )
            selection.run(argparse.Namespace(output_dir=str(baseline_output), **common))
            baseline_rows = [
                json.loads(line)
                for line in (baseline_output / "membership.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            excluded_group = baseline_rows[0]["connectivity_identity_sha256"]
            evidence = next(
                row
                for row in baseline_rows
                if row["connectivity_identity_sha256"] == excluded_group
            )
            _write_jsonl(
                support_path,
                [
                    {
                        "schema_version": selection.INELIGIBLE_GROUP_SCHEMA,
                        "connectivity_identity_sha256": excluded_group,
                        "reason": "ATOM_SELFIES_SUPPORT_LIMIT",
                        "member_evidence": [
                            {
                                "member_id": evidence["member_id"],
                                "sdf_record_index": evidence["sdf_record_index"],
                                "reason": "STRICT_ROUNDTRIP_MISMATCH",
                            }
                        ],
                    }
                ],
            )
            manifest = selection.run(
                argparse.Namespace(
                    output_dir=str(candidate_output),
                    ineligible_connectivity_groups=str(support_path),
                    baseline_membership=str(baseline_output / "membership.jsonl"),
                    support_coverage_status="candidate",
                    research_role="fallback_candidate",
                    **common,
                )
            )
            candidate_rows = [
                json.loads(line)
                for line in (candidate_output / "membership.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(manifest["status"], "candidate")
            self.assertEqual(
                manifest["research_role"],
                {"role": "fallback_candidate", "enters_current_mainline": False},
            )
            self.assertNotIn(
                excluded_group,
                {row["connectivity_identity_sha256"] for row in candidate_rows},
            )
            self.assertEqual(manifest["support_domain"]["excluded_group_count"], 1)
            self.assertEqual(
                manifest["support_domain"]["excluded_permitted_member_count"], 2
            )
            self.assertTrue(
                manifest["support_domain"][
                    "newly_added_members_require_chemistry_screen"
                ]
            )
            delta = manifest["baseline_comparison"]
            self.assertEqual(delta["removed_group_count"], 1)
            self.assertEqual(delta["added_group_count"], 1)
            self.assertEqual(delta["removed_member_count"], 2)
            self.assertEqual(delta["added_member_count"], 2)
            group_splits = {}
            for row in candidate_rows:
                group_splits.setdefault(
                    row["connectivity_identity_sha256"], set()
                ).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))


if __name__ == "__main__":
    unittest.main()
