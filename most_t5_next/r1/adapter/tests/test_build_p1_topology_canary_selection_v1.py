from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from most_t5_next.r1.adapter import build_p1_topology_canary_selection_v1 as builder
from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as runner


def member(ordinal: int, atom_count: int, sizes: tuple[int, ...]) -> builder.MemberFeatures:
    return builder.MemberFeatures(
        member_id=f"ogb_pcqm4mv2_train_row_index:{ordinal}",
        sdf_record_index=ordinal,
        model_atom_count=atom_count,
        motif_count=len(sizes),
        motif_group_sizes=sizes,
    )


def feature_fixture() -> list[builder.MemberFeatures]:
    result = []
    ordinal = 0
    templates = (
        (20, (20,)),
        (24, (3, 3, 3, 3, 3, 3, 3, 3)),
        (12, (4, 4, 4)),
        (18, (5, 5, 4, 4)),
        (20, (1, 1, 1, 1, 1, 1, 4, 4, 3, 3)),
        (30, (12, 6, 6, 6)),
    )
    for atom_count, sizes in templates:
        for _ in range(70):
            result.append(member(ordinal, atom_count, sizes))
            ordinal += 1
    for _ in range(100):
        result.append(member(ordinal, 24, (6, 6, 6, 6)))
        ordinal += 1
    return result


def build(features=None) -> dict:
    return builder.build_selection_document(
        features=feature_fixture() if features is None else features,
        release_binding={
            "release_id": "production-v2-fixture",
            "full_release_manifest_sha256": "a" * 64,
            "logical_release_root_sha256": "b" * 64,
        },
        source_scope={
            "shard_index": 0,
            "range_start": 0,
            "range_end": 25000,
            "admitted_record_count": 520,
            "shard_manifest_sha256": "c" * 64,
        },
        selection_id="p1-topology-canary-fixture-v1",
        seed="fixed-test-seed-v1",
    )


class BuildP1TopologyCanarySelectionTest(unittest.TestCase):
    def test_deterministic_exact_disjoint_selection_and_runner_compatibility(self):
        features = feature_fixture()
        first = build(features)
        second = build(list(reversed(features)))
        self.assertEqual(first, second)
        smoke = first["groups"]["smoke"]
        canary = first["groups"]["canary"]
        self.assertEqual(len(smoke), 32)
        self.assertEqual(len(canary), 256)
        selected = [row["sdf_record_index"] for row in smoke + canary]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected).issubset({item.sdf_record_index for item in features}))
        expected_smoke = sorted(
            features,
            key=lambda item: (
                hashlib.sha256(
                    f"fixed-test-seed-v1|smoke|{item.member_id}".encode("utf-8")
                ).hexdigest(),
                item.member_id,
            ),
        )[:32]
        self.assertEqual(
            [row["sdf_record_index"] for row in smoke],
            [item.sdf_record_index for item in expected_smoke],
        )
        self.assertEqual(first["selection_algorithm"]["algorithm_id"], builder.ALGORITHM_ID)
        self.assertEqual(
            sum(rule["quota"] for rule in first["selection_algorithm"]["canary"]["rules_in_order"]),
            256,
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            path.write_text(json.dumps(first), encoding="utf-8")
            _, parsed = runner.load_selection(path)
        self.assertEqual(len(parsed), 288)

    def test_every_boundary_rule_is_represented_in_canary_tags(self):
        document = build()
        observed = {
            tag
            for row in document["groups"]["canary"]
            for tag in row["selection_tags"]
        }
        expected = {rule["tag"] for rule in builder.BOUNDARY_RULES}
        self.assertTrue(expected.issubset(observed))
        for row in document["groups"]["canary"]:
            self.assertEqual(row["selection_tags"], sorted(set(row["selection_tags"])))

    def test_high_atom_rule_is_the_fixed_shard0_upper_boundary(self):
        rule = next(
            rule for rule in builder.BOUNDARY_RULES
            if rule["tag"] == "model_atom_count_ge_18"
        )
        self.assertEqual(rule["quota"], 40)
        self.assertEqual(rule["predicate"]["value"], 18)
        self.assertFalse(builder._matches(member(1, 17, (9, 8)), rule))
        self.assertTrue(builder._matches(member(2, 18, (9, 9)), rule))

    def test_fewer_than_288_admitted_members_is_rejected(self):
        with self.assertRaises(builder.SelectionBuildError):
            build(feature_fixture()[:287])

    def test_rejected_membership_never_loads_a_payload(self):
        admitted = {
            "disposition": "admit",
            "member_id": "ogb_pcqm4mv2_train_row_index:1",
            "sdf_record_index": 1,
            "record_storage_key": "000000001",
            "record_content_sha256": "d" * 64,
        }
        rejected = {
            "disposition": "reject",
            "member_id": "ogb_pcqm4mv2_train_row_index:2",
            "sdf_record_index": 2,
            "record_storage_key": None,
            "record_content_sha256": None,
        }
        loaded = []

        def payload_loader(row):
            loaded.append(row["sdf_record_index"])
            record = {
                "record_schema_version": runner.PRODUCTION_RECORD_SCHEMA,
                "member": {
                    "member_id": row["member_id"],
                    "sdf_record_index": row["sdf_record_index"],
                    "storage_key": row["record_storage_key"],
                },
                "atom_universe": {"model_atom_count": 4},
                "topology": {
                    "motif_count": 2,
                    "motif_atom_indices": [
                        np.asarray([0, 1], dtype=np.int32),
                        np.asarray([2, 3], dtype=np.int32),
                    ],
                },
            }
            return record, row["record_content_sha256"]

        features = builder.collect_admitted_features([rejected, admitted], payload_loader)
        self.assertEqual(loaded, [1])
        self.assertEqual([item.sdf_record_index for item in features], [1])


if __name__ == "__main__":
    unittest.main()
