from __future__ import print_function

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import derive_clean_pretrain_membership_v1 as derive
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_bytes(proof.canonical_json_bytes(value) + b"\n")


def write_rows(path, rows):
    rows = sorted(rows, key=lambda row: row["member_id"].encode("utf-8"))
    raw = b"".join(proof.canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    key_digest = hashlib.sha256()
    for row in rows:
        key_digest.update(row["member_id"].encode("utf-8") + b"\n")
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "key_lf_sha256": key_digest.hexdigest(),
    }


def make_collection(root, collection_id, role, split, task_family, specs):
    directory = root / collection_id
    directory.mkdir(parents=True)
    rows = [
        {
            "schema_version": proof.MOLECULE_ROW_SCHEMA,
            "collection_id": collection_id,
            "member_id": member_id,
            "connectivity_identity_sha256": digest("connectivity:" + connectivity),
            "stereo_identity_sha256": digest("stereo:" + stereo),
            "conformer_identity_sha256": None,
        }
        for member_id, connectivity, stereo in specs
    ]
    molecule_artifact = write_rows(directory / "molecules.jsonl", rows)
    phase = "p1" if role == "p1_structure_train" else "downstream"
    manifest = {
        "schema_version": proof.COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "dataset_id": "fixture-dataset-" + collection_id,
        "release_id": "fixture-release-" + collection_id,
        "phase": phase,
        "split": split,
        "role": role,
        "task_family": task_family,
        "identity_specs": {
            "connectivity_identity_spec_sha256": digest("shared-connectivity-spec"),
            "stereo_identity_spec_sha256": digest("shared-stereo-spec"),
            "conformer_identity": {"status": "unavailable", "spec_sha256": None},
            "text_identity": {
                "status": "unavailable",
                "exact_spec_sha256": None,
                "normalized_spec_sha256": None,
            },
        },
        "molecule_rows": molecule_artifact,
        "text_pair_rows": None,
        "provenance": {
            "source_identity_namespace": "fixture:" + collection_id,
            "source_release_manifest_sha256": digest("source:" + collection_id),
            "extractor_sha256": digest("fixture-extractor"),
            "excluded_source_metadata_keys": [],
        },
    }
    manifest_path = directory / "collection_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, proof.sha256_file(manifest_path)[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CleanMembershipDerivationTests(unittest.TestCase):
    def make_world(self, root):
        pretrain = make_collection(
            root,
            "pretrain",
            "p1_structure_train",
            "train",
            "none",
            [
                ("pretrain:4", "D", "report-only-stereo"),
                ("pretrain:2", "B", "B-pretrain"),
                ("pretrain:1", "A", "A-pretrain"),
                ("pretrain:3", "C", "C-pretrain"),
            ],
        )
        validation = make_collection(
            root,
            "task-x-validation",
            "downstream_validation",
            "validation",
            "task_x",
            [
                ("validation:1", "B", "B-validation-1"),
                ("validation:2", "B", "B-validation-2"),
                ("validation:3", "X", "report-only-stereo"),
            ],
        )
        test = make_collection(
            root,
            "task-y-test",
            "downstream_test",
            "test",
            "task_y",
            [
                ("test:1", "B", "B-test"),
                ("test:2", "C", "C-test"),
            ],
        )
        return pretrain, validation, test

    def run_world(self, protected_order=(1, 2)):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        refs = self.make_world(root)
        protected = [refs[index] for index in protected_order]
        output = root / "derived"
        manifest = derive.derive_clean_membership(
            refs[0][0], [path for path, _ in protected], output
        )
        return root, output, manifest

    def test_connectivity_union_partitions_members_and_deduplicates_protected_identity(self):
        _, output, manifest = self.run_world()
        permitted = read_jsonl(output / derive.PERMITTED_FILENAME)
        excluded = read_jsonl(output / derive.EXCLUDED_FILENAME)

        self.assertEqual([row["member_id"] for row in permitted], ["pretrain:1", "pretrain:4"])
        self.assertEqual([row["member_id"] for row in excluded], ["pretrain:2", "pretrain:3"])
        self.assertEqual(
            excluded[0]["matched_protected_collections"],
            [
                {
                    "collection_id": "task-x-validation",
                    "split": "validation",
                    "task_family": "task_x",
                },
                {
                    "collection_id": "task-y-test",
                    "split": "test",
                    "task_family": "task_y",
                },
            ],
        )
        self.assertEqual(len(excluded[1]["matched_protected_collections"]), 1)
        self.assertEqual(manifest["counts"]["pretrain_member_count"], 4)
        self.assertEqual(manifest["counts"]["permitted_member_count"], 2)
        self.assertEqual(manifest["counts"]["excluded_member_count"], 2)
        self.assertEqual(
            manifest["counts"]["excluded_members_matching_multiple_protected_collections"], 1
        )

    def test_stereo_overlap_is_reported_but_does_not_exclude(self):
        _, output, manifest = self.run_world()
        permitted_ids = [row["member_id"] for row in read_jsonl(output / derive.PERMITTED_FILENAME)]
        self.assertIn("pretrain:4", permitted_ids)
        validation = next(
            item
            for item in manifest["protected_collections"]
            if item["source"]["collection_id"] == "task-x-validation"
        )
        self.assertFalse(validation["report_only_stereo"]["used_for_exclusion"])
        self.assertEqual(
            validation["report_only_stereo"]["counts"]["pretrain_members_impacted"], 1
        )
        self.assertFalse(validation["report_only_text"]["used_for_exclusion"])
        self.assertEqual(validation["report_only_text"]["status"], "manifest_metadata_only")

    def test_output_is_deterministic_across_protected_argument_order(self):
        _, first_output, first_manifest = self.run_world((1, 2))
        first_bytes = {
            name: (first_output / name).read_bytes()
            for name in (derive.PERMITTED_FILENAME, derive.EXCLUDED_FILENAME, derive.MANIFEST_FILENAME)
        }
        _, second_output, second_manifest = self.run_world((2, 1))
        second_bytes = {
            name: (second_output / name).read_bytes()
            for name in (derive.PERMITTED_FILENAME, derive.EXCLUDED_FILENAME, derive.MANIFEST_FILENAME)
        }
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(
            first_manifest["derivation_binding_sha256"], second_manifest["derivation_binding_sha256"]
        )

    def test_artifact_and_manifest_hashes_are_recomputable(self):
        _, output, manifest = self.run_world()
        for key in ("permitted_member_ids", "excluded_member_ledger"):
            artifact = manifest["artifacts"][key]
            size, observed_sha256 = proof.sha256_file(output / artifact["path"])
            self.assertEqual((size, observed_sha256), (artifact["bytes"], artifact["sha256"]))
        payload = dict(manifest)
        expected_payload_sha256 = payload.pop("manifest_canonical_payload_sha256")
        self.assertEqual(
            proof.sha256_bytes(proof.canonical_json_bytes(payload)), expected_payload_sha256
        )
        self.assertTrue(manifest["release_handling"]["source_releases_preserved"])
        self.assertFalse(manifest["release_handling"]["molecule_payload_copied"])
        self.assertFalse(
            manifest["provenance_observation_policy"][
                "caller_supplied_digest_required"
            ]
        )

    def test_cli_accepts_manifest_paths_without_digest_arguments(self):
        parser = derive.build_parser()
        args = parser.parse_args(
            [
                "--pretrain-manifest",
                "pretrain.json",
                "--protected-manifest",
                "validation.json",
                "--protected-manifest",
                "test.json",
                "--output-dir",
                "derived",
            ]
        )
        self.assertEqual(args.pretrain_manifest, "pretrain.json")
        self.assertEqual(args.protected_manifest, ["validation.json", "test.json"])


if __name__ == "__main__":
    unittest.main()
