from __future__ import print_function

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as gate


R1_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = R1_ROOT / "contracts" / "p1_p2_downstream_overlap_proof_contract_v1.json"


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path, value):
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha(path):
    return gate.sha256_file(path)[1]


def write_rows(path, rows, key_name):
    rows = sorted(rows, key=lambda row: row[key_name].encode("utf-8"))
    raw = b"".join(gate.canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    key_hash = hashlib.sha256()
    for row in rows:
        key_hash.update(row[key_name].encode("utf-8") + b"\n")
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "key_lf_sha256": key_hash.hexdigest(),
    }


def make_collection(
    root,
    collection_id,
    role,
    split,
    task_family,
    molecule_specs,
    text_specs=None,
    connectivity_spec=None,
    stereo_spec=None,
):
    directory = root / collection_id
    directory.mkdir()
    connectivity_spec = connectivity_spec or digest("shared-connectivity-spec")
    stereo_spec = stereo_spec or digest("shared-stereo-spec")
    molecule_rows = []
    molecule_by_id = {}
    conformer_available = all(item[3] is not None for item in molecule_specs)
    for member_id, connectivity_label, stereo_label, conformer_label in molecule_specs:
        row = {
            "schema_version": gate.MOLECULE_ROW_SCHEMA,
            "collection_id": collection_id,
            "member_id": member_id,
            "connectivity_identity_sha256": digest("connectivity:" + connectivity_label),
            "stereo_identity_sha256": digest("stereo:" + stereo_label),
            "conformer_identity_sha256": digest("conformer:" + conformer_label) if conformer_label is not None else None,
        }
        molecule_rows.append(row)
        molecule_by_id[member_id] = row
    molecule_artifact = write_rows(directory / "molecules.jsonl", molecule_rows, "member_id")
    text_artifact = None
    if text_specs is not None:
        text_rows = []
        for pair_id, member_id, exact_label, normalized_label in text_specs:
            molecule = molecule_by_id[member_id]
            normalized_hash = digest("text-normalized:" + normalized_label)
            text_rows.append(
                {
                    "schema_version": gate.TEXT_ROW_SCHEMA,
                    "collection_id": collection_id,
                    "pair_id": pair_id,
                    "member_id": member_id,
                    "task_family": task_family,
                    "text_exact_sha256": digest("text-exact:" + exact_label),
                    "text_normalized_sha256": normalized_hash,
                    "connectivity_text_pair_sha256": gate.pair_digest(
                        "most-t5-r1/connectivity-text-pair/v1",
                        molecule["connectivity_identity_sha256"],
                        normalized_hash,
                    ),
                    "stereo_text_pair_sha256": gate.pair_digest(
                        "most-t5-r1/stereo-text-pair/v1",
                        molecule["stereo_identity_sha256"],
                        normalized_hash,
                    ),
                }
            )
        text_artifact = write_rows(directory / "text_pairs.jsonl", text_rows, "pair_id")
    phase = "p1" if role == "p1_structure_train" else ("p2" if role.startswith("p2_") else "downstream")
    manifest = {
        "schema_version": gate.COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "dataset_id": "fixture-dataset-" + collection_id,
        "release_id": "fixture-release-v1",
        "phase": phase,
        "split": split,
        "role": role,
        "task_family": task_family,
        "identity_specs": {
            "connectivity_identity_spec_sha256": connectivity_spec,
            "stereo_identity_spec_sha256": stereo_spec,
            "conformer_identity": {
                "status": "available" if conformer_available else "unavailable",
                "spec_sha256": digest("shared-conformer-spec") if conformer_available else None,
            },
            "text_identity": {
                "status": "available" if text_specs is not None else "unavailable",
                "exact_spec_sha256": digest("shared-text-exact-spec") if text_specs is not None else None,
                "normalized_spec_sha256": digest("shared-text-normalized-spec") if text_specs is not None else None,
            },
        },
        "molecule_rows": molecule_artifact,
        "text_pair_rows": text_artifact,
        "provenance": {
            "source_identity_namespace": "fixture_namespace_" + collection_id,
            "source_release_manifest_sha256": digest("source-release:" + collection_id),
            "extractor_sha256": digest("fixture-extractor"),
            "excluded_source_metadata_keys": ["__len__"] if role == "p2_alignment_train" else [],
        },
    }
    manifest_path = directory / "collection_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def comparison(comparison_id, left, right, relationship, text_bearing=False, p1_p2=False):
    required = ["connectivity_identity"]
    reported = ["stereo_identity", "conformer_identity"]
    if text_bearing:
        required.append("connectivity_text_pair")
        reported.extend(("text_exact", "text_normalized", "stereo_text_pair"))
    return {
        "comparison_id": comparison_id,
        "left_collection_id": left,
        "right_collection_id": right,
        "relationship": relationship,
        "policy": "disjoint_required" if not p1_p2 else "disjoint_required",
        "required_zero": required,
        "report_only": reported,
    }


def build_world(root, overrides=None):
    overrides = overrides or {}
    specs = {
        "p1": [("p1:1", "A", "A1", "A-conf")],
        "p2": [("p2:1", "B", "B1", "B-conf")],
        "dtrain": [("down:train:1", "C", "C1", "C-conf")],
        "dvalid": [("down:valid:1", "D", "D1", "D-conf")],
        "dtest": [("down:test:1", "E", "E1", "E-conf")],
    }
    specs.update(overrides.get("molecules", {}))
    text = {
        "dtrain": [("dtrain-pair-1", specs["dtrain"][0][0], "train exact", "train normalized")],
        "dvalid": [("dvalid-pair-1", specs["dvalid"][0][0], "valid exact", "valid normalized")],
        "dtest": [("dtest-pair-1", specs["dtest"][0][0], "test exact", "test normalized")],
    }
    text.update(overrides.get("texts", {}))
    manifests = {
        "p1": make_collection(root, "p1", "p1_structure_train", "train", "none", specs["p1"], None),
        "p2": make_collection(root, "p2", "p2_permitted_train_membership", "train", "none", specs["p2"], None),
        "dtrain": make_collection(root, "dtrain", "downstream_train", "train", "task_x", specs["dtrain"], text["dtrain"]),
        "dvalid": make_collection(
            root,
            "dvalid",
            "downstream_validation",
            "validation",
            "task_x",
            specs["dvalid"],
            text["dvalid"],
            connectivity_spec=overrides.get("dvalid_connectivity_spec"),
        ),
        "dtest": make_collection(root, "dtest", "downstream_test", "test", "task_x", specs["dtest"], text["dtest"]),
    }
    comparisons = [comparison("p1-p2", "p1", "p2", "p1_to_p2", p1_p2=True)]
    for pretrain in ("p1", "p2"):
        for downstream in ("dvalid", "dtest"):
            comparisons.append(
                comparison(
                    pretrain + "-" + downstream,
                    pretrain,
                    downstream,
                    "pretrain_to_downstream_eval",
                    text_bearing=False,
                )
            )
    for left, right in (("dtrain", "dvalid"), ("dtrain", "dtest"), ("dvalid", "dtest")):
        comparisons.append(comparison(left + "-" + right, left, right, "downstream_within_task_split", text_bearing=True))
    request = {
        "schema_version": gate.REQUEST_SCHEMA,
        "request_id": "fixture-overlap-proof-v1",
        "contract_sha256": file_sha(CONTRACT),
        "collections": [
            {"manifest_path": str(path), "manifest_sha256": file_sha(path)}
            for _, path in sorted(manifests.items())
        ],
        "comparisons": comparisons,
        "coverage": {
            "required_collection_roles": [
                "p1_structure_train",
                "p2_permitted_train_membership",
                "downstream_train",
                "downstream_validation",
                "downstream_test",
            ],
            "required_downstream_task_splits": [
                {"task_family": "task_x", "splits": ["train", "validation", "test"]}
            ],
            "downstream_eval_splits": ["validation", "test"],
            "require_p1_p2_comparison": True,
            "require_each_pretrain_vs_each_downstream_eval": True,
            "require_within_task_split_comparisons": True,
        },
    }
    request_path = root / "request.json"
    write_json(request_path, request)
    return request_path


class OverlapProofGateTests(unittest.TestCase):
    def run_world(self, overrides=None, mutate_request=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request_path = build_world(root, overrides)
        if mutate_request is not None:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            mutate_request(request)
            write_json(request_path, request)
        return gate.run_proof(CONTRACT, request_path, root / "report")

    def test_disjoint_complete_scope_passes(self):
        report = self.run_world()
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["coverage"]["passed"])
        self.assertFalse(report["p1_training_admission"])
        self.assertTrue(all(item["passed"] for item in report["comparisons"]))

    def test_connectivity_overlap_catches_different_stereo_and_conformer(self):
        report = self.run_world(
            {
                "molecules": {
                    "p2": [
                        ("p2:1", "B", "B1", "B-conf-1"),
                        ("p2:2", "B", "B1", "B-conf-2"),
                    ],
                    "dvalid": [("down:valid:1", "B", "B-other-stereo", "B-conf-3")],
                }
            }
        )
        self.assertEqual(report["status"], "fail")
        target = next(item for item in report["comparisons"] if item["comparison_id"] == "p2-dvalid")
        self.assertEqual(target["dimensions"]["connectivity_identity"]["counts"]["overlap_unique_count"], 1)
        self.assertEqual(target["dimensions"]["stereo_identity"]["counts"]["overlap_unique_count"], 0)
        self.assertEqual(target["cross_resolution"]["right_members_connectivity_overlap_without_stereo_match"], 1)
        p2_summary = next(item["summary"] for item in report["collections"] if item["summary"]["collection_id"] == "p2")
        self.assertEqual(p2_summary["multi_conformer_stereo_group_count"], 1)

    def test_normalized_text_overlap_can_be_required_independently(self):
        def mutate(request):
            target = next(item for item in request["comparisons"] if item["comparison_id"] == "dtrain-dvalid")
            target["report_only"].remove("text_normalized")
            target["required_zero"].append("text_normalized")

        report = self.run_world(
            {"texts": {"dvalid": [("dvalid-pair-1", "down:valid:1", "different exact", "train normalized")]}},
            mutate,
        )
        self.assertEqual(report["status"], "fail")
        target = next(item for item in report["comparisons"] if item["comparison_id"] == "dtrain-dvalid")
        self.assertEqual(target["dimensions"]["text_exact"]["counts"]["overlap_unique_count"], 0)
        self.assertEqual(target["dimensions"]["text_normalized"]["counts"]["overlap_unique_count"], 1)

    def test_identity_spec_mismatch_cannot_prove_required_zero(self):
        report = self.run_world({"dvalid_connectivity_spec": digest("different-connectivity-spec")})
        self.assertEqual(report["status"], "fail")
        target = next(item for item in report["comparisons"] if item["comparison_id"] == "p1-dvalid")
        self.assertEqual(target["dimensions"]["connectivity_identity"]["status"], "unavailable")
        self.assertTrue(any("required dimension connectivity_identity is unavailable" in item for item in target["violations"]))

    def test_missing_declared_downstream_split_fails_scope_coverage(self):
        def mutate(request):
            request["coverage"]["required_downstream_task_splits"][0]["splits"].append("external_test")

        report = self.run_world(mutate_request=mutate)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["coverage"]["passed"])
        self.assertTrue(any("task/split task_x/external_test" in item for item in report["coverage"]["errors"]))

    def test_geometry_replay_requires_each_downstream_eval_comparison(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request_path = build_world(root)
        replay_manifest = make_collection(
            root,
            "replay",
            "p2_geometry_replay_train",
            "train",
            "geometry_replay",
            [("replay:1", "R", "R1", "R-conf")],
            None,
        )
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["collections"].append(
            {"manifest_path": str(replay_manifest), "manifest_sha256": file_sha(replay_manifest)}
        )
        write_json(request_path, request)

        missing = gate.run_proof(CONTRACT, request_path, root / "report-missing")
        self.assertEqual(missing["status"], "fail")
        self.assertFalse(missing["coverage"]["passed"])
        self.assertTrue(
            any("missing pretrain_to_downstream_eval comparison replay vs dvalid" in item for item in missing["coverage"]["errors"])
        )
        self.assertTrue(
            any("missing pretrain_to_downstream_eval comparison replay vs dtest" in item for item in missing["coverage"]["errors"])
        )

        request["comparisons"].extend(
            [
                comparison("replay-dvalid", "replay", "dvalid", "pretrain_to_downstream_eval"),
                comparison("replay-dtest", "replay", "dtest", "pretrain_to_downstream_eval"),
            ]
        )
        write_json(request_path, request)
        complete = gate.run_proof(CONTRACT, request_path, root / "report-complete")
        self.assertEqual(complete["status"], "pass")
        self.assertTrue(complete["coverage"]["passed"])

    def test_manifest_hash_mismatch_is_rejected_before_set_proof(self):
        def mutate(request):
            request["collections"][0]["manifest_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "manifest SHA-256 differs"):
            self.run_world(mutate_request=mutate)

    def test_text_pair_must_reference_a_member_and_recompute_pair_hash(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        request_path = build_world(root)
        manifest_path = root / "dtrain" / "collection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row_path = root / "dtrain" / "text_pairs.jsonl"
        row = json.loads(row_path.read_text(encoding="utf-8").strip())
        row["connectivity_text_pair_sha256"] = "f" * 64
        manifest["text_pair_rows"] = write_rows(row_path, [row], "pair_id")
        write_json(manifest_path, manifest)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        for ref in request["collections"]:
            if Path(ref["manifest_path"]) == manifest_path:
                ref["manifest_sha256"] = file_sha(manifest_path)
        write_json(request_path, request)
        with self.assertRaisesRegex(ValueError, "pair digest is not derivable"):
            gate.run_proof(CONTRACT, request_path, root / "report")


if __name__ == "__main__":
    unittest.main()
