from __future__ import print_function

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import derive_downstream_connectivity_clean_view_v1 as clean
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_bytes(clean.canonical_json_bytes(value) + b"\n")


def write_rows(path, rows, key_name):
    rows = sorted(rows, key=lambda row: row[key_name].encode("utf-8"))
    raw = b"".join(clean.canonical_json_bytes(row) + b"\n" for row in rows)
    key_digest = hashlib.sha256()
    for row in rows:
        key_digest.update(row[key_name].encode("utf-8") + b"\n")
    path.write_bytes(raw)
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "key_lf_sha256": key_digest.hexdigest(),
    }


def make_collection(root, split, molecules, text=True, spec_label="shared"):
    directory = root / split
    directory.mkdir()
    collection_id = "fixture-{}-reported-v1".format(split)
    molecule_rows = []
    by_member = {}
    for member_id, connectivity_label, stereo_label in molecules:
        row = {
            "schema_version": proof.MOLECULE_ROW_SCHEMA,
            "collection_id": collection_id,
            "member_id": member_id,
            "connectivity_identity_sha256": digest("connectivity:" + connectivity_label),
            "stereo_identity_sha256": digest("stereo:" + stereo_label),
            "conformer_identity_sha256": None,
        }
        molecule_rows.append(row)
        by_member[member_id] = row
    molecule_declaration = write_rows(
        directory / "molecules.jsonl", molecule_rows, "member_id"
    )
    text_declaration = None
    if text:
        text_rows = []
        for member_id in sorted(by_member, key=lambda value: value.encode("utf-8")):
            pair_total = 2 if split == "test" and member_id == "test-a" else 1
            for pair_index in range(pair_total):
                pair_id = "pair:{}:{}".format(member_id, pair_index)
                normalized = digest("normalized:{}:{}".format(member_id, pair_index))
                molecule = by_member[member_id]
                text_rows.append(
                    {
                        "schema_version": proof.TEXT_ROW_SCHEMA,
                        "collection_id": collection_id,
                        "pair_id": pair_id,
                        "member_id": member_id,
                        "task_family": "fixture-property-prediction",
                        "text_exact_sha256": digest(
                            "exact:{}:{}".format(member_id, pair_index)
                        ),
                        "text_normalized_sha256": normalized,
                        "connectivity_text_pair_sha256": proof.pair_digest(
                            "most-t5-r1/connectivity-text-pair/v1",
                            molecule["connectivity_identity_sha256"],
                            normalized,
                        ),
                        "stereo_text_pair_sha256": proof.pair_digest(
                            "most-t5-r1/stereo-text-pair/v1",
                            molecule["stereo_identity_sha256"],
                            normalized,
                        ),
                    }
                )
        text_declaration = write_rows(
            directory / "text_pairs.jsonl", text_rows, "pair_id"
        )
    manifest = {
        "schema_version": proof.COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "dataset_id": "fixture-downstream-dataset",
        "release_id": "fixture-release-v1",
        "phase": "downstream",
        "split": split,
        "role": clean.EXPECTED_ROLES[split],
        "task_family": "fixture-property-prediction",
        "identity_specs": {
            "connectivity_identity_spec_sha256": digest(
                "connectivity-spec:" + spec_label
            ),
            "stereo_identity_spec_sha256": digest("stereo-spec:" + spec_label),
            "conformer_identity": {"status": "unavailable", "spec_sha256": None},
            "text_identity": {
                "status": "available" if text else "unavailable",
                "exact_spec_sha256": digest("text-exact-spec") if text else None,
                "normalized_spec_sha256": (
                    digest("text-normalized-spec") if text else None
                ),
            },
        },
        "molecule_rows": molecule_declaration,
        "text_pair_rows": text_declaration,
        "provenance": {
            "source_identity_namespace": "fixture-namespace",
            "source_release_manifest_sha256": digest("fixture-source-release"),
            "extractor_sha256": digest("fixture-extractor"),
            "excluded_source_metadata_keys": [],
        },
    }
    manifest_path = directory / "collection_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def make_world(root, text=True):
    manifests = {
        "train": make_collection(
            root,
            "train",
            [
                ("train-a", "A", "A-train"),
                ("train-b", "B", "B-train"),
                ("train-c1", "C", "C-one"),
                ("train-c2", "C", "C-two"),
            ],
            text=text,
        ),
        "validation": make_collection(
            root,
            "validation",
            [("validation-b", "B", "B-validation"), ("validation-d", "D", "D")],
            text=text,
        ),
        "test": make_collection(
            root,
            "test",
            [("test-a", "A", "A-test"), ("test-e", "E", "E")],
            text=text,
        ),
    }
    return manifests


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class DownstreamConnectivityCleanViewTests(unittest.TestCase):
    def build(self, root, text=True):
        manifests = make_world(root, text=text)
        output = root / "clean"
        original_bytes = {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        summary = clean.derive_clean_view(
            manifests["train"],
            manifests["validation"],
            manifests["test"],
            output,
        )
        return manifests, original_bytes, output, summary

    def test_priority_partition_keeps_owner_members_and_all_test_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests, original_bytes, output, summary = self.build(root)
            expected_members = {
                "train": {"train-c1", "train-c2"},
                "validation": {"validation-b", "validation-d"},
                "test": {"test-a", "test-e"},
            }
            connectivity_sets = {}
            for split in clean.SPLITS:
                manifest_path = output / "collections" / split / "collection_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                proof.validate_collection_manifest(manifest)
                rows = load_jsonl(manifest_path.parent / manifest["molecule_rows"]["path"])
                self.assertEqual({row["member_id"] for row in rows}, expected_members[split])
                connectivity_sets[split] = {
                    row["connectivity_identity_sha256"] for row in rows
                }
                if split == "test":
                    text_rows = load_jsonl(
                        manifest_path.parent / manifest["text_pair_rows"]["path"]
                    )
                    self.assertEqual(len(text_rows), 3)
                    self.assertEqual(
                        {row["pair_id"] for row in text_rows},
                        {"pair:test-a:0", "pair:test-a:1", "pair:test-e:0"},
                    )
            self.assertFalse(connectivity_sets["train"] & connectivity_sets["validation"])
            self.assertFalse(connectivity_sets["train"] & connectivity_sets["test"])
            self.assertFalse(connectivity_sets["validation"] & connectivity_sets["test"])
            self.assertTrue(summary["proofs"]["test_unchanged"])
            self.assertTrue(summary["proofs"]["connectivity_split_disjoint"])
            self.assertTrue(summary["proofs"]["emitted_counts_match_priority_view"])
            self.assertEqual(summary["counts"]["input_members"], 8)
            self.assertEqual(summary["counts"]["retained_members"], 6)
            self.assertEqual(summary["counts"]["removed_members"], 2)
            for relative_path, expected_bytes in original_bytes.items():
                self.assertEqual((root / relative_path).read_bytes(), expected_bytes)

    def test_disposition_covers_every_member_and_removed_rows_name_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, output, summary = self.build(root)
            rows = load_jsonl(output / clean.DISPOSITION_FILENAME)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({(row["source_collection_id"], row["member_id"]) for row in rows}), 8)
            removed = [row for row in rows if row["disposition"] == "removed"]
            self.assertEqual(
                {(row["member_id"], row["assigned_split"]) for row in removed},
                {("train-a", "test"), ("train-b", "validation")},
            )
            self.assertTrue(all(row["removal_reason"] for row in removed))
            self.assertTrue(
                summary["proofs"]["all_reported_members_have_one_disposition"]
            )
            self.assertTrue(summary["proofs"]["every_removed_member_has_reason"])
            self.assertFalse(
                summary["provenance_observation_policy"][
                    "caller_supplied_digest_required"
                ]
            )
            self.assertFalse(
                summary["provenance_observation_policy"][
                    "observed_digests_are_scientific_admission_gates"
                ]
            )

    def test_text_pairs_of_removed_members_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, output, summary = self.build(root)
            train_manifest = json.loads(
                (output / "collections" / "train" / "collection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            train_text = load_jsonl(
                output
                / "collections"
                / "train"
                / train_manifest["text_pair_rows"]["path"]
            )
            self.assertEqual(
                {row["member_id"] for row in train_text}, {"train-c1", "train-c2"}
            )
            self.assertEqual(
                summary["counts"]["by_split"]["train"]["removed_text_pair_count"],
                2,
            )
            self.assertTrue(
                summary["proofs"]["output_text_pairs_reference_retained_members"]
            )

    def test_collections_without_text_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, output, summary = self.build(root, text=False)
            for split in clean.SPLITS:
                manifest = json.loads(
                    (output / "collections" / split / "collection_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertIsNone(manifest["text_pair_rows"])
            self.assertTrue(summary["proofs"]["test_text_pairs_unchanged"])

    def test_identity_spec_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = make_world(root)
            validation = json.loads(manifests["validation"].read_text(encoding="utf-8"))
            validation["identity_specs"]["connectivity_identity_spec_sha256"] = digest(
                "different-spec"
            )
            write_json(manifests["validation"], validation)
            output = root / "clean"
            with self.assertRaisesRegex(clean.CleanViewError, "identity_specs"):
                clean.derive_clean_view(
                    manifests["train"], manifests["validation"], manifests["test"], output
                )
            self.assertFalse(output.exists())

    def test_dangling_text_reference_fails_row_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = make_world(root)
            train_manifest = json.loads(manifests["train"].read_text(encoding="utf-8"))
            text_path = manifests["train"].parent / train_manifest["text_pair_rows"]["path"]
            rows = load_jsonl(text_path)
            rows[0]["member_id"] = "missing-member"
            train_manifest["text_pair_rows"] = write_rows(text_path, rows, "pair_id")
            write_json(manifests["train"], train_manifest)
            output = root / "clean"
            with self.assertRaisesRegex(ValueError, "missing molecule member"):
                clean.derive_clean_view(
                    manifests["train"], manifests["validation"], manifests["test"], output
                )
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = make_world(root)
            output = root / "clean"
            output.mkdir()
            sentinel = output / "reported.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                clean.derive_clean_view(
                    manifests["train"], manifests["validation"], manifests["test"], output
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
