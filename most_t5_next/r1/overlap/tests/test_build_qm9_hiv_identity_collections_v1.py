from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import build_qm9_hiv_identity_collections_v1 as adapter
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def qm9_group_id(connectivity_label: str) -> str:
    return "qm9-canonical-connectivity-smiles-sha256:" + digest(connectivity_label)


def write_json(path: Path, value: object) -> None:
    path.write_bytes(adapter.canonical_json_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(adapter.canonical_json_bytes(row) + b"\n" for row in rows)
    )


class Qm9HivIdentityCollectionAdapterTests(unittest.TestCase):
    def make_inputs(self, root: Path, *, conflict: bool = False):
        qm9 = root / "qm9"
        hiv = root / "hiv"
        qm9.mkdir()
        hiv.mkdir()
        write_json(qm9 / "source_manifest.json", {"source": "qm9-fixture"})
        write_json(hiv / "source_manifest.json", {"source": "hiv-fixture"})

        identity_spec_id = adapter.identity_spec_id()
        qm9_rows = [
            {
                "schema_version": adapter.QM9_ROW_SCHEMA,
                "assigned_split": "validation",
                "group_id": qm9_group_id("conn-a"),
                "canonical_connectivity_smiles_sha256": digest("conn-a"),
                "strict_canonical_isomeric_smiles_sha256": digest("stereo-a"),
            },
            {
                "schema_version": adapter.QM9_ROW_SCHEMA,
                "assigned_split": "validation",
                "group_id": qm9_group_id("conn-a"),
                "canonical_connectivity_smiles_sha256": (
                    digest("conn-conflict") if conflict else digest("conn-a")
                ),
                "strict_canonical_isomeric_smiles_sha256": digest("stereo-a2"),
            },
            {
                "schema_version": adapter.QM9_ROW_SCHEMA,
                "assigned_split": "validation",
                "group_id": qm9_group_id("conn-a"),
                "canonical_connectivity_smiles_sha256": digest("conn-a"),
                "strict_canonical_isomeric_smiles_sha256": digest("stereo-a"),
            },
            {
                "schema_version": adapter.QM9_ROW_SCHEMA,
                "assigned_split": "validation",
                "group_id": qm9_group_id("conn-b"),
                "canonical_connectivity_smiles_sha256": digest("conn-b"),
                "strict_canonical_isomeric_smiles_sha256": digest("stereo-b"),
            },
            {
                "schema_version": adapter.QM9_ROW_SCHEMA,
                "assigned_split": "test",
                "group_id": qm9_group_id("conn-c"),
                "canonical_connectivity_smiles_sha256": digest("conn-c"),
                "strict_canonical_isomeric_smiles_sha256": digest("stereo-c"),
            },
            {
                "schema_version": adapter.QM9_ROW_SCHEMA,
                "assigned_split": "train",
                "group_id": qm9_group_id("conn-train"),
                "canonical_connectivity_smiles_sha256": digest("conn-train"),
                "strict_canonical_isomeric_smiles_sha256": digest("stereo-train"),
            },
        ]
        write_jsonl(qm9 / "split_manifest.jsonl", qm9_rows)
        write_json(
            qm9 / "split_summary.json",
            {
                "schema_version": adapter.QM9_SUMMARY_SCHEMA,
                "dataset_id": adapter.QM9_DATASET_ID,
                "split_protocol_id": adapter.QM9_PROTOCOL_ID,
                "identity_normalization_contract_sha256": identity_spec_id,
                "counts": {
                    "output_rows": {"train": 1, "validation": 4, "test": 1},
                    "output_groups": {"train": 1, "validation": 2, "test": 1},
                },
            },
        )

        hiv_rows = []
        for index, split in enumerate(("train", "validation", "validation", "test")):
            hiv_rows.append(
                {
                    "schema_version": adapter.HIV_ROW_SCHEMA,
                    "protocol_id": adapter.HIV_PROTOCOL_ID,
                    "dataset_id": adapter.HIV_DATASET_ID,
                    "assigned_split": split,
                    "member_id": "hiv-member-{}".format(index),
                    "connectivity_identity_sha256": digest("hiv-conn-{}".format(index)),
                    "stereo_identity_sha256": digest("hiv-stereo-{}".format(index)),
                }
            )
        write_jsonl(hiv / "member_manifest.jsonl", hiv_rows)
        write_json(
            hiv / "split_manifest.json",
            {
                "schema_version": adapter.HIV_SPLIT_SCHEMA,
                "dataset_id": adapter.HIV_DATASET_ID,
                "protocol_id": adapter.HIV_PROTOCOL_ID,
                "canonicalization": {
                    "identity_normalization_contract_sha256": identity_spec_id
                },
                "counts": {
                    "member_counts": {"train": 1, "validation": 2, "test": 1}
                },
            },
        )
        return qm9, hiv

    def build(self, root: Path, name: str = "derived"):
        qm9, hiv = self.make_inputs(root)
        output = root / name
        summary = adapter.build_identity_collections(
            qm9_split_manifest=qm9 / "split_manifest.jsonl",
            qm9_summary=qm9 / "split_summary.json",
            hiv_member_manifest=hiv / "member_manifest.jsonl",
            hiv_split_manifest=hiv / "split_manifest.json",
            output_dir=output,
        )
        return output, summary

    def test_projects_only_eval_members_and_preserves_qm9_stereo_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, summary = self.build(Path(temporary))
            observed = {
                (item["dataset_id"], item["split"]): item["member_count"]
                for item in summary["collections"]
            }
            self.assertEqual(
                observed,
                {
                    ("3dmolt5-e3fp-mol-instructions-qm9-clean-view", "validation"): 3,
                    ("3dmolt5-e3fp-mol-instructions-qm9-clean-view", "test"): 1,
                    ("HIV-MoleculeNet-DeepChem", "validation"): 2,
                    ("HIV-MoleculeNet-DeepChem", "test"): 1,
                },
            )
            for item in summary["collections"]:
                manifest_path = output / item["manifest"]["relative_path"]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                proof.validate_collection_manifest(manifest)
                self.assertIn(manifest["role"], ("downstream_validation", "downstream_test"))
                self.assertNotIn("train", manifest["collection_id"])
            self.assertEqual(
                summary["qm9_connectivity_groups_scanned"],
                {"validation": 2, "test": 1},
            )

    def test_projection_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root)
            outputs = []
            for name in ("first", "second"):
                output = root / name
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=output,
                )
                outputs.append(output)
            first_files = {
                path.relative_to(outputs[0]).as_posix(): path.read_bytes()
                for path in outputs[0].rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(outputs[1]).as_posix(): path.read_bytes()
                for path in outputs[1].rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_group_id_connectivity_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root, conflict=True)
            with self.assertRaisesRegex(adapter.IdentityAdapterError, "not derived"):
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=root / "derived",
                )
            self.assertFalse((root / "derived").exists())

    def test_identity_contract_mismatch_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root)
            summary = json.loads((qm9 / "split_summary.json").read_text(encoding="utf-8"))
            summary["identity_normalization_contract_sha256"] = digest("wrong-contract")
            write_json(qm9 / "split_summary.json", summary)
            with self.assertRaisesRegex(adapter.IdentityAdapterError, "contract differs"):
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=root / "derived",
                )
            self.assertFalse((root / "derived").exists())

    def test_connectivity_group_crossing_train_and_eval_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root)
            rows = [
                json.loads(line)
                for line in (qm9 / "split_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[-1]["group_id"] = qm9_group_id("conn-a")
            rows[-1]["canonical_connectivity_smiles_sha256"] = digest("conn-a")
            write_jsonl(qm9 / "split_manifest.jsonl", rows)
            with self.assertRaisesRegex(adapter.IdentityAdapterError, "crosses splits"):
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=root / "derived",
                )
            self.assertFalse((root / "derived").exists())

    def test_hiv_member_protocol_must_match_split_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root)
            rows = [
                json.loads(line)
                for line in (hiv / "member_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[1]["protocol_id"] = "HIV-old-protocol"
            write_jsonl(hiv / "member_manifest.jsonl", rows)
            with self.assertRaisesRegex(adapter.IdentityAdapterError, "row protocol"):
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=root / "derived",
                )
            self.assertFalse((root / "derived").exists())

    def test_qm9_group_id_must_bind_connectivity_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root)
            rows = [
                json.loads(line)
                for line in (qm9 / "split_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[-1]["canonical_connectivity_smiles_sha256"] = digest("conn-a")
            write_jsonl(qm9 / "split_manifest.jsonl", rows)
            with self.assertRaisesRegex(adapter.IdentityAdapterError, "not derived"):
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=root / "derived",
                )
            self.assertFalse((root / "derived").exists())

    def test_hiv_member_ids_must_be_unique_across_all_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qm9, hiv = self.make_inputs(root)
            rows = [
                json.loads(line)
                for line in (hiv / "member_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rows[-1]["member_id"] = rows[1]["member_id"]
            write_jsonl(hiv / "member_manifest.jsonl", rows)
            with self.assertRaisesRegex(adapter.IdentityAdapterError, "duplicate HIV"):
                adapter.build_identity_collections(
                    qm9_split_manifest=qm9 / "split_manifest.jsonl",
                    qm9_summary=qm9 / "split_summary.json",
                    hiv_member_manifest=hiv / "member_manifest.jsonl",
                    hiv_split_manifest=hiv / "split_manifest.json",
                    output_dir=root / "derived",
                )
            self.assertFalse((root / "derived").exists())


if __name__ == "__main__":
    unittest.main()
