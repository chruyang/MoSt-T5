from __future__ import annotations

import csv
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from most_t5_next.r1.overlap import build_kpgt_scaffold_manifests as kpgt
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


SMILES = (
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "c1ccoc1",
    "c1ccsc1",
    "C1CCNCC1",
)
DEFAULT_MEMBERSHIP = ([0, 1], [2, 3], [4, 5])


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_split(path: Path, membership=DEFAULT_MEMBERSHIP) -> None:
    payload = np.empty(3, dtype=object)
    for index, values in enumerate(membership):
        payload[index] = np.asarray(values, dtype=np.int64)
    np.save(path, payload, allow_pickle=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root: Path) -> None:
    labels = [0, 1, 0, 1, 0, 1]
    for spec in kpgt.TASK_SPECS:
        task_root = root / spec.task
        splits_root = task_root / "splits"
        splits_root.mkdir(parents=True)
        rows = []
        for index, smiles in enumerate(SMILES):
            row: dict[str, object] = {spec.smiles_column: smiles}
            for label_index, label in enumerate(spec.label_columns):
                row[label] = labels[index] if label_index == 0 else 1 - labels[index]
            rows.append(row)
        write_csv(
            task_root / spec.csv_name,
            [spec.smiles_column, *spec.label_columns],
            rows,
        )
        for split in kpgt.SPLIT_REPLICAS:
            write_split(splits_root / (split + ".npy"))


def make_archive(
    path: Path,
    source: Path,
    *,
    traversal_member: bool = False,
    symlink_member: bool = False,
) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for relative_path in kpgt._required_source_paths():
            archive.write(source / relative_path, "MoleculeNet/" + relative_path)
        if traversal_member:
            archive.writestr("../escape.txt", b"must be rejected")
        if symlink_member:
            info = zipfile.ZipInfo("MoleculeNet/unsafe-link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "bace/bace.csv")


class KpgtScaffoldManifestTests(unittest.TestCase):
    def build(self, root: Path, output_name: str = "derived") -> tuple[Path, dict[str, object]]:
        source = root / "source"
        if not source.exists():
            make_dataset(source)
        archive = root / "official-kpgt-fixture.zip"
        if not archive.exists():
            make_archive(archive, source)
        archive_sha256 = kpgt.sha256_file(archive).sha256
        output = root / output_name
        summary = kpgt.build_kpgt_scaffold_manifests(
            source,
            output,
            official_archive_path=archive,
            official_archive_sha256=archive_sha256,
            source_provenance=kpgt.OFFICIAL_SOURCE_PROVENANCE,
        )
        return output, summary

    def test_success_emits_official_bindings_checks_and_protected_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, summary = self.build(Path(temporary))
            source = read_json(output / kpgt.SOURCE_MANIFEST_FILENAME)
            self.assertEqual(source["source_provenance"], kpgt.OFFICIAL_SOURCE_PROVENANCE)
            archive = Path(temporary) / "official-kpgt-fixture.zip"
            self.assertEqual(
                source["official_archive"]["sha256"], kpgt.sha256_file(archive).sha256
            )
            self.assertEqual(source["official_archive"]["bytes"], archive.stat().st_size)
            self.assertEqual(source["official_archive"]["file_name"], archive.name)
            self.assertEqual(source["official_archive"]["format"], "zip")
            self.assertEqual(source["official_archive"]["figshare_doi"], kpgt.FIGSHARE_DOI)
            self.assertEqual(source["official_archive"]["figshare_file_id"], kpgt.FIGSHARE_FILE_ID)
            self.assertEqual(
                source["repository"]["paper_release_commit"],
                kpgt.KPGT_PAPER_RELEASE_COMMIT,
            )
            self.assertEqual(len(source["source_files"]), 12)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in source["source_files"]))
            self.assertTrue(
                all(item["archive_member_byte_identity_verified"] for item in source["source_files"])
            )
            self.assertTrue(
                all(item["archive_member_path"].startswith("MoleculeNet/") for item in source["source_files"])
            )

            self.assertEqual(len(summary["member_manifests"]), 9)
            self.assertEqual(len(summary["identity_collection_manifests"]), 27)
            for task in ("bace", "bbbp", "clintox"):
                for split in kpgt.SPLIT_REPLICAS:
                    checks = summary["tasks"][task]["split_replicas"][split]
                    self.assertEqual(
                        checks["partition_counts"],
                        {"train": 2, "validation": 2, "test": 2},
                    )
                    for flavor in ("achiral", "chiral"):
                        self.assertEqual(
                            set(checks["murcko_cross_partition_intersection_counts"][flavor].values()),
                            {0},
                        )
            protected = read_jsonl(output / kpgt.PROTECTED_UNION_FILENAME)
            self.assertEqual(len(protected), 4)
            self.assertEqual(
                {entry["partition"] for row in protected for entry in row["protected_by"]},
                {"validation", "test"},
            )

    def test_identity_collections_are_directly_accepted_by_existing_overlap_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, summary = self.build(root)
            artifact = next(
                item
                for item in summary["identity_collection_manifests"]
                if item["task"] == "bace"
                and item["split_replica"] == "scaffold-0"
                and item["partition"] == "validation"
            )
            manifest_path = output / artifact["relative_path"]
            manifest = read_json(manifest_path)
            proof.validate_collection_manifest(manifest)
            database = proof.create_database(str(root / "proof.sqlite"))
            try:
                loaded, observation = proof.load_collection(
                    database, manifest_path, artifact["sha256"]
                )
                self.assertEqual(loaded["role"], "downstream_validation")
                self.assertEqual(observation["molecule_rows"]["row_count"], 2)
            finally:
                database.close()

    def test_out_of_range_index_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root / "source")
            write_split(
                root / "source/bace/splits/scaffold-0.npy",
                ([0, 6], [2, 3], [4, 5]),
            )
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "out-of-range"):
                self.build(root)
            self.assertFalse((root / "derived").exists())

    def test_index_leakage_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root / "source")
            write_split(
                root / "source/bbbp/splits/scaffold-1.npy",
                ([0, 1], [1, 2, 3], [4, 5]),
            )
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "leaks"):
                self.build(root)
            self.assertFalse((root / "derived").exists())

    def test_single_evaluable_label_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root / "source")
            path = root / "source/bace/bace.csv"
            rows = [{"smiles": smiles, "Class": 0 if index < 2 else index % 2} for index, smiles in enumerate(SMILES)]
            write_csv(path, ["smiles", "Class"], rows)
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "single evaluable class"):
                self.build(root)
            self.assertFalse((root / "derived").exists())

    def test_wrong_provenance_never_reaches_pickle_or_creates_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root / "source")
            archive = root / "official-kpgt-fixture.zip"
            make_archive(archive, root / "source")
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "source_provenance"):
                kpgt.build_kpgt_scaffold_manifests(
                    root / "source",
                    root / "derived",
                    official_archive_path=archive,
                    official_archive_sha256=kpgt.sha256_file(archive).sha256,
                    source_provenance="kpgt_layout_candidate",
                )
            self.assertFalse((root / "derived").exists())

    def test_archive_hash_mismatch_is_rejected_before_pickle_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root / "source")
            archive = root / "official-kpgt-fixture.zip"
            make_archive(archive, root / "source")
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "observed SHA-256"):
                kpgt.build_kpgt_scaffold_manifests(
                    root / "source",
                    root / "derived",
                    official_archive_path=archive,
                    official_archive_sha256="a" * 64,
                    source_provenance=kpgt.OFFICIAL_SOURCE_PROVENANCE,
                )
            self.assertFalse((root / "derived").exists())

    def test_dataset_root_bytes_must_match_the_hashed_archive_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            make_dataset(source)
            archive = root / "official-kpgt-fixture.zip"
            make_archive(archive, source)
            with (source / "bace/bace.csv").open("a", encoding="utf-8") as handle:
                handle.write("c1ncccc1,1\n")
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "not byte-identical"):
                self.build(root)
            self.assertFalse((root / "derived").exists())

    def test_archive_path_traversal_is_rejected_before_pickle_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            make_dataset(source)
            archive = root / "official-kpgt-fixture.zip"
            make_archive(archive, source, traversal_member=True)
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "path traversal"):
                self.build(root)
            self.assertFalse((root / "derived").exists())

    def test_archive_symbolic_link_is_rejected_before_pickle_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            make_dataset(source)
            archive = root / "official-kpgt-fixture.zip"
            make_archive(archive, source, symlink_member=True)
            with self.assertRaisesRegex(kpgt.KpgtManifestError, "symbolic link"):
                self.build(root)
            self.assertFalse((root / "derived").exists())

    def test_all_artifacts_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = self.build(root, "first")
            second, _ = self.build(root, "second")
            first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
            second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
            self.assertEqual(first_files, second_files)
            for relative_path in first_files:
                self.assertEqual(
                    (first / relative_path).read_bytes(),
                    (second / relative_path).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
