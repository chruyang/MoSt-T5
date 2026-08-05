"""Hermetic fail-closed tests for the PCQM source-integrity boundary."""

from __future__ import print_function

import hashlib
import io
import json
import tempfile
import tarfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "adapter" / "pcqm_source_integrity.py"


def import_integrity():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("r1_pcqm_source_integrity_test", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


integrity = import_integrity()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PCQMSourceIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.archive = self.root / "pcqm4m-v2-train.sdf.tar.gz"
        self.csv = self.root / "data.csv.gz"
        self.split = self.root / "split_dict.pt"
        self.zip_path = self.root / "pcqm4m-v2.zip"
        self.source_manifest = self.root / "sdf_source_manifest.json"
        self.companion_manifest = self.root / "companion_source_manifest.json"
        self.content_manifest = self.root / "content_validation.json"
        self.member_hash_report = self.root / "train_sdf_member.sha256"
        self.contract_path = self.root / "contract.json"
        self.sdf_member_bytes = b"synthetic-sdf-member\n"
        with tarfile.open(str(self.archive), "w:gz") as archive:
            member = tarfile.TarInfo("pcqm4m-v2-train.sdf")
            member.size = len(self.sdf_member_bytes)
            archive.addfile(member, io.BytesIO(self.sdf_member_bytes))
        self.csv.write_bytes(b"synthetic-csv-gzip-member")
        self.split.write_bytes(b"synthetic-split-member")
        with zipfile.ZipFile(str(self.zip_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("pcqm4m-v2/raw/data.csv.gz", self.csv.read_bytes())
            archive.writestr("pcqm4m-v2/split_dict.pt", self.split.read_bytes())
        with zipfile.ZipFile(str(self.zip_path), "r") as archive:
            csv_info = archive.getinfo("pcqm4m-v2/raw/data.csv.gz")
            split_info = archive.getinfo("pcqm4m-v2/split_dict.pt")
        source_md5 = hashlib.md5(self.archive.read_bytes()).hexdigest()
        self.member_hash_report.write_bytes(
            (hashlib.sha256(self.sdf_member_bytes).hexdigest() + "  -\n").encode("ascii")
        )
        self.write_json(
            self.source_manifest,
            {
                "source_id": "ogb-pcqm4mv2-train-3d-v1",
                "official_train_3d_sdf_count": 3378606,
                "expected_md5": source_md5,
            },
        )
        self.write_json(
            self.companion_manifest,
            {
                "files": {
                    "archive": {"sha256": sha256(self.zip_path)},
                    "selected_members": {
                        "raw/data.csv.gz": {"sha256": sha256(self.csv)},
                        "split/split_dict.pt": {"sha256": sha256(self.split)},
                    },
                }
            },
        )
        self.write_json(
            self.content_manifest,
            {
                "csv_rows": 3746620,
                "csv_idx_is_zero_based_contiguous": True,
                "split_counts": {"train": 3378606},
                "train_is_contiguous_prefix": True,
                "split_minmax": {"train": [0, 3378605]},
                "split_overlap_counts": {"train__valid": 0},
            },
        )
        self.write_json(
            self.contract_path,
            {
                "schema_version": integrity.SOURCE_CONTRACT_SCHEMA,
                "source": {
                    "official_train_sdf_records": 3378606,
                    "official_md5": source_md5,
                    "remote_archive_path": str(self.archive),
                    "remote_archive_bytes": self.archive.stat().st_size,
                    "remote_archive_sha256": sha256(self.archive),
                    "remote_source_manifest": self.file_lock(self.source_manifest),
                    "train_sdf_member": {
                        "tar_member_name": "pcqm4m-v2-train.sdf",
                        "member_type": "regular_file",
                        "uncompressed_bytes": len(self.sdf_member_bytes),
                        "sha256": hashlib.sha256(self.sdf_member_bytes).hexdigest(),
                    },
                    "remote_sdf_member_hash_report": self.file_lock(self.member_hash_report),
                },
                "official_companion": {
                    "official_archive": self.file_lock(self.zip_path),
                    "data_csv": self.file_lock(self.csv),
                    "split_dict": self.file_lock(self.split),
                    "remote_source_manifest": self.file_lock(self.companion_manifest),
                    "remote_content_validation": self.file_lock(self.content_manifest),
                    "archive_members": {
                        "data_csv": self.member_lock(csv_info),
                        "split_dict": self.member_lock(split_info),
                    },
                    "validated_invariants": {
                        "csv_rows": 3746620,
                        "csv_idx_is_zero_based_contiguous": True,
                        "train_split_records": 3378606,
                        "train_split_is_contiguous_prefix": True,
                        "train_split_min": 0,
                        "train_split_max": 3378605,
                    },
                },
            },
        )

    def tearDown(self):
        # Each fixture path is explicit; tests must not use a recursive delete.
        for path in (
            self.archive,
            self.csv,
            self.split,
            self.zip_path,
            self.source_manifest,
            self.companion_manifest,
            self.content_manifest,
            self.member_hash_report,
            self.contract_path,
            self.root / "lookalike-data.csv.gz",
        ):
            if path.exists():
                path.unlink()
        self.root.rmdir()

    def write_json(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def file_lock(self, path):
        path = Path(path)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}

    def member_lock(self, info):
        return {
            "zip_member": info.filename,
            "crc32": "{:08x}".format(info.CRC),
            "compressed_bytes": info.compress_size,
            "uncompressed_bytes": info.file_size,
        }

    def test_verified_input_lock_binds_every_artifact_and_member(self):
        verified = integrity.verify_pcqm_inputs(
            self.contract_path, self.archive, self.csv, self.split
        )
        report = verified.report()
        self.assertEqual(report["source_record_count"], 3378606)
        self.assertEqual(
            report["artifacts"]["train_3d_sdf_archive"]["sha256"], sha256(self.archive)
        )
        self.assertEqual(
            report["companion_archive_members"]["data_csv"]["sha256"], sha256(self.csv)
        )
        self.assertEqual(
            report["companion_archive_members"]["split_dict"]["sha256"], sha256(self.split)
        )

    def test_changed_companion_file_fails_before_any_split_deserialization(self):
        self.split.write_bytes(b"substituted-split-member")
        with self.assertRaisesRegex(RuntimeError, "companion_split_dict_pt (byte count|SHA-256)"):
            integrity.verify_pcqm_inputs(self.contract_path, self.archive, self.csv, self.split)

    def test_noncanonical_cli_path_is_rejected_even_if_the_content_matches(self):
        copy = self.root / "lookalike-data.csv.gz"
        copy.write_bytes(self.csv.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "exact locked artifact"):
            integrity.verify_pcqm_inputs(self.contract_path, self.archive, copy, self.split)


if __name__ == "__main__":
    unittest.main(verbosity=2)
