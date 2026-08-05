"""Hermetic tests for the cross-region PCQM staging receipt gate."""

from __future__ import print_function

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "adapter" / "pcqm_staging_receipt.py"
PRODUCTION_STAGING_CONTRACT = ROOT / "contracts" / "pcqm4mv2_staging_receipt_contract.json"
PRODUCTION_SOURCE_CONTRACT = ROOT / "contracts" / "pcqm4mv2_source_contract.json"


def import_adapter():
    import sys

    spec = importlib.util.spec_from_file_location("r1_pcqm_staging_receipt_test", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = import_adapter()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


class PCQMStagingReceiptTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.created_files = []
        self.work_paths = {}
        self.source_specs = {}
        for index, role in enumerate(adapter.REQUIRED_ROLES):
            content = ("fixture:{}:{}\n".format(index, role)).encode("utf-8")
            work_path = (self.work / (role + ".bin")).resolve()
            work_path.write_bytes(content)
            self.created_files.append(work_path)
            self.work_paths[role] = work_path
            self.source_specs[role] = {
                "path": str((self.root / "canonical" / (role + ".bin")).resolve()),
                "bytes": len(content),
                "sha256": sha256_bytes(content),
            }

        self.source_contract_path = self.root / "pcqm4mv2_source_contract.json"
        self.staging_contract_path = self.root / "staging_contract.json"
        self.receipt_path = self.root / "receipt.json"
        self.created_files.extend(
            [self.source_contract_path, self.staging_contract_path, self.receipt_path]
        )
        self.write_json(self.source_contract_path, self.make_source_contract(), canonical=False)
        self.write_json(self.staging_contract_path, self.make_staging_contract(), canonical=False)
        self.receipt = self.make_receipt()
        self.write_json(self.receipt_path, self.receipt, canonical=True)

    def tearDown(self):
        for path in reversed(self.created_files):
            if path.is_symlink() or path.exists():
                path.unlink()
        if self.work.exists():
            self.work.rmdir()
        self.root.rmdir()

    def write_json(self, path, value, canonical):
        if canonical:
            path.write_bytes(adapter.canonical_json_bytes(value) + b"\n")
        else:
            path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def make_source_contract(self):
        specs = self.source_specs
        return {
            "schema_version": adapter.SOURCE_CONTRACT_SCHEMA,
            "source": {
                "remote_archive_path": specs["train_3d_sdf_archive"]["path"],
                "remote_archive_bytes": specs["train_3d_sdf_archive"]["bytes"],
                "remote_archive_sha256": specs["train_3d_sdf_archive"]["sha256"],
                "remote_source_manifest": specs["train_sdf_source_manifest"],
                "remote_sdf_member_hash_report": specs["train_sdf_member_hash_report"],
            },
            "official_companion": {
                "official_archive": specs["companion_archive"],
                "data_csv": specs["companion_data_csv_gz"],
                "split_dict": specs["companion_split_dict_pt"],
                "remote_source_manifest": specs["companion_source_manifest"],
                "remote_content_validation": specs["companion_content_validation"],
            },
        }

    def make_staging_contract(self):
        source_raw = self.source_contract_path.read_bytes()
        return {
            "schema_version": adapter.STAGING_CONTRACT_SCHEMA,
            "purpose": "test-only staging policy",
            "receipt_schema_version": adapter.RECEIPT_SCHEMA,
            "pinned_source_contract": {
                "repository_relative_path": "contracts/pcqm4mv2_source_contract.json",
                "filename": "pcqm4mv2_source_contract.json",
                "schema_version": adapter.SOURCE_CONTRACT_SCHEMA,
                "bytes": len(source_raw),
                "sha256": sha256_bytes(source_raw),
            },
            "required_roles": list(adapter.REQUIRED_ROLES),
            "required_artifact_fields": list(adapter.ARTIFACT_FIELDS),
            "required_transfer_fields": list(adapter.TRANSFER_FIELDS),
            "allowed_transfer_methods": list(adapter.ALLOWED_TRANSFER_METHODS),
            "required_cpu_runtime_fields": list(adapter.CPU_RUNTIME_FIELDS),
            "required_cpu_environment_fields": list(adapter.CPU_ENVIRONMENT_FIELDS),
            "required_cpu_packages": list(adapter.CPU_PACKAGES),
            "file_policy": dict(adapter.FILE_POLICY),
        }

    def make_runtime(self):
        runtime = {
            "captured_at_utc": "2026-08-05T03:00:00Z",
            "hostname": "cpu-worker",
            "platform_system": "Linux",
            "platform_release": "6.8.0",
            "machine": "x86_64",
            "python_implementation": "CPython",
            "python_version": "3.8.20",
            "python_executable": str((self.root / "runtime" / "python").resolve()),
            "cpu_model": "AMD EPYC 9654 96-Core Processor",
            "logical_cpu_count": 96,
            "affinity_cpu_count": 96,
            "memory_bytes": 180000000000,
            "working_root": str(self.work),
            "environment": {
                "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            "packages": {
                "python": "3.8.20",
                "numpy": "1.24.4",
                "scipy": "1.10.1",
                "rdkit": "2024.03.5",
                "e3fp": "1.2.5",
                "lmdb": "1.7.5",
                "torch": "2.1.0+cpu",
            },
        }
        runtime["snapshot_sha256"] = adapter.cpu_runtime_snapshot_sha256(runtime)
        return runtime

    def make_receipt(self):
        source_raw = self.source_contract_path.read_bytes()
        return {
            "schema_version": adapter.RECEIPT_SCHEMA,
            "receipt_created_at_utc": "2026-08-05T03:10:00Z",
            "source_contract": {
                "schema_version": adapter.SOURCE_CONTRACT_SCHEMA,
                "path": str(self.source_contract_path.resolve()),
                "bytes": len(source_raw),
                "sha256": sha256_bytes(source_raw),
            },
            "transfer": {
                "transfer_id": "cpu-stage-test-001",
                "method": "scp_over_ssh",
                "source_endpoint": "gpu-region:canonical-store",
                "destination_endpoint": "cpu-region:working-store",
                "started_at_utc": "2026-08-05T02:00:00Z",
                "completed_at_utc": "2026-08-05T02:30:00Z",
                "status": "completed",
                "verification": "post_transfer_bytes_and_sha256",
            },
            "cpu_runtime": self.make_runtime(),
            "artifacts": {
                role: {
                    "source_path": self.source_specs[role]["path"],
                    "work_path": str(self.work_paths[role]),
                    "bytes": self.source_specs[role]["bytes"],
                    "sha256": self.source_specs[role]["sha256"],
                }
                for role in adapter.REQUIRED_ROLES
            },
        }

    def verify(self):
        return adapter.verify_staging_receipt(
            self.staging_contract_path.resolve(),
            self.source_contract_path.resolve(),
            self.receipt_path.resolve(),
        )

    def rewrite_receipt(self):
        self.write_json(self.receipt_path, self.receipt, canonical=True)

    def test_valid_receipt_returns_only_verified_work_paths(self):
        verified = self.verify()
        self.assertEqual(verified.source_contract_sha256, self.receipt["source_contract"]["sha256"])
        self.assertEqual(
            verified.work_path("train_3d_sdf_archive"),
            str(self.work_paths["train_3d_sdf_archive"]),
        )
        report = verified.report()
        self.assertTrue(report["pass"])
        self.assertFalse(report["p1_training_admitted"])
        self.assertEqual(set(report["artifacts"]), set(adapter.REQUIRED_ROLES))

    def test_generator_hashes_then_verifies_every_work_role(self):
        runtime = self.make_runtime()
        del runtime["snapshot_sha256"]
        verified = adapter.generate_and_verify_staging_receipt(
            self.staging_contract_path.resolve(),
            self.source_contract_path.resolve(),
            self.receipt_path.resolve(),
            {role: str(path) for role, path in self.work_paths.items()},
            dict(self.receipt["transfer"]),
            runtime,
            self.receipt["receipt_created_at_utc"],
        )
        self.assertEqual(
            verified.cpu_runtime_sha256,
            adapter.cpu_runtime_snapshot_sha256(self.make_runtime()),
        )
        persisted = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["artifacts"]["train_3d_sdf_archive"]["sha256"],
            self.source_specs["train_3d_sdf_archive"]["sha256"],
        )

    def test_tampered_work_file_fails_closed(self):
        self.work_paths["companion_split_dict_pt"].write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "working (byte count|SHA-256) mismatch"):
            self.verify()

    def test_receipt_cannot_relabel_a_canonical_source_role(self):
        self.receipt["artifacts"]["companion_data_csv_gz"]["source_path"] = str(
            self.work_paths["companion_data_csv_gz"]
        )
        self.rewrite_receipt()
        with self.assertRaisesRegex(RuntimeError, "source path differs from source contract"):
            self.verify()

    def test_runtime_snapshot_mutation_fails_closed(self):
        self.receipt["cpu_runtime"]["affinity_cpu_count"] = 48
        self.rewrite_receipt()
        with self.assertRaisesRegex(RuntimeError, "runtime snapshot SHA-256 mismatch"):
            self.verify()

    def test_noncanonical_receipt_serialization_is_rejected(self):
        self.receipt_path.write_text(json.dumps(self.receipt, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not canonical UTF-8 JSON plus LF"):
            self.verify()

    def test_symlink_work_file_is_rejected_when_supported(self):
        role = "companion_content_validation"
        link = self.work / "linked-content-validation.bin"
        try:
            os.symlink(str(self.work_paths[role]), str(link))
        except (OSError, NotImplementedError) as exc:
            self.skipTest("symlink creation is unavailable: {}".format(type(exc).__name__))
        self.created_files.append(link)
        self.receipt["artifacts"][role]["work_path"] = str(link.resolve(strict=False))
        # Preserve the lexical symlink path: resolving it would hide the link.
        self.receipt["artifacts"][role]["work_path"] = str(link)
        self.rewrite_receipt()
        with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
            self.verify()

    def test_production_staging_contract_pins_current_v3_source_contract(self):
        policy = json.loads(PRODUCTION_STAGING_CONTRACT.read_text(encoding="utf-8"))
        source_raw = PRODUCTION_SOURCE_CONTRACT.read_bytes()
        pin = policy["pinned_source_contract"]
        self.assertEqual(pin["bytes"], len(source_raw))
        self.assertEqual(pin["sha256"], sha256_bytes(source_raw))
        self.assertEqual(pin["schema_version"], adapter.SOURCE_CONTRACT_SCHEMA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
