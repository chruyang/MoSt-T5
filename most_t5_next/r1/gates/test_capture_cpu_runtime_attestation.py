"""Hermetic tests for the CPU runtime attestation collector."""

from __future__ import print_function

import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "gates" / "capture_cpu_runtime_attestation.py"
VALIDATOR_PATH = ROOT / "gates" / "validate_cpu_runtime_attestation.py"
CONTRACT_PATH = ROOT / "contracts" / "cpu_runtime_attestation_contract.json"


def import_collector():
    spec = importlib.util.spec_from_file_location("r1_cpu_runtime_attestation_test", str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = import_collector()


def import_validator():
    import sys

    sys.modules["capture_cpu_runtime_attestation"] = collector
    spec = importlib.util.spec_from_file_location("r1_cpu_runtime_attestation_validator_test", str(VALIDATOR_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = import_validator()


class CPURuntimeAttestationTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.e3fp = self.root / "source" / "e3fp"
        self.fingerprint = self.e3fp / "fingerprint"
        self.config = self.e3fp / "config"
        self.cache = self.e3fp / "__pycache__"
        self.bundle = self.root / "bundle"
        for directory in (self.fingerprint, self.config, self.cache, self.bundle / "nested"):
            directory.mkdir(parents=True, exist_ok=True)
        self.files = {
            self.e3fp / "__init__.py": b'__version__ = "1.2.5"\n',
            self.e3fp / "pipeline.py": b"PIPELINE = True\n",
            self.fingerprint / "fprinter.py": b"FPRINTER = True\n",
            self.config / "defaults.cfg": b"[fingerprinting]\nlevel = 3\n",
            self.e3fp / "util.py": b"UTILITY = True\n",
            self.cache / "util.cpython-38.pyc": b"generated-bytecode",
            self.bundle / "runner.py": b"print('runner')\n",
            self.bundle / "nested" / "contract.json": b"{}\n",
        }
        for path, value in self.files.items():
            path.write_bytes(value)
        self.output = self.root / "attestation.json"

    def tearDown(self):
        for path in (self.output,):
            if path.exists():
                path.unlink()
        for path in reversed(tuple(self.files)):
            if path.exists():
                path.unlink()
        for directory in (
            self.cache,
            self.fingerprint,
            self.config,
            self.e3fp,
            self.root / "source",
            self.bundle / "nested",
            self.bundle,
            self.root,
        ):
            if directory.exists():
                directory.rmdir()

    def contract(self):
        return collector.load_contract(CONTRACT_PATH)[1]

    def test_canonical_json_is_sorted_float_free_and_new_only(self):
        value = {"z": [2, 1], "a": {"value": "ok"}}
        expected = b'{"a":{"value":"ok"},"z":[2,1]}\n'
        self.assertEqual(collector.canonical_json_bytes(value), expected)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            collector.canonical_json_bytes({"bad": 1.25})
        collector.write_new_canonical_json(self.output, value)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            collector.write_new_canonical_json(self.output, {"different": True})
        self.assertEqual(self.output.read_bytes(), before)

    def test_e3fp_closure_hashes_every_source_file_but_not_generated_bytecode(self):
        closure = collector.collect_e3fp_source_closure(self.e3fp.parent, self.contract())
        relative_paths = [item["relative_path"] for item in closure["files"]]
        self.assertEqual(
            relative_paths,
            ["__init__.py", "config/defaults.cfg", "fingerprint/fprinter.py", "pipeline.py", "util.py"],
        )
        self.assertEqual(closure["file_count"], 5)
        self.assertEqual(closure["excluded_generated_paths"], ["__pycache__/"])
        self.assertEqual(
            closure["closure_sha256"],
            hashlib.sha256(collector.canonical_json_bytes(closure["files"])).hexdigest(),
        )

    def test_bundle_lock_is_order_independent_and_rejects_traversal(self):
        contract = self.contract()
        first = collector.collect_bundle_file_lock(
            self.bundle, ["runner.py", "nested/contract.json"], contract
        )
        second = collector.collect_bundle_file_lock(
            self.bundle, ["nested/contract.json", "runner.py"], contract
        )
        self.assertEqual(first["closure_sha256"], second["closure_sha256"])
        self.assertEqual([item["relative_path"] for item in first["files"]], ["nested/contract.json", "runner.py"])
        with self.assertRaisesRegex(ValueError, "normalized relative"):
            collector.collect_bundle_file_lock(self.bundle, ["../outside.py"], contract)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            collector.collect_bundle_file_lock(self.bundle, ["runner.py", "runner.py"], contract)

    def test_cpu_parsers_preserve_fractional_quota_without_float(self):
        self.assertEqual(collector.parse_cpu_set("0-3,8,10-11"), (0, 1, 2, 3, 8, 10, 11))
        self.assertEqual(collector.encode_cpu_set((0, 1, 2, 3, 8, 10, 11)), "0-3,8,10-11")
        self.assertEqual(
            collector._parse_v2_cpu_max("150000 100000"),
            {"limited": True, "quota_us": 150000, "period_us": 100000},
        )
        with self.assertRaises(ValueError):
            collector.parse_cpu_set("8-3")

    def test_full_report_binds_contract_source_bundle_and_thread_policy(self):
        fake_python = {"implementation": "CPython", "version": "3.8.20"}
        fake_platform = {
            "operating_system": {"system": "Linux"},
            "libc": {"resolved_glibc_versions": ["2.35"]},
        }
        fake_dependencies = {
            role: {
                "status": "ok",
                "module_versions": {"module:__version__": "test"},
                "distributions": [],
            }
            for role in collector.REQUIRED_DEPENDENCY_ROLES
        }
        fake_cpu = {
            "logical_cpu_count": 96,
            "cgroup": {"version": 2, "quota": {"limited": False}},
            "effective_cpu_capacity": {"limiting_role": "process_affinity", "numerator": 96, "denominator": 1},
        }
        thread_values = {key: "1" for key in collector.REQUIRED_THREAD_KEYS}
        with mock.patch.object(collector, "collect_python_observation", return_value=fake_python), mock.patch.object(
            collector, "collect_platform_observation", return_value=(fake_platform, [])
        ), mock.patch.object(
            collector, "collect_dependency_observations", return_value=(fake_dependencies, [])
        ), mock.patch.object(collector, "collect_cpu_observation", return_value=(fake_cpu, [])), mock.patch.dict(
            os.environ, thread_values, clear=True
        ):
            report = collector.build_attestation(
                CONTRACT_PATH,
                self.e3fp,
                self.bundle,
                ["nested/contract.json", "runner.py"],
                created_utc="2026-08-05T00:00:00Z",
            )
        self.assertTrue(report["pass"])
        self.assertEqual(report["created_utc"], "2026-08-05T00:00:00Z")
        self.assertEqual(report["bundle_file_lock"]["file_count"], 2)
        self.assertEqual(report["e3fp_source_closure"]["file_count"], 5)
        self.assertEqual(len(report["attestation_payload_sha256"]), 64)
        collector.write_new_canonical_json(self.output, report)
        self.assertEqual(self.output.read_bytes(), collector.canonical_json_bytes(report))
        self.assertEqual(validator.validate_attestation(self.output, CONTRACT_PATH), [])

        tampered = dict(report)
        tampered["bundle_file_lock"] = dict(report["bundle_file_lock"])
        tampered["bundle_file_lock"]["closure_sha256"] = "0" * 64
        self.output.unlink()
        collector.write_new_canonical_json(self.output, tampered)
        errors = validator.validate_attestation(self.output, CONTRACT_PATH)
        self.assertIn("attestation payload SHA-256 is invalid", errors)
        self.assertIn("bundle closure SHA-256 is invalid", errors)

    def test_unset_thread_variables_fail_closed_without_hiding_observation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            observation, errors = collector.collect_thread_environment(self.contract())
        self.assertEqual(len(errors), 4)
        self.assertIsNone(observation["observed"]["OMP_NUM_THREADS"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
