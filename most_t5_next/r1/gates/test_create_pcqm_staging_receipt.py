"""Minimal CLI/API test for create_pcqm_staging_receipt.py."""

from __future__ import print_function

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CREATOR_PATH = Path(__file__).resolve().with_name("create_pcqm_staging_receipt.py")
SPEC = importlib.util.spec_from_file_location("r1_create_pcqm_staging_receipt_test", str(CREATOR_PATH))
creator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(creator)


class _VerifiedFixture:
    def report(self):
        return {
            "schema_version": "most-t5-r1/pcqm4mv2-staging-verification/v1",
            "pass": True,
            "p1_training_admitted": False,
        }


class _AdapterFixture:
    VERIFICATION_REPORT_SCHEMA = "most-t5-r1/pcqm4mv2-staging-verification/v1"

    def __init__(self):
        self.call = None

    @staticmethod
    def canonical_json_bytes(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def generate_and_verify_staging_receipt(self, *args):
        self.call = args
        return _VerifiedFixture()


class CreatePCQMStagingReceiptCLITest(unittest.TestCase):
    def test_cli_collects_runtime_and_forwards_all_eight_roles(self):
        fixture = _AdapterFixture()
        runtime = {
            "captured_at_utc": "2026-08-05T04:00:00Z",
            "hostname": "cpu-worker",
        }
        base = Path.cwd().resolve()
        argv = [
            "--output",
            str(base / "receipt-never-written-by-fixture.json"),
            "--working-root",
            str(base),
            "--e3fp-source",
            str(base / "3d_tokenization"),
            "--transfer-id",
            "cross-region-001",
            "--transfer-method",
            "scp_over_ssh",
            "--source-endpoint",
            "gpu-region",
            "--destination-endpoint",
            "cpu-region",
            "--transfer-started-at-utc",
            "2026-08-05T03:00:00Z",
            "--transfer-completed-at-utc",
            "2026-08-05T03:30:00Z",
            "--receipt-created-at-utc",
            "2026-08-05T04:00:00Z",
        ]
        for index, (_, flag) in enumerate(creator.ROLE_FLAGS):
            argv.extend([flag, str(base / "role-{}.bin".format(index))])

        stdout = io.StringIO()
        with mock.patch.object(creator, "import_adapter", return_value=fixture), mock.patch.object(
            creator, "collect_cpu_runtime", return_value=runtime
        ) as collect, contextlib.redirect_stdout(stdout):
            result = creator.main(argv)

        self.assertEqual(result, 0)
        collect.assert_called_once_with(
            base,
            str(base / "3d_tokenization"),
            captured_at_utc="2026-08-05T04:00:00Z",
        )
        self.assertIsNotNone(fixture.call)
        work_paths = fixture.call[3]
        self.assertEqual(set(work_paths), {role for role, _ in creator.ROLE_FLAGS})
        self.assertEqual(fixture.call[4]["status"], "completed")
        self.assertEqual(fixture.call[5], runtime)
        self.assertTrue(json.loads(stdout.getvalue())["pass"])

    def test_runtime_collection_requires_explicit_cpu_environment(self):
        base = Path.cwd().resolve()
        with mock.patch.object(creator.platform, "system", return_value="Linux"), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES is unset"):
                creator.collect_cpu_runtime(base, base)

    def test_vendored_e3fp_version_does_not_require_distribution_metadata(self):
        root = Path(tempfile.mkdtemp()).resolve()
        package = root / "e3fp"
        package.mkdir()
        init_path = package / "__init__.py"
        pipeline_path = package / "pipeline.py"
        init_path.write_text(
            "version_info = (1, 2, 5)\n"
            "version = '.'.join(str(c) for c in version_info)\n"
            "__version__ = version\n",
            encoding="utf-8",
        )
        pipeline_path.write_text("# fixture\n", encoding="utf-8")
        try:
            with mock.patch.object(
                creator.importlib_metadata,
                "version",
                side_effect=creator.importlib_metadata.PackageNotFoundError("e3fp"),
            ):
                self.assertEqual(creator._vendored_e3fp_version(root), "1.2.5")
                self.assertEqual(creator._vendored_e3fp_version(package), "1.2.5")
        finally:
            pipeline_path.unlink()
            init_path.unlink()
            package.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main(verbosity=2)
