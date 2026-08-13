from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from most_t5_next.p1.audit_fragsmiles_compact_stereo_domains_v1 import run_audit


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class CompactStereoDomainAuditTests(unittest.TestCase):
    def test_jsonl_audit_retains_source_index_and_classifies_source_failure(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tmp") as temp_text:
            temp = Path(temp_text)
            source = temp / "source.jsonl"
            source.write_text(
                "\n".join(
                    (
                        json.dumps({"smiles": "N[C@@H](C)C(=O)O"}),
                        json.dumps({"smiles": "F/C=C/F"}),
                        json.dumps({"wrong": "CC"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = temp / "audit"
            report = run_audit(
                input_path=source,
                input_format="jsonl",
                smiles_field="smiles",
                chemicalgof_root=CHEMICALGOF_ROOT,
                output_dir=output,
                workers=1,
                max_pending=1,
                max_records=None,
                progress_every=0,
                record_timeout_seconds=None,
            )
            self.assertEqual(report["counts"]["input"], 3)
            self.assertEqual(report["counts"]["pass"], 2)
            self.assertEqual(report["counts"]["reject"], 1)
            rejects = [
                json.loads(line)
                for line in (output / "rejects.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(sorted(row["source_index"] for row in rejects), [2])

    def test_max_records_stops_source_iteration(self):
        def guarded_source(*_args, **_kwargs):
            yield 0, "CC", None
            yield 1, "CO", None
            raise AssertionError("source was consumed beyond max_records")

        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tmp") as temp_text:
            temp = Path(temp_text)
            source = temp / "placeholder.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            with patch(
                "most_t5_next.p1.audit_fragsmiles_compact_stereo_domains_v1._iter_source",
                guarded_source,
            ):
                report = run_audit(
                    input_path=source,
                    input_format="jsonl",
                    smiles_field="smiles",
                    chemicalgof_root=CHEMICALGOF_ROOT,
                    output_dir=temp / "audit",
                    workers=1,
                    max_pending=1,
                    max_records=2,
                    progress_every=0,
                    record_timeout_seconds=None,
                )
            self.assertEqual(report["counts"]["input"], 2)
            self.assertEqual(report["counts"]["pass"], 2)


if __name__ == "__main__":
    unittest.main()
