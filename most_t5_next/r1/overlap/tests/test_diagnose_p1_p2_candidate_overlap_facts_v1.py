from __future__ import print_function

import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import diagnose_p1_p2_candidate_overlap_facts_v1 as diagnostic
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as gate
from most_t5_next.r1.overlap.tests.test_prove_membership_identity_overlap_v1 import (
    digest,
    file_sha,
    make_collection,
)


R1_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = R1_ROOT / "contracts" / "p1_p2_candidate_overlap_fact_contract_v1.json"


class CandidateOverlapFactDiagnosticTests(unittest.TestCase):
    def make_pair(self, root, p1_rows, p2_rows, p2_connectivity_spec=None, p2_stereo_spec=None):
        p1_manifest = make_collection(
            root,
            "p1",
            "p1_structure_train",
            "train",
            "none",
            p1_rows,
            None,
        )
        p2_manifest = make_collection(
            root,
            "p2",
            "p2_permitted_train_membership",
            "train",
            "none",
            p2_rows,
            None,
            connectivity_spec=p2_connectivity_spec,
            stereo_spec=p2_stereo_spec,
        )
        return p1_manifest, p2_manifest

    def run_pair(self, root, p1_manifest, p2_manifest, output_name="report"):
        return diagnostic.run_diagnostic(
            CONTRACT,
            p1_manifest,
            file_sha(p1_manifest),
            p2_manifest,
            file_sha(p2_manifest),
            root / output_name,
        )

    def test_overlap_reports_unique_and_impacted_counts_without_admission(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        p1_manifest, p2_manifest = self.make_pair(
            root,
            [
                ("p1:1", "X", "X1", None),
                ("p1:2", "X", "X1", None),
                ("p1:3", "Y", "Y1", None),
            ],
            [
                ("p2:1", "X", "X2", None),
                ("p2:2", "Y", "Y1", None),
                ("p2:3", "Z", "Z1", None),
            ],
        )
        report = self.run_pair(root, p1_manifest, p2_manifest)

        self.assertNotIn("status", report)
        self.assertEqual(report["diagnostic_completion"], "facts_reported")
        self.assertTrue(report["diagnostic_only"])
        self.assertTrue(report["admissions"])
        self.assertTrue(all(value is False for value in report["admissions"].values()))
        connectivity = report["facts"]["connectivity_identity"]
        self.assertEqual(connectivity["overlap_unique_count"], 2)
        self.assertEqual(connectivity["p1_members_impacted"], 3)
        self.assertEqual(connectivity["p2_members_impacted"], 2)
        stereo = report["facts"]["stereo_identity"]
        self.assertEqual(stereo["overlap_unique_count"], 1)
        self.assertEqual(stereo["p1_members_impacted"], 1)
        self.assertEqual(stereo["p2_members_impacted"], 1)
        cross = report["facts"]["cross_resolution"]
        self.assertEqual(cross["p1_members_connectivity_overlap_without_stereo_match"], 2)
        self.assertEqual(cross["p2_members_connectivity_overlap_without_stereo_match"], 1)
        self.assertIsNone(cross["p1_members_molecule_overlap_without_exact_conformer_match"])
        unavailable = report["facts"]["unavailable_dimensions"]
        self.assertEqual(unavailable["conformer_identity"]["status"], "unavailable_for_comparison")
        self.assertEqual(unavailable["text_identity"]["status"], "unavailable_for_comparison")
        self.assertEqual(diagnostic.canonical_payload_sha256(report), report["report_canonical_payload_sha256"])

        report_path = root / "report" / diagnostic.REPORT_FILENAME
        raw = report_path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
        self.assertEqual(raw, gate.canonical_json_bytes(parsed) + b"\n")
        bindings = report["input_artifact_bindings"]
        self.assertEqual([item["slot"] for item in bindings], ["p1", "p2"])
        for item in bindings:
            self.assertEqual(
                item["expected_manifest_sha256"],
                item["strict_load_observation"]["manifest_sha256"],
            )

    def test_no_overlap_is_reported_as_fact_not_policy_pass(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        p1_manifest, p2_manifest = self.make_pair(
            root,
            [("p1:1", "A", "A1", None)],
            [("p2:1", "B", "B1", None)],
        )
        report = self.run_pair(root, p1_manifest, p2_manifest)
        self.assertEqual(report["facts"]["connectivity_identity"]["overlap_unique_count"], 0)
        self.assertEqual(report["facts"]["stereo_identity"]["overlap_unique_count"], 0)
        self.assertFalse(report["admissions"]["p1_p2_policy_compliance_proven"])
        self.assertNotIn("policy", report)

    def test_external_manifest_sha_mismatch_is_rejected_by_strict_loader(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        p1_manifest, p2_manifest = self.make_pair(
            root,
            [("p1:1", "A", "A1", None)],
            [("p2:1", "B", "B1", None)],
        )
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 differs from request binding"):
            diagnostic.run_diagnostic(
                CONTRACT,
                p1_manifest,
                "0" * 64,
                p2_manifest,
                file_sha(p2_manifest),
                root / "bad-sha-report",
            )

    def test_wrong_slot_role_is_rejected(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        p1_manifest, p2_manifest = self.make_pair(
            root,
            [("p1:1", "A", "A1", None)],
            [("p2:1", "B", "B1", None)],
        )
        with self.assertRaisesRegex(ValueError, "p1 manifest role must be p1_structure_train"):
            diagnostic.run_diagnostic(
                CONTRACT,
                p2_manifest,
                file_sha(p2_manifest),
                p1_manifest,
                file_sha(p1_manifest),
                root / "bad-role-report",
            )

    def test_connectivity_or_stereo_spec_mismatch_is_rejected(self):
        for field in ("connectivity", "stereo"):
            with self.subTest(field=field):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                root = Path(temporary.name)
                kwargs = {
                    "p2_connectivity_spec": digest("different-connectivity-spec") if field == "connectivity" else None,
                    "p2_stereo_spec": digest("different-stereo-spec") if field == "stereo" else None,
                }
                p1_manifest, p2_manifest = self.make_pair(
                    root,
                    [("p1:1", "A", "A1", None)],
                    [("p2:1", "A", "A1", None)],
                    **kwargs
                )
                with self.assertRaisesRegex(ValueError, "values differ"):
                    self.run_pair(root, p1_manifest, p2_manifest, "bad-spec-report")


if __name__ == "__main__":
    unittest.main()
