from __future__ import print_function

import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import derive_clean_pretrain_membership_v1 as generic
from most_t5_next.r1.overlap import derive_phase2_chebi20_clean_membership_v1 as phase2
from most_t5_next.r1.overlap.tests.test_derive_clean_pretrain_membership_v1 import (
    make_collection,
    read_jsonl,
    write_json,
)


def rewrite_fields(path, fields):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(fields)
    write_json(path, manifest)
    return path


class Phase2Chebi20MembershipTests(unittest.TestCase):
    def make_world(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pretrain, _ = make_collection(
            root,
            phase2.EXPECTED_PRETRAIN["collection_id"],
            "p2_permitted_train_membership",
            "train",
            "none",
            [
                ("pubchem_cid:1", "A", "A-pretrain"),
                ("pubchem_cid:2", "C", "C-different-stereo"),
                ("pubchem_cid:3", "C", "C-pretrain-stereo"),
            ],
        )
        test, _ = make_collection(
            root,
            phase2.EXPECTED_PROTECTED["test"]["collection_id"],
            "downstream_test",
            "test",
            "text_to_molecule_generation",
            [("chebi_cid:t", "C", "C-pretrain-stereo")],
        )
        rewrite_fields(pretrain, phase2.EXPECTED_PRETRAIN)
        rewrite_fields(test, phase2.EXPECTED_PROTECTED["test"])
        return root, pretrain, test

    def test_exact_policy_excludes_test_matches_by_nonstereo_connectivity(self):
        root, pretrain, test = self.make_world()
        output = root / "derived"
        manifest, receipt = phase2.derive_phase2_chebi20_clean_membership(
            pretrain, [test], output
        )
        self.assertEqual(
            [row["member_id"] for row in read_jsonl(output / generic.PERMITTED_FILENAME)],
            ["pubchem_cid:1"],
        )
        self.assertEqual(
            [row["member_id"] for row in read_jsonl(output / generic.EXCLUDED_FILENAME)],
            ["pubchem_cid:2", "pubchem_cid:3"],
        )
        self.assertEqual(manifest["counts"]["excluded_member_count"], 2)
        self.assertEqual(
            manifest["policy"]["hard_exclusion_key"],
            "connectivity_identity_sha256",
        )
        self.assertEqual(manifest["policy"]["protected_roles"], ["downstream_test"])
        self.assertEqual(receipt["exclusion"]["action"], "exclude_entire_phase2_record")
        self.assertTrue(receipt["exclusion"]["connectivity_used_for_exclusion"])
        self.assertFalse(receipt["exclusion"]["stereo_used_for_exclusion"])
        self.assertFalse(receipt["scope"]["objective_specific_exceptions"])
        self.assertEqual(receipt["scope"]["other_downstream_datasets_used_for_exclusion"], [])

    def test_missing_protected_split_is_rejected_before_output(self):
        root, pretrain, _ = self.make_world()
        output = root / "derived"
        with self.assertRaisesRegex(ValueError, "exactly one protected manifest"):
            phase2.derive_phase2_chebi20_clean_membership(
                pretrain, [], output
            )
        self.assertFalse(output.exists())

    def test_non_chebi_protected_collection_is_rejected(self):
        root, pretrain, test = self.make_world()
        wrong = json.loads(test.read_text(encoding="utf-8"))
        wrong["dataset_id"] = "another-downstream-task"
        write_json(test, wrong)
        output = root / "derived"
        with self.assertRaisesRegex(ValueError, "frozen Phase-II/ChEBI-20 policy"):
            phase2.derive_phase2_chebi20_clean_membership(
                pretrain, [test], output
            )
        self.assertFalse(output.exists())

    def test_wrong_phase2_source_is_rejected(self):
        root, pretrain, test = self.make_world()
        wrong = json.loads(pretrain.read_text(encoding="utf-8"))
        wrong["release_id"] = "another-phase2-release"
        write_json(pretrain, wrong)
        output = root / "derived"
        with self.assertRaisesRegex(ValueError, "frozen Phase-II/ChEBI-20 policy"):
            phase2.derive_phase2_chebi20_clean_membership(
                pretrain, [test], output
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
