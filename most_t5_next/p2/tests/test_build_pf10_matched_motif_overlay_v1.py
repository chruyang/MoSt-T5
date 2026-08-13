from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest import mock
import uuid

from most_t5_next.p2 import build_pf10_matched_motif_overlay_v1 as subject
from most_t5_next.p2.tests.test_matched_motif_state_donor_v1 import _document


class _Reader:
    def __init__(self, _root: Path, *, lmdb_module=None) -> None:
        del lmdb_module
        self.dev_member_count = 3
        self.documents = (
            _document("left", ["shared"], [[0]], [[]], [], [[1, 2, 3, 4]]),
            _document("right", ["shared"], [[0]], [[]], [], [[11, 12, 13, 14]]),
            _document("single", ["unique"], [[0]], [[]], [], [[21, 22, 23, 24]]),
        )

    def iter_donor_atom_maps(self, *, split: str):
        self.assert_split(split)
        for index, document in enumerate(self.documents):
            yield {
                "member_id": document["member"]["member_id"],
                "storage_key": f"key-{index}",
                "overlay_planning_sidecar": copy.deepcopy(
                    document["overlay_planning_sidecar"]
                ),
            }

    def iter_raw_motif_documents(self, *, split: str):
        self.assert_split(split)
        for index, source in enumerate(self.documents):
            document = copy.deepcopy(source)
            document.pop("overlay_planning_sidecar")
            yield {
                "member_id": document["member"]["member_id"],
                "storage_key": f"key-{index}",
            }, document

    @staticmethod
    def assert_split(split: str) -> None:
        if split != "dev":
            raise AssertionError("fixture only exposes dev")


class PF10MatchedMotifOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path.cwd() / ("pf10_overlay_test_" + uuid.uuid4().hex)
        self.temporary.mkdir()

    def tearDown(self) -> None:
        for directory_name in ("overlay", "overlay.staging"):
            output = self.temporary / directory_name
            for name in (subject.MANIFEST_NAME, subject.ROWS_NAME):
                path = output / name
                if path.is_file():
                    path.unlink()
            if output.is_dir():
                output.rmdir()
        if self.temporary.is_dir():
            self.temporary.rmdir()

    def test_build_and_provider_keep_unmatched_motif_aligned(self) -> None:
        output = self.temporary / "overlay"
        with mock.patch.object(subject, "PF1PairedReleaseReader", _Reader):
            manifest = subject.build_pf10_matched_motif_overlay(
                paired_release=self.temporary / "paired",
                output_dir=output,
            )
        provider = subject.MatchedMotifStateProvider(output)

        self.assertEqual(provider.get("left"), ((11, 12, 13, 14),))
        self.assertEqual(provider.get("right"), ((1, 2, 3, 4),))
        self.assertEqual(provider.get("single"), ((21, 22, 23, 24),))
        self.assertEqual(provider.changed_motif_indices("left"), (0,))
        self.assertEqual(provider.changed_motif_indices("right"), (0,))
        self.assertEqual(provider.changed_motif_indices("single"), ())
        self.assertEqual(manifest["counts"]["published_rows"], 3)
        self.assertEqual(manifest["counts"]["records_with_any_matched_motif"], 2)
        self.assertEqual(manifest["coverage"]["eligible_motif_occurrences"], 2)
        self.assertTrue(
            manifest["semantics"]["unmatched_motifs_keep_aligned_state"]
        )
        provider.close()

    def test_provider_rejects_missing_record(self) -> None:
        output = self.temporary / "overlay"
        with mock.patch.object(subject, "PF1PairedReleaseReader", _Reader):
            subject.build_pf10_matched_motif_overlay(
                paired_release=self.temporary / "paired",
                output_dir=output,
            )
        provider = subject.MatchedMotifStateProvider(output)
        with self.assertRaisesRegex(
            subject.PF10MatchedMotifOverlayError, "absent"
        ):
            provider.get("missing")
        provider.close()


if __name__ == "__main__":
    unittest.main()
