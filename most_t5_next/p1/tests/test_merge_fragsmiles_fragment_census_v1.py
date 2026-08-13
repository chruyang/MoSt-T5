from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1.build_fragsmiles_fragment_census_v1 import run_census
from most_t5_next.p1.merge_fragsmiles_fragment_census_v1 import (
    FragSmilesCensusMergeError,
    merge_census_shards,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class FragSmilesFragmentCensusMergeTest(unittest.TestCase):
    def _run(
        self, root: Path, source: Path, name: str, start: int, count: int
    ) -> Path:
        output = root / name
        run_census(
            input_path=source,
            input_format="jsonl",
            smiles_field="smiles",
            chemicalgof_root=CHEMICALGOF_ROOT,
            output_dir=output,
            workers=1,
            max_pending=1,
            expected_records=count,
            start_record=start,
            max_records=count,
            progress_every=0,
            record_timeout_seconds=None,
        )
        return output

    def test_two_contiguous_shards_equal_monolithic_counts_and_order(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"smiles": smiles}) + "\n"
                    for smiles in ("CCO", "c1ccccc1", "CCO", "CCN", "CC")
                ),
                encoding="utf-8",
            )
            shard0 = self._run(root, source, "shard0", 0, 2)
            shard1 = self._run(root, source, "shard1", 2, 3)
            full = self._run(root, source, "full", 0, 5)
            merged = root / "merged"
            manifest = merge_census_shards(
                shard_dirs=[shard1, shard0],
                output_dir=merged,
                expected_records=5,
            )
            full_manifest = json.loads(
                (full / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(manifest["counts"], full_manifest["counts"])
            self.assertEqual(
                (merged / "fragment_census.jsonl").read_text(encoding="utf-8"),
                (full / "fragment_census.jsonl").read_text(encoding="utf-8"),
            )
            with gzip.open(
                merged / "molecule_fragments.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["selection_index"] for row in rows], list(range(5)))
            self.assertEqual([row["source_index"] for row in rows], list(range(5)))

    def test_gap_between_shards_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps({"smiles": "CC"}) + "\n" for _ in range(5)),
                encoding="utf-8",
            )
            shard0 = self._run(root, source, "shard0", 0, 2)
            shard1 = self._run(root, source, "shard1", 3, 2)
            with self.assertRaises(FragSmilesCensusMergeError):
                merge_census_shards(
                    shard_dirs=[shard0, shard1],
                    output_dir=root / "merged",
                    expected_records=5,
                )


if __name__ == "__main__":
    unittest.main()
