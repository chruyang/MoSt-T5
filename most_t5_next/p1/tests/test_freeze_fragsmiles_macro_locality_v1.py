from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1.freeze_fragsmiles_macro_locality_v1 import freeze_registry


def _census(root: Path, identities: list[list[str]]) -> None:
    root.mkdir()
    counts: dict[str, int] = {}
    for record in identities:
        for identity in record:
            counts[identity] = counts.get(identity, 0) + 1
    with (root / "fragment_census.jsonl").open("w", encoding="utf-8") as handle:
        for rank, (identity, count) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ):
            handle.write(json.dumps({"rank": rank, "fragment_identity": identity, "occurrences": count}) + "\n")
    with gzip.open(root / "molecule_fragments.jsonl.gz", "wt", encoding="utf-8") as handle:
        for index, record in enumerate(identities):
            handle.write(json.dumps({"selection_index": index, "mode": "compact", "fragment_identities": record, "fragment_macro_eligible": [True] * len(record)}) + "\n")


class FreezeMacroLocalityV1Tests(unittest.TestCase):
    def test_freeze_reranks_kept_rows_and_reports_lossless_record_coverage(self):
        # Keep the hermetic fixture inside the writable repository root.  Some
        # Windows test hosts expose the user TEMP directory read-only.
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw:
            root = Path(raw)
            candidate = root / "candidate.jsonl"
            rows = [
                {"rank": 0, "surface_token": "old0", "fragment_identity": "C", "selection_role": "base", "phase1_train_occurrences": 1},
                {"rank": 1, "surface_token": "old1", "fragment_identity": "CCCC", "selection_role": "base", "phase1_train_occurrences": 1},
            ]
            candidate.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            for name in ("p1", "p2", "chebi"):
                _census(root / name, [["C", "CCCC"]])
            manifest = freeze_registry(
                candidate_registry=candidate,
                phase1_census=root / "p1",
                phase2_census=root / "p2",
                chebi20_census=root / "chebi",
                output_dir=root / "out",
                max_atoms=2,
                max_glyphs=64,
            )
            frozen = [json.loads(line) for line in (root / "out" / "macro_registry.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(frozen), 1)
            self.assertEqual(frozen[0]["rank"], 0)
            self.assertEqual(frozen[0]["candidate_rank"], 0)
            self.assertEqual(frozen[0]["surface_token"], "<MOST:FM:000000>")
            coverage = manifest["record_level_train_coverage"]["phase1_train"]
            self.assertEqual(coverage["macro_occurrence_coverage"], 0.5)
            self.assertEqual(coverage["semantic_fragment_fallback_occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
