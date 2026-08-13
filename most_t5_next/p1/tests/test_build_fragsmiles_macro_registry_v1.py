from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1.build_fragsmiles_macro_registry_v1 import (
    BASE_T5_VOCAB_SIZE,
    NON_MACRO_ADDITIONS,
    SHARED_MOLECULE_BOUNDARY_TOKENS,
    build_registry,
    equal_stage_ranking,
    non_macro_token_universe,
)


def _write_census(
    root: Path,
    counts: dict[str, int],
    molecules: list[list[str]],
    *,
    write_eligibility: bool = True,
) -> None:
    root.mkdir()
    ranked = sorted(counts, key=lambda identity: (-counts[identity], identity.encode()))
    with (root / "fragment_census.jsonl").open("w", encoding="utf-8") as handle:
        for rank, identity in enumerate(ranked):
            handle.write(
                json.dumps(
                    {
                        "rank": rank,
                        "fragment_identity": identity,
                        "occurrences": counts[identity],
                    }
                )
                + "\n"
            )
    with gzip.open(root / "molecule_fragments.jsonl.gz", "wt", encoding="utf-8") as handle:
        for index, identities in enumerate(molecules):
            row = {
                "selection_index": index,
                "source_index": index,
                "mode": "compact",
                "fragment_identities": identities,
            }
            if write_eligibility:
                row["fragment_macro_eligible"] = [True] * len(identities)
            handle.write(json.dumps(row) + "\n")
    census_path = root / "fragment_census.jsonl"
    cache_path = root / "molecule_fragments.jsonl.gz"
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "most-t5-next/fragsmiles-fragment-census/v1",
                "status": "pass",
                "training_admission": False,
                "counts": {
                    "processed_records": len(molecules),
                    "fragment_occurrences": sum(counts.values()),
                },
                "artifacts": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (census_path, cache_path)
                },
            }
        ),
        encoding="utf-8",
    )


class FragSmilesMacroRegistryTest(unittest.TestCase):
    def test_non_macro_universe_is_exact_and_disjoint(self):
        tokens = non_macro_token_universe()
        self.assertEqual(len(tokens), NON_MACRO_ADDITIONS)
        self.assertEqual(len(set(tokens)), len(tokens))

    def test_equal_stage_mass_is_exact(self):
        self.assertEqual(
            equal_stage_ranking(
                {"A": 10, "B": 5, "C": 1}, {"B": 10, "D": 9}
            ),
            ("B", "A", "D", "C"),
        )

    def test_general_then_all_chebi_train_extension(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            p1 = root / "p1"
            p2 = root / "p2"
            chebi = root / "chebi"
            _write_census(
                p1,
                {"A": 2, "B": 1},
                [["A"], ["A", "B"]],
                write_eligibility=False,
            )
            _write_census(p2, {"B": 2, "D": 1}, [["B"], ["B", "D"]])
            _write_census(chebi, {"A": 1, "E": 2}, [["A", "E"], ["E"]])
            output = root / "registry"
            manifest = build_registry(
                phase1_census=p1,
                phase2_census=p2,
                chebi_train_census=chebi,
                output_dir=output,
                general_budget=2,
            )
            rows = [
                json.loads(line)
                for line in (output / "macro_registry.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([row["fragment_identity"] for row in rows], ["B", "A", "E"])
            self.assertEqual(
                [row["surface_token"] for row in rows],
                ["<MOST:FM:000000>", "<MOST:FM:000001>", "<MOST:FM:000002>"],
            )
            self.assertEqual(manifest["counts"]["general_macros"], 2)
            self.assertEqual(manifest["counts"]["chebi20_train_extension_macros"], 1)
            self.assertEqual(
                manifest["counts"]["projected_union_vocabulary"],
                BASE_T5_VOCAB_SIZE
                + NON_MACRO_ADDITIONS
                + len(SHARED_MOLECULE_BOUNDARY_TOKENS)
                + 3,
            )
            self.assertEqual(
                manifest["smiles_vocabulary"]["total_molecular_glyphs"], 160
            )
            self.assertEqual(
                manifest["smiles_vocabulary"]["utf8_byte_token_rows"], 0
            )
            self.assertEqual(
                manifest["smiles_vocabulary"]["fallback_class_token_rows"], 0
            )
            self.assertEqual(
                manifest["coverage"]["chebi20_train"]["macro_occurrence_coverage"],
                1.0,
            )
            self.assertEqual(
                manifest["coverage"]["phase1_train"][
                    "legacy_rows_without_explicit_fragment_eligibility"
                ],
                2,
            )


if __name__ == "__main__":
    unittest.main()
