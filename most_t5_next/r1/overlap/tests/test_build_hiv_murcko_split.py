from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import build_hiv_murcko_split as hiv


SCAFFOLD_FAMILIES = (
    ("c1ccccc1C", "c1ccccc1N", "c1ccccc1O"),
    ("C1CCCCC1C", "C1CCCCC1N", "C1CCCCC1O"),
    ("Cc1ccncc1", "Nc1ccncc1", "Oc1ccncc1"),
    ("CC1CCNCC1", "NC1CCNCC1", "OC1CCNCC1"),
    ("Cc1ncc[nH]1", "Nc1ncc[nH]1", "Oc1ncc[nH]1"),
    ("CC1CCOC1", "NC1CCOC1", "OC1CCOC1"),
    ("Cc1ccc2ccccc2c1", "Nc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1"),
    ("CC1CCC2CCCCC2C1", "NC1CCC2CCCCC2C1", "OC1CCC2CCCCC2C1"),
    ("Cc1ccc2[nH]ccc2c1", "Nc1ccc2[nH]ccc2c1", "Oc1ccc2[nH]ccc2c1"),
    ("CC1=CC=NN1", "NC1=CC=NN1", "OC1=CC=NN1"),
)


def write_csv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(hiv.EXPECTED_COLUMNS)
        for smiles, activity, label in rows:
            writer.writerow((smiles, activity, label))


def fixture_rows() -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for family in SCAFFOLD_FAMILIES:
        rows.extend(
            (
                (family[0], "CI", 0),
                (family[1], "CA", 1),
                (family[2], "CI", 0),
            )
        )
    return rows


def binding_for(path: Path, member_count: int) -> hiv.SourceBinding:
    payload = path.read_bytes()
    return hiv.SourceBinding(
        revision="fixture-hiv-source-v1",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_md5=hashlib.md5(payload).hexdigest(),
        expected_bytes=len(payload),
        expected_member_count=member_count,
        source_url="https://example.invalid/HIV-fixture.csv",
        authority="test fixture",
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class HivMurckoDerivedSplitTests(unittest.TestCase):
    def test_deepchem_order_full_coverage_and_protected_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "HIV.csv"
            write_csv(source, fixture_rows())
            output = root / "derived"
            result = hiv.build_hiv_murcko_split(
                source, output, source_binding=binding_for(source, 30)
            )

            self.assertEqual(result["protocol_id"], hiv.PROTOCOL_ID)
            self.assertFalse(result["official_exact_split_reproduction"])
            self.assertEqual(
                result["counts"]["member_counts"],
                {"train": 24, "validation": 3, "test": 3},
            )
            self.assertTrue(
                result["counts"]["invariants"][
                    "bemis_murcko_scaffolds_split_disjoint"
                ]
            )
            for counts in result["counts"]["class_counts"].values():
                self.assertGreater(counts["negative_0"], 0)
                self.assertGreater(counts["positive_1"], 0)

            member_rows = read_jsonl(output / hiv.MEMBER_MANIFEST_FILENAME)
            protected_rows = read_jsonl(output / hiv.PROTECTED_ROWS_FILENAME)
            protected_manifest = read_json(output / hiv.PROTECTED_MANIFEST_FILENAME)
            self.assertEqual(len(member_rows), 30)
            self.assertEqual(
                {row["source_member_index"] for row in member_rows}, set(range(30))
            )
            protected_member_ids = {
                member_id
                for row in protected_rows
                for member_id in row["source_member_ids"]
            }
            expected_ids = {
                row["member_id"]
                for row in member_rows
                if row["assigned_split"] in ("validation", "test")
            }
            self.assertEqual(protected_member_ids, expected_ids)
            self.assertEqual(protected_manifest["protected_splits"], ["validation", "test"])
            self.assertTrue(
                protected_manifest["training_split_is_not_protected_by_this_manifest"]
            )

            groups = result["scaffold_groups_in_assignment_order"]
            self.assertEqual(
                [group["first_source_member_index"] for group in groups],
                list(range(27, -1, -3)),
            )
            self.assertIsNone(result["algorithm"]["randomness"]["seed"])

    def test_artifacts_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "HIV.csv"
            write_csv(source, fixture_rows())
            binding = binding_for(source, 30)
            first = root / "first"
            second = root / "second"
            hiv.build_hiv_murcko_split(source, first, source_binding=binding)
            hiv.build_hiv_murcko_split(source, second, source_binding=binding)
            for name in (
                hiv.SOURCE_MANIFEST_FILENAME,
                hiv.SPLIT_MANIFEST_FILENAME,
                hiv.MEMBER_MANIFEST_FILENAME,
                hiv.PROTECTED_ROWS_FILENAME,
                hiv.PROTECTED_MANIFEST_FILENAME,
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_hash_invalid_smiles_and_label_mismatch_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "HIV.csv"
            write_csv(source, fixture_rows())
            binding = binding_for(source, 30)
            bad_binding = hiv.SourceBinding(
                revision=binding.revision,
                expected_sha256="0" * 64,
                source_url=binding.source_url,
            )
            with self.assertRaisesRegex(hiv.HivSplitProtocolError, "SHA-256"):
                hiv.build_hiv_murcko_split(
                    source, root / "hash-failure", source_binding=bad_binding
                )
            self.assertFalse((root / "hash-failure").exists())

            write_csv(source, [("not-a-smiles", "CI", 0)] + fixture_rows()[1:])
            with self.assertRaisesRegex(hiv.HivSplitProtocolError, "invalid SMILES"):
                hiv.build_hiv_murcko_split(
                    source,
                    root / "smiles-failure",
                    source_binding=binding_for(source, 30),
                )
            self.assertFalse((root / "smiles-failure").exists())

            write_csv(source, [("C", "CI", 1)] + fixture_rows()[1:])
            with self.assertRaisesRegex(hiv.HivSplitProtocolError, "inconsistent"):
                hiv.build_hiv_murcko_split(
                    source,
                    root / "label-failure",
                    source_binding=binding_for(source, 30),
                )
            self.assertFalse((root / "label-failure").exists())

    def test_production_cli_accepts_only_frozen_official_binding(self):
        parsed = hiv.parse_args(
            [
                "--source-csv",
                "HIV.csv",
                "--source-sha256",
                "0" * 64,
                "--source-revision",
                "unbound",
                "--output-dir",
                "derived",
            ]
        )
        with self.assertRaisesRegex(hiv.HivSplitProtocolError, "frozen official"):
            hiv._require_official_cli_binding(parsed)


if __name__ == "__main__":
    unittest.main()
