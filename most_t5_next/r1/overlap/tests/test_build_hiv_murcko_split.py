from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def binding_for(
    path: Path,
    member_count: int,
    *,
    invalid_member_count: int = 0,
) -> hiv.SourceBinding:
    payload = path.read_bytes()
    return hiv.SourceBinding(
        revision="fixture-hiv-source-v1",
        reference_sha256=hashlib.sha256(payload).hexdigest(),
        reference_md5=hashlib.md5(payload).hexdigest(),
        reference_bytes=len(payload),
        expected_member_count=member_count,
        expected_eligible_member_count=member_count - invalid_member_count,
        expected_invalid_member_count=invalid_member_count,
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
            self.assertEqual(result["counts"]["source_population"], 30)
            self.assertEqual(result["counts"]["eligible_population"], 30)
            self.assertEqual(result["counts"]["excluded_rdkit_invalid"], 0)
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
                hiv.INVALID_MEMBER_LEDGER_FILENAME,
                hiv.PROTECTED_ROWS_FILENAME,
                hiv.PROTECTED_MANIFEST_FILENAME,
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_hash_is_observational_and_invalid_smiles_are_ledgered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "HIV.csv"
            write_csv(source, fixture_rows())
            binding = binding_for(source, 30)
            bad_binding = hiv.SourceBinding(
                revision=binding.revision,
                reference_sha256="not-even-a-valid-digest",
                source_url=binding.source_url,
                expected_member_count=30,
                expected_eligible_member_count=30,
                expected_invalid_member_count=0,
            )
            hash_output = root / "hash-observation"
            hiv.build_hiv_murcko_split(
                source, hash_output, source_binding=bad_binding
            )
            source_manifest = read_json(hash_output / hiv.SOURCE_MANIFEST_FILENAME)
            self.assertFalse(
                source_manifest["file_integrity_observations"]["matches_reference"][
                    "sha256"
                ]
            )
            self.assertIn(
                "file_sha256", source_manifest["admission_criteria"]["not_hard"]
            )

            write_csv(source, [("not-a-smiles", "CI", 0)] + fixture_rows()[1:])
            invalid_output = root / "invalid-ledger"
            result = hiv.build_hiv_murcko_split(
                source,
                invalid_output,
                source_binding=binding_for(source, 30, invalid_member_count=1),
            )
            invalid_rows = read_jsonl(
                invalid_output / hiv.INVALID_MEMBER_LEDGER_FILENAME
            )
            eligible_rows = read_jsonl(invalid_output / hiv.MEMBER_MANIFEST_FILENAME)
            self.assertEqual(result["counts"]["source_population"], 30)
            self.assertEqual(result["counts"]["eligible_population"], 29)
            self.assertEqual(result["counts"]["excluded_rdkit_invalid"], 1)
            self.assertEqual(len(invalid_rows), 1)
            self.assertEqual(invalid_rows[0]["source_member_index"], 0)
            self.assertFalse(invalid_rows[0]["eligible_for_model_or_metric"])
            self.assertIsNone(invalid_rows[0]["assigned_split"])
            self.assertNotIn(
                invalid_rows[0]["member_id"],
                {row["member_id"] for row in eligible_rows},
            )

    def test_label_mismatch_and_population_mismatch_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "HIV.csv"

            write_csv(source, [("C", "CI", 1)] + fixture_rows()[1:])
            with self.assertRaisesRegex(hiv.HivSplitProtocolError, "inconsistent"):
                hiv.build_hiv_murcko_split(
                    source,
                    root / "label-failure",
                    source_binding=binding_for(source, 30),
                )
            self.assertFalse((root / "label-failure").exists())

            write_csv(source, fixture_rows())
            with self.assertRaisesRegex(hiv.HivSplitProtocolError, "population"):
                hiv.build_hiv_murcko_split(
                    source,
                    root / "population-failure",
                    source_binding=binding_for(source, 31),
                )
            self.assertFalse((root / "population-failure").exists())

    def test_production_cli_gates_url_and_revision_but_not_sha(self):
        parsed = hiv.parse_args(
            [
                "--source-csv",
                "HIV.csv",
                "--source-url",
                hiv.OFFICIAL_SOURCE_URL,
                "--source-sha256",
                "wrong-observation-does-not-block",
                "--source-revision",
                hiv.OFFICIAL_SOURCE_REVISION,
                "--output-dir",
                "derived",
            ]
        )
        hiv._require_official_cli_binding(parsed)

        no_sha = hiv.parse_args(
            [
                "--source-csv",
                "HIV.csv",
                "--source-url",
                hiv.OFFICIAL_SOURCE_URL,
                "--source-revision",
                hiv.OFFICIAL_SOURCE_REVISION,
                "--output-dir",
                "derived",
            ]
        )
        self.assertIsNone(no_sha.source_sha256)
        hiv._require_official_cli_binding(no_sha)

        parsed.source_revision = "unbound"
        with self.assertRaisesRegex(hiv.HivSplitProtocolError, "official HIV.csv revision"):
            hiv._require_official_cli_binding(parsed)

    def test_official_protocol_pins_rdkit_canonicalization_version(self):
        with mock.patch.object(
            hiv.rdBase, "rdkitVersion", hiv.PRODUCTION_RDKIT_VERSION
        ):
            hiv.require_official_rdkit_version(hiv.OFFICIAL_SOURCE_BINDING)
        with mock.patch.object(hiv.rdBase, "rdkitVersion", "2025.09.1"):
            with self.assertRaisesRegex(hiv.HivSplitProtocolError, "requires RDKit"):
                hiv.require_official_rdkit_version(hiv.OFFICIAL_SOURCE_BINDING)

    def test_canonical_forms_use_shared_explicit_h_projection(self):
        explicit = hiv.canonical_forms("C[C@]([H])(O)F")
        implicit = hiv.canonical_forms("C[C@H](O)F")
        self.assertIsNotNone(explicit)
        self.assertEqual(explicit, implicit)

        explicit_alkene = hiv.canonical_forms("[H]/C=C/F")
        implicit_alkene = hiv.canonical_forms("C=CF")
        self.assertIsNotNone(explicit_alkene)
        self.assertEqual(explicit_alkene, implicit_alkene)


if __name__ == "__main__":
    unittest.main()
