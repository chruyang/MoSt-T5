from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from most_t5_next.r1.overlap import build_controlled_editing_memberships_v1 as editing
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


TEST_SMILES = ("CCO", "N#N", "c1ccccc1", "C[C@H](O)F")
ZINC_SMILES = (
    "OCC",  # Same connectivity as the first sealed-test molecule.
    "CCN",
    "NCC",  # Same connectivity as the preceding ZINC row.
    "CCC",
    "CCCl\n",  # The published CSV contains cells needing strip().
    "CCBr",
    "CCF",
    "COC",
    "CNC",
    "CC(=O)O",
    "CC#N",
    "C1CC1",
    "C1CO1",
    "c1ccncc1",
)


def write_test_source(path: Path, smiles_values=TEST_SMILES) -> None:
    path.write_text("\n".join(smiles_values) + "\n", encoding="utf-8")


def write_zinc_source(path: Path, smiles_values=ZINC_SMILES) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(editing.ZINC_COLUMNS)
        for index, smiles in enumerate(smiles_values):
            writer.writerow((smiles, str(index / 10), "0.5", "2.0"))


def fixture_binding(
    *,
    test_count: int = len(TEST_SMILES),
    zinc_count: int = len(ZINC_SMILES),
    validation_count: int = 4,
    seed: int = 42,
) -> editing.SourceBinding:
    return editing.SourceBinding(
        repository_id="fixture/MoleculeSTM",
        repository_url="https://example.invalid/fixture-MoleculeSTM",
        revision="fixture-revision-v1",
        expected_test_source_count=test_count,
        expected_zinc_source_count=zinc_count,
        validation_member_count=validation_count,
        selection_seed=seed,
        authority="unit-test fixture",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ControlledEditingMembershipTests(unittest.TestCase):
    def make_sources(self, root: Path):
        sealed_test = root / "single_multi_property_SMILES.txt"
        zinc_csv = root / "250k_rndm_zinc_drugs_clean_3.csv"
        write_test_source(sealed_test)
        write_zinc_source(zinc_csv)
        return sealed_test, zinc_csv

    def build(self, root: Path, name: str = "derived"):
        sealed_test, zinc_csv = self.make_sources(root)
        output = root / name
        summary = editing.build_controlled_editing_memberships(
            sealed_test,
            zinc_csv,
            output,
            source_binding=fixture_binding(),
        )
        return output, summary

    def test_canonical_forms_apply_the_declared_explicit_h_projection(self):
        explicit = editing.canonical_forms("C[C@]([H])(O)F")
        implicit = editing.canonical_forms("C[C@H](O)F")
        self.assertIsNotNone(explicit)
        self.assertEqual(explicit, implicit)

    def test_builds_sealed_test_and_connectivity_disjoint_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, summary = self.build(Path(temporary))

            self.assertEqual(
                summary["membership_protocol"]["sealed_test_source_members"], 4
            )
            self.assertEqual(summary["membership_protocol"]["validation_members"], 4)
            self.assertEqual(summary["prompt_namespace"]["prompt_ids"], list(editing.PROMPT_IDS))
            self.assertEqual(
                summary["prompt_namespace"]["pairing_status"],
                "namespace_only_no_cartesian_or_explicit_molecule_prompt_pairs_assumed",
            )
            self.assertEqual(summary["census"]["zinc_source_members_scanned"], 14)
            self.assertEqual(summary["census"]["zinc_connectivity_duplicates_removed"], 1)
            self.assertEqual(
                summary["census"]["zinc_unique_connectivities_excluded_by_test"], 1
            )
            self.assertTrue(summary["invariants"]["validation_test_connectivity_disjoint"])
            self.assertTrue(
                summary["invariants"]
                ["sealed_test_used_only_for_connectivity_exclusion_from_validation"]
            )
            self.assertTrue(
                summary["invariants"]
                ["sealed_test_not_used_for_model_or_hyperparameter_selection"]
            )
            self.assertIsNone(summary["membership_protocol"]["supervised_train"])

            validation_manifest = read_json(
                output / "collections" / "validation" / "collection_manifest.json"
            )
            test_manifest = read_json(
                output / "collections" / "test" / "collection_manifest.json"
            )
            for manifest, split, count in (
                (validation_manifest, "validation", 4),
                (test_manifest, "test", 4),
            ):
                proof.validate_collection_manifest(manifest)
                self.assertEqual(manifest["split"], split)
                self.assertEqual(manifest["role"], "downstream_" + split)
                self.assertEqual(manifest["molecule_rows"]["row_count"], count)
                self.assertIsNone(manifest["text_pair_rows"])
                self.assertEqual(
                    manifest["identity_specs"]["text_identity"]["status"],
                    "unavailable",
                )

            validation_rows = read_jsonl(
                output / "collections" / "validation" / "molecule_identity_rows.jsonl"
            )
            test_rows = read_jsonl(
                output / "collections" / "test" / "molecule_identity_rows.jsonl"
            )
            validation_connectivity = {
                row["connectivity_identity_sha256"] for row in validation_rows
            }
            test_connectivity = {
                row["connectivity_identity_sha256"] for row in test_rows
            }
            self.assertEqual(len(validation_connectivity), 4)
            self.assertFalse(validation_connectivity & test_connectivity)
            self.assertEqual(len({row["member_id"] for row in test_rows}), 4)
            self.assertEqual(len({row["member_id"] for row in validation_rows}), 4)

    def test_splitmix_membership_and_all_output_bytes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed_test, zinc_csv = self.make_sources(root)
            outputs = []
            for name in ("first", "second"):
                output = root / name
                editing.build_controlled_editing_memberships(
                    sealed_test,
                    zinc_csv,
                    output,
                    source_binding=fixture_binding(),
                )
                outputs.append(output)

            first_rows = read_jsonl(
                outputs[0]
                / "collections"
                / "validation"
                / "molecule_identity_rows.jsonl"
            )
            self.assertEqual(
                [row["member_id"] for row in first_rows],
                [
                    "zinc250k-source-row:000008",
                    "zinc250k-source-row:000009",
                    "zinc250k-source-row:000011",
                    "zinc250k-source-row:000012",
                ],
            )

            first_files = {
                path.relative_to(outputs[0]).as_posix(): path.read_bytes()
                for path in outputs[0].rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(outputs[1]).as_posix(): path.read_bytes()
                for path in outputs[1].rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_sealed_test_requires_exact_nonempty_unique_parseable_rows(self):
        cases = (
            (("CCO", "", "CCC", "CCN"), "empty"),
            (("CCO", "CCO", "CCC", "CCN"), "unique"),
            (("CCO", "not-a-smiles", "CCC", "CCN"), "parseable"),
        )
        for values, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    sealed_test = root / "test.txt"
                    zinc_csv = root / "zinc.csv"
                    write_test_source(sealed_test, values)
                    write_zinc_source(zinc_csv)
                    with self.assertRaisesRegex(
                        editing.ControlledEditingProtocolError, expected_message
                    ):
                        editing.build_controlled_editing_memberships(
                            sealed_test,
                            zinc_csv,
                            root / "derived",
                            source_binding=fixture_binding(),
                        )
                    self.assertFalse((root / "derived").exists())

    def test_zinc_schema_population_and_full_parseability_are_hard_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed_test = root / "test.txt"
            zinc_csv = root / "zinc.csv"
            write_test_source(sealed_test)
            with zinc_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("logP", "smiles", "qed", "SAS"))
                writer.writerow(("0.1", "CCC", "0.5", "2.0"))
            with self.assertRaisesRegex(
                editing.ControlledEditingProtocolError, "header must be exactly"
            ):
                editing.build_controlled_editing_memberships(
                    sealed_test,
                    zinc_csv,
                    root / "wrong-header",
                    source_binding=fixture_binding(zinc_count=1, validation_count=1),
                )

            write_zinc_source(zinc_csv, ("CCC", "not-a-smiles"))
            with self.assertRaisesRegex(
                editing.ControlledEditingProtocolError, "not RDKit-parseable"
            ):
                editing.build_controlled_editing_memberships(
                    sealed_test,
                    zinc_csv,
                    root / "invalid-zinc",
                    source_binding=fixture_binding(zinc_count=2, validation_count=1),
                )

            write_zinc_source(zinc_csv, ("CCC", "CCN"))
            with self.assertRaisesRegex(
                editing.ControlledEditingProtocolError, "full population"
            ):
                editing.build_controlled_editing_memberships(
                    sealed_test,
                    zinc_csv,
                    root / "wrong-population",
                    source_binding=fixture_binding(zinc_count=3, validation_count=1),
                )

    def test_production_binding_requires_rdkit_2024_03_5(self):
        with mock.patch.object(editing.rdBase, "rdkitVersion", "2025.09.1"):
            with self.assertRaisesRegex(
                editing.ControlledEditingProtocolError, "requires RDKit 2024.03.5"
            ):
                editing.validate_binding(editing.PRODUCTION_SOURCE_BINDING)
        with mock.patch.object(
            editing.rdBase, "rdkitVersion", editing.PRODUCTION_RDKIT_VERSION
        ):
            editing.validate_binding(editing.PRODUCTION_SOURCE_BINDING)

    def test_existing_output_directory_is_rejected_before_source_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "already-exists"
            output.mkdir()
            with self.assertRaisesRegex(
                editing.ControlledEditingProtocolError, "must not already exist"
            ):
                editing.build_controlled_editing_memberships(
                    root / "missing-test.txt",
                    root / "missing-zinc.csv",
                    output,
                    source_binding=fixture_binding(),
                )


if __name__ == "__main__":
    unittest.main()
