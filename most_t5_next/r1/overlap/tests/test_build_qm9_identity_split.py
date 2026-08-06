from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from most_t5_next.r1.overlap import build_qm9_identity_split as qm9


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = b"".join(qm9.canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def row(
    smiles: str,
    instruction: str,
    output: object,
    molecule_fp: object,
    *,
    selfies: str | None = None,
) -> dict[str, object]:
    return {
        "smiles": smiles,
        "selfies": selfies if selfies is not None else "[" + smiles + "]",
        "instruction": instruction,
        "output": output,
        "molecule_fp": molecule_fp,
    }


class Qm9CleanIdentitySplitTests(unittest.TestCase):
    def make_sources(self, root: Path) -> tuple[Path, Path]:
        train = root / "train.jsonl"
        validation = root / "validation.jsonl"
        write_jsonl(
            train,
            [
                row("CC", " HOMO ", "1.00", [[1, 2], [3, 4]], selfies="[C][C]"),
                # Same semantic value after Decimal normalization and exact model-visible
                # payload: this later row is the sole removable duplicate.
                row("CC", "HOMO", "1.0", [[1, 2], [3, 4]], selfies="[C][C]"),
                # Equal semantic signature but different complete E3FP: retain as state.
                row("CC", "HOMO", 1, [[1, 2], [9, 9]], selfies="[C][C]"),
                # Same molecule through a non-canonical spelling; a different prompt.
                row("C(C)O", "LUMO", "-0.20", [[5]], selfies="[C][C][O]"),
                row("N", "gap", "-0", [[6]], selfies="[N]"),
                row("O", "gap", "2E-1", [[7]], selfies="[O]"),
            ],
        )
        write_jsonl(
            validation,
            [
                row("C", "HOMO", "0.3", [[8]], selfies="[C]"),
                row("F", "HOMO", "0.4", [[9]], selfies="[F]"),
                row("Cl", "HOMO", "0.5", [[10]], selfies="[Cl]"),
            ],
        )
        return train, validation

    def build(self, root: Path, output_name: str = "derived") -> tuple[Path, dict[str, object]]:
        train, validation = self.make_sources(root)
        output = root / output_name
        summary = qm9.build_qm9_identity_split(
            {"train": [train], "validation": [validation]},
            output,
            train_group_count=3,
            validation_group_count=1,
            enforce_production_protocol=False,
        )
        return output, summary

    def test_duplicate_boundary_and_distinct_e3fp_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, summary = self.build(Path(temporary))
            report = read_json(output / qm9.DUPLICATE_REPORT_FILENAME)
            rows = read_jsonl(output / qm9.SPLIT_MANIFEST_FILENAME)

            self.assertEqual(report["removed_model_visible_duplicate_count"], 1)
            removed = report["removed_model_visible_duplicates"][0]
            self.assertEqual(removed["kept"]["source_row_index"], 0)
            self.assertEqual(removed["removed"]["source_row_index"], 1)
            self.assertEqual(
                report["retained_equal_semantics_distinct_e3fp_signature_count"], 1
            )
            variants = report["retained_equal_semantics_distinct_e3fp_states"][0]
            self.assertEqual(variants["retained_record_count"], 2)
            self.assertEqual(variants["distinct_molecule_fp_serialization_count"], 2)
            cc_homo = [
                item
                for item in rows
                if item["strict_canonical_isomeric_smiles"] == "CC"
                and item["instruction_stripped"] == "HOMO"
            ]
            self.assertEqual(len(cc_homo), 2)
            self.assertEqual({item["normalized_numeric_target"] for item in cc_homo}, {"1"})
            self.assertEqual(summary["counts"]["input_rows"], 9)
            self.assertEqual(summary["counts"]["retained_rows"], 8)

    def test_released_numeric_sentence_grammar_is_exact(self):
        self.assertEqual(qm9.normalize_numeric_target("0.1913."), "0.1913")
        self.assertEqual(qm9.normalize_numeric_target("-0.243."), "-0.243")
        self.assertEqual(qm9.normalize_numeric_target("1."), "1")
        self.assertEqual(qm9.normalize_numeric_target(" -0.0000. "), "0")
        for invalid in (
            "0.1913 eV",
            "value=0.1913.",
            "0.1 0.2",
            "0.1913..",
            "NaN.",
            "inf.",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(qm9.Qm9SplitProtocolError):
                    qm9.normalize_numeric_target(invalid)

    def test_group_never_crosses_split_and_counts_cover_every_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, summary = self.build(Path(temporary))
            rows = read_jsonl(output / qm9.SPLIT_MANIFEST_FILENAME)
            assignments: dict[str, set[str]] = {}
            for item in rows:
                assignments.setdefault(item["group_id"], set()).add(item["assigned_split"])
            self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))
            counts = summary["counts"]
            self.assertEqual(
                counts["input_rows"],
                counts["retained_rows"] + counts["removed_model_visible_duplicates"],
            )
            self.assertEqual(sum(counts["output_rows"].values()), counts["retained_rows"])
            self.assertEqual(sum(counts["output_groups"].values()), counts["molecule_groups"])
            self.assertEqual(counts["output_groups"], {"train": 3, "validation": 1, "test": 3})

    def test_assignment_and_artifacts_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_output, _ = self.build(root, "first")
            # Reuse the exact same source files; rebuilding them writes identical bytes.
            second_output, _ = self.build(root, "second")
            for filename in (
                qm9.SOURCE_MANIFEST_FILENAME,
                qm9.SPLIT_MANIFEST_FILENAME,
                qm9.DUPLICATE_REPORT_FILENAME,
                qm9.SPLIT_SUMMARY_FILENAME,
            ):
                self.assertEqual(
                    (first_output / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                )
            summary = read_json(first_output / qm9.SPLIT_SUMMARY_FILENAME)
            self.assertEqual(summary["rng_contract"]["seed"], 42)
            self.assertEqual(summary["rng_contract"]["bit_generator"], "PCG64")

    def test_released_test_input_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, validation = self.make_sources(root)
            test = root / "test.jsonl"
            test.write_text("this file must not be read\n", encoding="utf-8")
            with self.assertRaisesRegex(qm9.Qm9SplitProtocolError, "forbids released test"):
                qm9.build_qm9_identity_split(
                    {"train": [train], "validation": [validation], "test": [test]},
                    root / "forbidden",
                    train_group_count=3,
                    validation_group_count=1,
                    enforce_production_protocol=False,
                )
            self.assertFalse((root / "forbidden").exists())

    def test_production_protocol_uses_layout_and_scale_not_file_observations(self):
        sources = [
            qm9.SourceFile(
                split="train",
                file_ordinal=0,
                path=Path("arbitrarily-renamed-train.parquet"),
                bytes=17,
                sha256="a" * 64,
            ),
            qm9.SourceFile(
                split="validation",
                file_ordinal=0,
                path=Path("arbitrarily-renamed-validation.parquet"),
                bytes=29,
                sha256="b" * 64,
            ),
        ]
        with mock.patch.object(
            qm9.rdBase, "rdkitVersion", qm9.PRODUCTION_RDKIT_VERSION
        ):
            qm9.require_production_protocol(
                sources,
                train_group_count=qm9.DEFAULT_TRAIN_GROUP_COUNT,
                validation_group_count=qm9.DEFAULT_VALIDATION_GROUP_COUNT,
            )

        changed_observations = [
            qm9.SourceFile(
                split=source.split,
                file_ordinal=source.file_ordinal,
                path=source.path,
                bytes=source.bytes + 10_000,
                sha256="f" * 64,
            )
            for source in sources
        ]
        with mock.patch.object(
            qm9.rdBase, "rdkitVersion", qm9.PRODUCTION_RDKIT_VERSION
        ):
            qm9.require_production_protocol(
                changed_observations,
                train_group_count=qm9.DEFAULT_TRAIN_GROUP_COUNT,
                validation_group_count=qm9.DEFAULT_VALIDATION_GROUP_COUNT,
            )

    def test_production_protocol_requires_one_parquet_per_split_and_fixed_scale(self):
        train = qm9.SourceFile("train", 0, Path("train.parquet"), 1, "a" * 64)
        validation = qm9.SourceFile(
            "validation", 0, Path("validation.parquet"), 1, "b" * 64
        )
        with mock.patch.object(
            qm9.rdBase, "rdkitVersion", qm9.PRODUCTION_RDKIT_VERSION
        ):
            with self.assertRaisesRegex(qm9.Qm9SplitProtocolError, "exactly one Parquet"):
                qm9.require_production_protocol(
                    [train, validation, qm9.SourceFile("train", 1, Path("part.parquet"), 1, "c" * 64)],
                    train_group_count=qm9.DEFAULT_TRAIN_GROUP_COUNT,
                    validation_group_count=qm9.DEFAULT_VALIDATION_GROUP_COUNT,
                )
            with self.assertRaisesRegex(qm9.Qm9SplitProtocolError, "must be one Parquet"):
                qm9.require_production_protocol(
                    [train, qm9.SourceFile("validation", 0, Path("validation.jsonl"), 1, "b" * 64)],
                    train_group_count=qm9.DEFAULT_TRAIN_GROUP_COUNT,
                    validation_group_count=qm9.DEFAULT_VALIDATION_GROUP_COUNT,
                )
            with self.assertRaisesRegex(qm9.Qm9SplitProtocolError, "train_group_count"):
                qm9.require_production_protocol(
                    [train, validation],
                    train_group_count=3,
                    validation_group_count=qm9.DEFAULT_VALIDATION_GROUP_COUNT,
                )

        with mock.patch.object(qm9.rdBase, "rdkitVersion", "2025.09.1"):
            with self.assertRaisesRegex(qm9.Qm9SplitProtocolError, "requires RDKit"):
                qm9.require_production_protocol(
                    [train, validation],
                    train_group_count=qm9.DEFAULT_TRAIN_GROUP_COUNT,
                    validation_group_count=qm9.DEFAULT_VALIDATION_GROUP_COUNT,
                )

    def test_production_cli_does_not_expose_fixture_split_sizes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                qm9.parse_args(
                    [
                        "--train",
                        "train.parquet",
                        "--validation",
                        "validation.parquet",
                        "--output-dir",
                        "derived",
                        "--fixture-train-group-count",
                        "3",
                    ]
                )

    def test_production_semantic_census_is_an_exact_admission_gate(self):
        observed = dict(qm9.PRODUCTION_EXPECTED_COUNTS)
        observed["output_rows"] = {
            "train": 298_518,
            "validation": 27_147,
            "test": 23_995,
        }
        qm9.require_production_semantic_census(observed)
        wrong = dict(observed)
        wrong["retained_rows"] = observed["retained_rows"] - 1
        with self.assertRaisesRegex(
            qm9.Qm9SplitProtocolError, "semantic census differs"
        ):
            qm9.require_production_semantic_census(wrong)

    def test_source_manifest_records_revision_rows_observations_and_rdkit_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self.build(Path(temporary))
            manifest = read_json(output / qm9.SOURCE_MANIFEST_FILENAME)
            self.assertEqual(manifest["frozen_source_revision"], qm9.SOURCE_REVISION)
            self.assertFalse(manifest["production_protocol_enforced"])
            self.assertIsNone(manifest["expected_production_counts"])
            self.assertIn(
                "do_not_decide_scientific_admission",
                manifest["source_file_observation_policy"],
            )
            self.assertEqual(
                [item["source_split"] for item in manifest["source_files"]],
                ["train", "validation"],
            )
            self.assertEqual(
                [item["row_count"] for item in manifest["source_files"]], [6, 3]
            )
            self.assertTrue(
                all(len(item["sha256"]) == 64 for item in manifest["source_files"])
            )
            self.assertTrue(
                all(
                    item["observation_role"]
                    == "provenance_metadata_not_admission_criterion"
                    for item in manifest["source_files"]
                )
            )
            self.assertEqual(manifest["canonicalization"]["library"], "RDKit")
            self.assertEqual(
                manifest["canonicalization"]["split_molecule_identity"]["parameters"],
                {"canonical": True, "isomericSmiles": False, "kekuleSmiles": False},
            )
            self.assertEqual(
                manifest["canonicalization"]["protected_union_identity"]["parameters"],
                {"canonical": True, "isomericSmiles": False, "kekuleSmiles": False},
            )

    def test_stereoisomers_share_connectivity_group_but_keep_distinct_stereo_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            write_jsonl(
                train,
                [
                    row("F[C@H](Cl)Br", "HOMO", "1", [[1]]),
                    row("F[C@H](Cl)Br", "gap", "2", [[2]]),
                    row("F[C@@H](Cl)Br", "HOMO", "1", [[3]]),
                    row("C", "HOMO", "1", [[4]]),
                ],
            )
            write_jsonl(validation, [row("N", "HOMO", "1", [[5]])])
            output = root / "derived"
            qm9.build_qm9_identity_split(
                {"train": [train], "validation": [validation]},
                output,
                train_group_count=1,
                validation_group_count=1,
                enforce_production_protocol=False,
            )
            rows = read_jsonl(output / qm9.SPLIT_MANIFEST_FILENAME)
            stereo_rows = [item for item in rows if "Br" in item["strict_canonical_isomeric_smiles"]]
            strict_identities = {
                item["strict_canonical_isomeric_smiles_sha256"] for item in stereo_rows
            }
            protection_identities = {
                item["canonical_connectivity_smiles_sha256"] for item in stereo_rows
            }
            self.assertEqual(len(strict_identities), 2)
            self.assertEqual(len(protection_identities), 1)
            grouped: dict[str, set[str]] = {}
            for item in stereo_rows:
                grouped.setdefault(item["group_id"], set()).add(item["assigned_split"])
            self.assertEqual(len(grouped), 1)
            self.assertTrue(all(len(splits) == 1 for splits in grouped.values()))

    def test_declared_explicit_h_projection_matches_implicit_form(self):
        explicit = qm9.canonicalize_identity_forms("C[C@]([H])(O)F")
        implicit = qm9.canonicalize_identity_forms("C[C@H](O)F")
        self.assertEqual(explicit, implicit)


if __name__ == "__main__":
    unittest.main()
