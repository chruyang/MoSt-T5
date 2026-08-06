from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

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
            enforce_frozen_source=False,
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
                    enforce_frozen_source=False,
                )
            self.assertFalse((root / "forbidden").exists())

    def test_production_source_hash_gate_precedes_row_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train-00000-of-00001.parquet"
            validation = root / "validation-00000-of-00001.parquet"
            train.write_bytes(b"not the frozen train parquet")
            validation.write_bytes(b"not the frozen validation parquet")
            with self.assertRaisesRegex(qm9.Qm9SplitProtocolError, "bytes or SHA-256"):
                qm9.build_qm9_identity_split(
                    {"train": [train], "validation": [validation]},
                    root / "derived",
                )
            self.assertFalse((root / "derived").exists())

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
        observed = dict(qm9.FROZEN_EXPECTED_COUNTS)
        qm9.require_frozen_output_counts(observed)
        wrong = dict(observed)
        wrong["retained_rows"] = observed["retained_rows"] - 1
        with self.assertRaisesRegex(
            qm9.Qm9SplitProtocolError, "semantic census differs"
        ):
            qm9.require_frozen_output_counts(wrong)

    def test_source_manifest_binds_revision_rows_hashes_and_rdkit_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self.build(Path(temporary))
            manifest = read_json(output / qm9.SOURCE_MANIFEST_FILENAME)
            self.assertEqual(manifest["frozen_source_revision"], qm9.SOURCE_REVISION)
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
            self.assertEqual(manifest["canonicalization"]["library"], "RDKit")
            self.assertEqual(
                manifest["canonicalization"]["split_molecule_identity"]["parameters"],
                {"canonical": True, "isomericSmiles": True, "kekuleSmiles": False},
            )
            self.assertEqual(
                manifest["canonicalization"]["protected_union_identity"]["parameters"],
                {"canonical": True, "isomericSmiles": False, "kekuleSmiles": False},
            )

    def test_stereoisomers_are_distinct_split_identities_but_share_protection_identity(self):
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
                enforce_frozen_source=False,
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
            self.assertTrue(all(len(splits) == 1 for splits in grouped.values()))


if __name__ == "__main__":
    unittest.main()
