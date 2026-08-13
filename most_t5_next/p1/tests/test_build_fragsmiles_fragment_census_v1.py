from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1.build_fragsmiles_fragment_census_v1 import (
    FragSmilesFragmentCensusError,
    _iter_legacy_lmdb,
    canonical_fragment_identity,
    run_census,
)
from most_t5_next.p1.build_phase2_anchored_pure_motif_census_v1 import (
    ACKNOWLEDGEMENT,
    EXPECTED_FIELDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMICALGOF_ROOT = REPO_ROOT / "reference_repos" / "chemicalgof-master"


class FragSmilesFragmentCensusTest(unittest.TestCase):
    def test_fragment_identity_is_traversal_invariant(self):
        self.assertEqual(
            canonical_fragment_identity("C1=CC=CC=C1"),
            canonical_fragment_identity("c1ccccc1"),
        )

    def test_small_jsonl_census_is_ordered_and_counted(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"smiles": smiles}) + "\n"
                    for smiles in ("CCO", "c1ccccc1", "CCO")
                ),
                encoding="utf-8",
            )
            output = root / "output"
            manifest = run_census(
                input_path=source,
                input_format="jsonl",
                smiles_field="smiles",
                chemicalgof_root=CHEMICALGOF_ROOT,
                output_dir=output,
                workers=1,
                max_pending=1,
                expected_records=3,
                progress_every=0,
                record_timeout_seconds=None,
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertTrue(
                manifest["contracts"][
                    "compact_fragments_all_in_finite_chemical_lexer_domain"
                ]
            )
            self.assertEqual(manifest["counts"]["processed_records"], 3)
            self.assertEqual(manifest["counts"]["modes"], {"compact": 3})
            rows = [
                json.loads(line)
                for line in (output / "fragment_census.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertTrue(rows)
            self.assertEqual([row["rank"] for row in rows], list(range(len(rows))))
            with gzip.open(
                output / "molecule_fragments.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                cache = [json.loads(line) for line in handle]
            self.assertEqual(
                [row["selection_index"] for row in cache], [0, 1, 2]
            )
            self.assertEqual([row["source_index"] for row in cache], [0, 1, 2])

    def test_explicit_hydrogen_identity_uses_lossless_original_surface(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"smiles": smiles}) + "\n"
                    for smiles in (
                        "[2H]O[2H]",
                        "[H-].[Na+]",
                        "[H+].[B-](F)(F)(F)F",
                        "C1=C/C/2=C\\3/C(=C/4\\C(=C\\5/C(=C2/C=C1)/C=CC=C5)\\C=CC=C4)/C=CC=C3",
                    )
                ),
                encoding="utf-8",
            )
            output = root / "output"
            manifest = run_census(
                input_path=source,
                input_format="jsonl",
                smiles_field="smiles",
                chemicalgof_root=CHEMICALGOF_ROOT,
                output_dir=output,
                workers=1,
                max_pending=1,
                expected_records=4,
                progress_every=0,
                record_timeout_seconds=None,
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(
                manifest["counts"]["modes"],
                {"whole_molecule_fallback": 4},
            )
            self.assertEqual(
                manifest["counts"]["projection_modes"],
                {
                    "heavy_atom_projection": 1,
                    "original_molecule_lossless_fallback": 3,
                },
            )
            self.assertEqual(
                manifest["counts"]["fallback_reasons"],
                {"RuntimeError": 1, "TrainingProjectionDomainError": 3},
            )
            with gzip.open(
                output / "molecule_fragments.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                rows = [json.loads(line) for line in handle]
            self.assertTrue(
                all(row["mode"] == "whole_molecule_fallback" for row in rows)
            )
            self.assertTrue(all(row["fragment_identities"] == [] for row in rows))

    def test_membership_selects_exact_pcqm_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"smiles": smiles}) + "\n"
                    for smiles in ("CC", "CO", "CN")
                ),
                encoding="utf-8",
            )
            membership = root / "membership.jsonl"
            membership.write_text(
                json.dumps(
                    {
                        "member_id": "ogb_pcqm4mv2_train_row_index:1",
                        "schema_version": "most-t5-r1/permitted-pretrain-member/v1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            manifest = run_census(
                input_path=source,
                input_format="jsonl",
                smiles_field="smiles",
                chemicalgof_root=CHEMICALGOF_ROOT,
                output_dir=output,
                membership_path=membership,
                workers=1,
                max_pending=1,
                expected_records=1,
                progress_every=0,
                record_timeout_seconds=None,
            )
            self.assertEqual(manifest["counts"]["processed_records"], 1)
            with gzip.open(
                output / "molecule_fragments.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                row = json.loads(next(handle))
            self.assertEqual(row["selection_index"], 0)
            self.assertEqual(row["source_index"], 1)

    def test_selected_record_range_preserves_global_indices(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"smiles": smiles}) + "\n"
                    for smiles in ("CC", "CO", "CN", "CF", "CS")
                ),
                encoding="utf-8",
            )
            output = root / "output"
            manifest = run_census(
                input_path=source,
                input_format="jsonl",
                smiles_field="smiles",
                chemicalgof_root=CHEMICALGOF_ROOT,
                output_dir=output,
                workers=1,
                max_pending=1,
                expected_records=2,
                start_record=2,
                max_records=2,
                progress_every=0,
                record_timeout_seconds=None,
            )
            self.assertEqual(
                manifest["selected_record_range"],
                {"start_inclusive": 2, "stop_exclusive": 4},
            )
            with gzip.open(
                output / "molecule_fragments.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["selection_index"] for row in rows], [2, 3])
            self.assertEqual([row["source_index"] for row in rows], [2, 3])

    def test_duplicate_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.jsonl"
            source.write_text('{"smiles":"CC"}\n', encoding="utf-8")
            row = '{"member_id":"ogb_pcqm4mv2_train_row_index:0"}\n'
            membership = root / "membership.jsonl"
            membership.write_text(row + row, encoding="utf-8")
            with self.assertRaises(FragSmilesFragmentCensusError):
                run_census(
                    input_path=source,
                    input_format="jsonl",
                    smiles_field="smiles",
                    chemicalgof_root=CHEMICALGOF_ROOT,
                    output_dir=root / "output",
                    membership_path=membership,
                    workers=1,
                    max_pending=1,
                    progress_every=0,
                    record_timeout_seconds=None,
                )

    def test_hash_locked_legacy_lmdb_source(self):
        try:
            import lmdb
        except ImportError:
            self.skipTest("python-lmdb is unavailable")
        import pickle

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "phase2.lmdb"
            environment = lmdb.open(str(source), subdir=False, map_size=1 << 20)
            payload = {field: None for field in EXPECTED_FIELDS}
            payload.update({"cid": 1, "smiles": "CCO"})
            with environment.begin(write=True) as transaction:
                transaction.put(b"1", pickle.dumps(payload, protocol=4))
            environment.close()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            output = root / "output"
            manifest = run_census(
                input_path=source,
                input_format="legacy-lmdb",
                smiles_field="smiles",
                chemicalgof_root=CHEMICALGOF_ROOT,
                output_dir=output,
                workers=1,
                max_pending=1,
                expected_records=1,
                progress_every=0,
                record_timeout_seconds=None,
                expected_source_sha256=digest,
                trusted_pickle_acknowledgement=ACKNOWLEDGEMENT,
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(manifest["source"]["sha256"], digest)
            self.assertTrue(
                manifest["contracts"][
                    "legacy_lmdb_payload_order_is_numeric_cid_ascending"
                ]
            )

    def test_legacy_lmdb_uses_numeric_key_order_and_declared_length(self):
        try:
            import lmdb
        except ImportError:
            self.skipTest("python-lmdb is unavailable")
        import pickle

        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "phase2.lmdb"
            environment = lmdb.open(str(source), subdir=False, map_size=1 << 20)
            with environment.begin(write=True) as transaction:
                for cid, smiles in ((10, "CN"), (2, "CO"), (1, "CC")):
                    payload = {field: None for field in EXPECTED_FIELDS}
                    payload.update({"cid": cid, "smiles": smiles})
                    transaction.put(
                        str(cid).encode("ascii"), pickle.dumps(payload, protocol=4)
                    )
                transaction.put(b"__len__", b"3")
            environment.close()
            rows = list(_iter_legacy_lmdb(source))
            self.assertEqual(
                rows,
                [(0, "CC", None), (1, "CO", None), (2, "CN", None)],
            )


if __name__ == "__main__":
    unittest.main()
