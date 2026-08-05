from __future__ import print_function

import csv
import importlib.util
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rdkit import Chem

from most_t5_next.r1.overlap import extract_identity_collection_v1 as extractor
from most_t5_next.r1.overlap import prove_membership_identity_overlap_v1 as proof


R1_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = R1_ROOT / "contracts" / "identity_collection_extraction_contract_v1.json"
IDENTITY_CONTRACT = R1_ROOT / "contracts" / "pcqm4mv2_identity_normalization_contract.json"
NORMALIZER = R1_ROOT / "gates" / "pcqm_identity_smoke.py"


def write_json(path, value):
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def source_observation(path):
    size, sha = extractor.sha256_file(path)
    return size, sha


def base_config(source_path, source_format, role="downstream_validation", text_available=True):
    size, sha = source_observation(source_path)
    if source_format == "json_array":
        options = {"json_top_level": "array", "csv_dialect": None, "parquet_batch_size": None, "lmdb": None}
        member_source = "field"
    elif source_format == "csv":
        options = {"json_top_level": None, "csv_dialect": "excel", "parquet_batch_size": None, "lmdb": None}
        member_source = "row_index"
    elif source_format == "parquet":
        options = {"json_top_level": None, "csv_dialect": None, "parquet_batch_size": 2, "lmdb": None}
        member_source = "field"
    else:
        options = {
            "json_top_level": None,
            "csv_dialect": None,
            "parquet_batch_size": None,
            "lmdb": {
                "subdir": False,
                "metadata_keys_permitted": ["__len__"],
                "metadata_keys_required": [],
                "trusted_pickle_source_sha256": sha,
            },
        }
        member_source = "lmdb_key"
    phase = "p2" if role.startswith("p2_") else "downstream"
    split = "train" if role in ("p2_permitted_train_membership", "p2_alignment_train", "p2_geometry_replay_train", "downstream_train") else ("validation" if role == "downstream_validation" else "test")
    task_family = "none" if role == "p2_permitted_train_membership" else "fixture_caption"
    text = {
        "status": "available",
        "normalization": "unicode_nfkc_whitespace_v1",
        "unit": {
            "unit_name": "caption_target",
            "semantic_role": "molecule_caption_decoder_target",
            "serialization": "canonical_component_object_utf8_v1",
            "components": [{"name": "target", "field": "description"}],
        },
    }
    if not text_available:
        text = {"status": "unavailable", "normalization": None, "unit": None}
    return {
        "schema_version": extractor.CONFIG_SCHEMA,
        "extraction_id": "fixture-extraction-v1",
        "contract_sha256": source_observation(CONTRACT)[1],
        "collection": {
            "collection_id": "fixture-collection",
            "dataset_id": "fixture-dataset",
            "release_id": "fixture-release",
            "phase": phase,
            "split": split,
            "role": role,
            "task_family": task_family,
            "source_identity_namespace": "fixture-source-row",
        },
        "source": {
            "path": str(source_path),
            "format": source_format,
            "expected_bytes": size,
            "expected_sha256": sha,
            "format_options": options,
        },
        "identity": {
            "normalization_contract_path": str(IDENTITY_CONTRACT),
            "normalization_contract_sha256": source_observation(IDENTITY_CONTRACT)[1],
            "normalizer_path": str(NORMALIZER),
            "normalizer_sha256": source_observation(NORMALIZER)[1],
            "required_rdkit_version": Chem.rdBase.rdkitVersion,
        },
        "mapping": {
            "member_id": {
                "source": member_source,
                "field": "cid" if member_source == "field" else None,
                "prefix": "fixture:",
                "crosscheck_field": "cid" if member_source == "lmdb_key" else None,
            },
            "smiles_field": "smiles",
            "record_filter": {"field": "split", "allowed_values": ["validation"]} if source_format == "json_array" else None,
            "text_identity": text,
        },
    }


class FakeCursor(object):
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeTransaction(object):
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return FakeCursor(self.rows)


class FakeEnvironment(object):
    def __init__(self, rows):
        self.rows = rows

    def begin(self):
        return FakeTransaction(self.rows)

    def close(self):
        return None


class IdentityCollectionExtractorTests(unittest.TestCase):
    def temporary_root(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_json_explicit_mapping_emits_hash_only_gate_compatible_collection(self):
        root = self.temporary_root()
        source = root / "rows.json"
        write_json(
            source,
            [
                {"cid": "1", "smiles": "CCO", "split": "validation", "description": "  Ethanol\t molecule  "},
                {"cid": "2", "smiles": "C", "split": "train", "description": "Methane"},
                {"cid": "3", "smiles": "CC", "split": "validation", "description": "Ethane"},
            ],
        )
        config = base_config(source, "json_array")
        config_path = root / "config.json"
        write_json(config_path, config)
        report = extractor.run_extraction(CONTRACT, config_path, root / "output")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["counts"]["source_data_records_seen"], 3)
        self.assertEqual(report["counts"]["records_filtered_by_explicit_policy"], 1)
        self.assertEqual(report["counts"]["selected_molecule_members"], 2)
        molecule_text = (root / "output" / "molecule_identity_rows.jsonl").read_text(encoding="utf-8")
        pair_text = (root / "output" / "text_pair_identity_rows.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("CCO", molecule_text)
        self.assertNotIn("Ethanol", pair_text)
        manifest_path = root / "output" / "collection_manifest.json"
        connection = proof.create_database(":memory:")
        try:
            collection, _ = proof.load_collection(connection, manifest_path, source_observation(manifest_path)[1])
            self.assertEqual(collection["collection_id"], "fixture-collection")
        finally:
            connection.close()

    def test_csv_row_index_mapping_is_explicit_and_does_not_guess_id(self):
        root = self.temporary_root()
        source = root / "rows.csv"
        with open(str(source), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["anything", "smiles"])
            writer.writeheader()
            writer.writerow({"anything": "not-an-id", "smiles": "O"})
            writer.writerow({"anything": "still-not-an-id", "smiles": "N"})
        config = base_config(source, "csv", role="downstream_test", text_available=False)
        config["mapping"]["member_id"]["prefix"] = "csv-row:"
        config_path = root / "config.json"
        write_json(config_path, config)
        extractor.run_extraction(CONTRACT, config_path, root / "output")
        rows = [json.loads(line) for line in (root / "output" / "molecule_identity_rows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["member_id"] for row in rows], ["csv-row:0", "csv-row:1"])

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow is unavailable")
    def test_parquet_explicit_fields_and_task_filter_run_end_to_end(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        root = self.temporary_root()
        source = root / "rows.parquet"
        table = arrow.table(
            {
                "cid": ["10", "11", "12"],
                "smiles": ["CO", "CN", "CCC"],
                "task": ["caption", "other", "caption"],
                "description": ["methanol", "methylamine", "propane"],
            }
        )
        parquet.write_table(table, str(source))
        config = base_config(source, "parquet", role="downstream_train", text_available=True)
        config["mapping"]["record_filter"] = {"field": "task", "allowed_values": ["caption"]}
        config_path = root / "config.json"
        write_json(config_path, config)
        report = extractor.run_extraction(CONTRACT, config_path, root / "output")
        self.assertEqual(report["counts"]["source_data_records_seen"], 3)
        self.assertEqual(report["counts"]["selected_molecule_members"], 2)
        self.assertEqual(report["counts"]["records_filtered_by_explicit_policy"], 1)

    def test_lmdb_fixture_excludes_declared_len_metadata_before_pickle(self):
        root = self.temporary_root()
        source = root / "fixture.lmdb"
        source.write_bytes(b"fixture-source-lock")
        config = base_config(source, "legacy_lmdb_pickle", role="p2_permitted_train_membership", text_available=False)
        config["source"]["format_options"]["lmdb"]["metadata_keys_required"] = ["__len__"]
        rows = [
            (b"1", pickle.dumps({"cid": "1", "smiles": "C"}, protocol=4)),
            (b"2", pickle.dumps({"cid": "2", "smiles": "CC"}, protocol=4)),
            (b"__len__", b"this-is-deliberately-not-a-pickle"),
        ]
        fake_lmdb = types.SimpleNamespace(open=lambda *args, **kwargs: FakeEnvironment(rows))
        observed = set()
        with mock.patch.dict(sys.modules, {"lmdb": fake_lmdb}):
            extracted = list(extractor.iter_source_records(source, config["source"], observed))
        self.assertEqual([item[1] for item in extracted], ["1", "2"])
        self.assertEqual(observed, {"__len__"})

    @unittest.skipUnless(importlib.util.find_spec("lmdb") is not None, "python-lmdb is unavailable")
    def test_real_single_file_lmdb_fixture_runs_end_to_end(self):
        import lmdb

        root = self.temporary_root()
        source = root / "fixture.lmdb"
        environment = lmdb.open(str(source), subdir=False, map_size=8 * 1024 * 1024)
        with environment.begin(write=True) as transaction:
            transaction.put(b"1", pickle.dumps({"cid": "1", "smiles": "C"}, protocol=4))
            transaction.put(b"2", pickle.dumps({"cid": "2", "smiles": "CC"}, protocol=4))
            transaction.put(b"__len__", pickle.dumps(2, protocol=4))
        environment.sync(True)
        environment.close()
        config = base_config(source, "legacy_lmdb_pickle", role="p2_permitted_train_membership", text_available=False)
        config["source"]["format_options"]["lmdb"]["metadata_keys_required"] = ["__len__"]
        config_path = root / "config.json"
        write_json(config_path, config)
        report = extractor.run_extraction(CONTRACT, config_path, root / "output")
        self.assertEqual(report["counts"]["selected_molecule_members"], 2)
        self.assertEqual(report["counts"]["observed_excluded_lmdb_metadata_keys"], 1)
        manifest = json.loads((root / "output" / "collection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["provenance"]["excluded_source_metadata_keys"], ["__len__"])

    def test_undeclared_lmdb_metadata_key_fails_closed(self):
        root = self.temporary_root()
        source = root / "fixture.lmdb"
        source.write_bytes(b"fixture-source-lock")
        config = base_config(source, "legacy_lmdb_pickle", role="p2_permitted_train_membership", text_available=False)
        config["source"]["format_options"]["lmdb"]["metadata_keys_permitted"] = []
        rows = [(b"__len__", pickle.dumps(0, protocol=4))]
        fake_lmdb = types.SimpleNamespace(open=lambda *args, **kwargs: FakeEnvironment(rows))
        with mock.patch.dict(sys.modules, {"lmdb": fake_lmdb}):
            with self.assertRaisesRegex(ValueError, "undeclared LMDB metadata-like key"):
                list(extractor.iter_source_records(source, config["source"], set()))

    def test_pickle_acknowledgement_must_equal_locked_source_sha(self):
        root = self.temporary_root()
        source = root / "fixture.lmdb"
        source.write_bytes(b"fixture-source-lock")
        config = base_config(source, "legacy_lmdb_pickle", role="p2_permitted_train_membership", text_available=False)
        config["source"]["format_options"]["lmdb"]["trusted_pickle_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "acknowledgement must equal"):
            extractor.validate_config(config, source_observation(CONTRACT)[1])

    def test_missing_explicit_smiles_field_never_falls_back(self):
        root = self.temporary_root()
        source = root / "rows.json"
        write_json(source, [{"cid": "1", "canonical_smiles": "CC", "split": "validation", "description": "Ethane"}])
        config = base_config(source, "json_array")
        config_path = root / "config.json"
        write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "SMILES field is absent"):
            extractor.run_extraction(CONTRACT, config_path, root / "output")


if __name__ == "__main__":
    unittest.main()
