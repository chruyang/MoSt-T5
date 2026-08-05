from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

from most_t5_next.r1.tokenizer import audit_p1_p2_motif_projection_compatibility_v1 as compatibility
from most_t5_next.r1.tokenizer import extract_p2_phase2_ready_motif_census_v1 as extractor
from most_t5_next.r1.tokenizer import verify_p2_phase2_ready_motif_census_rerun_v1 as rerun


REPO_ROOT = Path(__file__).resolve().parents[4]
P2_CONTRACT = REPO_ROOT / "most_t5_next" / "r1" / "contracts" / "p2_phase2_ready_motif_census_contract_v1.json"
COMPATIBILITY_CONTRACT = REPO_ROOT / "most_t5_next" / "r1" / "contracts" / "p1_p2_motif_projection_compatibility_contract_v1.json"
P1_CONTRACT = REPO_ROOT / "most_t5_next" / "r1" / "contracts" / "p1_pcqm_geometry_production_release_contract.json"


def write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def closed_payload(cid: int, motif_seq: str, mapping: list[list[int]]) -> dict[str, object]:
    return {
        "atom_to_motif_map": mapping,
        "atoms": [6, 8],
        "cid": cid,
        "coordinates": [[0.0, 0.0, 0.0]],
        "description": "fixture",
        "e3fp": [[1, 2]],
        "enriched_description": "fixture enriched",
        "motif_seq": motif_seq,
        "raw_smiles": "CO",
        "smiles": "CO",
    }


class FakeCursor:
    def __init__(self, entries: dict[bytes, bytes]) -> None:
        self.entries = entries

    def __iter__(self):
        for key in sorted(self.entries):
            yield key, self.entries[key]


class FakeTransaction:
    def __init__(self, entries: dict[bytes, bytes]) -> None:
        self.entries = entries

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def stat(self) -> dict[str, int]:
        return {"entries": len(self.entries)}

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.entries)


class FakeEnvironment:
    def __init__(self, entries: dict[bytes, bytes]) -> None:
        self.entries = entries
        self.closed = False

    def begin(self, write: bool = False) -> FakeTransaction:
        if write:
            raise AssertionError("extractor attempted a write transaction")
        return FakeTransaction(self.entries)

    def close(self) -> None:
        self.closed = True


def fake_lmdb_module(entries: dict[bytes, bytes]) -> types.ModuleType:
    module = types.ModuleType("lmdb")

    def open_lmdb(path: str, **kwargs: object) -> FakeEnvironment:
        if kwargs != {
            "subdir": False,
            "readonly": True,
            "lock": False,
            "readahead": False,
            "meminit": False,
            "max_readers": 1,
        }:
            raise AssertionError("extractor LMDB open flags are not the read-only locked fixture flags")
        if not isinstance(path, str):
            raise AssertionError("LMDB path must be text")
        return FakeEnvironment(entries)

    module.open = open_lmdb  # type: ignore[attr-defined]
    return module


class P2CensusCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # AGENTS.md forbids recursive/bulk deletion.  These tiny hermetic fixture
        # directories are intentionally left in the operating-system temp area.
        cls.root = Path(tempfile.mkdtemp(prefix="most_t5_p2_census_fixture_"))
        cls.source = cls.root / "phase2_pubchem_ready.lmdb"
        cls.source.write_bytes(b"hermetic fake LMDB content-address boundary\n")
        cls.entries = {
            b"1": pickle.dumps(
                closed_payload(
                    1,
                    "  <bom> [C<0*>] [<0*>O] [.] [N[C@@H](C)O] <eom> ",
                    [[0], [1], [2, 3]],
                ),
                protocol=4,
            ),
            b"2": pickle.dumps(
                closed_payload(2, "<bom>[C<1*>][N<1*>][O<1*>]<eom>", [[0], [1], [2]]),
                protocol=4,
            ),
            b"3": pickle.dumps(
                closed_payload(3, "<bom>[C][O]<eom>", [[0]]),
                protocol=4,
            ),
        }
        cls.source_lock = cls.root / "p2_source_lock.json"
        write_json_new(
            cls.source_lock,
            {
                "schema_version": extractor.SOURCE_LOCK_SCHEMA,
                "source_role": "phase2_pubchem_ready_lmdb",
                "source_format": "lmdb_single_file_pickle_values",
                "source_sha256": extractor.sha256_file(cls.source),
                "source_bytes": cls.source.stat().st_size,
                "expected_payload_entry_count": 3,
                "expected_metadata_keys": [],
                "expected_payload_fields": sorted(extractor.EXPECTED_PAYLOAD_FIELDS),
                "identity_namespace": "pubchem_cid",
                "membership_status": "candidate_geometry_ready",
                "source_copy_manifest_sha256": "1" * 64,
                "pickle_trust_basis_sha256": "2" * 64,
                "motif_sequence_producer_status": "unknown_legacy_producer",
                "motif_sequence_producer_sha256": None,
                "legacy_linearization_spec_sha256": "3" * 64,
            },
        )
        cls.release_a = cls.root / "p2_release_seed_11"
        cls.release_b = cls.root / "p2_release_seed_29"
        fake_module = fake_lmdb_module(cls.entries)
        with mock.patch.dict(sys.modules, {"lmdb": fake_module}):
            with mock.patch.dict("os.environ", {"PYTHONHASHSEED": "11"}):
                cls.receipt_a = extractor.extract_release(
                    cls.source, cls.source_lock, P2_CONTRACT, cls.release_a,
                    extractor.ACKNOWLEDGEMENT,
                )
            with mock.patch.dict("os.environ", {"PYTHONHASHSEED": "29"}):
                cls.receipt_b = extractor.extract_release(
                    cls.source, cls.source_lock, P2_CONTRACT, cls.release_b,
                    extractor.ACKNOWLEDGEMENT,
                )

    def test_bracket_depth_and_component_ranges(self) -> None:
        parsed = extractor.parse_motif_sequence(
            " <bom>[C<0*>][N[C@@H](C)O][.][Cl-]<eom> "
        )
        self.assertEqual(parsed["fragments"], ["C<0*>", "N[C@@H](C)O", "Cl-"])
        self.assertEqual(parsed["component_fragment_ranges"], [[0, 2], [2, 3]])
        self.assertEqual(extractor.project_fragment("N[C@@H](C)O<12*>")[0], "[N[C@@H](C)O]")
        for bad in (
            "<bom>[.][C]<eom>",
            "<bom>[C][.][.][O]<eom>",
            "<bom>[C][.]<eom>",
            "<bom>[C[NH2]<eom>",
        ):
            with self.assertRaises(extractor.RecordReject):
                extractor.parse_motif_sequence(bad)

    def test_candidate_census_partition_and_anchor_measurement(self) -> None:
        self.assertEqual(self.receipt_a["counts"]["payload"], 3)
        self.assertEqual(self.receipt_a["counts"]["admitted"], 2)
        self.assertEqual(self.receipt_a["counts"]["rejected"], 1)
        self.assertFalse(self.receipt_a["training_admission"])
        rejects = [json.loads(line) for line in (self.release_a / "reject_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["reason_code"] for row in rejects], ["MOTIF_MAPPING_CARDINALITY_MISMATCH"])
        summary = json.loads((self.release_a / "anchor_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["record_p1_pair_rule_pass_count"], 1)
        self.assertEqual(summary["record_p1_pair_rule_fail_count"], 1)
        self.assertEqual(summary["anchor_label_multiplicity_histogram"], {"2": 1, "3": 1})
        self.assertEqual(summary["max_anchor_id"], 1)
        self.assertEqual(summary["records_with_stereo_at_count"], 1)

    def test_independent_seed_outputs_are_byte_identical(self) -> None:
        self.assertEqual(
            self.receipt_a["logical_derivation_sha256"],
            self.receipt_b["logical_derivation_sha256"],
        )
        for name in extractor.DETERMINISTIC_ARTIFACTS:
            self.assertEqual((self.release_a / name).read_bytes(), (self.release_b / name).read_bytes(), name)
        validated = rerun.validate_receipt(self.release_a)
        self.assertEqual(validated["counts"], self.receipt_a["counts"])

    def test_source_hash_mismatch_precedes_pickle(self) -> None:
        bad_lock = self.root / "bad_source_lock.json"
        value = json.loads(self.source_lock.read_text(encoding="utf-8"))
        value["source_sha256"] = "f" * 64
        write_json_new(bad_lock, value)
        with mock.patch.object(extractor.pickle, "loads", side_effect=AssertionError("pickle must not run")):
            with self.assertRaises(extractor.CensusError):
                extractor.extract_release(
                    self.source,
                    bad_lock,
                    P2_CONTRACT,
                    self.root / "must_not_be_created",
                    extractor.ACKNOWLEDGEMENT,
                )
        self.assertFalse((self.root / "must_not_be_created").exists())

    def test_acknowledgement_and_unknown_metadata_fail_closed(self) -> None:
        no_ack_output = self.root / "no_ack_must_not_be_created"
        with mock.patch.object(extractor.pickle, "loads", side_effect=AssertionError("pickle must not run")):
            with self.assertRaises(extractor.CensusError):
                extractor.extract_release(
                    self.source,
                    self.source_lock,
                    P2_CONTRACT,
                    no_ack_output,
                    "not-an-acknowledgement",
                )
        self.assertFalse(no_ack_output.exists())

        metadata_lock = self.root / "unknown_metadata_source_lock.json"
        value = json.loads(self.source_lock.read_text(encoding="utf-8"))
        value["expected_payload_entry_count"] = 4
        write_json_new(metadata_lock, value)
        entries = dict(self.entries)
        entries[b"__len__"] = b"3"
        output = self.root / "unknown_metadata_partial_attempt"
        with mock.patch.dict(sys.modules, {"lmdb": fake_lmdb_module(entries)}):
            with self.assertRaisesRegex(extractor.CensusError, "unrecognized LMDB metadata"):
                extractor.extract_release(
                    self.source,
                    metadata_lock,
                    P2_CONTRACT,
                    output,
                    extractor.ACKNOWLEDGEMENT,
                )
        self.assertTrue(output.is_dir())
        self.assertFalse((output / "derivation_receipt.json").exists())

    def test_payload_field_and_cid_checks_are_closed_rejects(self) -> None:
        extra = closed_payload(1, "<bom>[C]<eom>", [[0]])
        extra["unexpected"] = 1
        with self.assertRaisesRegex(extractor.RecordReject, "FIELD_SET_MISMATCH"):
            extractor.process_payload("1", pickle.dumps(extra, protocol=4))
        wrong_cid = closed_payload(2, "<bom>[C]<eom>", [[0]])
        with self.assertRaisesRegex(extractor.RecordReject, "KEY_CID_MISMATCH"):
            extractor.process_payload("1", pickle.dumps(wrong_cid, protocol=4))
        canonical_text_cid = closed_payload(1, "<bom>[C]<eom>", [[0]])
        canonical_text_cid["cid"] = "1"
        self.assertEqual(
            extractor.process_payload("1", pickle.dumps(canonical_text_cid, protocol=4))["cid"],
            1,
        )

    def test_projection_domain_audit_is_fail_closed_and_quantified(self) -> None:
        p1_root = self.root / "p1_release"
        p1_root.mkdir()
        p1_census = p1_root / "motif_census.jsonl"
        p1_fragments = {"C<0*>": 4, "<0*>O": 4, "CCC": 2}
        with p1_census.open("xb") as handle:
            for fragment in sorted(p1_fragments, key=lambda item: hashlib.sha256(item.encode("utf-8")).hexdigest()):
                digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
                handle.write(compatibility.canonical_json_bytes({
                    "motif_lexeme_sha256": digest,
                    "motif_fragment": fragment,
                    "count": p1_fragments[fragment],
                }) + b"\n")
        manifest = {
            "schema_version": compatibility.P1_MANIFEST_SCHEMA,
            "release_status": "complete",
            "configuration": {
                "production_contract_sha256": compatibility.sha256_file(P1_CONTRACT),
                "harness": {"components": {"molecule_native_linearizer": "e" * 64}},
            },
            "global_motif_census": compatibility.observe(p1_census, "motif_census.jsonl"),
            "counts": {
                "unique_motif_count": len(p1_fragments),
                "motif_occurrence_count": sum(p1_fragments.values()),
            },
        }
        write_json_new(p1_root / "full_release_manifest.json", manifest)
        output = self.root / "compatibility_report.json"
        report = compatibility.audit(argparse.Namespace(
            p1_release=p1_root,
            p1_contract=P1_CONTRACT,
            p2_release=self.release_a,
            p2_contract=P2_CONTRACT,
            p2_source_lock=self.source_lock,
            contract=COMPATIBILITY_CONTRACT,
            output=output,
        ))
        self.assertFalse(report["direct_projection_domain_compatible"])
        self.assertFalse(report["union_decision_permitted"])
        self.assertFalse(report["training_admission"])
        self.assertGreater(report["overlap"]["exact_lexeme"]["shared_unique_count"], 0)
        self.assertIn("P2_MOTIF_SEQUENCE_PRODUCER_UNKNOWN", report["incompatibility_reasons"])
        self.assertIn("P1_P2_LINEARIZATION_SEMANTIC_MAPPING_NOT_PROVEN", report["incompatibility_reasons"])
        self.assertNotIn("P1_P2_LINEARIZATION_SPEC_SHA256_DIFFERS", report["incompatibility_reasons"])
        self.assertIn("P2_LEGACY_ANCHOR_MULTIPLICITY_VIOLATES_P1_PAIR_RULE", report["incompatibility_reasons"])
        self.assertIn("P2_RETAINS_AT_STEREOCHEMISTRY_WHILE_P1_DOMAIN_OMITS_IT", report["incompatibility_reasons"])

    def test_producer_semantic_reason_matrix_avoids_cross_artifact_hash_category_error(self) -> None:
        p1_sha = "a" * 64
        unknown = compatibility.producer_semantic_reasons({
            "motif_sequence_producer_status": "unknown_legacy_producer",
            "motif_sequence_producer_sha256": None,
            "legacy_linearization_spec_sha256": "c" * 64,
        }, p1_sha)
        self.assertEqual(unknown, [
            "P2_MOTIF_SEQUENCE_PRODUCER_UNKNOWN",
            "P1_P2_LINEARIZATION_SEMANTIC_MAPPING_NOT_PROVEN",
        ])

        different = compatibility.producer_semantic_reasons({
            "motif_sequence_producer_status": "hash_locked",
            "motif_sequence_producer_sha256": "b" * 64,
            "legacy_linearization_spec_sha256": "c" * 64,
        }, p1_sha)
        self.assertEqual(different, ["P1_P2_LINEARIZATION_SEMANTIC_MAPPING_NOT_PROVEN"])

        same_producer_different_retrospective_spec = compatibility.producer_semantic_reasons({
            "motif_sequence_producer_status": "hash_locked",
            "motif_sequence_producer_sha256": p1_sha,
            "legacy_linearization_spec_sha256": "c" * 64,
        }, p1_sha)
        self.assertEqual(same_producer_different_retrospective_spec, [])


if __name__ == "__main__":
    unittest.main()
