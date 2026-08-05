from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

from most_t5_next.r1.tokenizer import build_motif_tokenizer_binding_release_v1 as builder
from most_t5_next.r1.tokenizer import validate_motif_tokenizer_binding_release_v1 as validator


ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "most_t5_next/r1/contracts/motif_tokenizer_binding_release_contract_v1.json"
BUILDER = ROOT / "most_t5_next/r1/tokenizer/build_motif_tokenizer_binding_release_v1.py"
VALIDATOR = ROOT / "most_t5_next/r1/tokenizer/validate_motif_tokenizer_binding_release_v1.py"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(builder.canonical_json_bytes(row) + b"\n")


def _new_test_root() -> Path:
    # Do not recursively delete test artifacts: workspace policy forbids bulk
    # deletion.  Each run uses a new isolated directory under the system temp.
    return Path(tempfile.gettempdir()) / ("most-t5-tokenizer-test-" + uuid.uuid4().hex)


class ProjectionUnitTests(unittest.TestCase):
    def test_projection_preserves_exact_core_and_anchor_order(self):
        anchors, anchor_ids, token = builder.project_lexeme("<12*>[NH]<3*>")
        self.assertEqual(anchors, ["<12*>", "<3*>"])
        self.assertEqual(anchor_ids, [12, 3])
        self.assertEqual(token, "[[NH]]")

    def test_projection_rejects_noncanonical_or_empty_anchor_core(self):
        with self.assertRaises(builder.ContractError):
            builder.project_lexeme("C<00*>")
        with self.assertRaises(builder.ContractError):
            builder.project_lexeme("<0*>")

    def test_many_exact_lexemes_aggregate_to_one_pure_token(self):
        left = {
            _sha("C<0*>"): ("C<0*>", 5),
            _sha("<0*>C"): ("<0*>C", 3),
            _sha("O"): ("O", 2),
        }
        policy = {
            "min_selection_score": 2,
            "max_motif_tokens": 1,
            "selection_score": {"kind": "weighted_integer_count", "p1_weight": 1, "p2_weight": 0},
            "anchor_policy": {"max_anchor_id_inclusive": 9, "overflow_action": "fail_closed"},
        }
        bindings, pure, selected, stats = builder.aggregate_projection(left, {}, policy)
        self.assertEqual(selected, ["[C]"])
        by_token = {row["pure_motif_token"]: row for row in pure}
        self.assertEqual(by_token["[C]"]["p1_count"], 8)
        self.assertEqual(by_token["[C]"]["exact_lexeme_count"], 2)
        self.assertEqual(len(bindings), 3)
        self.assertEqual(stats["selected_pure_motif_count"], 1)

    def test_policy_requires_explicit_p2_union_weight_and_decision(self):
        policy = _candidate_policy()
        policy["discovery_scope"] = "p1_p2_permitted_train_union"
        with self.assertRaises(builder.ContractError):
            _validate_policy_object(policy)
        policy["selection_score"]["p2_weight"] = 1
        policy["decision_status"] = "UNRESOLVED"
        with self.assertRaises(builder.ContractError):
            _validate_policy_object(policy)

    def test_complete_p2_scope_requires_projection_domain_compatibility_audit(self):
        root = _new_test_root()
        root.mkdir(parents=True, exist_ok=False)
        fragment = "C"
        census = root / "p2_census.jsonl"
        _write_jsonl(census, [{
            "motif_lexeme_sha256": _sha(fragment),
            "motif_fragment": fragment,
            "count": 1,
        }])
        lock = root / "p2_scope.json"
        _write_json(lock, {
            "schema_version": builder.SCOPE_LOCK_SCHEMA,
            "phase": "P2",
            "scope_status": "complete",
            "identity_namespace": "pubchem_cid",
            "membership_manifest_sha256": "1" * 64,
            "membership_count": 1,
            "downstream_identity_exclusion_proof_sha256": "2" * 64,
            "census_sha256": builder.sha256_file(census),
            "census_unique_lexeme_count": 1,
            "census_occurrence_count": 1,
            "census_kind": "permitted_membership_derived",
            "census_derivation_audit_sha256": "3" * 64,
            "source_release_logical_root_sha256": "4" * 64,
            "motif_linearization_spec_sha256": "5" * 64,
            "motif_sequence_extraction_spec_sha256": "6" * 64,
            "projection_domain_compatibility_audit_sha256": None,
        })
        with self.assertRaises(builder.ContractError):
            builder.validate_scope_lock(lock, "P2", census, "9" * 64)

    def test_sample_sequences_require_hash_bound_extraction_receipt(self):
        root = _new_test_root()
        root.mkdir(parents=True, exist_ok=False)
        sample = root / "sample.jsonl"
        _write_jsonl(sample, [])
        with self.assertRaises(validator.ValidationError):
            validator.validate_sample_receipt(None, sample, "1" * 64, "2" * 64)

    def test_census_rejects_boolean_count(self):
        root = _new_test_root()
        root.mkdir(parents=True, exist_ok=False)
        census = root / "bad_count.jsonl"
        _write_jsonl(census, [{
            "motif_lexeme_sha256": _sha("C"),
            "motif_fragment": "C",
            "count": True,
        }])
        lock = {"census_unique_lexeme_count": 1, "census_occurrence_count": 1}
        with self.assertRaises(builder.ContractError):
            builder.read_census(census, lock, "P1")


def _candidate_policy():
    return {
        "schema_version": builder.POLICY_SCHEMA,
        "decision_status": "approved_for_candidate",
        "discovery_scope": "p1_only",
        "min_selection_score": 2,
        "max_motif_tokens": 2,
        "selection_score": {"kind": "weighted_integer_count", "p1_weight": 1, "p2_weight": 0},
        "oov_policy": {"kind": "base_unk", "token": "[UNK]"},
        "anchor_policy": {"max_anchor_id_inclusive": 9, "overflow_action": "fail_closed"},
        "reserved_special_token_count": 4,
        "base_model": {"identifier": "synthetic/bert-with-t5-sentinels", "revision": "fixture-v1"},
        "base_vocab_overlap_allowlist": [],
        "tie_break": ["selection_score_desc", "pure_motif_utf8_asc", "pure_motif_sha256_asc"],
        "p2_vocab_extension_forbidden": True,
    }


def _validate_policy_object(policy):
    root = _new_test_root()
    root.mkdir(parents=True, exist_ok=False)
    path = root / "policy.json"
    _write_json(path, policy)
    return builder.validate_policy(path)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("transformers") is not None,
    "transformers is required for the offline integration test",
)
class OfflineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AddedToken, BertTokenizer

        cls.root = _new_test_root()
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.production = cls.root / "production"
        cls.production.mkdir()
        fragments = ["C<0*>", "<0*>C", "O", "[NH]"]
        counts = {"C<0*>": 5, "<0*>C": 3, "O": 2, "[NH]": 1}
        rows = [
            {"motif_lexeme_sha256": _sha(fragment), "motif_fragment": fragment, "count": counts[fragment]}
            for fragment in sorted(fragments, key=_sha)
        ]
        cls.census = cls.production / "motif_census.jsonl"
        _write_jsonl(cls.census, rows)
        logical_root = "1" * 64
        production_manifest = {
            "schema_version": builder.PRODUCTION_SCHEMA,
            "release_status": "complete",
            "release_id": "synthetic-production-v2",
            "logical_release_root_sha256": logical_root,
            "global_motif_census": builder.observe_artifact(cls.census, "motif_census.jsonl"),
            "tokenizer_binding": "absent_and_forbidden",
            "p1_training_admission": False,
        }
        _write_json(cls.production / "full_release_manifest.json", production_manifest)

        cls.scope = cls.root / "p1_scope.json"
        _write_json(cls.scope, {
            "schema_version": builder.SCOPE_LOCK_SCHEMA,
            "phase": "P1",
            "scope_status": "candidate",
            "identity_namespace": "synthetic_member",
            "membership_manifest_sha256": "2" * 64,
            "membership_count": 3,
            "downstream_identity_exclusion_proof_sha256": None,
            "census_sha256": builder.sha256_file(cls.census),
            "census_unique_lexeme_count": len(rows),
            "census_occurrence_count": sum(counts.values()),
            "census_kind": "production_global_admitted_candidate",
            "census_derivation_audit_sha256": None,
            "source_release_logical_root_sha256": logical_root,
            "motif_linearization_spec_sha256": "9" * 64,
            "motif_sequence_extraction_spec_sha256": "a" * 64,
            "projection_domain_compatibility_audit_sha256": None,
        })
        cls.policy = cls.root / "policy.json"
        _write_json(cls.policy, _candidate_policy())

        cls.base = cls.root / "base_snapshot"
        cls.base.mkdir()
        vocab_path = cls.base / "vocab.txt"
        with vocab_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\natom\nbond\n")
        tokenizer = BertTokenizer(str(vocab_path), do_lower_case=False)
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=True)
                for token in builder.T5_SENTINELS
            ]
        })
        tokenizer.save_pretrained(str(cls.base))
        observed = builder.observe_tree(cls.base)
        cls.base_lock = cls.root / "base_lock.json"
        _write_json(cls.base_lock, {
            "schema_version": builder.BASE_LOCK_SCHEMA,
            "decision_status": "approved_for_candidate",
            "model_identifier": "synthetic/bert-with-t5-sentinels",
            "revision": "fixture-v1",
            "expected_tokenizer_class": "BertTokenizer",
            "tokenizer_and_model_same_revision": True,
            "snapshot_tree_sha256": observed["tree_sha256"],
            "files": observed["files"],
        })

        cls.release = cls.root / "binding_release"
        env = os.environ.copy()
        env.update({
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        command = [
            sys.executable, str(BUILDER),
            "--contract", str(CONTRACT),
            "--production-release-root", str(cls.production),
            "--p1-scope-lock", str(cls.scope),
            "--p1-census", str(cls.census),
            "--selection-policy", str(cls.policy),
            "--base-snapshot", str(cls.base),
            "--base-snapshot-lock", str(cls.base_lock),
            "--release-id", "synthetic-binding-candidate-v1",
            "--output-dir", str(cls.release),
            "--hash-seeds", "0", "1", "271828",
        ]
        cls.build = subprocess.run(command, cwd=str(ROOT), env=env, text=True, capture_output=True)
        if cls.build.returncode != 0:
            raise AssertionError("builder failed\nSTDOUT:\n{}\nSTDERR:\n{}".format(cls.build.stdout, cls.build.stderr))

        digests = [_sha("C<0*>"), _sha("<0*>C"), _sha("O")]
        cls.samples = cls.root / "samples.jsonl"
        _write_jsonl(cls.samples, [{
            "schema_version": validator.SAMPLE_SCHEMA,
            "member_id": "synthetic:000000000",
            "record_content_sha256": "3" * 64,
            "motif_count": 3,
            "motif_lexeme_sha256": digests,
            "motif_atom_indices_count": 3,
            "motif_geometry_valid_count": 3,
        }])
        cls.sample_receipt = cls.root / "sample_receipt.json"
        _write_json(cls.sample_receipt, {
            "schema_version": validator.SAMPLE_RECEIPT_SCHEMA,
            "status": "pass",
            "production_logical_release_root_sha256": logical_root,
            "production_manifest_sha256": builder.sha256_file(cls.production / "full_release_manifest.json"),
            "sample_schedule_sha256": "5" * 64,
            "sample_jsonl_sha256": builder.sha256_file(cls.samples),
            "sample_record_count": 1,
            "safe_payload_decoder_sha256": "6" * 64,
            "payload_index_verification_report_sha256": "7" * 64,
            "component_reference_audit_sha256": "8" * 64,
        })
        cls.report = cls.root / "validation_report.json"
        validate_command = [
            sys.executable, str(VALIDATOR),
            "--release-dir", str(cls.release),
            "--contract", str(CONTRACT),
            "--production-release-root", str(cls.production),
            "--base-snapshot", str(cls.base),
            "--base-snapshot-lock", str(cls.base_lock),
            "--selection-policy", str(cls.policy),
            "--p1-scope-lock", str(cls.scope),
            "--p1-census", str(cls.census),
            "--sample-digest-sequences", str(cls.samples),
            "--sample-extraction-receipt", str(cls.sample_receipt),
            "--require-sample-count", "1",
            "--output-report", str(cls.report),
        ]
        cls.validation = subprocess.run(validate_command, cwd=str(ROOT), env=env, text=True, capture_output=True)

    def test_builder_and_independent_validator_pass_offline(self):
        self.assertEqual(self.build.returncode, 0, self.build.stderr)
        self.assertEqual(self.validation.returncode, 0, self.validation.stderr)
        with (self.release / "tokenizer_release_manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        with self.report.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(manifest["release_status"], "candidate_tokenizer_built_non_release")
        self.assertTrue(manifest["determinism_gate"]["pass"])
        self.assertEqual(manifest["statistics"]["exact_lexeme_count"], 4)
        self.assertEqual(manifest["statistics"]["pure_motif_count"], 3)
        self.assertEqual(manifest["statistics"]["selected_pure_motif_count"], 2)
        self.assertFalse(manifest["p1_training_admission"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["sample_digest_sequence_validation"]["record_count"], 1)
        self.assertFalse(report["p1_training_admission"])

    def test_sample_validator_rejects_singleton_anchor(self):
        bad = self.root / "bad_singleton_sample.jsonl"
        _write_jsonl(bad, [{
            "schema_version": validator.SAMPLE_SCHEMA,
            "member_id": "synthetic:bad",
            "record_content_sha256": "4" * 64,
            "motif_count": 1,
            "motif_lexeme_sha256": [_sha("C<0*>")],
            "motif_atom_indices_count": 1,
            "motif_geometry_valid_count": 1,
        }])
        bindings = {}
        with (self.release / "motif_digest_binding.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                bindings[row["motif_lexeme_sha256"]] = row
        with self.assertRaises(validator.ValidationError):
            validator.validate_samples(bad, bindings, 9)


if __name__ == "__main__":
    unittest.main()
