#!/usr/bin/env python3
"""Tests for the logical-motif CE-first vNext contract gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


GATE_PATH = Path(__file__).resolve().with_name("validate_p1_logical_motif_vnext.py")
SPEC = importlib.util.spec_from_file_location("logical_motif_vnext_validator", str(GATE_PATH))
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class LogicalMotifVNextValidatorTest(unittest.TestCase):
    def make_valid_record(self, profile=validator.CE_PROFILE):
        record = {
            "schema_version": validator.RECORD_SCHEMA,
            "document_kind": validator.RECORD_KIND,
            "training_profile": profile,
            "bindings": {
                "release_id": "pcqm-production-v2-clean-fixture",
                "data_release_manifest_sha256": digest("release-manifest"),
                "geometry_record_schema_sha256": digest("geometry-schema-v2"),
                "geometry_record_content_sha256": digest("geometry-record-1"),
                "membership_manifest_sha256": digest("permitted-membership"),
                "tokenizer_contract_sha256": digest("tokenizer-contract"),
                "tokenizer_snapshot_sha256": digest("tokenizer-snapshot"),
                "identity_codec_sha256": digest("identity-codec"),
                "connection_codec_sha256": digest("connection-codec"),
            },
            "member": {
                "member_id": "pcqm4mv2:000000001",
                "storage_key": "000000001",
            },
            "dimensions": {
                "token_count": 6,
                "logical_motif_count": 2,
                "atom_count": 4,
                "source_atom_count": 4,
                "e3fp_level_count": 4,
            },
            "token_domain": {
                "input_ids": [0, 101, 201, 102, 201, 1],
                "attention_mask": [True, True, True, True, True, True],
                "token_to_logical_motif": [-1, 0, 0, 1, 1, -1],
                "token_role": ["boundary", "identity", "connection", "identity", "connection", "boundary"],
            },
            "logical_motif_domain": {
                "identity_spans": [[1, 2], [3, 4]],
                "connection_token_indices": [[2], [4]],
                "logical_to_carrier": [1, 3],
                "exact_identity_sha256": [digest("motif-0"), digest("motif-1")],
                "motif_geometry_valid": [True, True],
                "motif_atom_indices": [[0, 1], [2, 3]],
                "motif_slot_atom_indices": [[1], [2]],
                "slot_count": [1, 1],
                "cross_motif_bonds": [
                    {
                        "edge_id": 0,
                        "left": {"logical_motif_index": 0, "atom_index": 1, "slot_ordinal": 0},
                        "right": {"logical_motif_index": 1, "atom_index": 2, "slot_ordinal": 0},
                        "bond_type": "single",
                    }
                ],
            },
            "atom_domain": {
                "atom_to_logical_motif": [0, 0, 1, 1],
                "model_to_source_atom_index": [0, 1, 2, 3],
                "atom_valid_mask": [True, True, True, True],
                "atom_is_attachment": [False, True, True, False],
                "full_e3fp_ids": [
                    [10, 11, 12, 13],
                    [20, 21, -1, -1],
                    [30, 31, 32, -1],
                    [40, 41, 42, 43],
                ],
            },
            "masks": {
                "identity_recovery_mask": [True, False],
            },
            "mask_decision": {
                "objective": "identity_recovery_ce",
                "seed": 1,
                "epoch": 0,
                "mask_probability": 0.5,
                "selected_logical_motif_indices": [0],
                "decision_sha256": validator._mask_decision_sha256(
                    1,
                    0,
                    "pcqm4mv2:000000001",
                    "identity_recovery_ce",
                    0.5,
                    [0],
                ),
            },
        }
        if profile == validator.C3_PROFILE:
            record["masks"]["state_prediction_mask"] = [False, True]
            record["c3_teacher"] = {
                "contract_sha256": digest("c3-teacher-contract"),
                "target_source": "identity_query_free_full_e3fp_motif_mean",
                "reads_token_or_text": False,
                "reads_interface_role": False,
                "stop_gradient": True,
                "eval_mode": True,
                "ema_update_clock": "optimizer_step",
                "normalization": "unit_l2",
                "loss": "squared_l2",
                "target_dim": 3,
                "target_logical_motif_indices": [1],
                "target_vectors": [[1.0, 0.0, 0.0]],
            }
        return record

    def make_valid_decision(self, artifact_root, profile=validator.CE_PROFILE, decision="admit"):
        artifact_root = Path(artifact_root)
        release_id = "pcqm-production-v2-clean-fixture"

        def write_reference(relative_path, schema_version, artifact_kind, status=None, extra=None):
            content = {
                "schema_version": schema_version,
                "artifact_kind": artifact_kind,
                "subject_release_id": release_id,
            }
            if status is not None:
                content["status"] = status
            content.update(extra or {})
            destination = artifact_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            reference = {
                "path": relative_path.as_posix(),
                "sha256": file_digest(destination),
                "schema_version": schema_version,
                "artifact_kind": artifact_kind,
                "subject_release_id": release_id,
            }
            if status is not None:
                reference["status"] = status
            return reference

        release_manifest = write_reference(
            Path("candidate") / "release.json",
            "fixture/candidate-release/v1",
            validator.ADMISSION_REFERENCE_KINDS["release_manifest"],
            extra={"release_id": release_id, "release_status": "candidate"},
        )
        membership_manifest = write_reference(
            Path("candidate") / "membership.json",
            "fixture/p1-membership/v1",
            validator.ADMISSION_REFERENCE_KINDS["membership_manifest"],
        )
        tokenizer_contract = write_reference(
            Path("candidate") / "tokenizer.json",
            "fixture/p1-tokenizer/v1",
            validator.ADMISSION_REFERENCE_KINDS["tokenizer_contract"],
        )
        gate_names = set(validator.BASE_GATE_FIELDS)
        if profile == validator.C3_PROFILE:
            gate_names.update(validator.C3_GATE_FIELDS)
        evidence_receipts = {}
        for name in sorted(gate_names):
            status = "fail" if decision == "reject" and name == "pf_canary_passed" else "pass"
            evidence_receipts[name] = write_reference(
                Path("evidence") / (name + ".json"),
                validator.EVIDENCE_SCHEMA,
                validator.EVIDENCE_KIND_TEMPLATE.format(gate_name=name),
                status=status,
                extra={"gate_name": name},
            )
        document = {
            "schema_version": validator.ADMISSION_SCHEMA,
            "document_kind": validator.ADMISSION_KIND,
            "decision_id": "p1-admission-fixture-001",
            "created_at_utc": "2026-08-06T12:00:00+00:00",
            "training_profile": profile,
            "candidate_release": {
                "release_id": release_id,
                "release_status": "candidate",
                "release_manifest": release_manifest,
                "training_record_contract_sha256": file_digest(validator.RECORD_CONTRACT_PATH),
                "membership_manifest": membership_manifest,
                "tokenizer_contract": tokenizer_contract,
            },
            "decision": decision,
            "p1_admitted": decision == "admit",
            "authorized_stage": "pf_1" if decision == "admit" else "none",
            "evidence_receipts": evidence_receipts,
            "reason_codes": [] if decision == "admit" else ["PF_CANARY_FAILED"],
        }
        return document

    def test_contract_files_and_valid_ce_record_pass(self):
        self.assertEqual(
            validator.RECORD_CONTRACT["schema_version"],
            "most-t5-r1/p1-logical-motif-ce-first-contract/vnext1",
        )
        self.assertEqual(
            validator.ADMISSION_CONTRACT["schema_version"],
            "most-t5-r1/p1-admission-decision-contract/vnext1",
        )
        report = validator.validate_training_record(self.make_valid_record())
        self.assertTrue(report["pass"], report["errors"])
        self.assertEqual(report["artifact_kind"], validator.RECORD_KIND)

    def test_valid_record_passes_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path = root / "record.json"
            report_path = root / "report.json"
            artifact_path.write_text(json.dumps(self.make_valid_record(), indent=2), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GATE_PATH),
                    "--artifact",
                    str(artifact_path),
                    "--output",
                    str(report_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["pass"], report["errors"])

    def test_identity_recovery_is_mandatory_and_legacy_masks_are_forbidden(self):
        record = self.make_valid_record()
        record["masks"].pop("identity_recovery_mask")
        record["masks"]["joint_mask_positions"] = [True, False]
        report = validator.validate_training_record(record)
        self.assertFalse(report["pass"])
        paths = {error["path"] for error in report["errors"]}
        self.assertIn("masks.identity_recovery_mask", paths)
        self.assertIn("$.masks.joint_mask_positions", paths)

    def test_mask_decision_is_reproducible_and_bound_to_member(self):
        wrong_mask = self.make_valid_record()
        wrong_mask["masks"]["identity_recovery_mask"] = [False, True]
        wrong_mask["mask_decision"]["selected_logical_motif_indices"] = [1]
        wrong_mask["mask_decision"]["decision_sha256"] = validator._mask_decision_sha256(
            1,
            0,
            "pcqm4mv2:000000001",
            "identity_recovery_ce",
            0.5,
            [1],
        )
        report = validator.validate_training_record(wrong_mask)
        self.assertFalse(report["pass"])
        self.assertIn(
            "masks.identity_recovery_mask",
            {error["path"] for error in report["errors"]},
        )

        wrong_member = self.make_valid_record()
        wrong_member["member"]["member_id"] = "pcqm4mv2:000000002"
        report = validator.validate_training_record(wrong_member)
        self.assertFalse(report["pass"])
        paths = {error["path"] for error in report["errors"]}
        self.assertIn("masks.identity_recovery_mask", paths)
        self.assertIn("mask_decision.decision_sha256", paths)

    def test_ce_profile_forbids_state_mask_and_teacher(self):
        record = self.make_valid_record()
        record["masks"]["state_prediction_mask"] = [False, True]
        record["c3_teacher"] = copy.deepcopy(self.make_valid_record(validator.C3_PROFILE)["c3_teacher"])
        report = validator.validate_training_record(record)
        self.assertFalse(report["pass"])
        paths = {error["path"] for error in report["errors"]}
        self.assertIn("$.c3_teacher", paths)
        self.assertIn("masks.state_prediction_mask", paths)

    def test_three_domain_mapping_and_interface_invariants_are_enforced(self):
        record = self.make_valid_record()
        record["token_domain"]["token_to_logical_motif"][3] = 0
        record["logical_motif_domain"]["logical_to_carrier"][1] = 4
        record["atom_domain"]["atom_to_logical_motif"][2] = 0
        record["atom_domain"]["atom_is_attachment"] = [False, False, True, False]
        report = validator.validate_training_record(record)
        self.assertFalse(report["pass"])
        paths = {error["path"] for error in report["errors"]}
        self.assertIn("logical_motif_domain.identity_spans[1]", paths)
        self.assertIn("logical_motif_domain.logical_to_carrier[1]", paths)
        self.assertIn("atom_domain.atom_to_logical_motif[2]", paths)
        self.assertIn("atom_domain.atom_is_attachment", paths)

    def test_port_mapping_endpoint_order_and_dense_edge_ids_fail_closed(self):
        wrong_port = self.make_valid_record()
        wrong_port["logical_motif_domain"]["cross_motif_bonds"][0]["left"]["atom_index"] = 0
        wrong_port["atom_domain"]["atom_is_attachment"] = [True, False, True, False]
        report = validator.validate_training_record(wrong_port)
        self.assertFalse(report["pass"])
        self.assertIn(
            "logical_motif_domain.cross_motif_bonds[0].left.atom_index",
            {error["path"] for error in report["errors"]},
        )

        reversed_endpoints = self.make_valid_record()
        bond = reversed_endpoints["logical_motif_domain"]["cross_motif_bonds"][0]
        bond["left"], bond["right"] = bond["right"], bond["left"]
        report = validator.validate_training_record(reversed_endpoints)
        self.assertFalse(report["pass"])
        self.assertIn(
            "logical_motif_domain.cross_motif_bonds[0]",
            {error["path"] for error in report["errors"]},
        )

        nondense = self.make_valid_record()
        nondense["logical_motif_domain"]["cross_motif_bonds"][0]["edge_id"] = 99
        report = validator.validate_training_record(nondense)
        self.assertFalse(report["pass"])
        self.assertIn(
            "logical_motif_domain.cross_motif_bonds[0].edge_id",
            {error["path"] for error in report["errors"]},
        )

    def test_narrow_p1_geometry_and_source_mapping_fail_closed(self):
        invalid_geometry = self.make_valid_record()
        invalid_geometry["atom_domain"]["full_e3fp_ids"][0] = [-1, -1, -1, -1]
        report = validator.validate_training_record(invalid_geometry)
        self.assertFalse(report["pass"])
        self.assertIn(
            "atom_domain.full_e3fp_ids[0]",
            {error["path"] for error in report["errors"]},
        )

        permuted_source = self.make_valid_record()
        permuted_source["atom_domain"]["model_to_source_atom_index"] = [1, 0, 2, 3]
        report = validator.validate_training_record(permuted_source)
        self.assertFalse(report["pass"])
        self.assertIn(
            "atom_domain.model_to_source_atom_index",
            {error["path"] for error in report["errors"]},
        )

    def test_valid_c3_profile_passes(self):
        report = validator.validate_training_record(self.make_valid_record(validator.C3_PROFILE))
        self.assertTrue(report["pass"], report["errors"])

    def test_c3_profile_requires_disjoint_state_mask_and_query_free_teacher(self):
        record = self.make_valid_record(validator.C3_PROFILE)
        record["masks"]["state_prediction_mask"] = [True, False]
        record["c3_teacher"]["target_logical_motif_indices"] = [0]
        record["c3_teacher"]["reads_token_or_text"] = True
        record["c3_teacher"]["target_vectors"] = [[2.0, 0.0, 0.0]]
        report = validator.validate_training_record(record)
        self.assertFalse(report["pass"])
        paths = {error["path"] for error in report["errors"]}
        self.assertIn("masks.state_prediction_mask[0]", paths)
        self.assertIn("c3_teacher.reads_token_or_text", paths)
        self.assertIn("c3_teacher.target_vectors[0]", paths)

    def test_c3_profile_missing_conditional_fields_is_rejected(self):
        record = self.make_valid_record(validator.C3_PROFILE)
        record["masks"].pop("state_prediction_mask")
        record.pop("c3_teacher")
        report = validator.validate_training_record(record)
        self.assertFalse(report["pass"])
        paths = {error["path"] for error in report["errors"]}
        self.assertIn("$.c3_teacher", paths)
        self.assertIn("masks.state_prediction_mask", paths)

    def test_standalone_ce_admit_and_reject_decisions_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            admit_document = self.make_valid_decision(root)
            admit = validator.validate_admission_decision(admit_document, artifact_root=root)
            self.assertTrue(admit["pass"], admit["errors"])

            reject_document = self.make_valid_decision(root, decision="reject")
            reject_without_root = validator.validate_admission_decision(reject_document)
            reject_with_root = validator.validate_admission_decision(reject_document, artifact_root=root)
            self.assertTrue(reject_without_root["pass"], reject_without_root["errors"])
            self.assertTrue(reject_with_root["pass"], reject_with_root["errors"])

    def test_admit_requires_artifact_root_and_candidate_release_remains_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            no_root = validator.validate_admission_decision(self.make_valid_decision(root))
            self.assertFalse(no_root["pass"])
            self.assertIn("$.artifact_root", {error["path"] for error in no_root["errors"]})

            decision = self.make_valid_decision(root)
            decision["candidate_release"]["release_status"] = "admitted"
            decision["candidate_release"]["p1_admitted"] = True
            report = validator.validate_admission_decision(decision, artifact_root=root)
            self.assertFalse(report["pass"])
            paths = {error["path"] for error in report["errors"]}
            self.assertIn("candidate_release.release_status", paths)
            self.assertIn("candidate_release.p1_admitted", paths)

    def test_c3_admission_requires_conditional_gates_but_ce_forbids_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c3 = self.make_valid_decision(root / "c3", validator.C3_PROFILE)
            c3["evidence_receipts"].pop("teacher_optimizer_step_ema_passed")
            report = validator.validate_admission_decision(c3, artifact_root=root / "c3")
            self.assertFalse(report["pass"])
            self.assertIn(
                "evidence_receipts.teacher_optimizer_step_ema_passed",
                {error["path"] for error in report["errors"]},
            )

            ce = self.make_valid_decision(root / "ce")
            ce["evidence_receipts"]["teacher_profile_validation_passed"] = copy.deepcopy(
                c3["evidence_receipts"]["teacher_profile_validation_passed"]
            )
            report = validator.validate_admission_decision(ce, artifact_root=root / "ce")
            self.assertFalse(report["pass"])
            self.assertIn(
                "evidence_receipts.teacher_profile_validation_passed",
                {error["path"] for error in report["errors"]},
            )

    def test_zero_hashes_and_legacy_all_true_self_reports_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero_hashes = self.make_valid_decision(root)
            for name in ("release_manifest", "membership_manifest", "tokenizer_contract"):
                zero_hashes["candidate_release"][name]["sha256"] = "0" * 64
            zero_hashes["candidate_release"]["training_record_contract_sha256"] = "0" * 64
            for receipt in zero_hashes["evidence_receipts"].values():
                receipt["sha256"] = "0" * 64
            report = validator.validate_admission_decision(zero_hashes, artifact_root=root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "candidate_release.training_record_contract_sha256",
                {error["path"] for error in report["errors"]},
            )

            legacy = self.make_valid_decision(root)
            legacy["gate_results"] = {name: True for name in validator.BASE_GATE_FIELDS}
            legacy["evidence_sha256"] = {name: digest("evidence:" + name) for name in validator.BASE_GATE_FIELDS}
            report = validator.validate_admission_decision(legacy, artifact_root=root)
            self.assertFalse(report["pass"])
            paths = {error["path"] for error in report["errors"]}
            self.assertIn("$.gate_results", paths)
            self.assertIn("$.evidence_sha256", paths)

    def test_relative_paths_fail_closed_on_traversal_and_windows_absolute_syntax(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traversal = self.make_valid_decision(root)
            traversal["evidence_receipts"]["pf_canary_passed"]["path"] = "../outside.json"
            report = validator.validate_admission_decision(traversal, artifact_root=root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "evidence_receipts.pf_canary_passed.path",
                {error["path"] for error in report["errors"]},
            )

            absolute = self.make_valid_decision(root)
            absolute["candidate_release"]["membership_manifest"]["path"] = "C:\\outside\\membership.json"
            report = validator.validate_admission_decision(absolute, artifact_root=root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "candidate_release.membership_manifest.path",
                {error["path"] for error in report["errors"]},
            )

    def test_symlinked_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = self.make_valid_decision(root)
            receipt = decision["evidence_receipts"]["pf_canary_passed"]
            original = root / receipt["path"]
            link = original.with_name("pf_canary_link.json")
            try:
                link.symlink_to(original.name)
            except (OSError, NotImplementedError) as exc:
                # Windows commonly denies symlink creation outside Developer
                # Mode.  Keep the adversarial branch executable by simulating
                # the platform's link-like predicate on an otherwise identical
                # file; hosts with symlink support exercise the real path.
                link.write_bytes(original.read_bytes())
                original_detector = validator._is_link_like

                def simulated_link_detector(path):
                    if Path(path) == link:
                        return True
                    return original_detector(path)

                receipt["path"] = link.relative_to(root).as_posix()
                with mock.patch.object(validator, "_is_link_like", side_effect=simulated_link_detector):
                    report = validator.validate_admission_decision(decision, artifact_root=root)
                self.assertFalse(report["pass"], "symlink fallback was not rejected: {}".format(exc))
                self.assertIn(
                    "evidence_receipts.pf_canary_passed.path",
                    {error["path"] for error in report["errors"]},
                )
                return
            receipt["path"] = link.relative_to(root).as_posix()
            report = validator.validate_admission_decision(decision, artifact_root=root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "evidence_receipts.pf_canary_passed.path",
                {error["path"] for error in report["errors"]},
            )

    def test_hash_correct_but_metadata_mismatched_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = self.make_valid_decision(root)
            receipt = decision["evidence_receipts"]["pf_canary_passed"]
            evidence_path = root / receipt["path"]
            content = json.loads(evidence_path.read_text(encoding="utf-8"))
            content["subject_release_id"] = "a-different-release"
            evidence_path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            receipt["sha256"] = file_digest(evidence_path)
            report = validator.validate_admission_decision(decision, artifact_root=root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "evidence_receipts.pf_canary_passed.subject_release_id",
                {error["path"] for error in report["errors"]},
            )

    def test_candidate_manifest_binding_and_current_record_contract_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = self.make_valid_decision(root)
            manifest_ref = decision["candidate_release"]["release_manifest"]
            manifest_path = root / manifest_ref["path"]
            content = json.loads(manifest_path.read_text(encoding="utf-8"))
            content["release_id"] = "a-different-release"
            manifest_path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_ref["sha256"] = file_digest(manifest_path)
            decision["candidate_release"]["training_record_contract_sha256"] = digest("stale-contract")
            report = validator.validate_admission_decision(decision, artifact_root=root)
            self.assertFalse(report["pass"])
            paths = {error["path"] for error in report["errors"]}
            self.assertIn("candidate_release.release_manifest.path", paths)
            self.assertIn("candidate_release.training_record_contract_sha256", paths)

    def test_admit_cli_requires_and_uses_artifact_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision_path = root / "decision.json"
            decision_path.write_text(json.dumps(self.make_valid_decision(root), indent=2), encoding="utf-8")
            without_root = subprocess.run(
                [sys.executable, str(GATE_PATH), "--artifact", str(decision_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            with_root = subprocess.run(
                [
                    sys.executable,
                    str(GATE_PATH),
                    "--artifact",
                    str(decision_path),
                    "--artifact-root",
                    str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(without_root.returncode, 1, without_root.stdout + without_root.stderr)
            self.assertEqual(with_root.returncode, 0, with_root.stdout + with_root.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
