#!/usr/bin/env python3
"""Small hermetic self-test for validate_data_release_manifest.py.

It creates only temporary text fixtures, never opens a project LMDB or a
remote dataset, and exercises both the Python API and the CLI source-lock
verification path.
"""

from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


GATE_PATH = Path(__file__).resolve().with_name("validate_data_release_manifest.py")
SPEC = importlib.util.spec_from_file_location("r1_release_validator", str(GATE_PATH))
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReleaseManifestValidatorTest(unittest.TestCase):
    def make_source_lock(self, root, name, role, location_scope="remote_shared", directory=False):
        sources = root / "sources"
        sources.mkdir(exist_ok=True)
        if directory:
            target = sources / name
            target.mkdir()
            (target / "tokenizer.json").write_text("tokenizer-fixture\n", encoding="utf-8")
            observed = validator.sha256_directory_tree(target)
            kind = "directory"
        else:
            target = sources / (name + ".txt")
            target.write_text("{}:{}\n".format(name, role), encoding="utf-8")
            observed = validator.sha256_file(target)
            kind = "file"
        return {
            "name": name,
            "role": role,
            "kind": kind,
            "path": str(target.relative_to(root).as_posix()),
            "bytes": observed["bytes"],
            "sha256": observed["sha256"],
            "location_scope": location_scope,
            "immutable": True,
        }

    def make_valid_manifest(self, root):
        locks = [
            self.make_source_lock(root, "p1_split", "p1_membership_source"),
            self.make_source_lock(root, "p1_e3fp", "p1_geometry_source"),
            self.make_source_lock(root, "p2_split", "p2_membership_source"),
            self.make_source_lock(root, "p2_e3fp", "p2_geometry_source"),
            # These code/schema locks intentionally use remote_shared.  The
            # release can validate the exact adapter beside remote data; they
            # are not forced into a local-code-only location.
            self.make_source_lock(root, "p1_adapter", "p1_adapter_harness", "remote_shared"),
            self.make_source_lock(root, "p1_schema", "p1_record_schema", "remote_shared"),
            self.make_source_lock(root, "tokenizer_base", "tokenizer_base_snapshot", directory=True),
            self.make_source_lock(root, "tokenizer_builder", "tokenizer_builder_harness", "local_code_only"),
            self.make_source_lock(root, "tokenizer_contract", "stable_tokenizer_contract", "local_code_only"),
            self.make_source_lock(root, "geometry_policy", "geometry_policy_spec", "local_code_only"),
            self.make_source_lock(root, "identity_method", "identity_exclusion_method", "local_code_only"),
        ]
        by_role = {lock["role"]: lock for lock in locks}
        p1_member_manifest = digest("p1-member-manifest")
        p2_member_manifest = digest("p2-member-manifest")
        return {
            "schema_version": validator.MANIFEST_SCHEMA,
            "release": {
                "release_id": "self-test-candidate",
                "status": "candidate",
                "data_transfer_scope": "remote_only",
                "p1_admitted": False,
                "p2_admitted": False,
            },
            "source_profile": "legacy_3dmolm_control",
            "source_locks": locks,
            "phases": {
                "p1": {
                    "source_lock_names": ["p1_split", "p1_e3fp", "p1_adapter", "p1_schema"],
                    "same_mol_adapter": {
                        "schema_version": validator.P1_SAME_MOL_ADAPTER_SCHEMA,
                        "adapter_harness_sha256": by_role["p1_adapter_harness"]["sha256"],
                        "record_schema_sha256": by_role["p1_record_schema"]["sha256"],
                        "same_mol_derivation": validator.P1_SAME_MOL_DERIVATION,
                        "source_atom_index_tag": validator.P1_SOURCE_ATOM_INDEX_TAG,
                        "source_atom_index_tagged_pre_removehs": True,
                        "retained_source_index_validation_required": True,
                        "compacted_index_inference_forbidden": True,
                        "smiles_rebuild_for_alignment_forbidden": True,
                        "frozen_mol_linearizer_required": True,
                        "required_batch_fields": list(validator.P1_REQUIRED_MAINLINE_BATCH_FIELDS),
                        "legacy_mask_positions_forbidden": True,
                        "geometry_input_mask_definition": validator.P1_GEOMETRY_INPUT_MASK_DEFINITION,
                        "geometry_target_mask_required": True,
                        "geometry_target_mask_definition": validator.P1_GEOMETRY_TARGET_MASK_DEFINITION,
                    },
                    "membership": {
                        "identity_namespace": "pubchemqc_id",
                        "source_record_count": 3119717,
                        "geometry_admitted_record_count": 3119714,
                        "member_ids_sha256": digest("p1-member-ids"),
                        "manifest_file_sha256": p1_member_manifest,
                        "reject_ledger": {
                            "file_sha256": digest("p1-reject-file"),
                            "record_count": 3,
                            "ids_sha256": digest("p1-reject-ids"),
                            "reason_counts": {"E3FP_UNSUPPORTED_H2": 3},
                        },
                    },
                },
                "p2": {
                    "source_lock_names": ["p2_split", "p2_e3fp"],
                    "membership": {
                        "identity_namespace": "pubchem_cid",
                        "source_record_count": 301658,
                        "geometry_admitted_record_count": 301655,
                        "text_or_2d_only_record_count": 3,
                        "member_ids_sha256": digest("p2-member-ids"),
                        "manifest_file_sha256": p2_member_manifest,
                        "reject_ledger": {
                            "file_sha256": digest("p2-reject-file"),
                            "record_count": 3,
                            "ids_sha256": digest("p2-reject-ids"),
                            "reason_counts": {"E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS": 3},
                        },
                    },
                },
            },
            "geometry_mask_policy": {
                "schema_version": "most-t5-r1/geometry-mask-policy/v1",
                "policy_file_sha256": by_role["geometry_policy_spec"]["sha256"],
                "mse_task_scope": ["mmm"],
                "per_example_geometry_validity_mask": True,
                "zero_vector_as_geometry_target_forbidden": True,
                "mask_statistics_required": True,
                "padding_mapped_atom_action": "mask_or_exclude",
                "geometryless_motif_action": "mask_or_exclude",
                "mask_statistics_manifest_sha256": digest("mask-statistics"),
                "known_invalid_geometry_case_counts": {
                    "padding_mapped_atom_count": 353,
                    "geometryless_motif_group_count": 1508,
                },
                "reject_reason_actions": {
                    "E3FP_UNSUPPORTED_H2": {
                        "release_action": "exclude_from_geometry_release",
                        "geometry_mse_enabled": False,
                    },
                    "E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS": {
                        "release_action": "keep_text_or_2d_only_mask_geometry",
                        "geometry_mse_enabled": False,
                    },
                },
            },
            "downstream_identity_exclusion": {
                "method_sha256": by_role["identity_exclusion_method"]["sha256"],
                "downstream_validation_test_ids_sha256": digest("downstream-heldout-ids"),
                "p1_overlap_count": 0,
                "p2_overlap_count": 0,
            },
            "cross_phase_overlap": {
                "policy": "explicitly_declared",
                "overlap_count": 0,
                "evidence_sha256": digest("p1-p2-overlap"),
            },
            "tokenizer": {
                "contract_manifest_sha256": by_role["stable_tokenizer_contract"]["sha256"],
                "base_snapshot_tree_sha256": by_role["tokenizer_base_snapshot"]["sha256"],
                "builder_harness_sha256": by_role["tokenizer_builder_harness"]["sha256"],
                "special_token_spec_sha256": digest("special-token-order"),
                "ordered_motif_vocab_sha256": digest("ordered-motif-vocab"),
                "id_to_token_sha256": digest("id-to-token"),
                "vocab_size": 52306,
                "p1_p2_exact_same_mapping": True,
                "p2_vocab_extension_forbidden": True,
                "permitted_membership_manifest_sha256": {
                    "p1": p1_member_manifest,
                    "p2": p2_member_manifest,
                },
                "determinism_gate": {
                    "passed": True,
                    "pythonhashseeds": [0, 1, 271828],
                    "id_to_token_sha256": digest("id-to-token"),
                },
            },
            "checkpoint_prerequisites": {
                "p1_to_p2": {
                    "legacy_p1_checkpoint_allowed": False,
                    "same_tokenizer_mapping_required": True,
                    "strict_load_required": True,
                    "ignore_mismatched_sizes_forbidden": True,
                    "equal_vocab_size_required": True,
                    "embedding_shape_match_required": True,
                    "checkpoint_tokenizer_snapshot_required": True,
                    "checkpoint_release_manifest_required": True,
                }
            },
        }

    def make_valid_pcqm_manifest(self, root):
        """Build a PCQM candidate fixture without reading any molecular data."""
        manifest = self.make_valid_manifest(root)
        identity_lock = self.make_source_lock(
            root,
            "pcqm_identity_normalization",
            "pcqm_identity_normalization_contract",
            "remote_shared",
        )
        manifest["source_locks"].append(identity_lock)
        p1 = manifest["phases"]["p1"]
        p1["source_lock_names"].append("pcqm_identity_normalization")
        p1["same_mol_adapter"]["identity_normalization_contract_sha256"] = identity_lock["sha256"]
        p1["membership"].update(
            {
                "identity_namespace": "ogb_pcqm4mv2_train_row_index",
                "source_record_count": 3378606,
                "geometry_admitted_record_count": 3378604,
                "member_ids_sha256": digest("pcqm-p1-member-ids"),
                "manifest_file_sha256": digest("pcqm-p1-member-manifest"),
                "reject_ledger": {
                    "file_sha256": digest("pcqm-p1-reject-file"),
                    "record_count": 2,
                    "ids_sha256": digest("pcqm-p1-reject-ids"),
                    "reason_counts": {"PCQM_STEREO_2D3D_DIVERGENCE": 2},
                },
            }
        )
        manifest["tokenizer"]["permitted_membership_manifest_sha256"]["p1"] = p1["membership"]["manifest_file_sha256"]
        manifest["geometry_mask_policy"]["reject_reason_actions"]["PCQM_STEREO_2D3D_DIVERGENCE"] = {
            "release_action": "exclude_from_geometry_release",
            "geometry_mse_enabled": False,
        }
        manifest["source_profile"] = "pcqm4mv2_candidate"
        return manifest

    def test_valid_candidate_passes_api_and_cli_source_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_valid_manifest(root)
            manifest_path = root / "candidate.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            report = validator.validate_release_manifest(
                manifest,
                manifest_path=manifest_path,
                verify_source_locks=True,
            )
            self.assertTrue(report["pass"], report["errors"])
            self.assertEqual(report["warning_count"], 0)
            self.assertEqual(len(report["source_lock_observations"]), 11)

            output_path = root / "validation-report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GATE_PATH),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--verify-source-locks",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(persisted["pass"])

    def test_r0_invariants_and_tokenizer_preconditions_reject_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_valid_manifest(root)
            manifest["phases"]["p1"]["membership"]["geometry_admitted_record_count"] = 3119715
            manifest["tokenizer"]["p2_vocab_extension_forbidden"] = False
            report = validator.validate_release_manifest(manifest, manifest_path=root / "candidate.json")
            self.assertFalse(report["pass"])
            paths = set(error["path"] for error in report["errors"])
            self.assertIn("phases.p1.membership.geometry_admitted_record_count", paths)
            self.assertIn("tokenizer.p2_vocab_extension_forbidden", paths)

    def test_p1_same_mol_adapter_declarations_and_hash_binding_reject_regression(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_valid_manifest(root)
            adapter = manifest["phases"]["p1"]["same_mol_adapter"]
            adapter["same_mol_derivation"] = "separate_2d_and_3d_molecules"
            adapter["source_atom_index_tag"] = "legacy_atom_index"
            adapter["source_atom_index_tagged_pre_removehs"] = False
            adapter["retained_source_index_validation_required"] = False
            adapter["compacted_index_inference_forbidden"] = False
            adapter["smiles_rebuild_for_alignment_forbidden"] = False
            adapter["frozen_mol_linearizer_required"] = False
            adapter["required_batch_fields"] = ["mask_positions"]
            adapter["legacy_mask_positions_forbidden"] = False
            adapter["geometry_input_mask_definition"] = "mask_positions"
            adapter["geometry_target_mask_required"] = False
            adapter["geometry_target_mask_definition"] = "mask_positions AND token_geometry_valid_mask"
            adapter["mask_positions"] = "legacy_merged_mask"
            adapter["record_schema_sha256"] = digest("wrong-record-schema")
            report = validator.validate_release_manifest(manifest, manifest_path=root / "candidate.json")
            self.assertFalse(report["pass"])
            paths = set(error["path"] for error in report["errors"])
            self.assertIn("phases.p1.same_mol_adapter.same_mol_derivation", paths)
            self.assertIn("phases.p1.same_mol_adapter.source_atom_index_tag", paths)
            self.assertIn("phases.p1.same_mol_adapter.source_atom_index_tagged_pre_removehs", paths)
            self.assertIn("phases.p1.same_mol_adapter.retained_source_index_validation_required", paths)
            self.assertIn("phases.p1.same_mol_adapter.compacted_index_inference_forbidden", paths)
            self.assertIn("phases.p1.same_mol_adapter.smiles_rebuild_for_alignment_forbidden", paths)
            self.assertIn("phases.p1.same_mol_adapter.frozen_mol_linearizer_required", paths)
            self.assertIn("phases.p1.same_mol_adapter.required_batch_fields", paths)
            self.assertIn("phases.p1.same_mol_adapter.legacy_mask_positions_forbidden", paths)
            self.assertIn("phases.p1.same_mol_adapter.geometry_input_mask_definition", paths)
            self.assertIn("phases.p1.same_mol_adapter.geometry_target_mask_required", paths)
            self.assertIn("phases.p1.same_mol_adapter.geometry_target_mask_definition", paths)
            self.assertIn("phases.p1.same_mol_adapter.mask_positions", paths)
            self.assertIn("phases.p1.same_mol_adapter.record_schema_sha256", paths)

    def test_v1_merged_mask_definition_cannot_pass_v3_mainline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_valid_manifest(root)
            adapter = manifest["phases"]["p1"]["same_mol_adapter"]
            adapter["geometry_input_mask_definition"] = "mask_positions"
            adapter["geometry_target_mask_definition"] = "mask_positions AND token_geometry_valid_mask"
            report = validator.validate_release_manifest(manifest, manifest_path=root / "candidate.json")
            self.assertFalse(report["pass"])
            paths = set(error["path"] for error in report["errors"])
            self.assertIn("phases.p1.same_mol_adapter.geometry_input_mask_definition", paths)
            self.assertIn("phases.p1.same_mol_adapter.geometry_target_mask_definition", paths)

    def test_pcqm_profile_requires_identity_normalization_lock_and_p1_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_valid_pcqm_manifest(root)
            report = validator.validate_release_manifest(
                manifest,
                manifest_path=root / "pcqm-candidate.json",
                verify_source_locks=True,
            )
            self.assertTrue(report["pass"], report["errors"])
            self.assertEqual(len(report["source_lock_observations"]), 12)

            broken = copy.deepcopy(manifest)
            broken["phases"]["p1"]["source_lock_names"].remove("pcqm_identity_normalization")
            report = validator.validate_release_manifest(broken, manifest_path=root / "pcqm-candidate.json")
            self.assertFalse(report["pass"])
            paths = set(error["path"] for error in report["errors"])
            self.assertIn("phases.p1.source_lock_names", paths)
            self.assertIn("phases.p1.same_mol_adapter.identity_normalization_contract_sha256", paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
