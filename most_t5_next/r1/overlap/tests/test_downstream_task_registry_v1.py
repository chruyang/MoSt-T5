from __future__ import print_function

import hashlib
import json
import re
import unittest
from pathlib import Path


R1_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = R1_ROOT / "contracts" / "downstream_task_registry_contract_v1.json"
REGISTRY = R1_ROOT / "overlap" / "configs" / "downstream_task_registry_20260806_v1.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DownstreamTaskRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_json(CONTRACT)
        cls.registry = load_json(REGISTRY)
        cls.tasks = {task["task_id"]: task for task in cls.registry["tasks"]}
        cls.secondary_tasks = {task["task_id"]: task for task in cls.registry["secondary_tasks"]}

    def test_contract_binding_and_schema_are_exact(self):
        self.assertEqual(
            self.registry["schema_version"],
            self.contract["registry_schema_version"],
        )
        self.assertEqual(self.registry["contract_sha256"], sha256_file(CONTRACT))
        resolved_contract = (REGISTRY.parent / self.registry["contract_path"]).resolve()
        self.assertEqual(resolved_contract, CONTRACT.resolve())

    def test_task_ids_are_unique_and_expected_scope_is_explicit(self):
        task_ids = [task["task_id"] for task in self.registry["tasks"]]
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(
            set(task_ids),
            {
                "qm9_orbital_property_generation_3dmolt5_hf_v1",
                "chebi20_text_to_molecule_3dmolt5_hf_v1",
                "pubchemqc_four_property_generation_legacy_v1",
                "pubchem_captioning_p2_lmdb_v1",
                "controlled_motif_editing_finemoltex_informed_v1",
            },
        )
        self.assertEqual(
            set(task_ids),
            set(self.registry["scope_declaration"]["first_round_admission_core_task_ids"]),
        )
        self.assertEqual(
            self.registry["observation"]["legacy_root_caption_named_file_count"],
            0,
        )
        roots = self.registry["observation"]["remote_source_roots"]
        self.assertEqual(
            set(roots),
            {"legacy_and_downstream_evidence_r0_v1", "p2_pubchem_evidence_r0_v1"},
        )
        for task in self.registry["tasks"]:
            if task["source_root_id"] is None:
                self.assertEqual(task["protocol_type"], "zero_shot_evaluation_benchmark")
                self.assertEqual(task["splits"]["train"]["source"]["status"], "not_applicable")
                self.assertTrue(
                    all(task["splits"][split]["source"]["status"] == "missing" for split in ("validation", "test"))
                )
            else:
                self.assertIn(task["source_root_id"], roots)

    def test_every_task_has_explicit_train_validation_test_roles_and_source_state(self):
        required_splits = self.contract["required_splits"]
        role_matrix = self.contract["required_split_roles_by_protocol"]
        allowed_protocols = set(self.contract["protocol_types"])
        allowed_source_statuses = set(self.contract["source_statuses"])
        allowed_task_statuses = set(self.contract["task_statuses"])
        allowed_scope_statuses = set(self.contract["task_scope_statuses"])
        for task in self.registry["tasks"]:
            self.assertIn(task["protocol_type"], allowed_protocols)
            expected_roles = role_matrix[task["protocol_type"]]
            self.assertIn(task["task_status"], allowed_task_statuses)
            self.assertIn(task["scope_status"], allowed_scope_statuses)
            self.assertEqual(task["scope_status"], "first_round_admission_core")
            self.assertEqual(set(task["splits"]), set(required_splits))
            for split in required_splits:
                split_record = task["splits"][split]
                self.assertEqual(split_record["role"], expected_roles[split])
                source = split_record["source"]
                self.assertIn(source["status"], allowed_source_statuses)
                if source["status"] in {"missing", "not_applicable"}:
                    for field in ("relative_path", "format", "bytes", "sha256", "record_count", "fields"):
                        self.assertIsNone(source[field])
                else:
                    self.assertTrue(source["relative_path"])
                    self.assertFalse(Path(source["relative_path"]).is_absolute())
                    self.assertIn(source["format"], {"parquet", "json_array", "legacy_lmdb_pickle"})
                    self.assertGreater(source["bytes"], 0)
                    self.assertRegex(source["sha256"], SHA256_PATTERN)
                    self.assertGreater(source["record_count"], 0)
                    self.assertTrue(source["fields"])

    def test_protection_union_covers_present_eval_tasks_and_exposes_missing_core(self):
        declared = set(self.registry["protection_scope"]["present_task_ids_in_protected_union"])
        missing = set(self.registry["protection_scope"]["missing_task_ids_not_yet_protectable"])
        observed_present = set()
        observed_missing = set()
        for task in self.registry["tasks"]:
            eval_statuses = {
                task["splits"][split]["source"]["status"]
                for split in self.contract["protection_policy"]["protected_splits"]
            }
            if "missing" in eval_statuses:
                observed_missing.add(task["task_id"])
            else:
                observed_present.add(task["task_id"])
        self.assertEqual(declared, observed_present)
        self.assertEqual(missing, observed_missing)
        self.assertTrue(declared.isdisjoint(missing))

    def test_qm9_duplicate_eval_artifact_is_fail_closed_not_ready(self):
        task = self.tasks["qm9_orbital_property_generation_3dmolt5_hf_v1"]
        validation = task["splits"]["validation"]["source"]
        test = task["splits"]["test"]["source"]
        self.assertEqual(task["task_status"], "blocked_source_split_integrity")
        self.assertTrue(task["preliminary_split_observations"]["byte_identical_validation_and_test"])
        self.assertEqual(validation["sha256"], test["sha256"])
        self.assertEqual(validation["bytes"], test["bytes"])
        self.assertGreater(
            task["preliminary_split_observations"]["exact_smiles_overlap_counts"]["train_validation"],
            0,
        )
        self.assertFalse(task["preliminary_split_observations"]["explicit_task_field_present"])

    def test_chebi_is_direct_but_still_pending_canonical_overlap_proof(self):
        task = self.tasks["chebi20_text_to_molecule_3dmolt5_hf_v1"]
        self.assertEqual(task["task_status"], "candidate_pending_canonical_identity_proof")
        self.assertEqual(task["identity_mapping"]["generic_extractor_compatibility"], "direct")
        self.assertEqual(task["identity_mapping"]["member_id"]["field"], "cid")
        self.assertEqual(task["identity_mapping"]["smiles_source"]["field"], "smiles")
        self.assertEqual(
            task["identity_mapping"]["text_identity"]["semantic_role"],
            "text_to_molecule_encoder_input",
        )
        exact = task["preliminary_split_observations"]["exact_smiles_overlap_counts"]
        source_ids = task["preliminary_split_observations"]["source_member_id_overlap_counts"]
        self.assertEqual(set(exact.values()), {0})
        self.assertEqual(set(source_ids.values()), {0})

    def test_pubchem_property_requires_explicit_join_and_split_provenance(self):
        task = self.tasks["pubchemqc_four_property_generation_legacy_v1"]
        self.assertEqual(task["task_status"], "blocked_adapter_and_split_provenance")
        self.assertEqual(task["identity_mapping"]["generic_extractor_compatibility"], "requires_hash_locked_join_adapter")
        self.assertEqual(task["identity_mapping"]["smiles_source"]["lmdb_value_field"], "smi")
        self.assertFalse(task["join_source"]["full_split_join_coverage_verified"])
        unique_total = 0
        for split in ("train", "validation", "test"):
            source = task["splits"][split]["source"]
            self.assertEqual(source["record_count"], source["unique_source_member_id_count"] * 4)
            unique_total += source["unique_source_member_id_count"]
        self.assertEqual(
            unique_total,
            task["split_evidence"]["downstream_blacklist"]["unique_member_id_count"],
        )
        self.assertTrue(task["split_evidence"]["observed_json_union_equals_downstream_blacklist"])
        self.assertEqual(
            set(task["split_evidence"]["observed_json_member_id_pairwise_overlap"].values()),
            {0},
        )
        self.assertGreater(
            sum(task["split_evidence"]["positional_interpretation_mismatch_counts_each_side"].values()),
            0,
        )

    def test_pubchem_caption_sources_are_locked_but_split_is_fail_closed(self):
        task = self.tasks["pubchem_captioning_p2_lmdb_v1"]
        self.assertEqual(task["task_status"], "blocked_source_split_integrity")
        self.assertTrue(all(split_record["source"]["status"] == "present_direct" for split_record in task["splits"].values()))
        self.assertEqual(
            {split: record["source"]["record_count"] for split, record in task["splits"].items()},
            {"train": 12000, "validation": 1000, "test": 2000},
        )
        self.assertEqual(
            {split: record["source"]["sha256"] for split, record in task["splits"].items()},
            {
                "train": "971aa5bac9fa800e46b917dc8d176281b8f38b7642403a6433b6547fb21025f6",
                "validation": "2b8c285264c15403294d5d37e2356e2aad3115770191454779be1f9647c92030",
                "test": "b500f0cb7514c3e74def9d99959c53cb6d0399893aef7777e568e78cd462f3e5",
            },
        )
        self.assertEqual(task["identity_mapping"]["member_id"]["source"], "lmdb_key")
        self.assertEqual(task["identity_mapping"]["member_id"]["crosscheck_field"], "cid")
        self.assertEqual(task["identity_mapping"]["member_id"]["prefix"], "pubchem_cid:")
        self.assertEqual(task["identity_mapping"]["smiles_source"]["field"], "smiles")
        text_identity = task["identity_mapping"]["text_identity"]
        self.assertEqual(text_identity["status"], "available_compatibility_target_frozen")
        self.assertEqual(text_identity["component_fields"], ["description"])
        self.assertNotIn("enriched_description", text_identity["component_fields"])
        self.assertEqual(task["identity_mapping"]["generic_extractor_compatibility"], "direct")
        self.assertEqual(
            task["preliminary_split_observations"]["exact_smiles_overlap_counts"]["train_validation"],
            1,
        )
        self.assertEqual(
            set(task["preliminary_split_observations"]["source_member_id_overlap_counts"].values()),
            {0},
        )
        self.assertFalse(self.registry["registry_decision"]["protection_union_ready_now"])
        self.assertFalse(self.registry["registry_decision"]["benchmark_suite_ready_now"])

    def test_controlled_motif_editing_stays_in_core_scope_and_blocks_admission(self):
        task = self.tasks["controlled_motif_editing_finemoltex_informed_v1"]
        self.assertEqual(task["scope_status"], "first_round_admission_core")
        self.assertEqual(task["protocol_type"], "zero_shot_evaluation_benchmark")
        self.assertEqual(task["task_status"], "blocked_protocol_manifest_missing")
        self.assertEqual(task["splits"]["train"]["role"], "not_applicable_no_supervised_train")
        self.assertEqual(task["splits"]["train"]["source"]["status"], "not_applicable")
        self.assertTrue(all(task["splits"][split]["source"]["status"] == "missing" for split in ("validation", "test")))
        self.assertIn(
            task["task_id"],
            self.registry["protection_scope"]["missing_task_ids_not_yet_protectable"],
        )
        self.assertEqual(
            task["protocol_state"]["status"],
            "reference_protocol_understood_project_protocol_not_frozen",
        )
        compatibility = task["protocol_state"]["finemoltex_compatibility_layer"]
        self.assertEqual(compatibility["training_split"], "not_applicable_zero_shot_latent_editing")
        self.assertEqual(task["protocol_state"]["project_main_layer"]["development_target_count"], 200)
        self.assertGreaterEqual(task["protocol_state"]["project_main_layer"]["sealed_test_minimum_target_count"], 1000)
        self.assertGreaterEqual(len(task["protocol_state"]["required_before_source_admission"]), 5)
        self.assertFalse(self.registry["registry_decision"]["first_round_admission_ready_now"])

    def test_zero_shot_retrieval_is_secondary_and_never_claimed_as_protected(self):
        declared = self.registry["scope_declaration"]["secondary_not_in_first_round_admission_task_ids"]
        self.assertEqual(set(declared), set(self.secondary_tasks))
        task = self.secondary_tasks["zero_shot_retrieval_secondary_v1"]
        self.assertEqual(task["scope_status"], "secondary_not_in_first_round_admission")
        self.assertFalse(task["protected_union_inclusion"])
        self.assertNotIn(
            task["task_id"],
            self.registry["protection_scope"]["present_task_ids_in_protected_union"],
        )
        self.assertNotIn(
            task["task_id"],
            self.registry["protection_scope"]["missing_task_ids_not_yet_protectable"],
        )
        self.assertIn("No zero-shot retrieval", task["current_claim"])


if __name__ == "__main__":
    unittest.main()
