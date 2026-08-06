from __future__ import print_function

import json
import unittest
from pathlib import Path


R1_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = R1_ROOT / "contracts" / "downstream_scientific_registry_contract_v2.json"
REGISTRY = (
    R1_ROOT
    / "overlap"
    / "configs"
    / "downstream_scientific_registry_20260807_v2.json"
)
SOURCE_ACQUISITION_MANIFEST = (
    R1_ROOT
    / "overlap"
    / "configs"
    / "downstream_source_acquisition_manifest_20260807_v1.json"
)
V1_REGISTRY = (
    R1_ROOT
    / "overlap"
    / "configs"
    / "downstream_task_registry_20260806_v1.json"
)


CORE_TASK_IDS = {
    "qm9_orbital_property_scientific_v2",
    "pubchem_captioning_scientific_v2",
    "chebi20_text_to_molecule_scientific_v2",
    "controlled_motif_editing_scientific_v2",
}
SUPPORTING_TASK_IDS = {
    "moleculenet_bace_scientific_v2",
    "moleculenet_bbbp_scientific_v2",
    "moleculenet_hiv_scientific_v2",
    "moleculenet_clintox_scientific_v2",
}
DEFERRED_TASK_IDS = {
    "pubchemqc_property_deferred_v2",
    "zero_shot_retrieval_deferred_v2",
}


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def nested_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_keys(child)


class DownstreamScientificRegistryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_json(CONTRACT)
        cls.registry = load_json(REGISTRY)
        cls.tasks = {task["task_id"]: task for task in cls.registry["tasks"]}

    def test_contract_binding_is_semantic_and_v1_is_preserved(self):
        self.assertEqual(
            self.registry["schema_version"],
            self.contract["registry_schema_version"],
        )
        self.assertEqual(self.registry["contract_id"], self.contract["contract_id"])
        self.assertEqual(self.registry["contract_version"], self.contract["contract_version"])
        resolved_contract = (REGISTRY.parent / self.registry["contract_path"]).resolve()
        self.assertEqual(resolved_contract, CONTRACT.resolve())

        self.assertTrue(V1_REGISTRY.is_file())
        v1 = load_json(V1_REGISTRY)
        self.assertEqual(v1["registry_id"], self.registry["supersedes_for_scientific_planning"])
        self.assertTrue(self.registry["preserves_v1_as_evidence_inventory"])
        self.assertEqual(
            (REGISTRY.parent / self.registry["v1_evidence_registry_path"]).resolve(),
            V1_REGISTRY.resolve(),
        )
        acquisition = load_json(SOURCE_ACQUISITION_MANIFEST)
        self.assertEqual(
            (REGISTRY.parent / self.registry["source_acquisition_manifest_path"]).resolve(),
            SOURCE_ACQUISITION_MANIFEST.resolve(),
        )
        self.assertEqual(
            self.registry["source_acquisition_manifest_id"],
            acquisition["manifest_id"],
        )

    def test_contract_required_fields_and_allowed_values_are_enforced(self):
        for field in self.contract["required_top_level_fields"]:
            self.assertIn(field, self.registry)

        allowed = self.contract["allowed_values"]
        for task in self.registry["tasks"]:
            for field in self.contract["required_task_fields"]:
                self.assertIn(field, task, msg="{} missing {}".format(task["task_id"], field))
            self.assertIn(task["paper_role"], allowed["paper_roles"])
            self.assertIn(task["result_role"], allowed["result_roles"])
            self.assertIn(task["protocol_type"], allowed["protocol_types"])
            self.assertIn(task["source_basis"]["state"], allowed["source_states"])
            self.assertIn(task["split_protocol"]["status"], allowed["split_statuses"])
            self.assertIn(task["readiness_status"], allowed["readiness_statuses"])
            self.assertIn(
                task["decontamination"]["protection_timing"],
                allowed["protection_timings"],
            )

    def test_scope_is_exact_unique_and_role_partitioned(self):
        task_ids = [task["task_id"] for task in self.registry["tasks"]]
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(set(task_ids), CORE_TASK_IDS | SUPPORTING_TASK_IDS | DEFERRED_TASK_IDS)

        scope = self.registry["paper_scope"]
        self.assertEqual(set(scope["core_task_ids"]), CORE_TASK_IDS)
        self.assertEqual(set(scope["supporting_task_ids"]), SUPPORTING_TASK_IDS)
        self.assertEqual(set(scope["deferred_task_ids"]), DEFERRED_TASK_IDS)
        self.assertEqual(
            {task_id for task_id, task in self.tasks.items() if task["paper_role"] == "core"},
            CORE_TASK_IDS,
        )
        self.assertEqual(
            {task_id for task_id, task in self.tasks.items() if task["paper_role"] == "supporting"},
            SUPPORTING_TASK_IDS,
        )
        self.assertEqual(
            {task_id for task_id, task in self.tasks.items() if task["paper_role"] == "deferred"},
            DEFERRED_TASK_IDS,
        )
        self.assertEqual(
            set(scope["headline_priority_task_ids"]),
            {
                "qm9_orbital_property_scientific_v2",
                "pubchem_captioning_scientific_v2",
                "controlled_motif_editing_scientific_v2",
            },
        )
        self.assertEqual(
            scope["core_supporting_result_task_ids"],
            ["chebi20_text_to_molecule_scientific_v2"],
        )

    def test_minimum_decontamination_semantics_are_consistent(self):
        allowed = self.contract["allowed_values"]
        protected_splits = set(allowed["protected_split_names"])
        for task_id in CORE_TASK_IDS | SUPPORTING_TASK_IDS:
            task = self.tasks[task_id]
            policy = task["decontamination"]
            self.assertEqual(set(policy["protected_splits"]), protected_splits)
            self.assertEqual(policy["protection_timing"], "before_full_p1")
            self.assertEqual(policy["primary_identity"], allowed["primary_identity"])
            self.assertEqual(
                policy["downstream_train_overlap"],
                allowed["downstream_train_overlap_rule"],
            )
            self.assertEqual(
                policy["global_scaffold_removal"],
                allowed["global_scaffold_rule"],
            )
            self.assertFalse(policy["blocks_architecture_canary"])
            self.assertTrue(policy["blocks_full_p1"])
            self.assertTrue(task["split_protocol"]["required_before_full_p1"])
            self.assertIn(
                task["split_protocol"]["grouping_unit"],
                allowed["split_grouping_units"],
            )

        self.assertEqual(
            self.tasks["qm9_orbital_property_scientific_v2"]["split_protocol"]["grouping_unit"],
            "canonical_isomeric_molecular_identity",
        )
        for task_id in (CORE_TASK_IDS | SUPPORTING_TASK_IDS) - {
            "qm9_orbital_property_scientific_v2"
        }:
            self.assertEqual(
                self.tasks[task_id]["split_protocol"]["grouping_unit"],
                "canonical_connectivity",
            )

        for task_id in DEFERRED_TASK_IDS:
            task = self.tasks[task_id]
            policy = task["decontamination"]
            self.assertEqual(policy["protected_splits"], [])
            self.assertEqual(policy["protection_timing"], "not_in_current_protected_union")
            self.assertFalse(policy["blocks_architecture_canary"])
            self.assertFalse(policy["blocks_full_p1"])
            self.assertFalse(task["split_protocol"]["required_before_full_p1"])

        top_policy = self.registry["decontamination_policy"]
        self.assertEqual(top_policy["current_frozen_task_ids"], [])
        self.assertEqual(top_policy["downstream_train_overlap"], "retain_and_disclose")
        self.assertEqual(
            top_policy["global_scaffold_removal"],
            "no_global_pretraining_scaffold_exclusion",
        )

    def test_qm9_keeps_3dmolt5_payload_but_replaces_the_leaking_split(self):
        task = self.tasks["qm9_orbital_property_scientific_v2"]
        self.assertEqual(
            task["source_basis"]["state"],
            "3dmolt5_artifact_frozen_clean_split_materialization_pending",
        )
        self.assertEqual(
            task["split_protocol"]["status"],
            "derived_protocol_frozen_materialization_pending",
        )
        self.assertEqual(
            task["readiness_status"],
            "source_frozen_clean_split_materialization_pending",
        )
        self.assertIn("byte-identical", task["source_basis"]["observed_artifacts"])
        self.assertIn("349660", task["source_basis"]["observed_artifacts"])
        self.assertEqual(
            [
                task["split_protocol"][split]["molecule_count"]
                for split in ("train", "validation", "test")
            ],
            [110000, 10000, 8836],
        )
        self.assertEqual(
            [
                task["split_protocol"][split]["row_count"]
                for split in ("train", "validation", "test")
            ],
            [298529, 27211, 23920],
        )
        self.assertIn("canonical-isomeric", task["split_protocol"]["multi_view_grouping"])
        self.assertIn("non-isomeric", task["split_protocol"]["split_identity_boundary"])
        self.assertIn("HOMO", task["split_protocol"]["multi_view_grouping"])
        self.assertEqual(
            task["comparison_protocol"]["reported_view"]["status"],
            "diagnostic_only_not_headline",
        )
        self.assertIn("3dmolt5", task["comparison_protocol"]["clean_view"]["name"])
        self.assertIn("MAE_Hartree_per_property", task["metrics"]["primary"])
        self.assertIn("direct_regression_probe_MAE", task["metrics"]["secondary"])

    def test_caption_and_chebi_keep_reported_and_clean_views_separate(self):
        caption = self.tasks["pubchem_captioning_scientific_v2"]
        self.assertEqual(
            [
                caption["split_protocol"][split]["row_count"]
                for split in ("train", "validation", "test")
            ],
            [12000, 1000, 2000],
        )
        self.assertNotEqual(
            caption["comparison_protocol"]["reported_view"]["name"],
            caption["comparison_protocol"]["clean_view"]["name"],
        )
        for metric in ("BLEU-2", "BLEU-4", "ROUGE-L", "METEOR"):
            self.assertIn(metric, caption["metrics"]["primary"])
        self.assertIn("property_and_motif_factuality", caption["metrics"]["secondary"])

        chebi = self.tasks["chebi20_text_to_molecule_scientific_v2"]
        self.assertEqual(chebi["paper_role"], "core")
        self.assertEqual(chebi["result_role"], "core_supporting_result")
        self.assertEqual(
            [chebi["split_protocol"][split]["row_count"] for split in ("train", "validation", "test")],
            [26407, 3301, 3300],
        )
        self.assertNotEqual(
            chebi["comparison_protocol"]["reported_view"]["name"],
            chebi["comparison_protocol"]["clean_view"]["name"],
        )
        for metric in ("molecular_validity", "canonical_exact_match"):
            self.assertIn(metric, chebi["metrics"]["primary"])
        for metric in ("Morgan_similarity", "FCD", "Text2Mol_alignment"):
            self.assertIn(metric, chebi["metrics"]["secondary"])

    def test_editing_protocol_is_manifest_driven_not_arbitrary_size_driven(self):
        task = self.tasks["controlled_motif_editing_scientific_v2"]
        split = task["split_protocol"]
        self.assertEqual(task["protocol_type"], "zero_shot_controlled_editing")
        self.assertEqual(
            task["source_basis"]["state"],
            "sealed_test_member_source_frozen_development_members_pending",
        )
        self.assertEqual(split["train"]["status"], "not_applicable_no_supervised_training")
        self.assertEqual(split["validation"]["source_dataset"], "ZINC250K")
        self.assertIsNone(split["validation"]["source_molecule_count"])
        self.assertTrue(split["validation"]["must_be_disjoint_from_sealed_test"])
        self.assertEqual(
            split["validation"]["size_rule"],
            "justify_and_freeze_without_arbitrary_minimum",
        )
        self.assertIsNone(split["validation"]["evaluation_pair_count"])
        self.assertEqual(
            split["validation"]["pairing_rule"],
            "freeze_explicit_manifest_do_not_assume_full_cartesian_product",
        )
        self.assertEqual(split["test"]["source_molecule_count"], 200)
        self.assertEqual(split["test"]["prompt_count"], 12)
        self.assertTrue(split["test"]["must_be_independent"])
        self.assertTrue(split["test"]["sealed_for_checkpoint_selection"])
        self.assertIn("published_moleculestm_200_member", split["test"]["status"])
        self.assertIn(
            "sealed_test",
            task["comparison_protocol"]["reported_view"]["name"],
        )
        self.assertIn(
            "zinc250k_disjoint_development",
            task["comparison_protocol"]["clean_view"]["name"],
        )
        self.assertIn("target_edit_success_rate", task["metrics"]["primary"])
        self.assertIn("attachment_atom_correctness", task["metrics"]["secondary"])
        self.assertIn("stereochemistry_preservation", task["metrics"]["secondary"])

    def test_kpgt_layout_candidates_are_internally_valid_but_not_official(self):
        expected = {
            "moleculenet_bace_scientific_v2": (1513, [1210, 151, 152]),
            "moleculenet_bbbp_scientific_v2": (2039, [1631, 203, 205]),
            "moleculenet_clintox_scientific_v2": (1478, [1182, 147, 149]),
        }
        for task_id, (member_count, partition_sizes) in expected.items():
            task = self.tasks[task_id]
            self.assertEqual(task["paper_role"], "supporting")
            self.assertEqual(task["result_role"], "supporting_transfer_result")
            self.assertEqual(
                task["source_basis"]["state"],
                "kpgt_layout_candidate_present_official_comparison_pending",
            )
            self.assertIsNone(task["source_basis"]["declared_revision"])
            self.assertIn(
                "ff2de71fa6bb0533d5e740db6d88a0442a0d38e8",
                task["source_basis"]["verified_fallback_revision"],
            )
            self.assertEqual(
                set(task["source_basis"]["official_source_metadata_ids"]),
                {
                    "kpgt_official_github_revisions_v1",
                    "kpgt_figshare_19914811_metadata_v1",
                },
            )
            self.assertEqual(len(task["source_basis"]["candidate_source_acquisition_artifact_ids"]), 4)
            self.assertIn("http_403", task["source_basis"]["version_status"])
            self.assertIn("do not prove official KPGT equivalence", task["source_basis"]["observed_artifacts"])
            self.assertEqual(
                task["split_protocol"]["status"],
                "kpgt_layout_candidate_validated_official_comparison_pending",
            )
            self.assertEqual(
                task["split_protocol"]["split_method"],
                "candidate_murcko_scaffold_8_1_1_not_verified_official",
            )
            self.assertEqual(
                task["split_protocol"]["candidate_member_count"],
                member_count,
            )
            self.assertEqual(
                set(tuple(value) for value in task["split_protocol"]["candidate_partition_sizes"].values()),
                {tuple(partition_sizes)},
            )
            validation = task["split_protocol"]["candidate_validation"]
            self.assertTrue(
                validation["each_split_complete_and_within_split_pairwise_disjoint_duplicate_free"]
            )
            self.assertEqual(validation["murcko_overlap_include_chirality_false"], 0)
            self.assertEqual(validation["murcko_overlap_include_chirality_true"], 0)
            self.assertTrue(validation["all_smiles_valid"])
            self.assertTrue(validation["auroc_computable_each_partition"])
            self.assertIsNone(task["split_protocol"]["seed_or_manifest"])
            self.assertEqual(task["geometry_protocol"]["status"], "conformer_recipe_pending")
            self.assertEqual(
                task["readiness_status"],
                "kpgt_candidate_present_official_source_comparison_pending",
            )
            self.assertIn("3D-MolT5", task["next_actions"][0])
            self.assertIn("Figshare", task["next_actions"][0])
            self.assertIn("protected union", task["next_actions"][1])
            self.assertIn("In parallel", task["next_actions"][2])
            self.assertTrue(any(metric.startswith("ROC-AUC") for metric in task["metrics"]["primary"]))

        clintox = self.tasks["moleculenet_clintox_scientific_v2"]
        self.assertEqual(clintox["metrics"]["primary"], ["ROC-AUC_macro", "ROC-AUC_per_label"])
        self.assertIn("Both labels", clintox["split_protocol"]["multi_view_grouping"])

    def test_hiv_uses_no_kpgt_claim_and_requires_authoritative_derived_v1(self):
        task = self.tasks["moleculenet_hiv_scientific_v2"]
        self.assertEqual(
            task["source_basis"]["state"],
            "deepchem_authoritative_source_frozen_derived_split_materialization_pending",
        )
        self.assertNotIn("official_source_metadata_ids", task["source_basis"])
        self.assertEqual(
            task["source_basis"]["primary_source_acquisition_artifact_id"],
            "deepchem_hiv_csv_authoritative_v1",
        )
        self.assertIn("official 3D-MolT5", task["source_basis"]["observed_artifacts"])
        self.assertIn("KPGT's released eleven-task set also excludes HIV", task["source_basis"]["observed_artifacts"])
        self.assertIn("authoritative DeepChem", task["source_basis"]["reported_protocol"])
        self.assertIn("Never name", task["source_basis"]["clean_protocol"])
        split = task["split_protocol"]
        self.assertEqual(split["status"], "source_frozen_derived_membership_materialization_pending")
        self.assertEqual(
            split["protocol_id"],
            "HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1",
        )
        self.assertIn("deepchem_2_8_0_scaffoldsplitter", split["authority"])
        self.assertEqual(
            split["split_method"],
            "deepchem_2_8_0_nonchiral_bemis_murcko_groups_sorted_by_group_size_then_first_source_index_descending_and_greedily_assigned_at_0_8_0_9_row_cutoffs",
        )
        self.assertEqual(
            split["seed_or_manifest"],
            "no_rng_seed_unused_by_algorithm_publish_exact_member_manifest",
        )
        self.assertEqual(
            task["readiness_status"],
            "source_frozen_derived_split_materialization_pending",
        )
        self.assertIn("without using a 3D-MolT5, KPGT", task["next_actions"][1])

    def test_full_p1_gate_is_membership_only_not_downstream_adapter_completion(self):
        full_p1 = self.registry["planning_decisions"]["gate_summary"]["full_p1"]
        self.assertIn("dataset_versions_validation_test_memberships_canonical_identities", full_p1)
        self.assertIn("not_until_downstream_evaluators_or_conformer_adapters", full_p1)
        boundary = self.contract["decontamination_policy"]["downstream_readiness_boundary"]
        self.assertIn("not additional full-P1 admission conditions", boundary)

    def test_pubchemqc_and_retrieval_are_deferred_without_hidden_objectives(self):
        pubchemqc = self.tasks["pubchemqc_property_deferred_v2"]
        retrieval = self.tasks["zero_shot_retrieval_deferred_v2"]
        for task in (pubchemqc, retrieval):
            self.assertEqual(task["paper_role"], "deferred")
            self.assertEqual(task["result_role"], "deferred_no_current_result")
            self.assertEqual(task["readiness_status"], "deferred_not_in_current_protocol")
            self.assertEqual(task["metrics"]["primary"], [])
            self.assertEqual(task["metrics"]["secondary"], [])
        self.assertIn("after the QM9", pubchemqc["next_actions"][0])
        self.assertIn("Do not add a global contrastive objective", retrieval["next_actions"][0])

    def test_v2_stays_a_lightweight_scientific_protocol_not_a_file_audit(self):
        keys = set(nested_keys(self.registry))
        self.assertNotIn("contract_sha256", keys)
        self.assertNotIn("sha256", keys)
        self.assertNotIn("bytes", keys)
        self.assertNotIn("file_size", keys)


if __name__ == "__main__":
    unittest.main()
