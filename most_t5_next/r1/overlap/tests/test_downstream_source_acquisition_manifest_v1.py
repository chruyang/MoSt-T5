from __future__ import print_function

import json
import re
import unittest
from pathlib import Path


R1_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    R1_ROOT
    / "overlap"
    / "configs"
    / "downstream_source_acquisition_manifest_20260807_v1.json"
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


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


class DownstreamSourceAcquisitionManifestV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST)
        cls.sources = {source["source_id"]: source for source in cls.manifest["sources"]}
        cls.artifacts = {
            artifact["artifact_id"]: artifact
            for source in cls.manifest["sources"]
            for artifact in source["artifacts"]
        }
        cls.code_evidence = {
            evidence["evidence_id"]: evidence
            for evidence in cls.manifest["code_evidence"]
        }

    def test_manifest_is_versioned_remote_only_and_not_an_admission(self):
        self.assertEqual(
            self.manifest["schema_version"],
            "most-t5-r1/downstream-source-acquisition-manifest/v1",
        )
        self.assertEqual(
            self.manifest["manifest_id"],
            "downstream-source-acquisition-manifest-20260807-v1",
        )
        locator = self.manifest["storage_locator"]
        self.assertEqual(
            locator["storage_root"],
            "/root/autodl-fs/most-t5-r1/sources/downstream-scientific-v2-20260807",
        )
        self.assertEqual(
            locator["location_semantics"],
            "current_remote_locator_not_scientific_identity",
        )
        self.assertFalse(locator["local_download_materialized"])
        self.assertNotIn("local_path", set(nested_keys(self.manifest)))
        self.assertEqual(
            self.manifest["scientific_boundary"]["admission_status"],
            "not_full_p1_admitted",
        )

    def test_canonical_sources_and_revisions_are_exact(self):
        self.assertEqual(
            set(self.sources),
            {
                "moleculestm_hf_ff2de71_v1",
                "deepchem_hiv_authoritative_20200807_v1",
                "finemoltex_zenodo_15501037_v1",
                "kpgt_official_github_revisions_v1",
                "kpgt_figshare_19914811_metadata_v1",
                "kpgt_layout_candidate_legacy_r0_v1",
            },
        )
        molecule_stm = self.sources["moleculestm_hf_ff2de71_v1"]
        self.assertEqual(molecule_stm["origin"]["repository_id"], "chao1224/MoleculeSTM")
        self.assertEqual(
            molecule_stm["revision"],
            "ff2de71fa6bb0533d5e740db6d88a0442a0d38e8",
        )

        deepchem_hiv = self.sources["deepchem_hiv_authoritative_20200807_v1"]
        self.assertEqual(deepchem_hiv["origin"]["repository_id"], "deepchem/deepchem")
        self.assertEqual(deepchem_hiv["revision"]["deepchem_version"], "2.8.0")
        self.assertEqual(
            deepchem_hiv["revision"]["deepchem_commit"],
            "d5b293934d427062f52e2d92c1569d53d10418f9",
        )

        finemoltex = self.sources["finemoltex_zenodo_15501037_v1"]
        self.assertEqual(finemoltex["origin"]["record_id"], "15501037")
        self.assertEqual(finemoltex["origin"]["doi"], "10.5281/zenodo.15501037")
        self.assertEqual(finemoltex["revision"]["release"], "V0.0.0")
        self.assertEqual(
            finemoltex["revision"]["archive_root"],
            "liushiliushi-FineMolTex-fc56d6d",
        )

    def test_artifact_ids_paths_and_hashes_are_complete(self):
        artifact_ids = [
            artifact["artifact_id"]
            for source in self.manifest["sources"]
            for artifact in source["artifacts"]
        ]
        self.assertEqual(len(artifact_ids), len(set(artifact_ids)))
        self.assertEqual(len(artifact_ids), 23)
        for artifact_id, artifact in self.artifacts.items():
            self.assertFalse(Path(artifact["remote_relative_path"]).is_absolute())
            if artifact_id.startswith("kpgt_layout_candidate_"):
                self.assertNotIn("downloads/", artifact["remote_relative_path"])
            else:
                self.assertTrue(artifact["remote_relative_path"].startswith("downloads/"))

        for artifact_id, artifact in self.artifacts.items():
            if artifact_id == "finemoltex_v0_0_0_zip_v1":
                continue
            self.assertRegex(artifact["checksums"]["sha256"], SHA256_PATTERN)

        archive = self.artifacts["finemoltex_v0_0_0_zip_v1"]
        self.assertEqual(archive["bytes"], 225134229)
        self.assertRegex(
            archive["checksums"]["md5_from_zenodo_metadata"],
            MD5_PATTERN,
        )
        self.assertEqual(
            archive["checksums"]["md5_from_zenodo_metadata"],
            "cbfd0819189d27caa899f40b3f0a0afc",
        )
        self.assertRegex(
            archive["checksums"]["sha256_remote_recomputed"],
            SHA256_PATTERN,
        )
        self.assertEqual(
            archive["checksums"]["sha256_remote_recomputed"],
            "861cb0ceccd99cf07bb459dc70432be7d60006a27852abe540063b1eae013ae8",
        )

    def test_editing_member_source_and_code_evidence_are_exact(self):
        editing = self.artifacts["moleculestm_editing_single_multi_property_smiles_v1"]
        self.assertEqual(
            editing["checksums"]["sha256"],
            "6fa09f81402bebe7c14c12404c4c5edba5de32abf8d88f4f733485b992e79daa",
        )
        self.assertEqual(editing["count"]["record_count"], 200)
        self.assertEqual(editing["count"]["unique_record_count"], 200)
        self.assertIn("sealed_test", editing["scientific_role"])

        default_source = self.code_evidence["finemoltex_default_editing_source_v1"]
        self.assertEqual(
            default_source["archive_member_path"],
            "liushiliushi-FineMolTex-fc56d6d/scripts/generation_Optimization.py",
        )
        self.assertEqual((default_source["line_start"], default_source["line_end"]), (146, 146))

        prompts = self.code_evidence["finemoltex_prompt_ids_v1"]
        self.assertEqual(
            prompts["archive_member_path"],
            "liushiliushi-FineMolTex-fc56d6d/scripts/FineMolTex/downstream_molecule_edit_utils.py",
        )
        self.assertEqual((prompts["line_start"], prompts["line_end"]), (34, 72))
        self.assertEqual(
            prompts["prompt_ids"],
            [101, 102, 103, 104, 105, 106, 205, 206, 501, 502, 503, 504],
        )

        binding = self.manifest["task_bindings"]["controlled_motif_editing_scientific_v2"]
        self.assertIn("sealed", binding["frozen_fact"])
        self.assertIn("zinc250k_connectivity_disjoint_development_membership", binding["full_p1_membership_blockers"])
        self.assertNotIn(
            "source_conformer_recipe",
            binding["full_p1_membership_blockers"],
        )
        self.assertIn(
            "source_conformer_recipe",
            binding["downstream_evaluation_blockers_not_full_p1_blockers"],
        )

    def test_moleculenet_counts_hashes_and_fallback_boundary_are_exact(self):
        expected = {
            "moleculenet_bace_scientific_v2": (
                "moleculestm_bace_processed_smiles_v1",
                "moleculestm_bace_raw_csv_v1",
                1513,
                "12a893de0cb3b7ff174fb9ca7314534facfb8fbcfb8ef5e64a54449c9a7629cf",
                "894a1efd23d2c391c6f6ee0fb5f719bf9d6633a6b534ec02472fa11e1d5862a3",
            ),
            "moleculenet_bbbp_scientific_v2": (
                "moleculestm_bbbp_processed_smiles_v1",
                "moleculestm_bbbp_raw_csv_v1",
                2039,
                "9642ee43377730979f3771e8ea9ca505606f62cb294a78182e4aceea4fedc0c3",
                "6aecd84110e8c0549bc4e2e486c52267d4e775178fca8999f5465029e31e75bb",
            ),
            "moleculenet_hiv_scientific_v2": (
                "moleculestm_hiv_processed_smiles_v1",
                "moleculestm_hiv_raw_csv_v1",
                41127,
                "9d026d8ecd844190d9a627686d35f1ca0925db8c92a5d5ff95596d8c028755da",
                "8df84857edc602ae9267a73ebf2b68451f75c159637726cdc392dd7ef297b592",
            ),
            "moleculenet_clintox_scientific_v2": (
                "moleculestm_clintox_processed_smiles_v1",
                "moleculestm_clintox_raw_csv_v1",
                1478,
                "d19b6585041992eec7b32d67912cbafca3b14b68eeb0dbab5392b4dd34f2f86f",
                "f5c9da01b5f5b188db9e0fb611796d344e9259acd3ef59e2cfcb17557b455563",
            ),
        }
        for task_id, (artifact_id, raw_id, count, sha256_value, raw_sha256) in expected.items():
            binding = self.manifest["task_bindings"][task_id]
            self.assertEqual(binding["fallback_member_source_artifact_id"], artifact_id)
            self.assertEqual(binding["raw_companion_artifact_id"], raw_id)
            artifact = self.artifacts[artifact_id]
            raw_artifact = self.artifacts[raw_id]
            self.assertEqual(artifact["scientific_role"], "verified_fallback_member_source_evidence")
            self.assertEqual(artifact["count"]["record_count"], count)
            self.assertTrue(artifact["count"]["matches_3dmolt5_table7_reported_sample_count"])
            self.assertEqual(artifact["checksums"]["sha256"], sha256_value)
            self.assertEqual(raw_artifact["checksums"]["sha256"], raw_sha256)
            self.assertNotIn("conformer_recipe", binding["full_p1_membership_blockers"])
            self.assertIn(
                "conformer_recipe",
                binding["downstream_training_blockers_not_full_p1_blockers"],
            )

        boundary = self.manifest["scientific_boundary"]["member_source_freeze_is_not_split_recovery"]
        self.assertIn("fallback evidence", boundary)

        authoritative_hiv = self.artifacts["deepchem_hiv_csv_authoritative_v1"]
        self.assertEqual(authoritative_hiv["count"]["record_count"], 41127)
        self.assertEqual(authoritative_hiv["bytes"], 2193844)
        self.assertEqual(
            authoritative_hiv["checksums"]["sha256"],
            "9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22",
        )
        self.assertEqual(
            authoritative_hiv["checksums"]["md5_etag"],
            "9ad10c88f82f1dac7eb5c52b668c30a7",
        )

    def test_kpgt_official_metadata_and_http_403_boundary_are_exact(self):
        github = self.sources["kpgt_official_github_revisions_v1"]
        self.assertEqual(github["origin"]["repository_id"], "lihan97/KPGT")
        self.assertEqual(
            github["revision"]["current_commit"],
            "47dc1646c70b2138a157de481d24a1ac35d174cd",
        )
        self.assertEqual(github["revision"]["paper_release_tag"], "v1.0.0")
        self.assertEqual(
            github["revision"]["paper_release_commit"],
            "390f29529dde268fed19203e7435307ae15dc083",
        )
        self.assertEqual(github["task_coverage"]["released_downstream_task_count"], 11)
        self.assertFalse(github["task_coverage"]["includes_hiv"])

        figshare = self.sources["kpgt_figshare_19914811_metadata_v1"]
        self.assertEqual(figshare["origin"]["doi"], "10.6084/m9.figshare.19914811")
        self.assertEqual(
            figshare["origin"]["share_url"],
            "https://figshare.com/s/8bbb8cad9ac644bf9caa",
        )
        self.assertEqual(figshare["origin"]["file_id"], "35391163")
        self.assertEqual(figshare["release_file_metadata"]["page_size_label"], "27.74MB")
        self.assertIsNone(figshare["release_file_metadata"]["payload_checksum"])
        self.assertEqual(figshare["access_boundary"]["observed_http_status"], 403)
        self.assertEqual(
            set(figshare["access_boundary"]["surfaces"]),
            {"figshare_page_or_share_link", "figshare_api", "figshare_ndownloader"},
        )
        self.assertIn("not evidence", figshare["access_boundary"]["consequence"])

    def test_kpgt_layout_candidate_files_hashes_counts_and_splits_are_exact(self):
        expected_csv = {
            "bace": (
                "kpgt_layout_candidate_bace_csv_v1",
                1513,
                "d7c3b46a688e87ec8489c8e4e0809139e0098ec69c7d3d527f297eedccece921",
                [1210, 151, 152],
            ),
            "bbbp": (
                "kpgt_layout_candidate_bbbp_csv_v1",
                2039,
                "a61e17872cbec0743a6147dc802a1f7550acfe8dc1053f881f4c42a3ce2dc597",
                [1631, 203, 205],
            ),
            "clintox": (
                "kpgt_layout_candidate_clintox_csv_v1",
                1478,
                "cc561402d058a0d996a9f5e5d02bc20f6652b7003525b32c3598b8da4b24e9b7",
                [1182, 147, 149],
            ),
        }
        expected_split_hashes = {
            "bace": [
                "22a697541418aeee65f130a9aa240e1fd98eb1dda99188f4b0d98500ae0ee9ed",
                "bae52768a4cc6926a072a42b0054bf35327e73cc12c40b8002ec09d5fc29e483",
                "7f529721508a0c17fc216c1076234c0fac1c07c94041e1232779796ed10909f1",
            ],
            "bbbp": [
                "f8b4ed48398e082a4a46702ee95daa9a1e19e90a51513f6804fccb6d5133dae4",
                "d3375a262d52b021e7e2511d2e0af4de691e3b3024fb559dccc107eb3996e2c8",
                "aa4cd63e7d77575f03615ae13746f0e9f5781aa07539f4b22b5c0da3afcbc6fa",
            ],
            "clintox": [
                "49642f93fad8ceb6351111342d6b90f4602e0780315f64949775d7f94a39641e",
                "6220bfe30d678b05c6bbd9db4e6281d99af2b4e93ad53a2c9a8ee78978f50796",
                "ee1e58979f36abd30934e9ccdbfc90f09eb831ec1ec1ecc72dc3c74595cd5612",
            ],
        }
        candidate = self.sources["kpgt_layout_candidate_legacy_r0_v1"]
        self.assertEqual(
            candidate["remote_storage_root"],
            "/root/autodl-fs/most-t5-r1/sources/legacy-and-downstream-evidence-r0-v1/moleculenet-datasets",
        )
        self.assertEqual(
            candidate["origin"]["layout_claim"],
            "kpgt_layout_compatible_candidate_only",
        )
        self.assertIn("not_verified", candidate["origin"]["canonical_origin_status"])

        for dataset, (csv_id, row_count, csv_hash, sizes) in expected_csv.items():
            csv_artifact = self.artifacts[csv_id]
            self.assertEqual(csv_artifact["remote_relative_path"], "{0}/{0}.csv".format(dataset))
            self.assertEqual(csv_artifact["count"]["record_count"], row_count)
            self.assertEqual(csv_artifact["checksums"]["sha256"], csv_hash)
            self.assertIn("not_verified_official", csv_artifact["scientific_role"])
            for split_index, split_hash in enumerate(expected_split_hashes[dataset]):
                split_id = "kpgt_layout_candidate_{}_scaffold_{}_v1".format(dataset, split_index)
                split_artifact = self.artifacts[split_id]
                self.assertEqual(
                    split_artifact["remote_relative_path"],
                    "{}/splits/scaffold-{}.npy".format(dataset, split_index),
                )
                self.assertEqual(split_artifact["checksums"]["sha256"], split_hash)
                self.assertEqual(
                    [
                        split_artifact["count"]["train_count"],
                        split_artifact["count"]["validation_count"],
                        split_artifact["count"]["test_count"],
                    ],
                    sizes,
                )
                self.assertEqual(split_artifact["count"]["total_count"], row_count)
                self.assertEqual(split_artifact["count"]["ratio_protocol"], "8:1:1")

        validation = candidate["candidate_validation"]
        self.assertEqual(validation["split_artifact_count"], 9)
        self.assertTrue(validation["each_split_membership_complete_coverage"])
        self.assertTrue(validation["within_each_split_train_validation_test_pairwise_disjoint"])
        self.assertTrue(validation["within_each_split_duplicate_free"])
        self.assertEqual(
            validation["within_each_split_murcko_scaffold_cross_partition_overlap"],
            {"include_chirality_false": 0, "include_chirality_true": 0},
        )
        self.assertTrue(validation["all_smiles_valid"])
        self.assertTrue(
            validation["label_validation"]["auroc_computable_for_every_dataset_split"]
        )
        self.assertIn("do not prove", validation["scientific_boundary"])

    def test_kpgt_candidate_bindings_and_hiv_separation_are_explicit(self):
        for dataset in ("bace", "bbbp", "clintox"):
            binding = self.manifest["task_bindings"]["moleculenet_{}_scientific_v2".format(dataset)]
            self.assertEqual(
                binding["primary_protocol_status"],
                "candidate_internally_valid_but_not_verified_official_due_figshare_http_403",
            )
            self.assertEqual(len(binding["kpgt_layout_candidate_split_artifact_ids"]), 3)
            self.assertIn(
                "official_source_comparison_or_authoritative_replacement",
                binding["full_p1_membership_blockers"],
            )
            self.assertNotIn("conformer_recipe", binding["full_p1_membership_blockers"])

        hiv = self.manifest["task_bindings"]["moleculenet_hiv_scientific_v2"]
        self.assertEqual(
            hiv["kpgt_applicability"],
            "not_applicable_kpgt_released_11_task_set_excludes_hiv",
        )
        self.assertEqual(
            hiv["primary_member_source_artifact_id"],
            "deepchem_hiv_csv_authoritative_v1",
        )
        self.assertEqual(
            hiv["primary_protocol_status"],
            "deepchem_authoritative_source_frozen_murcko_8_1_1_derived_v1_materialization_pending",
        )
        self.assertIn("derived_v1_formal_member_manifest", hiv["full_p1_membership_blockers"])
        self.assertNotIn("authoritative_hiv_source_revision", hiv["full_p1_membership_blockers"])
        self.assertIn("never described as an exact", self.manifest["scientific_boundary"]["hiv_boundary"])


if __name__ == "__main__":
    unittest.main()
