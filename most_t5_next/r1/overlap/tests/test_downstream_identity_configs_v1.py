from __future__ import print_function

import json
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import extract_identity_collection_v1 as extractor


R1_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = R1_ROOT / "contracts" / "identity_collection_extraction_contract_v1.json"
CONFIG_ROOT = R1_ROOT / "overlap" / "configs"
REGISTRY = CONFIG_ROOT / "downstream_task_registry_20260806_v1.json"

CONFIGS = {
    ("qm9", "train"): CONFIG_ROOT / "downstream_qm9_train_connectivity_diagnostic_identity_config_20260806_v1.json",
    ("qm9", "validation"): CONFIG_ROOT / "downstream_qm9_validation_connectivity_diagnostic_identity_config_20260806_v1.json",
    ("qm9", "test"): CONFIG_ROOT / "downstream_qm9_test_connectivity_diagnostic_identity_config_20260806_v1.json",
    ("chebi", "train"): CONFIG_ROOT / "downstream_chebi20_train_identity_config_20260806_v1.json",
    ("chebi", "validation"): CONFIG_ROOT / "downstream_chebi20_validation_identity_config_20260806_v1.json",
    ("chebi", "test"): CONFIG_ROOT / "downstream_chebi20_test_identity_config_20260806_v1.json",
    ("caption", "train"): CONFIG_ROOT / "downstream_pubchem_caption_train_identity_config_20260806_v1.json",
    ("caption", "validation"): CONFIG_ROOT / "downstream_pubchem_caption_validation_identity_config_20260806_v1.json",
    ("caption", "test"): CONFIG_ROOT / "downstream_pubchem_caption_test_identity_config_20260806_v1.json",
}

SOURCE_ROOTS = {
    "qm9": Path("/root/autodl-fs/most-t5-r1/sources/legacy-and-downstream-evidence-r0-v1"),
    "chebi": Path("/root/autodl-fs/most-t5-r1/sources/legacy-and-downstream-evidence-r0-v1"),
    "caption": Path("/root/autodl-fs/most-t5-r1/sources/p2-pubchem-evidence-r0-v1/pubchem"),
}

TASK_IDS = {
    "qm9": "qm9_orbital_property_generation_3dmolt5_hf_v1",
    "chebi": "chebi20_text_to_molecule_3dmolt5_hf_v1",
    "caption": "pubchem_captioning_p2_lmdb_v1",
}

EXPECTED_ROLES = {
    "train": "downstream_train",
    "validation": "downstream_validation",
    "test": "downstream_test",
}


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


class DownstreamIdentityConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_sha256 = extractor.sha256_file(CONTRACT)[1]
        cls.configs = {key: load_json(path) for key, path in CONFIGS.items()}
        registry = load_json(REGISTRY)
        cls.registry_tasks = {task["task_id"]: task for task in registry["tasks"]}

    def test_all_nine_configs_pass_the_frozen_extractor_contract(self):
        for key, config in self.configs.items():
            with self.subTest(config=key):
                extractor.validate_config(config, self.contract_sha256)

    def test_split_roles_and_identifiers_are_unique_and_explicit(self):
        extraction_ids = set()
        collection_ids = set()
        for (family, split), config in self.configs.items():
            with self.subTest(family=family, split=split):
                collection = config["collection"]
                self.assertEqual(collection["phase"], "downstream")
                self.assertEqual(collection["split"], split)
                self.assertEqual(collection["role"], EXPECTED_ROLES[split])
                self.assertNotIn(config["extraction_id"], extraction_ids)
                self.assertNotIn(collection["collection_id"], collection_ids)
                extraction_ids.add(config["extraction_id"])
                collection_ids.add(collection["collection_id"])

    def test_locked_sources_match_the_downstream_registry(self):
        for (family, split), config in self.configs.items():
            with self.subTest(family=family, split=split):
                task = self.registry_tasks[TASK_IDS[family]]
                registry_source = task["splits"][split]["source"]
                expected_path = SOURCE_ROOTS[family] / registry_source["relative_path"]
                self.assertEqual(Path(config["source"]["path"]), expected_path)
                self.assertEqual(config["source"]["format"], registry_source["format"])
                self.assertEqual(config["source"]["expected_bytes"], registry_source["bytes"])
                self.assertEqual(config["source"]["expected_sha256"], registry_source["sha256"])
                self.assertEqual(config["collection"]["dataset_id"], task["dataset_id"])
                self.assertEqual(config["collection"]["task_family"], task["task_family"])

    def test_chebi_mapping_freezes_the_two_component_encoder_input(self):
        expected_components = [
            {"name": "instruction", "field": "instruction"},
            {"name": "description", "field": "input"},
        ]
        for split in ("train", "validation", "test"):
            config = self.configs[("chebi", split)]
            member = config["mapping"]["member_id"]
            text = config["mapping"]["text_identity"]
            self.assertEqual(
                member,
                {"source": "field", "field": "cid", "prefix": "chebi_cid:", "crosscheck_field": None},
            )
            self.assertEqual(config["mapping"]["smiles_field"], "smiles")
            self.assertEqual(text["normalization"], "unicode_nfkc_whitespace_v1")
            self.assertEqual(text["unit"]["unit_name"], "text_to_molecule_encoder_input")
            self.assertEqual(text["unit"]["semantic_role"], "text_to_molecule_encoder_input")
            self.assertEqual(text["unit"]["components"], expected_components)

    def test_qm9_configs_are_connectivity_diagnostics_without_text_routing(self):
        task = self.registry_tasks[TASK_IDS["qm9"]]
        self.assertEqual(task["task_status"], "blocked_source_split_integrity")
        for split in ("train", "validation", "test"):
            config = self.configs[("qm9", split)]
            member = config["mapping"]["member_id"]
            self.assertEqual(config["collection"]["release_id"], task["source_revision"])
            self.assertIn("connectivity-diagnostic", config["extraction_id"])
            self.assertIn("connectivity-diagnostic", config["collection"]["collection_id"])
            self.assertEqual(
                member,
                {
                    "source": "row_index",
                    "field": None,
                    "prefix": "qm9-hf:{}:row:".format(split),
                    "crosscheck_field": None,
                },
            )
            self.assertEqual(config["mapping"]["smiles_field"], "smiles")
            self.assertEqual(
                config["mapping"]["text_identity"],
                {"status": "unavailable", "normalization": None, "unit": None},
            )

    def test_qm9_byte_identical_eval_sources_remain_distinct_diagnostic_addresses_only(self):
        validation = self.configs[("qm9", "validation")]
        test = self.configs[("qm9", "test")]
        task = self.registry_tasks[TASK_IDS["qm9"]]
        self.assertTrue(task["preliminary_split_observations"]["byte_identical_validation_and_test"])
        self.assertEqual(validation["source"]["expected_bytes"], test["source"]["expected_bytes"])
        self.assertEqual(validation["source"]["expected_sha256"], test["source"]["expected_sha256"])
        self.assertNotEqual(validation["collection"]["collection_id"], test["collection"]["collection_id"])
        self.assertNotEqual(
            validation["mapping"]["member_id"]["prefix"],
            test["mapping"]["member_id"]["prefix"],
        )
        self.assertIn("diagnostic", validation["collection"]["collection_id"])
        self.assertIn("diagnostic", test["collection"]["collection_id"])
        self.assertNotEqual(task["task_status"], "ready")

    def test_caption_compatibility_target_is_legacy_description_only(self):
        for split in ("train", "validation", "test"):
            config = self.configs[("caption", split)]
            member = config["mapping"]["member_id"]
            text = config["mapping"]["text_identity"]
            self.assertEqual(
                member,
                {"source": "lmdb_key", "field": None, "prefix": "pubchem_cid:", "crosscheck_field": "cid"},
            )
            self.assertEqual(config["mapping"]["smiles_field"], "smiles")
            self.assertEqual(text["normalization"], "unicode_nfkc_whitespace_v1")
            self.assertEqual(text["unit"]["semantic_role"], "molecule_caption_decoder_target")
            self.assertEqual(text["unit"]["components"], [{"name": "target", "field": "description"}])
            self.assertNotIn("enriched_description", json.dumps(text, sort_keys=True))

    def test_lmdb_pickle_acknowledgement_is_bound_per_caption_split(self):
        for split in ("train", "validation", "test"):
            source = self.configs[("caption", split)]["source"]
            lmdb = source["format_options"]["lmdb"]
            self.assertFalse(lmdb["subdir"])
            self.assertEqual(lmdb["metadata_keys_permitted"], ["__len__"])
            self.assertEqual(lmdb["metadata_keys_required"], [])
            self.assertEqual(lmdb["trusted_pickle_source_sha256"], source["expected_sha256"])


if __name__ == "__main__":
    unittest.main()
