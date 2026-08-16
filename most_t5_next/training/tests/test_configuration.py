from copy import deepcopy
from pathlib import Path
import unittest

from most_t5_next.configuration import (
    ConfigurationError,
    apply_overrides,
    load_pretraining_config,
    validate_pretraining_config,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pretrain.yaml"


class ConfigurationTest(unittest.TestCase):
    def test_existing_public_parameters_can_be_overridden(self) -> None:
        original = load_pretraining_config(CONFIG_PATH)
        updated = apply_overrides(
            original,
            (
                "seed=7",
                "model.dropout_rate=0.1",
                "batching.micro_batch_size=4",
                "batching.gradient_accumulation_steps=32",
            ),
        )
        self.assertEqual(updated["seed"], 7)
        self.assertEqual(updated["model"]["dropout_rate"], 0.1)
        self.assertEqual(updated["batching"]["micro_batch_size"], 4)
        self.assertEqual(original["seed"], 42)

    def test_unknown_override_fails_instead_of_becoming_a_noop(self) -> None:
        config = load_pretraining_config(CONFIG_PATH)
        with self.assertRaisesRegex(ConfigurationError, "unknown configuration key"):
            apply_overrides(config, ("model.dropuot_rate=0.1",))

    def test_frozen_semantic_defaults_are_not_repository_style_options(self) -> None:
        config = load_pretraining_config(CONFIG_PATH)
        self.assertEqual(config["seed"], 42)
        self.assertNotIn("adapter_seed", config["model"])
        self.assertNotIn("max_identity_span_length", config["model"])
        self.assertNotIn("max_fragment_tokens", config["model"])
        self.assertEqual(config["optimization"]["warmup_start_factor"], 0.5)
        self.assertEqual(config["optimization"]["phase_one"]["base_learning_rate"], 2.0e-3)
        self.assertEqual(config["optimization"]["phase_two"]["base_learning_rate"], 1.0e-3)
        self.assertEqual(config["optimization"]["final_learning_rate"], 1.0e-5)
        self.assertEqual(config["curriculum"]["phase_one"]["tasks"], ["M", "MG"])
        self.assertEqual(config["curriculum"]["phase_one"]["total_updates"], 100_000)
        self.assertEqual(
            config["curriculum"]["phase_two"]["tasks"],
            ["SYN", "TXT", "CAP", "T2M"],
        )
        self.assertEqual(config["curriculum"]["phase_two"]["total_updates"], 200_000)
        self.assertTrue(config["curriculum"]["restart_optimizer_at_phase_two"])
        self.assertEqual(config["optimization"]["phase_one"]["warmup_updates"], 10_000)
        self.assertEqual(config["optimization"]["phase_two"]["warmup_updates"], 10_000)
        self.assertFalse(config["data"]["pretraining_validation_split"])
        self.assertEqual(config["data"]["txt"]["dataset"], "MedRAG/pubmed")
        self.assertEqual(config["data"]["txt"]["text_column"], "contents")
        self.assertTrue(config["data"]["txt"]["parquet_export_partial"])
        self.assertEqual(config["data"]["txt"]["training_shards"], ["train", "dev"])
        self.assertEqual(config["data"]["txt"]["pretraining_holdout_documents"], 0)
        self.assertFalse(config["monitoring"]["pretraining_evaluation"])
        self.assertEqual(config["monitoring"]["checkpoint_every_updates"], 10_000)
        self.assertNotIn("evaluate_every_updates", config["monitoring"])
        self.assertEqual(
            config["data"]["paired_text"]["text_column"],
            "enriched_description",
        )
        self.assertEqual(
            config["data"]["paired_text"]["downstream_reference_column"],
            "description",
        )
        self.assertEqual(config["corruption"]["molecule_sampling"], "heavy_atom_weighted")
        self.assertEqual(
            config["corruption"]["motif_unit"],
            "fragment_with_owned_explicit_endpoints",
        )
        self.assertEqual(
            config["batching"]["micro_batch_size"]
            * config["batching"]["gradient_accumulation_steps"],
            config["batching"]["effective_batch_size"],
        )
        self.assertEqual(config["batching"]["effective_batch_size"], 96)
        self.assertEqual(config["batching"]["global_effective_batch_size"], 384)
        self.assertEqual(
            config["distributed"]["rank_tasks"]["phase_one"],
            ["M", "M", "MG", "MG"],
        )
        self.assertEqual(
            config["distributed"]["rank_tasks"]["phase_two"],
            ["SYN", "TXT", "CAP", "T2M"],
        )
        self.assertEqual(
            config["distributed"]["loss_weighting"],
            "equal_rank_after_rank_local_token_normalization",
        )
        self.assertEqual(
            config["batching"]["task_partitions"]["phase_one"]["M"],
            {"micro_batch_size": 48, "gradient_accumulation_steps": 2},
        )
        self.assertEqual(
            config["batching"]["task_partitions"]["phase_one"]["MG"],
            {"micro_batch_size": 96, "gradient_accumulation_steps": 1},
        )
        self.assertEqual(
            config["batching"]["task_partitions"]["phase_two"]["SYN"],
            {"micro_batch_size": 48, "gradient_accumulation_steps": 2},
        )
        self.assertEqual(
            config["batching"]["task_partitions"]["phase_two"]["CAP"],
            {"micro_batch_size": 32, "gradient_accumulation_steps": 3},
        )
        self.assertEqual(
            config["batching"]["task_partitions"]["phase_two"]["T2M"],
            {"micro_batch_size": 32, "gradient_accumulation_steps": 3},
        )

    def test_distributed_layout_cannot_silently_change_global_batch(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["batching"]["global_effective_batch_size"] = 96
        with self.assertRaisesRegex(ConfigurationError, "global batch"):
            validate_pretraining_config(config)

        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["distributed"]["rank_tasks"]["phase_two"] = [
            "SYN", "SYN", "CAP", "T2M"
        ]
        with self.assertRaisesRegex(ConfigurationError, "balance"):
            validate_pretraining_config(config)

    def test_public_config_is_launch_complete(self) -> None:
        config = load_pretraining_config(CONFIG_PATH)
        validate_pretraining_config(config, require_launch_values=True)

    def test_phase_local_launch_values_are_global_ddp_update_budgets(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        for phase_name, updates in (("phase_one", 20_001), ("phase_two", 40_003)):
            config["curriculum"][phase_name]["total_updates"] = updates
            config["optimization"][phase_name]["base_learning_rate"] = 1.0e-3
        config["optimization"]["warmup_start_factor"] = 0.5
        config["optimization"]["final_learning_rate"] = 1.0e-5
        config["monitoring"]["checkpoint_every_updates"] = 10_000
        validate_pretraining_config(config, require_launch_values=True)

    def test_checkpoint_interval_is_open_but_must_be_positive(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["monitoring"]["checkpoint_every_updates"] = 0
        with self.assertRaisesRegex(ConfigurationError, "must be positive"):
            validate_pretraining_config(config)

    def test_phase_warmup_cannot_drift_from_10000_updates(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["optimization"]["phase_two"]["warmup_updates"] = 1_000
        with self.assertRaisesRegex(ConfigurationError, "must remain 10000"):
            validate_pretraining_config(config)

    def test_txt_source_cannot_silently_revert_to_c4(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["data"]["txt"]["dataset"] = "c4"
        with self.assertRaisesRegex(ConfigurationError, "TXT dataset has drifted"):
            validate_pretraining_config(config)

    def test_seed_is_open_but_must_be_reproducible(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["seed"] = 7
        validate_pretraining_config(config)
        config["seed"] = -1
        with self.assertRaisesRegex(ConfigurationError, "seed"):
            validate_pretraining_config(config)

    def test_pretraining_cannot_reserve_a_dev_population(self) -> None:
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["data"]["pretraining_validation_split"] = True
        with self.assertRaisesRegex(ConfigurationError, "every admitted task record"):
            validate_pretraining_config(config)

        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["data"]["txt"]["training_shards"] = ["train"]
        config["data"]["txt"]["pretraining_holdout_documents"] = 2_389_870
        with self.assertRaisesRegex(ConfigurationError, "all physical TXT shards"):
            validate_pretraining_config(config)

        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        config["monitoring"]["pretraining_evaluation"] = True
        with self.assertRaisesRegex(ConfigurationError, "must not run an evaluation"):
            validate_pretraining_config(config)


if __name__ == "__main__":
    unittest.main()
