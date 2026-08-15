"""Loading and scientific-contract checks for MoSt-T5 pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from most_t5_next.interfaces import PHASE_TASKS


class ConfigurationError(ValueError):
    pass


def apply_overrides(
    config: Mapping[str, Any], overrides: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Return a copy with explicit ``section.key=value`` overrides.

    Only existing keys may be changed.  This keeps the public command line
    compact while exposing the same ordinary research parameters as the YAML
    configuration, without silently accepting misspelled options.
    """

    import copy
    import yaml

    result = copy.deepcopy(dict(config))
    for assignment in overrides:
        if "=" not in assignment:
            raise ConfigurationError("configuration override must contain =")
        dotted_key, raw_value = assignment.split("=", 1)
        parts = dotted_key.split(".")
        if not parts or any(not part for part in parts):
            raise ConfigurationError("configuration override has an invalid key")
        target: Any = result
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ConfigurationError(f"unknown configuration key: {dotted_key}")
            target = target[part]
        leaf = parts[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise ConfigurationError(f"unknown configuration key: {dotted_key}")
        target[leaf] = yaml.safe_load(raw_value)
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def validate_pretraining_config(
    config: Mapping[str, Any], *, require_launch_values: bool = False
) -> None:
    """Check parameters that determine the frozen experimental protocol."""

    seed = config["seed"]
    _require(
        not isinstance(seed, bool) and isinstance(seed, int) and seed >= 0,
        "seed must be a nonnegative integer",
    )

    model = config["model"]
    for name in (
        "vocabulary_size",
        "input_length",
        "target_length",
        "e3fp_bits",
        "e3fp_embedding_dim",
    ):
        _require(int(model[name]) > 0, f"model.{name} must be positive")
    _require(0.0 <= float(model["dropout_rate"]) < 1.0, "invalid dropout")
    _require(0.0 < float(model["geometry_fraction"]) < 1.0, "invalid geometry fraction")

    corruption = config["corruption"]
    _require(0.0 < float(corruption["noise_density"]) < 1.0, "invalid noise density")
    _require(float(corruption["mean_span_length"]) > 0.0, "invalid span length")
    _require(
        corruption["molecule_sampling"] == "heavy_atom_weighted",
        "molecular sampling policy has drifted",
    )
    _require(
        corruption["motif_unit"] == "fragment_with_owned_explicit_endpoints",
        "compound motif policy has drifted",
    )

    data = config["data"]
    _require(
        data["pretraining_validation_split"] is False,
        "formal pretraining must use every admitted task record",
    )
    paired_text = data["paired_text"]
    _require(
        paired_text["source"] == "3D-MoLM/PubChem",
        "paired-text source has drifted",
    )
    _require(
        paired_text["text_column"] == "enriched_description",
        "paired-text pretraining field has drifted",
    )
    _require(
        paired_text["downstream_reference_column"] == "description",
        "paired-text downstream reference has drifted",
    )

    text_source = data["txt"]
    _require(text_source["dataset"] == "MedRAG/pubmed", "TXT dataset has drifted")
    _require(
        text_source["revision"] == "33da3593d5756bc04c8909f170003c0b14197957",
        "TXT dataset revision has drifted",
    )
    _require(
        text_source["parquet_ref"] == "refs/convert/parquet",
        "TXT Parquet ref has drifted",
    )
    _require(text_source["text_column"] == "contents", "TXT text column has drifted")
    _require(
        text_source["parquet_export_partial"] is True,
        "TXT Parquet coverage declaration has drifted",
    )
    _require(
        tuple(text_source["training_shards"]) == ("train", "dev"),
        "all physical TXT shards must remain in the training population",
    )
    _require(
        int(text_source["pretraining_holdout_documents"]) == 0,
        "pretraining must not reserve a semantic holdout",
    )

    curriculum = config["curriculum"]
    _require(
        curriculum["restart_optimizer_at_phase_two"] is True,
        "Phase II must restart optimizer and schedule state",
    )
    phase_updates: dict[str, int | None] = {}
    for phase_name, phase_number in (("phase_one", 1), ("phase_two", 2)):
        phase = curriculum[phase_name]
        tasks = tuple(phase["tasks"])
        _require(tasks == PHASE_TASKS[phase_number], f"{phase_name} tasks have drifted")
        expected_ratio = 1.0 / len(tasks)
        _require(
            len(phase["ratios"]) == len(tasks)
            and all(abs(float(ratio) - expected_ratio) < 1.0e-12 for ratio in phase["ratios"]),
            f"{phase_name} must use balanced update ratios",
        )
        updates = phase["total_updates"]
        if updates is None:
            _require(not require_launch_values, f"{phase_name}.total_updates is unresolved")
        else:
            updates = int(updates)
            _require(updates > 0 and updates % len(tasks) == 0, f"{phase_name} budget is invalid")
        phase_updates[phase_name] = updates

    optimization = config["optimization"]
    _require(optimization["name"] == "adamwscale", "optimizer has drifted")
    _require(
        optimization["scheduler"] == "linear_warmup_cosine",
        "scheduler has drifted",
    )
    _require(
        optimization["precision"] in {"fp32", "bf16", "fp16"},
        "precision must be fp32, bf16, or fp16",
    )
    _require(0.0 <= float(optimization["weight_decay"]), "invalid weight decay")
    _require(float(optimization["gradient_clip_norm"]) > 0.0, "invalid gradient clip")
    _require(0.0 <= float(optimization["beta1"]) < 1.0, "invalid beta1")
    _require(0.0 <= float(optimization["beta2"]) < 1.0, "invalid beta2")
    _require(float(optimization["epsilon"]) > 0.0, "invalid epsilon")
    warmup_start_factor = optimization["warmup_start_factor"]
    if warmup_start_factor is None:
        _require(not require_launch_values, "optimization.warmup_start_factor is unresolved")
    else:
        _require(
            0.0 < float(warmup_start_factor) <= 1.0,
            "invalid warmup start factor",
        )
    final_learning_rate = optimization["final_learning_rate"]
    if final_learning_rate is None:
        _require(not require_launch_values, "optimization.final_learning_rate is unresolved")
    else:
        _require(float(final_learning_rate) >= 0.0, "invalid final learning rate")
    for phase_name in ("phase_one", "phase_two"):
        phase = optimization[phase_name]
        warmup = int(phase["warmup_updates"])
        _require(
            warmup == 10_000,
            f"optimization.{phase_name}.warmup_updates must remain 10000",
        )
        updates = phase_updates[phase_name]
        _require(updates is None or warmup < updates, f"{phase_name} warmup exceeds its budget")
        if phase["base_learning_rate"] is None:
            _require(not require_launch_values, f"optimization.{phase_name} is unresolved")
            continue
        learning_rate = float(phase["base_learning_rate"])
        _require(learning_rate > 0.0, f"optimization.{phase_name} is invalid")
        if final_learning_rate is not None:
            _require(
                float(final_learning_rate) < learning_rate,
                f"{phase_name} final learning rate must be below its base rate",
            )

    batching = config["batching"]
    micro_batch_size = int(batching["micro_batch_size"])
    accumulation_steps = int(batching["gradient_accumulation_steps"])
    _require(micro_batch_size > 0, "micro batch size must be positive")
    _require(accumulation_steps > 0, "gradient accumulation must be positive")
    _require(
        micro_batch_size * accumulation_steps == int(batching["effective_batch_size"]),
        "batching values do not produce effective_batch_size",
    )
    dataloader = config["dataloader"]
    _require(int(dataloader["num_workers"]) >= 0, "num_workers must be nonnegative")
    _require(int(dataloader["prefetch_factor"]) > 0, "prefetch_factor must be positive")

    monitoring = config["monitoring"]
    _require(int(monitoring["log_every_updates"]) > 0, "log interval must be positive")
    _require(
        monitoring["pretraining_evaluation"] is False,
        "formal pretraining must not run an evaluation loop",
    )
    checkpoint_interval = monitoring["checkpoint_every_updates"]
    if checkpoint_interval is None:
        _require(not require_launch_values, "monitoring.checkpoint_every_updates is unresolved")
    else:
        _require(int(checkpoint_interval) > 0, "checkpoint interval must be positive")


def load_pretraining_config(
    path: str | Path,
    *,
    overrides: list[str] | tuple[str, ...] = (),
    require_launch_values: bool = False,
) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = apply_overrides(config, overrides)
    validate_pretraining_config(config, require_launch_values=require_launch_values)
    return config


__all__ = [
    "ConfigurationError",
    "apply_overrides",
    "load_pretraining_config",
    "validate_pretraining_config",
]
