#!/usr/bin/env python3
"""Build the one shared raw-T5 checkpoint for the P1 four-cell canary.

The base T5 checkpoint and its tokenizer do not necessarily have the same
vocabulary dimension.  In the production input, for example, the tokenizer
has 32,100 addressable IDs while the model has 32,128 embedding rows.  Once
the union tokenizer assigns meanings to IDs at and above 32,100, *all* of
those rows must be initialized as new vocabulary rows.  Merely relying on
``resize_token_embeddings`` would retain the previously unreachable
32,100--32,127 checkpoint rows and would therefore make their initialization
history differ from later union rows.

This module consequently performs one exact resize on a raw
``T5ForConditionalGeneration`` before any wrapper, optimizer, or distributed
training object exists.  It then replaces the half-open row interval
``[base_tokenizer_size, union_vocab_size)`` using T5's own shared-embedding
and untied-LM-head initialization law: Normal(0, initializer_factor).  The
two untied matrices use separate, private CPU generators; tied models retain
their tie and are initialized once.  No semantic or data-dependent
initialization is attempted.  This module makes no *explicit* call to
``tie_weights``; Transformers 4.45.2's ``resize_token_embeddings`` API itself
calls ``self.tie_weights()`` internally after resizing, and the resulting tie
state is verified against the unchanged T5 configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


SCHEMA_VERSION = "most-t5-p1/union-init-checkpoint/v1"
MANIFEST_NAME = "init_manifest.json"
CHECKPOINT_DIRECTORY = "shared_raw_t5"
STAGING_SUFFIX = ".staging"
INPUT_STREAM_OFFSET = 0
OUTPUT_STREAM_OFFSET = 1
T5_VOCAB_STATE_KEYS = frozenset(
    {
        "shared.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    }
)


class UnionInitCheckpointError(RuntimeError):
    """The shared raw-T5 initialization checkpoint is not reproducible."""


@dataclass(frozen=True)
class VerifiedUnionInitCheckpoint:
    """A model/tokenizer pair admitted only after semantic replay checks."""

    model: Any
    tokenizer_build: Any
    manifest: dict[str, object]
    checkpoint_path: Path


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime boundary
        raise UnionInitCheckpointError("PyTorch is required to build the checkpoint") from exc
    return torch


def _load_verified_tokenizer(
    *, base_tokenizer_snapshot: Path, union_tokenizer_dir: Path
) -> Any:
    from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
        load_verified_canary_union_tokenizer,
    )

    return load_verified_canary_union_tokenizer(
        base_snapshot=base_tokenizer_snapshot,
        output_dir=union_tokenizer_dir,
    )


def _load_raw_t5_from_pretrained(path: Path) -> Any:
    """Load raw T5 on CPU without consuming the caller's CPU RNG stream."""

    torch = _load_torch()
    try:
        from transformers import T5ForConditionalGeneration
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime boundary
        raise UnionInitCheckpointError("Transformers is required to load raw T5") from exc

    rng_state = torch.random.get_rng_state()
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            str(path),
            local_files_only=True,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise UnionInitCheckpointError(
            "raw T5 checkpoint could not be loaded offline: {}".format(path)
        ) from exc
    finally:
        torch.random.set_rng_state(rng_state)

    if not isinstance(model, T5ForConditionalGeneration):
        raise UnionInitCheckpointError("checkpoint did not load as raw T5ForConditionalGeneration")
    if getattr(model.config, "model_type", None) != "t5":
        raise UnionInitCheckpointError("checkpoint config is not T5")
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise UnionInitCheckpointError("union initialization must run on CPU before wrapping")
    return model


def _require_new_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise UnionInitCheckpointError("{} must be a new path: {}".format(label, resolved))
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise UnionInitCheckpointError("{} must be an existing directory".format(label))
    return resolved


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise UnionInitCheckpointError("union-init manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise UnionInitCheckpointError("union-init manifest must be a JSON object")
    return value


def _write_manifest_new(path: Path, value: Mapping[str, object]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise UnionInitCheckpointError("union-init manifest could not be written") from exc


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnionInitCheckpointError("{} must be a positive integer".format(label))
    return int(value)


def _tokenizer_dimensions(verified: Any) -> tuple[int, int, str, str]:
    manifest = getattr(verified, "manifest", None)
    runtime = getattr(verified, "runtime", None)
    tokenizer = getattr(verified, "tokenizer", None)
    if not isinstance(manifest, Mapping) or runtime is None or tokenizer is None:
        raise UnionInitCheckpointError("verified tokenizer result is incomplete")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise UnionInitCheckpointError("verified tokenizer manifest lacks counts")
    base_size = _positive_int(counts.get("base_vocab_size"), "base tokenizer size")
    union_size = _positive_int(counts.get("final_vocab_size"), "union tokenizer size")
    runtime_size = _positive_int(getattr(runtime, "vocab_size", None), "runtime vocabulary size")
    try:
        observed_size = len(tokenizer)
    except (TypeError, AttributeError) as exc:
        raise UnionInitCheckpointError("verified tokenizer has no finite length") from exc
    if union_size != runtime_size or union_size != observed_size:
        raise UnionInitCheckpointError("verified tokenizer dimensions disagree")
    if union_size <= base_size:
        raise UnionInitCheckpointError("union vocabulary must extend the base tokenizer")
    contract_hash = manifest.get("tokenizer_contract_sha256")
    snapshot_hash = manifest.get("tokenizer_snapshot_sha256")
    if not isinstance(contract_hash, str) or not contract_hash:
        raise UnionInitCheckpointError("verified tokenizer contract identity is absent")
    if not isinstance(snapshot_hash, str) or not snapshot_hash:
        raise UnionInitCheckpointError("verified tokenizer snapshot identity is absent")
    return base_size, union_size, contract_hash, snapshot_hash


def _weight_pair(model: Any) -> tuple[Any, Any]:
    try:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
    except (AttributeError, TypeError) as exc:
        raise UnionInitCheckpointError("raw T5 does not expose input/output weights") from exc
    if input_weight.ndim != 2 or output_weight.ndim != 2:
        raise UnionInitCheckpointError("T5 vocabulary weights must be rank two")
    if tuple(input_weight.shape) != tuple(output_weight.shape):
        raise UnionInitCheckpointError("T5 input and output vocabulary shapes differ")
    return input_weight, output_weight


def _weights_are_tied(input_weight: Any, output_weight: Any) -> bool:
    return (
        input_weight is output_weight
        or (
            input_weight.device == output_weight.device
            and input_weight.data_ptr() == output_weight.data_ptr()
            and input_weight.storage_offset() == output_weight.storage_offset()
        )
    )


def _inspect_base_model(model: Any, base_tokenizer_size: int, union_size: int) -> dict[str, object]:
    torch = _load_torch()
    input_weight, output_weight = _weight_pair(model)
    checkpoint_vocab, hidden_size = (int(value) for value in input_weight.shape)
    config_vocab = getattr(model.config, "vocab_size", None)
    if config_vocab != checkpoint_vocab:
        raise UnionInitCheckpointError("base config and vocabulary weights disagree")
    if base_tokenizer_size > checkpoint_vocab:
        raise UnionInitCheckpointError("base tokenizer is larger than the base T5 model")
    if union_size < checkpoint_vocab:
        raise UnionInitCheckpointError(
            "union vocabulary must not shrink the base T5 checkpoint"
        )
    config_hidden = getattr(model.config, "d_model", None)
    if config_hidden != hidden_size:
        raise UnionInitCheckpointError("base config.d_model differs from vocabulary width")
    config_tied = getattr(model.config, "tie_word_embeddings", None)
    if not isinstance(config_tied, bool):
        raise UnionInitCheckpointError("base config must declare tie_word_embeddings")
    observed_tied = _weights_are_tied(input_weight, output_weight)
    if observed_tied != config_tied:
        raise UnionInitCheckpointError("base T5 tie state differs from its config")
    if not input_weight.dtype.is_floating_point or not output_weight.dtype.is_floating_point:
        raise UnionInitCheckpointError("T5 vocabulary weights must be floating point")
    factor = getattr(model.config, "initializer_factor", None)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise UnionInitCheckpointError("T5 config.initializer_factor is invalid")
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise UnionInitCheckpointError("T5 initializer_factor must be finite and positive")
    input_dtype_name = str(input_weight.dtype)
    output_dtype_name = str(output_weight.dtype)
    if input_dtype_name.startswith("torch."):
        input_dtype_name = input_dtype_name[len("torch.") :]
    if output_dtype_name.startswith("torch."):
        output_dtype_name = output_dtype_name[len("torch.") :]
    return {
        "checkpoint_vocab_size": checkpoint_vocab,
        "hidden_size": hidden_size,
        "tie_word_embeddings": config_tied,
        "input_dtype": input_dtype_name,
        "output_dtype": output_dtype_name,
        "initializer_factor": factor,
        "model_type": str(model.config.model_type),
    }


def _validate_seed(seed: object, label: str = "seed") -> int:
    # The second private stream is seed + 1 for an untied LM head.
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2**63 - 1):
        raise UnionInitCheckpointError(
            "{} must be an integer in [0, 2**63 - 2]".format(label)
        )
    return int(seed)


def _validate_num_e3fp_embeddings(value: object) -> int:
    return _positive_int(value, "num_e3fp_embeddings")


def _t5_initialized_rows(
    *, rows: int, width: int, dtype: Any, seed: int, initializer_factor: float
) -> Any:
    """Draw rows by the T5 vocabulary law using a private CPU RNG stream."""

    torch = _load_torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # T5 checkpoints are conventionally initialized in fp32.  Drawing in fp32
    # and casting makes the stream explicit for lower-precision checkpoints.
    values = torch.empty((rows, width), dtype=torch.float32, device="cpu")
    values.normal_(mean=0.0, std=initializer_factor, generator=generator)
    return values.to(dtype=dtype)


def _resize_and_initialize(
    model: Any,
    *,
    base_tokenizer_size: int,
    union_size: int,
    seed: int,
    base_info: Mapping[str, object],
) -> None:
    """Perform the sole exact resize and deterministic vocabulary reset."""

    torch = _load_torch()
    input_before, output_before = _weight_pair(model)
    old_input = input_before[:base_tokenizer_size].detach().clone()
    old_output = output_before[:base_tokenizer_size].detach().clone()
    tied = bool(base_info["tie_word_embeddings"])

    # HF initializes newly allocated rows and calls ``self.tie_weights()``
    # internally as part of this public resize API.  We make no separate
    # tie_weights call; after the API returns, its resulting tie state is
    # checked against the unchanged config.  Preserve the external RNG stream
    # because every row receiving new tokenizer semantics is overwritten below
    # from our private generators.
    rng_state = torch.random.get_rng_state()
    try:
        model.resize_token_embeddings(union_size)
    finally:
        torch.random.set_rng_state(rng_state)

    input_weight, output_weight = _weight_pair(model)
    if tuple(input_weight.shape) != (union_size, int(base_info["hidden_size"])):
        raise UnionInitCheckpointError("resize did not produce the exact union dimension")
    if getattr(model.config, "vocab_size", None) != union_size:
        raise UnionInitCheckpointError("resize did not update config.vocab_size exactly")
    if _weights_are_tied(input_weight, output_weight) != tied:
        raise UnionInitCheckpointError("resize changed the T5 tie state")

    row_count = union_size - base_tokenizer_size
    input_rows = _t5_initialized_rows(
        rows=row_count,
        width=int(base_info["hidden_size"]),
        dtype=input_weight.dtype,
        seed=seed + INPUT_STREAM_OFFSET,
        initializer_factor=float(base_info["initializer_factor"]),
    )
    with torch.no_grad():
        input_weight[base_tokenizer_size:union_size].copy_(input_rows)
        if not tied:
            output_rows = _t5_initialized_rows(
                rows=row_count,
                width=int(base_info["hidden_size"]),
                dtype=output_weight.dtype,
                seed=seed + OUTPUT_STREAM_OFFSET,
                initializer_factor=float(base_info["initializer_factor"]),
            )
            output_weight[base_tokenizer_size:union_size].copy_(output_rows)

    if not torch.equal(input_weight[:base_tokenizer_size], old_input):
        raise UnionInitCheckpointError("resize changed an addressable base input row")
    if not torch.equal(output_weight[:base_tokenizer_size], old_output):
        raise UnionInitCheckpointError("resize changed an addressable base output row")
    if not torch.isfinite(input_weight[base_tokenizer_size:]).all().item():
        raise UnionInitCheckpointError("initialized input vocabulary rows are non-finite")
    if not torch.isfinite(output_weight[base_tokenizer_size:]).all().item():
        raise UnionInitCheckpointError("initialized output vocabulary rows are non-finite")


def _expected_manifest(
    *,
    base_info: Mapping[str, object],
    base_tokenizer_size: int,
    union_size: int,
    contract_hash: str,
    snapshot_hash: str,
    seed: int,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
) -> dict[str, object]:
    checkpoint_vocab = int(base_info["checkpoint_vocab_size"])
    tied = bool(base_info["tie_word_embeddings"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "one_shared_raw_t5_for_A0_A1_M0_M1",
        "construction_order": [
            "load_raw_base_t5_on_cpu",
            "resize_exactly_to_union_tokenizer_length",
            "initialize_all_newly_addressable_rows",
            "save_and_offline_reload",
            "construct_wrappers_optimizers_or_ddp_only_after_this_checkpoint",
        ],
        "tokenizer": {
            "base_vocab_size": base_tokenizer_size,
            "union_vocab_size": union_size,
            "tokenizer_contract_sha256": contract_hash,
            "tokenizer_snapshot_sha256": snapshot_hash,
        },
        "base_model": dict(base_info),
        "initialization": {
            "seed": seed,
            "input_stream_seed": seed + INPUT_STREAM_OFFSET,
            "output_stream_seed": None if tied else seed + OUTPUT_STREAM_OFFSET,
            "distribution": "normal",
            "mean": 0.0,
            "std": float(base_info["initializer_factor"]),
            "law": "T5 shared embedding and untied lm_head initialization",
            "preserved_id_range_half_open": [0, base_tokenizer_size],
            "initialized_id_range_half_open": [base_tokenizer_size, union_size],
            "reclaimed_checkpoint_id_range_half_open": [
                base_tokenizer_size,
                checkpoint_vocab,
            ],
            "newly_allocated_id_range_half_open": [checkpoint_vocab, union_size],
            "input_and_output_tied": tied,
            "input_initialized_once_when_tied": tied,
            "semantic_initialization": False,
        },
        "resize": {
            "requested_vocab_size": union_size,
            "pad_to_multiple_of": None,
            "explicit_tie_weights_call": False,
            "hf_resize_api_internally_calls_tie_weights": True,
            "post_resize_tie_state_verified_against_config": True,
        },
        "four_grid_wrapper": {
            "geometry_fusion_seed": geometry_fusion_seed,
            "num_e3fp_embeddings": num_e3fp_embeddings,
            "geometry_parameter_initialization": (
                "native FourGridT5Wrapper construction under a private CPU RNG state"
            ),
            "condition_id_is_non_parameter_metadata": True,
            "one_raw_checkpoint_shared_by_all_conditions": True,
            "global_cpu_rng_restored_after_construction": True,
        },
        "checkpoint": {
            "directory": CHECKPOINT_DIRECTORY,
            "raw_model_class": "T5ForConditionalGeneration",
            "offline_reload_verified": True,
            "old_rows_bitwise_preserved": True,
            "non_vocabulary_state_bitwise_preserved": True,
            "initialized_rows_replayed_from_private_seeds": True,
            "unexpected_input_output_alias": False,
        },
    }


def _validate_manifest_shape(manifest: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "status",
        "scope",
        "construction_order",
        "tokenizer",
        "base_model",
        "initialization",
        "resize",
        "four_grid_wrapper",
        "checkpoint",
    }
    if set(manifest) != required:
        raise UnionInitCheckpointError("union-init manifest fields changed")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise UnionInitCheckpointError("union-init manifest state is invalid")


def _verify_model_pair(
    *,
    base_model: Any,
    union_model: Any,
    expected_manifest: Mapping[str, object],
) -> None:
    """Replay every changed vocabulary row and preserve all other state."""

    torch = _load_torch()
    tokenizer_info = expected_manifest["tokenizer"]
    initialization = expected_manifest["initialization"]
    base_info = expected_manifest["base_model"]
    assert isinstance(tokenizer_info, Mapping)
    assert isinstance(initialization, Mapping)
    assert isinstance(base_info, Mapping)
    base_size = int(tokenizer_info["base_vocab_size"])
    union_size = int(tokenizer_info["union_vocab_size"])
    seed = int(initialization["seed"])
    tied = bool(base_info["tie_word_embeddings"])

    observed_base = _inspect_base_model(base_model, base_size, union_size)
    if observed_base != dict(base_info):
        raise UnionInitCheckpointError("base model differs from the initialization manifest")
    union_input, union_output = _weight_pair(union_model)
    base_input, base_output = _weight_pair(base_model)
    if tuple(union_input.shape) != (union_size, int(base_info["hidden_size"])):
        raise UnionInitCheckpointError("saved checkpoint input vocabulary shape is invalid")
    if tuple(union_output.shape) != tuple(union_input.shape):
        raise UnionInitCheckpointError("saved checkpoint output vocabulary shape is invalid")
    if getattr(union_model.config, "vocab_size", None) != union_size:
        raise UnionInitCheckpointError("saved checkpoint config vocabulary is invalid")
    if getattr(union_model.config, "tie_word_embeddings", None) != tied:
        raise UnionInitCheckpointError("saved checkpoint config tie state changed")
    if _weights_are_tied(union_input, union_output) != tied:
        raise UnionInitCheckpointError("saved checkpoint weight tie state changed")
    if not torch.equal(union_input[:base_size], base_input[:base_size]):
        raise UnionInitCheckpointError("saved checkpoint changed old input rows")
    if not torch.equal(union_output[:base_size], base_output[:base_size]):
        raise UnionInitCheckpointError("saved checkpoint changed old output rows")

    row_count = union_size - base_size
    expected_input = _t5_initialized_rows(
        rows=row_count,
        width=int(base_info["hidden_size"]),
        dtype=union_input.dtype,
        seed=seed + INPUT_STREAM_OFFSET,
        initializer_factor=float(base_info["initializer_factor"]),
    )
    if not torch.equal(union_input[base_size:], expected_input):
        raise UnionInitCheckpointError("saved input rows do not replay from the fixed seed")
    if tied:
        if not torch.equal(union_output[base_size:], expected_input):
            raise UnionInitCheckpointError("tied output rows differ from shared initialization")
    else:
        expected_output = _t5_initialized_rows(
            rows=row_count,
            width=int(base_info["hidden_size"]),
            dtype=union_output.dtype,
            seed=seed + OUTPUT_STREAM_OFFSET,
            initializer_factor=float(base_info["initializer_factor"]),
        )
        if not torch.equal(union_output[base_size:], expected_output):
            raise UnionInitCheckpointError("saved output rows do not replay from the fixed seed")

    checkpoint_vocab = int(base_info["checkpoint_vocab_size"])
    if checkpoint_vocab > base_size:
        if torch.equal(
            union_input[base_size:checkpoint_vocab],
            base_input[base_size:checkpoint_vocab],
        ):
            raise UnionInitCheckpointError("reclaimed base-model input rows were not reset")
        if not tied and torch.equal(
            union_output[base_size:checkpoint_vocab],
            base_output[base_size:checkpoint_vocab],
        ):
            raise UnionInitCheckpointError("reclaimed base-model output rows were not reset")
    if not torch.isfinite(union_input[base_size:]).all().item() or not torch.isfinite(
        union_output[base_size:]
    ).all().item():
        raise UnionInitCheckpointError("saved initialized rows are non-finite")

    base_state = base_model.state_dict()
    union_state = union_model.state_dict()
    if set(base_state) != set(union_state):
        raise UnionInitCheckpointError("raw T5 state keys changed during initialization")
    if not T5_VOCAB_STATE_KEYS.issubset(base_state):
        raise UnionInitCheckpointError("raw T5 vocabulary state keys are incomplete")
    for key in sorted(set(base_state).difference(T5_VOCAB_STATE_KEYS)):
        if not torch.equal(base_state[key], union_state[key]):
            raise UnionInitCheckpointError(
                "non-vocabulary state changed during initialization: {}".format(key)
            )


def _verified_tokenizer_and_dimensions(
    *,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    verified_tokenizer_loader: Optional[Callable[..., Any]] = None,
) -> tuple[Any, int, int, str, str]:
    if verified_tokenizer_loader is None:
        verified = _load_verified_tokenizer(
            base_tokenizer_snapshot=base_tokenizer_snapshot,
            union_tokenizer_dir=union_tokenizer_dir,
        )
    else:
        verified = verified_tokenizer_loader(
            base_snapshot=base_tokenizer_snapshot,
            output_dir=union_tokenizer_dir,
        )
    base_size, union_size, contract_hash, snapshot_hash = _tokenizer_dimensions(verified)
    return verified, base_size, union_size, contract_hash, snapshot_hash


def build_union_init_checkpoint(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    output_dir: Path,
    seed: int,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    verified_tokenizer_loader: Optional[Callable[..., Any]] = None,
) -> VerifiedUnionInitCheckpoint:
    """Build, reload, verify, then publish the single four-grid initializer."""

    seed = _validate_seed(seed, "vocabulary initialization seed")
    geometry_fusion_seed = _validate_seed(
        geometry_fusion_seed, "geometry fusion initialization seed"
    )
    num_e3fp_embeddings = _validate_num_e3fp_embeddings(num_e3fp_embeddings)
    base_model_snapshot = _require_directory(base_model_snapshot, "base model snapshot")
    base_tokenizer_snapshot = _require_directory(
        base_tokenizer_snapshot, "base tokenizer snapshot"
    )
    union_tokenizer_dir = _require_directory(union_tokenizer_dir, "union tokenizer directory")
    output_dir = _require_new_directory(output_dir, "output directory")
    staging_dir = output_dir.with_name(output_dir.name + STAGING_SUFFIX)
    _require_new_directory(staging_dir, "staging directory")

    verified_tokenizer, base_size, union_size, contract_hash, snapshot_hash = (
        _verified_tokenizer_and_dimensions(
            base_tokenizer_snapshot=base_tokenizer_snapshot,
            union_tokenizer_dir=union_tokenizer_dir,
            verified_tokenizer_loader=verified_tokenizer_loader,
        )
    )
    model = _load_raw_t5_from_pretrained(base_model_snapshot)
    base_info = _inspect_base_model(model, base_size, union_size)
    _resize_and_initialize(
        model,
        base_tokenizer_size=base_size,
        union_size=union_size,
        seed=seed,
        base_info=base_info,
    )
    manifest = _expected_manifest(
        base_info=base_info,
        base_tokenizer_size=base_size,
        union_size=union_size,
        contract_hash=contract_hash,
        snapshot_hash=snapshot_hash,
        seed=seed,
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    checkpoint_path = staging_dir / CHECKPOINT_DIRECTORY
    try:
        model.save_pretrained(str(checkpoint_path), safe_serialization=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise UnionInitCheckpointError("shared raw-T5 checkpoint could not be saved") from exc
    _write_manifest_new(staging_dir / MANIFEST_NAME, manifest)

    # Release the construction model before the two-model semantic replay;
    # this keeps peak memory bounded for T5-base while remaining CPU-only.
    del model
    gc.collect()
    base_replay = _load_raw_t5_from_pretrained(base_model_snapshot)
    initialized_replay = _load_raw_t5_from_pretrained(checkpoint_path)
    _verify_model_pair(
        base_model=base_replay,
        union_model=initialized_replay,
        expected_manifest=manifest,
    )
    del base_replay
    gc.collect()

    staging_dir.rename(output_dir)
    return VerifiedUnionInitCheckpoint(
        model=initialized_replay,
        tokenizer_build=verified_tokenizer,
        manifest=manifest,
        checkpoint_path=output_dir / CHECKPOINT_DIRECTORY,
    )


def load_verified_union_init_checkpoint(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    output_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    verified_tokenizer_loader: Optional[Callable[..., Any]] = None,
) -> VerifiedUnionInitCheckpoint:
    """Load one independent raw-T5 copy after replaying the init contract."""

    base_model_snapshot = _require_directory(base_model_snapshot, "base model snapshot")
    base_tokenizer_snapshot = _require_directory(
        base_tokenizer_snapshot, "base tokenizer snapshot"
    )
    union_tokenizer_dir = _require_directory(union_tokenizer_dir, "union tokenizer directory")
    output_dir = _require_directory(output_dir, "union-init output directory")
    geometry_fusion_seed = _validate_seed(
        geometry_fusion_seed, "geometry fusion initialization seed"
    )
    num_e3fp_embeddings = _validate_num_e3fp_embeddings(num_e3fp_embeddings)
    checkpoint_path = output_dir / CHECKPOINT_DIRECTORY
    if not checkpoint_path.is_dir():
        raise UnionInitCheckpointError("union-init checkpoint directory is absent")
    manifest = _read_manifest(output_dir / MANIFEST_NAME)
    _validate_manifest_shape(manifest)

    verified_tokenizer, base_size, union_size, contract_hash, snapshot_hash = (
        _verified_tokenizer_and_dimensions(
            base_tokenizer_snapshot=base_tokenizer_snapshot,
            union_tokenizer_dir=union_tokenizer_dir,
            verified_tokenizer_loader=verified_tokenizer_loader,
        )
    )
    tokenizer_manifest = manifest.get("tokenizer")
    if not isinstance(tokenizer_manifest, Mapping) or dict(tokenizer_manifest) != {
        "base_vocab_size": base_size,
        "union_vocab_size": union_size,
        "tokenizer_contract_sha256": contract_hash,
        "tokenizer_snapshot_sha256": snapshot_hash,
    }:
        raise UnionInitCheckpointError("union-init checkpoint names a different tokenizer")

    base_model = _load_raw_t5_from_pretrained(base_model_snapshot)
    expected_base_info = _inspect_base_model(base_model, base_size, union_size)
    if manifest.get("base_model") != expected_base_info:
        raise UnionInitCheckpointError("union-init checkpoint names a different base T5")
    expected = _expected_manifest(
        base_info=expected_base_info,
        base_tokenizer_size=base_size,
        union_size=union_size,
        contract_hash=contract_hash,
        snapshot_hash=snapshot_hash,
        seed=_validate_seed(
            manifest.get("initialization", {}).get("seed")
            if isinstance(manifest.get("initialization"), Mapping)
            else None,
            "vocabulary initialization seed",
        ),
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
    )
    if manifest != expected:
        raise UnionInitCheckpointError("union-init manifest differs from its replayed contract")

    union_model = _load_raw_t5_from_pretrained(checkpoint_path)
    _verify_model_pair(
        base_model=base_model,
        union_model=union_model,
        expected_manifest=expected,
    )
    del base_model
    gc.collect()
    return VerifiedUnionInitCheckpoint(
        model=union_model,
        tokenizer_build=verified_tokenizer,
        manifest=dict(manifest),
        checkpoint_path=checkpoint_path,
    )


def load_verified_four_grid_wrapper(
    *,
    condition_id: str,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    output_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    verified_tokenizer_loader: Optional[Callable[..., Any]] = None,
) -> Any:
    """Construct one grid cell with deterministic, shared fusion parameters.

    Each call loads an independent raw-T5 object from the same published
    checkpoint.  ``condition_id`` is passed only to the wrapper's metadata;
    the complete parameter state is therefore identical for A0/A1/M0/M1.
    Wrapper construction uses a private deterministic CPU RNG state and
    restores the caller's state even if construction fails.
    """

    geometry_fusion_seed = _validate_seed(
        geometry_fusion_seed, "geometry fusion initialization seed"
    )
    num_e3fp_embeddings = _validate_num_e3fp_embeddings(num_e3fp_embeddings)
    verified = load_verified_union_init_checkpoint(
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=union_tokenizer_dir,
        output_dir=output_dir,
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        verified_tokenizer_loader=verified_tokenizer_loader,
    )
    from most_t5_next.p1.four_grid_t5_wrapper import FourGridT5Wrapper

    torch = _load_torch()
    rng_state = torch.random.get_rng_state()
    try:
        # Seed only the CPU generator used by the CPU-resident wrapper.  Do not
        # touch CUDA generator state before DDP/device placement exists.
        torch.random.default_generator.manual_seed(geometry_fusion_seed)
        wrapper = FourGridT5Wrapper(
            t5_model=verified.model,
            condition_id=condition_id,
            num_e3fp_embeddings=num_e3fp_embeddings,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in wrapper.parameters()} != {"cpu"}:
        raise UnionInitCheckpointError("verified wrapper must be constructed on CPU")
    return wrapper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    build = build_union_init_checkpoint(
        base_model_snapshot=args.base_model_snapshot,
        base_tokenizer_snapshot=args.base_tokenizer_snapshot,
        union_tokenizer_dir=args.union_tokenizer_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        geometry_fusion_seed=args.geometry_fusion_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "checkpoint_path": str(build.checkpoint_path),
                "base_vocab_size": build.manifest["tokenizer"]["base_vocab_size"],
                "union_vocab_size": build.manifest["tokenizer"]["union_vocab_size"],
                "seed": build.manifest["initialization"]["seed"],
                "geometry_fusion_seed": build.manifest["four_grid_wrapper"][
                    "geometry_fusion_seed"
                ],
                "num_e3fp_embeddings": build.manifest["four_grid_wrapper"][
                    "num_e3fp_embeddings"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "UnionInitCheckpointError",
    "VerifiedUnionInitCheckpoint",
    "build_union_init_checkpoint",
    "load_verified_four_grid_wrapper",
    "load_verified_union_init_checkpoint",
]
