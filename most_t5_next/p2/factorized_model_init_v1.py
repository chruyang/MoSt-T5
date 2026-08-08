"""Deterministic loader for the PF-10 factorized T5 mechanism cells.

The published union-init checkpoint owns the T5 and vocabulary rows.  This
module adds the new motif-state adapter under one private CPU RNG stream, so
B2D and F3D start from bitwise-identical but storage-independent parameters.
It does not create a second T5 checkpoint or derive initialization from data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_union_init_checkpoint,
)

from .factorized_motif_t5_v1 import FactorizedMotifT5V1


FACTORIZED_INIT_ID = "most-t5-p2/factorized-model-init/v1"


class FactorizedModelInitError(RuntimeError):
    """The shared T5 or deterministic adapter initialization is invalid."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FactorizedModelInitError(f"{name} must be a positive integer")
    return int(value)


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 2**63 - 1
    ):
        raise FactorizedModelInitError("adapter_seed must be in [0, 2**63-1)")
    return int(value)


def factorized_initialization_contract(
    *,
    adapter_seed: int,
    num_e3fp_embeddings: int,
    state_level2_weight: float,
    state_embedding_dim: int,
    atom_memory_dim: int,
    max_identity_span_length: int,
) -> dict[str, object]:
    """Return the data-independent adapter initialization contract."""

    adapter_seed = _seed(adapter_seed)
    num_e3fp_embeddings = _positive_int(
        num_e3fp_embeddings, "num_e3fp_embeddings"
    )
    if num_e3fp_embeddings <= 1:
        raise FactorizedModelInitError("num_e3fp_embeddings must exceed one")
    dimensions = {
        "state_embedding_dim": _positive_int(
            state_embedding_dim, "state_embedding_dim"
        ),
        "atom_memory_dim": _positive_int(atom_memory_dim, "atom_memory_dim"),
        "max_identity_span_length": _positive_int(
            max_identity_span_length, "max_identity_span_length"
        ),
    }
    if (
        isinstance(state_level2_weight, bool)
        or not isinstance(state_level2_weight, (int, float))
        or float(state_level2_weight) < 0.0
    ):
        raise FactorizedModelInitError(
            "state_level2_weight must be a non-negative number"
        )
    return {
        "schema_version": FACTORIZED_INIT_ID,
        "adapter_seed": adapter_seed,
        "num_e3fp_embeddings": num_e3fp_embeddings,
        "state_level2_weight": float(state_level2_weight),
        **dimensions,
        "rng_scope": "private_cpu_default_generator",
        "data_dependent_initialization": False,
        "paired_cells": ["B2D", "F3D"],
    }


def load_deterministic_factorized_model(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    union_init_dir: Path,
    union_geometry_fusion_seed: int,
    adapter_seed: int,
    num_e3fp_embeddings: int,
    state_level2_weight: float = 0.25,
    state_embedding_dim: int = 64,
    atom_memory_dim: int = 128,
    max_identity_span_length: int = 128,
    union_loader: Callable[..., Any] = load_verified_union_init_checkpoint,
) -> FactorizedMotifT5V1:
    """Load one independent T5 and deterministically add adapter-v1.

    Repeated calls with the same published union-init and contract yield
    bitwise-equal state dictionaries, while every tensor has independent
    storage.  The caller's CPU RNG state is restored even if construction
    fails.  CUDA RNG is never touched.
    """

    contract = factorized_initialization_contract(
        adapter_seed=adapter_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        state_level2_weight=state_level2_weight,
        state_embedding_dim=state_embedding_dim,
        atom_memory_dim=atom_memory_dim,
        max_identity_span_length=max_identity_span_length,
    )
    rng_state = torch.random.get_rng_state()
    try:
        verified = union_loader(
            base_model_snapshot=Path(base_model_snapshot),
            base_tokenizer_snapshot=Path(base_tokenizer_snapshot),
            union_tokenizer_dir=Path(union_tokenizer_dir),
            output_dir=Path(union_init_dir),
            geometry_fusion_seed=int(union_geometry_fusion_seed),
            num_e3fp_embeddings=int(num_e3fp_embeddings),
        )
        t5_model = getattr(verified, "model", None)
        if not isinstance(t5_model, torch.nn.Module):
            raise FactorizedModelInitError(
                "verified union-init loader did not return a torch model"
            )
        torch.random.default_generator.manual_seed(int(contract["adapter_seed"]))
        model = FactorizedMotifT5V1(
            t5_model,
            num_e3fp_embeddings=int(contract["num_e3fp_embeddings"]),
            state_level2_weight=float(contract["state_level2_weight"]),
            state_embedding_dim=int(contract["state_embedding_dim"]),
            atom_memory_dim=int(contract["atom_memory_dim"]),
            max_identity_span_length=int(contract["max_identity_span_length"]),
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise FactorizedModelInitError(
            "factorized model must be constructed on CPU before device placement"
        )
    return model


__all__ = [
    "FACTORIZED_INIT_ID",
    "FactorizedModelInitError",
    "factorized_initialization_contract",
    "load_deterministic_factorized_model",
]
