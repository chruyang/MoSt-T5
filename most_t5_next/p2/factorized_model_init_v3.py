"""Deterministic construction contract for carrier+endpoint 3D-MotifT5 V3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_union_init_checkpoint,
)

from .factorized_motif_t5_v3 import FACTORISATION_ID, FactorizedMotifT5V3
from .motif_geometry_adapter_v3 import ADAPTER_ID


FACTORIZED_INIT_ID = "most-t5-p2/factorized-model-init/v3-carrier-endpoint"


class FactorizedModelInitV3Error(RuntimeError):
    """The shared T5 or deterministic V3 adapter initialization is invalid."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FactorizedModelInitV3Error(f"{name} must be a positive integer")
    return int(value)


def factorized_initialization_contract_v3(
    *,
    adapter_seed: int,
    num_e3fp_embeddings: int,
    state_level2_weight: float,
    state_embedding_dim: int,
    atom_memory_dim: int,
    max_identity_span_length: int,
    max_atoms_per_motif: int,
    geometry_fraction: float,
) -> dict[str, object]:
    if (
        isinstance(adapter_seed, bool)
        or not isinstance(adapter_seed, int)
        or not 0 <= adapter_seed < 2**63 - 1
    ):
        raise FactorizedModelInitV3Error(
            "adapter_seed must be in [0, 2**63-1)"
        )
    dimensions = {
        "num_e3fp_embeddings": _positive_int(
            num_e3fp_embeddings, "num_e3fp_embeddings"
        ),
        "state_embedding_dim": _positive_int(
            state_embedding_dim, "state_embedding_dim"
        ),
        "atom_memory_dim": _positive_int(atom_memory_dim, "atom_memory_dim"),
        "max_identity_span_length": _positive_int(
            max_identity_span_length, "max_identity_span_length"
        ),
        "max_atoms_per_motif": _positive_int(
            max_atoms_per_motif, "max_atoms_per_motif"
        ),
    }
    if dimensions["num_e3fp_embeddings"] <= 1:
        raise FactorizedModelInitV3Error("num_e3fp_embeddings must exceed one")
    if (
        isinstance(state_level2_weight, bool)
        or not isinstance(state_level2_weight, (int, float))
        or float(state_level2_weight) < 0.0
    ):
        raise FactorizedModelInitV3Error(
            "state_level2_weight must be non-negative"
        )
    if (
        isinstance(geometry_fraction, bool)
        or not isinstance(geometry_fraction, (int, float))
        or not 0.0 < float(geometry_fraction) < 1.0
    ):
        raise FactorizedModelInitV3Error(
            "geometry_fraction must lie strictly between zero and one"
        )
    return {
        "schema_version": FACTORIZED_INIT_ID,
        "factorisation_id": FACTORISATION_ID,
        "adapter_id": ADAPTER_ID,
        "adapter_seed": int(adapter_seed),
        "state_level2_weight": float(state_level2_weight),
        "geometry_fraction": float(geometry_fraction),
        **dimensions,
        "geometry_injection": "fixed_normalized_carrier_and_endpoint_mix",
        "geometry_shells": [1, 2],
        "endpoint_address": "graphports_connection_token_to_model_atom",
        "text_tokens_added": 0,
        "trainable_collapse_gate": False,
        "rng_scope": "private_cpu_default_generator",
        "data_dependent_initialization": False,
        "paired_cells": ["B2D", "F3D"],
    }


def load_deterministic_factorized_model_v3(
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
    max_atoms_per_motif: int = 128,
    geometry_fraction: float = 0.5,
    union_loader: Callable[..., Any] = load_verified_union_init_checkpoint,
) -> FactorizedMotifT5V3:
    contract = factorized_initialization_contract_v3(
        adapter_seed=adapter_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        state_level2_weight=state_level2_weight,
        state_embedding_dim=state_embedding_dim,
        atom_memory_dim=atom_memory_dim,
        max_identity_span_length=max_identity_span_length,
        max_atoms_per_motif=max_atoms_per_motif,
        geometry_fraction=geometry_fraction,
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
            raise FactorizedModelInitV3Error(
                "verified union-init loader did not return a torch model"
            )
        torch.random.default_generator.manual_seed(int(contract["adapter_seed"]))
        model = FactorizedMotifT5V3(
            t5_model,
            num_e3fp_embeddings=int(contract["num_e3fp_embeddings"]),
            state_level2_weight=float(contract["state_level2_weight"]),
            state_embedding_dim=int(contract["state_embedding_dim"]),
            atom_memory_dim=int(contract["atom_memory_dim"]),
            max_identity_span_length=int(contract["max_identity_span_length"]),
            max_atoms_per_motif=int(contract["max_atoms_per_motif"]),
            geometry_fraction=float(contract["geometry_fraction"]),
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise FactorizedModelInitV3Error(
            "factorized model must be constructed on CPU before device placement"
        )
    return model


__all__ = [
    "FACTORIZED_INIT_ID",
    "FactorizedModelInitV3Error",
    "factorized_initialization_contract_v3",
    "load_deterministic_factorized_model_v3",
]
