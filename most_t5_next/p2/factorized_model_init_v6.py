"""Deterministic initialization for the minimal L0/high-shell V6 reducer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from most_t5_next.p1.build_anchored_union_init_checkpoint_v1 import (
    load_verified_anchored_union_init_checkpoint,
)

from .factorized_model_init_v5 import (
    FactorizedModelInitV5Error,
    factorized_initialization_contract_v5,
)
from .factorized_motif_t5_v6 import FACTORISATION_ID, FactorizedMotifT5V6
from .motif_geometry_adapter_v6 import (
    ADAPTER_ID,
    ATOM_ENCODER_VARIANT,
    INITIAL_L0_WEIGHT,
    SHELL_REDUCER_MODES,
)


FACTORIZED_INIT_ID = "most-t5-p2/factorized-model-init/v6-global-l0-high-mix"


class FactorizedModelInitV6Error(FactorizedModelInitV5Error):
    """The anchored T5 or deterministic V6 reducer contract is invalid."""


def factorized_initialization_contract_v6(
    *,
    semantic_plan_sha256: str,
    adapter_seed: int,
    num_e3fp_embeddings: int,
    state_level2_weight: float,
    state_embedding_dim: int,
    atom_memory_dim: int,
    max_identity_span_length: int,
    max_atoms_per_motif: int,
    geometry_fraction: float,
    shell_reducer_mode: str,
) -> dict[str, object]:
    if shell_reducer_mode not in SHELL_REDUCER_MODES:
        raise FactorizedModelInitV6Error("shell reducer mode is invalid")
    contract = factorized_initialization_contract_v5(
        semantic_plan_sha256=semantic_plan_sha256,
        adapter_seed=adapter_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        state_level2_weight=state_level2_weight,
        state_embedding_dim=state_embedding_dim,
        atom_memory_dim=atom_memory_dim,
        max_identity_span_length=max_identity_span_length,
        max_atoms_per_motif=max_atoms_per_motif,
        geometry_fraction=geometry_fraction,
    )
    contract.update(
        {
            "schema_version": FACTORIZED_INIT_ID,
            "factorisation_id": FACTORISATION_ID,
            "adapter_id": ADAPTER_ID,
            "atom_encoder_variant": ATOM_ENCODER_VARIANT,
            "shell_reducer_mode": shell_reducer_mode,
            "initial_l0_weight": INITIAL_L0_WEIGHT,
            "shell_reduction": (
                "arithmetic_mean_fixed_denominator_4"
                if shell_reducer_mode == "fixed_four_mean"
                else "global_convex_l0_vs_fixed_denominator_3_high_shell_mean"
            ),
            "learned_shell_parameters": (
                0 if shell_reducer_mode == "fixed_four_mean" else 1
            ),
        }
    )
    return contract


def load_deterministic_factorized_model_v6(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    anchored_tokenizer_dir: Path,
    semantic_plan_sha256: str,
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
    shell_reducer_mode: str = "adaptive_l0_high",
    union_loader: Callable[..., Any] = load_verified_anchored_union_init_checkpoint,
) -> FactorizedMotifT5V6:
    contract = factorized_initialization_contract_v6(
        semantic_plan_sha256=semantic_plan_sha256,
        adapter_seed=adapter_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        state_level2_weight=state_level2_weight,
        state_embedding_dim=state_embedding_dim,
        atom_memory_dim=atom_memory_dim,
        max_identity_span_length=max_identity_span_length,
        max_atoms_per_motif=max_atoms_per_motif,
        geometry_fraction=geometry_fraction,
        shell_reducer_mode=shell_reducer_mode,
    )
    rng_state = torch.random.get_rng_state()
    try:
        verified = union_loader(
            base_model_snapshot=Path(base_model_snapshot),
            base_tokenizer_snapshot=Path(base_tokenizer_snapshot),
            anchored_tokenizer_dir=Path(anchored_tokenizer_dir),
            semantic_plan_sha256=semantic_plan_sha256,
            output_dir=Path(union_init_dir),
            geometry_fusion_seed=int(union_geometry_fusion_seed),
            num_e3fp_embeddings=int(num_e3fp_embeddings),
        )
        t5_model = getattr(verified, "model", None)
        if not isinstance(t5_model, torch.nn.Module):
            raise FactorizedModelInitV6Error(
                "verified anchored union-init loader did not return a torch model"
            )
        torch.random.default_generator.manual_seed(int(contract["adapter_seed"]))
        model = FactorizedMotifT5V6(
            t5_model,
            num_e3fp_embeddings=int(contract["num_e3fp_embeddings"]),
            state_level2_weight=float(contract["state_level2_weight"]),
            state_embedding_dim=int(contract["state_embedding_dim"]),
            atom_memory_dim=int(contract["atom_memory_dim"]),
            max_identity_span_length=int(contract["max_identity_span_length"]),
            max_atoms_per_motif=int(contract["max_atoms_per_motif"]),
            geometry_fraction=float(contract["geometry_fraction"]),
            shell_reducer_mode=str(contract["shell_reducer_mode"]),
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise FactorizedModelInitV6Error(
            "factorized model must be constructed on CPU before device placement"
        )
    return model


__all__ = [
    "FACTORIZED_INIT_ID",
    "FactorizedModelInitV6Error",
    "factorized_initialization_contract_v6",
    "load_deterministic_factorized_model_v6",
]
