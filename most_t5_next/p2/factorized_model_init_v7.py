"""Deterministic initialization for the one-linear L0/high-shell V7 reducer."""

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
from .factorized_motif_t5_v7 import FACTORISATION_ID, FactorizedMotifT5V7
from .motif_geometry_adapter_v7 import ADAPTER_ID, ATOM_ENCODER_VARIANT


FACTORIZED_INIT_ID = "most-t5-p2/factorized-model-init/v7-linear-l0-high"


class FactorizedModelInitV7Error(FactorizedModelInitV5Error):
    """The anchored T5 or deterministic V7 reducer contract is invalid."""


def factorized_initialization_contract_v7(**kwargs: object) -> dict[str, object]:
    contract = factorized_initialization_contract_v5(**kwargs)
    atom_memory_dim = int(contract["atom_memory_dim"])
    contract.update(
        {
            "schema_version": FACTORIZED_INIT_ID,
            "factorisation_id": FACTORISATION_ID,
            "adapter_id": ADAPTER_ID,
            "atom_encoder_variant": ATOM_ENCODER_VARIANT,
            "shell_reduction": "concat_l0_fixed_denominator_3_high_mean_then_linear",
            "projection": "one_bias_free_linear_2d_to_d",
            "projection_parameter_count": 2 * atom_memory_dim * atom_memory_dim,
            "initial_projection": "0.25I_for_l0_and_0.75I_for_high",
            "level_embedding": False,
            "attachment_role_is_learned_atom_input": False,
            "presence_feature": False,
            "atom_mlp": False,
        }
    )
    return contract


def load_deterministic_factorized_model_v7(
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
    union_loader: Callable[..., Any] = load_verified_anchored_union_init_checkpoint,
) -> FactorizedMotifT5V7:
    contract = factorized_initialization_contract_v7(
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
            raise FactorizedModelInitV7Error(
                "verified anchored union-init loader did not return a torch model"
            )
        torch.random.default_generator.manual_seed(int(contract["adapter_seed"]))
        model = FactorizedMotifT5V7(
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
        raise FactorizedModelInitV7Error(
            "factorized model must be constructed on CPU before device placement"
        )
    return model


__all__ = [
    "FACTORIZED_INIT_ID",
    "FactorizedModelInitV7Error",
    "factorized_initialization_contract_v7",
    "load_deterministic_factorized_model_v7",
]
