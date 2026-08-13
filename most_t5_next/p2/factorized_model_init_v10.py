"""Deterministic initialization for the E3FP parameter-tying experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from most_t5_next.p1.build_anchored_union_init_checkpoint_v1 import (
    load_verified_anchored_union_init_checkpoint,
)

from .e3fp_atom_embedding_v1 import REFERENCE_SHARED_FIXED4
from .factorized_motif_t5_v10 import FACTORISATION_ID, FactorizedMotifT5V10
from .motif_geometry_adapter_v10 import ADAPTER_ID, SUPPORTED_PARAMETER_TYING


FACTORIZED_INIT_ID = "most-t5-p2/factorized-model-init/v10-e3fp-parameter-tying"


class FactorizedModelInitV10Error(RuntimeError):
    pass


def factorized_initialization_contract_v10(
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
    parameter_tying: str,
) -> dict[str, object]:
    if parameter_tying not in SUPPORTED_PARAMETER_TYING:
        raise FactorizedModelInitV10Error("unsupported E3FP parameter tying")
    if len(semantic_plan_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in semantic_plan_sha256
    ):
        raise FactorizedModelInitV10Error("semantic plan must be a lower-case SHA-256")
    dimensions = {
        "num_e3fp_embeddings": num_e3fp_embeddings,
        "state_embedding_dim": state_embedding_dim,
        "atom_memory_dim": atom_memory_dim,
        "max_identity_span_length": max_identity_span_length,
        "max_atoms_per_motif": max_atoms_per_motif,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions.values()):
        raise FactorizedModelInitV10Error("all model dimensions must be positive integers")
    if isinstance(adapter_seed, bool) or not isinstance(adapter_seed, int) or not 0 <= adapter_seed < 2**63 - 1:
        raise FactorizedModelInitV10Error("adapter seed is outside the supported domain")
    if not 0.0 < float(geometry_fraction) < 1.0:
        raise FactorizedModelInitV10Error("geometry fraction must be in (0,1)")
    table_count = {
        "reference_shared_fixed4": 1,
        "l0_state_fixed4": 2,
        "level_specific_fixed4": 4,
    }[parameter_tying]
    return {
        "schema_version": FACTORIZED_INIT_ID,
        "factorisation_id": FACTORISATION_ID,
        "adapter_id": ADAPTER_ID,
        "semantic_plan_sha256": semantic_plan_sha256,
        "adapter_seed": adapter_seed,
        "state_level2_weight": float(state_level2_weight),
        "geometry_fraction": float(geometry_fraction),
        **dimensions,
        "e3fp_parameter_tying": parameter_tying,
        "e3fp_table_count": table_count,
        "e3fp_table_rows": num_e3fp_embeddings + 1,
        "e3fp_parameter_count": table_count * (num_e3fp_embeddings + 1) * atom_memory_dim,
        "e3fp_external_padding_id": -1,
        "e3fp_internal_padding_row": 0,
        "shell_reduction": "arithmetic_mean_fixed_denominator_4",
        "missing_shell_contribution": "zero",
        "update_zero_reference_equivalence_required": True,
        "attachment_role_is_learned_atom_input": False,
        "atom_mlp": False,
    }


def _first_table_weight(model: FactorizedMotifT5V10) -> torch.Tensor:
    encoder = model.adapter.e3fp_atom_embedding
    if encoder.variant == "l0_state_fixed4":
        return encoder.l0_embedding.weight
    if encoder.variant == "level_specific_fixed4":
        return encoder.level_embeddings[0].weight
    return encoder.shared_embedding.weight


def load_deterministic_factorized_model_v10(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    anchored_tokenizer_dir: Path,
    semantic_plan_sha256: str,
    union_init_dir: Path,
    union_geometry_fusion_seed: int,
    adapter_seed: int,
    num_e3fp_embeddings: int,
    parameter_tying: str,
    state_level2_weight: float = 0.25,
    state_embedding_dim: int = 64,
    atom_memory_dim: int = 768,
    max_identity_span_length: int = 128,
    max_atoms_per_motif: int = 128,
    geometry_fraction: float = 0.5,
    union_loader: Callable[..., Any] = load_verified_anchored_union_init_checkpoint,
) -> FactorizedMotifT5V10:
    contract = factorized_initialization_contract_v10(
        semantic_plan_sha256=semantic_plan_sha256,
        adapter_seed=adapter_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        state_level2_weight=state_level2_weight,
        state_embedding_dim=state_embedding_dim,
        atom_memory_dim=atom_memory_dim,
        max_identity_span_length=max_identity_span_length,
        max_atoms_per_motif=max_atoms_per_motif,
        geometry_fraction=geometry_fraction,
        parameter_tying=parameter_tying,
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
            raise FactorizedModelInitV10Error("verified union init did not return a torch model")
        torch.random.default_generator.manual_seed(int(adapter_seed))
        model = FactorizedMotifT5V10(
            t5_model,
            num_e3fp_embeddings=num_e3fp_embeddings,
            parameter_tying=parameter_tying,
            state_level2_weight=state_level2_weight,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
            max_atoms_per_motif=max_atoms_per_motif,
            geometry_fraction=geometry_fraction,
        )
        encoder = model.adapter.e3fp_atom_embedding
        if parameter_tying != REFERENCE_SHARED_FIXED4:
            encoder.initialize_tied_tables_from_shared(_first_table_weight(model).detach().clone())
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise FactorizedModelInitV10Error("model must be built on CPU before placement")
    model.initialization_contract = contract
    return model


__all__ = [
    "FACTORIZED_INIT_ID",
    "FactorizedModelInitV10Error",
    "factorized_initialization_contract_v10",
    "load_deterministic_factorized_model_v10",
]
