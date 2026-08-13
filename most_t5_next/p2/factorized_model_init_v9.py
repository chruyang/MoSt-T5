"""Deterministic initialization for the level-aware minimal atom-phi V9."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from most_t5_next.p1.build_anchored_union_init_checkpoint_v1 import (
    load_verified_anchored_union_init_checkpoint,
)

from .factorized_model_init_v8 import (
    FactorizedModelInitV8Error,
    factorized_initialization_contract_v8,
)
from .factorized_motif_t5_v9 import FACTORISATION_ID, FactorizedMotifT5V9
from .motif_geometry_adapter_v9 import ADAPTER_ID, ATOM_ENCODER_VARIANT


FACTORIZED_INIT_ID = "most-t5-p2/factorized-model-init/v9-level-aware-phi"


class FactorizedModelInitV9Error(FactorizedModelInitV8Error):
    """The anchored T5 or deterministic V9 atom-phi contract is invalid."""


def factorized_initialization_contract_v9(**kwargs: object) -> dict[str, object]:
    contract = factorized_initialization_contract_v8(**kwargs)
    contract.update(
        {
            "schema_version": FACTORIZED_INIT_ID,
            "factorisation_id": FACTORISATION_ID,
            "adapter_id": ADAPTER_ID,
            "atom_encoder_variant": ATOM_ENCODER_VARIANT,
            "level_embedding": True,
            "level_embedding_count": 4,
            "attachment_role_is_learned_atom_input": False,
            "presence_feature": False,
        }
    )
    return contract


def load_deterministic_factorized_model_v9(
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
) -> FactorizedMotifT5V9:
    contract = factorized_initialization_contract_v9(
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
            raise FactorizedModelInitV9Error(
                "verified anchored union-init loader did not return a torch model"
            )
        torch.random.default_generator.manual_seed(int(contract["adapter_seed"]))
        model = FactorizedMotifT5V9(
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
        raise FactorizedModelInitV9Error(
            "factorized model must be constructed on CPU before device placement"
        )
    return model


__all__ = [
    "FACTORIZED_INIT_ID",
    "FactorizedModelInitV9Error",
    "factorized_initialization_contract_v9",
    "load_deterministic_factorized_model_v9",
]
