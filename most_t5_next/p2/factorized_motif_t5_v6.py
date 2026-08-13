"""Anchored 3D-MotifT5 with the minimal global L0/high-shell reducer."""

from __future__ import annotations

from torch import nn

from .factorized_motif_t5_v1 import FactorizedMotifT5V1
from .factorized_motif_t5_v3 import FactorizedMotifT5V3
from .motif_geometry_adapter_v6 import MotifGeometryAdapterV6


FACTORISATION_ID = (
    "most-t5-p2/factorized-motif-geometry-t5/v6-global-l0-high-mix-anchored"
)


class FactorizedMotifT5V6(FactorizedMotifT5V3):
    """Reuse V3 carrier/endpoint routing while constructing one V6 adapter."""

    def __init__(
        self,
        t5_model: nn.Module,
        *,
        num_e3fp_embeddings: int,
        state_level2_weight: float = 0.25,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 128,
        max_identity_span_length: int = 128,
        max_atoms_per_motif: int = 128,
        geometry_fraction: float = 0.5,
        shell_reducer_mode: str = "adaptive_l0_high",
    ) -> None:
        nn.Module.__init__(self)
        hidden_size = FactorizedMotifT5V1._bind_t5_boundary(
            self,
            t5_model=t5_model,
            state_level2_weight=state_level2_weight,
        )
        self.adapter = MotifGeometryAdapterV6(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
            max_atoms_per_motif=max_atoms_per_motif,
            geometry_fraction=geometry_fraction,
            shell_reducer_mode=shell_reducer_mode,
        )

    @property
    def factorisation_id(self) -> str:
        return FACTORISATION_ID


__all__ = ["FACTORISATION_ID", "FactorizedMotifT5V6"]
