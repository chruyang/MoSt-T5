"""3D-MotifT5 wrapper for the controlled E3FP parameter-tying screen."""

from __future__ import annotations

from torch import nn

from .factorized_motif_t5_v1 import FactorizedMotifT5V1
from .factorized_motif_t5_v3 import FactorizedMotifT5V3
from .motif_geometry_adapter_v10 import MotifGeometryAdapterV10


FACTORISATION_ID = "most-t5-p2/factorized-motif-t5/v10-e3fp-parameter-tying"


class FactorizedMotifT5V10(FactorizedMotifT5V3):
    def __init__(
        self,
        t5_model: nn.Module,
        *,
        num_e3fp_embeddings: int,
        parameter_tying: str,
        state_level2_weight: float = 0.25,
        state_embedding_dim: int = 64,
        atom_memory_dim: int = 768,
        max_identity_span_length: int = 128,
        max_atoms_per_motif: int = 128,
        geometry_fraction: float = 0.5,
    ) -> None:
        nn.Module.__init__(self)
        hidden_size = FactorizedMotifT5V1._bind_t5_boundary(
            self,
            t5_model=t5_model,
            state_level2_weight=state_level2_weight,
        )
        self.adapter = MotifGeometryAdapterV10(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
            state_embedding_dim=state_embedding_dim,
            atom_memory_dim=atom_memory_dim,
            max_identity_span_length=max_identity_span_length,
            max_atoms_per_motif=max_atoms_per_motif,
            geometry_fraction=geometry_fraction,
            parameter_tying=parameter_tying,
        )

    @property
    def factorisation_id(self) -> str:
        return FACTORISATION_ID


__all__ = ["FACTORISATION_ID", "FactorizedMotifT5V10"]
