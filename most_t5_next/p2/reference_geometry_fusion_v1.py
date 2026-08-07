"""Reference-calibrated E3FP carrier fusion for the PF-2A screen.

The numerical path deliberately follows the official 3D-MolT5 ``sum``
embedding setting before applying this project's motif carrier mapping:

``one shared E3FP table -> mean over four fixed shell slots -> atom mean``
``within each carrier -> 0.5 * identity + 0.5 * geometry at carriers``.

The fixed four-slot mean includes the zero padding embedding for an absent
shell, matching ``molecule_fp_embed_tokens(...).mean(dim=-2)`` in the
reference implementation.  Tokens without an atom carrier remain exactly
unchanged.  This module adds no gate, projection, teacher, or auxiliary loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_union_init_checkpoint,
)
from most_t5_next.p1.experiment_grid import GeometryBatchSidecar
from most_t5_next.p1.four_grid_t5_wrapper import FourGridT5Wrapper
from most_t5_next.p1.shared_geometry_fusion import (
    GeometryFusionError,
    GeometryInput,
    GeometryTensorSidecar,
    SharedE3FPCarrierFusion,
)


FUSION_ID = "shared-e3fp-fixed4-mean-carrier-atom-mean-balanced-half-v1"


class ReferenceE3FPCarrierFusion(SharedE3FPCarrierFusion):
    """Fuse a shared, fixed-shell-mean E3FP state at atom/motif carriers."""

    def __init__(self, *, num_e3fp_embeddings: int, hidden_size: int) -> None:
        # Reuse the mature tensor/contract validation from the P1 module while
        # intentionally replacing its four level-specific parameter tables.
        nn.Module.__init__(self)
        if isinstance(num_e3fp_embeddings, bool) or not isinstance(
            num_e3fp_embeddings, int
        ):
            raise GeometryFusionError("num_e3fp_embeddings must be an integer")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise GeometryFusionError("hidden_size must be an integer")
        if num_e3fp_embeddings <= 0 or hidden_size <= 0:
            raise GeometryFusionError("embedding dimensions must be positive")

        self.num_e3fp_embeddings = num_e3fp_embeddings
        self.hidden_size = hidden_size
        self.num_e3fp_levels = 4
        self.shared_embedding = nn.Embedding(
            num_e3fp_embeddings + 1,
            hidden_size,
            padding_idx=0,
        )

    @property
    def level_embeddings(self) -> tuple[nn.Embedding, ...]:
        """Compatibility view used only by the inherited input validator."""

        return (self.shared_embedding,) * self.num_e3fp_levels

    def forward(
        self,
        input_embeddings: Tensor,
        geometry: GeometryInput,
        *,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return unchanged noncarriers and balanced identity/3D carriers."""

        contract = geometry if isinstance(geometry, GeometryBatchSidecar) else None
        tensor_geometry = self._tensorize(geometry, device=input_embeddings.device)
        batch_size, token_width, _atom_width, _valid_level = self._validate(
            input_embeddings,
            attention_mask,
            tensor_geometry,
            contract=contract,
        )

        shifted_ids = (tensor_geometry.e3fp_ids + 1).to(dtype=torch.long)
        # Row zero is the fixed padding vector.  Averaging all four ordered
        # slots is the exact reference reduction used by 3D-MolT5.
        atom_states = self.shared_embedding(shifted_ids).mean(dim=2)

        atom_mask = tensor_geometry.e3fp_atom_mask
        batch_offsets = (
            torch.arange(batch_size, device=input_embeddings.device).unsqueeze(1)
            * token_width
        )
        flat_carriers = (
            tensor_geometry.e3fp_atom_to_token.to(dtype=torch.long) + batch_offsets
        )[atom_mask]
        flat_atom_states = atom_states[atom_mask]

        flat_carrier_states = input_embeddings.new_zeros(
            (batch_size * token_width, self.hidden_size)
        )
        flat_carrier_states.index_add_(0, flat_carriers, flat_atom_states)
        carrier_counts = input_embeddings.new_zeros((batch_size * token_width, 1))
        carrier_counts.index_add_(
            0,
            flat_carriers,
            input_embeddings.new_ones((flat_carriers.numel(), 1)),
        )
        flat_carrier_states = flat_carrier_states / carrier_counts.clamp_min(1)
        carrier_states = flat_carrier_states.view(
            batch_size,
            token_width,
            self.hidden_size,
        )
        carrier_mask = carrier_counts.view(batch_size, token_width, 1) > 0
        balanced = 0.5 * input_embeddings + 0.5 * carrier_states
        return torch.where(carrier_mask, balanced, input_embeddings)


def load_verified_reference_four_grid_wrapper(
    *,
    condition_id: str,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    output_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
) -> Any:
    """Load the frozen raw T5 and attach the deterministic PF-2A fusion."""

    verified = load_verified_union_init_checkpoint(
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=union_tokenizer_dir,
        output_dir=output_dir,
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
    )
    rng_state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(geometry_fusion_seed)
        wrapper = FourGridT5Wrapper(
            t5_model=verified.model,
            condition_id=condition_id,
            num_e3fp_embeddings=num_e3fp_embeddings,
            geometry_fusion_factory=ReferenceE3FPCarrierFusion,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in wrapper.parameters()} != {"cpu"}:
        raise GeometryFusionError("verified reference wrapper must start on CPU")
    return wrapper


__all__ = [
    "FUSION_ID",
    "ReferenceE3FPCarrierFusion",
    "load_verified_reference_four_grid_wrapper",
]
