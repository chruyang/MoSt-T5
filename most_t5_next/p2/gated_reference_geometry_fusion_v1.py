"""Zero-initialized gated residual fusion for the independent F-Gate screen.

The geometry state is deliberately identical to PF-2A:

``one shared E3FP table -> mean over four fixed shell slots -> atom mean``
``within each carrier``.

Only the injection rule changes.  Carrier tokens use
``identity + tanh(alpha) * geometry`` with one scalar ``alpha`` initialized to
zero.  The complete wrapper is therefore function-identical to the geometry-
free T5 at initialization.  This is the smallest carrier-level analogue of the
zero-initialized gated residual used by Flamingo and LLaMA-Adapter; it adds no
projection, normalization, teacher, or auxiliary objective.
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
from most_t5_next.p2.reference_geometry_fusion_v1 import (
    ReferenceE3FPCarrierFusion,
)


FUSION_ID = (
    "shared-e3fp-fixed4-mean-carrier-atom-mean-"
    "zero-init-tanh-residual-v1"
)


class ZeroInitGatedE3FPCarrierFusion(ReferenceE3FPCarrierFusion):
    """Inject the PF-2A geometry state through one zero-initialized gate."""

    def __init__(self, *, num_e3fp_embeddings: int, hidden_size: int) -> None:
        super().__init__(
            num_e3fp_embeddings=num_e3fp_embeddings,
            hidden_size=hidden_size,
        )
        # Match the one-element scalar convention used by the reference gated
        # multimodal implementations and make its checkpoint shape explicit.
        self.geometry_gate_logit = nn.Parameter(torch.zeros(1))

    @property
    def effective_geometry_gate(self) -> Tensor:
        return torch.tanh(self.geometry_gate_logit)

    def forward(
        self,
        input_embeddings: Tensor,
        geometry: Any,
        *,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return exact identity at init and a gated residual at carriers."""

        contract = geometry if isinstance(geometry, GeometryBatchSidecar) else None
        tensor_geometry = self._tensorize(geometry, device=input_embeddings.device)
        batch_size, token_width, _atom_width, _valid_level = self._validate(
            input_embeddings,
            attention_mask,
            tensor_geometry,
            contract=contract,
        )

        shifted_ids = (tensor_geometry.e3fp_ids + 1).to(dtype=torch.long)
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
        gated = input_embeddings + self.effective_geometry_gate.to(
            dtype=input_embeddings.dtype
        ) * carrier_states
        return torch.where(carrier_mask, gated, input_embeddings)


def load_verified_gated_four_grid_wrapper(
    *,
    condition_id: str,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    output_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
) -> Any:
    """Load the frozen T5 and attach the deterministic F-Gate module."""

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
            geometry_fusion_factory=ZeroInitGatedE3FPCarrierFusion,
        )
    finally:
        torch.random.set_rng_state(rng_state)
    if {parameter.device.type for parameter in wrapper.parameters()} != {"cpu"}:
        raise RuntimeError("verified F-Gate wrapper must start on CPU")
    return wrapper


__all__ = [
    "FUSION_ID",
    "ZeroInitGatedE3FPCarrierFusion",
    "load_verified_gated_four_grid_wrapper",
]
