"""V3 factorized collator with addressable GraphPorts endpoint geometry.

The tokenizer surface is unchanged.  This wrapper carries the validated
GraphPorts connection-marker -> attachment-model-atom relation through the
same whole-identity corruption used by V1/V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)

from .factorized_view_collator_v1 import AtomStateProvider, FactorizedViewCollatorError
from .factorized_view_collator_v2 import (
    CanonicalAtomAddressProvider,
    FactorizedMotifViewBatchV2,
    collate_factorized_motif_view_v2,
)


VIEW_COLLATOR_ID = "most-t5-p2/factorized-motif-view-collator/v3-endpoint-atom"


@dataclass(frozen=True)
class FactorizedMotifViewBatchV3:
    """A V2 view plus endpoint-token addresses on the corrupted token axis."""

    base: FactorizedMotifViewBatchV2
    endpoint_token_to_atom: Tensor

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def model_inputs(self) -> dict[str, object]:
        values = self.base.model_inputs()
        values["endpoint_token_to_atom"] = self.endpoint_token_to_atom
        return values


def _transform_endpoint_mapping(
    record: ProductionMotifRecord,
    current_span_bounds: Tensor,
    *,
    current_token_count: int,
) -> tuple[int, ...]:
    mapping = record.connection_token_to_atom
    if len(mapping) != len(record.input_ids):
        raise FactorizedViewCollatorError(
            "V3 requires a complete endpoint-token-to-atom production sidecar"
        )
    if any(
        (role == "connection") != (mapping[index] >= 0)
        for index, role in enumerate(record.token_role)
    ):
        raise FactorizedViewCollatorError(
            "endpoint-to-atom sidecar disagrees with connection token roles"
        )
    for atom_index in (value for value in mapping if value >= 0):
        if (
            atom_index >= len(record.atom_valid_mask)
            or not record.atom_valid_mask[atom_index]
            or not record.atom_is_attachment
            or not record.atom_is_attachment[atom_index]
        ):
            raise FactorizedViewCollatorError(
                "endpoint token must address one valid attachment atom"
            )

    motif_count = len(record.identity_spans)
    if current_span_bounds.shape != (motif_count, 2):
        raise FactorizedViewCollatorError("current identity-span row is malformed")
    reductions: list[tuple[int, int]] = []
    for motif_id, original in enumerate(record.identity_spans):
        start = int(current_span_bounds[motif_id, 0])
        stop = int(current_span_bounds[motif_id, 1])
        current_length = stop - start
        original_length = original.stop - original.start
        reduction = original_length - current_length
        if reduction < 0:
            raise FactorizedViewCollatorError(
                "identity corruption cannot expand an identity span"
            )
        reductions.append((original.stop, reduction))

    def transform(position: int) -> int:
        return position - sum(
            reduction for stop, reduction in reductions if stop <= position
        )

    transformed = [-1] * current_token_count
    for original_position, atom_index in enumerate(mapping):
        if atom_index < 0:
            continue
        position = transform(original_position)
        if not 0 <= position < current_token_count or transformed[position] != -1:
            raise FactorizedViewCollatorError(
                "identity corruption damaged endpoint-token addressing"
            )
        transformed[position] = atom_index
    if sum(value >= 0 for value in transformed) != sum(value >= 0 for value in mapping):
        raise FactorizedViewCollatorError("identity corruption lost a connection endpoint")
    return tuple(transformed)


def collate_factorized_motif_view_v3(
    records: Sequence[ProductionMotifRecord],
    *,
    tokenizer: ProductionTokenizerRuntime,
    objective_mode: str,
    seed: int,
    epoch: int,
    atom_address_provider: CanonicalAtomAddressProvider,
    identity_mask_probability: float = 0.15,
    state_mask_probability: float = 0.15,
    state_masking_strategy: str = "motif_atom_row",
    num_e3fp_embeddings: int = 4096,
    atom_state_provider: AtomStateProvider | None = None,
    device: object | None = None,
) -> FactorizedMotifViewBatchV3:
    """Build the existing V2 batch and retain exact endpoint atom addresses."""

    rows = tuple(records)
    if not rows:
        raise FactorizedViewCollatorError("records cannot be empty")
    base = collate_factorized_motif_view_v2(
        rows,
        tokenizer=tokenizer,
        objective_mode=objective_mode,
        seed=seed,
        epoch=epoch,
        atom_address_provider=atom_address_provider,
        identity_mask_probability=identity_mask_probability,
        state_mask_probability=state_mask_probability,
        state_masking_strategy=state_masking_strategy,
        num_e3fp_embeddings=num_e3fp_embeddings,
        atom_state_provider=atom_state_provider,
        device=device,
    )
    endpoint_map = torch.full_like(base.input_ids, -1)
    for batch_index, record in enumerate(rows):
        token_count = int(base.attention_mask[batch_index].sum().item())
        motif_count = len(record.identity_spans)
        row = _transform_endpoint_mapping(
            record,
            base.identity_span_bounds[batch_index, :motif_count].cpu(),
            current_token_count=token_count,
        )
        endpoint_map[batch_index, :token_count] = torch.as_tensor(
            row,
            dtype=torch.long,
            device=endpoint_map.device,
        )
    return FactorizedMotifViewBatchV3(
        base=base,
        endpoint_token_to_atom=endpoint_map,
    )


__all__ = [
    "VIEW_COLLATOR_ID",
    "FactorizedMotifViewBatchV3",
    "collate_factorized_motif_view_v3",
]
