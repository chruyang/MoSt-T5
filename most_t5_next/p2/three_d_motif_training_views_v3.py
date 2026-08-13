"""Explicit 1D/3D input views for 3D-MotifT5 V3.

This module intentionally does not choose a sampling ratio.  A training
protocol selects one of the three declared views per batch/update, while all
views reuse the same tokenizer, records, corruption seed, and V3 collator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)

from .factorized_view_collator_v1 import AtomStateProvider, FactorizedViewCollatorError
from .factorized_view_collator_v2 import CanonicalAtomAddressProvider
from .factorized_view_collator_v3 import (
    FactorizedMotifViewBatchV3,
    collate_factorized_motif_view_v3,
)


TRAINING_VIEW_ID = "most-t5-p2/3d-motif-training-views/v3"
TRAINING_VIEWS = ("m_only", "m_plus_g", "g_only")


@dataclass(frozen=True)
class ThreeDMotifTrainingBatchV3:
    view_id: str
    state_memory_mode: str
    batch: FactorizedMotifViewBatchV3

    def __getattr__(self, name: str):
        return getattr(self.batch, name)

    def model_inputs(self) -> dict[str, object]:
        values = self.batch.model_inputs()
        values["state_memory_mode"] = self.state_memory_mode
        return values


def collate_3d_motif_training_view_v3(
    records: Sequence[ProductionMotifRecord],
    *,
    view_id: str,
    tokenizer: ProductionTokenizerRuntime,
    seed: int,
    epoch: int,
    atom_address_provider: CanonicalAtomAddressProvider,
    identity_mask_probability: float = 0.15,
    num_e3fp_embeddings: int = 4096,
    atom_state_provider: AtomStateProvider | None = None,
    device: object | None = None,
) -> ThreeDMotifTrainingBatchV3:
    """Build one explicit 1D-only, joint, or geometry-required T5 view."""

    if view_id not in TRAINING_VIEWS:
        raise FactorizedViewCollatorError(
            "view_id must be m_only, m_plus_g or g_only"
        )
    if view_id == "m_only":
        objective_mode = "grammar"
        probability = float(identity_mask_probability)
        memory_mode = "zero"
    elif view_id == "m_plus_g":
        objective_mode = "cross_view"
        probability = float(identity_mask_probability)
        memory_mode = "aligned"
    else:
        objective_mode = "cross_view"
        probability = 1.0
        memory_mode = "aligned"

    batch = collate_factorized_motif_view_v3(
        records,
        tokenizer=tokenizer,
        objective_mode=objective_mode,
        seed=seed,
        epoch=epoch,
        atom_address_provider=atom_address_provider,
        identity_mask_probability=probability,
        num_e3fp_embeddings=num_e3fp_embeddings,
        atom_state_provider=atom_state_provider,
        device=device,
    )
    return ThreeDMotifTrainingBatchV3(
        view_id=view_id,
        state_memory_mode=memory_mode,
        batch=batch,
    )


__all__ = [
    "TRAINING_VIEW_ID",
    "TRAINING_VIEWS",
    "ThreeDMotifTrainingBatchV3",
    "collate_3d_motif_training_view_v3",
]
