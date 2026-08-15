"""CAP/T2M collation and the single final wrapper-input mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .fragsmiles_training_tensor_cache_v1 import (
    CachedFragSmilesRecord,
    collate_cached_fragsmiles,
)


class FragSmilesPairedPretrainingError(ValueError):
    pass


@dataclass(frozen=True)
class Phase2PairedSample:
    record: CachedFragSmilesRecord
    text_input_ids: np.ndarray


def _pad_rows(
    rows: Sequence[Sequence[int]], *, pad_value: int, cap: int, name: str
) -> torch.Tensor:
    if not rows:
        raise FragSmilesPairedPretrainingError(f"{name} rows are empty")
    maximum = max(len(row) for row in rows)
    if maximum > cap:
        raise FragSmilesPairedPretrainingError(f"{name} exceeds cap {cap}")
    result = torch.full((len(rows), maximum), int(pad_value), dtype=torch.long)
    for index, row in enumerate(rows):
        # Cache rows are read-only mmap views.  ``torch.as_tensor`` aliases
        # their storage and emits a non-writable warning even though this
        # assignment only reads the source.  A small per-row copy keeps the
        # collator contract explicit and avoids undefined aliasing semantics.
        result[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def _truncate_text_rows(
    rows: Sequence[Sequence[int]], *, cap: int, eos_token_id: int
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for row in rows:
        values = tuple(int(value) for value in row)
        if len(values) > cap:
            values = values[: cap - 1] + (int(eos_token_id),)
        result.append(values)
    return tuple(result)


def collate_phase2_cap_samples(
    samples: Sequence[Phase2PairedSample],
    *,
    pad_token_id: int,
    encoder_cap: int = 512,
    target_cap: int = 512,
    eos_token_id: int = 1,
) -> dict[str, object]:
    if not samples:
        raise FragSmilesPairedPretrainingError("CAP batch is empty")
    if max(len(sample.record.input_ids) for sample in samples) > encoder_cap:
        raise FragSmilesPairedPretrainingError("CAP molecule exceeds encoder cap")
    batch = collate_cached_fragsmiles(
        tuple(sample.record for sample in samples), pad_token_id=pad_token_id
    )
    text_targets = _truncate_text_rows(
        tuple(sample.text_input_ids for sample in samples),
        cap=target_cap,
        eos_token_id=eos_token_id,
    )
    batch["labels"] = _pad_rows(
        text_targets,
        pad_value=-100,
        cap=target_cap,
        name="CAP target",
    )
    batch["fragment_geometry_mask"] = batch["fragment_mask"].clone()
    batch["endpoint_geometry_mask"] = batch["endpoint_mask"].clone()
    batch["view"] = "P2-CAP"
    return batch


def collate_phase2_t2m_samples(
    samples: Sequence[Phase2PairedSample],
    *,
    pad_token_id: int,
    encoder_cap: int = 512,
    target_cap: int = 512,
    eos_token_id: int = 1,
) -> dict[str, torch.Tensor | str]:
    if not samples:
        raise FragSmilesPairedPretrainingError("T2M batch is empty")
    text_sources = _truncate_text_rows(
        tuple(sample.text_input_ids for sample in samples),
        cap=encoder_cap,
        eos_token_id=eos_token_id,
    )
    input_ids = _pad_rows(
        text_sources,
        pad_value=pad_token_id,
        cap=encoder_cap,
        name="T2M text source",
    )
    labels = _pad_rows(
        tuple(sample.record.input_ids for sample in samples),
        pad_value=-100,
        cap=target_cap,
        name="T2M molecule target",
    )
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(int(pad_token_id)).to(torch.long),
        "labels": labels,
        "view": "P2-T2M",
    }


def factorized_model_inputs_from_batch(
    batch: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Translate the cache/collator ABI into the final wrapper ABI once."""

    mapping = {
        "input_ids": "input_ids",
        "attention_mask": "attention_mask",
        "e3fp_input_ids": "e3fp_ids",
        "atom_mask": "e3fp_atom_mask",
        "atom_to_fragment": "atom_to_fragment",
        "fragment_mask": "fragment_mask",
        "fragment_to_carrier": "fragment_to_carrier",
        "identity_span_bounds": "identity_span_bounds",
        "endpoint_mask": "endpoint_mask",
        "endpoint_to_atom": "endpoint_to_atom",
        "endpoint_to_token": "endpoint_to_token",
        "endpoint_to_fragment": "endpoint_to_fragment",
        "endpoint_is_explicit": "endpoint_is_explicit",
        "token_is_connector_endpoint": "connector_endpoint_mask",
        "atom_is_attachment": "atom_is_attachment",
        "fragment_geometry_mask": "fragment_geometry_mask",
        "endpoint_geometry_mask": "endpoint_geometry_mask",
        "labels": "labels",
    }
    result: dict[str, torch.Tensor] = {}
    for model_name, batch_name in mapping.items():
        value = batch.get(batch_name)
        if not isinstance(value, torch.Tensor):
            raise FragSmilesPairedPretrainingError(
                f"factorized batch lacks tensor {batch_name}"
            )
        result[model_name] = value
    return result


__all__ = [
    "FragSmilesPairedPretrainingError",
    "Phase2PairedSample",
    "collate_phase2_cap_samples",
    "collate_phase2_t2m_samples",
    "factorized_model_inputs_from_batch",
]
