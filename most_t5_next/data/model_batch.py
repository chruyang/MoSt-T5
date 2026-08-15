"""Translate cache/collator fields into the public MoSt-T5 model interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from most_t5_next.interfaces import (
    GEOMETRY_INPUT_NAMES,
    OPTIONAL_GEOMETRY_INPUT_NAMES,
    REQUIRED_GEOMETRY_INPUT_NAMES,
)


REQUIRED_BASE_FIELDS = ("input_ids", "attention_mask")
CACHE_ALIASES = {
    "atom_mask": "e3fp_atom_mask",
    "token_is_connector_endpoint": "connector_endpoint_mask",
}
def _tensor(batch: Mapping[str, Any], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"batch lacks tensor {name}")
    return value


def text_model_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Keep only fields accepted by a plain T5 forward or generation call."""

    result = {name: _tensor(batch, name) for name in REQUIRED_BASE_FIELDS}
    labels = batch.get("labels")
    if labels is not None:
        if not isinstance(labels, Tensor):
            raise ValueError("batch field labels is not a tensor")
        result["labels"] = labels
    return result


def model_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Map text, molecular, joint, or mixed rows onto one stable model ABI.

    Geometry is optional at the batch level.  Once any structural field is
    present, the complete required structural schema must be present.  Rows
    without geometry stay in the same batch by using false masks and E3FP
    values of ``-1``.
    """

    result = text_model_batch(batch)
    present: set[str] = set()
    for model_name in GEOMETRY_INPUT_NAMES:
        cache_name = model_name
        value = batch.get(model_name)
        if value is None and model_name in CACHE_ALIASES:
            cache_name = CACHE_ALIASES[model_name]
            value = batch.get(cache_name)
        if value is None:
            continue
        if not isinstance(value, Tensor):
            raise ValueError(f"batch field {cache_name} is not a tensor")
        result[model_name] = value
        present.add(model_name)

    if present:
        missing = sorted(REQUIRED_GEOMETRY_INPUT_NAMES - present)
        if missing:
            raise ValueError(
                "partial molecular batch is missing: " + ", ".join(missing)
            )
    return result


def molecular_model_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Validate that a batch contains the molecular schema."""

    result = model_batch(batch)
    if not REQUIRED_GEOMETRY_INPUT_NAMES.issubset(result):
        raise ValueError("batch does not contain molecular structure fields")
    return result


def disable_geometry(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return a shallow batch copy whose E3FP payload is explicitly absent."""

    result = dict(batch)
    e3fp_ids = result.get("e3fp_ids")
    if e3fp_ids is not None:
        result["e3fp_ids"] = e3fp_ids.new_full(e3fp_ids.shape, -1)
    return result


__all__ = [
    "disable_geometry",
    "model_batch",
    "molecular_model_batch",
    "text_model_batch",
]
