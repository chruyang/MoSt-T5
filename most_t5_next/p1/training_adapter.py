"""Minimal tensor boundary between :class:`PaddedCEBatch` and standard T5.

Torch and Transformers are imported lazily so the data-contract tests remain
usable in a CPU-only environment without either package.  Record identifiers,
audit documents and artifact hashes deliberately stay outside the returned
``BatchEncoding``.
"""

from __future__ import annotations

from typing import Any, Mapping

from .runtime_bridge import LABEL_PAD_ID, PaddedCEBatch


MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "labels")
GEOMETRY_MODEL_INPUT_KEYS = (
    "e3fp_ids",
    "e3fp_atom_mask",
    "e3fp_atom_to_token",
)
FOUR_GRID_MODEL_INPUT_KEYS = MODEL_INPUT_KEYS + GEOMETRY_MODEL_INPUT_KEYS


class TrainingAdapterError(ValueError):
    """The padded CE batch cannot be represented as a standard T5 batch."""


def _validate_batch(batch: PaddedCEBatch) -> None:
    if not isinstance(batch, PaddedCEBatch):
        raise TrainingAdapterError("batch must be a PaddedCEBatch")

    batch_size = len(batch.record_ids)
    if batch_size == 0:
        raise TrainingAdapterError("batch must contain at least one record")
    fields = (
        batch.input_ids,
        batch.attention_mask,
        batch.labels,
        batch.input_lengths,
        batch.target_lengths,
    )
    if any(len(field) != batch_size for field in fields):
        raise TrainingAdapterError("all batch fields must share the batch dimension")

    input_widths = {len(row) for row in batch.input_ids}
    attention_widths = {len(row) for row in batch.attention_mask}
    label_widths = {len(row) for row in batch.labels}
    if len(input_widths) != 1 or 0 in input_widths:
        raise TrainingAdapterError("input_ids must be a nonempty rectangular matrix")
    if attention_widths != input_widths:
        raise TrainingAdapterError("attention_mask shape must equal input_ids shape")
    if len(label_widths) != 1 or 0 in label_widths:
        raise TrainingAdapterError("labels must be a nonempty rectangular matrix")

    input_width = next(iter(input_widths))
    label_width = next(iter(label_widths))
    for index in range(batch_size):
        input_length = batch.input_lengths[index]
        target_length = batch.target_lengths[index]
        if not 0 < input_length <= input_width:
            raise TrainingAdapterError("input_lengths are inconsistent with input_ids")
        if not 0 < target_length <= label_width:
            raise TrainingAdapterError("target_lengths are inconsistent with labels")

        mask = batch.attention_mask[index]
        if any(value not in (False, True, 0, 1) for value in mask):
            raise TrainingAdapterError("attention_mask values must be binary")
        if tuple(bool(value) for value in mask) != (
            (True,) * input_length + (False,) * (input_width - input_length)
        ):
            raise TrainingAdapterError("attention_mask must describe right padding")

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in batch.input_ids[index]
        ):
            raise TrainingAdapterError("input_ids must contain nonnegative integers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or (value < 0 and value != LABEL_PAD_ID)
            for value in batch.labels[index]
        ):
            raise TrainingAdapterError("labels must contain token IDs or -100")
        if LABEL_PAD_ID in batch.labels[index][:target_length]:
            raise TrainingAdapterError("-100 may appear only in label padding")
        if any(value != LABEL_PAD_ID for value in batch.labels[index][target_length:]):
            raise TrainingAdapterError("label padding must use -100")


def to_t5_batch_encoding(
    batch: PaddedCEBatch,
    *,
    device: object | None = None,
    torch_module: Any | None = None,
    batch_encoding_cls: Any | None = None,
) -> Any:
    """Convert a padded CE batch to three ``torch.long`` model tensors.

    ``torch_module`` and ``batch_encoding_cls`` are injectable only to keep the
    structural unit test independent of the optional training dependencies.
    Normal callers should leave both unset.
    """

    _validate_batch(batch)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - exercised remotely
            raise TrainingAdapterError("PyTorch is required for tensor conversion") from exc
    if batch_encoding_cls is None:
        try:
            from transformers.tokenization_utils_base import BatchEncoding
        except ImportError as exc:  # pragma: no cover - exercised remotely
            raise TrainingAdapterError("Transformers is required for BatchEncoding") from exc
        batch_encoding_cls = BatchEncoding

    python_inputs = batch.model_inputs()
    tensors = {
        key: torch_module.as_tensor(
            python_inputs[key], dtype=torch_module.long, device=device
        )
        for key in MODEL_INPUT_KEYS
    }
    return batch_encoding_cls(tensors)


def select_t5_forward_inputs(batch_encoding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict standard-T5 allowlist, ignoring all audit metadata."""

    missing = [key for key in MODEL_INPUT_KEYS if key not in batch_encoding]
    if missing:
        raise TrainingAdapterError(
            "BatchEncoding is missing model inputs: {}".format(", ".join(missing))
        )
    return {key: batch_encoding[key] for key in MODEL_INPUT_KEYS}


def to_four_grid_batch_encoding(
    batch: Any,
    *,
    device: object | None = None,
    torch_module: Any | None = None,
    batch_encoding_cls: Any | None = None,
) -> Any:
    """Convert one validated A0/A1/M0/M1 batch to wrapper tensor kwargs.

    CE-only cells return the ordinary three T5 tensors.  Geometry-enabled
    cells add exactly the three fields declared by :class:`FourGridT5Wrapper`;
    record IDs and ``model_to_source_atom_index`` remain audit provenance and
    never cross the model boundary.
    """

    from .experiment_grid import P1ConditionBatch

    if not isinstance(batch, P1ConditionBatch):
        raise TrainingAdapterError("batch must be a P1ConditionBatch")
    encoded = to_t5_batch_encoding(
        batch.ce_batch,
        device=device,
        torch_module=torch_module,
        batch_encoding_cls=batch_encoding_cls,
    )
    if batch.geometry is None:
        return encoded
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - exercised remotely
            raise TrainingAdapterError("PyTorch is required for tensor conversion") from exc
    geometry_values = batch.geometry.model_inputs()
    dtypes = {
        "e3fp_ids": torch_module.long,
        "e3fp_atom_mask": torch_module.bool,
        "e3fp_atom_to_token": torch_module.long,
    }
    for key in GEOMETRY_MODEL_INPUT_KEYS:
        encoded[key] = torch_module.as_tensor(
            geometry_values[key], dtype=dtypes[key], device=device
        )
    return encoded


def select_four_grid_forward_inputs(
    batch_encoding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the strict wrapper allowlist and enforce all-or-none geometry."""

    selected = select_t5_forward_inputs(batch_encoding)
    present = tuple(key in batch_encoding for key in GEOMETRY_MODEL_INPUT_KEYS)
    if any(present) and not all(present):
        raise TrainingAdapterError("four-grid geometry tensors are all-or-none")
    if all(present):
        selected.update(
            {key: batch_encoding[key] for key in GEOMETRY_MODEL_INPUT_KEYS}
        )
    return selected
