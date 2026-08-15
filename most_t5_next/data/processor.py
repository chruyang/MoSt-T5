"""Public input preparation for text and fragSMILES examples.

The processor keeps modality decisions in the data.  A molecular row carries
the same structural address tensors whether geometry is available or not; an
all-minus-one E3FP payload disables geometry for that row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


class InputPreparationError(ValueError):
    """Raised when text and molecular inputs cannot be aligned safely."""


def _tuple_1d(value: Any, name: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise InputPreparationError(f"{name} must be one-dimensional")
    return tuple(int(item) for item in array.tolist())


def _tuple_2d(value: Any, columns: int, name: str) -> tuple[tuple[int, ...], ...]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns:
        raise InputPreparationError(f"{name} must have shape [N,{columns}]")
    return tuple(tuple(int(item) for item in row) for row in array.tolist())


@dataclass(frozen=True)
class MolecularInput:
    """One token-aligned fragSMILES representation.

    The class intentionally contains only fields consumed by ``MoStT5``.  It
    accepts the formal mmap cache records through :meth:`from_cache_record`
    without making the public model depend on the cache implementation.
    """

    input_ids: tuple[int, ...]
    e3fp_ids: tuple[tuple[int, int, int, int], ...]
    atom_to_fragment: tuple[int, ...]
    fragment_to_carrier: tuple[int, ...]
    identity_span_bounds: tuple[tuple[int, int], ...]
    endpoint_to_atom: tuple[int, ...]
    endpoint_to_token: tuple[int, ...]
    endpoint_to_fragment: tuple[int, ...]
    endpoint_is_explicit: tuple[bool, ...]
    token_is_connector_endpoint: tuple[bool, ...]
    atom_is_attachment: tuple[bool, ...]

    def __post_init__(self) -> None:
        tokens = len(self.input_ids)
        atoms = len(self.e3fp_ids)
        fragments = len(self.fragment_to_carrier)
        endpoints = len(self.endpoint_to_atom)
        if not tokens:
            raise InputPreparationError("molecular input cannot be empty")
        if len(self.token_is_connector_endpoint) != tokens:
            raise InputPreparationError("connector mask must match molecular tokens")
        if len(self.atom_to_fragment) != atoms or len(self.atom_is_attachment) != atoms:
            raise InputPreparationError("atom fields must share one axis")
        if len(self.identity_span_bounds) != fragments:
            raise InputPreparationError("fragment fields must share one axis")
        if not (
            len(self.endpoint_to_token)
            == len(self.endpoint_to_fragment)
            == len(self.endpoint_is_explicit)
            == endpoints
        ):
            raise InputPreparationError("endpoint fields must share one axis")
        for start, stop in self.identity_span_bounds:
            if not 0 <= start < stop <= tokens:
                raise InputPreparationError("fragment span lies outside molecular tokens")
        if any(not 0 <= carrier < tokens for carrier in self.fragment_to_carrier):
            raise InputPreparationError("fragment carrier lies outside molecular tokens")
        if fragments:
            if any(not 0 <= owner < fragments for owner in self.atom_to_fragment):
                raise InputPreparationError("atom owner lies outside fragment axis")
        elif any(owner != -1 for owner in self.atom_to_fragment):
            raise InputPreparationError("whole-molecule atoms must use owner -1")
        for atom, token, fragment in zip(
            self.endpoint_to_atom, self.endpoint_to_token, self.endpoint_to_fragment
        ):
            if not 0 <= atom < atoms or not 0 <= token < tokens:
                raise InputPreparationError("endpoint address is out of range")
            if not 0 <= fragment < fragments:
                raise InputPreparationError("endpoint fragment is out of range")
            if not self.atom_is_attachment[atom]:
                raise InputPreparationError("endpoint must address an attachment atom")
            if self.atom_to_fragment[atom] != fragment:
                raise InputPreparationError("endpoint and atom ownership disagree")

    @classmethod
    def from_cache_record(
        cls, record: Any, *, use_geometry: bool = True
    ) -> "MolecularInput":
        """Create a public input from a formal ``CachedFragSmilesRecord``."""

        endpoint_rows = np.asarray(record.endpoints)
        if endpoint_rows.ndim != 2 or endpoint_rows.shape[1] != 6:
            raise InputPreparationError("cached endpoints must have shape [E,6]")
        e3fp = np.asarray(record.e3fp)
        if e3fp.ndim != 2 or e3fp.shape[1] != 4:
            raise InputPreparationError("cached E3FP must have shape [A,4]")
        if not use_geometry:
            e3fp = np.full(e3fp.shape, -1, dtype=np.int64)
        connector_mask = np.zeros(len(record.input_ids), dtype=np.bool_)
        for endpoint in endpoint_rows:
            if bool(endpoint[5]):
                connector_mask[int(endpoint[4])] = True
        return cls(
            input_ids=_tuple_1d(record.input_ids, "input_ids"),
            e3fp_ids=_tuple_2d(e3fp, 4, "e3fp_ids"),
            atom_to_fragment=_tuple_1d(record.atom_to_fragment, "atom_to_fragment"),
            fragment_to_carrier=_tuple_1d(
                record.fragment_carriers, "fragment_to_carrier"
            ),
            identity_span_bounds=_tuple_2d(
                record.fragment_spans, 2, "identity_span_bounds"
            ),
            endpoint_to_atom=tuple(int(item) for item in endpoint_rows[:, 3]),
            endpoint_to_token=tuple(int(item) for item in endpoint_rows[:, 4]),
            endpoint_to_fragment=tuple(int(item) for item in endpoint_rows[:, 2]),
            endpoint_is_explicit=tuple(bool(item) for item in endpoint_rows[:, 5]),
            token_is_connector_endpoint=tuple(
                bool(item) for item in connector_mask
            ),
            atom_is_attachment=tuple(
                bool(item) for item in np.asarray(record.atom_is_attachment)
            ),
        )

    def without_geometry(self) -> "MolecularInput":
        """Keep topology and addresses while replacing E3FP with ``-1``."""

        return replace(
            self,
            e3fp_ids=tuple((-1, -1, -1, -1) for _ in self.e3fp_ids),
        )

    def with_text_prefix(self, prefix_ids: Sequence[int]) -> "MolecularInput":
        """Prepend text and shift every token address by the same amount."""

        prefix = tuple(int(item) for item in prefix_ids)
        shift = len(prefix)
        if not shift:
            return self
        return replace(
            self,
            input_ids=prefix + self.input_ids,
            fragment_to_carrier=tuple(item + shift for item in self.fragment_to_carrier),
            identity_span_bounds=tuple(
                (start + shift, stop + shift)
                for start, stop in self.identity_span_bounds
            ),
            endpoint_to_token=tuple(item + shift for item in self.endpoint_to_token),
            token_is_connector_endpoint=(False,) * shift
            + self.token_is_connector_endpoint,
        )


@dataclass(frozen=True)
class MoStT5Example:
    """One model example, with optional molecular alignment and target."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...] | None = None
    molecule: MolecularInput | None = None

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise InputPreparationError("input_ids cannot be empty")
        if self.molecule is not None and self.input_ids != self.molecule.input_ids:
            raise InputPreparationError("example and molecule token axes differ")
        if self.labels is not None and not self.labels:
            raise InputPreparationError("labels cannot be empty")


class MoStT5Processor:
    """Prepare text, molecular, and joint examples for one public collator."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ) -> None:
        if max_input_length <= 0 or max_target_length <= 0:
            raise InputPreparationError("length limits must be positive")
        self.tokenizer = tokenizer
        self.max_input_length = int(max_input_length)
        self.max_target_length = int(max_target_length)

    def _encode_text(self, text: str, *, limit: int) -> tuple[int, ...]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=limit,
        )
        ids = tuple(int(item) for item in encoded["input_ids"])
        if not ids:
            raise InputPreparationError("tokenizer returned an empty sequence")
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None and len(ids) == limit and ids[-1] != int(eos):
            ids = ids[:-1] + (int(eos),)
        return ids

    def text(self, text: str, *, target: str | None = None) -> MoStT5Example:
        labels = (
            self._encode_text(target, limit=self.max_target_length)
            if target is not None
            else None
        )
        return MoStT5Example(
            input_ids=self._encode_text(text, limit=self.max_input_length),
            labels=labels,
        )

    def molecule(
        self,
        molecule: MolecularInput,
        *,
        target: str | Sequence[int] | None = None,
        use_geometry: bool = True,
    ) -> MoStT5Example:
        molecular_input = molecule if use_geometry else molecule.without_geometry()
        if len(molecular_input.input_ids) > self.max_input_length:
            raise InputPreparationError(
                "molecular input exceeds max_input_length; structural token truncation is unsafe"
            )
        labels = self._target_ids(target)
        return MoStT5Example(
            input_ids=molecular_input.input_ids,
            labels=labels,
            molecule=molecular_input,
        )

    def joint(
        self,
        text: str,
        molecule: MolecularInput,
        *,
        target: str | Sequence[int] | None = None,
        use_geometry: bool = True,
    ) -> MoStT5Example:
        """Prepend truncated text without ever truncating molecular tokens."""

        if len(molecule.input_ids) > self.max_input_length:
            raise InputPreparationError(
                "molecular input exceeds max_input_length; structural token truncation is unsafe"
            )
        remaining = self.max_input_length - len(molecule.input_ids)
        if remaining == 0:
            combined = molecule
        else:
            combined = molecule.with_text_prefix(
                self._encode_text(text, limit=remaining)
            )
        if not use_geometry:
            combined = combined.without_geometry()
        return MoStT5Example(
            input_ids=combined.input_ids,
            labels=self._target_ids(target),
            molecule=combined,
        )

    def _target_ids(
        self, target: str | Sequence[int] | None
    ) -> tuple[int, ...] | None:
        if target is None:
            return None
        if isinstance(target, str):
            return self._encode_text(target, limit=self.max_target_length)
        ids = tuple(int(item) for item in target)
        if len(ids) > self.max_target_length:
            raise InputPreparationError(
                "token target exceeds max_target_length; structural token truncation is unsafe"
            )
        return ids


class MoStT5Collator:
    """Dynamically pad text-only, molecular, joint, or mixed examples."""

    def __init__(self, *, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, examples: Sequence[MoStT5Example]) -> dict[str, Tensor]:
        if not examples:
            raise InputPreparationError("cannot collate an empty batch")
        has_labels = [example.labels is not None for example in examples]
        if any(has_labels) and not all(has_labels):
            raise InputPreparationError("one batch cannot mix labeled and unlabeled rows")
        batch = len(examples)
        tokens = max(len(example.input_ids) for example in examples)
        input_ids = torch.full(
            (batch, tokens), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch, tokens), dtype=torch.bool)
        result: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        for row, example in enumerate(examples):
            width = len(example.input_ids)
            input_ids[row, :width] = torch.tensor(example.input_ids, dtype=torch.long)
            attention_mask[row, :width] = True
        if all(has_labels):
            label_width = max(len(example.labels or ()) for example in examples)
            labels = torch.full(
                (batch, label_width), self.label_pad_token_id, dtype=torch.long
            )
            for row, example in enumerate(examples):
                values = example.labels or ()
                labels[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            result["labels"] = labels

        molecules = [example.molecule for example in examples]
        if not any(molecule is not None for molecule in molecules):
            return result
        atoms = max((len(molecule.e3fp_ids) if molecule else 0) for molecule in molecules)
        fragments = max(
            (len(molecule.fragment_to_carrier) if molecule else 0)
            for molecule in molecules
        )
        endpoints = max(
            (len(molecule.endpoint_to_atom) if molecule else 0)
            for molecule in molecules
        )
        result.update(self._empty_geometry(batch, tokens, atoms, fragments, endpoints))
        for row, molecule in enumerate(molecules):
            if molecule is None:
                continue
            atom_count = len(molecule.e3fp_ids)
            fragment_count = len(molecule.fragment_to_carrier)
            endpoint_count = len(molecule.endpoint_to_atom)
            token_count = len(molecule.input_ids)
            result["e3fp_ids"][row, :atom_count] = torch.tensor(
                molecule.e3fp_ids, dtype=torch.long
            )
            result["atom_mask"][row, :atom_count] = True
            result["atom_to_fragment"][row, :atom_count] = torch.tensor(
                molecule.atom_to_fragment, dtype=torch.long
            )
            result["atom_is_attachment"][row, :atom_count] = torch.tensor(
                molecule.atom_is_attachment, dtype=torch.bool
            )
            result["fragment_mask"][row, :fragment_count] = True
            result["fragment_to_carrier"][row, :fragment_count] = torch.tensor(
                molecule.fragment_to_carrier, dtype=torch.long
            )
            result["identity_span_bounds"][row, :fragment_count] = torch.tensor(
                molecule.identity_span_bounds, dtype=torch.long
            )
            result["endpoint_mask"][row, :endpoint_count] = True
            for name in (
                "endpoint_to_atom",
                "endpoint_to_token",
                "endpoint_to_fragment",
                "endpoint_is_explicit",
            ):
                values = getattr(molecule, name)
                result[name][row, :endpoint_count] = torch.tensor(
                    values, dtype=result[name].dtype
                )
            result["token_is_connector_endpoint"][row, :token_count] = torch.tensor(
                molecule.token_is_connector_endpoint, dtype=torch.bool
            )
        return result

    @staticmethod
    def _empty_geometry(
        batch: int, tokens: int, atoms: int, fragments: int, endpoints: int
    ) -> dict[str, Tensor]:
        return {
            "e3fp_ids": torch.full((batch, atoms, 4), -1, dtype=torch.long),
            "atom_mask": torch.zeros((batch, atoms), dtype=torch.bool),
            "atom_to_fragment": torch.full((batch, atoms), -1, dtype=torch.long),
            "fragment_mask": torch.zeros((batch, fragments), dtype=torch.bool),
            "fragment_to_carrier": torch.full(
                (batch, fragments), -1, dtype=torch.long
            ),
            "identity_span_bounds": torch.full(
                (batch, fragments, 2), -1, dtype=torch.long
            ),
            "endpoint_mask": torch.zeros((batch, endpoints), dtype=torch.bool),
            "endpoint_to_atom": torch.full(
                (batch, endpoints), -1, dtype=torch.long
            ),
            "endpoint_to_token": torch.full(
                (batch, endpoints), -1, dtype=torch.long
            ),
            "endpoint_to_fragment": torch.full(
                (batch, endpoints), -1, dtype=torch.long
            ),
            "endpoint_is_explicit": torch.zeros(
                (batch, endpoints), dtype=torch.bool
            ),
            "token_is_connector_endpoint": torch.zeros(
                (batch, tokens), dtype=torch.bool
            ),
            "atom_is_attachment": torch.zeros((batch, atoms), dtype=torch.bool),
        }


__all__ = [
    "InputPreparationError",
    "MolecularInput",
    "MoStT5Collator",
    "MoStT5Example",
    "MoStT5Processor",
]
