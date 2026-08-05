"""Bridge the synthetic P1 runtime objects to auditable contracts and T5 CE.

The three layers are deliberately separate:

* :class:`BoundRecord` is the immutable, uncorrupted L/M/A binding;
* :class:`CEFirstExample` is one epoch-keyed corruption realization;
* :class:`PaddedCEBatch` contains only the three arrays accepted by the
  standard T5 conditional-generation forward path.

``materialize_training_record`` creates a JSON-compatible audit view of the
first two layers.  It never mutates the record and it cannot authorize P1.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from .bound_record import BoundRecord
from .ce_collator import CEFirstExample, CollatorContractError
from .hybrid_codec import HybridMotifCodec


TRAINING_RECORD_SCHEMA_VERSION = (
    "most-t5-r1/p1-logical-motif-training-record/vnext1"
)
TRAINING_RECORD_KIND = "logical_motif_training_record"
CE_FIRST_PROFILE = "ce_first"
LABEL_PAD_ID = -100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeBridgeError(ValueError):
    """Raised when independently valid layers cannot be bound safely."""


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise RuntimeBridgeError(f"{field} must be a lower-case SHA-256")


@dataclass(frozen=True)
class P1ArtifactBindings:
    """Hash locks needed by one runtime audit view."""

    release_id: str
    data_release_manifest_sha256: str
    geometry_record_schema_sha256: str
    geometry_record_content_sha256: str
    membership_manifest_sha256: str
    tokenizer_contract_sha256: str
    tokenizer_snapshot_sha256: str
    identity_codec_sha256: str
    connection_codec_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id.strip():
            raise RuntimeBridgeError("release_id must be nonempty")
        for field in (
            "data_release_manifest_sha256",
            "geometry_record_schema_sha256",
            "geometry_record_content_sha256",
            "membership_manifest_sha256",
            "tokenizer_contract_sha256",
            "tokenizer_snapshot_sha256",
            "identity_codec_sha256",
            "connection_codec_sha256",
        ):
            _require_sha256(getattr(self, field), field)

    def as_dict(self) -> dict[str, str]:
        return {
            "release_id": self.release_id,
            "data_release_manifest_sha256": self.data_release_manifest_sha256,
            "geometry_record_schema_sha256": self.geometry_record_schema_sha256,
            "geometry_record_content_sha256": self.geometry_record_content_sha256,
            "membership_manifest_sha256": self.membership_manifest_sha256,
            "tokenizer_contract_sha256": self.tokenizer_contract_sha256,
            "tokenizer_snapshot_sha256": self.tokenizer_snapshot_sha256,
            "identity_codec_sha256": self.identity_codec_sha256,
            "connection_codec_sha256": self.connection_codec_sha256,
        }


@dataclass(frozen=True)
class P1MemberRef:
    """Stable member identity; storage location is never inferred from order."""

    member_id: str
    storage_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.member_id, str) or not self.member_id.strip():
            raise RuntimeBridgeError("member_id must be nonempty")
        if not isinstance(self.storage_key, str) or not self.storage_key.strip():
            raise RuntimeBridgeError("storage_key must be nonempty")

    def as_dict(self) -> dict[str, str]:
        return {"member_id": self.member_id, "storage_key": self.storage_key}


def materialize_training_record(
    *,
    record: BoundRecord,
    example: CEFirstExample,
    bindings: P1ArtifactBindings,
    member: P1MemberRef,
    codec: HybridMotifCodec,
    token_to_id: Mapping[str, int],
    sentinel_token_ids: Sequence[int],
    eos_token_id: int,
) -> dict[str, object]:
    """Create the JSON-compatible uncorrupted audit view for one CE example.

    The mask comes from ``example`` while token, motif, connection, atom and
    E3FP data come only from the validated uncorrupted ``record``.  This makes
    accidental persistence of corrupted input or decoder labels impossible.
    """

    record.validate(codec, token_to_id)
    example.validate_against(record, sentinel_token_ids, eos_token_id)
    if member.member_id != record.record_id:
        raise RuntimeBridgeError("member_id must equal the bound record_id")
    if bindings.tokenizer_snapshot_sha256 != record.token_table_sha256:
        raise RuntimeBridgeError(
            "tokenizer_snapshot_sha256 must bind the exact BoundRecord token table"
        )

    bonds = []
    for connection in record.cross_motif_connections:
        bonds.append(
            {
                "edge_id": connection.edge_id,
                "left": {
                    "logical_motif_index": connection.endpoint_a.logical_motif_id,
                    "atom_index": connection.endpoint_a.atom_index,
                    "slot_ordinal": connection.endpoint_a.slot_id,
                },
                "right": {
                    "logical_motif_index": connection.endpoint_b.logical_motif_id,
                    "atom_index": connection.endpoint_b.atom_index,
                    "slot_ordinal": connection.endpoint_b.slot_id,
                },
                "bond_type": connection.bond_type,
            }
        )

    return {
        "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
        "document_kind": TRAINING_RECORD_KIND,
        "training_profile": CE_FIRST_PROFILE,
        "bindings": bindings.as_dict(),
        "member": member.as_dict(),
        "dimensions": {
            "token_count": len(record.input_ids),
            "logical_motif_count": len(record.identity_spans),
            "atom_count": len(record.atom_to_logical_motif),
            "source_atom_count": record.source_atom_count,
            "e3fp_level_count": len(record.full_e3fp_ids[0]),
        },
        "token_domain": {
            "input_ids": list(record.input_ids),
            "attention_mask": [True] * len(record.input_ids),
            "token_to_logical_motif": list(record.token_to_logical_motif),
            "token_role": list(record.token_role),
        },
        "logical_motif_domain": {
            "identity_spans": [
                [span.start, span.stop] for span in record.identity_spans
            ],
            "connection_token_indices": [
                list(span.indices()) for span in record.connection_spans
            ],
            "logical_to_carrier": list(record.logical_to_carrier),
            "exact_identity_sha256": list(record.exact_identity_digest),
            "motif_geometry_valid": list(record.motif_geometry_valid),
            "motif_atom_indices": [list(row) for row in record.motif_atom_indices],
            "motif_slot_atom_indices": [
                list(row) for row in record.motif_slot_atom_indices
            ],
            "slot_count": [len(row) for row in record.motif_slot_atom_indices],
            "cross_motif_bonds": bonds,
        },
        "atom_domain": {
            "atom_to_logical_motif": list(record.atom_to_logical_motif),
            "model_to_source_atom_index": list(record.model_to_source_atom_index),
            "atom_valid_mask": list(record.atom_valid_mask),
            "atom_is_attachment": list(record.atom_is_attachment),
            "full_e3fp_ids": [list(row) for row in record.full_e3fp_ids],
        },
        "masks": {
            "identity_recovery_mask": list(example.identity_recovery_mask),
        },
        "mask_decision": {
            "objective": example.objective,
            "seed": example.seed,
            "epoch": example.epoch,
            "mask_probability": example.mask_probability,
            "selected_logical_motif_indices": [
                motif_id
                for motif_id, selected in enumerate(example.identity_recovery_mask)
                if selected
            ],
            "decision_sha256": example.mask_decision_sha256,
        },
    }


@dataclass(frozen=True)
class PaddedCEBatch:
    """Right-padded pure-Python batch with a strict model input allowlist."""

    record_ids: tuple[str, ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    input_lengths: tuple[int, ...]
    target_lengths: tuple[int, ...]

    def model_inputs(self) -> dict[str, tuple[tuple[object, ...], ...]]:
        """Return only names accepted by standard T5 CE forward."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


def pad_ce_first_batch(
    examples: Sequence[CEFirstExample],
    *,
    pad_token_id: int,
    label_pad_id: int = LABEL_PAD_ID,
) -> PaddedCEBatch:
    """Right-pad CE examples without importing torch or leaking audit fields."""

    rows = tuple(examples)
    if not rows:
        raise RuntimeBridgeError("a CE batch must contain at least one example")
    for field, value in (("pad_token_id", pad_token_id), ("label_pad_id", label_pad_id)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeBridgeError(f"{field} must be an integer")
    if pad_token_id < 0:
        raise RuntimeBridgeError("pad_token_id must be nonnegative")
    if label_pad_id != LABEL_PAD_ID:
        raise RuntimeBridgeError("standard T5 CE requires label padding at -100")

    record_ids = tuple(row.record_id for row in rows)
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeBridgeError("a batch cannot contain duplicate record_ids")
    if any(not row.input_ids or not row.labels for row in rows):
        raise RuntimeBridgeError("every CE example needs nonempty input and labels")
    if any(pad_token_id in row.input_ids for row in rows):
        raise RuntimeBridgeError("unpadded CE input already contains pad_token_id")

    input_lengths = tuple(len(row.input_ids) for row in rows)
    target_lengths = tuple(len(row.labels) for row in rows)
    max_input = max(input_lengths)
    max_target = max(target_lengths)
    padded_inputs = tuple(
        row.input_ids + (pad_token_id,) * (max_input - len(row.input_ids))
        for row in rows
    )
    attention = tuple(
        (True,) * len(row.input_ids) + (False,) * (max_input - len(row.input_ids))
        for row in rows
    )
    padded_labels = tuple(
        row.labels + (label_pad_id,) * (max_target - len(row.labels))
        for row in rows
    )
    return PaddedCEBatch(
        record_ids=record_ids,
        input_ids=padded_inputs,
        attention_mask=attention,
        labels=padded_labels,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )
