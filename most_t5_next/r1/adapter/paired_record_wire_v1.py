"""Canonical JSON wire for one validated P1 atom/motif training pair.

Only the two training documents, their shared lineage receipt, bounded source
indices and a small diagnostic surface summary cross this boundary.  The
in-memory graph encoding is deliberately not serialized: the complete M
document is validated again by :func:`load_production_motif_record`, while the
A artifact is recomputed from its JSON semantic fields with the same producer
hash function used at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from most_t5_next.p1.atom_production_bridge import ProductionAtomSelfiesRecord
from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    load_production_motif_record,
)
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.r1.adapter.production_paired_identity_records_v1 import (
    ProductionPairedIdentityRecords,
    ProductionPairReceipt,
    _atom_record_artifact_sha256,
)


PAIRED_RECORD_WIRE_SCHEMA = "most-t5-next/p1-paired-training-wire/v1"

_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "schedule_index",
        "sdf_record_index",
        "atom_document",
        "motif_training_document",
        "receipt",
        "surface_summary",
    }
)
_ATOM_FIELDS = frozenset(
    {
        "schema_version",
        "record_artifact_sha256",
        "record_id",
        "storage_key",
        "release_id",
        "geometry_record_content_sha256",
        "union_tokenizer_contract_sha256",
        "union_tokenizer_snapshot_sha256",
        "selfies",
        "input_ids",
        "token_to_atom",
        "token_role",
        "atom_identity_spans",
        "atom_to_carrier",
        "source_atom_count",
        "model_to_source_atom_index",
        "full_e3fp_ids",
        "atom_valid_mask",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "member_id",
        "storage_key",
        "release_id",
        "base_geometry_record_content_sha256",
        "effective_inherited_overlay_content_sha256",
        "strict_isomeric_identity",
    }
)
_SURFACE_SUMMARY_FIELDS = frozenset(
    {
        "atom_input_token_count",
        "motif_input_token_count",
        "motif_identity_modes",
        "motif_identity_token_counts",
        "graph_token_count",
        "cross_motif_connection_count",
    }
)


class PairedRecordWireError(ValueError):
    """A persisted paired row is not the exact validated training pair."""


def _plain_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PairedRecordWireError(f"{field} must be a nonnegative integer")
    return value


def _require_exact_fields(value: object, expected: frozenset[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PairedRecordWireError(f"{field} must be one JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PairedRecordWireError(
            f"{field} fields differ (missing={missing}, extra={extra})"
        )
    return value


def _assert_json_tree(value: object, field: str = "$", *, decoded: bool) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PairedRecordWireError(f"{field} contains NaN or infinity")
        return
    sequence_types = (list,) if decoded else (list, tuple)
    if isinstance(value, sequence_types):
        for index, item in enumerate(value):
            _assert_json_tree(item, f"{field}[{index}]", decoded=decoded)
        return
    mapping_types = (dict,) if decoded else (dict, Mapping)
    if isinstance(value, mapping_types):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PairedRecordWireError(f"{field} has a non-string JSON key")
            _assert_json_tree(item, f"{field}.{key}", decoded=decoded)
        return
    raise PairedRecordWireError(f"{field} is not JSON-safe")


def _canonical_json_bytes(value: object) -> bytes:
    _assert_json_tree(value, decoded=False)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PairedRecordWireError("wire value is not canonical JSON-safe data") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairedRecordWireError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise PairedRecordWireError(f"non-finite JSON constant {value!r} is forbidden")


@dataclass(frozen=True)
class PairedSurfaceSummary:
    """Non-semantic diagnostics retained without graph or identity payloads."""

    atom_input_token_count: int
    motif_input_token_count: int
    motif_identity_modes: tuple[str, ...]
    motif_identity_token_counts: tuple[int, ...]
    graph_token_count: int
    cross_motif_connection_count: int

    def __post_init__(self) -> None:
        for field in (
            "atom_input_token_count",
            "motif_input_token_count",
            "graph_token_count",
            "cross_motif_connection_count",
        ):
            _plain_nonnegative_int(getattr(self, field), field)
        if self.atom_input_token_count == 0 or self.motif_input_token_count == 0:
            raise PairedRecordWireError("surface token counts must be positive")
        if not isinstance(self.motif_identity_modes, tuple) or not isinstance(
            self.motif_identity_token_counts, tuple
        ):
            raise PairedRecordWireError("surface-summary arrays must be immutable tuples")
        if len(self.motif_identity_modes) != len(self.motif_identity_token_counts):
            raise PairedRecordWireError("surface-summary motif arrays disagree")
        if any(mode not in {"macro", "fallback"} for mode in self.motif_identity_modes):
            raise PairedRecordWireError("unknown motif identity surface mode")
        if any(
            _plain_nonnegative_int(count, "motif_identity_token_counts") == 0
            for count in self.motif_identity_token_counts
        ):
            raise PairedRecordWireError("motif identity surfaces must contain tokens")

    def as_dict(self) -> dict[str, object]:
        return {
            "atom_input_token_count": self.atom_input_token_count,
            "motif_input_token_count": self.motif_input_token_count,
            "motif_identity_modes": list(self.motif_identity_modes),
            "motif_identity_token_counts": list(self.motif_identity_token_counts),
            "graph_token_count": self.graph_token_count,
            "cross_motif_connection_count": self.cross_motif_connection_count,
        }


@dataclass(frozen=True)
class LoadedPairedTrainingRecord:
    """Lightweight immutable row returned by the JSON Dataset boundary."""

    schedule_index: int
    sdf_record_index: int
    atom_record: ProductionAtomSelfiesRecord
    motif_record: ProductionMotifRecord
    receipt: ProductionPairReceipt
    surface_summary: PairedSurfaceSummary

    def __post_init__(self) -> None:
        _plain_nonnegative_int(self.schedule_index, "schedule_index")
        _plain_nonnegative_int(self.sdf_record_index, "sdf_record_index")
        if not isinstance(self.atom_record, ProductionAtomSelfiesRecord):
            raise PairedRecordWireError("loaded A record has an unknown type")
        if not isinstance(self.motif_record, ProductionMotifRecord):
            raise PairedRecordWireError("loaded M record has an unknown type")
        if not isinstance(self.receipt, ProductionPairReceipt):
            raise PairedRecordWireError("loaded receipt has an unknown type")
        if not isinstance(self.surface_summary, PairedSurfaceSummary):
            raise PairedRecordWireError("loaded surface summary has an unknown type")

        atom = self.atom_record
        motif = self.motif_record
        receipt = self.receipt
        if not all(
            (
                atom.record_id == motif.record_id == receipt.member_id,
                atom.storage_key == motif.storage_key == receipt.storage_key,
                atom.release_id == motif.release_id == receipt.release_id,
                atom.geometry_record_content_sha256
                == motif.geometry_record_content_sha256
                == receipt.effective_inherited_overlay_content_sha256,
                atom.union_tokenizer_contract_sha256 == motif.tokenizer_contract_sha256,
                atom.union_tokenizer_snapshot_sha256 == motif.tokenizer_snapshot_sha256,
                atom.source_atom_count == motif.source_atom_count,
                atom.model_to_source_atom_index == motif.model_to_source_atom_index,
                atom.full_e3fp_ids == motif.full_e3fp_ids,
                atom.atom_valid_mask == motif.atom_valid_mask,
            )
        ):
            raise PairedRecordWireError(
                "loaded A/M/receipt member, tokenizer, source or effective geometry parity failed"
            )

        summary = self.surface_summary
        span_lengths = tuple(span.stop - span.start for span in motif.identity_spans)
        inferred_graph_tokens = len(motif.input_ids) - 2 - sum(span_lengths)
        endpoint_marker_count = sum(
            len(row) for row in motif.connection_token_indices
        )
        if (
            summary.atom_input_token_count != len(atom.input_ids)
            or summary.motif_input_token_count != len(motif.input_ids)
            or summary.motif_identity_token_counts != span_lengths
            or len(summary.motif_identity_modes) != len(motif.identity_spans)
            or summary.graph_token_count != inferred_graph_tokens
        ):
            raise PairedRecordWireError("surface summary disagrees with the loaded A/M records")
        if (
            endpoint_marker_count % 2 != 0
            or endpoint_marker_count
            != 2 * summary.cross_motif_connection_count
        ):
            raise PairedRecordWireError(
                "surface summary cross-motif connection count disagrees with M endpoint markers"
            )


def _atom_document(record: ProductionAtomSelfiesRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "record_artifact_sha256": record.record_artifact_sha256,
        "record_id": record.record_id,
        "storage_key": record.storage_key,
        "release_id": record.release_id,
        "geometry_record_content_sha256": record.geometry_record_content_sha256,
        "union_tokenizer_contract_sha256": record.union_tokenizer_contract_sha256,
        "union_tokenizer_snapshot_sha256": record.union_tokenizer_snapshot_sha256,
        "selfies": record.selfies,
        "input_ids": list(record.input_ids),
        "token_to_atom": list(record.token_to_atom),
        "token_role": list(record.token_role),
        "atom_identity_spans": [
            [span.start, span.stop] for span in record.atom_identity_spans
        ],
        "atom_to_carrier": list(record.atom_to_carrier),
        "source_atom_count": record.source_atom_count,
        "model_to_source_atom_index": list(record.model_to_source_atom_index),
        "full_e3fp_ids": [list(row) for row in record.full_e3fp_ids],
        "atom_valid_mask": list(record.atom_valid_mask),
    }


def _receipt_document(receipt: ProductionPairReceipt) -> dict[str, str]:
    return {
        "member_id": receipt.member_id,
        "storage_key": receipt.storage_key,
        "release_id": receipt.release_id,
        "base_geometry_record_content_sha256": receipt.base_geometry_record_content_sha256,
        "effective_inherited_overlay_content_sha256": (
            receipt.effective_inherited_overlay_content_sha256
        ),
        "strict_isomeric_identity": receipt.strict_isomeric_identity,
    }


def paired_record_to_document(
    pair: ProductionPairedIdentityRecords,
    *,
    schedule_index: int,
    sdf_record_index: int,
) -> dict[str, object]:
    """Materialize one exact JSON envelope from a validated producer pair."""

    if not isinstance(pair, ProductionPairedIdentityRecords):
        raise PairedRecordWireError("pair must be ProductionPairedIdentityRecords")
    schedule_index = _plain_nonnegative_int(schedule_index, "schedule_index")
    sdf_record_index = _plain_nonnegative_int(sdf_record_index, "sdf_record_index")
    motif_document_text = getattr(pair, "motif_document_canonical_json", None)
    if not isinstance(motif_document_text, str) or not motif_document_text:
        raise PairedRecordWireError(
            "pair does not retain its canonical motif_document_canonical_json"
        )
    try:
        motif_document = json.loads(
            motif_document_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
        if (
            not isinstance(motif_document, dict)
            or _canonical_json_bytes(motif_document).decode("utf-8")
            != motif_document_text
        ):
            raise PairedRecordWireError(
                "producer-retained M document text is not canonical JSON"
            )
        reloaded_motif = load_production_motif_record(motif_document)
    except Exception as exc:
        if isinstance(exc, PairedRecordWireError):
            raise
        raise PairedRecordWireError("producer-retained M document failed validation") from exc
    if reloaded_motif != pair.motif_record:
        raise PairedRecordWireError("producer-retained M document differs from motif_record")

    summary = PairedSurfaceSummary(
        atom_input_token_count=len(pair.atom_record.input_ids),
        motif_input_token_count=len(pair.motif_record.input_ids),
        motif_identity_modes=tuple(surface.mode for surface in pair.motif_identity_surfaces),
        motif_identity_token_counts=tuple(
            len(surface.tokens) for surface in pair.motif_identity_surfaces
        ),
        graph_token_count=len(pair.graph_encoding.graph_token_stream.tokens),
        cross_motif_connection_count=len(pair.graph_encoding.connections),
    )
    return {
        "schema_version": PAIRED_RECORD_WIRE_SCHEMA,
        "schedule_index": schedule_index,
        "sdf_record_index": sdf_record_index,
        "atom_document": _atom_document(pair.atom_record),
        "motif_training_document": motif_document,
        "receipt": _receipt_document(pair.receipt),
        "surface_summary": summary.as_dict(),
    }


def encode_paired_training_record(
    pair: ProductionPairedIdentityRecords,
    *,
    schedule_index: int,
    sdf_record_index: int,
) -> bytes:
    """Encode one pair as deterministic UTF-8 canonical JSON bytes."""

    return _canonical_json_bytes(
        paired_record_to_document(
            pair,
            schedule_index=schedule_index,
            sdf_record_index=sdf_record_index,
        )
    )


def _load_atom_record(
    raw: object,
    motif_document: dict[str, Any],
) -> ProductionAtomSelfiesRecord:
    document = _require_exact_fields(raw, _ATOM_FIELDS, "atom_document")
    raw_spans = document["atom_identity_spans"]
    if not isinstance(raw_spans, list):
        raise PairedRecordWireError("atom_identity_spans must be one JSON array")
    spans: list[Span] = []
    for index, row in enumerate(raw_spans):
        if not isinstance(row, list) or len(row) != 2:
            raise PairedRecordWireError(
                f"atom_identity_spans[{index}] must be [start, stop]"
            )
        try:
            spans.append(Span(row[0], row[1]))
        except Exception as exc:
            raise PairedRecordWireError(
                f"atom_identity_spans[{index}] is invalid"
            ) from exc

    for field in (
        "input_ids",
        "token_to_atom",
        "token_role",
        "atom_to_carrier",
        "model_to_source_atom_index",
        "full_e3fp_ids",
        "atom_valid_mask",
    ):
        if not isinstance(document[field], list):
            raise PairedRecordWireError(f"atom_document.{field} must be one JSON array")
    if any(not isinstance(row, list) for row in document["full_e3fp_ids"]):
        raise PairedRecordWireError("atom_document.full_e3fp_ids rows must be JSON arrays")

    try:
        record = ProductionAtomSelfiesRecord(
            schema_version=document["schema_version"],
            record_artifact_sha256=document["record_artifact_sha256"],
            record_id=document["record_id"],
            storage_key=document["storage_key"],
            release_id=document["release_id"],
            geometry_record_content_sha256=document["geometry_record_content_sha256"],
            union_tokenizer_contract_sha256=document[
                "union_tokenizer_contract_sha256"
            ],
            union_tokenizer_snapshot_sha256=document[
                "union_tokenizer_snapshot_sha256"
            ],
            selfies=document["selfies"],
            input_ids=tuple(document["input_ids"]),
            token_to_atom=tuple(document["token_to_atom"]),
            token_role=tuple(document["token_role"]),
            atom_identity_spans=tuple(spans),
            atom_to_carrier=tuple(document["atom_to_carrier"]),
            source_atom_count=document["source_atom_count"],
            model_to_source_atom_index=tuple(document["model_to_source_atom_index"]),
            full_e3fp_ids=tuple(tuple(row) for row in document["full_e3fp_ids"]),
            atom_valid_mask=tuple(document["atom_valid_mask"]),
        )
    except Exception as exc:
        raise PairedRecordWireError("A document failed semantic validation") from exc

    try:
        raw_bindings = dict(motif_document["bindings"])
        raw_bindings.update(
            {
                "release_id": record.release_id,
                "geometry_record_content_sha256": record.geometry_record_content_sha256,
                "tokenizer_contract_sha256": record.union_tokenizer_contract_sha256,
                "tokenizer_snapshot_sha256": record.union_tokenizer_snapshot_sha256,
            }
        )
        bindings = P1ArtifactBindings(**raw_bindings)
        member = P1MemberRef(record.record_id, record.storage_key)
        recomputed = _atom_record_artifact_sha256(
            member=member,
            bindings=bindings,
            alignment=record,
            source_atom_count=record.source_atom_count,
            model_to_source_atom_index=record.model_to_source_atom_index,
            inherited_e3fp=record.full_e3fp_ids,
        )
    except Exception as exc:
        raise PairedRecordWireError("A artifact inputs could not be reconstructed") from exc
    if recomputed != record.record_artifact_sha256:
        raise PairedRecordWireError("A record artifact hash does not match its JSON semantics")
    return record


def _load_receipt(raw: object) -> ProductionPairReceipt:
    document = _require_exact_fields(raw, _RECEIPT_FIELDS, "receipt")
    try:
        return ProductionPairReceipt(**document)
    except Exception as exc:
        raise PairedRecordWireError("paired receipt failed validation") from exc


def _load_surface_summary(raw: object) -> PairedSurfaceSummary:
    document = _require_exact_fields(raw, _SURFACE_SUMMARY_FIELDS, "surface_summary")
    for field in ("motif_identity_modes", "motif_identity_token_counts"):
        if not isinstance(document[field], list):
            raise PairedRecordWireError(f"surface_summary.{field} must be one JSON array")
    try:
        return PairedSurfaceSummary(
            atom_input_token_count=document["atom_input_token_count"],
            motif_input_token_count=document["motif_input_token_count"],
            motif_identity_modes=tuple(document["motif_identity_modes"]),
            motif_identity_token_counts=tuple(document["motif_identity_token_counts"]),
            graph_token_count=document["graph_token_count"],
            cross_motif_connection_count=document["cross_motif_connection_count"],
        )
    except Exception as exc:
        if isinstance(exc, PairedRecordWireError):
            raise
        raise PairedRecordWireError("surface summary failed validation") from exc


def load_paired_training_record(document: Mapping[str, object]) -> LoadedPairedTrainingRecord:
    """Validate one decoded JSON object and return immutable A/M training rows."""

    _assert_json_tree(document, decoded=True)
    envelope = _require_exact_fields(document, _ENVELOPE_FIELDS, "paired envelope")
    if envelope["schema_version"] != PAIRED_RECORD_WIRE_SCHEMA:
        raise PairedRecordWireError("unknown paired-record wire schema")
    schedule_index = _plain_nonnegative_int(envelope["schedule_index"], "schedule_index")
    sdf_record_index = _plain_nonnegative_int(
        envelope["sdf_record_index"], "sdf_record_index"
    )
    motif_document = envelope["motif_training_document"]
    if not isinstance(motif_document, dict):
        raise PairedRecordWireError("motif_training_document must be one JSON object")
    try:
        motif_record = load_production_motif_record(motif_document)
    except Exception as exc:
        raise PairedRecordWireError("M document failed production vNext validation") from exc
    atom_record = _load_atom_record(envelope["atom_document"], motif_document)
    receipt = _load_receipt(envelope["receipt"])
    summary = _load_surface_summary(envelope["surface_summary"])

    loaded = LoadedPairedTrainingRecord(
        schedule_index=schedule_index,
        sdf_record_index=sdf_record_index,
        atom_record=atom_record,
        motif_record=motif_record,
        receipt=receipt,
        surface_summary=summary,
    )
    raw_cross_bonds = motif_document["logical_motif_domain"]["cross_motif_bonds"]
    if loaded.surface_summary.cross_motif_connection_count != len(raw_cross_bonds):
        raise PairedRecordWireError(
            "surface summary cross-motif connection count disagrees with M document"
        )
    return loaded


def decode_paired_training_record(payload: bytes) -> LoadedPairedTrainingRecord:
    """Decode canonical bytes; non-canonical or ambiguous JSON is rejected."""

    if not isinstance(payload, bytes):
        raise PairedRecordWireError("paired-record payload must be bytes")
    try:
        text = payload.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except PairedRecordWireError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PairedRecordWireError("paired-record payload is not valid UTF-8 JSON") from exc
    _assert_json_tree(document, decoded=True)
    if _canonical_json_bytes(document) != payload:
        raise PairedRecordWireError("paired-record payload is not canonical JSON")
    return load_paired_training_record(document)


__all__ = [
    "LoadedPairedTrainingRecord",
    "PAIRED_RECORD_WIRE_SCHEMA",
    "PairedRecordWireError",
    "PairedSurfaceSummary",
    "decode_paired_training_record",
    "encode_paired_training_record",
    "load_paired_training_record",
    "paired_record_to_document",
]
