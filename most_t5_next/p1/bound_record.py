"""Synthetic three-domain ``BoundRecord`` candidate.

The record makes token (L), logical motif (M), and atom (A) indices explicit.
It is intentionally string-surface backed and marked synthetic; it cannot be
passed to the historical dataset/model or treated as a frozen P1 release.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from .hybrid_codec import (
    CodecContractError,
    CrossMotifConnection,
    FALLBACK_BEGIN,
    HybridMotifCodec,
    LogicalMoleculeSchema,
    LogicalMotif,
    LogicalMotifIdentity,
)


BOUND_RECORD_SCHEMA_VERSION = "most-t5-next/p1-bound-record-synthetic/v1"
MOLECULE_BEGIN = "<MOLECULE_BEGIN>"
MOLECULE_END = "<MOLECULE_END>"
CONNECTION_BEGIN = "<CONNECTION_BEGIN>"
CONNECTION_END = "<CONNECTION_END>"
TOKEN_ROLES = frozenset({"boundary", "identity", "connection"})


class BoundRecordInvariantError(ValueError):
    """Raised when a token/motif/atom binding is inconsistent."""


@dataclass(frozen=True)
class Span:
    """Half-open token-domain span."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop < self.start
        ):
            raise BoundRecordInvariantError("span must be a valid nonnegative half-open range")

    def indices(self) -> range:
        return range(self.start, self.stop)


def _connection_tokens(
    schema: LogicalMoleculeSchema, logical_motif_id: int
) -> tuple[str, ...]:
    incident: dict[int, CrossMotifConnection] = {}
    for connection in schema.connections:
        for endpoint in (connection.endpoint_a, connection.endpoint_b):
            if endpoint.logical_motif_id == logical_motif_id:
                incident[endpoint.slot_id] = connection
    if not incident:
        return ()

    tokens: list[str] = [CONNECTION_BEGIN]
    for slot_id in sorted(incident):
        connection = incident[slot_id]
        tokens.extend(
            (
                f"<SLOT:{slot_id:04d}>",
                f"<EDGE:{connection.edge_id:06d}>",
                f"<BOND:{connection.bond_type}>",
            )
        )
    tokens.append(CONNECTION_END)
    return tuple(tokens)


def build_synthetic_token_table(
    codec: HybridMotifCodec, schema: LogicalMoleculeSchema
) -> dict[str, int]:
    """Build one deterministic table covering both surfaces for a fixture.

    The returned table is not a production tokenizer.  Including macro and
    forced-fallback tokens in one table ensures tests compare surface encodings
    without silently changing token IDs.
    """

    schema.validate()
    tokens = {MOLECULE_BEGIN, MOLECULE_END}
    for motif in schema.motifs:
        tokens.update(codec.encode(motif.identity).tokens)
        tokens.update(codec.encode(motif.identity, force_fallback=True).tokens)
        tokens.update(_connection_tokens(schema, motif.logical_motif_id))
    return {token: token_id for token_id, token in enumerate(sorted(tokens))}


@dataclass(frozen=True)
class BoundRecord:
    """Immutable synthetic binding across L, M and A domains."""

    schema_version: str
    record_id: str
    training_admission: bool
    surface_modes: tuple[str, ...]
    surface_tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    token_to_logical_motif: tuple[int, ...]
    token_role: tuple[str, ...]
    identity_spans: tuple[Span, ...]
    connection_spans: tuple[Span, ...]
    logical_to_carrier: tuple[int, ...]
    motif_atom_indices: tuple[tuple[int, ...], ...]
    motif_slot_atom_indices: tuple[tuple[int, ...], ...]
    atom_to_logical_motif: tuple[int, ...]
    source_atom_count: int
    model_to_source_atom_index: tuple[int, ...]
    atom_is_attachment: tuple[bool, ...]
    atom_valid_mask: tuple[bool, ...]
    motif_geometry_valid: tuple[bool, ...]
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    exact_identity_digest: tuple[str, ...]
    cross_motif_connections: tuple[CrossMotifConnection, ...]
    token_table_sha256: str

    def validate(
        self, codec: HybridMotifCodec, token_to_id: Mapping[str, int]
    ) -> None:
        """Fail closed on any token/motif/atom-domain inconsistency."""

        if self.schema_version != BOUND_RECORD_SCHEMA_VERSION:
            raise BoundRecordInvariantError("unexpected BoundRecord schema version")
        if self.training_admission is not False:
            raise BoundRecordInvariantError("synthetic BoundRecord cannot admit training")
        if not isinstance(self.record_id, str) or not self.record_id:
            raise BoundRecordInvariantError("record_id must be nonempty")

        token_count = len(self.surface_tokens)
        parallel_token_lengths = (
            len(self.input_ids),
            len(self.token_to_logical_motif),
            len(self.token_role),
        )
        if not token_count or any(length != token_count for length in parallel_token_lengths):
            raise BoundRecordInvariantError("all token-domain arrays must have equal length")
        if self.surface_tokens[0] != MOLECULE_BEGIN or self.surface_tokens[-1] != MOLECULE_END:
            raise BoundRecordInvariantError("molecule boundaries are missing")
        if any(role not in TOKEN_ROLES for role in self.token_role):
            raise BoundRecordInvariantError("unknown token role")

        normalized_table = _validate_token_table(token_to_id)
        expected_table_sha = _token_table_sha256(normalized_table)
        if self.token_table_sha256 != expected_table_sha:
            raise BoundRecordInvariantError("token table SHA-256 mismatch")
        try:
            expected_ids = tuple(normalized_table[token] for token in self.surface_tokens)
        except KeyError as exc:
            raise BoundRecordInvariantError("surface token is absent from token table") from exc
        if self.input_ids != expected_ids:
            raise BoundRecordInvariantError("input_ids do not encode surface_tokens")

        motif_count = len(self.identity_spans)
        motif_parallel_lengths = (
            len(self.surface_modes),
            len(self.connection_spans),
            len(self.logical_to_carrier),
            len(self.motif_atom_indices),
            len(self.motif_slot_atom_indices),
            len(self.motif_geometry_valid),
            len(self.exact_identity_digest),
        )
        if not motif_count or any(length != motif_count for length in motif_parallel_lengths):
            raise BoundRecordInvariantError("all logical-motif arrays must have equal length")

        covered_token_positions: set[int] = {0, token_count - 1}
        carriers: set[int] = set()
        decoded_identities = []
        for motif_id in range(motif_count):
            identity_span = self.identity_spans[motif_id]
            connection_span = self.connection_spans[motif_id]
            if identity_span.stop > token_count or connection_span.stop > token_count:
                raise BoundRecordInvariantError("motif span exceeds token domain")
            if identity_span.stop == identity_span.start:
                raise BoundRecordInvariantError("identity span cannot be empty")
            if identity_span.stop > connection_span.start:
                raise BoundRecordInvariantError("identity and connection spans overlap")
            carrier = self.logical_to_carrier[motif_id]
            if carrier != identity_span.start or carrier in carriers:
                raise BoundRecordInvariantError("each motif needs one unique first-token carrier")
            carriers.add(carrier)

            for position in identity_span.indices():
                if position in covered_token_positions:
                    raise BoundRecordInvariantError("logical motif spans overlap")
                covered_token_positions.add(position)
                if self.token_to_logical_motif[position] != motif_id:
                    raise BoundRecordInvariantError("identity token maps to the wrong motif")
                if self.token_role[position] != "identity":
                    raise BoundRecordInvariantError("identity span has a non-identity token role")
            for position in connection_span.indices():
                if position in covered_token_positions:
                    raise BoundRecordInvariantError("logical motif spans overlap")
                covered_token_positions.add(position)
                if self.token_to_logical_motif[position] != motif_id:
                    raise BoundRecordInvariantError("connection token maps to the wrong motif")
                if self.token_role[position] != "connection":
                    raise BoundRecordInvariantError("connection span has a non-connection role")

            identity_tokens = self.surface_tokens[identity_span.start : identity_span.stop]
            try:
                decoded = codec.decode(identity_tokens)
            except CodecContractError as exc:
                raise BoundRecordInvariantError("identity span does not decode") from exc
            decoded_identities.append(decoded)
            if decoded.exact_identity_digest != self.exact_identity_digest[motif_id]:
                raise BoundRecordInvariantError("decoded identity digest mismatch")
            actual_mode = "fallback" if identity_tokens[0] == FALLBACK_BEGIN else "macro"
            if self.surface_modes[motif_id] != actual_mode:
                raise BoundRecordInvariantError("declared and observed identity surfaces differ")

        if covered_token_positions != set(range(token_count)):
            raise BoundRecordInvariantError("token positions exist outside declared motif spans")
        if self.token_to_logical_motif[0] != -1 or self.token_to_logical_motif[-1] != -1:
            raise BoundRecordInvariantError("molecule boundaries cannot map to a motif")
        if self.token_role[0] != "boundary" or self.token_role[-1] != "boundary":
            raise BoundRecordInvariantError("molecule boundary roles are invalid")
        try:
            decoded_schema = _schema_from_record(self, tuple(decoded_identities))
        except CodecContractError as exc:
            raise BoundRecordInvariantError("logical connection schema is invalid") from exc
        for motif_id, span in enumerate(self.connection_spans):
            expected_connection_tokens = _connection_tokens(decoded_schema, motif_id)
            if self.surface_tokens[span.start : span.stop] != expected_connection_tokens:
                raise BoundRecordInvariantError("connection span is not the canonical rendering")

        atom_count = len(self.atom_to_logical_motif)
        if (
            len(self.model_to_source_atom_index) != atom_count
            or len(self.atom_is_attachment) != atom_count
            or len(self.atom_valid_mask) != atom_count
            or len(self.full_e3fp_ids) != atom_count
            or atom_count == 0
        ):
            raise BoundRecordInvariantError("atom-domain arrays must have equal nonzero length")
        if (
            isinstance(self.source_atom_count, bool)
            or not isinstance(self.source_atom_count, int)
            or self.source_atom_count <= 0
        ):
            raise BoundRecordInvariantError("source_atom_count must be a positive integer")
        if any(
            isinstance(source_atom, bool)
            or not isinstance(source_atom, int)
            or source_atom < 0
            or source_atom >= self.source_atom_count
            for source_atom in self.model_to_source_atom_index
        ):
            raise BoundRecordInvariantError(
                "model-to-source atom mapping must stay inside the declared source domain"
            )
        if tuple(sorted(set(self.model_to_source_atom_index))) != self.model_to_source_atom_index:
            raise BoundRecordInvariantError(
                "model-to-source atom mapping must be strictly increasing"
            )
        if any(value is not True for value in self.atom_valid_mask):
            raise BoundRecordInvariantError("narrow P1 policy requires every model atom valid")
        if any(value is not True for value in self.motif_geometry_valid):
            raise BoundRecordInvariantError("narrow P1 policy requires every motif geometry valid")
        expected_atom_map = [-1] * atom_count
        for motif_id, atoms in enumerate(self.motif_atom_indices):
            if not atoms or tuple(sorted(set(atoms))) != atoms:
                raise BoundRecordInvariantError("motif atom indices must be nonempty and sorted")
            for atom in atoms:
                if atom < 0 or atom >= atom_count or expected_atom_map[atom] != -1:
                    raise BoundRecordInvariantError("motif atom groups overlap or leave the atom domain")
                expected_atom_map[atom] = motif_id
            expected_slots = tuple(
                atoms[position] for position in decoded_identities[motif_id].slot_atom_positions
            )
            if self.motif_slot_atom_indices[motif_id] != expected_slots:
                raise BoundRecordInvariantError("motif slot atoms do not match identity positions")
        if tuple(expected_atom_map) != self.atom_to_logical_motif or -1 in expected_atom_map:
            raise BoundRecordInvariantError("atom-to-logical-motif mapping is not a partition")

        attachment_atoms = {
            atom for motif_slots in self.motif_slot_atom_indices for atom in motif_slots
        }
        expected_attachment = tuple(atom in attachment_atoms for atom in range(atom_count))
        if self.atom_is_attachment != expected_attachment:
            raise BoundRecordInvariantError(
                "atom_is_attachment must be derived exactly from declared motif slots"
            )

        e3fp_widths = {len(row) for row in self.full_e3fp_ids}
        if len(e3fp_widths) != 1 or next(iter(e3fp_widths)) == 0:
            raise BoundRecordInvariantError("E3FP rows need one common positive width")
        for row in self.full_e3fp_ids:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -1
                or value > 4095
                for value in row
            ):
                raise BoundRecordInvariantError(
                    "synthetic E3FP IDs must be integers in [-1, 4095]"
                )
            if row[0] == -1:
                raise BoundRecordInvariantError(
                    "narrow P1 policy requires a valid level-0 E3FP ID for every atom"
                )

        decoded_schema.validate()


def _schema_from_record(
    record: BoundRecord, decoded_identities: Sequence[LogicalMotifIdentity]
) -> LogicalMoleculeSchema:
    motifs = tuple(
        LogicalMotif(
            logical_motif_id=motif_id,
            identity=decoded_identities[motif_id],
            atom_indices=record.motif_atom_indices[motif_id],
            geometry_valid=record.motif_geometry_valid[motif_id],
        )
        for motif_id in range(len(record.motif_atom_indices))
    )
    return LogicalMoleculeSchema(motifs, record.cross_motif_connections)


def _validate_token_table(token_to_id: Mapping[str, int]) -> dict[str, int]:
    normalized = dict(token_to_id)
    if not normalized:
        raise BoundRecordInvariantError("token table cannot be empty")
    if any(not isinstance(token, str) or not token for token in normalized):
        raise BoundRecordInvariantError("token table keys must be nonempty strings")
    ids = tuple(normalized.values())
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in ids):
        raise BoundRecordInvariantError("token IDs must be nonnegative integers")
    if len(set(ids)) != len(ids):
        raise BoundRecordInvariantError("token IDs must be unique")
    return normalized


def _token_table_sha256(token_to_id: Mapping[str, int]) -> str:
    payload = sorted(token_to_id.items(), key=lambda item: (item[1], item[0]))
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_bound_record(
    *,
    record_id: str,
    schema: LogicalMoleculeSchema,
    codec: HybridMotifCodec,
    token_to_id: Mapping[str, int],
    full_e3fp_ids: Sequence[Sequence[int]],
    source_atom_count: int,
    model_to_source_atom_index: Sequence[int],
    force_fallback_motif_ids: Sequence[int] = (),
) -> BoundRecord:
    """Build and immediately validate one synthetic candidate record."""

    schema.validate()
    normalized_table = _validate_token_table(token_to_id)
    forced = frozenset(force_fallback_motif_ids)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in forced):
        raise BoundRecordInvariantError("forced fallback motif IDs must be integers")
    if not forced.issubset(set(range(len(schema.motifs)))):
        raise BoundRecordInvariantError("forced fallback references an unknown motif")

    tokens: list[str] = [MOLECULE_BEGIN]
    token_to_motif: list[int] = [-1]
    roles: list[str] = ["boundary"]
    modes: list[str] = []
    identity_spans: list[Span] = []
    connection_spans: list[Span] = []
    carriers: list[int] = []
    digests: list[str] = []

    for motif in schema.motifs:
        surface = codec.verify_round_trip(
            motif.identity,
            force_fallback=motif.logical_motif_id in forced,
        )
        identity_start = len(tokens)
        tokens.extend(surface.tokens)
        token_to_motif.extend([motif.logical_motif_id] * len(surface.tokens))
        roles.extend(["identity"] * len(surface.tokens))
        identity_stop = len(tokens)
        identity_spans.append(Span(identity_start, identity_stop))
        carriers.append(identity_start + surface.carrier_offset)
        modes.append(surface.mode)
        digests.append(surface.exact_identity_digest)

        connection_start = len(tokens)
        connection_tokens = _connection_tokens(schema, motif.logical_motif_id)
        tokens.extend(connection_tokens)
        token_to_motif.extend([motif.logical_motif_id] * len(connection_tokens))
        roles.extend(["connection"] * len(connection_tokens))
        connection_spans.append(Span(connection_start, len(tokens)))

    tokens.append(MOLECULE_END)
    token_to_motif.append(-1)
    roles.append("boundary")
    try:
        input_ids = tuple(normalized_table[token] for token in tokens)
    except KeyError as exc:
        raise BoundRecordInvariantError("token table does not cover this surface") from exc

    atom_count = schema.atom_count
    source_mapping = tuple(model_to_source_atom_index)
    atom_to_motif = [-1] * atom_count
    for motif in schema.motifs:
        for atom in motif.atom_indices:
            atom_to_motif[atom] = motif.logical_motif_id

    record = BoundRecord(
        schema_version=BOUND_RECORD_SCHEMA_VERSION,
        record_id=record_id,
        training_admission=False,
        surface_modes=tuple(modes),
        surface_tokens=tuple(tokens),
        input_ids=input_ids,
        token_to_logical_motif=tuple(token_to_motif),
        token_role=tuple(roles),
        identity_spans=tuple(identity_spans),
        connection_spans=tuple(connection_spans),
        logical_to_carrier=tuple(carriers),
        motif_atom_indices=tuple(motif.atom_indices for motif in schema.motifs),
        motif_slot_atom_indices=tuple(motif.slot_atom_indices for motif in schema.motifs),
        atom_to_logical_motif=tuple(atom_to_motif),
        source_atom_count=source_atom_count,
        model_to_source_atom_index=source_mapping,
        atom_is_attachment=tuple(
            atom in {
                slot_atom
                for motif in schema.motifs
                for slot_atom in motif.slot_atom_indices
            }
            for atom in range(atom_count)
        ),
        atom_valid_mask=tuple(True for _ in range(atom_count)),
        motif_geometry_valid=tuple(motif.geometry_valid for motif in schema.motifs),
        full_e3fp_ids=tuple(tuple(row) for row in full_e3fp_ids),
        exact_identity_digest=tuple(digests),
        cross_motif_connections=schema.connections,
        token_table_sha256=_token_table_sha256(normalized_table),
    )
    record.validate(codec, normalized_table)
    return record
