"""Pure producer for paired atom/SELFIES and motif/graph P1 records.

The producer accepts one already-projected RDKit molecule and one frozen
logical-motif partition.  Both A and M records are derived in the same call,
from the same atom row axis, inherited E3FP matrix, source mapping and union
tokenizer.  It performs no filesystem or network I/O.

The M record is not instantiated directly: a minimal vNext CE-first document
is materialized and passed through ``load_production_motif_record`` so the
existing graph/token/atom-domain gate remains the authoritative validator.
The document's deterministic all-motif mask is validation scaffolding only;
epoch-specific masking is still applied by the production collator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import re
from typing import Any, Mapping, Sequence

from most_t5_next.p1.atom_production_bridge import (
    ATOM_SELFIES_RECORD_SCHEMA,
    ProductionAtomSelfiesRecord,
)
from most_t5_next.p1.ce_collator import (
    IDENTITY_RECOVERY_OBJECTIVE,
    _mask_decision_sha256,
)
from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    load_production_motif_record,
)
from most_t5_next.p1.runtime_bridge import P1ArtifactBindings, P1MemberRef
from most_t5_next.r1.gates import validate_p1_logical_motif_vnext as motif_validator
from most_t5_next.r1.tokenizer.production_atom_selfies_codec_v1 import (
    AtomSelfiesAlignment,
    AtomSelfiesSurface,
    bind_atom_selfies_surface,
    discover_atom_selfies_surface,
)
from most_t5_next.r1.tokenizer.production_graph_ports_codec_v1 import (
    CrossEdgeInput,
    GraphPortsEncoding,
    IdentitySurface,
    ProductionGraphPortsCodecV1,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MOTIF_MACRO_RE = re.compile(r"^<MOST:M:[0-9]{6}>$")
MOTIF_IDENTITY_ROLE = "identity"
BOUNDARY_ROLE = "boundary"


class ProductionPairedIdentityError(ValueError):
    """The common A/M production boundary cannot be proven."""


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProductionPairedIdentityError(f"{field} must be a lower-case SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_text(value).encode("utf-8")).hexdigest()


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProductionPairedIdentityError(f"{field} must be an integer")
    return int(value)


def _normalize_source_mapping(
    values: Sequence[int], atom_count: int, source_atom_count: int
) -> tuple[int, ...]:
    try:
        result = tuple(
            _plain_int(value, f"model_to_source_atom_index[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise ProductionPairedIdentityError(
            "model_to_source_atom_index must be a finite sequence"
        ) from exc
    if len(result) != atom_count:
        raise ProductionPairedIdentityError(
            "model_to_source_atom_index length must equal projected atom count"
        )
    if result != tuple(sorted(set(result))) or any(
        value < 0 or value >= source_atom_count for value in result
    ):
        raise ProductionPairedIdentityError(
            "model_to_source_atom_index must be strictly increasing and inside the source domain"
        )
    return result


def _normalize_e3fp(
    rows: Sequence[Sequence[int]], atom_count: int
) -> tuple[tuple[int, ...], ...]:
    try:
        result = tuple(
            tuple(
                _plain_int(value, f"inherited_e3fp[{atom_id}][{level}]")
                for level, value in enumerate(row)
            )
            for atom_id, row in enumerate(rows)
        )
    except TypeError as exc:
        raise ProductionPairedIdentityError(
            "inherited_e3fp must be a finite rectangular sequence"
        ) from exc
    if len(result) != atom_count or not result:
        raise ProductionPairedIdentityError(
            "inherited_e3fp row count must equal projected atom count"
        )
    level_counts = {len(row) for row in result}
    if len(level_counts) != 1 or 0 in level_counts:
        raise ProductionPairedIdentityError("inherited_e3fp must be non-empty and rectangular")
    for row in result:
        if row[0] < 0 or any(value < -1 or value > 4095 for value in row):
            raise ProductionPairedIdentityError(
                "inherited_e3fp values are outside the narrow P1 domain"
            )
    return result


def _exact_token_ids(
    tokenizer: Any,
    tokens: Sequence[str],
    *,
    complete_surface: str,
    vocab_size: int,
) -> tuple[int, ...]:
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if isinstance(unk_token_id, bool) or not isinstance(unk_token_id, Integral):
        raise ProductionPairedIdentityError("union tokenizer lacks a valid unk_token_id")
    unk_token_id = int(unk_token_id)
    ids: list[int] = []
    for offset, token in enumerate(tokens):
        if not isinstance(token, str) or not token:
            raise ProductionPairedIdentityError(f"M token {offset} is not a non-empty string")
        try:
            raw_id = tokenizer.convert_tokens_to_ids(token)
            singleton = tokenizer.encode(token, add_special_tokens=False)
        except Exception as exc:
            raise ProductionPairedIdentityError("union tokenizer API failed") from exc
        if isinstance(raw_id, bool) or not isinstance(raw_id, Integral):
            raise ProductionPairedIdentityError(f"M token {token!r} has no integral id")
        token_id = int(raw_id)
        if token_id < 0 or token_id >= vocab_size or token_id == unk_token_id:
            raise ProductionPairedIdentityError(f"M token {token!r} is outside the frozen vocabulary")
        if not isinstance(singleton, (list, tuple)) or tuple(singleton) != (token_id,):
            raise ProductionPairedIdentityError(f"M token {token!r} is not encoded exactly once")
        try:
            reverse = tokenizer.convert_ids_to_tokens(token_id)
        except Exception as exc:
            raise ProductionPairedIdentityError("union tokenizer reverse API failed") from exc
        if reverse != token:
            raise ProductionPairedIdentityError(f"M token {token!r} is not reversible")
        ids.append(token_id)
    try:
        whole = tokenizer.encode(complete_surface, add_special_tokens=False)
    except Exception as exc:
        raise ProductionPairedIdentityError("union tokenizer whole-surface API failed") from exc
    if not isinstance(whole, (list, tuple)) or tuple(whole) != tuple(ids):
        raise ProductionPairedIdentityError(
            "union tokenizer does not preserve the complete M surface token boundaries"
        )
    return tuple(ids)


def _atom_record_artifact_sha256(
    *,
    member: P1MemberRef,
    bindings: P1ArtifactBindings,
    alignment: AtomSelfiesAlignment,
    source_atom_count: int,
    model_to_source_atom_index: tuple[int, ...],
    inherited_e3fp: tuple[tuple[int, ...], ...],
) -> str:
    """Compute the one summary required by ProductionAtomSelfiesRecord."""

    return _canonical_sha256(
        {
            "schema_version": ATOM_SELFIES_RECORD_SCHEMA,
            "member": member.as_dict(),
            "release_id": bindings.release_id,
            "geometry_record_content_sha256": bindings.geometry_record_content_sha256,
            "union_tokenizer_contract_sha256": bindings.tokenizer_contract_sha256,
            "union_tokenizer_snapshot_sha256": bindings.tokenizer_snapshot_sha256,
            "selfies": alignment.selfies,
            "input_ids": list(alignment.input_ids),
            "token_to_atom": list(alignment.token_to_atom),
            "token_role": list(alignment.token_role),
            "atom_identity_spans": [
                [span.start, span.stop] for span in alignment.atom_identity_spans
            ],
            "atom_to_carrier": list(alignment.atom_to_carrier),
            "source_atom_count": source_atom_count,
            "model_to_source_atom_index": list(model_to_source_atom_index),
            "full_e3fp_ids": [list(row) for row in inherited_e3fp],
            "atom_valid_mask": [True] * len(inherited_e3fp),
        }
    )


@dataclass(frozen=True)
class ProductionPairReceipt:
    """Small lineage receipt; the base digest never enters either model row."""

    member_id: str
    storage_key: str
    release_id: str
    base_geometry_record_content_sha256: str
    effective_inherited_overlay_content_sha256: str
    strict_isomeric_identity: str

    def __post_init__(self) -> None:
        for field in (
            "base_geometry_record_content_sha256",
            "effective_inherited_overlay_content_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.base_geometry_record_content_sha256 == self.effective_inherited_overlay_content_sha256:
            raise ProductionPairedIdentityError(
                "effective inherited-overlay content must not be represented by the base geometry digest"
            )
        for field in ("member_id", "storage_key", "release_id", "strict_isomeric_identity"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ProductionPairedIdentityError(f"{field} must be non-empty")


@dataclass(frozen=True)
class PreparedPairedIdentitySurfaces:
    """Tokenizer-independent A/M chemistry evidence for the second pass."""

    atom_surface: AtomSelfiesSurface
    graph_encoding: GraphPortsEncoding

    def __post_init__(self) -> None:
        if not isinstance(self.atom_surface, AtomSelfiesSurface) or not isinstance(
            self.graph_encoding, GraphPortsEncoding
        ):
            raise ProductionPairedIdentityError(
                "prepared surfaces contain an unknown codec result"
            )
        if (
            self.atom_surface.canonical_isomeric_smiles
            != self.graph_encoding.strict_isomeric_identity
        ):
            raise ProductionPairedIdentityError(
                "A SELFIES and M graph strict identities disagree"
            )
        atom_count = len(self.atom_surface.canonical_position_to_model_atom)
        graph_atoms = tuple(
            atom_id
            for group in self.graph_encoding.logical_motif_atom_groups
            for atom_id in group
        )
        if sorted(graph_atoms) != list(range(atom_count)):
            raise ProductionPairedIdentityError(
                "A SELFIES and M graph atom axes disagree"
            )


def discover_production_paired_identity_surfaces(
    Chem: Any,
    sf: Any,
    projected_mol: Any,
    logical_motif_atom_groups: Sequence[Sequence[int]],
    cross_edges: Sequence[CrossEdgeInput],
) -> PreparedPairedIdentitySurfaces:
    """Run both strict chemistry codecs once, before union-vocab binding."""

    atom_surface = discover_atom_selfies_surface(Chem, sf, projected_mol)
    graph = ProductionGraphPortsCodecV1().encode(
        projected_mol,
        logical_motif_atom_groups,
        cross_edges,
    )
    return PreparedPairedIdentitySurfaces(
        atom_surface=atom_surface,
        graph_encoding=graph,
    )


@dataclass(frozen=True)
class ProductionPairedIdentityRecords:
    """The accepted A/M pair and the common graph/lineage evidence."""

    atom_record: ProductionAtomSelfiesRecord
    motif_record: ProductionMotifRecord
    receipt: ProductionPairReceipt
    graph_encoding: GraphPortsEncoding
    motif_identity_surfaces: tuple[IdentitySurface, ...]
    motif_document_canonical_json: str

    def __post_init__(self) -> None:
        atom = self.atom_record
        motif = self.motif_record
        receipt = self.receipt
        if not isinstance(atom, ProductionAtomSelfiesRecord) or not isinstance(
            motif, ProductionMotifRecord
        ):
            raise ProductionPairedIdentityError("paired result contains an unknown record type")
        shared = (
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
        if not all(shared):
            raise ProductionPairedIdentityError("A/M member, tokenizer or shared geometry parity failed")
        if self.graph_encoding.strict_isomeric_identity != receipt.strict_isomeric_identity:
            raise ProductionPairedIdentityError("receipt and M graph identities disagree")
        identities = tuple(motif.identity_smiles for motif in self.graph_encoding.motifs)
        if len(self.motif_identity_surfaces) != len(identities):
            raise ProductionPairedIdentityError("motif identity surface count disagrees with graph")
        if len(motif.identity_spans) != len(identities):
            raise ProductionPairedIdentityError("motif record and graph motif counts disagree")
        expected_spans = []
        cursor = 1
        for logical_id, surface in enumerate(self.motif_identity_surfaces):
            if surface.mode not in {"macro", "fallback"} or not surface.tokens:
                raise ProductionPairedIdentityError(
                    f"motif {logical_id} has an invalid identity surface"
                )
            expected_spans.append((cursor, cursor + len(surface.tokens)))
            cursor += len(surface.tokens)
        graph_offset = cursor
        graph_stream = self.graph_encoding.graph_token_stream
        expected_roles = (
            BOUNDARY_ROLE,
            *(
                MOTIF_IDENTITY_ROLE
                for surface in self.motif_identity_surfaces
                for _ in surface.tokens
            ),
            *graph_stream.token_roles,
            BOUNDARY_ROLE,
        )
        expected_mapping = (
            -1,
            *(
                logical_id
                for logical_id, surface in enumerate(self.motif_identity_surfaces)
                for _ in surface.tokens
            ),
            *graph_stream.token_to_logical_motif,
            -1,
        )
        expected_connections = tuple(
            tuple(graph_offset + index for index in row)
            for row in graph_stream.connection_token_indices
        )
        if (
            motif.identity_spans != tuple(Span(start, stop) for start, stop in expected_spans)
            or motif.logical_to_carrier != tuple(start for start, _stop in expected_spans)
            or motif.connection_token_indices != expected_connections
            or motif.token_role != expected_roles
            or motif.token_to_logical_motif != expected_mapping
        ):
            raise ProductionPairedIdentityError("M token spans or graph offsets were tampered")
        expected_identity_digests = tuple(
            hashlib.sha256(identity.encode("utf-8")).hexdigest() for identity in identities
        )
        if motif.exact_identity_sha256 != expected_identity_digests:
            raise ProductionPairedIdentityError("M exact motif identity digests disagree with graph")
        expected_owner = [-1] * len(motif.atom_to_logical_motif)
        for logical_id, group in enumerate(self.graph_encoding.logical_motif_atom_groups):
            for atom_id in group:
                if not 0 <= atom_id < len(expected_owner):
                    raise ProductionPairedIdentityError("M graph atom owner is outside the record domain")
                expected_owner[atom_id] = logical_id
        if tuple(expected_owner) != motif.atom_to_logical_motif:
            raise ProductionPairedIdentityError("M atom owners disagree with the frozen graph partition")

        try:
            document = json.loads(self.motif_document_canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProductionPairedIdentityError(
                "retained M document is not canonical JSON"
            ) from exc
        if not isinstance(document, dict) or _canonical_json_text(document) != (
            self.motif_document_canonical_json
        ):
            raise ProductionPairedIdentityError(
                "retained M document is not canonical JSON"
            )

    @property
    def motif_training_document(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible copy of the validated vNext document."""

        return json.loads(self.motif_document_canonical_json)

    @property
    def motif_document(self) -> dict[str, Any]:
        """Backward-readable alias for ``motif_training_document``."""

        return self.motif_training_document


def _build_motif_record(
    *,
    graph_codec: ProductionGraphPortsCodecV1,
    graph: GraphPortsEncoding,
    member: P1MemberRef,
    bindings: P1ArtifactBindings,
    source_atom_count: int,
    model_to_source_atom_index: tuple[int, ...],
    inherited_e3fp: tuple[tuple[int, ...], ...],
    union_tokenizer: Any,
    tokenizer_binding: ProductionTokenizerRuntime,
    macro_by_identity: Mapping[str, str] | None,
) -> tuple[ProductionMotifRecord, tuple[IdentitySurface, ...], str]:
    macro_by_identity = dict(macro_by_identity or {})
    if any(MOTIF_MACRO_RE.fullmatch(token) is None for token in macro_by_identity.values()):
        raise ProductionPairedIdentityError(
            "motif macros must use the opaque <MOST:M:000000> namespace"
        )
    identity_by_macro = {token: identity for identity, token in macro_by_identity.items()}
    if len(identity_by_macro) != len(macro_by_identity):
        raise ProductionPairedIdentityError("motif macro mapping must be injective")

    tokens: list[str] = ["<bom>"]
    roles: list[str] = [BOUNDARY_ROLE]
    token_to_motif: list[int] = [-1]
    spans: list[list[int]] = []
    surfaces: list[IdentitySurface] = []
    identity_digests: list[str] = []
    for logical_id, motif in enumerate(graph.motifs):
        surface = graph_codec.encode_identity_surface(
            motif.identity_smiles,
            macro_by_identity=macro_by_identity,
        )
        decoded = graph_codec.decode_identity_surface(
            surface,
            identity_by_macro=identity_by_macro,
        )
        if decoded != motif.identity_smiles:
            raise ProductionPairedIdentityError(
                f"motif {logical_id} identity surface is not lossless"
            )
        start = len(tokens)
        tokens.extend(surface.tokens)
        roles.extend(MOTIF_IDENTITY_ROLE for _ in surface.tokens)
        token_to_motif.extend(logical_id for _ in surface.tokens)
        spans.append([start, len(tokens)])
        surfaces.append(surface)
        identity_digests.append(hashlib.sha256(motif.identity_smiles.encode("utf-8")).hexdigest())

    graph_offset = len(tokens)
    graph_stream = graph.graph_token_stream
    tokens.extend(graph_stream.tokens)
    roles.extend(graph_stream.token_roles)
    token_to_motif.extend(graph_stream.token_to_logical_motif)
    tokens.append("<eom>")
    roles.append(BOUNDARY_ROLE)
    token_to_motif.append(-1)
    input_ids = _exact_token_ids(
        union_tokenizer,
        tokens,
        complete_surface="".join(tokens),
        vocab_size=tokenizer_binding.vocab_size,
    )

    motif_atoms = [list(group) for group in graph.logical_motif_atom_groups]
    atom_to_motif = [-1] * len(inherited_e3fp)
    for logical_id, group in enumerate(graph.logical_motif_atom_groups):
        for atom_id in group:
            atom_to_motif[atom_id] = logical_id
    if any(owner < 0 for owner in atom_to_motif):
        raise ProductionPairedIdentityError("logical motif groups do not own every model atom")

    motif_slot_atoms = [
        [port.source_atom_index for port in motif.ports]
        for motif in graph.motifs
    ]
    port_by_ref = {
        (motif.motif_id, port.port_id): port
        for motif in graph.motifs
        for port in motif.ports
    }
    cross_motif_bonds = []
    attachment_atoms: set[int] = set()
    for edge_index, connection in enumerate(graph.connections):
        left_port = port_by_ref[(connection.endpoint_a.motif_id, connection.endpoint_a.port_id)]
        right_port = port_by_ref[(connection.endpoint_b.motif_id, connection.endpoint_b.port_id)]
        attachment_atoms.update((left_port.source_atom_index, right_port.source_atom_index))
        cross_motif_bonds.append(
            {
                "edge_id": edge_index,
                "left": {
                    "logical_motif_index": connection.endpoint_a.motif_id,
                    "atom_index": left_port.source_atom_index,
                    "slot_ordinal": connection.endpoint_a.port_id - 1,
                },
                "right": {
                    "logical_motif_index": connection.endpoint_b.motif_id,
                    "atom_index": right_port.source_atom_index,
                    "slot_ordinal": connection.endpoint_b.port_id - 1,
                },
                "bond_type": connection.bond_type.lower(),
            }
        )

    selected_ids = list(range(len(graph.motifs)))
    mask_probability = 1.0
    mask_seed = 0
    mask_epoch = 0
    document = {
        "schema_version": motif_validator.RECORD_SCHEMA,
        "document_kind": motif_validator.RECORD_KIND,
        "training_profile": motif_validator.CE_PROFILE,
        "bindings": bindings.as_dict(),
        "member": member.as_dict(),
        "dimensions": {
            "token_count": len(input_ids),
            "logical_motif_count": len(graph.motifs),
            "atom_count": len(inherited_e3fp),
            "source_atom_count": source_atom_count,
            "e3fp_level_count": len(inherited_e3fp[0]),
        },
        "token_domain": {
            "input_ids": list(input_ids),
            "attention_mask": [True] * len(input_ids),
            "token_to_logical_motif": token_to_motif,
            "token_role": roles,
        },
        "logical_motif_domain": {
            "identity_spans": spans,
            "connection_token_indices": [
                [graph_offset + index for index in row]
                for row in graph_stream.connection_token_indices
            ],
            "logical_to_carrier": [span[0] for span in spans],
            "exact_identity_sha256": identity_digests,
            "motif_geometry_valid": [True] * len(graph.motifs),
            "motif_atom_indices": motif_atoms,
            "motif_slot_atom_indices": motif_slot_atoms,
            "slot_count": [len(row) for row in motif_slot_atoms],
            "cross_motif_bonds": cross_motif_bonds,
        },
        "atom_domain": {
            "atom_to_logical_motif": atom_to_motif,
            "model_to_source_atom_index": list(model_to_source_atom_index),
            "atom_valid_mask": [True] * len(inherited_e3fp),
            "atom_is_attachment": [
                atom_id in attachment_atoms for atom_id in range(len(inherited_e3fp))
            ],
            "full_e3fp_ids": [list(row) for row in inherited_e3fp],
        },
        "masks": {"identity_recovery_mask": [True] * len(graph.motifs)},
        "mask_decision": {
            "objective": IDENTITY_RECOVERY_OBJECTIVE,
            "seed": mask_seed,
            "epoch": mask_epoch,
            "mask_probability": mask_probability,
            "selected_logical_motif_indices": selected_ids,
            "decision_sha256": _mask_decision_sha256(
                seed=mask_seed,
                epoch=mask_epoch,
                record_id=member.member_id,
                objective=IDENTITY_RECOVERY_OBJECTIVE,
                mask_probability=mask_probability,
                selected_logical_motif_ids=selected_ids,
            ),
        },
    }
    document_json = _canonical_json_text(document)
    # The loader receives its own JSON round-tripped copy.  The paired result
    # retains only immutable canonical text and exposes fresh copies, so no
    # caller can mutate the document that was accepted at this boundary.
    record = load_production_motif_record(json.loads(document_json))
    return record, tuple(surfaces), document_json


def build_production_paired_identity_records_from_prepared(
    *,
    prepared: PreparedPairedIdentitySurfaces,
    member: P1MemberRef,
    bindings: P1ArtifactBindings,
    base_geometry_record_content_sha256: str,
    effective_inherited_overlay_content_sha256: str,
    source_atom_count: int,
    model_to_source_atom_index: Sequence[int],
    inherited_e3fp: Sequence[Sequence[int]],
    union_tokenizer: Any,
    tokenizer_binding: ProductionTokenizerRuntime,
    macro_by_identity: Mapping[str, str] | None = None,
) -> ProductionPairedIdentityRecords:
    """Bind one prepared A/M surface pair and materialize both records."""

    if not isinstance(prepared, PreparedPairedIdentitySurfaces):
        raise ProductionPairedIdentityError(
            "prepared must be PreparedPairedIdentitySurfaces"
        )
    if not isinstance(member, P1MemberRef):
        raise ProductionPairedIdentityError("member must be P1MemberRef")
    if not isinstance(bindings, P1ArtifactBindings):
        raise ProductionPairedIdentityError("bindings must be P1ArtifactBindings")
    if not isinstance(tokenizer_binding, ProductionTokenizerRuntime):
        raise ProductionPairedIdentityError(
            "tokenizer_binding must be ProductionTokenizerRuntime"
        )
    base_sha = _require_sha256(
        base_geometry_record_content_sha256,
        "base_geometry_record_content_sha256",
    )
    effective_sha = _require_sha256(
        effective_inherited_overlay_content_sha256,
        "effective_inherited_overlay_content_sha256",
    )
    if bindings.geometry_record_content_sha256 != effective_sha:
        raise ProductionPairedIdentityError(
            "bindings.geometry_record_content_sha256 must name the effective inherited overlay"
        )
    if base_sha == effective_sha:
        raise ProductionPairedIdentityError(
            "base and effective inherited-overlay content digests must be distinct"
        )
    if (
        bindings.tokenizer_contract_sha256 != tokenizer_binding.tokenizer_contract_sha256
        or bindings.tokenizer_snapshot_sha256 != tokenizer_binding.tokenizer_snapshot_sha256
    ):
        raise ProductionPairedIdentityError(
            "release bindings and tokenizer binding describe different union tokenizers"
        )
    source_atom_count = _plain_int(source_atom_count, "source_atom_count")
    if source_atom_count <= 0:
        raise ProductionPairedIdentityError("source_atom_count must be positive")
    atom_count = len(prepared.atom_surface.canonical_position_to_model_atom)
    source_mapping = _normalize_source_mapping(
        model_to_source_atom_index, atom_count, source_atom_count
    )
    e3fp = _normalize_e3fp(inherited_e3fp, atom_count)

    alignment = bind_atom_selfies_surface(prepared.atom_surface, union_tokenizer)
    if any(token_id >= tokenizer_binding.vocab_size for token_id in alignment.input_ids):
        raise ProductionPairedIdentityError("A surface contains an ID outside the frozen vocabulary")

    graph_codec = ProductionGraphPortsCodecV1()
    graph = prepared.graph_encoding
    if alignment.canonical_isomeric_smiles != graph.strict_isomeric_identity:
        raise ProductionPairedIdentityError("A SELFIES and M graph strict identities disagree")

    atom_record = ProductionAtomSelfiesRecord(
        record_artifact_sha256=_atom_record_artifact_sha256(
            member=member,
            bindings=bindings,
            alignment=alignment,
            source_atom_count=source_atom_count,
            model_to_source_atom_index=source_mapping,
            inherited_e3fp=e3fp,
        ),
        record_id=member.member_id,
        storage_key=member.storage_key,
        release_id=bindings.release_id,
        geometry_record_content_sha256=effective_sha,
        union_tokenizer_contract_sha256=tokenizer_binding.tokenizer_contract_sha256,
        union_tokenizer_snapshot_sha256=tokenizer_binding.tokenizer_snapshot_sha256,
        selfies=alignment.selfies,
        input_ids=alignment.input_ids,
        token_to_atom=alignment.token_to_atom,
        token_role=alignment.token_role,
        atom_identity_spans=alignment.atom_identity_spans,
        atom_to_carrier=alignment.atom_to_carrier,
        source_atom_count=source_atom_count,
        model_to_source_atom_index=source_mapping,
        full_e3fp_ids=e3fp,
        atom_valid_mask=(True,) * atom_count,
    )
    motif_record, surfaces, motif_document_json = _build_motif_record(
        graph_codec=graph_codec,
        graph=graph,
        member=member,
        bindings=bindings,
        source_atom_count=source_atom_count,
        model_to_source_atom_index=source_mapping,
        inherited_e3fp=e3fp,
        union_tokenizer=union_tokenizer,
        tokenizer_binding=tokenizer_binding,
        macro_by_identity=macro_by_identity,
    )
    receipt = ProductionPairReceipt(
        member_id=member.member_id,
        storage_key=member.storage_key,
        release_id=bindings.release_id,
        base_geometry_record_content_sha256=base_sha,
        effective_inherited_overlay_content_sha256=effective_sha,
        strict_isomeric_identity=graph.strict_isomeric_identity,
    )
    return ProductionPairedIdentityRecords(
        atom_record=atom_record,
        motif_record=motif_record,
        receipt=receipt,
        graph_encoding=graph,
        motif_identity_surfaces=surfaces,
        motif_document_canonical_json=motif_document_json,
    )


def build_production_paired_identity_records(
    Chem: Any,
    sf: Any,
    *,
    projected_mol: Any,
    logical_motif_atom_groups: Sequence[Sequence[int]],
    cross_edges: Sequence[CrossEdgeInput],
    member: P1MemberRef,
    bindings: P1ArtifactBindings,
    base_geometry_record_content_sha256: str,
    effective_inherited_overlay_content_sha256: str,
    source_atom_count: int,
    model_to_source_atom_index: Sequence[int],
    inherited_e3fp: Sequence[Sequence[int]],
    union_tokenizer: Any,
    tokenizer_binding: ProductionTokenizerRuntime,
    macro_by_identity: Mapping[str, str] | None = None,
) -> ProductionPairedIdentityRecords:
    """Compatibility wrapper: discover once, then use the prepared boundary."""

    prepared = discover_production_paired_identity_surfaces(
        Chem,
        sf,
        projected_mol,
        logical_motif_atom_groups,
        cross_edges,
    )
    return build_production_paired_identity_records_from_prepared(
        prepared=prepared,
        member=member,
        bindings=bindings,
        base_geometry_record_content_sha256=base_geometry_record_content_sha256,
        effective_inherited_overlay_content_sha256=(
            effective_inherited_overlay_content_sha256
        ),
        source_atom_count=source_atom_count,
        model_to_source_atom_index=model_to_source_atom_index,
        inherited_e3fp=inherited_e3fp,
        union_tokenizer=union_tokenizer,
        tokenizer_binding=tokenizer_binding,
        macro_by_identity=macro_by_identity,
    )


__all__ = [
    "PreparedPairedIdentitySurfaces",
    "ProductionPairReceipt",
    "ProductionPairedIdentityError",
    "ProductionPairedIdentityRecords",
    "build_production_paired_identity_records",
    "build_production_paired_identity_records_from_prepared",
    "discover_production_paired_identity_surfaces",
]
