"""Lossless endpoint-pair graph grammar over the frozen GraphPorts chemistry.

Version 1 deliberately used a verbose, easy-to-audit five-token connection
surface.  This module changes only that model-facing graph serialization:
canonical connection order supplies the connection id and every edge is two
self-delimiting endpoint varints.  Motif identities, ports, components,
stereochemistry, reconstruction, and source-bound validation remain delegated
to the already validated version-1 chemistry codec.

The first byte of each endpoint is its sole ``connection`` carrier and maps to
the owning logical motif.  Continuation bytes are structural
``boundary``/``-1`` tokens.  This preserves the frozen P1 invariant of exactly
two connection carriers per edge; the carrier indices must not be interpreted
as complete endpoint-mask spans.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from rdkit import Chem

from . import production_graph_ports_codec_v1 as v1


FORMAT_VERSION = "production_graph_ports_endpoint_pair_v2"
GPORTS_V2_BOUNDARY_TOKENS = (
    v1.FALLBACK_BEGIN,
    v1.FALLBACK_END,
    v1.GRAPH_BEGIN,
    v1.GRAPH_END,
    v1.PORT_RADIX,
)
GPORTS_V2_UNION_TOKENS = (
    *GPORTS_V2_BOUNDARY_TOKENS,
    *v1.GPORTS_BYTE_TOKENS,
)


def _append_endpoint(
    *,
    tokens: list[str],
    roles: list[str],
    token_to_logical: list[int],
    packed_endpoint: int,
    logical_motif_id: int,
) -> tuple[int, ...]:
    encoded = v1._uvarint_tokens(packed_endpoint)
    start = len(tokens)
    tokens.extend(encoded)
    roles.append("connection")
    token_to_logical.append(logical_motif_id)
    roles.extend("boundary" for _ in encoded[1:])
    token_to_logical.extend(-1 for _ in encoded[1:])
    return (start,)


def _validate_mapping_permutation(
    mapping: Sequence[int],
    *,
    name: str,
) -> None:
    if (
        not mapping
        or any(isinstance(value, bool) or not isinstance(value, int) for value in mapping)
        or tuple(sorted(mapping)) != tuple(range(len(mapping)))
    ):
        raise v1.GraphPortsContractError(
            f"{name} must be a non-empty permutation of the motif domain"
        )


def _validate_canonical_connections(
    *,
    motif_count: int,
    components: Sequence[tuple[int, ...]],
    connections: Sequence[v1.ConnectionRecord],
) -> None:
    previous_key = None
    used_ports: set[v1.PortRef] = set()
    for expected_connection_id, connection in enumerate(connections, start=1):
        if connection.connection_id != expected_connection_id:
            raise v1.GraphPortsContractError(
                "endpoint-pair connections must have contiguous canonical ids"
            )
        if not (
            connection.bond_type == "SINGLE"
            and connection.bond_stereo == "STEREONONE"
            and connection.stereo_atoms is None
        ):
            raise v1.GraphPortsContractError(
                "endpoint-pair grammar requires SINGLE/STEREONONE connections"
            )
        if not (
            0 <= connection.endpoint_a.motif_id < motif_count
            and 0 <= connection.endpoint_b.motif_id < motif_count
            and connection.endpoint_a.port_id > 0
            and connection.endpoint_b.port_id > 0
            and connection.endpoint_a < connection.endpoint_b
        ):
            raise v1.GraphPortsContractError(
                "endpoint-pair connection has a non-canonical endpoint"
            )
        key = (connection.endpoint_a, connection.endpoint_b)
        if previous_key is not None and key <= previous_key:
            raise v1.GraphPortsContractError(
                "endpoint-pair connections are not in strict canonical order"
            )
        previous_key = key
        for endpoint in key:
            if endpoint in used_ports:
                raise v1.GraphPortsContractError(
                    "one graph port occurs in more than one connection"
                )
            used_ports.add(endpoint)
    expected_components = v1._connected_component_motifs(motif_count, connections)
    if tuple(components) != expected_components:
        raise v1.GraphPortsContractError(
            "endpoint-pair components disagree with the connection graph"
        )


def _build_endpoint_pair_graph_token_stream(
    components: Sequence[tuple[int, ...]],
    connections: Sequence[v1.ConnectionRecord],
    canonical_to_logical_motif_ids: Sequence[int],
) -> v1.ProductionGraphTokenStream:
    _validate_mapping_permutation(
        canonical_to_logical_motif_ids,
        name="canonical_to_logical_motif_ids",
    )
    _validate_canonical_connections(
        motif_count=len(canonical_to_logical_motif_ids),
        components=components,
        connections=connections,
    )
    max_port_id = max(
        (
            endpoint.port_id
            for connection in connections
            for endpoint in (connection.endpoint_a, connection.endpoint_b)
        ),
        default=0,
    )
    port_radix = max_port_id + 1
    header = (
        v1.GRAPH_BEGIN,
        v1.PORT_RADIX,
        *v1._uvarint_tokens(port_radix),
    )
    tokens = list(header)
    roles = ["boundary"] * len(header)
    token_to_logical = [-1] * len(header)
    endpoint_indices: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    connection_indices_by_logical: list[list[int]] = [
        [] for _ in canonical_to_logical_motif_ids
    ]

    for connection in connections:
        logical_a = canonical_to_logical_motif_ids[connection.endpoint_a.motif_id]
        logical_b = canonical_to_logical_motif_ids[connection.endpoint_b.motif_id]
        indices_a = _append_endpoint(
            tokens=tokens,
            roles=roles,
            token_to_logical=token_to_logical,
            packed_endpoint=(
                logical_a * port_radix + connection.endpoint_a.port_id
            ),
            logical_motif_id=logical_a,
        )
        indices_b = _append_endpoint(
            tokens=tokens,
            roles=roles,
            token_to_logical=token_to_logical,
            packed_endpoint=(
                logical_b * port_radix + connection.endpoint_b.port_id
            ),
            logical_motif_id=logical_b,
        )
        endpoint_indices.append((indices_a, indices_b))
        connection_indices_by_logical[logical_a].extend(indices_a)
        connection_indices_by_logical[logical_b].extend(indices_b)

    tokens.append(v1.GRAPH_END)
    roles.append("boundary")
    token_to_logical.append(-1)
    return v1.ProductionGraphTokenStream(
        port_radix=port_radix,
        tokens=tuple(tokens),
        token_roles=tuple(roles),
        token_to_logical_motif=tuple(token_to_logical),
        component_token_indices=tuple(() for _ in components),
        connection_endpoint_token_indices=tuple(endpoint_indices),
        connection_token_indices=tuple(
            tuple(indices) for indices in connection_indices_by_logical
        ),
    )


def _decode_endpoint_pair_graph_token_stream(
    stream: v1.ProductionGraphTokenStream,
    components: Sequence[tuple[int, ...]],
    logical_to_canonical_motif_ids: Sequence[int],
) -> tuple[v1.ConnectionRecord, ...]:
    _validate_mapping_permutation(
        logical_to_canonical_motif_ids,
        name="logical_to_canonical_motif_ids",
    )
    tokens = stream.tokens
    if not (
        len(tokens)
        == len(stream.token_roles)
        == len(stream.token_to_logical_motif)
    ):
        raise v1.GraphPortsContractError("graph token arrays differ in length")
    if not tokens or tokens[0] != v1.GRAPH_BEGIN or tokens[-1] != v1.GRAPH_END:
        raise v1.GraphPortsContractError("endpoint-pair graph framing is invalid")

    def require_boundary(index: int) -> None:
        if (
            stream.token_roles[index] != "boundary"
            or stream.token_to_logical_motif[index] != -1
        ):
            raise v1.GraphPortsContractError(
                "endpoint-pair graph header/trailer must be boundary/-1"
            )

    def read_endpoint(cursor: int) -> tuple[v1.PortRef, int, tuple[int, ...]]:
        start = cursor
        packed, cursor = v1._read_uvarint(tokens, cursor)
        payload_indices = tuple(range(start, cursor))
        logical_motif_id, port_id = divmod(packed, port_radix)
        if not (
            0 <= logical_motif_id < len(logical_to_canonical_motif_ids)
            and port_id > 0
        ):
            raise v1.GraphPortsContractError(
                "endpoint-pair payload is outside its motif/port domain"
            )
        if (
            stream.token_roles[start] != "connection"
            or stream.token_to_logical_motif[start] != logical_motif_id
        ):
            raise v1.GraphPortsContractError(
                "endpoint-pair first byte must carry its logical motif owner"
            )
        for index in payload_indices[1:]:
            require_boundary(index)
        return (
            v1.PortRef(
                logical_to_canonical_motif_ids[logical_motif_id],
                port_id,
            ),
            cursor,
            (start,),
        )

    require_boundary(0)
    cursor = 1
    if cursor >= len(tokens) or tokens[cursor] != v1.PORT_RADIX:
        raise v1.GraphPortsContractError("endpoint-pair graph omits PORT_RADIX")
    require_boundary(cursor)
    cursor += 1
    radix_start = cursor
    port_radix, cursor = v1._read_uvarint(tokens, cursor)
    for index in range(radix_start, cursor):
        require_boundary(index)
    if port_radix <= 0 or port_radix != stream.port_radix:
        raise v1.GraphPortsContractError("endpoint-pair port radix is inconsistent")

    decoded: list[v1.ConnectionRecord] = []
    endpoint_indices: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    by_logical: list[list[int]] = [
        [] for _ in logical_to_canonical_motif_ids
    ]
    while cursor < len(tokens) - 1:
        endpoint_a, cursor, indices_a = read_endpoint(cursor)
        if cursor >= len(tokens) - 1:
            raise v1.GraphPortsContractError(
                "endpoint-pair graph ends after only one endpoint"
            )
        endpoint_b, cursor, indices_b = read_endpoint(cursor)
        if not endpoint_a < endpoint_b:
            raise v1.GraphPortsContractError(
                "endpoint-pair connection is not in canonical endpoint order"
            )
        decoded.append(
            v1.ConnectionRecord(
                connection_id=len(decoded) + 1,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                bond_type="SINGLE",
                bond_stereo="STEREONONE",
                stereo_atoms=None,
            )
        )
        endpoint_indices.append((indices_a, indices_b))
        logical_a = stream.token_to_logical_motif[indices_a[0]]
        logical_b = stream.token_to_logical_motif[indices_b[0]]
        by_logical[logical_a].extend(indices_a)
        by_logical[logical_b].extend(indices_b)

    if cursor != len(tokens) - 1:
        raise v1.GraphPortsContractError("endpoint-pair graph has trailing payload")
    require_boundary(cursor)
    if stream.component_token_indices != tuple(() for _ in components):
        raise v1.GraphPortsContractError("components must remain metadata-only")
    if stream.connection_endpoint_token_indices != tuple(endpoint_indices):
        raise v1.GraphPortsContractError(
            "endpoint-pair token indices disagree with decoded endpoints"
        )
    if stream.connection_token_indices != tuple(tuple(row) for row in by_logical):
        raise v1.GraphPortsContractError(
            "logical connection indices disagree with endpoint owners"
        )
    expected_radix = max(
        (
            endpoint.port_id
            for connection in decoded
            for endpoint in (connection.endpoint_a, connection.endpoint_b)
        ),
        default=0,
    ) + 1
    if port_radix != expected_radix:
        raise v1.GraphPortsContractError(
            "endpoint-pair port radix is not the canonical minimum"
        )
    _validate_canonical_connections(
        motif_count=len(logical_to_canonical_motif_ids),
        components=components,
        connections=decoded,
    )
    return tuple(decoded)


def upgrade_v1_encoding(
    encoding: v1.GraphPortsEncoding,
) -> v1.GraphPortsEncoding:
    """Replace only a valid v1 graph surface with the v2 endpoint-pair surface."""

    if encoding.format_version != v1.FORMAT_VERSION:
        raise v1.GraphPortsContractError("upgrade requires one v1 encoding")
    v1.ProductionGraphPortsCodecV1().validate(encoding)
    upgraded = replace(
        encoding,
        format_version=FORMAT_VERSION,
        graph_token_stream=_build_endpoint_pair_graph_token_stream(
            encoding.component_motif_ids,
            encoding.connections,
            encoding.canonical_to_logical_motif_ids,
        ),
    )
    ProductionGraphPortsCodecV2().validate(upgraded)
    return upgraded


def _downgrade_to_validated_v1(
    encoding: v1.GraphPortsEncoding,
) -> v1.GraphPortsEncoding:
    if encoding.format_version != FORMAT_VERSION:
        raise v1.GraphPortsContractError(
            "endpoint-pair codec received another format"
        )
    decoded = _decode_endpoint_pair_graph_token_stream(
        encoding.graph_token_stream,
        encoding.component_motif_ids,
        encoding.logical_to_canonical_motif_ids,
    )
    if decoded != encoding.connections:
        raise v1.GraphPortsContractError(
            "endpoint-pair graph tokens disagree with connection metadata"
        )
    return replace(
        encoding,
        format_version=v1.FORMAT_VERSION,
        graph_token_stream=v1._build_graph_token_stream(
            encoding.component_motif_ids,
            encoding.connections,
            encoding.canonical_to_logical_motif_ids,
        ),
    )


class ProductionGraphPortsCodecV2:
    """Version-1 chemistry with a two-endpoint-per-edge model surface."""

    def __init__(self) -> None:
        self._chemistry_codec = v1.ProductionGraphPortsCodecV1()

    @staticmethod
    def required_union_tokens() -> tuple[str, ...]:
        return GPORTS_V2_UNION_TOKENS

    @staticmethod
    def encode_identity_surface(*args, **kwargs):
        return v1.ProductionGraphPortsCodecV1.encode_identity_surface(*args, **kwargs)

    @staticmethod
    def decode_identity_surface(*args, **kwargs):
        return v1.ProductionGraphPortsCodecV1.decode_identity_surface(*args, **kwargs)

    @staticmethod
    def decode_graph_token_stream(
        encoding: v1.GraphPortsEncoding,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[v1.ConnectionRecord, ...]]:
        connections = _decode_endpoint_pair_graph_token_stream(
            encoding.graph_token_stream,
            encoding.component_motif_ids,
            encoding.logical_to_canonical_motif_ids,
        )
        return encoding.component_motif_ids, connections

    def encode(
        self,
        projected_mol: Chem.Mol,
        motif_atom_groups: Sequence[Sequence[int]],
        cross_edges: Sequence[v1.CrossEdgeInput],
    ) -> v1.GraphPortsEncoding:
        return upgrade_v1_encoding(
            self._chemistry_codec.encode(
                projected_mol,
                motif_atom_groups,
                cross_edges,
            )
        )

    def validate(self, encoding: v1.GraphPortsEncoding) -> None:
        self.reconstruct(encoding, verify_identity=True)

    def validate_against_source(
        self,
        projected_mol: Chem.Mol,
        motif_atom_groups: Sequence[Sequence[int]],
        cross_edges: Sequence[v1.CrossEdgeInput],
        encoding: v1.GraphPortsEncoding,
    ) -> None:
        expected = self.encode(projected_mol, motif_atom_groups, cross_edges)
        if encoding != expected:
            raise v1.GraphPortsContractError(
                "endpoint-pair encoding differs from canonical source re-encoding"
            )

    def reconstruct(
        self,
        encoding: v1.GraphPortsEncoding,
        *,
        verify_identity: bool = True,
    ) -> Chem.Mol:
        downgraded = _downgrade_to_validated_v1(encoding)
        return self._chemistry_codec.reconstruct(
            downgraded,
            verify_identity=verify_identity,
        )


__all__ = [
    "FORMAT_VERSION",
    "GPORTS_V2_BOUNDARY_TOKENS",
    "GPORTS_V2_UNION_TOKENS",
    "ProductionGraphPortsCodecV2",
    "_build_endpoint_pair_graph_token_stream",
    "_decode_endpoint_pair_graph_token_stream",
    "upgrade_v1_encoding",
]
