"""Lossless graph-and-ports codec for production motif records.

This module is intentionally independent from the frozen linearizer and legacy
tokenizer.  Its input contract is deliberately small and explicit:

* a sanitized, projected RDKit molecule without dummy atoms;
* a complete partition of its atom indices into connected motif groups; and
* every bond crossing that partition, represented by its two atom endpoints
  and the required ``SINGLE`` RDKit bond-type name.

The codec verifies the supplied cross-edge table against the molecule instead
of guessing the schema of an upstream topology object.  It then cuts those
bonds, assigns canonical *motif-local* distinguishable ports, and records a
canonical connection table.  The CAMT5-derived production partition cuts only
non-stereogenic single bonds; motif payloads retain tetrahedral stereochemistry
and internal E/Z, including E/Z whose support atom is replaced by a port.  A
non-single or stereogenic cross-motif bond is a contract error rather than a
second, implicit fragmentation domain.

Canonicality is deliberately local, not a claim of whole-graph automorphism
canonicalization: motif identity, local atom maps, and local port labels are
canonical under atom renumbering, while the molecule-level motif sequence and
connection-ID domain preserve the frozen input logical motif order.

The byte fallback below is a closed, reversible surface encoding for motif
identities.  It is not intended to replace a tokenizer; it provides a strict
no-loss path when a motif has no frozen macro token.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

from rdkit import Chem


FORMAT_VERSION = "production_graph_ports_v1"
FALLBACK_BEGIN = "<GPORTS:FALLBACK:BEGIN>"
FALLBACK_END = "<GPORTS:FALLBACK:END>"
GRAPH_BEGIN = "<GPORTS:GRAPH:BEGIN>"
GRAPH_END = "<GPORTS:GRAPH:END>"
EDGE_ENDPOINT_A = "<GPORTS:EDGE:A>"
EDGE_ENDPOINT_B = "<GPORTS:EDGE:B>"
PORT_RADIX = "<GPORTS:RADIX:PORT>"
GPORTS_BYTE_TOKENS = tuple(f"<GPORTS:B{value:02X}>" for value in range(256))
GPORTS_BOUNDARY_TOKENS = (
    FALLBACK_BEGIN,
    FALLBACK_END,
    GRAPH_BEGIN,
    GRAPH_END,
    EDGE_ENDPOINT_A,
    EDGE_ENDPOINT_B,
    PORT_RADIX,
)
GPORTS_UNION_TOKENS = (
    *GPORTS_BOUNDARY_TOKENS,
    *GPORTS_BYTE_TOKENS,
)
_REAL_ATOM_MAP_BASE = 1


class GraphPortsContractError(ValueError):
    """Raised when the graph/partition/cross-edge contract is inconsistent."""


@dataclass(frozen=True)
class CrossEdgeInput:
    """One complete cross-motif bond supplied by the caller.

    ``bond_type`` must be the uppercase RDKit name ``"SINGLE"``, matching
    ``str(bond.GetBondType()).upper()``.
    """

    atom_a: int
    atom_b: int
    bond_type: str


@dataclass(frozen=True, order=True)
class PortRef:
    """Canonical reference to one motif-local port."""

    motif_id: int
    port_id: int


@dataclass(frozen=True, order=True)
class AtomRef:
    """Canonical reference to one real atom in one motif payload."""

    motif_id: int
    local_atom_id: int


@dataclass(frozen=True)
class PortRecord:
    """A distinguishable local cut site on a motif.

    ``port_id`` is one-based and appears as the dummy isotope in
    ``identity_smiles`` (for example ``[1*]``).  Source indices are lineage
    metadata only; reconstruction uses ``local_atom_id``.
    """

    port_id: int
    local_atom_id: int
    source_atom_index: int
    remote_source_atom_index: int
    bond_type: str


@dataclass(frozen=True)
class MotifRecord:
    """One frozen-logical motif with a canonical motif-local graph payload."""

    motif_id: int
    identity_smiles: str
    reconstruction_smiles: str
    ports: tuple[PortRecord, ...]
    source_atom_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConnectionRecord:
    """One canonical CAMT5-partition connection between motif-local ports.

    The three chemical fields remain explicit, but their production values
    are fixed to ``SINGLE``, ``STEREONONE`` and ``None`` and therefore consume
    no model tokens.
    """

    connection_id: int
    endpoint_a: PortRef
    endpoint_b: PortRef
    bond_type: str
    bond_stereo: str
    stereo_atoms: tuple[AtomRef, AtomRef] | None


@dataclass(frozen=True)
class ProductionGraphTokenStream:
    """Tokenizer-ready deterministic connection/component grammar.

    Graph grammar (integers are unsigned-varint GPORTS byte tokens)::

        GRAPH_BEGIN
          PORT_RADIX uvarint(port_radix)
          EDGE_ENDPOINT_A
            uvarint(edge_id)
            uvarint(logical_a * port_radix + port_a)
            uvarint(logical_b * port_radix + port_b)
          EDGE_ENDPOINT_B
          ...
        GRAPH_END

    ``token_roles`` and ``token_to_logical_motif`` follow the frozen P1 record
    contract: A/B endpoint markers are the only ``connection`` tokens and each
    maps to one frozen logical motif; structural/payload tokens are
    ``boundary``/``-1``.  ``connection_token_indices`` is indexed by logical
    motif and partitions all endpoint markers.  Component membership remains
    validated metadata and consumes no model tokens.  A producer inserts each
    motif identity surface, sets its first identity token as
    ``logical_to_carrier``, then offsets these graph-fragment indices.
    """

    port_radix: int
    tokens: tuple[str, ...]
    token_roles: tuple[str, ...]
    token_to_logical_motif: tuple[int, ...]
    component_token_indices: tuple[tuple[int, ...], ...]
    connection_endpoint_token_indices: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...
    ]
    connection_token_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GraphPortsEncoding:
    """Complete lossless motif graph for one projected molecule."""

    format_version: str
    strict_isomeric_identity: str
    motifs: tuple[MotifRecord, ...]
    connections: tuple[ConnectionRecord, ...]
    component_motif_ids: tuple[tuple[int, ...], ...]
    logical_motif_atom_groups: tuple[tuple[int, ...], ...]
    logical_to_canonical_motif_ids: tuple[int, ...]
    canonical_to_logical_motif_ids: tuple[int, ...]
    graph_token_stream: ProductionGraphTokenStream


@dataclass(frozen=True)
class IdentitySurface:
    """Either one frozen macro token or a framed reversible byte sequence."""

    mode: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class _Edge:
    atom_a: int
    atom_b: int
    bond_type: str
    bond_index: int

    @property
    def atom_key(self) -> tuple[int, int]:
        return (min(self.atom_a, self.atom_b), max(self.atom_a, self.atom_b))


@dataclass(frozen=True)
class _BuiltMotif:
    old_motif_id: int
    identity_smiles: str
    reconstruction_smiles: str
    ports: tuple[PortRecord, ...]
    source_atom_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _InternalBondStereo:
    """Stereo carried by a motif bond whose support may be a cut port.

    RDKit can serialize the correct E/Z state against a dummy port in a motif
    payload, but ``molzip`` removes that dummy and may choose the other
    substituent without changing the stored E/Z enum.  Keep the payload's
    actual support references until the port can be resolved to the real atom
    on the opposite motif.
    """

    bond_atoms: tuple[AtomRef, AtomRef]
    bond_stereo: str
    stereo_atoms: tuple[AtomRef | PortRef, AtomRef | PortRef]


def _bond_type_name(bond: Chem.Bond) -> str:
    return str(bond.GetBondType()).upper()


def _normalize_bond_type(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphPortsContractError("cross-edge bond_type must be a non-empty string")
    return value.strip().upper()


def _strict_isomeric_smiles(mol: Chem.Mol, *, clean_stereo: bool) -> str:
    probe = Chem.Mol(mol)
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(probe, cleanIt=clean_stereo, force=True)
    return Chem.MolToSmiles(probe, canonical=True, isomericSmiles=True)


def _canonical_atom_positions(mol: Chem.Mol) -> tuple[int, ...]:
    """Return inverse canonical-SMILES atom order, independent of input indices."""

    probe = Chem.Mol(mol)
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.MolToSmiles(probe, canonical=True, isomericSmiles=True)
    if probe.HasProp("_smilesAtomOutputOrder"):
        try:
            order = tuple(int(value) for value in ast.literal_eval(probe.GetProp("_smilesAtomOutputOrder")))
        except (SyntaxError, ValueError, TypeError) as exc:
            raise GraphPortsContractError("RDKit returned an invalid canonical atom order") from exc
    else:
        # The private output-order property is present in supported RDKit
        # releases.  This fallback keeps the codec usable on lean builds.
        ranks = tuple(
            int(value)
            for value in Chem.CanonicalRankAtoms(
                probe,
                breakTies=True,
                includeChirality=True,
                includeIsotopes=True,
            )
        )
        order = tuple(sorted(range(probe.GetNumAtoms()), key=lambda idx: (ranks[idx], idx)))
    if sorted(order) != list(range(probe.GetNumAtoms())):
        raise GraphPortsContractError("canonical atom order is not a permutation")
    positions = [0] * probe.GetNumAtoms()
    for position, atom_index in enumerate(order):
        positions[atom_index] = position
    return tuple(positions)


def _validate_partition(
    mol: Chem.Mol,
    motif_atom_groups: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    if not motif_atom_groups:
        raise GraphPortsContractError("motif_atom_groups must not be empty")
    groups: list[tuple[int, ...]] = []
    owner = [-1] * mol.GetNumAtoms()
    for motif_id, raw_group in enumerate(motif_atom_groups):
        try:
            group = tuple(sorted(int(atom_index) for atom_index in raw_group))
        except (TypeError, ValueError) as exc:
            raise GraphPortsContractError(f"motif {motif_id} contains a non-integer atom index") from exc
        if not group:
            raise GraphPortsContractError(f"motif {motif_id} is empty")
        if len(group) != len(set(group)):
            raise GraphPortsContractError(f"motif {motif_id} repeats an atom index")
        for atom_index in group:
            if atom_index < 0 or atom_index >= mol.GetNumAtoms():
                raise GraphPortsContractError(
                    f"motif {motif_id} atom index {atom_index} is outside the molecule"
                )
            if owner[atom_index] != -1:
                raise GraphPortsContractError(
                    f"atom {atom_index} occurs in motifs {owner[atom_index]} and {motif_id}"
                )
            owner[atom_index] = motif_id
        groups.append(group)
    missing = tuple(index for index, motif_id in enumerate(owner) if motif_id == -1)
    if missing:
        raise GraphPortsContractError(f"motif partition omits atoms {missing}")
    return tuple(groups), tuple(owner)


def _validate_cross_edges(
    mol: Chem.Mol,
    owner: Sequence[int],
    cross_edges: Sequence[CrossEdgeInput],
) -> tuple[_Edge, ...]:
    actual: dict[tuple[int, int], _Edge] = {}
    unsupported_cross_edges: list[tuple[tuple[int, int], str, str]] = []
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtomIdx()
        atom_b = bond.GetEndAtomIdx()
        if owner[atom_a] == owner[atom_b]:
            continue
        key = (min(atom_a, atom_b), max(atom_a, atom_b))
        actual[key] = _Edge(
            atom_a=key[0],
            atom_b=key[1],
            bond_type=_bond_type_name(bond),
            bond_index=bond.GetIdx(),
        )
        bond_type = _bond_type_name(bond)
        bond_stereo = str(bond.GetStereo())
        if bond_type != "SINGLE" or bond_stereo != "STEREONONE":
            unsupported_cross_edges.append((key, bond_type, bond_stereo))

    if unsupported_cross_edges:
        raise GraphPortsContractError(
            "CAMT5-derived cross edges must be SINGLE/STEREONONE: "
            f"{tuple(sorted(unsupported_cross_edges))}"
        )

    supplied: dict[tuple[int, int], str] = {}
    for offset, raw_edge in enumerate(cross_edges):
        if not isinstance(raw_edge, CrossEdgeInput):
            raise GraphPortsContractError(
                "cross_edges must contain CrossEdgeInput instances; upstream schema adaptation belongs at the caller"
            )
        if isinstance(raw_edge.atom_a, bool) or isinstance(raw_edge.atom_b, bool):
            raise GraphPortsContractError(f"cross edge {offset} has a boolean atom endpoint")
        try:
            atom_a = int(raw_edge.atom_a)
            atom_b = int(raw_edge.atom_b)
        except (TypeError, ValueError) as exc:
            raise GraphPortsContractError(f"cross edge {offset} has a non-integer endpoint") from exc
        if atom_a == atom_b:
            raise GraphPortsContractError(f"cross edge {offset} is a self-edge")
        if atom_a < 0 or atom_b < 0 or atom_a >= mol.GetNumAtoms() or atom_b >= mol.GetNumAtoms():
            raise GraphPortsContractError(f"cross edge {offset} endpoint is outside the molecule")
        if owner[atom_a] == owner[atom_b]:
            raise GraphPortsContractError(
                f"cross edge {offset} does not cross the motif partition"
            )
        key = (min(atom_a, atom_b), max(atom_a, atom_b))
        if key in supplied:
            raise GraphPortsContractError(f"duplicate cross edge for atoms {key}")
        supplied[key] = _normalize_bond_type(raw_edge.bond_type)

    if set(supplied) != set(actual):
        missing = tuple(sorted(set(actual) - set(supplied)))
        extra = tuple(sorted(set(supplied) - set(actual)))
        raise GraphPortsContractError(
            f"cross-edge table is not complete: missing={missing}, extra={extra}"
        )
    for key, supplied_type in supplied.items():
        if supplied_type != actual[key].bond_type:
            raise GraphPortsContractError(
                f"cross edge {key} declares {supplied_type}, molecule contains {actual[key].bond_type}"
            )
    return tuple(actual[key] for key in sorted(actual))


def _connected_component_motifs(
    motif_count: int,
    connections: Sequence[ConnectionRecord],
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(motif_count)]
    for connection in connections:
        left = connection.endpoint_a.motif_id
        right = connection.endpoint_b.motif_id
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(range(motif_count))
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(adjacency[current] - found)
        remaining -= found
        components.append(tuple(sorted(found)))
    return tuple(sorted(components))


_GPORTS_BYTE_VALUE_BY_TOKEN = {
    token: value for value, token in enumerate(GPORTS_BYTE_TOKENS)
}


def _uvarint_tokens(value: int) -> tuple[str, ...]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraphPortsContractError("uvarint value must be a non-negative integer")
    encoded: list[str] = []
    remaining = value
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            byte |= 0x80
        encoded.append(GPORTS_BYTE_TOKENS[byte])
        if not remaining:
            return tuple(encoded)


def _read_uvarint(tokens: Sequence[str], cursor: int) -> tuple[int, int]:
    start = cursor
    value = 0
    shift = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token not in _GPORTS_BYTE_VALUE_BY_TOKEN:
            raise GraphPortsContractError(f"expected uvarint byte token at graph position {cursor}")
        byte = _GPORTS_BYTE_VALUE_BY_TOKEN[token]
        value |= (byte & 0x7F) << shift
        cursor += 1
        if not byte & 0x80:
            if tuple(tokens[start:cursor]) != _uvarint_tokens(value):
                raise GraphPortsContractError("graph stream contains a non-canonical uvarint")
            return value, cursor
        shift += 7
    raise GraphPortsContractError("graph stream ends inside a uvarint")


def _build_graph_token_stream(
    components: Sequence[tuple[int, ...]],
    connections: Sequence[ConnectionRecord],
    canonical_to_logical_motif_ids: Sequence[int],
) -> ProductionGraphTokenStream:
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
        GRAPH_BEGIN,
        PORT_RADIX,
        *_uvarint_tokens(port_radix),
    )
    tokens: list[str] = list(header)
    roles: list[str] = ["boundary"] * len(header)
    token_to_logical: list[int] = [-1] * len(header)
    endpoint_indices: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    connection_indices_by_logical: list[list[int]] = [
        [] for _ in canonical_to_logical_motif_ids
    ]

    for connection in connections:
        if not (
            connection.bond_type == "SINGLE"
            and connection.bond_stereo == "STEREONONE"
            and connection.stereo_atoms is None
        ):
            raise GraphPortsContractError(
                "compact CAMT5 graph grammar requires SINGLE/STEREONONE connections"
            )
        logical_a = canonical_to_logical_motif_ids[connection.endpoint_a.motif_id]
        logical_b = canonical_to_logical_motif_ids[connection.endpoint_b.motif_id]
        payload = (
            *_uvarint_tokens(connection.connection_id),
            *_uvarint_tokens(logical_a * port_radix + connection.endpoint_a.port_id),
            *_uvarint_tokens(logical_b * port_radix + connection.endpoint_b.port_id),
        )

        endpoint_a_index = len(tokens)
        tokens.append(EDGE_ENDPOINT_A)
        roles.append("connection")
        token_to_logical.append(logical_a)
        tokens.extend(payload)
        roles.extend("boundary" for _ in payload)
        token_to_logical.extend(-1 for _ in payload)
        endpoint_b_index = len(tokens)
        tokens.append(EDGE_ENDPOINT_B)
        roles.append("connection")
        token_to_logical.append(logical_b)
        endpoint_indices.append(((endpoint_a_index,), (endpoint_b_index,)))
        connection_indices_by_logical[logical_a].append(endpoint_a_index)
        connection_indices_by_logical[logical_b].append(endpoint_b_index)

    tokens.append(GRAPH_END)
    roles.append("boundary")
    token_to_logical.append(-1)
    return ProductionGraphTokenStream(
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


def _decode_graph_token_stream(
    stream: ProductionGraphTokenStream,
    components: Sequence[tuple[int, ...]],
    logical_to_canonical_motif_ids: Sequence[int],
) -> tuple[ConnectionRecord, ...]:
    tokens = stream.tokens
    if not (
        len(tokens)
        == len(stream.token_roles)
        == len(stream.token_to_logical_motif)
    ):
        raise GraphPortsContractError("graph token arrays differ in length")
    if not tokens or tokens[0] != GRAPH_BEGIN or tokens[-1] != GRAPH_END:
        raise GraphPortsContractError("graph token stream has invalid framing")

    def require_boundary(index: int) -> None:
        if (
            stream.token_roles[index] != "boundary"
            or stream.token_to_logical_motif[index] != -1
        ):
            raise GraphPortsContractError(
                f"graph payload position {index} must be boundary/-1"
            )

    require_boundary(0)
    cursor = 1
    if cursor >= len(tokens) or tokens[cursor] != PORT_RADIX:
        raise GraphPortsContractError("graph stream omits PORT_RADIX")
    require_boundary(cursor)
    cursor += 1
    start = cursor
    port_radix, cursor = _read_uvarint(tokens, cursor)
    for index in range(start, cursor):
        require_boundary(index)
    if port_radix <= 0 or port_radix != stream.port_radix:
        raise GraphPortsContractError("graph port radix is invalid or inconsistent")

    decoded: list[ConnectionRecord] = []
    endpoint_indices: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    by_logical: list[list[int]] = [[] for _ in logical_to_canonical_motif_ids]
    while cursor < len(tokens) - 1:
        endpoint_a_index = cursor
        if tokens[cursor] != EDGE_ENDPOINT_A or stream.token_roles[cursor] != "connection":
            raise GraphPortsContractError(f"expected EDGE_ENDPOINT_A at graph position {cursor}")
        marker_logical_a = stream.token_to_logical_motif[cursor]
        cursor += 1

        payload_start = cursor
        connection_id, cursor = _read_uvarint(tokens, cursor)
        packed_a, cursor = _read_uvarint(tokens, cursor)
        packed_b, cursor = _read_uvarint(tokens, cursor)
        for index in range(payload_start, cursor):
            require_boundary(index)

        logical_a, port_a = divmod(packed_a, port_radix)
        logical_b, port_b = divmod(packed_b, port_radix)
        if (
            not (0 <= logical_a < len(logical_to_canonical_motif_ids))
            or not (0 <= logical_b < len(logical_to_canonical_motif_ids))
            or port_a <= 0
            or port_b <= 0
        ):
            raise GraphPortsContractError("graph endpoint payload is outside its radix/domain")
        endpoint_b_index = cursor
        if (
            cursor >= len(tokens)
            or tokens[cursor] != EDGE_ENDPOINT_B
            or stream.token_roles[cursor] != "connection"
        ):
            raise GraphPortsContractError(f"expected EDGE_ENDPOINT_B at graph position {cursor}")
        marker_logical_b = stream.token_to_logical_motif[cursor]
        cursor += 1
        if marker_logical_a != logical_a or marker_logical_b != logical_b:
            raise GraphPortsContractError("endpoint marker mapping disagrees with packed logical motif")
        endpoint_a = PortRef(logical_to_canonical_motif_ids[logical_a], port_a)
        endpoint_b = PortRef(logical_to_canonical_motif_ids[logical_b], port_b)
        decoded.append(
            ConnectionRecord(
                connection_id=connection_id,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                bond_type="SINGLE",
                bond_stereo="STEREONONE",
                stereo_atoms=None,
            )
        )
        endpoint_indices.append(((endpoint_a_index,), (endpoint_b_index,)))
        by_logical[logical_a].append(endpoint_a_index)
        by_logical[logical_b].append(endpoint_b_index)

    if cursor != len(tokens) - 1:
        raise GraphPortsContractError("graph token stream has trailing payload")
    require_boundary(cursor)
    if stream.component_token_indices != tuple(() for _ in components):
        raise GraphPortsContractError("components must remain metadata-only in compact graph grammar")
    if stream.connection_endpoint_token_indices != tuple(endpoint_indices):
        raise GraphPortsContractError("connection endpoint token indices disagree with compact grammar")
    if stream.connection_token_indices != tuple(tuple(row) for row in by_logical):
        raise GraphPortsContractError("logical connection token indices disagree with compact grammar")
    return tuple(decoded)


class ProductionGraphPortsCodecV1:
    """Encode and reconstruct projected molecules through motif graph ports."""

    @staticmethod
    def required_union_tokens() -> tuple[str, ...]:
        """Return every fixed GPORTS byte/boundary token required by the union tokenizer."""

        return GPORTS_UNION_TOKENS

    @staticmethod
    def decode_graph_token_stream(
        encoding: GraphPortsEncoding,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[ConnectionRecord, ...]]:
        """Decode compact model tokens back to component metadata and connections."""

        connections = _decode_graph_token_stream(
            encoding.graph_token_stream,
            encoding.component_motif_ids,
            encoding.logical_to_canonical_motif_ids,
        )
        return encoding.component_motif_ids, connections

    def encode(
        self,
        mol: Chem.Mol,
        motif_atom_groups: Sequence[Sequence[int]],
        cross_edges: Sequence[CrossEdgeInput],
    ) -> GraphPortsEncoding:
        if not isinstance(mol, Chem.Mol):
            raise GraphPortsContractError("mol must be an RDKit Mol")
        if mol.GetNumAtoms() == 0:
            raise GraphPortsContractError("mol must contain at least one atom")
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            raise GraphPortsContractError("input mol must not contain dummy atoms; they are reserved for ports")

        work = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(work)
        except Exception as exc:
            raise GraphPortsContractError("input mol cannot be sanitized") from exc
        Chem.AssignStereochemistry(work, cleanIt=True, force=True)

        groups, owner = _validate_partition(work, motif_atom_groups)
        edges = _validate_cross_edges(work, owner, cross_edges)
        positions = _canonical_atom_positions(work)

        edge_order = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    min(positions[edge.atom_a], positions[edge.atom_b]),
                    max(positions[edge.atom_a], positions[edge.atom_b]),
                    edge.bond_type,
                ),
            )
        )
        if edge_order:
            fragmented = Chem.FragmentOnBonds(
                work,
                [edge.bond_index for edge in edge_order],
                addDummies=True,
                dummyLabels=[(offset + 1, offset + 1) for offset in range(len(edge_order))],
            )
        else:
            fragmented = Chem.Mol(work)

        edge_by_tag = {offset + 1: edge for offset, edge in enumerate(edge_order)}
        dummy_by_group_and_edge: dict[tuple[int, tuple[int, int]], int] = {}
        port_specs: dict[int, list[tuple[_Edge, int, int, int]]] = {
            motif_id: [] for motif_id in range(len(groups))
        }
        for atom in fragmented.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            tag = atom.GetIsotope()
            edge = edge_by_tag.get(tag)
            if edge is None:
                raise GraphPortsContractError(f"fragmentation produced unknown dummy label {tag}")
            neighbors = tuple(atom.GetNeighbors())
            if len(neighbors) != 1:
                raise GraphPortsContractError("each port dummy must have exactly one real neighbor")
            source_atom = neighbors[0].GetIdx()
            if source_atom == edge.atom_a:
                remote_atom = edge.atom_b
            elif source_atom == edge.atom_b:
                remote_atom = edge.atom_a
            else:
                raise GraphPortsContractError("fragmentation dummy does not match its source edge")
            motif_id = owner[source_atom]
            lookup_key = (motif_id, edge.atom_key)
            if lookup_key in dummy_by_group_and_edge:
                raise GraphPortsContractError("fragmentation produced duplicate motif endpoint")
            dummy_by_group_and_edge[lookup_key] = atom.GetIdx()
            port_specs[motif_id].append((edge, source_atom, remote_atom, atom.GetIdx()))

        local_atom_ids: dict[int, dict[int, int]] = {}
        local_port_ids: dict[tuple[int, tuple[int, int]], int] = {}
        for motif_id, group in enumerate(groups):
            ordered_atoms = tuple(sorted(group, key=lambda atom_index: positions[atom_index]))
            local_atom_ids[motif_id] = {
                atom_index: local_id
                for local_id, atom_index in enumerate(ordered_atoms, start=_REAL_ATOM_MAP_BASE)
            }
            ordered_ports = sorted(
                port_specs[motif_id],
                key=lambda item: (
                    positions[item[1]],
                    # The remote logical motif row is already frozen by the
                    # upstream topology record.  Use it before a whole-graph
                    # atom-position tie-break so symmetry-related neighbours
                    # cannot exchange port IDs after RDKit atom renumbering.
                    owner[item[2]],
                    positions[item[2]],
                    item[0].bond_type,
                ),
            )
            for port_id, (edge, _source, _remote, dummy_index) in enumerate(ordered_ports, start=1):
                fragmented.GetAtomWithIdx(dummy_index).SetIsotope(port_id)
                fragmented.GetAtomWithIdx(dummy_index).SetAtomMapNum(0)
                local_port_ids[(motif_id, edge.atom_key)] = port_id

        fragment_atom_maps: list[tuple[int, ...]] = []
        fragment_mols = Chem.GetMolFrags(
            fragmented,
            asMols=True,
            sanitizeFrags=False,
            fragsMolAtomMapping=fragment_atom_maps,
        )
        fragment_by_group: dict[int, tuple[Chem.Mol, tuple[int, ...]]] = {}
        source_atom_count = work.GetNumAtoms()
        for fragment_mol, atom_mapping in zip(fragment_mols, fragment_atom_maps):
            real_atoms = tuple(sorted(index for index in atom_mapping if index < source_atom_count))
            matching = [motif_id for motif_id, group in enumerate(groups) if group == real_atoms]
            if len(matching) != 1:
                raise GraphPortsContractError(
                    "each motif group must induce exactly one connected fragment after cross-edge cutting"
                )
            motif_id = matching[0]
            if motif_id in fragment_by_group:
                raise GraphPortsContractError(f"motif {motif_id} produced more than one fragment")
            fragment_by_group[motif_id] = (fragment_mol, tuple(atom_mapping))
        if set(fragment_by_group) != set(range(len(groups))):
            raise GraphPortsContractError("not every motif group produced a fragment")

        built: list[_BuiltMotif] = []
        for motif_id, group in enumerate(groups):
            fragment_mol, atom_mapping = fragment_by_group[motif_id]
            payload = Chem.Mol(fragment_mol)
            for local_index, fragmented_index in enumerate(atom_mapping):
                atom = payload.GetAtomWithIdx(local_index)
                if fragmented_index < source_atom_count:
                    atom.SetAtomMapNum(local_atom_ids[motif_id][fragmented_index])
                else:
                    atom.SetAtomMapNum(0)
            try:
                Chem.SanitizeMol(payload)
            except Exception as exc:
                raise GraphPortsContractError(f"motif {motif_id} payload cannot be sanitized") from exc
            identity = Chem.Mol(payload)
            for atom in identity.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    atom.SetAtomMapNum(0)
            identity_smiles = Chem.MolToSmiles(identity, canonical=True, isomericSmiles=True)
            reconstruction_smiles = Chem.MolToSmiles(payload, canonical=True, isomericSmiles=True)

            records: list[PortRecord] = []
            for edge, source_atom, remote_atom, _dummy_index in port_specs[motif_id]:
                port_id = local_port_ids[(motif_id, edge.atom_key)]
                records.append(
                    PortRecord(
                        port_id=port_id,
                        local_atom_id=local_atom_ids[motif_id][source_atom],
                        source_atom_index=source_atom,
                        remote_source_atom_index=remote_atom,
                        bond_type=edge.bond_type,
                    )
                )
            records.sort(key=lambda record: record.port_id)
            source_map = tuple(
                sorted(
                    (
                        (local_id, source_atom)
                        for source_atom, local_id in local_atom_ids[motif_id].items()
                    ),
                    key=lambda item: item[0],
                )
            )
            built.append(
                _BuiltMotif(
                    old_motif_id=motif_id,
                    identity_smiles=identity_smiles,
                    reconstruction_smiles=reconstruction_smiles,
                    ports=tuple(records),
                    source_atom_map=source_map,
                )
            )

        # Molecule-level order is the frozen logical order supplied by the
        # linearizer.  Sorting identical motif identities by a whole-molecule
        # atom rank is not automorphism-canonical and can change under harmless
        # RDKit atom renumbering.  Keep compatibility mapping fields as the
        # identity permutation; canonicality remains motif-local.
        logical_to_canonical = tuple(range(len(groups)))
        canonical_to_logical = tuple(range(len(groups)))
        new_id_by_old = {motif_id: motif_id for motif_id in range(len(groups))}
        motifs = tuple(
            MotifRecord(
                motif_id=record.old_motif_id,
                identity_smiles=record.identity_smiles,
                reconstruction_smiles=record.reconstruction_smiles,
                ports=record.ports,
                source_atom_map=record.source_atom_map,
            )
            for record in built
        )

        pending_connections: list[tuple[PortRef, PortRef]] = []
        for edge in edges:
            old_a = owner[edge.atom_a]
            old_b = owner[edge.atom_b]
            endpoint_a = PortRef(
                new_id_by_old[old_a],
                local_port_ids[(old_a, edge.atom_key)],
            )
            endpoint_b = PortRef(
                new_id_by_old[old_b],
                local_port_ids[(old_b, edge.atom_key)],
            )
            if endpoint_b < endpoint_a:
                endpoint_a, endpoint_b = endpoint_b, endpoint_a

            pending_connections.append((endpoint_a, endpoint_b))

        pending_connections.sort()
        connections = tuple(
            ConnectionRecord(
                connection_id=connection_id,
                endpoint_a=item[0],
                endpoint_b=item[1],
                bond_type="SINGLE",
                bond_stereo="STEREONONE",
                stereo_atoms=None,
            )
            for connection_id, item in enumerate(pending_connections, start=1)
        )
        components = _connected_component_motifs(len(motifs), connections)
        graph_token_stream = _build_graph_token_stream(
            components,
            connections,
            canonical_to_logical,
        )
        encoding = GraphPortsEncoding(
            format_version=FORMAT_VERSION,
            strict_isomeric_identity=_strict_isomeric_smiles(work, clean_stereo=True),
            motifs=motifs,
            connections=connections,
            component_motif_ids=components,
            logical_motif_atom_groups=groups,
            logical_to_canonical_motif_ids=logical_to_canonical,
            canonical_to_logical_motif_ids=canonical_to_logical,
            graph_token_stream=graph_token_stream,
        )
        # Encoding is not considered valid until its independent reconstruction
        # reproduces the strict molecular identity.
        self.validate(encoding)
        return encoding

    def validate(self, encoding: GraphPortsEncoding) -> None:
        """Reject any structural/payload inconsistency and require strict round-trip identity."""

        self.reconstruct(encoding, verify_identity=True)

    def validate_against_source(
        self,
        projected_mol: Chem.Mol,
        motif_atom_groups: Sequence[Sequence[int]],
        cross_edges: Sequence[CrossEdgeInput],
        encoding: GraphPortsEncoding,
    ) -> None:
        """Bind an encoding to its projected Mol and frozen logical partition.

        Source-free validation proves internal graph/identity consistency.  It
        cannot distinguish a coordinated relabeling of otherwise valid ports.
        The producer should call this source-bound validator before persisting
        a record; deterministic re-encoding makes every lineage, permutation,
        port, connection, component, and token-grammar field comparable.
        """

        expected = self.encode(projected_mol, motif_atom_groups, cross_edges)
        if encoding != expected:
            raise GraphPortsContractError(
                "encoding is internally valid but does not equal the canonical encoding of its source"
            )

    def reconstruct(
        self,
        encoding: GraphPortsEncoding,
        *,
        verify_identity: bool = True,
    ) -> Chem.Mol:
        if encoding.format_version != FORMAT_VERSION:
            raise GraphPortsContractError(
                f"unsupported graph-ports format {encoding.format_version!r}"
            )
        motif_count = len(encoding.motifs)
        if motif_count == 0:
            raise GraphPortsContractError("encoding contains no motifs")
        if tuple(motif.motif_id for motif in encoding.motifs) != tuple(range(motif_count)):
            raise GraphPortsContractError("motif ids must be contiguous and canonical")
        expected_ids = tuple(range(motif_count))
        if len(encoding.logical_motif_atom_groups) != motif_count:
            raise GraphPortsContractError("logical motif groups and canonical motifs differ in count")
        if encoding.logical_to_canonical_motif_ids != expected_ids:
            raise GraphPortsContractError(
                "logical_to_canonical_motif_ids must be identity in frozen-logical order"
            )
        if encoding.canonical_to_logical_motif_ids != expected_ids:
            raise GraphPortsContractError(
                "canonical_to_logical_motif_ids must be identity in frozen-logical order"
            )

        flattened_logical_atoms: list[int] = []
        for logical_id, group in enumerate(encoding.logical_motif_atom_groups):
            if not group or tuple(sorted(set(group))) != tuple(group):
                raise GraphPortsContractError(
                    f"logical motif group {logical_id} must be a non-empty sorted unique tuple"
                )
            flattened_logical_atoms.extend(group)
        if tuple(sorted(flattened_logical_atoms)) != tuple(range(len(flattened_logical_atoms))):
            raise GraphPortsContractError("logical motif groups are not a complete atom partition")

        expected_connection_ids = tuple(range(1, len(encoding.connections) + 1))
        if tuple(connection.connection_id for connection in encoding.connections) != expected_connection_ids:
            raise GraphPortsContractError("connection ids must be contiguous and canonical")
        connection_sort_keys = tuple(
            (
                connection.endpoint_a,
                connection.endpoint_b,
                connection.bond_type,
                connection.bond_stereo,
                connection.stereo_atoms or (),
            )
            for connection in encoding.connections
        )
        if connection_sort_keys != tuple(sorted(connection_sort_keys)):
            raise GraphPortsContractError("connections are not in canonical endpoint order")
        for connection in encoding.connections:
            if not (
                0 <= connection.endpoint_a.motif_id < motif_count
                and 0 <= connection.endpoint_b.motif_id < motif_count
                and connection.endpoint_a.port_id > 0
                and connection.endpoint_b.port_id > 0
                and connection.endpoint_a < connection.endpoint_b
            ):
                raise GraphPortsContractError(
                    f"connection {connection.connection_id} has non-canonical endpoints"
                )
            if not (
                connection.bond_type == "SINGLE"
                and connection.bond_stereo == "STEREONONE"
                and connection.stereo_atoms is None
            ):
                raise GraphPortsContractError(
                    f"connection {connection.connection_id} is outside the "
                    "CAMT5 SINGLE/STEREONONE contract"
                )

        expected_components = _connected_component_motifs(motif_count, encoding.connections)
        if encoding.component_motif_ids != expected_components:
            raise GraphPortsContractError("component_motif_ids does not match the connection graph")
        expected_token_stream = _build_graph_token_stream(
            encoding.component_motif_ids,
            encoding.connections,
            encoding.canonical_to_logical_motif_ids,
        )
        if encoding.graph_token_stream != expected_token_stream:
            raise GraphPortsContractError("graph_token_stream does not match components/connections")
        decoded_components, decoded_connections = self.decode_graph_token_stream(encoding)
        if (
            decoded_components != encoding.component_motif_ids
            or decoded_connections != encoding.connections
        ):
            raise GraphPortsContractError("compact graph token decode disagrees with encoding metadata")

        connection_by_port: dict[PortRef, int] = {}
        connection_record_by_port: dict[PortRef, ConnectionRecord] = {}
        for connection in encoding.connections:
            for endpoint in (connection.endpoint_a, connection.endpoint_b):
                if endpoint in connection_by_port:
                    raise GraphPortsContractError(f"port {endpoint} occurs in more than one connection")
                connection_by_port[endpoint] = connection.connection_id
                connection_record_by_port[endpoint] = connection

        combined: Chem.Mol | None = None
        global_map_by_atom_ref: dict[AtomRef, int] = {}
        source_atom_ref_by_index: dict[int, AtomRef] = {}
        next_global_map = 1
        port_record_by_ref: dict[PortRef, PortRecord] = {}
        internal_bond_stereo: list[_InternalBondStereo] = []
        for motif in encoding.motifs:
            logical_id = encoding.canonical_to_logical_motif_ids[motif.motif_id]
            expected_source_atoms = encoding.logical_motif_atom_groups[logical_id]
            expected_local_ids = tuple(range(1, len(expected_source_atoms) + 1))
            if tuple(local_id for local_id, _source in motif.source_atom_map) != expected_local_ids:
                raise GraphPortsContractError(
                    f"motif {motif.motif_id} source atom maps are not contiguous"
                )
            if tuple(sorted(source for _local_id, source in motif.source_atom_map)) != expected_source_atoms:
                raise GraphPortsContractError(
                    f"motif {motif.motif_id} source atom map disagrees with its frozen logical motif"
                )
            source_by_local_id = dict(motif.source_atom_map)
            for local_id, source_atom in motif.source_atom_map:
                atom_ref = AtomRef(motif.motif_id, local_id)
                if source_atom in source_atom_ref_by_index:
                    raise GraphPortsContractError(f"source atom {source_atom} occurs in multiple motifs")
                source_atom_ref_by_index[source_atom] = atom_ref

            payload = Chem.MolFromSmiles(motif.reconstruction_smiles)
            if payload is None:
                raise GraphPortsContractError(f"motif {motif.motif_id} payload is not parseable")
            identity_probe = Chem.Mol(payload)
            for atom in identity_probe.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    atom.SetAtomMapNum(0)
            observed_identity = Chem.MolToSmiles(
                identity_probe,
                canonical=True,
                isomericSmiles=True,
            )
            if observed_identity != motif.identity_smiles:
                raise GraphPortsContractError(
                    f"motif {motif.motif_id} identity_smiles does not match its payload"
                )
            expected_port_ids = tuple(range(1, len(motif.ports) + 1))
            if tuple(port.port_id for port in motif.ports) != expected_port_ids:
                raise GraphPortsContractError(f"motif {motif.motif_id} port ids are not contiguous")
            port_by_id = {port.port_id: port for port in motif.ports}
            for port in motif.ports:
                if source_by_local_id.get(port.local_atom_id) != port.source_atom_index:
                    raise GraphPortsContractError(
                        f"motif {motif.motif_id} port {port.port_id} source atom map is inconsistent"
                    )
                if port.remote_source_atom_index not in flattened_logical_atoms:
                    raise GraphPortsContractError(
                        f"motif {motif.motif_id} port {port.port_id} has an unknown remote source atom"
                    )
            expected_local_id_set = set(expected_local_ids)
            seen_local_ids: set[int] = set()
            seen_port_ids: set[int] = set()

            # Capture internal E/Z before molzip removes dummy ports.  The
            # payload itself is the serialized source of this information; a
            # dummy stereo support is later resolved through its connection to
            # the real atom in the neighbouring motif.
            for internal_bond in payload.GetBonds():
                if internal_bond.GetBondType() != Chem.BondType.DOUBLE:
                    continue
                raw_stereo_atoms = tuple(int(index) for index in internal_bond.GetStereoAtoms())
                if not raw_stereo_atoms:
                    continue
                if len(raw_stereo_atoms) != 2:
                    raise GraphPortsContractError(
                        f"motif {motif.motif_id} internal stereo bond must have two references"
                    )
                begin = payload.GetAtomWithIdx(internal_bond.GetBeginAtomIdx())
                end = payload.GetAtomWithIdx(internal_bond.GetEndAtomIdx())
                begin_local = begin.GetAtomMapNum()
                end_local = end.GetAtomMapNum()
                if begin_local not in expected_local_id_set or end_local not in expected_local_id_set:
                    raise GraphPortsContractError(
                        f"motif {motif.motif_id} internal stereo bond has an unknown atom map"
                    )
                bond_stereo = str(internal_bond.GetStereo())
                if bond_stereo not in {"STEREOE", "STEREOZ", "STEREOCIS", "STEREOTRANS"}:
                    raise GraphPortsContractError(
                        f"motif {motif.motif_id} has unsupported internal bond stereo {bond_stereo!r}"
                    )
                support_refs: list[AtomRef | PortRef] = []
                for atom_index in raw_stereo_atoms:
                    support_atom = payload.GetAtomWithIdx(atom_index)
                    if support_atom.GetAtomicNum() == 0:
                        port_id = support_atom.GetIsotope()
                        if port_id not in port_by_id:
                            raise GraphPortsContractError(
                                f"motif {motif.motif_id} internal stereo references an unknown port"
                            )
                        support_refs.append(PortRef(motif.motif_id, port_id))
                    else:
                        local_atom_id = support_atom.GetAtomMapNum()
                        if local_atom_id not in expected_local_id_set:
                            raise GraphPortsContractError(
                                f"motif {motif.motif_id} internal stereo references an unknown atom map"
                            )
                        support_refs.append(AtomRef(motif.motif_id, local_atom_id))
                internal_bond_stereo.append(
                    _InternalBondStereo(
                        bond_atoms=(
                            AtomRef(motif.motif_id, begin_local),
                            AtomRef(motif.motif_id, end_local),
                        ),
                        bond_stereo=bond_stereo,
                        stereo_atoms=(support_refs[0], support_refs[1]),
                    )
                )

            for atom in payload.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    port_id = atom.GetIsotope()
                    if port_id <= 0 or port_id in seen_port_ids:
                        raise GraphPortsContractError(
                            f"motif {motif.motif_id} has an invalid or duplicate dummy port"
                        )
                    if port_id not in port_by_id:
                        raise GraphPortsContractError(
                            f"motif {motif.motif_id} payload contains unknown port {port_id}"
                        )
                    seen_port_ids.add(port_id)
                    ref = PortRef(motif.motif_id, port_id)
                    if ref not in connection_by_port:
                        raise GraphPortsContractError(f"port {ref} has no connection")
                    neighbors = tuple(atom.GetNeighbors())
                    if len(neighbors) != 1:
                        raise GraphPortsContractError(f"port {ref} must have exactly one real neighbor")
                    dummy_bond = payload.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                    if dummy_bond is None or _bond_type_name(dummy_bond) != port_by_id[port_id].bond_type:
                        raise GraphPortsContractError(
                            f"port {ref} bond_type does not match its motif payload"
                        )
                    atom.SetIsotope(connection_by_port[ref])
                    atom.SetAtomMapNum(0)
                else:
                    local_atom_id = atom.GetAtomMapNum()
                    if local_atom_id <= 0 or local_atom_id in seen_local_ids:
                        raise GraphPortsContractError(
                            f"motif {motif.motif_id} real-atom maps are missing or duplicated"
                        )
                    seen_local_ids.add(local_atom_id)
                    atom_ref = AtomRef(motif.motif_id, local_atom_id)
                    global_map_by_atom_ref[atom_ref] = next_global_map
                    atom.SetAtomMapNum(next_global_map)
                    next_global_map += 1
            if seen_local_ids != expected_local_id_set:
                raise GraphPortsContractError(
                    f"motif {motif.motif_id} payload and source atom map disagree"
                )
            if seen_port_ids != set(expected_port_ids):
                raise GraphPortsContractError(f"motif {motif.motif_id} payload and ports disagree")
            for port in motif.ports:
                port_record_by_ref[PortRef(motif.motif_id, port.port_id)] = port
            combined = payload if combined is None else Chem.CombineMols(combined, payload)

        if combined is None:
            raise GraphPortsContractError("encoding contains no motifs")
        params = Chem.MolzipParams()
        params.label = Chem.MolzipLabel.Isotope
        try:
            rebuilt = Chem.molzip(combined, params)
        except Exception as exc:
            raise GraphPortsContractError("motif ports could not be reconnected") from exc
        if any(atom.GetAtomicNum() == 0 for atom in rebuilt.GetAtoms()):
            raise GraphPortsContractError("reconstruction left unmatched port dummies")

        atom_index_by_global_map: dict[int, int] = {}
        for atom in rebuilt.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map > 0:
                atom_index_by_global_map[atom_map] = atom.GetIdx()
        atom_index_by_ref = {
            atom_ref: atom_index_by_global_map[global_map]
            for atom_ref, global_map in global_map_by_atom_ref.items()
        }

        for connection in encoding.connections:
            left_port = port_record_by_ref.get(connection.endpoint_a)
            right_port = port_record_by_ref.get(connection.endpoint_b)
            if left_port is None or right_port is None:
                raise GraphPortsContractError("connection references an unknown port")
            if (
                left_port.bond_type != connection.bond_type
                or right_port.bond_type != connection.bond_type
            ):
                raise GraphPortsContractError(
                    f"connection {connection.connection_id} and PortRecord bond types disagree"
                )
            if (
                left_port.remote_source_atom_index != right_port.source_atom_index
                or right_port.remote_source_atom_index != left_port.source_atom_index
            ):
                raise GraphPortsContractError(
                    f"connection {connection.connection_id} source/remote atom lineage disagrees"
                )
            left_ref = AtomRef(connection.endpoint_a.motif_id, left_port.local_atom_id)
            right_ref = AtomRef(connection.endpoint_b.motif_id, right_port.local_atom_id)
            bond = rebuilt.GetBondBetweenAtoms(atom_index_by_ref[left_ref], atom_index_by_ref[right_ref])
            if bond is None:
                raise GraphPortsContractError(f"connection {connection.connection_id} was not reconstructed")
            if _bond_type_name(bond) != connection.bond_type:
                raise GraphPortsContractError(
                    f"connection {connection.connection_id} reconstructed the wrong bond type"
                )

        def resolve_internal_support(ref: AtomRef | PortRef) -> int:
            if isinstance(ref, AtomRef):
                return atom_index_by_ref[ref]
            connection = connection_record_by_port.get(ref)
            if connection is None:
                raise GraphPortsContractError(f"internal stereo references unknown port {ref}")
            if connection.endpoint_a == ref:
                remote_ref = connection.endpoint_b
            elif connection.endpoint_b == ref:
                remote_ref = connection.endpoint_a
            else:  # pragma: no cover - guarded by connection_record_by_port construction
                raise GraphPortsContractError(f"internal stereo port {ref} has no remote endpoint")
            remote_port = port_record_by_ref.get(remote_ref)
            if remote_port is None:
                raise GraphPortsContractError(
                    f"internal stereo port {ref} resolves to unknown remote port {remote_ref}"
                )
            return atom_index_by_ref[
                AtomRef(remote_ref.motif_id, remote_port.local_atom_id)
            ]

        for internal_stereo in internal_bond_stereo:
            left_index, right_index = (
                atom_index_by_ref[internal_stereo.bond_atoms[0]],
                atom_index_by_ref[internal_stereo.bond_atoms[1]],
            )
            bond = rebuilt.GetBondBetweenAtoms(left_index, right_index)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                raise GraphPortsContractError("internal stereo bond was not reconstructed")
            stereo_indices = tuple(
                resolve_internal_support(ref) for ref in internal_stereo.stereo_atoms
            )
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            if (
                rebuilt.GetBondBetweenAtoms(begin_atom, stereo_indices[0]) is not None
                and rebuilt.GetBondBetweenAtoms(end_atom, stereo_indices[1]) is not None
            ):
                begin_reference, end_reference = stereo_indices
            elif (
                rebuilt.GetBondBetweenAtoms(begin_atom, stereo_indices[1]) is not None
                and rebuilt.GetBondBetweenAtoms(end_atom, stereo_indices[0]) is not None
            ):
                begin_reference, end_reference = stereo_indices[1], stereo_indices[0]
            else:
                raise GraphPortsContractError(
                    "internal stereo references are not adjacent to their double bond"
                )
            bond.SetStereoAtoms(begin_reference, end_reference)
            bond.SetStereo(getattr(Chem.BondStereo, internal_stereo.bond_stereo))

        try:
            Chem.SanitizeMol(rebuilt)
        except Exception as exc:
            raise GraphPortsContractError("reconnected full molecule cannot be sanitized") from exc
        for atom in rebuilt.GetAtoms():
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistry(rebuilt, cleanIt=False, force=True)
        rebuilt_identity = _strict_isomeric_smiles(rebuilt, clean_stereo=False)
        if verify_identity and rebuilt_identity != encoding.strict_isomeric_identity:
            raise GraphPortsContractError(
                "strict-isomeric round-trip mismatch: "
                f"expected={encoding.strict_isomeric_identity!r}, rebuilt={rebuilt_identity!r}"
            )
        return rebuilt

    @staticmethod
    def encode_identity_surface(
        identity: str,
        *,
        macro_by_identity: Mapping[str, str] | None = None,
        force_fallback: bool = False,
    ) -> IdentitySurface:
        """Encode one strict motif identity as macro or reversible byte tokens."""

        if not isinstance(identity, str) or not identity:
            raise GraphPortsContractError("identity must be a non-empty string")
        macros = macro_by_identity or {}
        macro_tokens: list[str] = []
        for macro_identity, token in macros.items():
            if not isinstance(macro_identity, str) or not macro_identity:
                raise GraphPortsContractError("macro identity must be a non-empty string")
            if not isinstance(token, str) or not token:
                raise GraphPortsContractError("macro token must be a non-empty string")
            macro_tokens.append(token)
        if len(macro_tokens) != len(set(macro_tokens)):
            raise GraphPortsContractError(
                "macro identity-to-token mapping must be injective"
            )
        if not force_fallback and identity in macros:
            token = macros[identity]
            return IdentitySurface(mode="macro", tokens=(token,))
        byte_tokens = tuple(GPORTS_BYTE_TOKENS[value] for value in identity.encode("utf-8"))
        return IdentitySurface(
            mode="fallback",
            tokens=(FALLBACK_BEGIN, *byte_tokens, FALLBACK_END),
        )

    @staticmethod
    def decode_identity_surface(
        surface: IdentitySurface,
        *,
        identity_by_macro: Mapping[str, str] | None = None,
    ) -> str:
        """Invert :meth:`encode_identity_surface` with strict framing checks."""

        if surface.mode == "macro":
            if len(surface.tokens) != 1:
                raise GraphPortsContractError("macro surface must contain exactly one token")
            macros = identity_by_macro or {}
            try:
                return macros[surface.tokens[0]]
            except KeyError as exc:
                raise GraphPortsContractError(f"unknown macro token {surface.tokens[0]!r}") from exc
        if surface.mode != "fallback":
            raise GraphPortsContractError(f"unknown identity surface mode {surface.mode!r}")
        if len(surface.tokens) < 3 or surface.tokens[0] != FALLBACK_BEGIN or surface.tokens[-1] != FALLBACK_END:
            raise GraphPortsContractError("fallback surface has invalid framing")
        values: list[int] = []
        for token in surface.tokens[1:-1]:
            if len(token) != len("<GPORTS:B00>") or not token.startswith("<GPORTS:B") or not token.endswith(">"):
                raise GraphPortsContractError(f"invalid fallback byte token {token!r}")
            try:
                values.append(int(token[-3:-1], 16))
            except ValueError as exc:
                raise GraphPortsContractError(f"invalid fallback byte token {token!r}") from exc
        try:
            return bytes(values).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GraphPortsContractError("fallback bytes are not valid UTF-8") from exc


__all__ = [
    "AtomRef",
    "ConnectionRecord",
    "CrossEdgeInput",
    "EDGE_ENDPOINT_A",
    "EDGE_ENDPOINT_B",
    "FALLBACK_BEGIN",
    "FALLBACK_END",
    "GPORTS_BOUNDARY_TOKENS",
    "GPORTS_BYTE_TOKENS",
    "GPORTS_UNION_TOKENS",
    "FORMAT_VERSION",
    "GRAPH_BEGIN",
    "GRAPH_END",
    "GraphPortsContractError",
    "GraphPortsEncoding",
    "IdentitySurface",
    "MotifRecord",
    "PortRecord",
    "PortRef",
    "PORT_RADIX",
    "ProductionGraphTokenStream",
    "ProductionGraphPortsCodecV1",
]
