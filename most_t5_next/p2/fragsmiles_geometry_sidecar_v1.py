"""Unified model-facing geometry addresses for fragSMILES surfaces.

The authoritative chemistry codecs deliberately remain in :mod:`p1`.  This
module is a pure projection into the training ABI shared by registered macro
fragments, locally lexed fragments, and the whole-molecule lossless fallback.

Official fragSMILES omits a connector ``<n>`` whenever an endpoint fragment
has exactly one possible attachment atom.  Consequently an edge does not
always own two serialized connector records.  The ABI records that distinction
explicitly: a serialized endpoint uses the terminal connector token as its
carrier, while an implicit endpoint uses its fragment carrier.  No token is
invented and both endpoints still retain exact atom/E3FP addresses.  The
whole-molecule fallback is deliberately *not* projected as a degenerate motif:
it owns zero fragments and uses ``<eom>`` only as a molecule-summary carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Optional, Sequence

from rdkit import Chem

from most_t5_next.p1.audit_fragsmiles_adoption_v1 import (
    FragSmilesRecord,
)
from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import (
    ATOM_PREFIX,
    BOND_PREFIX,
    CompactStereoSurface,
    _is_fragment_token,
)
from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import (
    LosslessFallbackSurface,
)
from most_t5_next.p1.fragsmiles_macro_fallback_surface_v1 import (
    BRANCH_CLOSE,
    BRANCH_OPEN,
    COMPONENT,
    CONNECTOR_END,
    CONNECTOR_PREFIX,
    FragSmilesModelSurface,
    decode_model_tokens,
)
from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
    smiles_glyph_token_map,
)


SCHEMA_VERSION = "most-t5-next/fragsmiles-geometry-sidecar/v2"
MOLECULE_BEGIN = "<bom>"
MOLECULE_END = "<eom>"
_CONNECTOR_RE = re.compile(r"^<([0-9]+)>$")
_NUMBER_WIDTH = 3


class FragSmilesGeometrySidecarError(ValueError):
    """The serialized surface and its geometry addresses disagree."""


@dataclass(frozen=True)
class AtomAxisAddress:
    """One explicit correspondence across source, projection and E3FP axes."""

    source_sdf_atom_index: int
    projected_atom_index: int
    e3fp_row: int

    def __post_init__(self) -> None:
        if min(
            self.source_sdf_atom_index,
            self.projected_atom_index,
            self.e3fp_row,
        ) < 0:
            raise FragSmilesGeometrySidecarError("atom-axis addresses must be nonnegative")


@dataclass(frozen=True)
class FragmentGeometryAddress:
    fragment_index: int
    component_index: int
    identity: str
    token_start: int
    token_stop: int
    carrier_token_index: int
    representation: str
    e3fp_rows: tuple[int, ...]
    projected_atom_indices: tuple[int, ...]
    source_sdf_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class AtomGeometryAddress:
    e3fp_row: Optional[int]
    has_e3fp_row: bool
    projected_atom_index: int
    source_sdf_atom_index: int
    component_index: int
    fragment_index: Optional[int]
    fragment_local_atom_index: Optional[int]
    fragment_carrier_token_index: Optional[int]
    token_start: Optional[int]
    token_stop: Optional[int]
    is_attachment: bool


@dataclass(frozen=True)
class ConnectorEndpointGeometryAddress:
    connector_index: int
    side: str
    fragment_index: int
    fragment_local_atom_index: int
    e3fp_row: int
    projected_atom_index: int
    source_sdf_atom_index: int
    explicit_in_surface: bool
    token_start: Optional[int]
    token_stop: Optional[int]
    carrier_token_index: int


@dataclass(frozen=True)
class ConnectorGeometryAddress:
    connector_index: int
    left: ConnectorEndpointGeometryAddress
    right: ConnectorEndpointGeometryAddress


@dataclass(frozen=True)
class FragSmilesGeometrySidecar:
    schema_version: str
    mode: str
    model_tokens: tuple[str, ...]
    token_roles: tuple[str, ...]
    token_to_fragment: tuple[int, ...]
    fragments: tuple[FragmentGeometryAddress, ...]
    atoms: tuple[AtomGeometryAddress, ...]
    connectors: tuple[ConnectorGeometryAddress, ...]
    component_count: int
    molecule_carrier_token_index: Optional[int]
    fallback_mode: bool
    padding_materialized: bool

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise FragSmilesGeometrySidecarError("unexpected sidecar schema")
        if self.mode not in {"compact", "whole_molecule_fallback"}:
            raise FragSmilesGeometrySidecarError("unknown sidecar mode")
        if not self.model_tokens or self.component_count <= 0:
            raise FragSmilesGeometrySidecarError("empty model-facing surface")
        if not (
            len(self.model_tokens)
            == len(self.token_roles)
            == len(self.token_to_fragment)
        ):
            raise FragSmilesGeometrySidecarError("token metadata lengths disagree")
        if self.fallback_mode != (self.mode == "whole_molecule_fallback"):
            raise FragSmilesGeometrySidecarError("fallback mode flag disagrees")
        if self.padding_materialized:
            raise FragSmilesGeometrySidecarError(
                "record sidecars must remain unpadded; padding belongs in collate"
            )
        if tuple(row.fragment_index for row in self.fragments) != tuple(
            range(len(self.fragments))
        ):
            raise FragSmilesGeometrySidecarError("fragment indices are not dense")
        e3fp_rows = sorted(
            row.e3fp_row for row in self.atoms if row.e3fp_row is not None
        )
        if e3fp_rows != list(range(len(e3fp_rows))):
            raise FragSmilesGeometrySidecarError("E3FP rows are not a dense partition")
        if any(row.has_e3fp_row != (row.e3fp_row is not None) for row in self.atoms):
            raise FragSmilesGeometrySidecarError("has_e3fp_row flag disagrees")
        if len({row.projected_atom_index for row in self.atoms}) != len(self.atoms):
            raise FragSmilesGeometrySidecarError("projected atom indices repeat")
        if len({row.source_sdf_atom_index for row in self.atoms}) != len(self.atoms):
            raise FragSmilesGeometrySidecarError("source SDF atom indices repeat")
        atom_by_row = {
            row.e3fp_row: row for row in self.atoms if row.e3fp_row is not None
        }
        if self.mode == "compact":
            if self.molecule_carrier_token_index is not None or not self.fragments:
                raise FragSmilesGeometrySidecarError(
                    "compact records require fragments and no molecule carrier"
                )
            if any(
                row.fragment_index is None
                or row.fragment_local_atom_index is None
                or row.fragment_carrier_token_index is None
                or row.e3fp_row is None
                for row in self.atoms
            ):
                raise FragSmilesGeometrySidecarError(
                    "compact atoms require fragment ownership and E3FP rows"
                )
        else:
            if self.fragments or self.connectors:
                raise FragSmilesGeometrySidecarError(
                    "whole-molecule fallback cannot masquerade as a motif graph"
                )
            if (
                self.molecule_carrier_token_index != len(self.model_tokens) - 1
                or self.model_tokens[0] != MOLECULE_BEGIN
                or self.model_tokens[-1] != MOLECULE_END
                or any(row != -1 for row in self.token_to_fragment)
            ):
                raise FragSmilesGeometrySidecarError(
                    "whole-molecule fallback has an invalid molecule envelope"
                )
            if any(
                row.fragment_index is not None
                or row.fragment_local_atom_index is not None
                or row.fragment_carrier_token_index is not None
                or row.is_attachment
                for row in self.atoms
            ):
                raise FragSmilesGeometrySidecarError(
                    "whole-molecule atoms cannot own motif fields"
                )
        for fragment in self.fragments:
            if not (
                0 <= fragment.component_index < self.component_count
                and 0 <= fragment.token_start < fragment.token_stop <= len(self.model_tokens)
                and fragment.token_start
                <= fragment.carrier_token_index
                < fragment.token_stop
            ):
                raise FragSmilesGeometrySidecarError("invalid fragment address")
            if not (
                len(fragment.e3fp_rows)
                == len(fragment.projected_atom_indices)
                == len(fragment.source_sdf_atom_indices)
            ):
                raise FragSmilesGeometrySidecarError("fragment atom arrays disagree")
            for e3fp_row, projected_atom, source_atom in zip(
                fragment.e3fp_rows,
                fragment.projected_atom_indices,
                fragment.source_sdf_atom_indices,
            ):
                atom = atom_by_row.get(e3fp_row)
                if (
                    atom is None
                    or atom.fragment_index != fragment.fragment_index
                    or atom.projected_atom_index != projected_atom
                    or atom.source_sdf_atom_index != source_atom
                ):
                    raise FragSmilesGeometrySidecarError(
                        "fragment and atom geometry addresses disagree"
                    )
        seen_connectors = set()
        for connector in self.connectors:
            if connector.connector_index in seen_connectors:
                raise FragSmilesGeometrySidecarError("duplicate connector index")
            seen_connectors.add(connector.connector_index)
            for endpoint in (connector.left, connector.right):
                atom = atom_by_row.get(endpoint.e3fp_row)
                if (
                    atom is None
                    or endpoint.connector_index != connector.connector_index
                    or atom.fragment_index != endpoint.fragment_index
                    or atom.fragment_local_atom_index
                    != endpoint.fragment_local_atom_index
                    or atom.projected_atom_index != endpoint.projected_atom_index
                    or atom.source_sdf_atom_index != endpoint.source_sdf_atom_index
                    or not atom.is_attachment
                ):
                    raise FragSmilesGeometrySidecarError(
                        "connector endpoint does not bind its attachment atom"
                    )
                if endpoint.explicit_in_surface:
                    if (
                        endpoint.token_start is None
                        or endpoint.token_stop is None
                        or not (
                            0
                            <= endpoint.token_start
                            < endpoint.token_stop
                            <= len(self.model_tokens)
                        )
                        or endpoint.carrier_token_index != endpoint.token_stop - 1
                        or self.model_tokens[endpoint.carrier_token_index]
                        != CONNECTOR_END
                    ):
                        raise FragSmilesGeometrySidecarError(
                            "explicit connector endpoint has an invalid token span"
                        )
                elif endpoint.token_start is not None or endpoint.token_stop is not None:
                    raise FragSmilesGeometrySidecarError(
                        "implicit connector endpoint unexpectedly owns tokens"
                    )


@dataclass(frozen=True)
class _Endpoint:
    local_atom_index: int
    connectivity_token_index: Optional[int]


@dataclass(frozen=True)
class _Node:
    fragment_index: int
    potential_linkers: tuple[int, ...]


@dataclass(frozen=True)
class _ParsedEdge:
    first_fragment: int
    second_fragment: int
    first_endpoint: _Endpoint
    second_endpoint: _Endpoint


def _fragment_potential_linkers(fragment_smiles: str) -> tuple[int, ...]:
    identity = fragment_smiles.split("|", 1)[0]
    mol = Chem.MolFromSmiles(identity)
    if mol is None:
        raise FragSmilesGeometrySidecarError("invalid fragment identity")
    return tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetTotalNumHs() > 0)


class _ConnectivityParser:
    """Mirror the pinned chemicalgof parser while retaining token origins."""

    def __init__(self, record: FragSmilesRecord):
        self.record = record
        self.fragment_cursor = 0
        self.edges: list[_ParsedEdge] = []

    @staticmethod
    def _extract_branching(
        sequence: list[tuple[str, int]], starting_index: int
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        main = sequence[:starting_index]
        branch = sequence[starting_index + 1 :]
        opened, closed, index = 1, 0, 0
        while opened != closed and index < len(branch):
            token = branch[index][0]
            if token == "(":
                opened += 1
            elif token == ")":
                closed += 1
            index += 1
        if opened != closed:
            raise FragSmilesGeometrySidecarError("unbalanced fragSMILES branch")
        return main + branch[index:], branch[: index - 1]

    def parse(
        self,
        sequence: list[tuple[str, int]],
        ascendent_node: Optional[_Node] = None,
        ascendent_edge: Optional[dict[str, object]] = None,
    ) -> None:
        asc = {} if ascendent_edge is None else ascendent_edge
        desc: dict[str, object] = {}
        index = 0
        while index < len(sequence):
            token, origin = sequence[index]
            if token == "(":
                if ascendent_node is None:
                    raise FragSmilesGeometrySidecarError("orphan branch")
                sequence, branch = self._extract_branching(sequence, index)
                if not desc and not asc:
                    raise FragSmilesGeometrySidecarError("branch lacks connector")
                branch_edge = dict(desc if desc else asc)
                self.parse(branch, ascendent_node, branch_edge)
                desc.clear()
                index -= 1
            elif _is_fragment_token(token):
                if self.fragment_cursor >= len(self.record.fragments):
                    raise FragSmilesGeometrySidecarError("too many fragment tokens")
                row = self.record.fragments[self.fragment_cursor]
                if row.fragment_smiles != token:
                    raise FragSmilesGeometrySidecarError(
                        "fragment sequence differs from authoritative sidecar"
                    )
                node = _Node(
                    fragment_index=self.fragment_cursor,
                    potential_linkers=_fragment_potential_linkers(token),
                )
                self.fragment_cursor += 1
                if ascendent_node is not None:
                    if asc and desc and len(node.potential_linkers) == 1:
                        raise FragSmilesGeometrySidecarError(
                            "single-linker fragment also specifies a connector"
                        )
                    if asc and not desc and len(node.potential_linkers) > 1:
                        raise FragSmilesGeometrySidecarError(
                            "multi-linker fragment omits its connector"
                        )
                    if not asc:
                        raise FragSmilesGeometrySidecarError(
                            "ascendent fragment omits its connector"
                        )
                    if not desc and len(node.potential_linkers) == 1:
                        desc.update(
                            local_atom_index=node.potential_linkers[0],
                            connectivity_token_index=None,
                        )
                    self.edges.append(
                        _ParsedEdge(
                            first_fragment=ascendent_node.fragment_index,
                            second_fragment=node.fragment_index,
                            first_endpoint=_Endpoint(
                                int(asc["local_atom_index"]),
                                asc.get("connectivity_token_index"),  # type: ignore[arg-type]
                            ),
                            second_endpoint=_Endpoint(
                                int(desc["local_atom_index"]),
                                desc.get("connectivity_token_index"),  # type: ignore[arg-type]
                            ),
                        )
                    )
                    desc.clear()
                    asc.clear()
                if len(node.potential_linkers) == 1:
                    asc.update(
                        local_atom_index=node.potential_linkers[0],
                        connectivity_token_index=None,
                    )
                ascendent_node = node
            else:
                match = _CONNECTOR_RE.fullmatch(token)
                if match is None:
                    raise FragSmilesGeometrySidecarError(
                        f"unexpected connectivity token: {token}"
                    )
                edge = {
                    "local_atom_index": int(match.group(1)),
                    "connectivity_token_index": origin,
                }
                if asc and desc:
                    raise FragSmilesGeometrySidecarError(
                        "too many consecutive connector records"
                    )
                if asc:
                    desc.update(edge)
                else:
                    asc.update(edge)
            index += 1

    def run(self) -> tuple[_ParsedEdge, ...]:
        component: list[tuple[str, int]] = []
        for origin, token in enumerate(self.record.tokens):
            if token == "<COMP>":
                if not component:
                    raise FragSmilesGeometrySidecarError("empty component")
                self.parse(component)
                component = []
            else:
                component.append((token, origin))
        if not component:
            raise FragSmilesGeometrySidecarError("empty component")
        self.parse(component)
        if self.fragment_cursor != len(self.record.fragments):
            raise FragSmilesGeometrySidecarError("fragment traversal is incomplete")
        return tuple(self.edges)


def _connectivity_to_compact_indices(
    compact: CompactStereoSurface,
) -> tuple[int, ...]:
    indices = []
    index = 0
    while index < len(compact.tokens):
        token = compact.tokens[index]
        if token.startswith(ATOM_PREFIX):
            index += 1 + _NUMBER_WIDTH
            continue
        if token.startswith(BOND_PREFIX):
            index += 1 + 2 * _NUMBER_WIDTH
            continue
        indices.append(index)
        index += 1
    if tuple(compact.tokens[index] for index in indices) != compact.connectivity_record.tokens:
        raise FragSmilesGeometrySidecarError(
            "compact stereo records do not reduce to connectivity tokens"
        )
    return tuple(indices)


def _compact_model_token_spans(
    compact: CompactStereoSurface, model: FragSmilesModelSurface
) -> tuple[tuple[int, int], ...]:
    spans = []
    cursor = 0
    fragment_index = -1
    for token in compact.tokens:
        if _is_fragment_token(token):
            fragment_index += 1
            phrase = model.fragment_phrases[fragment_index]
            if phrase.token_start != cursor:
                raise FragSmilesGeometrySidecarError("fragment/model cursor drift")
            span = (phrase.token_start, phrase.token_stop)
        else:
            match = _CONNECTOR_RE.fullmatch(token)
            width = 2 + len(match.group(1)) if match is not None else 1
            span = (cursor, cursor + width)
        if not (0 <= span[0] < span[1] <= len(model.tokens)):
            raise FragSmilesGeometrySidecarError("compact/model span is invalid")
        spans.append(span)
        cursor = span[1]
    if cursor != len(model.tokens):
        raise FragSmilesGeometrySidecarError("compact/model token count drift")
    return tuple(spans)


def _component_by_fragment(record: FragSmilesRecord) -> tuple[int, ...]:
    output = []
    component = 0
    for token in record.tokens:
        if token == "<COMP>":
            component += 1
        elif _is_fragment_token(token):
            output.append(component)
    if len(output) != len(record.fragments):
        raise FragSmilesGeometrySidecarError("component/fragment count drift")
    return tuple(output)


def build_compact_geometry_sidecar(
    compact: CompactStereoSurface,
    model: FragSmilesModelSurface,
    macro_rows: Sequence[dict[str, object]],
    atom_axes: Sequence[AtomAxisAddress],
) -> FragSmilesGeometrySidecar:
    """Bind a reversible compact macro/local-fallback surface to geometry rows."""

    record = compact.connectivity_record
    if model.compact_tokens != compact.tokens:
        raise FragSmilesGeometrySidecarError("model surface belongs to another record")
    if decode_model_tokens(model.tokens, macro_rows) != compact.tokens:
        raise FragSmilesGeometrySidecarError("model surface is not reversible")
    compact_spans = _compact_model_token_spans(compact, model)
    connectivity_compact = _connectivity_to_compact_indices(compact)
    connectivity_model_spans = tuple(compact_spans[index] for index in connectivity_compact)

    e3fp_by_local = {
        (row.fragment_index, row.fragment_local_atom_index): row.e3fp_row
        for row in model.atom_addresses
    }
    axes_by_e3fp = {row.e3fp_row: row for row in atom_axes}
    if len(axes_by_e3fp) != len(atom_axes):
        raise FragSmilesGeometrySidecarError("duplicate E3FP row in atom axes")
    expected_e3fp = set(e3fp_by_local.values())
    if set(axes_by_e3fp) != expected_e3fp:
        raise FragSmilesGeometrySidecarError(
            "atom axes do not exactly cover the compact E3FP rows"
        )
    if len({row.projected_atom_index for row in atom_axes}) != len(atom_axes):
        raise FragSmilesGeometrySidecarError("projected atom axis is not bijective")
    if len({row.source_sdf_atom_index for row in atom_axes}) != len(atom_axes):
        raise FragSmilesGeometrySidecarError("source SDF atom axis is not bijective")
    carrier_by_fragment = {
        row.fragment_index: row.carrier_token_index for row in model.fragment_phrases
    }
    canonical_keys = {
        (fragment.sequence_fragment_index, local)
        for fragment in record.fragments
        for local, _source in enumerate(fragment.source_atom_indices)
    }
    if canonical_keys != set(e3fp_by_local):
        raise FragSmilesGeometrySidecarError("source and E3FP atom partitions differ")

    parsed_edges = _ConnectivityParser(record).run()
    parsed_by_pair = {
        frozenset((row.first_fragment, row.second_fragment)): row
        for row in parsed_edges
    }
    if len(parsed_by_pair) != len(parsed_edges):
        raise FragSmilesGeometrySidecarError("parallel fragment edges are unsupported")

    attachment_keys = set()
    connectors = []
    for authoritative in record.connectors:
        pair = frozenset(
            (authoritative.left_fragment_index, authoritative.right_fragment_index)
        )
        parsed = parsed_by_pair.pop(pair, None)
        if parsed is None:
            raise FragSmilesGeometrySidecarError("connector edge is absent from syntax")

        def endpoint(side: str) -> ConnectorEndpointGeometryAddress:
            fragment_index = (
                authoritative.left_fragment_index
                if side == "left"
                else authoritative.right_fragment_index
            )
            local_index = (
                authoritative.left_local_atom_index
                if side == "left"
                else authoritative.right_local_atom_index
            )
            parsed_endpoint = (
                parsed.first_endpoint
                if parsed.first_fragment == fragment_index
                else parsed.second_endpoint
            )
            if parsed_endpoint.local_atom_index != local_index:
                raise FragSmilesGeometrySidecarError(
                    "serialized connector selects a different local atom"
                )
            explicit = parsed_endpoint.connectivity_token_index is not None
            if explicit:
                token_start, token_stop = connectivity_model_spans[
                    parsed_endpoint.connectivity_token_index  # type: ignore[index]
                ]
                if (
                    model.tokens[token_start] != CONNECTOR_PREFIX
                    or model.tokens[token_stop - 1] != CONNECTOR_END
                ):
                    raise FragSmilesGeometrySidecarError(
                        "connector model span has invalid boundaries"
                    )
                carrier = token_stop - 1
            else:
                token_start = token_stop = None
                carrier = carrier_by_fragment[fragment_index]
            key = (fragment_index, local_index)
            attachment_keys.add(key)
            axis = axes_by_e3fp[e3fp_by_local[key]]
            return ConnectorEndpointGeometryAddress(
                connector_index=authoritative.connector_index,
                side=side,
                fragment_index=fragment_index,
                fragment_local_atom_index=local_index,
                e3fp_row=e3fp_by_local[key],
                projected_atom_index=axis.projected_atom_index,
                source_sdf_atom_index=axis.source_sdf_atom_index,
                explicit_in_surface=explicit,
                token_start=token_start,
                token_stop=token_stop,
                carrier_token_index=carrier,
            )

        connectors.append(
            ConnectorGeometryAddress(
                connector_index=authoritative.connector_index,
                left=endpoint("left"),
                right=endpoint("right"),
            )
        )
    if parsed_by_pair:
        raise FragSmilesGeometrySidecarError("syntax contains an unknown edge")

    component_by_fragment = _component_by_fragment(record)
    fragments = []
    for phrase, authoritative in zip(model.fragment_phrases, record.fragments):
        local_rows = tuple(
            sorted(
                (local, e3fp_by_local[(phrase.fragment_index, local)])
                for local in range(len(authoritative.source_atom_indices))
            )
        )
        fragments.append(
            FragmentGeometryAddress(
                fragment_index=phrase.fragment_index,
                component_index=component_by_fragment[phrase.fragment_index],
                identity=phrase.fragment_smiles,
                token_start=phrase.token_start,
                token_stop=phrase.token_stop,
                carrier_token_index=phrase.carrier_token_index,
                representation="macro" if phrase.macro_used else "fragment_lexer",
                e3fp_rows=tuple(row for _local, row in local_rows),
                projected_atom_indices=tuple(
                    axes_by_e3fp[row].projected_atom_index
                    for _local, row in local_rows
                ),
                source_sdf_atom_indices=tuple(
                    axes_by_e3fp[row].source_sdf_atom_index
                    for _local, row in local_rows
                ),
            )
        )
    atoms = tuple(
        AtomGeometryAddress(
            e3fp_row=e3fp_row,
            has_e3fp_row=True,
            projected_atom_index=axes_by_e3fp[e3fp_row].projected_atom_index,
            source_sdf_atom_index=axes_by_e3fp[e3fp_row].source_sdf_atom_index,
            component_index=component_by_fragment[fragment_index],
            fragment_index=fragment_index,
            fragment_local_atom_index=local_index,
            fragment_carrier_token_index=carrier_by_fragment[fragment_index],
            token_start=None,
            token_stop=None,
            is_attachment=(fragment_index, local_index) in attachment_keys,
        )
        for (fragment_index, local_index), e3fp_row in sorted(
            e3fp_by_local.items(), key=lambda item: item[1]
        )
    )

    roles = ["control"] * len(model.tokens)
    token_to_fragment = [-1] * len(model.tokens)
    for fragment in fragments:
        for index in range(fragment.token_start, fragment.token_stop):
            roles[index] = "fragment_phrase"
            token_to_fragment[index] = fragment.fragment_index
    for connector in connectors:
        for endpoint in (connector.left, connector.right):
            if endpoint.explicit_in_surface:
                for index in range(endpoint.token_start, endpoint.token_stop):  # type: ignore[arg-type]
                    roles[index] = "connector_endpoint"
                    token_to_fragment[index] = endpoint.fragment_index
    for index, token in enumerate(model.tokens):
        if token in {BRANCH_OPEN, BRANCH_CLOSE}:
            roles[index] = "branch"
        elif token == COMPONENT:
            roles[index] = "component"
        elif token.startswith("<ST:"):
            roles[index] = "stereo_record"

    # Attribute the complete fixed-arity stereo record, including its numeric
    # payload, to the fragment most recently serialized.  These tokens are not
    # part of the identity phrase span, but their ownership is useful to a
    # cache/collator and must not be inferred again there.
    fragment_index = -1
    compact_index = 0
    while compact_index < len(compact.tokens):
        token = compact.tokens[compact_index]
        if _is_fragment_token(token):
            fragment_index += 1
            compact_index += 1
            continue
        if token.startswith(ATOM_PREFIX):
            width = 1 + _NUMBER_WIDTH
        elif token.startswith(BOND_PREFIX):
            width = 1 + 2 * _NUMBER_WIDTH
        else:
            compact_index += 1
            continue
        if fragment_index < 0:
            raise FragSmilesGeometrySidecarError("orphan compact stereo record")
        for owned_compact_index in range(compact_index, compact_index + width):
            start, stop = compact_spans[owned_compact_index]
            for model_index in range(start, stop):
                roles[model_index] = "stereo_record"
                token_to_fragment[model_index] = fragment_index
        compact_index += width

    # Every molecular representation shares the same explicit modality
    # envelope.  Official fragSMILES remains unchanged inside the envelope;
    # only model-facing addresses are shifted by the leading boundary.
    offset = 1
    bounded_fragments = tuple(
        replace(
            row,
            token_start=row.token_start + offset,
            token_stop=row.token_stop + offset,
            carrier_token_index=row.carrier_token_index + offset,
        )
        for row in fragments
    )
    bounded_atoms = tuple(
        replace(
            row,
            fragment_carrier_token_index=(
                None
                if row.fragment_carrier_token_index is None
                else row.fragment_carrier_token_index + offset
            ),
        )
        for row in atoms
    )

    def _bounded_endpoint(row: ConnectorEndpointGeometryAddress):
        return replace(
            row,
            token_start=(None if row.token_start is None else row.token_start + offset),
            token_stop=(None if row.token_stop is None else row.token_stop + offset),
            carrier_token_index=row.carrier_token_index + offset,
        )

    bounded_connectors = tuple(
        replace(
            row,
            left=_bounded_endpoint(row.left),
            right=_bounded_endpoint(row.right),
        )
        for row in connectors
    )
    return FragSmilesGeometrySidecar(
        schema_version=SCHEMA_VERSION,
        mode="compact",
        model_tokens=(MOLECULE_BEGIN,) + model.tokens + (MOLECULE_END,),
        token_roles=("molecule_boundary",) + tuple(roles) + ("molecule_boundary",),
        token_to_fragment=(-1,) + tuple(token_to_fragment) + (-1,),
        fragments=bounded_fragments,
        atoms=bounded_atoms,
        connectors=bounded_connectors,
        component_count=len(record.component_surfaces),
        molecule_carrier_token_index=None,
        fallback_mode=False,
        padding_materialized=False,
    )


def build_fallback_geometry_sidecar(
    fallback: LosslessFallbackSurface,
    atom_axes: Sequence[AtomAxisAddress],
) -> FragSmilesGeometrySidecar:
    """Expose fallback as one molecule envelope with zero logical motifs."""

    dot_token = dict(smiles_glyph_token_map())["."]
    dot_positions = tuple(
        index for index, token in enumerate(fallback.tokens) if token == dot_token
    )
    component_identities = tuple(fallback.canonical_stereo_free_smiles.split("."))
    if len(component_identities) != len(dot_positions) + 1:
        raise FragSmilesGeometrySidecarError(
            "fallback SMILES/component boundaries disagree"
        )
    axes_by_pair = {
        (row.source_sdf_atom_index, row.projected_atom_index): row
        for row in atom_axes
    }
    if len(axes_by_pair) != len(atom_axes):
        raise FragSmilesGeometrySidecarError("duplicate source/projected atom axis")
    if len({row.e3fp_row for row in atom_axes}) != len(atom_axes):
        raise FragSmilesGeometrySidecarError("duplicate fallback E3FP row")
    fallback_pairs = {
        (row.source_atom_index, row.projected_atom_index)
        for row in fallback.atom_addresses
    }
    if not set(axes_by_pair).issubset(fallback_pairs):
        raise FragSmilesGeometrySidecarError(
            "fallback atom axes contain an unknown lexical atom"
        )
    if sorted(row.e3fp_row for row in atom_axes) != list(range(len(atom_axes))):
        raise FragSmilesGeometrySidecarError("fallback E3FP rows are not dense")
    output = (MOLECULE_BEGIN,) + fallback.tokens + (MOLECULE_END,)
    atoms = []
    for row in fallback.atom_addresses:
        if any(row.token_start <= dot < row.token_stop for dot in dot_positions):
            raise FragSmilesGeometrySidecarError(
                "fallback atom span crosses a component boundary"
            )
        component_index = sum(dot < row.token_start for dot in dot_positions)
        axis = axes_by_pair.get((row.source_atom_index, row.projected_atom_index))
        atoms.append(
            AtomGeometryAddress(
                e3fp_row=None if axis is None else axis.e3fp_row,
                has_e3fp_row=axis is not None,
                projected_atom_index=row.projected_atom_index,
                source_sdf_atom_index=row.source_atom_index,
                component_index=component_index,
                fragment_index=None,
                fragment_local_atom_index=None,
                fragment_carrier_token_index=None,
                token_start=row.token_start + 1,
                token_stop=row.token_stop + 1,
                is_attachment=False,
            )
        )
    atoms.sort(key=lambda row: row.projected_atom_index)

    if output[1:-1] != fallback.tokens:
        raise FragSmilesGeometrySidecarError("fallback model projection is not reversible")

    roles = ["molecule_boundary"] + list(fallback.roles) + ["molecule_boundary"]
    token_to_fragment = [-1] * len(output)
    for atom in atoms:
        if atom.token_start is not None and atom.token_stop is not None:
            for index in range(atom.token_start, atom.token_stop):
                roles[index] = "atom_glyph"
    return FragSmilesGeometrySidecar(
        schema_version=SCHEMA_VERSION,
        mode="whole_molecule_fallback",
        model_tokens=output,
        token_roles=tuple(roles),
        token_to_fragment=tuple(token_to_fragment),
        fragments=(),
        atoms=tuple(atoms),
        connectors=(),
        component_count=len(component_identities),
        molecule_carrier_token_index=len(output) - 1,
        fallback_mode=True,
        padding_materialized=False,
    )


__all__ = [
    "AtomAxisAddress",
    "AtomGeometryAddress",
    "ConnectorEndpointGeometryAddress",
    "ConnectorGeometryAddress",
    "FragSmilesGeometrySidecar",
    "FragSmilesGeometrySidecarError",
    "FragmentGeometryAddress",
    "SCHEMA_VERSION",
    "build_compact_geometry_sidecar",
    "build_fallback_geometry_sidecar",
]
