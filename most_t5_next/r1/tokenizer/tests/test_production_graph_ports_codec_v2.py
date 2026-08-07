from __future__ import annotations

from dataclasses import replace
import random

import pytest
from rdkit import Chem

from most_t5_next.r1.tokenizer import production_graph_ports_codec_v1 as v1
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v2 as v2


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _cross_edges(mol, groups):
    owner = {
        atom_index: motif_id
        for motif_id, group in enumerate(groups)
        for atom_index in group
    }
    return tuple(
        v1.CrossEdgeInput(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            str(bond.GetBondType()),
        )
        for bond in mol.GetBonds()
        if owner[bond.GetBeginAtomIdx()] != owner[bond.GetEndAtomIdx()]
    )


def _identity(mol: Chem.Mol) -> str:
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    return Chem.MolToSmiles(probe, canonical=True, isomericSmiles=True)


@pytest.mark.parametrize(
    ("smiles", "groups"),
    (
        ("CCO", ((0,), (1,), (2,))),
        ("CC.O", ((0,), (1,), (2,))),
        ("F[C@H](Cl)Br", ((1,), (0,), (2,), (3,))),
        ("F/C=C/F", ((1, 2), (0,), (3,))),
        ("F/C=C\\F", ((1, 2), (0,), (3,))),
        ("[13CH3][NH2+]C(=O)[O-]", ((0,), (1,), (2, 3, 4))),
    ),
)
def test_v2_preserves_v1_chemistry_and_shortens_graph(smiles, groups) -> None:
    mol = _mol(smiles)
    edges = _cross_edges(mol, groups)
    old = v1.ProductionGraphPortsCodecV1().encode(mol, groups, edges)
    new = v2.ProductionGraphPortsCodecV2().encode(mol, groups, edges)

    assert new.format_version == v2.FORMAT_VERSION
    assert new.connections == old.connections
    assert new.motifs == old.motifs
    assert new.logical_motif_atom_groups == old.logical_motif_atom_groups
    assert len(old.graph_token_stream.tokens) == 4 + 5 * len(edges)
    assert len(new.graph_token_stream.tokens) == 4 + 2 * len(edges)
    assert _identity(v2.ProductionGraphPortsCodecV2().reconstruct(new)) == _identity(mol)
    v2.ProductionGraphPortsCodecV2().validate_against_source(
        mol, groups, edges, new
    )


def test_multibyte_endpoint_has_one_owner_carrier_and_boundary_continuation() -> None:
    motif_count = 21
    mappings = tuple(range(motif_count))
    connections = (
        v1.ConnectionRecord(
            connection_id=1,
            endpoint_a=v1.PortRef(0, 1),
            endpoint_b=v1.PortRef(20, 8),
            bond_type="SINGLE",
            bond_stereo="STEREONONE",
            stereo_atoms=None,
        ),
    )
    components = v1._connected_component_motifs(motif_count, connections)
    stream = v2._build_endpoint_pair_graph_token_stream(
        components, connections, mappings
    )
    endpoint_indices = stream.connection_endpoint_token_indices[0][1]
    assert len(endpoint_indices) == 1
    endpoint_start = endpoint_indices[0]
    assert stream.token_roles[endpoint_start] == "connection"
    assert stream.token_to_logical_motif[endpoint_start] == 20
    assert stream.token_roles[endpoint_start + 1] == "boundary"
    assert stream.token_to_logical_motif[endpoint_start + 1] == -1
    for logical_id, indices in enumerate(stream.connection_token_indices):
        assert all(stream.token_roles[index] == "connection" for index in indices)
        assert all(stream.token_to_logical_motif[index] == logical_id for index in indices)
    assert sum(map(len, stream.connection_token_indices)) == 2
    assert (
        v2._decode_endpoint_pair_graph_token_stream(stream, components, mappings)
        == connections
    )


@pytest.mark.parametrize(
    ("packed_endpoint", "expected_bytes"),
    ((127, 1), (128, 2), (16383, 2), (16384, 3)),
)
def test_endpoint_owner_contract_crosses_uvarint_boundaries(
    packed_endpoint, expected_bytes
) -> None:
    tokens: list[str] = []
    roles: list[str] = []
    owners: list[int] = []
    carrier = v2._append_endpoint(
        tokens=tokens,
        roles=roles,
        token_to_logical=owners,
        packed_endpoint=packed_endpoint,
        logical_motif_id=2,
    )
    assert len(tokens) == expected_bytes
    assert carrier == (0,)
    assert roles[0] == "connection"
    assert owners[0] == 2
    assert roles[1:] == ["boundary"] * (expected_bytes - 1)
    assert owners[1:] == [-1] * (expected_bytes - 1)


def test_connection_id_is_implicit_in_canonical_pair_order() -> None:
    connections = (
        v1.ConnectionRecord(
            1,
            v1.PortRef(0, 1),
            v1.PortRef(1, 1),
            "SINGLE",
            "STEREONONE",
            None,
        ),
        v1.ConnectionRecord(
            2,
            v1.PortRef(0, 2),
            v1.PortRef(2, 2),
            "SINGLE",
            "STEREONONE",
            None,
        ),
        v1.ConnectionRecord(
            3,
            v1.PortRef(1, 2),
            v1.PortRef(2, 1),
            "SINGLE",
            "STEREONONE",
            None,
        ),
    )
    components = v1._connected_component_motifs(3, connections)
    stream = v2._build_endpoint_pair_graph_token_stream(
        components, connections, (0, 1, 2)
    )
    assert len(stream.tokens) == 4 + 2 * len(connections)
    assert (
        v2._decode_endpoint_pair_graph_token_stream(
            stream, components, (0, 1, 2)
        )
        == connections
    )


def test_zero_edge_disconnected_cycle_and_parallel_motif_pair_domains() -> None:
    zero_components = ((0,),)
    zero = v2._build_endpoint_pair_graph_token_stream(
        zero_components, (), (0,)
    )
    assert len(zero.tokens) == 4
    assert v2._decode_endpoint_pair_graph_token_stream(
        zero, zero_components, (0,)
    ) == ()

    connections = (
        v1.ConnectionRecord(1, v1.PortRef(0, 1), v1.PortRef(1, 1), "SINGLE", "STEREONONE", None),
        v1.ConnectionRecord(2, v1.PortRef(0, 2), v1.PortRef(1, 2), "SINGLE", "STEREONONE", None),
    )
    components = v1._connected_component_motifs(3, connections)
    assert components == ((0, 1), (2,))
    parallel = v2._build_endpoint_pair_graph_token_stream(
        components, connections, (0, 1, 2)
    )
    assert v2._decode_endpoint_pair_graph_token_stream(
        parallel, components, (0, 1, 2)
    ) == connections


def test_direct_builder_rejects_noncanonical_order_reused_port_and_bad_components() -> None:
    first = v1.ConnectionRecord(
        1, v1.PortRef(0, 1), v1.PortRef(1, 1), "SINGLE", "STEREONONE", None
    )
    later = v1.ConnectionRecord(
        2, v1.PortRef(0, 2), v1.PortRef(2, 1), "SINGLE", "STEREONONE", None
    )
    with pytest.raises(v1.GraphPortsContractError):
        v2._build_endpoint_pair_graph_token_stream(
            ((0, 1, 2),), (replace(later, connection_id=1), replace(first, connection_id=2)), (0, 1, 2)
        )

    reused = replace(later, endpoint_a=first.endpoint_a)
    with pytest.raises(v1.GraphPortsContractError):
        v2._build_endpoint_pair_graph_token_stream(
            ((0, 1, 2),), (first, reused), (0, 1, 2)
        )

    with pytest.raises(v1.GraphPortsContractError):
        v2._build_endpoint_pair_graph_token_stream(
            ((0, 1), (2,)), (first, later), (0, 1, 2)
        )


def test_v2_rejects_surface_role_and_endpoint_tampering() -> None:
    mol = _mol("CCO")
    groups = ((0,), (1,), (2,))
    encoding = v2.ProductionGraphPortsCodecV2().encode(
        mol, groups, _cross_edges(mol, groups)
    )
    stream = encoding.graph_token_stream
    connection_index = stream.connection_endpoint_token_indices[0][0][0]
    wrong_roles = list(stream.token_roles)
    wrong_roles[connection_index] = "boundary"
    tampered_role = replace(
        encoding,
        graph_token_stream=replace(stream, token_roles=tuple(wrong_roles)),
    )
    with pytest.raises(v1.GraphPortsContractError):
        v2.ProductionGraphPortsCodecV2().validate(tampered_role)

    tokens = list(stream.tokens)
    tokens[connection_index] = v1.GPORTS_BYTE_TOKENS[0]
    tampered_endpoint = replace(
        encoding,
        graph_token_stream=replace(stream, tokens=tuple(tokens)),
    )
    with pytest.raises(v1.GraphPortsContractError):
        v2.ProductionGraphPortsCodecV2().validate(tampered_endpoint)


def test_random_atom_renumbering_keeps_v2_scientific_identity() -> None:
    source = _mol("N[C@@H](C)C(=O)O")
    groups = ((1,), (0,), (2,), (3, 4), (5,))
    reference = v2.ProductionGraphPortsCodecV2().encode(
        source, groups, _cross_edges(source, groups)
    )
    rng = random.Random(20260808)
    for _ in range(32):
        order = list(range(source.GetNumAtoms()))
        rng.shuffle(order)
        mol = Chem.RenumberAtoms(source, order)
        old_to_new = {old: new for new, old in enumerate(order)}
        renumbered_groups = tuple(
            tuple(old_to_new[atom] for atom in group) for group in groups
        )
        encoded = v2.ProductionGraphPortsCodecV2().encode(
            mol,
            renumbered_groups,
            _cross_edges(mol, renumbered_groups),
        )
        assert encoded.strict_isomeric_identity == reference.strict_isomeric_identity
        assert encoded.graph_token_stream.tokens == reference.graph_token_stream.tokens
        assert _identity(v2.ProductionGraphPortsCodecV2().reconstruct(encoded)) == _identity(source)


def test_v2_union_registry_omits_redundant_edge_markers() -> None:
    tokens = v2.ProductionGraphPortsCodecV2.required_union_tokens()
    assert v1.EDGE_ENDPOINT_A not in tokens
    assert v1.EDGE_ENDPOINT_B not in tokens
    assert v1.PORT_RADIX in tokens
    assert len(tokens) == 261
    assert len(tokens) == len(set(tokens))
