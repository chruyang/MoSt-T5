from __future__ import annotations

import base64
import random
from dataclasses import replace

import pytest
from rdkit import Chem

from most_t5_next.r1.tokenizer.production_graph_ports_codec_v1 import (
    CrossEdgeInput,
    EDGE_ENDPOINT_A,
    EDGE_ENDPOINT_B,
    GPORTS_BOUNDARY_TOKENS,
    GPORTS_BYTE_TOKENS,
    GPORTS_UNION_TOKENS,
    GRAPH_BEGIN,
    GRAPH_END,
    GraphPortsContractError,
    ProductionGraphPortsCodecV1,
    _build_graph_token_stream,
)


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _molblock_fixture(encoded: str) -> Chem.Mol:
    mol = Chem.MolFromMolBlock(
        base64.b64decode(encoded).decode("ascii"),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    assert mol is not None
    return mol


def _strict_chemical_projection(mol: Chem.Mol) -> tuple[object, ...]:
    """Expose graph, bond, radical, and assigned stereo invariants separately."""

    probe = Chem.Mol(mol)
    for atom in probe.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    return (
        Chem.MolToSmiles(probe, canonical=True, isomericSmiles=False),
        tuple(
            sorted(
                (
                    atom.GetAtomicNum(),
                    atom.GetIsotope(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                )
                for atom in probe.GetAtoms()
            )
        ),
        tuple(
            sorted(
                (str(bond.GetBondType()), str(bond.GetStereo()))
                for bond in probe.GetBonds()
            )
        ),
        tuple(
            sorted(
                label
                for _atom_index, label in Chem.FindMolChiralCenters(
                    probe,
                    includeUnassigned=True,
                    includeCIP=True,
                    useLegacyImplementation=False,
                )
                # RDKit reports traversal-relative ``Tet_CW``/``Tet_CCW`` for
                # a symmetric centre without an absolute CIP descriptor.  The
                # local winding legitimately changes after canonical atom
                # order changes; only assigned R/S and pseudoasymmetric r/s
                # labels are chemical invariants.
                if label in {"R", "S", "r", "s"}
            )
        ),
    )


# Exact hydrogen-projected PCQM4Mv2 SDF records from the frozen PF-1 rejects.
_PF1_GRAPH_PORT_FIXTURES = (
    (
        63194,
        "CiAgICAgUkRLaXQgICAgICAgICAgM0QKCiAxNCAxNCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMDk5OSBWMjAwMAogICAgMS4zNDI5ICAgIDEuMTQ0MyAgICAwLjYwNzcgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDQuNjg4OSAgICAwLjI5MzUgICAgNS44MTczIEMgICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAzLjQ3NzYgICAtMi41NzMxICAgIDIuNTc4NSBDICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgMi42NDUzICAgLTEuOTQ4MiAgICAxLjQ1NjkgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDMuNTEzNiAgIC0xLjY0MDcgICAgMy43OTY0IEMgICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAyLjMyNTMgICAgMC4xMzQ3ICAgIDAuMDc3OCBDICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgNC40MTg5ICAgIDAuNjUyMyAgICA0LjM4MTggQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDMuMjU5NCAgIC0wLjYwNDYgICAgMS4wMjU3IEMgICAwICAwICAxICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAzLjg2MTMgICAtMC4yMTQ0ICAgIDMuNDQwOSBDICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgMy41NjY5ICAgIDAuMjIwOSAgICAyLjE5MzEgTiAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDIuMjg1MSAgIC0wLjA5MDMgICAtMS4xODQzIE4gICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICA0LjcwOTYgICAgMS45MDYwICAgIDMuODc4NCBOICAgMCAgMCAgMCAgMCAgMCAgMiAgMCAgMCAgMCAgMCAgMCAgMAogICAgMy4yMTU0ICAgLTEuMDc5MiAgIC0xLjU3OTEgTyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDUuMTk2OSAgICAyLjc0NTIgICAgNC42NTY5IE8gICAwICAwICAwICAwICAwICAxICAwICAwICAwICAwICAwICAwCiAgMyAgNSAgMSAgMAogIDQgIDMgIDEgIDAKICA2ICAxICAxICAwCiAgOCAgNiAgMSAgNgogIDcgIDIgIDEgIDAKICA4ICA0ICAxICAwCiAgOCAxMCAgMSAgMAogIDkgIDUgIDEgIDAKICA5ICA3ICAyICAwCiAxMCAgOSAgMSAgMAogMTEgIDYgIDIgIDAKIDEyICA3ICAxICAwCiAxMiAxNCAgMSAgMAogMTMgMTEgIDEgIDAKTSAgUkFEICAyICAxMiAgIDIgIDE0ICAgMgpNICBFTkQK",
        ((2, 3, 4, 6, 7, 8, 9), (1,), (5, 10), (0,), (12,), (11,), (13,)),
        r"C/C(=N/O)[C@@H]1CCC/C(=C(\C)[N][O])N1",
    ),
    (
        2447063,
        "CiAgICAgUkRLaXQgICAgICAgICAgM0QKCiAxNSAxNyAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMDk5OSBWMjAwMAogICAgNS43NjA5ICAgIDMuNjQ2MiAgICA0LjAzNzAgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDQuNDc0MiAgICAyLjg1NzYgICAgNC4yOTY2IEMgICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAwLjUwNTcgICAtMC41NTE3ICAgIDUuNzA4MCBDICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgMC42MTUyICAgIDAuNDM0OCAgICAzLjM5MDEgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDIuMDUyNSAgIC0xLjU0ODggICAgNC4wMDIwIEMgICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAxLjUzMjQgICAgMS43MTI4ICAgIDUuMzM0NyBDICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgMy4wMDAyICAgLTAuMjM3OSAgICA1LjkzMDIgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDAuNjgxMyAgIC0wLjg3MzUgICAgNC4yMDkxIEMgICAwICAwICAxICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAxLjYyMDMgICAgMC40MTQwICAgIDYuMTYwOCBDICAgMCAgMCAgMiAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgMS43NDI2ICAgIDEuMzk4NSAgICAzLjgzNTIgQyAgIDAgIDAgIDEgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgMAogICAgMy4xODAyICAgLTAuNTg5MCAgICA0LjQzODQgQyAgIDAgIDAgIDIgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgMAogICAgNC4zNjkyICAgIDEuNTI2NCAgICAzLjU3NjQgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDMuMDk1MSAgICAwLjY4OTkgICAgMy41NDUyIEMgICAwICAwICAyICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICA1LjQzODYgICAgMS4xNjU3ICAgIDIuOTYxMSBOICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgNS4yOTI1ICAgLTAuMDc1OCAgICAyLjI5NDYgTyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAxICAyICAxICAwCiAgOSAgMyAgMSAgNgogMTAgIDQgIDEgIDEKICA4ICA0ICAxICA2CiAgNSA4ICAxICAwCiAxMSAgNSAgMSAgNgogIDYgIDkgIDEgIDAKICA3ICA5ICAxICAwCiAgOCAgMyAgMSAgMAogMTAgIDYgIDEgIDAKIDExICA3ICAxICAwCiAxMiAgMiAgMSAgMAogMTMgMTIgIDEgIDEKIDEzIDEwICAxICAwCiAxMyAxMSAgMSAgMAogMTQgMTIgIDIgIDAKIDE1IDE0ICAxICAwCk0gIEVORAo=",
        ((11, 13), (1,), (0,), (2, 3, 4, 5, 6, 7, 8, 9, 10, 12), (14,)),
        r"CC/C(=N/O)[C@H]1[C@@H]2C[C@H]3C[C@@H](C2)C[C@@H]1C3",
    ),
    (
        3140645,
        "CiAgICAgUkRLaXQgICAgICAgICAgM0QKCiAgOSAxMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMDk5OSBWMjAwMAogICAgMS4yNDY2ICAgLTAuNjE2NSAgICAwLjY2MzcgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDEuMjE2MCAgICAwLjcxNTMgICAgMC41NzAwIEMgICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgIC0xLjIzMzAgICAtMC42ODUyICAgIDAuNDkyNCBDICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAtMS4yNjQwICAgIDAuNjQ1OCAgICAwLjM5OTAgQyAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDEuMzAyNCAgIC0wLjE1MDYgICAtMi4yNzYxIEMgICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAwLjA3NDIgICAtMS4xNTk5ICAgLTAuMTgwOCBDICAgMCAgMCAgMSAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogICAgMC4wMjI3ICAgIDEuMDgxMyAgIC0wLjMzODAgQyAgIDAgIDAgIDIgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAgIDAKICAgIDAuMDk5OCAgIC0wLjExNTIgICAtMS4zNTcwIEMgICAwICAwICAxICAwICAwICAwICAwICAwICAwICAwICAwICAwCiAgICAxLjE5ODUgICAtMC4yMzQxICAgLTMuNDgzMSBPICAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMCAgMAogIDIgIDEgIDIgIDAKICA0ICAzICAyICAwCiAgOCAgNSAgMSAgMQogIDYgIDMgIDEgIDEKICA2ICAxICAxICAwCiAgNyAgNCAgMSAgMQogIDcgIDIgIDEgIDAKICA4ICA3ICAxICAwCiAgOCAgNiAgMSAgMAogIDkgIDUgIDIgIDAKTSAgRU5ECg==",
        ((0, 1, 2, 3, 5, 6, 7), (4, 8)),
        r"O=C[C@H]1[C@H]2C=C[C@@H]1C=C2",
    ),
)


def _cross_edges(
    mol: Chem.Mol,
    groups: tuple[tuple[int, ...], ...],
) -> tuple[CrossEdgeInput, ...]:
    owner: dict[int, int] = {}
    for motif_id, group in enumerate(groups):
        for atom_index in group:
            owner[atom_index] = motif_id
    edges = []
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtomIdx()
        atom_b = bond.GetEndAtomIdx()
        if owner[atom_a] != owner[atom_b]:
            edges.append(CrossEdgeInput(atom_a, atom_b, str(bond.GetBondType())))
    return tuple(reversed(edges))  # Input order is deliberately non-canonical.


def _scientific_projection(encoding):
    """Drop source-index lineage while retaining every chemical codec field."""

    return (
        encoding.strict_isomeric_identity,
        tuple(
            (
                motif.identity_smiles,
                motif.reconstruction_smiles,
                tuple(
                    (port.port_id, port.local_atom_id, port.bond_type)
                    for port in motif.ports
                ),
            )
            for motif in encoding.motifs
        ),
        tuple(
            (
                connection.endpoint_a,
                connection.endpoint_b,
                connection.bond_type,
                connection.bond_stereo,
                connection.stereo_atoms,
            )
            for connection in encoding.connections
        ),
        encoding.component_motif_ids,
    )


def test_atom_renumbering_keeps_canonical_graph_and_round_trip() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("N[C@@H](C)C(=O)O")
    groups = ((1,), (0,), (2,), (3, 4), (5,))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    assert tuple(
        encoded.canonical_to_logical_motif_ids[canonical_id]
        for canonical_id in encoded.logical_to_canonical_motif_ids
    ) == tuple(range(len(groups)))
    for logical_id, group in enumerate(groups):
        canonical_id = encoded.logical_to_canonical_motif_ids[logical_id]
        assert tuple(sorted(source for _local, source in encoded.motifs[canonical_id].source_atom_map)) == tuple(
            sorted(group)
        )

    renumber_order = tuple(reversed(range(mol.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(mol, renumber_order)
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(renumber_order)}
    # Atom IDs may change, but the upstream frozen logical motif sequence is
    # part of the production contract and therefore keeps the same row order.
    renumbered_groups = tuple(
        tuple(old_to_new[atom_index] for atom_index in group)
        for group in groups
    )
    renumbered_encoded = codec.encode(
        renumbered,
        renumbered_groups,
        _cross_edges(renumbered, renumbered_groups),
    )

    assert _scientific_projection(renumbered_encoded) == _scientific_projection(encoded)
    assert Chem.MolToSmiles(codec.reconstruct(encoded), isomericSmiles=True) == encoded.strict_isomeric_identity
    assert (
        Chem.MolToSmiles(codec.reconstruct(renumbered_encoded), isomericSmiles=True)
        == encoded.strict_isomeric_identity
    )


def test_symmetric_motifs_keep_frozen_logical_order_under_many_atom_renumberings() -> None:
    """Do not invent an unstable whole-graph ordering for equivalent motifs.

    The two methyl branches in this ibuprofen-like graph are symmetry-related.
    Earlier revisions sorted motif records with RDKit canonical atom positions;
    an automorphism could then exchange the two equal identities after a benign
    atom renumbering.  Production instead preserves the already-frozen logical
    motif order and requires only motif-local canonicality.
    """

    codec = ProductionGraphPortsCodecV1()
    mol = _mol("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    frozen_groups = (
        (1,),
        (2,),
        (0,),
        (3,),
        (4, 5, 6, 7, 8, 9),
        (10,),
        (11,),
        (12, 13),
        (14,),
    )
    reference = codec.encode(mol, frozen_groups, _cross_edges(mol, frozen_groups))

    flagged_order = (9, 6, 1, 13, 7, 2, 14, 12, 11, 0, 4, 3, 5, 8, 10)
    rng = random.Random(20260807)
    orders = [flagged_order]
    for _ in range(100):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        orders.append(tuple(order))

    for order in orders:
        renumbered = Chem.RenumberAtoms(mol, order)
        old_to_new = {old_index: new_index for new_index, old_index in enumerate(order)}
        mapped_groups = tuple(
            tuple(old_to_new[atom_index] for atom_index in group)
            for group in frozen_groups
        )
        candidate = codec.encode(
            renumbered,
            mapped_groups,
            _cross_edges(renumbered, mapped_groups),
        )
        assert candidate.logical_to_canonical_motif_ids == tuple(range(len(frozen_groups)))
        assert candidate.canonical_to_logical_motif_ids == tuple(range(len(frozen_groups)))
        assert _scientific_projection(candidate) == _scientific_projection(reference)


@pytest.mark.parametrize(
    ("smiles", "cip"),
    [
        ("C[C@H](O)F", "R"),
        ("C[C@@H](O)F", "S"),
    ],
)
def test_tetrahedral_stereo_and_multiple_ports_on_one_atom(smiles: str, cip: str) -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol(smiles)
    groups = tuple((atom_index,) for atom_index in range(mol.GetNumAtoms()))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    rebuilt = codec.reconstruct(encoded)

    assert Chem.MolToSmiles(rebuilt, isomericSmiles=True) == encoded.strict_isomeric_identity
    assert [label for _atom_index, label in Chem.FindMolChiralCenters(rebuilt, includeUnassigned=True)] == [cip]
    central = next(
        motif
        for motif in encoded.motifs
        if any(source_atom == 1 for _local_id, source_atom in motif.source_atom_map)
    )
    assert tuple(port.port_id for port in central.ports) == (1, 2, 3)
    assert len({port.local_atom_id for port in central.ports}) == 1
    assert "[1*]" in central.identity_smiles
    assert "[2*]" in central.identity_smiles
    assert "[3*]" in central.identity_smiles


def test_opposite_tetrahedral_isomers_have_different_strict_identities() -> None:
    codec = ProductionGraphPortsCodecV1()
    identities = []
    for smiles in ("C[C@H](O)F", "C[C@@H](O)F"):
        mol = _mol(smiles)
        groups = tuple((atom_index,) for atom_index in range(mol.GetNumAtoms()))
        identities.append(codec.encode(mol, groups, _cross_edges(mol, groups)).strict_isomeric_identity)
    assert identities[0] != identities[1]


def test_port_reconnection_refreshes_stereo_ranks_and_keeps_relative_isomers_distinct() -> None:
    """A molzip neighbourhood change must not reuse fragment-era CIP ranks."""

    codec = ProductionGraphPortsCodecV1()
    groups = (
        (1, 11, 12, 13),
        (0,),
        (2,),
        (3,),
        (4, 5, 6, 7, 9, 10),
        (8,),
    )
    relative_isomers = (
        "CC1(CN[C@H]2CC[C@H](C)CC2)COC1",
        "CC1(CN[C@H]2CC[C@@H](C)CC2)COC1",
    )
    references = []
    for isomer_offset, smiles in enumerate(relative_isomers):
        mol = _mol(smiles)
        reference = codec.encode(mol, groups, _cross_edges(mol, groups))
        references.append(reference)
        rebuilt = codec.reconstruct(reference)

        assert (
            Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=True)
            == reference.strict_isomeric_identity
        )
        assert sorted(
            label
            for _atom_index, label in Chem.FindMolChiralCenters(
                rebuilt,
                includeUnassigned=True,
                includeCIP=True,
                useLegacyImplementation=False,
            )
        ) == sorted(
            label
            for _atom_index, label in Chem.FindMolChiralCenters(
                mol,
                includeUnassigned=True,
                includeCIP=True,
                useLegacyImplementation=False,
            )
        )

        rng = random.Random(20260807 + isomer_offset)
        for _ in range(40):
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            renumbered = Chem.RenumberAtoms(mol, order)
            old_to_new = {
                old_index: new_index for new_index, old_index in enumerate(order)
            }
            renumbered_groups = tuple(
                tuple(old_to_new[atom_index] for atom_index in group)
                for group in groups
            )
            candidate = codec.encode(
                renumbered,
                renumbered_groups,
                _cross_edges(renumbered, renumbered_groups),
            )
            assert _scientific_projection(candidate) == _scientific_projection(reference)

    assert references[0].strict_isomeric_identity != references[1].strict_isomeric_identity


def test_four_distinguishable_ports_on_one_chiral_atom() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("F[C@](Cl)(Br)I")
    groups = tuple((atom_index,) for atom_index in range(mol.GetNumAtoms()))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    central = next(
        motif
        for motif in encoded.motifs
        if any(source_atom == 1 for _local_id, source_atom in motif.source_atom_map)
    )

    assert tuple(port.port_id for port in central.ports) == (1, 2, 3, 4)
    assert len({port.local_atom_id for port in central.ports}) == 1
    rebuilt = codec.reconstruct(encoded)
    assert Chem.MolToSmiles(rebuilt, isomericSmiles=True) == "F[C@](Cl)(Br)I"
    assert [label for _atom, label in Chem.FindMolChiralCenters(rebuilt)] == ["S"]


@pytest.mark.parametrize(
    ("ordinal", "smiles", "groups"),
    [
        (
            442310,
            "CC/N=C(/O)O[C@@H](O/C(O)=N/CC)N(C)C",
            (
                (2, 3), (1,), (0,), (4,), (5,), (6,), (7,), (8, 10),
                (9,), (11,), (12,), (13,), (14,), (15,),
            ),
        ),
        (
            2822098,
            "C=C/C(O)=N/C[C@H](CC)C/N=C(/O)C=C",
            (
                (2, 4), (0, 1), (3,), (5,), (6,), (7,), (8,), (9,),
                (10, 11), (12,), (13, 14),
            ),
        ),
        (
            2995617,
            "C/C=C(/C)[C@@](O)(CN(C)C)/C(C)=C/C",
            (
                (4,), (1, 2), (0,), (3,), (5,), (6,), (7,), (8,), (9,),
                (10, 12), (11,), (13,),
            ),
        ),
    ],
)
def test_pf10_stereo_dependent_tetrahedral_centres_survive_preliminary_cleaning(
    ordinal: int,
    smiles: str,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    """PF-10 rejects whose R/S identity depends on cut-supported E/Z state."""

    del ordinal  # The parameter documents the exact rejected PCQM row.
    codec = ProductionGraphPortsCodecV1()
    mol = _mol(smiles)
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    rebuilt = codec.reconstruct(encoded)

    assert (
        Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=True)
        == encoded.strict_isomeric_identity
    )
    def assigned_labels(candidate: Chem.Mol) -> list[str]:
        return sorted(
            label
            for _atom_index, label in Chem.FindMolChiralCenters(
                candidate,
                includeUnassigned=True,
                includeCIP=True,
                useLegacyImplementation=False,
            )
        )

    assert assigned_labels(rebuilt) == assigned_labels(mol) == ["R"]


def test_cam_t5_partition_rejects_cross_motif_double_bond() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("F/C=C/Cl")
    groups = ((0, 1), (2, 3))
    with pytest.raises(GraphPortsContractError, match="SINGLE/STEREONONE"):
        codec.encode(mol, groups, _cross_edges(mol, groups))


@pytest.mark.parametrize(
    ("smiles", "groups"),
    [
        (
            "CC/N=C(/O)C[N]C1=N[CH]NS1",
            ((2, 3), (1,), (0,), (4,), (5,), (6,), (7, 8, 9, 10, 11)),
        ),
        (
            "NCCCC/N=C(/O)c1ccc(F)nc1",
            (
                (5, 6),
                (4,),
                (3,),
                (2,),
                (1,),
                (0,),
                (7,),
                (8, 9, 10, 11, 13, 14),
                (12,),
            ),
        ),
    ],
)
def test_internal_c_n_stereo_supported_by_cut_ports_round_trips_and_renumbers(
    smiles: str,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    """Molzip must replace dummy stereo supports without changing E/Z."""

    codec = ProductionGraphPortsCodecV1()
    mol = _mol(smiles)
    reference = codec.encode(mol, groups, _cross_edges(mol, groups))

    # These production boundary cases used dummy ports as both serialized
    # stereo supports.  Reconstruct repeatedly because the original RDKit
    # molzip failure could depend on the substituent selected after zipping.
    for _ in range(20):
        rebuilt = codec.reconstruct(reference)
        assert Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=True) == smiles

    rng = random.Random(20260807)
    orders = [tuple(reversed(range(mol.GetNumAtoms())))]
    for _ in range(12):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        orders.append(tuple(order))
    for order in orders:
        renumbered = Chem.RenumberAtoms(mol, order)
        old_to_new = {old_index: new_index for new_index, old_index in enumerate(order)}
        mapped_groups = tuple(
            tuple(old_to_new[atom_index] for atom_index in group)
            for group in groups
        )
        candidate = codec.encode(
            renumbered,
            mapped_groups,
            _cross_edges(renumbered, mapped_groups),
        )
        assert _scientific_projection(candidate) == _scientific_projection(reference)
        assert (
            Chem.MolToSmiles(codec.reconstruct(candidate), canonical=True, isomericSmiles=True)
            == smiles
        )


def test_internal_c_n_opposite_geometries_remain_distinguishable() -> None:
    codec = ProductionGraphPortsCodecV1()
    groups = ((2, 3), (1,), (0,), (4,), (5,), (6,), (7, 8, 9, 10, 11))
    smiles_pair = (
        "CC/N=C(/O)C[N]C1=N[CH]NS1",
        r"CC/N=C(\O)C[N]C1=N[CH]NS1",
    )
    encodings = tuple(
        codec.encode(mol, groups, _cross_edges(mol, groups))
        for mol in map(_mol, smiles_pair)
    )

    assert encodings[0].strict_isomeric_identity != encodings[1].strict_isomeric_identity
    assert encodings[0].motifs[0].reconstruction_smiles != encodings[1].motifs[0].reconstruction_smiles
    assert tuple(
        Chem.MolToSmiles(codec.reconstruct(encoding), canonical=True, isomericSmiles=True)
        for encoding in encodings
    ) == smiles_pair


def test_adjacent_c_n_stereo_states_share_a_cut_support_without_collapsing() -> None:
    """Dummy priority must not replace the support-relative stereo relation."""

    codec = ProductionGraphPortsCodecV1()
    groups = (
        (5, 6),
        (4,),
        (3,),
        (2,),
        (1,),
        (0,),
        (7,),
        (8, 9, 10, 11, 12, 13),
    )
    smiles_grid = (
        r"COCCC/N=C(S)\N=c1\cco[nH]1",
        r"COCCC/N=C(S)\N=c1/cco[nH]1",
        r"COCCC/N=C(S)/N=c1/cco[nH]1",
        r"COCCC/N=C(S)/N=c1\cco[nH]1",
    )
    references = []
    for grid_index, smiles in enumerate(smiles_grid):
        mol = _mol(smiles)
        reference = codec.encode(mol, groups, _cross_edges(mol, groups))
        references.append(reference)
        assert (
            Chem.MolToSmiles(
                codec.reconstruct(reference),
                canonical=True,
                isomericSmiles=True,
            )
            == reference.strict_isomeric_identity
        )

        rng = random.Random(20260807 + grid_index)
        for _ in range(24):
            order = list(range(mol.GetNumAtoms()))
            rng.shuffle(order)
            renumbered = Chem.RenumberAtoms(mol, order)
            old_to_new = {
                old_index: new_index for new_index, old_index in enumerate(order)
            }
            renumbered_groups = tuple(
                tuple(old_to_new[atom_index] for atom_index in group)
                for group in groups
            )
            candidate = codec.encode(
                renumbered,
                renumbered_groups,
                _cross_edges(renumbered, renumbered_groups),
            )
            assert _scientific_projection(candidate) == _scientific_projection(reference)

    assert len({reference.strict_isomeric_identity for reference in references}) == 4


@pytest.mark.parametrize(
    ("ordinal", "molblock", "groups", "strict_identity"),
    _PF1_GRAPH_PORT_FIXTURES,
)
def test_pf1_real_stereo_and_polycycle_rejects_are_closed_and_renumber_stable(
    ordinal: int,
    molblock: str,
    groups: tuple[tuple[int, ...], ...],
    strict_identity: str,
) -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _molblock_fixture(molblock)
    reference = codec.encode(mol, groups, _cross_edges(mol, groups))
    rebuilt = codec.reconstruct(reference)

    assert reference.strict_isomeric_identity == strict_identity
    assert (
        Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=True)
        == strict_identity
    )
    assert _strict_chemical_projection(rebuilt) == _strict_chemical_projection(mol)
    assert sorted(atom.GetNumRadicalElectrons() for atom in rebuilt.GetAtoms() if atom.GetNumRadicalElectrons()) == (
        [1, 1] if ordinal == 63194 else []
    )
    for motif in reference.motifs:
        independently_loaded = _mol(motif.identity_smiles)
        assert (
            Chem.MolToSmiles(
                independently_loaded,
                canonical=True,
                isomericSmiles=True,
            )
            == motif.identity_smiles
        )

    rng = random.Random(ordinal)
    for _ in range(32):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        renumbered = Chem.RenumberAtoms(mol, order)
        old_to_new = {
            old_index: new_index for new_index, old_index in enumerate(order)
        }
        renumbered_groups = tuple(
            tuple(old_to_new[atom_index] for atom_index in group)
            for group in groups
        )
        candidate = codec.encode(
            renumbered,
            renumbered_groups,
            _cross_edges(renumbered, renumbered_groups),
        )
        assert _scientific_projection(candidate) == _scientific_projection(reference)


def test_disconnected_components_survive_round_trip() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("CC.O")
    groups = ((0,), (1,), (2,))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    rebuilt = codec.reconstruct(encoded)

    assert encoded.strict_isomeric_identity == "CC.O"
    assert len(encoded.component_motif_ids) == 2
    assert len(Chem.GetMolFrags(rebuilt)) == 2
    assert Chem.MolToSmiles(rebuilt, isomericSmiles=True) == "CC.O"


def test_non_single_cross_edge_is_outside_cam_t5_partition_contract() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("c1ccncc1")
    groups = tuple((atom_index,) for atom_index in range(mol.GetNumAtoms()))
    with pytest.raises(GraphPortsContractError, match="SINGLE/STEREONONE"):
        codec.encode(mol, groups, _cross_edges(mol, groups))

    # Aromatic motifs remain valid when the frozen boundary is the exocyclic
    # single bond; only an aromatic *cross edge* is out of domain.
    toluene = _mol("Cc1ccccc1")
    frozen_groups = ((0,), (1, 2, 3, 4, 5, 6))
    encoded = codec.encode(toluene, frozen_groups, _cross_edges(toluene, frozen_groups))
    assert Chem.MolToSmiles(codec.reconstruct(encoded), isomericSmiles=True) == "Cc1ccccc1"


def test_production_graph_token_grammar_is_union_tokenizer_ready() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("CC.O")
    groups = ((0,), (1,), (2,))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    stream = encoded.graph_token_stream

    assert stream.tokens[0] == GRAPH_BEGIN
    assert stream.tokens[-1] == GRAPH_END
    assert len(stream.tokens) == len(stream.token_roles) == len(stream.token_to_logical_motif)
    assert set(stream.tokens) <= set(GPORTS_UNION_TOKENS)
    assert codec.required_union_tokens() == GPORTS_UNION_TOKENS
    assert len(GPORTS_UNION_TOKENS) == len(set(GPORTS_UNION_TOKENS)) == (
        len(GPORTS_BOUNDARY_TOKENS)
        + len(GPORTS_BYTE_TOKENS)
    )
    assert len(stream.component_token_indices) == len(encoded.component_motif_ids)
    assert len(stream.connection_endpoint_token_indices) == len(encoded.connections)
    assert len(stream.connection_token_indices) == len(encoded.logical_motif_atom_groups)

    for indices in stream.component_token_indices:
        assert indices == ()  # Component membership is compact metadata, not model input.
    for endpoint_pair, connection in zip(
        stream.connection_endpoint_token_indices,
        encoded.connections,
    ):
        for side, indices, endpoint in zip(
            (EDGE_ENDPOINT_A, EDGE_ENDPOINT_B),
            endpoint_pair,
            (connection.endpoint_a, connection.endpoint_b),
        ):
            logical_id = encoded.canonical_to_logical_motif_ids[endpoint.motif_id]
            assert len(indices) == 1
            assert stream.tokens[indices[0]] == side
            assert all(stream.token_roles[index] == "connection" for index in indices)
            assert all(stream.token_to_logical_motif[index] == logical_id for index in indices)

    expected_connection_positions = {
        index for index, role in enumerate(stream.token_roles) if role == "connection"
    }
    observed_connection_positions = {
        index for row in stream.connection_token_indices for index in row
    }
    assert observed_connection_positions == expected_connection_positions
    for logical_id, indices in enumerate(stream.connection_token_indices):
        assert tuple(sorted(set(indices))) == indices
        assert all(stream.token_to_logical_motif[index] == logical_id for index in indices)
    for index, role in enumerate(stream.token_roles):
        if role == "boundary":
            assert stream.token_to_logical_motif[index] == -1
    assert codec.decode_graph_token_stream(encoded) == (
        encoded.component_motif_ids,
        encoded.connections,
    )


@pytest.mark.parametrize(
    ("smiles", "atom_count", "expected_graph_tokens"),
    [
        ("CCO", 3, 14),
        ("CC(C)(C)C", 5, 24),
        ("CCOCCNCC(C)OCC(F)(Cl)Br", 15, 74),
        ("CCCCCCCCCCCCCCCCCCCC", 20, 99),
    ],
)
def test_compact_graph_tokens_scale_with_motifs_and_edges(
    smiles: str,
    atom_count: int,
    expected_graph_tokens: int,
) -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol(smiles)
    assert mol.GetNumAtoms() == atom_count
    groups = tuple((atom_index,) for atom_index in range(atom_count))
    edges = _cross_edges(mol, groups)
    encoded = codec.encode(mol, groups, edges)

    assert len(encoded.graph_token_stream.tokens) == expected_graph_tokens
    assert len(encoded.graph_token_stream.tokens) == 4 + 5 * len(edges)
    assert codec.decode_graph_token_stream(encoded) == (
        encoded.component_motif_ids,
        encoded.connections,
    )
    codec.validate_against_source(mol, groups, edges, encoded)
    if atom_count == 20:
        assert len(encoded.graph_token_stream.tokens) < 160


def test_reconstruct_rejects_tampered_contract_fields() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("C[C@H](O)F")
    groups = tuple((atom_index,) for atom_index in range(mol.GetNumAtoms()))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))

    first_motif = encoded.motifs[0]
    bad_identity = replace(first_motif, identity_smiles="tampered")
    with pytest.raises(GraphPortsContractError, match="identity_smiles"):
        codec.reconstruct(replace(encoded, motifs=(bad_identity, *encoded.motifs[1:])))

    motif_with_port = next(motif for motif in encoded.motifs if motif.ports)
    bad_port = replace(motif_with_port.ports[0], bond_type="DOUBLE")
    bad_port_motif = replace(
        motif_with_port,
        ports=(bad_port, *motif_with_port.ports[1:]),
    )
    bad_motifs = tuple(
        bad_port_motif if motif.motif_id == motif_with_port.motif_id else motif
        for motif in encoded.motifs
    )
    with pytest.raises(GraphPortsContractError, match="bond_type"):
        codec.reconstruct(replace(encoded, motifs=bad_motifs))

    with pytest.raises(GraphPortsContractError, match="component_motif_ids"):
        codec.reconstruct(replace(encoded, component_motif_ids=((0,),)))

    bad_connection = replace(encoded.connections[0], connection_id=99)
    with pytest.raises(GraphPortsContractError, match="connection ids"):
        codec.reconstruct(replace(encoded, connections=(bad_connection, *encoded.connections[1:])))

    reversed_connections = tuple(
        replace(connection, connection_id=connection_id)
        for connection_id, connection in enumerate(reversed(encoded.connections), start=1)
    )
    synchronized_stream = _build_graph_token_stream(
        encoded.component_motif_ids,
        reversed_connections,
        encoded.canonical_to_logical_motif_ids,
    )
    with pytest.raises(GraphPortsContractError, match="canonical endpoint order"):
        codec.reconstruct(
            replace(
                encoded,
                connections=reversed_connections,
                graph_token_stream=synchronized_stream,
            )
        )

    bad_source_map = replace(
        first_motif,
        source_atom_map=((first_motif.source_atom_map[0][0], 999), *first_motif.source_atom_map[1:]),
    )
    with pytest.raises(GraphPortsContractError, match="source atom map"):
        codec.reconstruct(replace(encoded, motifs=(bad_source_map, *encoded.motifs[1:])))

    single = encoded.connections[0]
    outside_partition = replace(single, bond_type="DOUBLE")
    with pytest.raises(GraphPortsContractError, match="CAMT5 SINGLE/STEREONONE"):
        codec.reconstruct(
            replace(encoded, connections=(outside_partition, *encoded.connections[1:]))
        )


def test_source_bound_validation_rejects_coordinated_logical_relabeling() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("CCO")
    frozen_groups = ((0,), (1,), (2,))
    frozen_edges = _cross_edges(mol, frozen_groups)
    encoded = codec.encode(mol, frozen_groups, frozen_edges)
    codec.validate_against_source(mol, frozen_groups, frozen_edges, encoded)

    relabeled_groups = tuple(reversed(frozen_groups))
    relabeled = codec.encode(mol, relabeled_groups, _cross_edges(mol, relabeled_groups))
    codec.validate(relabeled)  # It is internally self-consistent.
    with pytest.raises(GraphPortsContractError, match="canonical encoding of its source"):
        codec.validate_against_source(mol, frozen_groups, frozen_edges, relabeled)


def test_forced_fallback_and_macro_are_identity_equivalent() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("C[C@H](O)F")
    groups = tuple((atom_index,) for atom_index in range(mol.GetNumAtoms()))
    encoded = codec.encode(mol, groups, _cross_edges(mol, groups))
    identity = encoded.motifs[0].identity_smiles
    macro_token = "<MOTIF:KNOWN>"

    macro = codec.encode_identity_surface(
        identity,
        macro_by_identity={identity: macro_token},
    )
    fallback = codec.encode_identity_surface(
        identity,
        macro_by_identity={identity: macro_token},
        force_fallback=True,
    )

    assert macro.mode == "macro"
    assert fallback.mode == "fallback"
    assert macro.tokens == (macro_token,)
    assert fallback.tokens != macro.tokens
    assert codec.decode_identity_surface(macro, identity_by_macro={macro_token: identity}) == identity
    assert codec.decode_identity_surface(fallback) == identity
    assert Chem.MolToSmiles(codec.reconstruct(encoded), isomericSmiles=True) == encoded.strict_isomeric_identity
    with pytest.raises(GraphPortsContractError, match="injective"):
        codec.encode_identity_surface(
            identity,
            macro_by_identity={identity: macro_token, identity + "-alias": macro_token},
        )


def test_incomplete_or_wrong_cross_edge_table_is_rejected() -> None:
    codec = ProductionGraphPortsCodecV1()
    mol = _mol("CCO")
    groups = ((0,), (1,), (2,))
    edges = _cross_edges(mol, groups)

    with pytest.raises(GraphPortsContractError, match="not complete"):
        codec.encode(mol, groups, edges[:-1])
    with pytest.raises(GraphPortsContractError, match="declares DOUBLE"):
        codec.encode(mol, groups, (replace(edges[0], bond_type="DOUBLE"), *edges[1:]))
