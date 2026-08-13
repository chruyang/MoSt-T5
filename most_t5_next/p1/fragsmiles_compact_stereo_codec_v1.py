"""Compact, fragment-local stereo records for the fragSMILES candidate.

The official fragSMILES tokens remain the connectivity language.  This module
adds fixed-arity records immediately after their owning fragment token:

* ``<ST:A:R|S> d d d`` for a CIP-labelled tetrahedral atom;
* ``<ST:B:E|Z:CIS|TRANS> d d d d d d`` for a double bond.

The first integer is a fragment-local atom index or a canonical local double-
bond ordinal.  The second bond integer packs the selected support-neighbour
indices (two two-bit values).  Full molecule/source atom indices never enter
the model surface.

This is a preflight codec.  A retained tetrahedral source tag is serialized
only when RDKit assigns an R/S CIP identity; non-stereogenic Tet_CW/Tet_CCW
source-state tags that disappear from canonical isomeric identity are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from most_t5_next.p1.audit_fragsmiles_adoption_v1 import (
    FragSmilesAuditError,
    FragSmilesRecord,
    _chemicalgof_import,
    encode_with_sidecar,
)


ATOM_PREFIX = "<ST:A:"
BOND_PREFIX = "<ST:B:"
_DIGITS = frozenset("0123456789")
_NUMBER_WIDTH = 3


class CompactStereoCodecError(FragSmilesAuditError):
    pass


@dataclass(frozen=True)
class CompactAtomStereoRecord:
    fragment_index: int
    local_atom_index: int
    label: str
    local_parity: str


@dataclass(frozen=True)
class CompactBondStereoRecord:
    fragment_index: int
    local_bond_ordinal: int
    label: str
    support_relation: str
    support_selector_code: int


@dataclass(frozen=True)
class CompactStereoSurface:
    source_isomeric_smiles: str
    connectivity_record: FragSmilesRecord
    tokens: tuple[str, ...]
    atom_records: tuple[CompactAtomStereoRecord, ...]
    bond_records: tuple[CompactBondStereoRecord, ...]


def _number_tokens(value: int) -> tuple[str, ...]:
    if value < 0 or value > 255:
        raise CompactStereoCodecError("stereo local integer exceeds byte domain")
    return tuple(f"{value:03d}")


def compact_stereo_token_universe() -> tuple[str, ...]:
    """Return the complete finite ordinary-token domain of the sidecar."""

    atom_tokens = tuple(
        f"<ST:A:{label}:{parity}>"
        for label in ("R", "S", "X")
        for parity in ("CCW", "CW")
    )
    bond_tokens = tuple(
        f"<ST:B:{label}:{relation}>"
        for label in ("E", "Z")
        for relation in ("CIS", "TRANS")
    )
    return atom_tokens + bond_tokens


def _parse_number(tokens: Sequence[str], start: int) -> int:
    digits = tuple(tokens[start : start + _NUMBER_WIDTH])
    if len(digits) != _NUMBER_WIDTH or any(token not in _DIGITS for token in digits):
        raise CompactStereoCodecError(f"invalid stereo integer tokens: {digits!r}")
    value = int("".join(digits))
    if value > 255:
        raise CompactStereoCodecError("stereo local integer exceeds byte domain")
    return value


def _is_fragment_token(token: str) -> bool:
    return (
        token not in {"(", ")", "<COMP>"}
        and token not in _DIGITS
        and not token.startswith("<")
    )


def _project_heavy_connectivity(source_mol: Chem.Mol) -> Chem.Mol:
    projected = Chem.Mol(source_mol)
    Chem.RemoveStereochemistry(projected)
    projected = Chem.RemoveHs(projected, sanitize=True)
    if projected.GetNumAtoms() == 0:
        raise CompactStereoCodecError("empty heavy-atom projection")
    return projected


def _address_map(record: FragSmilesRecord) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for fragment in record.fragments:
        for local_index, source_index in enumerate(fragment.source_atom_indices):
            if source_index in result:
                raise CompactStereoCodecError("atom occurs in multiple fragments")
            result[source_index] = (fragment.sequence_fragment_index, local_index)
    return result


def _local_double_bonds(fragment_smiles: str) -> tuple[tuple[int, int], ...]:
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        raise CompactStereoCodecError("invalid fragment SMILES")
    return tuple(
        sorted(
            (min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
             max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.DOUBLE
        )
    )


def _support_candidates(
    projected: Chem.Mol,
    endpoint: int,
    opposite: int,
    addresses: dict[int, tuple[int, int]],
) -> tuple[int | None, ...]:
    heavy = sorted(
        (
            neighbour.GetIdx()
            for neighbour in projected.GetAtomWithIdx(endpoint).GetNeighbors()
            if neighbour.GetIdx() != opposite
        ),
        key=lambda atom_index: addresses[atom_index],
    )
    candidates: list[int | None] = list(heavy)
    if projected.GetAtomWithIdx(endpoint).GetTotalNumHs() > 0:
        candidates.append(None)
    if not candidates or len(candidates) > 4:
        raise CompactStereoCodecError("invalid stereo support candidate count")
    return tuple(candidates)


def _canonical_local_chiral_tags(
    source_mol: Chem.Mol,
    record: FragSmilesRecord,
) -> dict[int, str]:
    """Return tetrahedral tags after renumbering onto the fragment-local axis."""
    annotated = Chem.Mol(source_mol)
    for atom in annotated.GetAtoms():
        atom.SetIntProp("most_source_atom_index", atom.GetIdx())
    Chem.AssignStereochemistry(annotated, cleanIt=True, force=True)

    stereo_free = Chem.Mol(annotated)
    Chem.RemoveStereochemistry(stereo_free)
    stereo_free = Chem.RemoveHs(stereo_free, sanitize=True)
    original_to_projected = {
        atom.GetIntProp("most_source_atom_index"): atom.GetIdx()
        for atom in stereo_free.GetAtoms()
    }
    projected_to_original = {
        projected: original for original, projected in original_to_projected.items()
    }
    addresses = _address_map(record)

    stereo_heavy = Chem.RemoveHs(Chem.Mol(annotated), sanitize=True)
    original_to_stereo_heavy = {
        atom.GetIntProp("most_source_atom_index"): atom.GetIdx()
        for atom in stereo_heavy.GetAtoms()
    }
    result = {}
    for center in record.stereo_identity.atom_centers:
        original = projected_to_original[center.projected_atom_index]
        atom = stereo_heavy.GetAtomWithIdx(original_to_stereo_heavy[original])
        tag = atom.GetChiralTag()
        neighbour_addresses = []
        for neighbour in atom.GetNeighbors():
            neighbour_original = neighbour.GetIntProp("most_source_atom_index")
            neighbour_projected = original_to_projected.get(neighbour_original)
            if neighbour_projected is None:
                # A retained explicit H is a deterministic final support.
                neighbour_addresses.append((1, neighbour_original, 0))
            else:
                fragment_index, local_index = addresses[neighbour_projected]
                neighbour_addresses.append((0, fragment_index, local_index))
        if _permutation_is_odd(neighbour_addresses):
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                tag = Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            result[center.projected_atom_index] = "CW"
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            result[center.projected_atom_index] = "CCW"
        else:
            raise CompactStereoCodecError("defined tetrahedral center lost local parity")
    return result


def _permutation_is_odd(values: Sequence[tuple[int, int, int]]) -> bool:
    if len(set(values)) != len(values):
        raise CompactStereoCodecError("tetrahedral neighbour address is not unique")
    positions = {value: index for index, value in enumerate(sorted(values))}
    permutation = [positions[value] for value in values]
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return bool(inversions % 2)


def _build_compact_records(
    source_mol: Chem.Mol,
    record: FragSmilesRecord,
) -> tuple[tuple[CompactAtomStereoRecord, ...], tuple[CompactBondStereoRecord, ...]]:
    projected = _project_heavy_connectivity(source_mol)
    addresses = _address_map(record)
    local_chiral_tags = _canonical_local_chiral_tags(source_mol, record)
    atoms: list[CompactAtomStereoRecord] = []
    for center in record.stereo_identity.atom_centers:
        label = center.cip_label.upper()
        if label not in {"R", "S"}:
            # The input was first canonical-SMILES-normalized.  Consequently a
            # remaining Tet_CW/Tet_CCW tag is part of that serialized identity,
            # even when no stable R/S CIP label exists.
            label = "X"
        if center.fragment_index is None or center.fragment_local_atom_index is None:
            raise CompactStereoCodecError("tetrahedral center is not fragment-addressed")
        atoms.append(
            CompactAtomStereoRecord(
                fragment_index=center.fragment_index,
                local_atom_index=center.fragment_local_atom_index,
                label=label,
                local_parity=local_chiral_tags[center.projected_atom_index],
            )
        )

    bonds: list[CompactBondStereoRecord] = []
    for stereo_bond in record.stereo_identity.double_bonds:
        if stereo_bond.cip_label not in {"E", "Z"}:
            raise CompactStereoCodecError("defined double bond lacks E/Z CIP label")
        if (
            stereo_bond.begin_fragment_index is None
            or stereo_bond.end_fragment_index is None
            or stereo_bond.begin_fragment_index != stereo_bond.end_fragment_index
        ):
            raise CompactStereoCodecError("stereo double bond crosses fragments")
        fragment_index = stereo_bond.begin_fragment_index
        begin_local = int(stereo_bond.begin_fragment_local_atom_index)
        end_local = int(stereo_bond.end_fragment_local_atom_index)
        local_pair = (min(begin_local, end_local), max(begin_local, end_local))
        local_bonds = _local_double_bonds(record.fragments[fragment_index].fragment_smiles)
        try:
            ordinal = local_bonds.index(local_pair)
        except ValueError as exc:
            raise CompactStereoCodecError("stereo bond absent from owning fragment") from exc

        if begin_local <= end_local:
            endpoint0 = stereo_bond.projected_begin_atom_index
            endpoint1 = stereo_bond.projected_end_atom_index
            support0 = stereo_bond.begin_support_atom_index
            support1 = stereo_bond.end_support_atom_index
        else:
            endpoint0 = stereo_bond.projected_end_atom_index
            endpoint1 = stereo_bond.projected_begin_atom_index
            support0 = stereo_bond.end_support_atom_index
            support1 = stereo_bond.begin_support_atom_index
        candidates0 = _support_candidates(projected, endpoint0, endpoint1, addresses)
        candidates1 = _support_candidates(projected, endpoint1, endpoint0, addresses)
        try:
            selector0 = candidates0.index(support0)
            selector1 = candidates1.index(support1)
        except ValueError as exc:
            raise CompactStereoCodecError("stored support is not a connectivity neighbour") from exc
        relation = (
            stereo_bond.stereo[len("STEREO") :]
            if stereo_bond.stereo.startswith("STEREO")
            else stereo_bond.stereo
        )
        if relation not in {"CIS", "TRANS"}:
            raise CompactStereoCodecError("unsupported double-bond support relation")
        bonds.append(
            CompactBondStereoRecord(
                fragment_index=fragment_index,
                local_bond_ordinal=ordinal,
                label=stereo_bond.cip_label,
                support_relation=relation,
                support_selector_code=selector0 * 4 + selector1,
            )
        )
    return tuple(atoms), tuple(bonds)


def _render_tokens(
    record: FragSmilesRecord,
    atom_records: Sequence[CompactAtomStereoRecord],
    bond_records: Sequence[CompactBondStereoRecord],
) -> tuple[str, ...]:
    atoms_by_fragment: dict[int, list[CompactAtomStereoRecord]] = {}
    bonds_by_fragment: dict[int, list[CompactBondStereoRecord]] = {}
    for item in atom_records:
        atoms_by_fragment.setdefault(item.fragment_index, []).append(item)
    for item in bond_records:
        bonds_by_fragment.setdefault(item.fragment_index, []).append(item)

    output: list[str] = []
    fragment_index = -1
    for token in record.tokens:
        output.append(token)
        if not _is_fragment_token(token):
            continue
        fragment_index += 1
        for item in sorted(
            atoms_by_fragment.get(fragment_index, ()),
            key=lambda row: row.local_atom_index,
        ):
            output.append(f"<ST:A:{item.label}:{item.local_parity}>")
            output.extend(_number_tokens(item.local_atom_index))
        for item in sorted(
            bonds_by_fragment.get(fragment_index, ()),
            key=lambda row: row.local_bond_ordinal,
        ):
            output.append(f"<ST:B:{item.label}:{item.support_relation}>")
            output.extend(_number_tokens(item.local_bond_ordinal))
            output.extend(_number_tokens(item.support_selector_code))
    if fragment_index + 1 != len(record.fragments):
        raise CompactStereoCodecError("fragment/token traversal count drifted")
    return tuple(output)


def _same_stereo_identity(expected: Chem.Mol, actual: Chem.Mol) -> bool:
    if (
        expected.GetNumAtoms() != actual.GetNumAtoms()
        or expected.GetNumBonds() != actual.GetNumBonds()
    ):
        return False
    expected_connectivity = Chem.MolToSmiles(
        expected, canonical=True, isomericSmiles=False
    )
    actual_connectivity = Chem.MolToSmiles(
        actual, canonical=True, isomericSmiles=False
    )
    if expected_connectivity != actual_connectivity:
        return False
    expected_key = Chem.MolToInchiKey(expected)
    actual_key = Chem.MolToInchiKey(actual)
    if expected_key and actual_key:
        return expected_key == actual_key
    return (
        expected.HasSubstructMatch(actual, useChirality=True)
        and actual.HasSubstructMatch(expected, useChirality=True)
    )


def _select_local_parities(
    *,
    expected: Chem.Mol,
    record: FragSmilesRecord,
    atom_records: tuple[CompactAtomStereoRecord, ...],
    bond_records: tuple[CompactBondStereoRecord, ...],
    chemicalgof_root: Path,
) -> tuple[tuple[CompactAtomStereoRecord, ...], tuple[str, ...]]:
    initial_tokens = _render_tokens(record, atom_records, bond_records)
    if not atom_records:
        return atom_records, initial_tokens
    try:
        initial_mol = decode_compact_stereo_surface(
            initial_tokens, chemicalgof_root=chemicalgof_root
        )
    except CompactStereoCodecError:
        initial_mol = None
    if initial_mol is not None and _same_stereo_identity(expected, initial_mol):
        return atom_records, initial_tokens
    if len(atom_records) > 12:
        raise CompactStereoCodecError(
            "tetrahedral parity search exceeds preflight center limit"
        )
    matches: list[tuple[tuple[str, ...], tuple[CompactAtomStereoRecord, ...]]] = []
    for parities in product(("CCW", "CW"), repeat=len(atom_records)):
        candidate_records = tuple(
            replace(item, local_parity=parity)
            for item, parity in zip(atom_records, parities)
        )
        candidate_tokens = _render_tokens(record, candidate_records, bond_records)
        try:
            candidate_mol = decode_compact_stereo_surface(
                candidate_tokens, chemicalgof_root=chemicalgof_root
            )
        except CompactStereoCodecError:
            continue
        if _same_stereo_identity(expected, candidate_mol):
            matches.append((candidate_tokens, candidate_records))
    if not matches:
        raise CompactStereoCodecError(
            "no fragment-local parity assignment restores stereo identity"
        )
    matches.sort(key=lambda item: item[0])
    return matches[0][1], matches[0][0]


def _encode_compact_stereo_once(
    source_mol: Chem.Mol,
    *,
    chemicalgof_root: Path,
) -> CompactStereoSurface:
    source_isomeric_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(source_mol), sanitize=True),
        canonical=True,
        isomericSmiles=True,
    )
    normalized_source = Chem.MolFromSmiles(source_isomeric_smiles)
    if normalized_source is None:
        raise CompactStereoCodecError("canonical isomeric source cannot be reparsed")
    record = encode_with_sidecar(
        normalized_source,
        chemicalgof_root=chemicalgof_root,
        stereo_policy="connectivity_only",
    )
    atom_records, bond_records = _build_compact_records(normalized_source, record)
    atom_records, output_tokens = _select_local_parities(
        expected=normalized_source,
        record=record,
        atom_records=atom_records,
        bond_records=bond_records,
        chemicalgof_root=chemicalgof_root,
    )
    return CompactStereoSurface(
        source_isomeric_smiles=source_isomeric_smiles,
        connectivity_record=record,
        tokens=output_tokens,
        atom_records=atom_records,
        bond_records=bond_records,
    )


def encode_compact_stereo_surface(
    source_mol: Chem.Mol,
    *,
    chemicalgof_root: Path,
) -> CompactStereoSurface:
    """Return the minimum surface in the eventual decode/re-encode cycle.

    A source representation can have a short transient before it enters the
    deterministic surface cycle.  Transient members must not participate in
    canonical selection: starting again from the decoded molecule would then
    omit that transient and could select a different surface.
    """
    current = Chem.Mol(source_mol)
    seen_at: dict[tuple[str, ...], int] = {}
    orbit: list[CompactStereoSurface] = []
    for _ in range(16):
        surface = _encode_compact_stereo_once(
            current, chemicalgof_root=chemicalgof_root
        )
        if not surface.atom_records:
            return surface
        cycle_start = seen_at.get(surface.tokens)
        if cycle_start is not None:
            return min(orbit[cycle_start:], key=lambda item: item.tokens)
        seen_at[surface.tokens] = len(orbit)
        orbit.append(surface)
        current = decode_compact_stereo_surface(
            surface.tokens, chemicalgof_root=chemicalgof_root
        )
    raise CompactStereoCodecError("stereo canonicalization orbit did not close")


def _parse_surface(
    tokens: Sequence[str],
) -> tuple[tuple[str, ...], tuple[CompactAtomStereoRecord, ...], tuple[CompactBondStereoRecord, ...]]:
    connectivity: list[str] = []
    atoms: list[CompactAtomStereoRecord] = []
    bonds: list[CompactBondStereoRecord] = []
    fragment_index = -1
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith(ATOM_PREFIX):
            if fragment_index < 0 or index + _NUMBER_WIDTH >= len(tokens):
                raise CompactStereoCodecError("orphan atom stereo record")
            fields = token[1:-1].split(":")
            if len(fields) != 4 or fields[:2] != ["ST", "A"]:
                raise CompactStereoCodecError("invalid atom stereo record")
            label, local_parity = fields[2], fields[3]
            if label not in {"R", "S", "X"} or local_parity not in {"CW", "CCW"}:
                raise CompactStereoCodecError("invalid atom stereo labels")
            atoms.append(
                CompactAtomStereoRecord(
                    fragment_index,
                    _parse_number(tokens, index + 1),
                    label,
                    local_parity,
                )
            )
            index += 1 + _NUMBER_WIDTH
            continue
        if token.startswith(BOND_PREFIX):
            if fragment_index < 0 or index + 2 * _NUMBER_WIDTH >= len(tokens):
                raise CompactStereoCodecError("orphan bond stereo record")
            fields = token[1:-1].split(":")
            if len(fields) != 4 or fields[:2] != ["ST", "B"]:
                raise CompactStereoCodecError("invalid bond stereo record")
            label, relation = fields[2], fields[3]
            if label not in {"E", "Z"} or relation not in {"CIS", "TRANS"}:
                raise CompactStereoCodecError("invalid bond stereo labels")
            bonds.append(
                CompactBondStereoRecord(
                    fragment_index,
                    _parse_number(tokens, index + 1),
                    label,
                    relation,
                    _parse_number(tokens, index + 1 + _NUMBER_WIDTH),
                )
            )
            index += 1 + 2 * _NUMBER_WIDTH
            continue
        connectivity.append(token)
        if _is_fragment_token(token):
            fragment_index += 1
        index += 1
    return tuple(connectivity), tuple(atoms), tuple(bonds)


def _assemble_connectivity(
    connectivity_tokens: Sequence[str], chemicalgof_root: Path
) -> tuple[Chem.Mol, dict[tuple[int, int], int], dict[int, str]]:
    components: list[list[str]] = [[]]
    for token in connectivity_tokens:
        if token == "<COMP>":
            if not components[-1]:
                raise CompactStereoCodecError("empty disconnected component")
            components.append([])
        else:
            components[-1].append(token)
    if not components[-1]:
        raise CompactStereoCodecError("empty disconnected component")

    combined: Chem.Mol | None = None
    address_to_atom: dict[tuple[int, int], int] = {}
    fragment_smiles: dict[int, str] = {}
    atom_offset = 0
    fragment_offset = 0
    with _chemicalgof_import(chemicalgof_root):
        parse_module = __import__("chemicalgof.parse", fromlist=["Sequence2GoF"])
        explode_module = __import__("chemicalgof.explode", fromlist=["Assembler"])
        for component in components:
            graph = parse_module.Sequence2GoF(component)
            assembler = explode_module.Assembler(graph, strict_chirality=True)
            assembler.assemble_mol()
            component_mol = assembler.mol
            local_offset = 0
            for local_fragment_index, node in enumerate(graph._node):
                node_mol = Chem.MolFromSmiles(node.smiles)
                if node_mol is None:
                    raise CompactStereoCodecError("invalid decoded fragment")
                global_fragment = fragment_offset + local_fragment_index
                fragment_smiles[global_fragment] = node.smiles
                for local_atom in range(node_mol.GetNumAtoms()):
                    address_to_atom[(global_fragment, local_atom)] = (
                        atom_offset + local_offset + local_atom
                    )
                local_offset += node_mol.GetNumAtoms()
            if local_offset != component_mol.GetNumAtoms():
                raise CompactStereoCodecError("decoded fragment atom offsets drifted")
            combined = (
                Chem.Mol(component_mol)
                if combined is None
                else Chem.CombineMols(combined, component_mol)
            )
            atom_offset += component_mol.GetNumAtoms()
            fragment_offset += len(graph._node)
    if combined is None:
        raise CompactStereoCodecError("empty decoded surface")
    Chem.SanitizeMol(combined)
    return combined, address_to_atom, fragment_smiles


def _restore_atom_stereo(
    mol: Chem.Mol,
    records: Sequence[CompactAtomStereoRecord],
    address_to_atom: dict[tuple[int, int], int],
) -> None:
    targets: dict[int, CompactAtomStereoRecord] = {}
    for record in records:
        address = (record.fragment_index, record.local_atom_index)
        if address not in address_to_atom or address_to_atom[address] in targets:
            raise CompactStereoCodecError("invalid or duplicate atom stereo address")
        atom_index = address_to_atom[address]
        targets[atom_index] = record
        atom = mol.GetAtomWithIdx(atom_index)
        tag = (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
            if record.local_parity == "CW"
            else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        )
        atom_to_address = {value: key for key, value in address_to_atom.items()}
        neighbour_addresses = []
        for neighbour in atom.GetNeighbors():
            if neighbour.GetAtomicNum() == 1:
                neighbour_addresses.append((1, neighbour.GetIdx(), 0))
            else:
                fragment_index, local_index = atom_to_address[neighbour.GetIdx()]
                neighbour_addresses.append((0, fragment_index, local_index))
        if _permutation_is_odd(neighbour_addresses):
            tag = (
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW
                else Chem.ChiralType.CHI_TETRAHEDRAL_CW
            )
        atom.SetChiralTag(tag)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    rdCIPLabeler.AssignCIPLabels(mol)
    # The local parity is the reconstruction field.  R/S remains a readable
    # identity label, but in symmetric/pseudo-asymmetric systems RDKit may move
    # an equivalent centre or change R/S to r/s after canonical relabelling.
    # Whole-molecule chiral isomorphism plus decode/re-encode fixed point below
    # is the authoritative identity gate.


def decode_compact_stereo_surface(
    tokens: Sequence[str], *, chemicalgof_root: Path
) -> Chem.Mol:
    connectivity, atom_records, bond_records = _parse_surface(tokens)
    heavy, address_to_atom, fragment_smiles = _assemble_connectivity(
        connectivity, chemicalgof_root
    )
    atom_to_address = {value: key for key, value in address_to_atom.items()}
    expanded = Chem.AddHs(heavy)
    for record in bond_records:
        if record.fragment_index not in fragment_smiles:
            raise CompactStereoCodecError("invalid bond fragment address")
        local_bonds = _local_double_bonds(fragment_smiles[record.fragment_index])
        if record.local_bond_ordinal >= len(local_bonds):
            raise CompactStereoCodecError("invalid local double-bond ordinal")
        local_begin, local_end = local_bonds[record.local_bond_ordinal]
        begin = address_to_atom[(record.fragment_index, local_begin)]
        end = address_to_atom[(record.fragment_index, local_end)]
        bond = expanded.GetBondBetweenAtoms(begin, end)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise CompactStereoCodecError("addressed decoded bond is not double")

        def candidates(endpoint: int, opposite: int) -> list[int]:
            heavy_candidates = sorted(
                (
                    neighbour.GetIdx()
                    for neighbour in expanded.GetAtomWithIdx(endpoint).GetNeighbors()
                    if neighbour.GetIdx() != opposite
                    and neighbour.GetAtomicNum() != 1
                ),
                key=lambda atom_index: atom_to_address[atom_index],
            )
            hydrogen_candidates = sorted(
                neighbour.GetIdx()
                for neighbour in expanded.GetAtomWithIdx(endpoint).GetNeighbors()
                if neighbour.GetIdx() != opposite and neighbour.GetAtomicNum() == 1
            )
            return heavy_candidates + hydrogen_candidates

        support0 = record.support_selector_code // 4
        support1 = record.support_selector_code % 4
        candidates0 = candidates(begin, end)
        candidates1 = candidates(end, begin)
        if support0 >= len(candidates0) or support1 >= len(candidates1):
            raise CompactStereoCodecError("bond support selector is out of range")
        bond.SetStereoAtoms(candidates0[support0], candidates1[support1])
        bond.SetStereo(
            Chem.BondStereo.STEREOCIS
            if record.support_relation == "CIS"
            else Chem.BondStereo.STEREOTRANS
        )

    if bond_records:
        Chem.SetDoubleBondNeighborDirections(expanded)
    restored = Chem.RemoveHs(expanded)
    _restore_atom_stereo(restored, atom_records, address_to_atom)
    Chem.AssignStereochemistry(restored, cleanIt=False, force=True)
    rdCIPLabeler.AssignCIPLabels(restored)
    for record in bond_records:
        local_begin, local_end = _local_double_bonds(
            fragment_smiles[record.fragment_index]
        )[record.local_bond_ordinal]
        begin = address_to_atom[(record.fragment_index, local_begin)]
        end = address_to_atom[(record.fragment_index, local_end)]
        bond = restored.GetBondBetweenAtoms(begin, end)
        current = bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else None
        if current != record.label:
            raise CompactStereoCodecError(
                f"restored bond CIP mismatch: expected {record.label}, got {current}"
            )
    return restored


def strict_round_trip(
    source_mol: Chem.Mol, *, chemicalgof_root: Path
) -> CompactStereoSurface:
    surface = encode_compact_stereo_surface(
        source_mol, chemicalgof_root=chemicalgof_root
    )
    restored = decode_compact_stereo_surface(
        surface.tokens, chemicalgof_root=chemicalgof_root
    )
    actual = Chem.MolToSmiles(restored, canonical=True, isomericSmiles=True)
    expected_mol = Chem.MolFromSmiles(surface.source_isomeric_smiles)
    if expected_mol is None:
        raise CompactStereoCodecError("stored source identity cannot be reparsed")
    same_identity = _same_stereo_identity(expected_mol, restored)
    if not same_identity:
        raise CompactStereoCodecError(
            f"strict stereo round trip mismatch: {surface.source_isomeric_smiles} != {actual}"
        )
    replay = encode_compact_stereo_surface(
        restored, chemicalgof_root=chemicalgof_root
    )
    if replay.tokens != surface.tokens:
        raise CompactStereoCodecError("compact stereo surface is not a decode/re-encode fixed point")
    return surface
