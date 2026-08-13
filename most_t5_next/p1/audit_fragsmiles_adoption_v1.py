from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import platform
import sys
from typing import Iterator, Sequence

import networkx as nx
from rdkit import Chem, rdBase
from rdkit.Chem import rdCIPLabeler


SCHEMA_VERSION = "most_t5.fragsmiles_adoption_audit.v1"
DEFAULT_CLEAVAGE_SMARTS = "[!$([+1,-1]~[-1,+1])]-&!@[*]"
STEREO_POLICIES = ("official_isomeric", "connectivity_only")


class FragSmilesAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class FragmentSidecar:
    sequence_fragment_index: int
    fragment_smiles: str
    canonical_molecule_atom_indices: tuple[int, ...]
    source_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class ConnectorSidecar:
    connector_index: int
    left_fragment_index: int
    right_fragment_index: int
    left_local_atom_index: int
    right_local_atom_index: int
    left_source_atom_index: int
    right_source_atom_index: int


@dataclass(frozen=True)
class AtomStereoSidecar:
    projected_atom_index: int
    fragment_index: int | None
    fragment_local_atom_index: int | None
    cip_label: str
    chiral_tag: str


@dataclass(frozen=True)
class BondStereoSidecar:
    projected_begin_atom_index: int
    projected_end_atom_index: int
    begin_support_atom_index: int | None
    end_support_atom_index: int | None
    begin_fragment_index: int | None
    begin_fragment_local_atom_index: int | None
    end_fragment_index: int | None
    end_fragment_local_atom_index: int | None
    begin_support_fragment_index: int | None
    begin_support_fragment_local_atom_index: int | None
    end_support_fragment_index: int | None
    end_support_fragment_local_atom_index: int | None
    stereo: str
    cip_label: str | None = None


@dataclass(frozen=True)
class StereoIdentitySidecar:
    atom_centers: tuple[AtomStereoSidecar, ...]
    double_bonds: tuple[BondStereoSidecar, ...]


@dataclass(frozen=True)
class FragSmilesRecord:
    canonical_smiles: str
    fragsmiles: str
    component_surfaces: tuple[str, ...]
    tokens: tuple[str, ...]
    fragments: tuple[FragmentSidecar, ...]
    connectors: tuple[ConnectorSidecar, ...]
    stereo_identity: StereoIdentitySidecar


def _extract_stereo_identity(source_mol: Chem.Mol) -> StereoIdentitySidecar:
    annotated = Chem.Mol(source_mol)
    for atom in annotated.GetAtoms():
        atom.SetIntProp("most_source_atom_index", atom.GetIdx())
    Chem.AssignStereochemistry(annotated, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(annotated)

    projected = Chem.Mol(annotated)
    Chem.RemoveStereochemistry(projected)
    projected = Chem.RemoveHs(projected, sanitize=True)
    old_to_projected = {
        atom.GetIntProp("most_source_atom_index"): atom.GetIdx()
        for atom in projected.GetAtoms()
    }

    centers = []
    for atom_index, cip_label in Chem.FindMolChiralCenters(
        annotated,
        includeUnassigned=False,
        useLegacyImplementation=False,
    ):
        if atom_index not in old_to_projected:
            continue
        centers.append(
            AtomStereoSidecar(
                projected_atom_index=old_to_projected[atom_index],
                fragment_index=None,
                fragment_local_atom_index=None,
                cip_label=str(cip_label),
                chiral_tag=str(annotated.GetAtomWithIdx(atom_index).GetChiralTag()),
            )
        )

    double_bonds = []
    for bond in annotated.GetBonds():
        stereo = str(bond.GetStereo())
        if stereo in {"STEREONONE", "STEREOANY"}:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in old_to_projected or end not in old_to_projected:
            raise FragSmilesAuditError("stereo bond endpoint removed by projection")
        supports = tuple(int(value) for value in bond.GetStereoAtoms())
        if len(supports) != 2:
            raise FragSmilesAuditError("defined bond stereo lacks two support atoms")
        double_bonds.append(
            BondStereoSidecar(
                projected_begin_atom_index=old_to_projected[begin],
                projected_end_atom_index=old_to_projected[end],
                begin_support_atom_index=old_to_projected.get(supports[0]),
                end_support_atom_index=old_to_projected.get(supports[1]),
                begin_fragment_index=None,
                begin_fragment_local_atom_index=None,
                end_fragment_index=None,
                end_fragment_local_atom_index=None,
                begin_support_fragment_index=None,
                begin_support_fragment_local_atom_index=None,
                end_support_fragment_index=None,
                end_support_fragment_local_atom_index=None,
                stereo=stereo,
                cip_label=(
                    bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else None
                ),
            )
        )
    return StereoIdentitySidecar(
        atom_centers=tuple(centers),
        double_bonds=tuple(double_bonds),
    )


def _bind_stereo_identity_to_fragments(
    stereo: StereoIdentitySidecar,
    fragments: Sequence[FragmentSidecar],
) -> StereoIdentitySidecar:
    addresses = {}
    for fragment in fragments:
        for local_index, source_index in enumerate(fragment.source_atom_indices):
            if source_index in addresses:
                raise FragSmilesAuditError("source atom occurs in multiple fragments")
            addresses[source_index] = (
                fragment.sequence_fragment_index,
                local_index,
            )

    def address(source_index: int | None) -> tuple[int | None, int | None]:
        if source_index is None:
            return None, None
        if source_index not in addresses:
            raise FragSmilesAuditError("stereo atom is absent from fragment sidecar")
        return addresses[source_index]

    centers = []
    for center in stereo.atom_centers:
        fragment_index, local_index = address(center.projected_atom_index)
        centers.append(
            AtomStereoSidecar(
                projected_atom_index=center.projected_atom_index,
                fragment_index=fragment_index,
                fragment_local_atom_index=local_index,
                cip_label=center.cip_label,
                chiral_tag=center.chiral_tag,
            )
        )
    bonds = []
    for bond in stereo.double_bonds:
        begin = address(bond.projected_begin_atom_index)
        end = address(bond.projected_end_atom_index)
        begin_support = address(bond.begin_support_atom_index)
        end_support = address(bond.end_support_atom_index)
        bonds.append(
            BondStereoSidecar(
                projected_begin_atom_index=bond.projected_begin_atom_index,
                projected_end_atom_index=bond.projected_end_atom_index,
                begin_support_atom_index=bond.begin_support_atom_index,
                end_support_atom_index=bond.end_support_atom_index,
                begin_fragment_index=begin[0],
                begin_fragment_local_atom_index=begin[1],
                end_fragment_index=end[0],
                end_fragment_local_atom_index=end[1],
                begin_support_fragment_index=begin_support[0],
                begin_support_fragment_local_atom_index=begin_support[1],
                end_support_fragment_index=end_support[0],
                end_support_fragment_local_atom_index=end_support[1],
                stereo=bond.stereo,
                cip_label=bond.cip_label,
            )
        )
    return StereoIdentitySidecar(atom_centers=tuple(centers), double_bonds=tuple(bonds))


def stereo_address_signature(record: FragSmilesRecord) -> tuple:
    atoms = tuple(
        sorted(
            (
                center.fragment_index,
                center.fragment_local_atom_index,
                center.cip_label,
            )
            for center in record.stereo_identity.atom_centers
        )
    )
    bonds = tuple(
        sorted(
            (
                bond.begin_fragment_index,
                bond.begin_fragment_local_atom_index,
                bond.end_fragment_index,
                bond.end_fragment_local_atom_index,
                bond.begin_support_fragment_index,
                bond.begin_support_fragment_local_atom_index,
                bond.end_support_fragment_index,
                bond.end_support_fragment_local_atom_index,
                bond.stereo,
            )
            for bond in record.stereo_identity.double_bonds
        )
    )
    return atoms, bonds


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _chemicalgof_import(reference_root: Path) -> Iterator[object]:
    if not hasattr(itertools, "pairwise"):
        # chemicalgof uses the Python 3.10 stdlib helper.  The production
        # chemistry runtime is Python 3.8, so install its exact two-item
        # sliding-window semantics without changing chemicalgof traversal.
        def _pairwise(iterable):
            iterator = iter(iterable)
            try:
                previous = next(iterator)
            except StopIteration:
                return
            for current in iterator:
                yield previous, current
                previous = current

        itertools.pairwise = _pairwise
    root = reference_root.resolve()
    package_init = root / "chemicalgof" / "__init__.py"
    if not package_init.is_file():
        raise FragSmilesAuditError(f"chemicalgof package missing: {package_init}")
    existing = sys.modules.get("chemicalgof")
    if existing is not None:
        loaded = Path(existing.__file__).resolve()
        if root not in loaded.parents:
            raise FragSmilesAuditError(
                f"chemicalgof already loaded from unexpected path: {loaded}"
            )
        yield existing
        return
    sys.path.insert(0, str(root))
    try:
        package = importlib.import_module("chemicalgof")
        loaded = Path(package.__file__).resolve()
        if root not in loaded.parents:
            raise FragSmilesAuditError(
                f"chemicalgof resolved outside requested source: {loaded}"
            )
        yield package
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)


def _canonicalize_source_mol(
    source_mol: Chem.Mol, *, stereo_policy: str
) -> tuple[Chem.Mol, str, tuple[int, ...]]:
    if stereo_policy not in STEREO_POLICIES:
        raise FragSmilesAuditError(f"unsupported stereo policy: {stereo_policy}")
    mol = Chem.Mol(source_mol)
    if stereo_policy == "connectivity_only":
        Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol, sanitize=True)
    if mol.GetNumAtoms() == 0:
        raise FragSmilesAuditError("empty molecule after heavy-atom projection")
    canonical_smiles = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )
    raw_order = mol.GetProp("_smilesAtomOutputOrder")
    order = tuple(int(value) for value in ast.literal_eval(raw_order))
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None or len(order) != canonical_mol.GetNumAtoms():
        raise FragSmilesAuditError("RDKit canonical atom-output order is invalid")
    return canonical_mol, canonical_smiles, order


def _node_match(left: dict, right: dict) -> bool:
    return (
        left.get("smiles") == right.get("smiles")
        and left.get("chirality", {}) == right.get("chirality", {})
    )


def _edge_match(left: dict, right: dict) -> bool:
    return left.get("aB") == right.get("aB") and left.get("stereo") == right.get(
        "stereo"
    )


def _sequence_to_reduction_mapping(parsed_graph, reduction_graph, memberships):
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        parsed_graph,
        reduction_graph,
        node_match=_node_match,
        edge_match=_edge_match,
    )
    parsed_nodes = tuple(parsed_graph.nodes)
    candidates = []
    for mapping in matcher.isomorphisms_iter():
        key = tuple(tuple(memberships[mapping[node]]) for node in parsed_nodes)
        candidates.append((key, mapping))
    if not candidates:
        raise FragSmilesAuditError("serialized fragSMILES graph does not match reduction graph")
    candidates.sort(key=lambda item: item[0])
    return parsed_nodes, candidates[0][1]


def encode_with_sidecar(
    source_mol: Chem.Mol,
    *,
    chemicalgof_root: Path,
    capitalize_chirality: bool = True,
    stereo_policy: str = "official_isomeric",
) -> FragSmilesRecord:
    stereo_identity = _extract_stereo_identity(source_mol)
    canonical_mol, canonical_smiles, canonical_to_source = _canonicalize_source_mol(
        source_mol, stereo_policy=stereo_policy
    )
    component_atom_maps: list[tuple[int, ...]] = []
    component_mols = Chem.GetMolFrags(
        canonical_mol,
        asMols=True,
        sanitizeFrags=True,
        fragsMolAtomMapping=component_atom_maps,
    )
    if len(component_mols) > 1:
        component_records = [
            encode_with_sidecar(
                component,
                chemicalgof_root=chemicalgof_root,
                capitalize_chirality=capitalize_chirality,
                stereo_policy=stereo_policy,
            )
            for component in component_mols
        ]
        fragments = []
        connectors = []
        tokens = []
        for component_index, (record, atom_map) in enumerate(
            zip(component_records, component_atom_maps)
        ):
            if len(record.component_surfaces) != 1:
                raise FragSmilesAuditError("nested disconnected component encoding")
            fragment_offset = len(fragments)
            if component_index:
                tokens.append("<COMP>")
            tokens.extend(record.tokens)
            for fragment in record.fragments:
                canonical_indices = tuple(
                    atom_map[index]
                    for index in fragment.canonical_molecule_atom_indices
                )
                fragments.append(
                    FragmentSidecar(
                        sequence_fragment_index=fragment_offset
                        + fragment.sequence_fragment_index,
                        fragment_smiles=fragment.fragment_smiles,
                        canonical_molecule_atom_indices=canonical_indices,
                        source_atom_indices=tuple(
                            canonical_to_source[index] for index in canonical_indices
                        ),
                    )
                )
            for connector in record.connectors:
                left_index = fragment_offset + connector.left_fragment_index
                right_index = fragment_offset + connector.right_fragment_index
                connectors.append(
                    ConnectorSidecar(
                        connector_index=len(connectors),
                        left_fragment_index=left_index,
                        right_fragment_index=right_index,
                        left_local_atom_index=connector.left_local_atom_index,
                        right_local_atom_index=connector.right_local_atom_index,
                        left_source_atom_index=fragments[
                            left_index
                        ].source_atom_indices[connector.left_local_atom_index],
                        right_source_atom_index=fragments[
                            right_index
                        ].source_atom_indices[connector.right_local_atom_index],
                    )
                )
        component_surfaces = tuple(
            record.fragsmiles for record in component_records
        )
        fragment_rows = tuple(fragments)
        return FragSmilesRecord(
            canonical_smiles=canonical_smiles,
            fragsmiles="<COMP>".join(component_surfaces),
            component_surfaces=component_surfaces,
            tokens=tuple(tokens),
            fragments=fragment_rows,
            connectors=tuple(connectors),
            stereo_identity=_bind_stereo_identity_to_fragments(
                stereo_identity, fragment_rows
            ),
        )
    with _chemicalgof_import(chemicalgof_root) as chemicalgof:
        reduce_module = importlib.import_module("chemicalgof.reduce")
        gof_module = importlib.import_module("chemicalgof.gof")
        write_module = importlib.import_module("chemicalgof.write")
        parse_module = importlib.import_module("chemicalgof.parse")

        decompositer = reduce_module.Decompositer(DEFAULT_CLEAVAGE_SMARTS)
        bond_matches = canonical_mol.GetSubstructMatches(decompositer.cleavage_pattern)
        chiral = Chem.FindMolChiralCenters(
            canonical_mol, useLegacyImplementation=False
        )
        if capitalize_chirality:
            all_chiral = {idx: label.upper() for idx, label in chiral}
        else:
            all_chiral = dict(chiral)

        if bond_matches:
            fragment_mols = decompositer.fragment(canonical_mol, bond_matches)
            fragment_bonds = decompositer.frag_bonds(fragment_mols, bond_matches)
            fragment_smiles = decompositer.ultimate_smiles(fragment_mols)
            inter_atoms = {atom for pair in bond_matches for atom in pair}
            node_chirality = decompositer.setup_nodes_attributes(
                fragment_smiles, all_chiral, inter_atoms
            )
            canonical_memberships = tuple(
                tuple(mapping[index] for index in sorted(mapping))
                for mapping in decompositer.mapsFrag2Mol
            )
        else:
            fragment_bonds = ((),)
            fragment_smiles = (Chem.MolToSmiles(canonical_mol),)
            node_chirality = (
                {idx: label for idx, label in sorted(all_chiral.items())},
            )
            canonical_memberships = (tuple(range(canonical_mol.GetNumAtoms())),)

        graph = gof_module.DiGraphFrags()
        nodes = [
            gof_module.FragNode(smiles=smiles, chirality=chirality)
            for smiles, chirality in zip(fragment_smiles, node_chirality)
        ]
        graph.add_nodes_from(nodes)
        memberships = {
            node: canonical_memberships[index] for index, node in enumerate(nodes)
        }
        if bond_matches:
            for map_mol_to_fragment, node, neighbours in zip(
                decompositer.mapsMol2Frag, nodes, fragment_bonds
            ):
                for atom_index, neighbour_index in neighbours:
                    neighbour = nodes[decompositer.fragsIdxs[neighbour_index]]
                    stereo = (
                        all_chiral.get(atom_index)
                        if node.numPotAtomLinkers > 1
                        else None
                    )
                    graph.add_edge(
                        node,
                        neighbour,
                        aB=map_mol_to_fragment[atom_index],
                        stereo=stereo,
                    )

        fragsmiles = write_module.GoF2fragSMILES(graph, canonize=True)
        public_surface = chemicalgof.encode(
            canonical_smiles,
            canonical=True,
            random=False,
            capitalize_chirality=capitalize_chirality,
        )
        if fragsmiles != public_surface:
            raise FragSmilesAuditError("sidecar reduction diverges from official encode()")

        parsed_graph = parse_module.fragSMILES2GoF(fragsmiles)
        sequence_nodes, sequence_to_reduction = _sequence_to_reduction_mapping(
            parsed_graph, graph, memberships
        )
        sequence_index = {node: index for index, node in enumerate(sequence_nodes)}
        fragments = []
        for index, parsed_node in enumerate(sequence_nodes):
            reduction_node = sequence_to_reduction[parsed_node]
            canonical_indices = tuple(memberships[reduction_node])
            fragments.append(
                FragmentSidecar(
                    sequence_fragment_index=index,
                    fragment_smiles=parsed_node.smiles,
                    canonical_molecule_atom_indices=canonical_indices,
                    source_atom_indices=tuple(
                        canonical_to_source[atom] for atom in canonical_indices
                    ),
                )
            )

        connectors = []
        seen = set()
        for left, right in parsed_graph.edges:
            pair = frozenset((left, right))
            if pair in seen:
                continue
            seen.add(pair)
            left_index = sequence_index[left]
            right_index = sequence_index[right]
            if right_index < left_index:
                left, right = right, left
                left_index, right_index = right_index, left_index
            left_local = int(parsed_graph.edges[left, right]["aB"])
            right_local = int(parsed_graph.edges[right, left]["aB"])
            left_source = fragments[left_index].source_atom_indices[left_local]
            right_source = fragments[right_index].source_atom_indices[right_local]
            connectors.append(
                ConnectorSidecar(
                    connector_index=len(connectors),
                    left_fragment_index=left_index,
                    right_fragment_index=right_index,
                    left_local_atom_index=left_local,
                    right_local_atom_index=right_local,
                    left_source_atom_index=left_source,
                    right_source_atom_index=right_source,
                )
            )

        fragment_rows = tuple(fragments)
        return FragSmilesRecord(
            canonical_smiles=canonical_smiles,
            fragsmiles=fragsmiles,
            component_surfaces=(fragsmiles,),
            tokens=tuple(chemicalgof.split(fragsmiles)),
            fragments=fragment_rows,
            connectors=tuple(connectors),
            stereo_identity=_bind_stereo_identity_to_fragments(
                stereo_identity, fragment_rows
            ),
        )


def _canonical_identity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise FragSmilesAuditError("decoded fragSMILES is not valid SMILES")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _canonical_connectivity_identity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise FragSmilesAuditError("decoded fragSMILES is not valid SMILES")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def audit_molecules(
    molecules: Sequence[Chem.Mol],
    *,
    chemicalgof_root: Path,
    stereo_policy: str = "official_isomeric",
) -> dict:
    failures = []
    rows = []
    for index, mol in enumerate(molecules):
        try:
            record = encode_with_sidecar(
                mol,
                chemicalgof_root=chemicalgof_root,
                stereo_policy=stereo_policy,
            )
            with _chemicalgof_import(chemicalgof_root) as chemicalgof:
                decoded = ".".join(
                    chemicalgof.decode(surface, strict_chirality=True)
                    for surface in record.component_surfaces
                )
            identity_ok = _canonical_identity(decoded) == record.canonical_smiles
            connectivity_ok = _canonical_connectivity_identity(
                decoded
            ) == _canonical_connectivity_identity(record.canonical_smiles)
            source_atoms = sorted(
                atom
                for fragment in record.fragments
                for atom in fragment.source_atom_indices
            )
            projected_mol, _, _ = _canonicalize_source_mol(
                mol, stereo_policy=stereo_policy
            )
            sidecar_ok = source_atoms == list(range(projected_mol.GetNumAtoms()))
            reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
            reverse_surface = encode_with_sidecar(
                reverse,
                chemicalgof_root=chemicalgof_root,
                stereo_policy=stereo_policy,
            ).fragsmiles
            rows.append(
                {
                    "record_index": index,
                    "identity_round_trip": identity_ok,
                    "connectivity_round_trip": connectivity_ok,
                    "sidecar_atom_partition": sidecar_ok,
                    "renumbering_surface_invariant": reverse_surface == record.fragsmiles,
                    "token_count": len(record.tokens),
                    "fragment_count": len(record.fragments),
                    "connector_count": len(record.connectors),
                    "component_count": len(record.component_surfaces),
                    "stereo_atom_center_count": len(
                        record.stereo_identity.atom_centers
                    ),
                    "stereo_double_bond_count": len(
                        record.stereo_identity.double_bonds
                    ),
                }
            )
        except Exception as exc:  # audit must classify upstream failures
            try:
                projected, source_smiles, _ = _canonicalize_source_mol(
                    mol, stereo_policy=stereo_policy
                )
                del projected
            except Exception:
                source_smiles = None
            failures.append(
                {
                    "record_index": index,
                    "source_canonical_smiles": source_smiles,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    def count_true(field: str) -> int:
        return sum(bool(row[field]) for row in rows)

    source_init = chemicalgof_root / "chemicalgof" / "__init__.py"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "audit_only",
        "training_admission": False,
        "representation": {
            "backend": "chemicalgof/fragSMILES",
            "cleavage_smarts": DEFAULT_CLEAVAGE_SMARTS,
            "stereo_policy": stereo_policy,
            "explicit_hydrogens_removed": True,
            "sidecar": "fragment atoms plus connector attachment atoms",
            "stereo_identity_sidecar": "audit extraction only; decode restoration not admitted",
            "disconnected_component_separator": "<COMP>",
        },
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "networkx": nx.__version__,
            "chemicalgof_init_sha256": _sha256(source_init),
            "chemicalgof_source": str(chemicalgof_root.resolve()),
        },
        "counts": {
            "input": len(molecules),
            "encoded": len(rows),
            "failures": len(failures),
            "identity_round_trip_pass": count_true("identity_round_trip"),
            "connectivity_round_trip_pass": count_true("connectivity_round_trip"),
            "sidecar_atom_partition_pass": count_true("sidecar_atom_partition"),
            "renumbering_surface_invariant_pass": count_true(
                "renumbering_surface_invariant"
            ),
        },
        "lengths": {
            "mean_frag_tokens": (
                sum(row["token_count"] for row in rows) / len(rows) if rows else None
            ),
            "max_frag_tokens": max((row["token_count"] for row in rows), default=None),
        },
        "rows": rows,
        "failures": failures,
    }


def _read_sdf(path: Path, max_records: int) -> tuple[Chem.Mol, ...]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    molecules = []
    for mol in supplier:
        if mol is not None:
            molecules.append(mol)
        if len(molecules) >= max_records:
            break
    return tuple(molecules)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official fragSMILES adoption")
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument("--chemicalgof-root", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument(
        "--stereo-policy", choices=STEREO_POLICIES, default="official_isomeric"
    )
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    report = audit_molecules(
        _read_sdf(args.input_sdf, args.max_records),
        chemicalgof_root=args.chemicalgof_root,
        stereo_policy=args.stereo_policy,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
