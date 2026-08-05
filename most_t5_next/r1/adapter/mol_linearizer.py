"""Deterministic, molecule-native motif linearization for the R1 sidecar.

The historical ``CAMT5.representation.linearize`` starts from a SMILES string,
re-parses it, mutates the parsed molecule while removing stereochemistry and
kekulizing it, and relies on unordered containers during motif traversal.  This
module deliberately has a narrower contract:

* the caller supplies an existing :class:`rdkit.Chem.rdchem.Mol`;
* its atom indices are the only atom identifiers used in the result;
* the supplied molecule is read only -- all fragment construction happens in
  newly allocated molecules; and
* every ordering decision is defined by atom indices, bond endpoints, and a
  documented DFS tie break.

It preserves the legacy *motif notion*: each ring and each non-single bond is
seeded, overlapping seeds are merged, every remaining atom is a singleton, and
each bond between motifs is represented by a matched pair of isotope-labelled
dummy anchors.  It intentionally does not claim byte-identical historical
fragment strings: deterministic output and index provenance are the R1
contracts to be released before a new vocabulary/checkpoint is created.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Chem.rdchem import Bond, BondType, Mol


_DUMMY_ISOTOPE_OFFSET = 10_000
_DUMMY_ISOTOPE_LIMIT = 65_535
_DUMMY_PATTERN = re.compile(r"\[(\d+)\*[^\]]*\]")


class LinearizationError(ValueError):
    """Raised when a supplied molecule cannot satisfy the sidecar contract."""


@dataclass(frozen=True)
class CrossMotifBond:
    """One inter-motif bond and its deterministic dummy-anchor identifier.

    ``motif_a < motif_b`` refers to the canonical motif IDs in
    :attr:`LinearizationMetadata.canonical_motif_atom_groups`.  ``atom_a`` and
    ``atom_b`` remain *original input* atom indices, not fragment-local ones.
    """

    anchor_id: int
    motif_a: int
    motif_b: int
    atom_a: int
    atom_b: int
    bond_type: str


@dataclass(frozen=True)
class LinearizationMetadata:
    """Traceability metadata accompanying a :class:`LinearizationResult`."""

    schema_version: str
    input_atom_count: int
    input_bond_count: int
    input_atom_indices: tuple[int, ...]
    canonical_motif_atom_groups: tuple[tuple[int, ...], ...]
    fragment_motif_ids: tuple[int, ...]
    component_fragment_ranges: tuple[tuple[int, int], ...]
    cross_motif_bonds: tuple[CrossMotifBond, ...]
    input_mutation_policy: str
    ordering_policy: str
    fragment_rendering_policy: str


@dataclass(frozen=True)
class LinearizationResult:
    """A deterministic motif sequence for one existing RDKit molecule.

    ``fragment_sequence`` contains only real motif fragment SMILES.  It is
    one-to-one with ``motif_atom_groups``.  For each position ``i``,
    ``motif_atom_groups[i]`` contains sorted original input atom indices for
    that fragment; it never contains local fragment indices or dummy atoms.

    Disconnected molecular components are represented in the metadata as
    half-open ``component_fragment_ranges``.  :attr:`fragment_string` is a
    legacy-friendly rendering that inserts ``[.]`` between those ranges.
    """

    fragment_sequence: tuple[str, ...]
    motif_atom_groups: tuple[tuple[int, ...], ...]
    metadata: LinearizationMetadata

    @property
    def fragment_string(self) -> str:
        """Render the sequence using the historical outer-token convention."""

        rendered: list[str] = []
        for component_index, (start, stop) in enumerate(
            self.metadata.component_fragment_ranges
        ):
            if component_index:
                rendered.append("[.]")
            rendered.extend(f"[{fragment}]" for fragment in self.fragment_sequence[start:stop])
        return " ".join(rendered)


class _UnionFind:
    """Index-ordered union-find with deterministic smallest-root selection."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


@dataclass(frozen=True)
class _Anchor:
    """Internal anchor record retaining RDKit's bond type for rendering."""

    anchor_id: int
    motif_a: int
    motif_b: int
    atom_a: int
    atom_b: int
    bond_type: BondType
    bond_type_name: str


def linearize_mol(mol: Mol) -> LinearizationResult:
    """Linearize an existing RDKit molecule without changing it.

    Parameters
    ----------
    mol:
        An already prepared RDKit molecule.  Ring information must be
        available; ordinary sanitized SDF/RDKit molecules satisfy this.

    Returns
    -------
    LinearizationResult
        Deterministically ordered fragment SMILES, atom groups expressed in
        the original ``mol`` index space, and provenance metadata.

    Notes
    -----
    This function never calls ``MolFromSmiles`` and never creates a replacement
    molecule for the input.  It reads the supplied molecule directly and
    allocates new, fragment-only molecules solely to render the output tokens.
    """

    if not isinstance(mol, Mol):
        raise TypeError("linearize_mol expects an RDKit Mol instance")

    atom_count = mol.GetNumAtoms()
    bond_count = mol.GetNumBonds()
    canonical_groups = _canonical_motif_groups(mol)
    atom_to_motif = _atom_to_motif(canonical_groups, atom_count)
    anchors = _cross_motif_anchors(mol, atom_to_motif)
    adjacency = _motif_adjacency(len(canonical_groups), anchors)
    motif_order, component_ranges = _deterministic_dfs_order(adjacency)

    fragments = tuple(
        _render_fragment(mol, canonical_groups[motif_id], motif_id, anchors)
        for motif_id in motif_order
    )
    ordered_groups = tuple(canonical_groups[motif_id] for motif_id in motif_order)

    metadata = LinearizationMetadata(
        schema_version="r1-molecule-native-linearizer/v1",
        input_atom_count=atom_count,
        input_bond_count=bond_count,
        input_atom_indices=tuple(range(atom_count)),
        canonical_motif_atom_groups=tuple(canonical_groups),
        fragment_motif_ids=tuple(motif_order),
        component_fragment_ranges=tuple(component_ranges),
        cross_motif_bonds=tuple(
            CrossMotifBond(
                anchor_id=anchor.anchor_id,
                motif_a=anchor.motif_a,
                motif_b=anchor.motif_b,
                atom_a=anchor.atom_a,
                atom_b=anchor.atom_b,
                bond_type=anchor.bond_type_name,
            )
            for anchor in anchors
        ),
        input_mutation_policy="read_only_input; fragment-only allocations",
        ordering_policy=(
            "canonical groups by minimum original atom index; cross bonds by "
            "(motif_a,motif_b,atom_a,atom_b,bond_type); components by minimum "
            "motif id; DFS root by highest degree then lowest motif id; "
            "neighbors ascending motif id"
        ),
        fragment_rendering_policy=(
            "new fragment-only RWMol; input stereochemistry omitted to match "
            "legacy intent; canonical kekule SMILES when possible; anchor "
            "[10000+id*] rendered as <id*>"
        ),
    )
    return LinearizationResult(fragments, ordered_groups, metadata)


def _canonical_motif_groups(mol: Mol) -> list[tuple[int, ...]]:
    """Create a sorted partition following the legacy ring/non-single rule."""

    atom_count = mol.GetNumAtoms()
    if atom_count == 0:
        return []

    try:
        raw_rings = mol.GetRingInfo().AtomRings()
    except RuntimeError as exc:
        raise LinearizationError(
            "input Mol has no initialized ring information; prepare/sanitize it upstream"
        ) from exc

    ring_groups = [tuple(sorted(int(atom) for atom in ring)) for ring in raw_rings]
    non_single_groups = [
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE
    ]
    seed_groups = _deduplicate_sorted_groups(ring_groups + non_single_groups)

    union_find = _UnionFind(atom_count)
    atom_in_seed = [False] * atom_count
    for group in seed_groups:
        if not group:
            continue
        first_atom = group[0]
        for atom in group:
            atom_in_seed[atom] = True
            union_find.union(first_atom, atom)

    merged_by_root: dict[int, list[int]] = {}
    singleton_groups: list[list[int]] = []
    for atom_idx in range(atom_count):
        if not atom_in_seed[atom_idx]:
            singleton_groups.append([atom_idx])
            continue
        root = union_find.find(atom_idx)
        if root not in merged_by_root:
            merged_by_root[root] = []
        merged_by_root[root].append(atom_idx)

    groups = [tuple(group) for group in merged_by_root.values()]
    groups.extend(tuple(group) for group in singleton_groups)
    groups.sort(key=lambda group: (group[0], len(group), group))
    return groups


def _deduplicate_sorted_groups(groups: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Sort groups lexicographically and remove exact duplicates without a set."""

    ordered = sorted(group for group in groups if group)
    unique: list[tuple[int, ...]] = []
    previous: tuple[int, ...] | None = None
    for group in ordered:
        if group != previous:
            unique.append(group)
            previous = group
    return unique


def _atom_to_motif(
    canonical_groups: Sequence[tuple[int, ...]], atom_count: int
) -> list[int]:
    atom_to_motif = [-1] * atom_count
    for motif_id, group in enumerate(canonical_groups):
        for atom_idx in group:
            if atom_idx < 0 or atom_idx >= atom_count or atom_to_motif[atom_idx] != -1:
                raise LinearizationError("motif partition is invalid")
            atom_to_motif[atom_idx] = motif_id
    if any(motif_id < 0 for motif_id in atom_to_motif):
        raise LinearizationError("motif partition does not cover every input atom")
    return atom_to_motif


def _cross_motif_anchors(mol: Mol, atom_to_motif: Sequence[int]) -> list[_Anchor]:
    """Assign stable anchor IDs to all inter-motif bonds."""

    candidates: list[tuple[int, int, int, int, str, BondType]] = []
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        begin_motif = atom_to_motif[begin_atom]
        end_motif = atom_to_motif[end_atom]
        if begin_motif == end_motif:
            continue
        if begin_motif < end_motif:
            motif_a, motif_b = begin_motif, end_motif
            atom_a, atom_b = begin_atom, end_atom
        else:
            motif_a, motif_b = end_motif, begin_motif
            atom_a, atom_b = end_atom, begin_atom
        candidates.append(
            (
                motif_a,
                motif_b,
                atom_a,
                atom_b,
                str(bond.GetBondType()),
                bond.GetBondType(),
            )
        )

    candidates.sort(key=lambda item: item[:5])
    if len(candidates) + _DUMMY_ISOTOPE_OFFSET > _DUMMY_ISOTOPE_LIMIT:
        raise LinearizationError("too many cross-motif bonds for RDKit dummy isotope IDs")

    return [
        _Anchor(
            anchor_id=anchor_id,
            motif_a=motif_a,
            motif_b=motif_b,
            atom_a=atom_a,
            atom_b=atom_b,
            bond_type=bond_type,
            bond_type_name=bond_type_name,
        )
        for anchor_id, (motif_a, motif_b, atom_a, atom_b, bond_type_name, bond_type) in enumerate(
            candidates
        )
    ]


def _motif_adjacency(motif_count: int, anchors: Sequence[_Anchor]) -> list[list[int]]:
    adjacency = [[] for _ in range(motif_count)]
    for anchor in anchors:
        # Multiple molecular bonds may join the same pair of motifs.  They are
        # separate anchors, but only one graph-neighbor relation is needed for DFS.
        if anchor.motif_b not in adjacency[anchor.motif_a]:
            adjacency[anchor.motif_a].append(anchor.motif_b)
        if anchor.motif_a not in adjacency[anchor.motif_b]:
            adjacency[anchor.motif_b].append(anchor.motif_a)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def _deterministic_dfs_order(
    adjacency: Sequence[Sequence[int]],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Traverse each motif component using only explicit, stable tie breaks."""

    motif_count = len(adjacency)
    assigned_to_component = [False] * motif_count
    traversed = [False] * motif_count
    motif_order: list[int] = []
    component_ranges: list[tuple[int, int]] = []

    for lowest_motif in range(motif_count):
        if assigned_to_component[lowest_motif]:
            continue

        component: list[int] = []
        pending = [lowest_motif]
        assigned_to_component[lowest_motif] = True
        while pending:
            motif_id = pending.pop()
            component.append(motif_id)
            for neighbor in adjacency[motif_id]:
                if not assigned_to_component[neighbor]:
                    assigned_to_component[neighbor] = True
                    pending.append(neighbor)
        component.sort()

        # Match the historical intent of beginning at a highly connected motif,
        # while resolving every tie by the canonical motif ID.
        root = min(component, key=lambda motif_id: (-len(adjacency[motif_id]), motif_id))
        start = len(motif_order)
        pending = [root]
        while pending:
            motif_id = pending.pop()
            if traversed[motif_id]:
                continue
            traversed[motif_id] = True
            motif_order.append(motif_id)
            # Reverse push preserves ascending neighbor visitation in LIFO DFS.
            for neighbor in reversed(adjacency[motif_id]):
                if not traversed[neighbor]:
                    pending.append(neighbor)
        component_ranges.append((start, len(motif_order)))

    return motif_order, component_ranges


def _render_fragment(
    mol: Mol,
    motif_atoms: Sequence[int],
    motif_id: int,
    anchors: Sequence[_Anchor],
) -> str:
    """Render one motif in a newly allocated fragment-only RWMol."""

    fragment = Chem.RWMol()
    local_atom_by_original = [-1] * mol.GetNumAtoms()
    for atom_idx in motif_atoms:
        source_atom = mol.GetAtomWithIdx(atom_idx)
        copied_atom = Chem.Atom(source_atom)
        # The historical path removed stereochemistry before token rendering.
        # Do the same only on this copied atom, never on the caller's molecule.
        copied_atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        local_atom_by_original[atom_idx] = fragment.AddAtom(copied_atom)

    for source_bond in mol.GetBonds():
        begin_atom = source_bond.GetBeginAtomIdx()
        end_atom = source_bond.GetEndAtomIdx()
        local_begin = local_atom_by_original[begin_atom]
        local_end = local_atom_by_original[end_atom]
        if local_begin < 0 or local_end < 0:
            continue
        fragment.AddBond(local_begin, local_end, source_bond.GetBondType())
        copied_bond = fragment.GetBondBetweenAtoms(local_begin, local_end)
        copied_bond.SetIsAromatic(source_bond.GetIsAromatic())
        copied_bond.SetIsConjugated(source_bond.GetIsConjugated())
        copied_bond.SetBondDir(Chem.rdchem.BondDir.NONE)
        copied_bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)

    for anchor in anchors:
        if motif_id == anchor.motif_a:
            local_atom = local_atom_by_original[anchor.atom_a]
        elif motif_id == anchor.motif_b:
            local_atom = local_atom_by_original[anchor.atom_b]
        else:
            continue
        if local_atom < 0:
            raise LinearizationError("cross-motif anchor does not match its motif group")
        dummy = Chem.Atom(0)
        dummy.SetIsotope(_DUMMY_ISOTOPE_OFFSET + anchor.anchor_id)
        dummy.SetNoImplicit(True)
        dummy_atom = fragment.AddAtom(dummy)
        fragment.AddBond(local_atom, dummy_atom, anchor.bond_type)

    fragment_mol = fragment.GetMol()
    try:
        Chem.SanitizeMol(fragment_mol)
    except Exception as exc:  # RDKit exposes several exception subclasses.
        raise LinearizationError(
            f"cannot sanitize motif {motif_id} built from input atoms {tuple(motif_atoms)}"
        ) from exc

    # Kekulization is intentionally limited to the fresh fragment.  It matches
    # the legacy token convention without touching the caller-owned molecule.
    kekulized = False
    try:
        Chem.Kekulize(fragment_mol, clearAromaticFlags=True)
        kekulized = True
    except Exception:
        # The normal canonical form remains deterministic and keeps the source
        # aromatic bond representation when Kekulization is not available.
        kekulized = False

    smiles = Chem.MolToSmiles(
        fragment_mol,
        canonical=True,
        # ``isomericSmiles`` must remain enabled for RDKit to emit the dummy
        # isotope labels.  Stereochemistry itself was cleared on copied atoms
        # and bonds above, so this does not reintroduce input stereochemistry.
        isomericSmiles=True,
        kekuleSmiles=kekulized,
    )
    return _DUMMY_PATTERN.sub(_replace_dummy_anchor, smiles)


def _replace_dummy_anchor(match: re.Match[str]) -> str:
    isotope = int(match.group(1))
    if isotope < _DUMMY_ISOTOPE_OFFSET:
        return match.group(0)
    return f"<{isotope - _DUMMY_ISOTOPE_OFFSET}*>"
