"""Stereo-free pure-motif phrases with ordered molecule-local anchors.

This module is deliberately tokenizer independent.  It projects the already
validated P1 topology augmentation into the logical language that will later
be consumed by a chemical lexer.  GraphPorts is not part of this surface.

The projection preserves three distinct domains:

* a stereo-free pure-motif identity string;
* ordered anchor occurrences, each bound to one attachment atom;
* the component and cross-motif topology sidecar used for reconstruction.

Two candidate renderings are exposed for the later phrase-boundary experiment:
one explicit prefix boundary per motif and an implicit rendering whose phrase
spans live in the sidecar.  Both decode to the same logical phrase sequence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

from rdkit import Chem

from most_t5_next.r1.adapter.p1_topology_augmentation_v1 import (
    validate_topology_augmentation,
)


SURFACE_SCHEMA_VERSION = "most-t5-next/stereo-free-anchored-motif-surface/v2"
MOTIF_CANONICALIZATION_POLICY = (
    "stereo-free port-labelled RDKit graph; anchor labels excluded from "
    "canonical traversal; canonical output-order remaps occurrence slots"
)
EXPLICIT_MOTIF_BOUNDARY = "<MOST:MOTIF>"
COMPONENT_SEPARATOR = "[.]"
BOUNDARY_MODES = frozenset({"explicit", "implicit"})
ANCHOR_RE = re.compile(r"^<([0-9]+)\*>$")
ANCHOR_LIKE_RE = re.compile(r"<[^>]*\*>")
STEREO_MARKERS = ("@", "/", "\\")
_ANCHOR_IN_FRAGMENT_RE = re.compile(r"<([0-9]+)\*>")
_DUMMY_ISOTOPE_OFFSET = 10000
_DUMMY_ISOTOPE_LIMIT = 65535


class AnchoredMotifSurfaceError(ValueError):
    """The anchored surface cannot be derived or decoded losslessly."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AnchoredMotifSurfaceError(f"{field} must be an integer")
    return int(value)


def anchor_token(anchor_id: int) -> str:
    anchor_id = _plain_int(anchor_id, "anchor_id")
    if anchor_id < 0:
        raise AnchoredMotifSurfaceError("anchor_id must be nonnegative")
    return f"<{anchor_id}*>"


def parse_anchor_token(token: str) -> int:
    if not isinstance(token, str):
        raise AnchoredMotifSurfaceError("anchor token must be a string")
    match = ANCHOR_RE.fullmatch(token)
    if match is None or ANCHOR_LIKE_RE.findall(token) != [token]:
        raise AnchoredMotifSurfaceError("malformed anchor token")
    decimal = match.group(1)
    if len(decimal) > 1 and decimal.startswith("0"):
        raise AnchoredMotifSurfaceError("anchor token uses noncanonical decimal")
    return int(decimal)


def _require_pure_motif(value: object) -> str:
    if not isinstance(value, str) or len(value) < 2 or not (
        value.startswith("[") and value.endswith("]")
    ):
        raise AnchoredMotifSurfaceError("pure motif must use the frozen outer-bracket form")
    if any(character in value for character in ("\x00", "\t", "\r", "\n")):
        raise AnchoredMotifSurfaceError("pure motif contains a forbidden control character")
    if any(marker in value for marker in STEREO_MARKERS):
        raise AnchoredMotifSurfaceError("pure motif leaks a stereochemical marker")
    if ANCHOR_LIKE_RE.search(value) is not None:
        raise AnchoredMotifSurfaceError("pure motif still contains an anchor-like lexeme")
    return value


def project_legacy_fragment(fragment: str) -> tuple[str, tuple[int, ...]]:
    """Lexically project one historical fragment without graph canonicalization.

    This compatibility helper preserves encounter order.  New vocabulary and
    surface code must use :func:`canonicalize_legacy_fragment`, which removes
    traversal aliases and returns the corresponding canonical slot order.
    """

    if not isinstance(fragment, str) or not fragment:
        raise AnchoredMotifSurfaceError("legacy fragment must be nonempty")
    matches = list(re.finditer(r"<([0-9]+)\*>", fragment))
    tokens = [match.group(0) for match in matches]
    if ANCHOR_LIKE_RE.findall(fragment) != tokens:
        raise AnchoredMotifSurfaceError("legacy fragment contains malformed anchor text")
    anchor_ids = tuple(parse_anchor_token(token) for token in tokens)
    if len(anchor_ids) != len(set(anchor_ids)):
        raise AnchoredMotifSurfaceError("one motif cannot contain the same anchor twice")
    core = re.sub(r"<[0-9]+\*>", "", fragment)
    if not core or "<" in core or ">" in core:
        raise AnchoredMotifSurfaceError("anchor deletion did not leave one closed motif core")
    pure = _require_pure_motif(f"[{core}]")
    return pure, anchor_ids


def _anchor_fragment_to_mol(
    fragment: str,
) -> tuple[Chem.Mol, dict[int, int]]:
    """Parse custom anchors as degree-one dummy atoms and retain their IDs."""

    _pure, lexical_anchor_ids = project_legacy_fragment(fragment)
    if any(anchor_id + _DUMMY_ISOTOPE_OFFSET > _DUMMY_ISOTOPE_LIMIT for anchor_id in lexical_anchor_ids):
        raise AnchoredMotifSurfaceError("anchor ID exceeds the RDKit isotope transport domain")
    rdkit_smiles = _ANCHOR_IN_FRAGMENT_RE.sub(
        lambda match: f"[{_DUMMY_ISOTOPE_OFFSET + int(match.group(1))}*]",
        fragment,
    )
    molecule = Chem.MolFromSmiles(rdkit_smiles)
    if molecule is None or molecule.GetNumAtoms() <= 0:
        raise AnchoredMotifSurfaceError("legacy fragment is not a valid molecular graph")
    anchor_by_atom: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        isotope = int(atom.GetIsotope())
        if isotope < _DUMMY_ISOTOPE_OFFSET:
            raise AnchoredMotifSurfaceError("legacy fragment contains a non-anchor dummy atom")
        if atom.GetDegree() != 1:
            raise AnchoredMotifSurfaceError("anchor dummy must have exactly one motif neighbor")
        bond = atom.GetBonds()[0]
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            raise AnchoredMotifSurfaceError("anchored surface only admits SINGLE boundary bonds")
        anchor_id = isotope - _DUMMY_ISOTOPE_OFFSET
        if anchor_id in anchor_by_atom.values():
            raise AnchoredMotifSurfaceError("one motif cannot contain the same anchor twice")
        anchor_by_atom[atom.GetIdx()] = anchor_id
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    if tuple(sorted(anchor_by_atom.values())) != tuple(sorted(lexical_anchor_ids)):
        raise AnchoredMotifSurfaceError("anchor transport changed the lexical anchor domain")
    Chem.RemoveStereochemistry(molecule)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise AnchoredMotifSurfaceError("cannot sanitize the port-labelled motif graph") from exc
    return molecule, anchor_by_atom


def _smiles_output_order(molecule: Chem.Mol) -> tuple[int, ...]:
    try:
        text = molecule.GetProp("_smilesAtomOutputOrder").strip()
        if not (text.startswith("[") and text.endswith("]")):
            raise ValueError("missing brackets")
        raw = [int(value) for value in text[1:-1].split(",") if value.strip()]
    except Exception as exc:
        raise AnchoredMotifSurfaceError("RDKit did not expose canonical atom output order") from exc
    if (
        not isinstance(raw, list)
        or len(raw) != molecule.GetNumAtoms()
        or sorted(raw) != list(range(molecule.GetNumAtoms()))
    ):
        raise AnchoredMotifSurfaceError("RDKit canonical atom output order is invalid")
    return tuple(int(value) for value in raw)


def canonicalize_legacy_fragment(fragment: str) -> tuple[str, tuple[int, ...]]:
    """Canonicalize one stereo-free port-labelled motif graph.

    Anchor numbers are occurrence-local edge labels, so they are deliberately
    excluded while RDKit chooses the motif traversal.  The returned anchor ID
    sequence is then read in that canonical atom-output order.  It is the exact
    slot permutation that callers must apply to attachment atoms and edges.
    """

    molecule, anchor_by_atom = _anchor_fragment_to_mol(fragment)
    rendered = Chem.Mol(molecule)
    kekulized = False
    try:
        Chem.Kekulize(rendered, clearAromaticFlags=True)
        kekulized = True
    except Exception:
        kekulized = False
    canonical_with_ports = Chem.MolToSmiles(
        rendered,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=kekulized,
    )
    if "." in canonical_with_ports:
        raise AnchoredMotifSurfaceError("one motif fragment must be connected")
    output_order = _smiles_output_order(rendered)
    canonical_anchor_ids = tuple(
        anchor_by_atom[atom_index]
        for atom_index in output_order
        if atom_index in anchor_by_atom
    )
    if canonical_with_ports.count("*") != len(canonical_anchor_ids):
        raise AnchoredMotifSurfaceError("canonical SMILES did not preserve every motif port")
    parts = canonical_with_ports.split("*")
    canonical_fragment = parts[0]
    for anchor_id, suffix in zip(canonical_anchor_ids, parts[1:]):
        canonical_fragment += anchor_token(anchor_id) + suffix
    pure, observed_anchor_ids = project_legacy_fragment(canonical_fragment)
    if observed_anchor_ids != canonical_anchor_ids:
        raise AnchoredMotifSurfaceError("canonical port order is not lexically reversible")
    if restore_canonical_legacy_fragment(pure, canonical_anchor_ids) != canonical_fragment:
        raise AnchoredMotifSurfaceError("canonical motif and slot permutation are not reversible")
    return pure, canonical_anchor_ids


def restore_canonical_legacy_fragment(
    pure_motif: str, anchor_ids: Sequence[int]
) -> str:
    """Return the deterministic legacy spelling used by the original tokenizer.

    The old decoder filled empty ``()`` slots from left to right.  If an
    occurrence was not represented by an empty branch, it used one prefix and
    then a suffix.  This function preserves that behaviour only as a reversible
    audit rendering; the model-facing phrase keeps anchors as separate tokens.
    """

    pure = _require_pure_motif(pure_motif)
    anchors = [anchor_token(value) for value in anchor_ids]
    parsed_ids = [parse_anchor_token(token) for token in anchors]
    if len(parsed_ids) != len(set(parsed_ids)):
        raise AnchoredMotifSurfaceError("one motif cannot contain the same anchor twice")
    inner = pure[1:-1]
    slot_count = inner.count("()")
    if len(anchors) > slot_count and anchors:
        inner = anchors.pop(0) + inner
    for _ in range(slot_count):
        if not anchors:
            raise AnchoredMotifSurfaceError("pure motif declares more slots than anchors")
        inner = inner.replace("()", f"({anchors.pop(0)})", 1)
    inner += "".join(anchors)
    restored = inner
    projected_pure, projected_anchors = project_legacy_fragment(restored)
    if projected_pure != pure or projected_anchors != tuple(parsed_ids):
        raise AnchoredMotifSurfaceError("legacy fragment projection is not a fixed point")
    return restored


@dataclass(frozen=True)
class AnchorOccurrence:
    anchor_id: int
    slot_ordinal: int
    model_atom_index: int
    source_atom_index: int

    def __post_init__(self) -> None:
        for field in (
            "anchor_id",
            "slot_ordinal",
            "model_atom_index",
            "source_atom_index",
        ):
            value = _plain_int(getattr(self, field), field)
            if value < 0:
                raise AnchoredMotifSurfaceError(f"{field} must be nonnegative")

    @property
    def token(self) -> str:
        return anchor_token(self.anchor_id)


@dataclass(frozen=True)
class AnchoredMotifPhrase:
    logical_motif_index: int
    pure_motif: str
    motif_atom_indices: tuple[int, ...]
    anchors: tuple[AnchorOccurrence, ...]

    def __post_init__(self) -> None:
        logical = _plain_int(self.logical_motif_index, "logical_motif_index")
        if logical < 0:
            raise AnchoredMotifSurfaceError("logical_motif_index must be nonnegative")
        _require_pure_motif(self.pure_motif)
        atoms = tuple(
            _plain_int(value, "motif_atom_indices") for value in self.motif_atom_indices
        )
        if not atoms or atoms != tuple(sorted(set(atoms))) or atoms[0] < 0:
            raise AnchoredMotifSurfaceError("motif atoms must be nonempty, sorted and unique")
        if any(not isinstance(value, AnchorOccurrence) for value in self.anchors):
            raise AnchoredMotifSurfaceError("anchors contain an unknown occurrence type")
        if tuple(anchor.slot_ordinal for anchor in self.anchors) != tuple(
            range(len(self.anchors))
        ):
            raise AnchoredMotifSurfaceError("anchor occurrences must follow dense slot order")
        if len({anchor.anchor_id for anchor in self.anchors}) != len(self.anchors):
            raise AnchoredMotifSurfaceError("one motif repeats an anchor ID")
        if any(anchor.model_atom_index not in atoms for anchor in self.anchors):
            raise AnchoredMotifSurfaceError("anchor occurrence is outside the motif atom set")
        restored = restore_canonical_legacy_fragment(
            self.pure_motif, tuple(anchor.anchor_id for anchor in self.anchors)
        )
        if canonicalize_legacy_fragment(restored) != (
            self.pure_motif,
            tuple(anchor.anchor_id for anchor in self.anchors),
        ):
            raise AnchoredMotifSurfaceError(
                "motif identity or occurrence slots are not graph-canonical"
            )

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(anchor.token for anchor in self.anchors) + (self.pure_motif,)

    @property
    def carrier_offset(self) -> int:
        return len(self.anchors)


@dataclass(frozen=True)
class AnchoredSurfaceRendering:
    boundary_mode: str
    tokens: tuple[str, ...]
    phrase_spans: tuple[tuple[int, int], ...]
    motif_to_carrier: tuple[int, ...]
    anchor_token_positions: tuple[tuple[int, ...], ...]
    component_token_ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.boundary_mode not in BOUNDARY_MODES:
            raise AnchoredMotifSurfaceError("unknown phrase boundary mode")
        motif_count = len(self.phrase_spans)
        if not (
            len(self.motif_to_carrier)
            == len(self.anchor_token_positions)
            == motif_count
        ):
            raise AnchoredMotifSurfaceError("rendering motif arrays are misaligned")
        cursor = 0
        for start, stop in self.phrase_spans:
            if not (0 <= start < stop <= len(self.tokens)) or start < cursor:
                raise AnchoredMotifSurfaceError("phrase spans are invalid or unordered")
            cursor = stop
        if any(
            not (start <= carrier < stop)
            for (start, stop), carrier in zip(self.phrase_spans, self.motif_to_carrier)
        ):
            raise AnchoredMotifSurfaceError("motif carrier is outside its phrase")
        for motif_index, positions in enumerate(self.anchor_token_positions):
            start, stop = self.phrase_spans[motif_index]
            if any(not start <= position < stop for position in positions):
                raise AnchoredMotifSurfaceError("anchor token is outside its phrase")


@dataclass(frozen=True)
class StereoFreeAnchoredSurface:
    schema_version: str
    member_id: str
    model_atom_count: int
    source_atom_count: int
    model_to_source_atom_index: tuple[int, ...]
    atom_is_attachment: tuple[bool, ...]
    phrases: tuple[AnchoredMotifPhrase, ...]
    component_motif_ranges: tuple[tuple[int, int], ...]
    cross_motif_bonds: tuple[tuple[int, int, int, int, int, int, int], ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SURFACE_SCHEMA_VERSION:
            raise AnchoredMotifSurfaceError("unexpected anchored surface schema")
        if not isinstance(self.member_id, str) or not self.member_id:
            raise AnchoredMotifSurfaceError("member_id must be nonempty")
        model_count = _plain_int(self.model_atom_count, "model_atom_count")
        source_count = _plain_int(self.source_atom_count, "source_atom_count")
        if model_count <= 0 or source_count <= 0:
            raise AnchoredMotifSurfaceError("atom counts must be positive")
        source_map = tuple(
            _plain_int(value, "model_to_source_atom_index")
            for value in self.model_to_source_atom_index
        )
        if (
            len(source_map) != model_count
            or source_map != tuple(sorted(set(source_map)))
            or any(value < 0 or value >= source_count for value in source_map)
        ):
            raise AnchoredMotifSurfaceError("model/source atom mapping is invalid")
        if len(self.atom_is_attachment) != model_count or any(
            not isinstance(value, bool) for value in self.atom_is_attachment
        ):
            raise AnchoredMotifSurfaceError("attachment mask is invalid")
        if not self.phrases or tuple(
            phrase.logical_motif_index for phrase in self.phrases
        ) != tuple(range(len(self.phrases))):
            raise AnchoredMotifSurfaceError("phrases are not in dense logical order")
        owned = tuple(sorted(atom for phrase in self.phrases for atom in phrase.motif_atom_indices))
        if owned != tuple(range(model_count)):
            raise AnchoredMotifSurfaceError("motif phrases are not an atom partition")
        _validate_component_ranges(self.component_motif_ranges, len(self.phrases))
        _validate_surface_connections(self)
        if self.artifact_sha256 != _surface_artifact_sha256(self):
            raise AnchoredMotifSurfaceError("anchored surface artifact digest mismatch")

    def render(self, boundary_mode: str) -> AnchoredSurfaceRendering:
        return _render(self, boundary_mode)


def _validate_component_ranges(
    ranges: Sequence[tuple[int, int]], motif_count: int
) -> None:
    if not ranges:
        raise AnchoredMotifSurfaceError("component ranges must be nonempty")
    cursor = 0
    for start, stop in ranges:
        if start != cursor or stop <= start or stop > motif_count:
            raise AnchoredMotifSurfaceError("component ranges are not a contiguous partition")
        cursor = stop
    if cursor != motif_count:
        raise AnchoredMotifSurfaceError("component ranges do not cover all motifs")


def _validate_surface_connections(surface: StereoFreeAnchoredSurface) -> None:
    occurrences: dict[int, list[tuple[int, AnchorOccurrence]]] = {}
    for motif_index, phrase in enumerate(surface.phrases):
        for occurrence in phrase.anchors:
            occurrences.setdefault(occurrence.anchor_id, []).append((motif_index, occurrence))
    if sorted(occurrences) != list(range(len(surface.cross_motif_bonds))):
        raise AnchoredMotifSurfaceError("anchor IDs are not dense over cross-motif bonds")
    attachment_atoms: set[int] = set()
    for edge_id, row in enumerate(surface.cross_motif_bonds):
        if len(row) != 7:
            raise AnchoredMotifSurfaceError("cross-motif bond row has the wrong width")
        values = tuple(_plain_int(value, "cross_motif_bond") for value in row)
        (
            declared_edge,
            anchor_id,
            left_motif,
            left_slot,
            right_motif,
            right_slot,
            bond_order,
        ) = values
        if declared_edge != edge_id or anchor_id != edge_id or bond_order != 1:
            raise AnchoredMotifSurfaceError("only dense SINGLE anchor bonds are admitted")
        paired = occurrences.get(anchor_id, [])
        if len(paired) != 2 or paired[0][0] == paired[1][0]:
            raise AnchoredMotifSurfaceError("every anchor must occur in two distinct motifs")
        observed = {(motif, occurrence.slot_ordinal) for motif, occurrence in paired}
        if observed != {(left_motif, left_slot), (right_motif, right_slot)}:
            raise AnchoredMotifSurfaceError("anchor occurrences disagree with the bond sidecar")
        attachment_atoms.update(occurrence.model_atom_index for _, occurrence in paired)
    expected_mask = tuple(
        atom_index in attachment_atoms for atom_index in range(surface.model_atom_count)
    )
    if expected_mask != surface.atom_is_attachment:
        raise AnchoredMotifSurfaceError("attachment mask differs from anchor endpoints")


def _surface_projection_values(
    *,
    schema_version: str,
    member_id: str,
    model_atom_count: int,
    source_atom_count: int,
    model_to_source_atom_index: Sequence[int],
    atom_is_attachment: Sequence[bool],
    phrases: Sequence[AnchoredMotifPhrase],
    component_motif_ranges: Sequence[tuple[int, int]],
    cross_motif_bonds: Sequence[tuple[int, int, int, int, int, int, int]],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "motif_canonicalization_policy": MOTIF_CANONICALIZATION_POLICY,
        "member_id": member_id,
        "model_atom_count": model_atom_count,
        "source_atom_count": source_atom_count,
        "model_to_source_atom_index": list(model_to_source_atom_index),
        "atom_is_attachment": list(atom_is_attachment),
        "phrases": [
            {
                "logical_motif_index": phrase.logical_motif_index,
                "pure_motif": phrase.pure_motif,
                "pure_motif_sha256": _sha256_text(phrase.pure_motif),
                "motif_atom_indices": list(phrase.motif_atom_indices),
                "anchors": [
                    {
                        "anchor_id": occurrence.anchor_id,
                        "slot_ordinal": occurrence.slot_ordinal,
                        "model_atom_index": occurrence.model_atom_index,
                        "source_atom_index": occurrence.source_atom_index,
                    }
                    for occurrence in phrase.anchors
                ],
                "canonical_legacy_fragment": restore_canonical_legacy_fragment(
                    phrase.pure_motif,
                    tuple(occurrence.anchor_id for occurrence in phrase.anchors),
                ),
            }
            for phrase in phrases
        ],
        "component_motif_ranges": [list(row) for row in component_motif_ranges],
        "cross_motif_bonds": [list(row) for row in cross_motif_bonds],
        "geometry_or_e3fp_recomputed": False,
        "graphports_exposed_to_model": False,
    }


def _surface_projection(surface: StereoFreeAnchoredSurface) -> dict[str, object]:
    return _surface_projection_values(
        schema_version=surface.schema_version,
        member_id=surface.member_id,
        model_atom_count=surface.model_atom_count,
        source_atom_count=surface.source_atom_count,
        model_to_source_atom_index=surface.model_to_source_atom_index,
        atom_is_attachment=surface.atom_is_attachment,
        phrases=surface.phrases,
        component_motif_ranges=surface.component_motif_ranges,
        cross_motif_bonds=surface.cross_motif_bonds,
    )


def _surface_artifact_sha256(surface: StereoFreeAnchoredSurface) -> str:
    return _canonical_json_sha256(_surface_projection(surface))


def surface_document(surface: StereoFreeAnchoredSurface) -> dict[str, object]:
    """Return a canonical JSON-compatible document including its content hash."""

    document = _surface_projection(surface)
    document["artifact_sha256"] = surface.artifact_sha256
    return document


def stereo_free_canonical_smiles(molecule: Chem.Mol) -> str:
    """Return the identity domain used by the anchored language.

    The architecture deliberately carries stereochemical state outside the
    motif token stream.  Isotopes, charges, radicals, aromaticity and bond
    orders remain part of this identity; only atom/bond stereochemistry is
    removed.
    """

    if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() <= 0:
        raise AnchoredMotifSurfaceError("stereo-free identity requires a nonempty RDKit Mol")
    normalized = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(normalized)
    try:
        Chem.SanitizeMol(normalized)
    except Exception as exc:
        raise AnchoredMotifSurfaceError("cannot sanitize molecule for stereo-free identity") from exc
    return Chem.MolToSmiles(
        normalized,
        canonical=True,
        isomericSmiles=True,
    )


def reconstruct_stereo_free_molecule(surface: StereoFreeAnchoredSurface) -> Chem.Mol:
    """Decode all canonical phrases and reconnect their paired anchors."""

    if not isinstance(surface, StereoFreeAnchoredSurface):
        raise AnchoredMotifSurfaceError("reconstruction requires an anchored surface")
    fragments: list[Chem.Mol] = []
    for phrase in surface.phrases:
        fragment = restore_canonical_legacy_fragment(
            phrase.pure_motif,
            tuple(occurrence.anchor_id for occurrence in phrase.anchors),
        )
        rdkit_smiles = _ANCHOR_IN_FRAGMENT_RE.sub(
            lambda match: f"[{_DUMMY_ISOTOPE_OFFSET + int(match.group(1))}*]",
            fragment,
        )
        molecule = Chem.MolFromSmiles(rdkit_smiles)
        if molecule is None:
            raise AnchoredMotifSurfaceError("canonical motif phrase could not be decoded")
        fragments.append(molecule)
    combined = fragments[0]
    for fragment in fragments[1:]:
        combined = Chem.CombineMols(combined, fragment)
    editable = Chem.RWMol(combined)
    dummy_rows: dict[int, list[int]] = {}
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        isotope = int(atom.GetIsotope())
        if isotope < _DUMMY_ISOTOPE_OFFSET or atom.GetDegree() != 1:
            raise AnchoredMotifSurfaceError("decoded motif contains an invalid anchor dummy")
        dummy_rows.setdefault(isotope - _DUMMY_ISOTOPE_OFFSET, []).append(atom.GetIdx())
    if sorted(dummy_rows) != list(range(len(surface.cross_motif_bonds))):
        raise AnchoredMotifSurfaceError("decoded anchor domain differs from surface edges")
    dummy_indices: list[int] = []
    for anchor_id in sorted(dummy_rows):
        pair = dummy_rows[anchor_id]
        if len(pair) != 2:
            raise AnchoredMotifSurfaceError("every decoded anchor must occur exactly twice")
        left_neighbor = editable.GetAtomWithIdx(pair[0]).GetNeighbors()[0].GetIdx()
        right_neighbor = editable.GetAtomWithIdx(pair[1]).GetNeighbors()[0].GetIdx()
        if left_neighbor == right_neighbor or editable.GetBondBetweenAtoms(left_neighbor, right_neighbor):
            raise AnchoredMotifSurfaceError("decoded anchor would create an invalid connection")
        editable.AddBond(left_neighbor, right_neighbor, Chem.rdchem.BondType.SINGLE)
        dummy_indices.extend(pair)
    for atom_index in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_index)
    reconstructed = editable.GetMol()
    Chem.RemoveStereochemistry(reconstructed)
    try:
        Chem.SanitizeMol(reconstructed)
    except Exception as exc:
        raise AnchoredMotifSurfaceError("decoded molecule failed sanitization") from exc
    if reconstructed.GetNumAtoms() != surface.model_atom_count:
        raise AnchoredMotifSurfaceError("decoded molecule atom count differs from the model domain")
    return reconstructed


def validate_stereo_free_molecule_round_trip(
    surface: StereoFreeAnchoredSurface,
    model_molecule: Chem.Mol,
) -> str:
    """Require encode/decode identity with the projected source molecule."""

    if not isinstance(model_molecule, Chem.Mol):
        raise AnchoredMotifSurfaceError("round-trip source must be an RDKit Mol")
    if model_molecule.GetNumAtoms() != surface.model_atom_count:
        raise AnchoredMotifSurfaceError("round-trip source atom count differs from the surface")
    expected = stereo_free_canonical_smiles(model_molecule)
    observed = stereo_free_canonical_smiles(reconstruct_stereo_free_molecule(surface))
    if observed != expected:
        raise AnchoredMotifSurfaceError(
            f"stereo-free molecule round trip changed identity: {expected!r} -> {observed!r}"
        )
    return observed


def build_stereo_free_anchored_surface(
    topology_document: Mapping[str, object],
) -> StereoFreeAnchoredSurface:
    """Derive one anchored surface from a validated P1 topology augmentation."""

    validate_topology_augmentation(topology_document)
    member = topology_document["member"]
    atoms = topology_document["atom_universe"]
    motifs = topology_document["logical_motif_domain"]
    assert isinstance(member, dict) and isinstance(atoms, dict) and isinstance(motifs, dict)

    pure_rows = motifs["pure_motif_token"]
    groups = motifs["motif_atom_indices"]
    anchor_rows = motifs["motif_slot_anchor_ids"]
    atom_rows = motifs["motif_slot_atom_indices"]
    source_rows = motifs["motif_slot_source_atom_indices"]
    edge_id_by_source_anchor = {
        int(bond["source_anchor_id"]): int(bond["edge_id"])
        for bond in motifs["cross_motif_bonds"]
    }
    phrases = []
    surface_slot_by_source_anchor: dict[tuple[int, int], int] = {}
    for motif_index, (pure, group, anchors, model_atoms, source_atoms) in enumerate(
        zip(pure_rows, groups, anchor_rows, atom_rows, source_rows)
    ):
        source_anchor_ids = tuple(_plain_int(value, "source anchor") for value in anchors)
        old_rows = {
            source_anchor_id: (
                _plain_int(model_atom, "model_atom_index"),
                _plain_int(source_atom, "source_atom_index"),
            )
            for source_anchor_id, model_atom, source_atom in zip(
                source_anchor_ids, model_atoms, source_atoms
            )
        }
        canonical_pure, canonical_source_anchor_ids = canonicalize_legacy_fragment(
            restore_canonical_legacy_fragment(_require_pure_motif(pure), source_anchor_ids)
        )
        occurrences = tuple(
            AnchorOccurrence(
                anchor_id=edge_id_by_source_anchor[source_anchor_id],
                slot_ordinal=slot,
                model_atom_index=old_rows[source_anchor_id][0],
                source_atom_index=old_rows[source_anchor_id][1],
            )
            for slot, source_anchor_id in enumerate(canonical_source_anchor_ids)
        )
        for slot, source_anchor_id in enumerate(canonical_source_anchor_ids):
            surface_slot_by_source_anchor[(motif_index, source_anchor_id)] = slot
        phrases.append(
            AnchoredMotifPhrase(
                logical_motif_index=motif_index,
                pure_motif=canonical_pure,
                motif_atom_indices=tuple(int(value) for value in group),
                anchors=occurrences,
            )
        )

    bond_rows = []
    for bond in motifs["cross_motif_bonds"]:
        if bond["bond_type"] != "single":
            raise AnchoredMotifSurfaceError(
                "anchored surface v1 only admits SINGLE cross-motif bonds"
            )
        left = bond["left"]
        right = bond["right"]
        bond_rows.append(
            (
                int(bond["edge_id"]),
                int(bond["edge_id"]),
                int(left["logical_motif_index"]),
                surface_slot_by_source_anchor[
                    (int(left["logical_motif_index"]), int(bond["source_anchor_id"]))
                ],
                int(right["logical_motif_index"]),
                surface_slot_by_source_anchor[
                    (int(right["logical_motif_index"]), int(bond["source_anchor_id"]))
                ],
                1,
            )
        )

    arguments = dict(
        schema_version=SURFACE_SCHEMA_VERSION,
        member_id=str(member["member_id"]),
        model_atom_count=int(atoms["model_atom_count"]),
        source_atom_count=int(atoms["source_atom_count"]),
        model_to_source_atom_index=tuple(int(value) for value in atoms["model_to_source_atom_index"]),
        atom_is_attachment=tuple(bool(value) for value in atoms["atom_is_attachment"]),
        phrases=tuple(phrases),
        component_motif_ranges=tuple(
            (int(row[0]), int(row[1])) for row in motifs["component_logical_motif_ranges"]
        ),
        cross_motif_bonds=tuple(bond_rows),
    )
    digest = _canonical_json_sha256(_surface_projection_values(**arguments))
    return StereoFreeAnchoredSurface(artifact_sha256=digest, **arguments)


def _component_ranges_from_bonds(
    motif_count: int, cross_motif_bonds: Sequence[Mapping[str, object]]
) -> tuple[tuple[int, int], ...]:
    adjacency = [set() for _ in range(motif_count)]
    for bond in cross_motif_bonds:
        try:
            left = int(bond["left"]["logical_motif_index"])  # type: ignore[index]
            right = int(bond["right"]["logical_motif_index"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise AnchoredMotifSurfaceError("cross-motif bond endpoints are malformed") from exc
        if not (0 <= left < motif_count and 0 <= right < motif_count) or left == right:
            raise AnchoredMotifSurfaceError("cross-motif bond endpoints are invalid")
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited = [False] * motif_count
    components: list[tuple[int, ...]] = []
    for root in range(motif_count):
        if visited[root]:
            continue
        pending = [root]
        visited[root] = True
        members = []
        while pending:
            motif = pending.pop()
            members.append(motif)
            for neighbor in sorted(adjacency[motif], reverse=True):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        component = tuple(sorted(members))
        if component != tuple(range(component[0], component[-1] + 1)):
            raise AnchoredMotifSurfaceError(
                "disconnected components are interleaved in logical motif order"
            )
        components.append(component)
    components.sort(key=lambda row: row[0])
    return tuple((row[0], row[-1] + 1) for row in components)


def build_stereo_free_anchored_surface_from_persisted_pair(
    *,
    member_id: str,
    source_atom_count: int,
    model_to_source_atom_index: Sequence[int],
    atom_is_attachment: Sequence[bool],
    motif_atom_indices: Sequence[Sequence[int]],
    exact_motif_lexemes: Sequence[str],
    motif_slot_atom_indices: Sequence[Sequence[int]],
    cross_motif_bonds: Sequence[Mapping[str, object]],
) -> StereoFreeAnchoredSurface:
    """Join production-v2 exact lexemes with the persisted paired topology.

    The paired wire intentionally omitted the historical exact lexeme strings
    and their molecule-local source anchor IDs.  Production-v2 retained their
    digests, while its global motif census retains the digest-to-lexeme map.
    This boundary reattaches those strings to the paired slot/atom sidecar and
    renumbers model-facing anchors by the paired canonical ``edge_id``.  No SDF,
    coordinates or E3FP are read.
    """

    if not isinstance(member_id, str) or not member_id:
        raise AnchoredMotifSurfaceError("member_id must be nonempty")
    source_count = _plain_int(source_atom_count, "source_atom_count")
    source_map = tuple(
        _plain_int(value, "model_to_source_atom_index")
        for value in model_to_source_atom_index
    )
    model_count = len(source_map)
    if (
        source_count <= 0
        or model_count <= 0
        or source_map != tuple(sorted(set(source_map)))
        or any(value < 0 or value >= source_count for value in source_map)
    ):
        raise AnchoredMotifSurfaceError("model/source atom mapping is invalid")
    attachment = tuple(atom_is_attachment)
    if len(attachment) != model_count or any(not isinstance(value, bool) for value in attachment):
        raise AnchoredMotifSurfaceError("attachment mask is invalid")
    groups = tuple(tuple(_plain_int(atom, "motif atom") for atom in row) for row in motif_atom_indices)
    slots = tuple(tuple(_plain_int(atom, "slot atom") for atom in row) for row in motif_slot_atom_indices)
    lexemes = tuple(exact_motif_lexemes)
    motif_count = len(groups)
    if motif_count <= 0 or len(slots) != motif_count or len(lexemes) != motif_count:
        raise AnchoredMotifSurfaceError("persisted motif arrays are misaligned")
    owner = [-1] * model_count
    for motif_index, group in enumerate(groups):
        if not group or group != tuple(sorted(set(group))):
            raise AnchoredMotifSurfaceError("motif atoms must be nonempty, sorted and unique")
        for atom in group:
            if not 0 <= atom < model_count or owner[atom] != -1:
                raise AnchoredMotifSurfaceError("motif groups are not an atom partition")
            owner[atom] = motif_index
        if any(atom not in group for atom in slots[motif_index]):
            raise AnchoredMotifSurfaceError("slot atom is outside its motif")
    if -1 in owner:
        raise AnchoredMotifSurfaceError("motif groups leave atoms uncovered")

    slot_to_edge: dict[tuple[int, int], int] = {}
    persisted_edge_endpoints: list[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ] = []
    for expected_edge, bond in enumerate(cross_motif_bonds):
        if not isinstance(bond, Mapping) or bond.get("bond_type") != "single":
            raise AnchoredMotifSurfaceError("anchored surface v1 only admits SINGLE bonds")
        edge_id = _plain_int(bond.get("edge_id"), "edge_id")
        if edge_id != expected_edge:
            raise AnchoredMotifSurfaceError("cross-motif edge IDs must be dense and ordered")
        endpoints = []
        for side in ("left", "right"):
            endpoint = bond.get(side)
            if not isinstance(endpoint, Mapping):
                raise AnchoredMotifSurfaceError("cross-motif endpoint is malformed")
            motif = _plain_int(endpoint.get("logical_motif_index"), "endpoint motif")
            slot = _plain_int(endpoint.get("slot_ordinal"), "endpoint slot")
            atom = _plain_int(endpoint.get("atom_index"), "endpoint atom")
            if not 0 <= motif < motif_count or not 0 <= slot < len(slots[motif]):
                raise AnchoredMotifSurfaceError("cross-motif endpoint is out of range")
            if slots[motif][slot] != atom:
                raise AnchoredMotifSurfaceError("cross-motif endpoint differs from its slot atom")
            if (motif, slot) in slot_to_edge:
                raise AnchoredMotifSurfaceError("one motif slot occurs in multiple edges")
            slot_to_edge[(motif, slot)] = edge_id
            endpoints.append((motif, slot, atom))
        if endpoints[0][:2] >= endpoints[1][:2] or endpoints[0][0] == endpoints[1][0]:
            raise AnchoredMotifSurfaceError("cross-motif endpoints are not canonical")
        persisted_edge_endpoints.append((endpoints[0], endpoints[1]))
    expected_slots = {
        (motif_index, slot_index)
        for motif_index, row in enumerate(slots)
        for slot_index in range(len(row))
    }
    if set(slot_to_edge) != expected_slots:
        raise AnchoredMotifSurfaceError("cross-motif bonds do not cover every motif slot")

    # Historical anchor occurrences follow their lexical encounter order.  That
    # order is not the persisted GraphPorts slot order (for example ``<5*>C<8*>``
    # may have persisted slots ``[edge-to-8, edge-to-5]``).  Pair anchors first,
    # then recover the matching persisted endpoint; never zip the two orders.
    source_anchor_occurrences: dict[int, list[tuple[int, int]]] = {}
    projected_lexemes: list[tuple[str, tuple[int, ...]]] = []
    for motif_index, (lexeme, slot_atoms) in enumerate(zip(lexemes, slots)):
        pure, source_anchor_ids = canonicalize_legacy_fragment(lexeme)
        if len(source_anchor_ids) != len(slot_atoms):
            raise AnchoredMotifSurfaceError(
                "exact lexeme anchor count differs from persisted slot count"
            )
        projected_lexemes.append((pure, source_anchor_ids))
        for lexical_slot, source_anchor_id in enumerate(source_anchor_ids):
            source_anchor_occurrences.setdefault(source_anchor_id, []).append(
                (motif_index, lexical_slot)
            )
    if sorted(source_anchor_occurrences) != list(range(len(persisted_edge_endpoints))):
        raise AnchoredMotifSurfaceError("source lexeme anchor IDs are not dense")

    anchor_for_motif: list[list[AnchorOccurrence | None]] = [
        [None] * len(source_ids) for _, source_ids in projected_lexemes
    ]
    surface_slot_for_edge: dict[int, dict[int, int]] = {}
    unused_edges = set(range(len(persisted_edge_endpoints)))
    for source_anchor_id in sorted(source_anchor_occurrences):
        rows = source_anchor_occurrences[source_anchor_id]
        if len(rows) != 2 or rows[0][0] == rows[1][0]:
            raise AnchoredMotifSurfaceError(
                "source lexeme anchor pairing disagrees with the persisted edge pairing"
            )
        motif_pair = {rows[0][0], rows[1][0]}
        candidates = [
            edge_id
            for edge_id in sorted(unused_edges)
            if {endpoint[0] for endpoint in persisted_edge_endpoints[edge_id]}
            == motif_pair
        ]
        if len(candidates) > 1:
            atom_matched = [
                edge_id
                for edge_id in candidates
                if source_anchor_id
                in {endpoint[2] for endpoint in persisted_edge_endpoints[edge_id]}
            ]
            if len(atom_matched) == 1:
                candidates = atom_matched
        if len(candidates) != 1:
            raise AnchoredMotifSurfaceError(
                "source anchor cannot be mapped uniquely to a persisted edge"
            )
        edge_id = candidates[0]
        unused_edges.remove(edge_id)
        endpoints_by_motif = {
            endpoint[0]: endpoint for endpoint in persisted_edge_endpoints[edge_id]
        }
        surface_slots: dict[int, int] = {}
        for motif_index, lexical_slot in rows:
            endpoint = endpoints_by_motif.get(motif_index)
            if endpoint is None:
                raise AnchoredMotifSurfaceError(
                    "source anchor motif is absent from its persisted edge"
                )
            atom = endpoint[2]
            anchor_for_motif[motif_index][lexical_slot] = AnchorOccurrence(
                anchor_id=edge_id,
                slot_ordinal=lexical_slot,
                model_atom_index=atom,
                source_atom_index=source_map[atom],
            )
            surface_slots[motif_index] = lexical_slot
        surface_slot_for_edge[edge_id] = surface_slots
    if unused_edges:
        raise AnchoredMotifSurfaceError("source anchors do not form one bijection to persisted edges")

    phrases = []
    for motif_index, (group, projected, row) in enumerate(
        zip(groups, projected_lexemes, anchor_for_motif)
    ):
        pure, _ = projected
        if any(occurrence is None for occurrence in row):
            raise AnchoredMotifSurfaceError("one lexical anchor was not mapped")
        phrases.append(
            AnchoredMotifPhrase(
                logical_motif_index=motif_index,
                pure_motif=pure,
                motif_atom_indices=group,
                anchors=tuple(occurrence for occurrence in row if occurrence is not None),
            )
        )

    edge_rows: list[tuple[int, int, int, int, int, int, int]] = []
    for edge_id, endpoints in enumerate(persisted_edge_endpoints):
        left, right = endpoints
        edge_rows.append(
            (
                edge_id,
                edge_id,
                left[0],
                surface_slot_for_edge[edge_id][left[0]],
                right[0],
                surface_slot_for_edge[edge_id][right[0]],
                1,
            )
        )

    arguments = dict(
        schema_version=SURFACE_SCHEMA_VERSION,
        member_id=member_id,
        model_atom_count=model_count,
        source_atom_count=source_count,
        model_to_source_atom_index=source_map,
        atom_is_attachment=attachment,
        phrases=tuple(phrases),
        component_motif_ranges=_component_ranges_from_bonds(motif_count, cross_motif_bonds),
        cross_motif_bonds=tuple(edge_rows),
    )
    digest = _canonical_json_sha256(_surface_projection_values(**arguments))
    return StereoFreeAnchoredSurface(artifact_sha256=digest, **arguments)


def _render(
    surface: StereoFreeAnchoredSurface, boundary_mode: str
) -> AnchoredSurfaceRendering:
    if boundary_mode not in BOUNDARY_MODES:
        raise AnchoredMotifSurfaceError("unknown phrase boundary mode")
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    carriers: list[int] = []
    anchor_positions: list[tuple[int, ...]] = []
    component_ranges: list[tuple[int, int]] = []
    for component_index, (motif_start, motif_stop) in enumerate(
        surface.component_motif_ranges
    ):
        if component_index:
            tokens.append(COMPONENT_SEPARATOR)
        component_token_start = len(tokens)
        for motif_index in range(motif_start, motif_stop):
            phrase = surface.phrases[motif_index]
            phrase_start = len(tokens)
            if boundary_mode == "explicit":
                tokens.append(EXPLICIT_MOTIF_BOUNDARY)
            positions = []
            for occurrence in phrase.anchors:
                positions.append(len(tokens))
                tokens.append(occurrence.token)
            carrier = len(tokens)
            tokens.append(phrase.pure_motif)
            phrase_stop = len(tokens)
            spans.append((phrase_start, phrase_stop))
            carriers.append(carrier)
            anchor_positions.append(tuple(positions))
        component_ranges.append((component_token_start, len(tokens)))
    rendering = AnchoredSurfaceRendering(
        boundary_mode=boundary_mode,
        tokens=tuple(tokens),
        phrase_spans=tuple(spans),
        motif_to_carrier=tuple(carriers),
        anchor_token_positions=tuple(anchor_positions),
        component_token_ranges=tuple(component_ranges),
    )
    decoded = decode_rendering(rendering.tokens, boundary_mode)
    expected = tuple(
        (tuple(occurrence.anchor_id for occurrence in phrase.anchors), phrase.pure_motif)
        for phrase in surface.phrases
    )
    if decoded[0] != expected or decoded[1] != surface.component_motif_ranges:
        raise AnchoredMotifSurfaceError("rendered surface did not decode to the source phrases")
    return rendering


def decode_rendering(
    tokens: Sequence[str], boundary_mode: str
) -> tuple[tuple[tuple[tuple[int, ...], str], ...], tuple[tuple[int, int], ...]]:
    """Decode model-facing tokens to logical ``(anchor IDs, pure motif)`` rows."""

    if boundary_mode not in BOUNDARY_MODES:
        raise AnchoredMotifSurfaceError("unknown phrase boundary mode")
    rows: list[tuple[tuple[int, ...], str]] = []
    component_ranges: list[tuple[int, int]] = []
    component_start = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == COMPONENT_SEPARATOR:
            if len(rows) == component_start:
                raise AnchoredMotifSurfaceError("empty molecular component")
            component_ranges.append((component_start, len(rows)))
            component_start = len(rows)
            index += 1
            continue
        if boundary_mode == "explicit":
            if token != EXPLICIT_MOTIF_BOUNDARY:
                raise AnchoredMotifSurfaceError("explicit phrase is missing its boundary token")
            index += 1
        anchors: list[int] = []
        while index < len(tokens) and ANCHOR_RE.fullmatch(tokens[index] or ""):
            anchors.append(parse_anchor_token(tokens[index]))
            index += 1
        if index >= len(tokens):
            raise AnchoredMotifSurfaceError("phrase is missing its pure-motif carrier")
        pure = _require_pure_motif(tokens[index])
        index += 1
        if len(anchors) != len(set(anchors)):
            raise AnchoredMotifSurfaceError("one decoded phrase repeats an anchor")
        restored = restore_canonical_legacy_fragment(pure, anchors)
        if canonicalize_legacy_fragment(restored) != (pure, tuple(anchors)):
            raise AnchoredMotifSurfaceError("decoded phrase is a noncanonical traversal alias")
        rows.append((tuple(anchors), pure))
    if len(rows) == component_start:
        raise AnchoredMotifSurfaceError("surface ends with an empty component")
    component_ranges.append((component_start, len(rows)))
    return tuple(rows), tuple(component_ranges)


__all__ = [
    "ANCHOR_RE",
    "BOUNDARY_MODES",
    "COMPONENT_SEPARATOR",
    "EXPLICIT_MOTIF_BOUNDARY",
    "MOTIF_CANONICALIZATION_POLICY",
    "SURFACE_SCHEMA_VERSION",
    "AnchorOccurrence",
    "AnchoredMotifPhrase",
    "AnchoredMotifSurfaceError",
    "AnchoredSurfaceRendering",
    "StereoFreeAnchoredSurface",
    "anchor_token",
    "build_stereo_free_anchored_surface",
    "build_stereo_free_anchored_surface_from_persisted_pair",
    "canonicalize_legacy_fragment",
    "decode_rendering",
    "parse_anchor_token",
    "project_legacy_fragment",
    "reconstruct_stereo_free_molecule",
    "restore_canonical_legacy_fragment",
    "stereo_free_canonical_smiles",
    "surface_document",
    "validate_stereo_free_molecule_round_trip",
]
