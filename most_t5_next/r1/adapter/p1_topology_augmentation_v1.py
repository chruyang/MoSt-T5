"""Build the missing port topology for bounded P1 canary records.

Production-v2 already stores motif groups, ordered motif-lexeme digests,
model/source atom provenance and E3FP.  It does not retain which real atom is
attached to each anchor occurrence.  This module reruns only the frozen
molecule-native linearizer, proves its groups and lexeme digests match the
existing record, and emits that missing association.  It never computes
coordinates or E3FP and cannot admit training.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
import re
from typing import Any, Sequence


SCHEMA_VERSION = "most-t5-r1/p1-topology-augmentation/v1"
DOCUMENT_KIND = "p1_topology_augmentation"
PROJECTION_POLICY = (
    "anchor occurrence order in each logical fragment defines slot ordinal; "
    "canonical motif IDs are remapped through fragment_motif_ids; edges are "
    "sorted by logical endpoints and densely reindexed"
)
ANCHOR_RE = re.compile(r"<([0-9]+)\*>")
ANCHOR_LIKE_RE = re.compile(r"<[^<>]*\*>")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_BOND_TYPES = frozenset(
    {"single", "double", "triple", "aromatic", "dative", "other"}
)


class TopologyAugmentationError(ValueError):
    """Raised when topology cannot be proven against the existing record."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise TopologyAugmentationError(f"{name} must be a lower-case SHA-256")
    return value


def _project_fragment(fragment: str) -> tuple[str, tuple[int, ...]]:
    if not isinstance(fragment, str) or not fragment:
        raise TopologyAugmentationError("motif fragment must be nonempty text")
    matches = list(ANCHOR_RE.finditer(fragment))
    if ANCHOR_LIKE_RE.findall(fragment) != [match.group(0) for match in matches]:
        raise TopologyAugmentationError("motif fragment contains malformed anchor text")
    anchor_ids: list[int] = []
    for match in matches:
        decimal = match.group(1)
        if len(decimal) > 1 and decimal.startswith("0"):
            raise TopologyAugmentationError("anchor IDs must use canonical decimal")
        anchor_ids.append(int(decimal))
    if len(anchor_ids) != len(set(anchor_ids)):
        raise TopologyAugmentationError("one motif cannot contain the same anchor twice")
    core = ANCHOR_RE.sub("", fragment)
    if not core or "<" in core or ">" in core:
        raise TopologyAugmentationError("anchor removal did not leave one closed motif core")
    return f"[{core}]", tuple(anchor_ids)


def _normalize_bond_type(value: object) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in ALLOWED_BOND_TYPES - {"other"} else "other"


def _endpoint(raw: tuple[int, int, int, int]) -> dict[str, int]:
    return {
        "logical_motif_index": raw[0],
        "slot_ordinal": raw[1],
        "model_atom_index": raw[2],
        "source_atom_index": raw[3],
    }


def build_topology_augmentation(
    *,
    linearization_result: Any,
    member_id: str,
    base_record_content_sha256: str,
    linearizer_spec_sha256: str,
    expected_motif_atom_indices: Sequence[Sequence[int]],
    expected_motif_lexeme_sha256: Sequence[str],
    source_atom_count: int,
    model_to_source_atom_index: Sequence[int],
) -> dict[str, object]:
    """Build and self-validate one augmentation bound to production-v2 data."""

    if not isinstance(member_id, str) or not member_id.strip():
        raise TopologyAugmentationError("member_id must be nonempty")
    _require_sha256(base_record_content_sha256, "base_record_content_sha256")
    _require_sha256(linearizer_spec_sha256, "linearizer_spec_sha256")
    fragments = tuple(linearization_result.fragment_sequence)
    groups = tuple(tuple(int(atom) for atom in row) for row in linearization_result.motif_atom_groups)
    expected_groups = tuple(tuple(int(atom) for atom in row) for row in expected_motif_atom_indices)
    digests = tuple(_sha256_bytes(fragment.encode("utf-8")) for fragment in fragments)
    if groups != expected_groups:
        raise TopologyAugmentationError("rerun motif groups differ from production-v2")
    if digests != tuple(expected_motif_lexeme_sha256):
        raise TopologyAugmentationError("rerun motif lexeme digests differ from production-v2")
    motif_count = len(groups)
    if motif_count == 0 or len(fragments) != motif_count:
        raise TopologyAugmentationError("linearization has no aligned motif sequence")

    metadata = linearization_result.metadata
    canonical_groups = tuple(tuple(int(atom) for atom in row) for row in metadata.canonical_motif_atom_groups)
    logical_to_canonical = tuple(int(value) for value in metadata.fragment_motif_ids)
    if sorted(logical_to_canonical) != list(range(motif_count)):
        raise TopologyAugmentationError("fragment_motif_ids must be a full permutation")
    canonical_to_logical = [-1] * motif_count
    for logical_id, canonical_id in enumerate(logical_to_canonical):
        canonical_to_logical[canonical_id] = logical_id
        if canonical_groups[canonical_id] != groups[logical_id]:
            raise TopologyAugmentationError("canonical/logical remap changed motif membership")

    if isinstance(source_atom_count, bool) or not isinstance(source_atom_count, int) or source_atom_count <= 0:
        raise TopologyAugmentationError("source_atom_count must be a positive integer")
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in model_to_source_atom_index):
        raise TopologyAugmentationError("model/source mapping must contain integers")
    source_map = tuple(int(value) for value in model_to_source_atom_index)
    model_count = int(metadata.input_atom_count)
    if len(source_map) != model_count:
        raise TopologyAugmentationError("model/source mapping length mismatch")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < source_atom_count for value in source_map):
        raise TopologyAugmentationError("model/source atom mapping is out of range")
    if tuple(sorted(set(source_map))) != source_map:
        raise TopologyAugmentationError("model/source mapping must be strictly increasing")

    anchors = {int(anchor.anchor_id): anchor for anchor in metadata.cross_motif_bonds}
    if len(anchors) != len(tuple(metadata.cross_motif_bonds)) or sorted(anchors) != list(range(len(anchors))):
        raise TopologyAugmentationError("source anchor IDs must be unique and dense")

    pure_tokens: list[str] = []
    slot_anchor_ids: list[list[int]] = []
    slot_atoms: list[list[int]] = []
    slot_source_atoms: list[list[int]] = []
    occurrences: dict[int, list[tuple[int, int, int, int]]] = {}
    for logical_id, fragment in enumerate(fragments):
        pure, anchor_ids = _project_fragment(fragment)
        pure_tokens.append(pure)
        slot_anchor_ids.append(list(anchor_ids))
        canonical_id = logical_to_canonical[logical_id]
        model_row: list[int] = []
        source_row: list[int] = []
        for slot, anchor_id in enumerate(anchor_ids):
            anchor = anchors.get(anchor_id)
            if anchor is None:
                raise TopologyAugmentationError("fragment references an unknown anchor")
            if int(anchor.motif_a) == canonical_id:
                atom = int(anchor.atom_a)
            elif int(anchor.motif_b) == canonical_id:
                atom = int(anchor.atom_b)
            else:
                raise TopologyAugmentationError("anchor occurrence is attached to the wrong motif")
            if atom not in groups[logical_id]:
                raise TopologyAugmentationError("anchor endpoint is outside its motif")
            source_atom = source_map[atom]
            model_row.append(atom)
            source_row.append(source_atom)
            occurrences.setdefault(anchor_id, []).append((logical_id, slot, atom, source_atom))
        slot_atoms.append(model_row)
        slot_source_atoms.append(source_row)

    edge_rows = []
    for anchor_id in sorted(anchors):
        anchor = anchors[anchor_id]
        endpoints = sorted(occurrences.get(anchor_id, []))
        if len(endpoints) != 2 or endpoints[0][0] == endpoints[1][0]:
            raise TopologyAugmentationError("every anchor must occur once in each of two motifs")
        expected = {
            (int(anchor.motif_a), int(anchor.atom_a)),
            (int(anchor.motif_b), int(anchor.atom_b)),
        }
        observed = {(logical_to_canonical[row[0]], row[2]) for row in endpoints}
        if expected != observed:
            raise TopologyAugmentationError("fragment anchors disagree with edge metadata")
        edge_rows.append(
            (endpoints[0], endpoints[1], _normalize_bond_type(anchor.bond_type), str(anchor.bond_type), anchor_id)
        )
    edge_rows.sort(key=lambda row: (row[0], row[1], row[2], row[4]))
    bonds = [
        {
            "edge_id": edge_id,
            "source_anchor_id": row[4],
            "left": _endpoint(row[0]),
            "right": _endpoint(row[1]),
            "bond_type": row[2],
            "source_bond_type": row[3],
        }
        for edge_id, row in enumerate(edge_rows)
    ]
    attachment_atoms = {atom for row in slot_atoms for atom in row}
    document = {
        "schema_version": SCHEMA_VERSION,
        "document_kind": DOCUMENT_KIND,
        "training_admission": False,
        "member": {
            "member_id": member_id,
            "base_record_content_sha256": base_record_content_sha256,
        },
        "provenance": {
            "linearizer_schema_version": str(metadata.schema_version),
            "linearizer_spec_sha256": linearizer_spec_sha256,
            "projection_policy": PROJECTION_POLICY,
            "geometry_or_e3fp_recomputed": False,
        },
        "atom_universe": {
            "model_atom_count": model_count,
            "source_atom_count": source_atom_count,
            "model_to_source_atom_index": list(source_map),
            "atom_is_attachment": [atom in attachment_atoms for atom in range(model_count)],
        },
        "logical_motif_domain": {
            "logical_motif_count": motif_count,
            "logical_to_canonical_motif_index": list(logical_to_canonical),
            "canonical_to_logical_motif_index": canonical_to_logical,
            "component_logical_motif_ranges": [[int(a), int(b)] for a, b in metadata.component_fragment_ranges],
            "motif_atom_indices": [list(row) for row in groups],
            "exact_motif_lexeme_sha256": list(digests),
            "pure_motif_token": pure_tokens,
            "pure_motif_token_sha256": [_sha256_bytes(token.encode("utf-8")) for token in pure_tokens],
            "motif_slot_anchor_ids": slot_anchor_ids,
            "motif_slot_atom_indices": slot_atoms,
            "motif_slot_source_atom_indices": slot_source_atoms,
            "cross_motif_bonds": bonds,
        },
    }
    validate_topology_augmentation(document)
    return document


def validate_topology_augmentation(document: object) -> None:
    """Fail closed on cross-domain mutations of a derived augmentation."""

    top = {
        "schema_version", "document_kind", "training_admission", "member",
        "provenance", "atom_universe", "logical_motif_domain",
    }
    if not isinstance(document, dict) or set(document) != top:
        raise TopologyAugmentationError("augmentation top-level fields are not closed")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("document_kind") != DOCUMENT_KIND:
        raise TopologyAugmentationError("augmentation schema/kind mismatch")
    if document.get("training_admission") is not False:
        raise TopologyAugmentationError("augmentation cannot admit training")
    member = document.get("member")
    if not isinstance(member, dict) or set(member) != {"member_id", "base_record_content_sha256"}:
        raise TopologyAugmentationError("member binding fields are not closed")
    if not isinstance(member.get("member_id"), str) or not member["member_id"]:
        raise TopologyAugmentationError("member_id must be nonempty")
    _require_sha256(member.get("base_record_content_sha256"), "base_record_content_sha256")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "linearizer_schema_version", "linearizer_spec_sha256", "projection_policy",
        "geometry_or_e3fp_recomputed",
    }:
        raise TopologyAugmentationError("provenance fields are not closed")
    if provenance.get("projection_policy") != PROJECTION_POLICY or provenance.get("geometry_or_e3fp_recomputed") is not False:
        raise TopologyAugmentationError("projection policy or no-recompute guarantee changed")
    _require_sha256(provenance.get("linearizer_spec_sha256"), "linearizer_spec_sha256")

    atoms = document.get("atom_universe")
    motifs = document.get("logical_motif_domain")
    if not isinstance(atoms, dict) or set(atoms) != {
        "model_atom_count", "source_atom_count", "model_to_source_atom_index", "atom_is_attachment",
    }:
        raise TopologyAugmentationError("atom fields are not closed")
    motif_fields = {
        "logical_motif_count", "logical_to_canonical_motif_index",
        "canonical_to_logical_motif_index", "component_logical_motif_ranges",
        "motif_atom_indices", "exact_motif_lexeme_sha256", "pure_motif_token",
        "pure_motif_token_sha256", "motif_slot_anchor_ids",
        "motif_slot_atom_indices", "motif_slot_source_atom_indices", "cross_motif_bonds",
    }
    if not isinstance(motifs, dict) or set(motifs) != motif_fields:
        raise TopologyAugmentationError("motif fields are not closed")
    model_count, source_count, motif_count = (
        atoms.get("model_atom_count"), atoms.get("source_atom_count"), motifs.get("logical_motif_count")
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (model_count, source_count, motif_count)):
        raise TopologyAugmentationError("domain counts must be positive integers")
    source_map = atoms.get("model_to_source_atom_index")
    attachment = atoms.get("atom_is_attachment")
    if not isinstance(source_map, list) or len(source_map) != model_count or source_map != sorted(set(source_map)):
        raise TopologyAugmentationError("model/source mapping is not strictly ordered")
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < source_count for value in source_map):
        raise TopologyAugmentationError("model/source mapping is out of range")
    if not isinstance(attachment, list) or len(attachment) != model_count or any(not isinstance(value, bool) for value in attachment):
        raise TopologyAugmentationError("attachment mask shape/type mismatch")

    parallel = (
        "logical_to_canonical_motif_index", "canonical_to_logical_motif_index",
        "motif_atom_indices", "exact_motif_lexeme_sha256", "pure_motif_token",
        "pure_motif_token_sha256", "motif_slot_anchor_ids", "motif_slot_atom_indices",
        "motif_slot_source_atom_indices",
    )
    if any(not isinstance(motifs.get(key), list) or len(motifs[key]) != motif_count for key in parallel):
        raise TopologyAugmentationError("logical motif arrays are misaligned")
    forward, inverse = motifs[parallel[0]], motifs[parallel[1]]
    if sorted(forward) != list(range(motif_count)) or sorted(inverse) != list(range(motif_count)):
        raise TopologyAugmentationError("motif remaps must be permutations")
    if any(inverse[canonical] != logical for logical, canonical in enumerate(forward)):
        raise TopologyAugmentationError("motif remaps are not inverses")

    atom_owner = [-1] * model_count
    all_slots: set[tuple[int, int]] = set()
    all_anchor_ids: set[int] = set()
    declared_attachment: set[int] = set()
    for motif_id in range(motif_count):
        group = motifs["motif_atom_indices"][motif_id]
        if not isinstance(group, list) or not group or group != sorted(set(group)):
            raise TopologyAugmentationError("motif group must be nonempty/sorted/unique")
        for atom in group:
            if not isinstance(atom, int) or isinstance(atom, bool) or not 0 <= atom < model_count or atom_owner[atom] != -1:
                raise TopologyAugmentationError("motif groups are not an atom partition")
            atom_owner[atom] = motif_id
        _require_sha256(motifs["exact_motif_lexeme_sha256"][motif_id], "motif lexeme digest")
        pure = motifs["pure_motif_token"][motif_id]
        if not isinstance(pure, str) or _sha256_bytes(pure.encode("utf-8")) != motifs["pure_motif_token_sha256"][motif_id]:
            raise TopologyAugmentationError("pure motif digest mismatch")
        anchors = motifs["motif_slot_anchor_ids"][motif_id]
        slots = motifs["motif_slot_atom_indices"][motif_id]
        source_slots = motifs["motif_slot_source_atom_indices"][motif_id]
        if not all(isinstance(row, list) for row in (anchors, slots, source_slots)) or not (len(anchors) == len(slots) == len(source_slots)):
            raise TopologyAugmentationError("motif slot arrays are misaligned")
        if len(anchors) != len(set(anchors)):
            raise TopologyAugmentationError("one motif repeats an anchor")
        for slot, (anchor, atom, source_atom) in enumerate(zip(anchors, slots, source_slots)):
            if not isinstance(anchor, int) or isinstance(anchor, bool) or anchor < 0 or atom not in group or source_atom != source_map[atom]:
                raise TopologyAugmentationError("slot/anchor/atom binding is invalid")
            all_slots.add((motif_id, slot))
            all_anchor_ids.add(anchor)
            declared_attachment.add(atom)
    if -1 in atom_owner:
        raise TopologyAugmentationError("motif groups leave atoms uncovered")

    ranges = motifs["component_logical_motif_ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise TopologyAugmentationError("component ranges must be nonempty")
    cursor = 0
    for row in ranges:
        if not isinstance(row, list) or len(row) != 2 or row[0] != cursor or not isinstance(row[1], int) or row[1] <= row[0]:
            raise TopologyAugmentationError("component ranges are not a contiguous partition")
        cursor = row[1]
    if cursor != motif_count:
        raise TopologyAugmentationError("component ranges do not cover motifs")

    bonds = motifs["cross_motif_bonds"]
    if not isinstance(bonds, list):
        raise TopologyAugmentationError("cross_motif_bonds must be an array")
    used_slots: set[tuple[int, int]] = set()
    edge_atoms: set[int] = set()
    source_anchors: list[int] = []
    previous_key = None
    for edge_id, bond in enumerate(bonds):
        if not isinstance(bond, dict) or set(bond) != {"edge_id", "source_anchor_id", "left", "right", "bond_type", "source_bond_type"}:
            raise TopologyAugmentationError("edge fields are not closed")
        if bond.get("edge_id") != edge_id or bond.get("bond_type") not in ALLOWED_BOND_TYPES:
            raise TopologyAugmentationError("edge ID/bond type is invalid")
        source_anchor = bond.get("source_anchor_id")
        if not isinstance(source_anchor, int) or isinstance(source_anchor, bool) or source_anchor < 0:
            raise TopologyAugmentationError("source anchor is invalid")
        source_anchors.append(source_anchor)
        endpoint_rows = []
        for side in ("left", "right"):
            endpoint = bond.get(side)
            required = {"logical_motif_index", "slot_ordinal", "model_atom_index", "source_atom_index"}
            if not isinstance(endpoint, dict) or set(endpoint) != required:
                raise TopologyAugmentationError("endpoint fields are not closed")
            logical = endpoint["logical_motif_index"]
            slot = endpoint["slot_ordinal"]
            atom = endpoint["model_atom_index"]
            source_atom = endpoint["source_atom_index"]
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (logical, slot, atom, source_atom)):
                raise TopologyAugmentationError("endpoint values must be integers")
            if not 0 <= logical < motif_count or not 0 <= slot < len(motifs["motif_slot_atom_indices"][logical]):
                raise TopologyAugmentationError("endpoint motif/slot is out of range")
            if atom != motifs["motif_slot_atom_indices"][logical][slot] or source_atom != motifs["motif_slot_source_atom_indices"][logical][slot]:
                raise TopologyAugmentationError("endpoint does not match its slot")
            if motifs["motif_slot_anchor_ids"][logical][slot] != source_anchor:
                raise TopologyAugmentationError("endpoint does not match its source anchor")
            used_slots.add((logical, slot))
            edge_atoms.add(atom)
            endpoint_rows.append((logical, slot, atom, source_atom))
        if endpoint_rows[0] >= endpoint_rows[1] or endpoint_rows[0][0] == endpoint_rows[1][0]:
            raise TopologyAugmentationError("edge endpoints are not canonical cross-motif endpoints")
        key = (endpoint_rows[0], endpoint_rows[1], bond["bond_type"], source_anchor)
        if previous_key is not None and key < previous_key:
            raise TopologyAugmentationError("edges are not canonically ordered")
        previous_key = key
    if used_slots != all_slots:
        raise TopologyAugmentationError("every motif slot must occur in exactly one edge")
    if sorted(source_anchors) != list(range(len(bonds))) or set(source_anchors) != all_anchor_ids:
        raise TopologyAugmentationError("source anchors are not paired densely")
    if attachment != [atom in edge_atoms for atom in range(model_count)] or edge_atoms != declared_attachment:
        raise TopologyAugmentationError("attachment mask is not derived from endpoints")


def augmentation_sha256(document: object) -> str:
    """Return the canonical content digest after complete validation."""

    validate_topology_augmentation(document)
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)
