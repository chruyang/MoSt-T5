"""Lossless SMILES-vocabulary fallback for the fragSMILES candidate language.

The ordinary path remains the compact fragSMILES stereo codec.  This module
adds a universal path for molecules outside that codec's support domain.  The
fallback serializes RDKit's canonical stereo-free SMILES with the same pinned
Smirk-derived molecular glyph vocabulary used by fragment misses, records
every atom token span, and binds heavy atoms
back to the projected source/E3FP row.  It therefore never needs ``<unk>`` and
does not weaken the frozen two-dimensional identity contract to admit an
exceptional molecule.  R/S, E/Z and conformational state remain exclusively
in the aligned E3FP sidecar, exactly as on the ordinary motif path.

The fallback is a routing mode in release metadata, not a separate learned
molecule class or byte namespace.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from rdkit import Chem

from most_t5_next.p1.audit_fragsmiles_adoption_v1 import (
    FragSmilesAuditError,
    encode_with_sidecar,
)
from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import (
    CompactStereoCodecError,
    CompactStereoSurface,
    _is_fragment_token,
    strict_round_trip,
)
from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
    SmirkSmilesVocabularyError,
    decode_smiles_glyphs,
    encode_smiles_glyphs,
    require_stereo_free_fragment,
    smiles_glyph_token_map,
)


SCHEMA_VERSION = "most-t5-next/fragsmiles-lossless-fallback/v3"
_SOURCE_INDEX_PROP = "most_fallback_source_atom_index"
_PROJECTED_INDEX_PROP = "most_fallback_projected_atom_index"
_E3FP_ROW_PROP = "most_fallback_e3fp_row"


class LosslessFallbackError(ValueError):
    """The universal fallback or its atom-address sidecar is malformed."""


@dataclass(frozen=True)
class FallbackAtomAddress:
    smiles_atom_ordinal: int
    token_start: int
    token_stop: int
    projected_atom_index: int
    source_atom_index: int
    e3fp_row: Optional[int]

    def __post_init__(self) -> None:
        if (
            self.smiles_atom_ordinal < 0
            or self.token_start < 0
            or self.token_stop <= self.token_start
            or self.projected_atom_index < 0
            or self.source_atom_index < 0
        ):
            raise LosslessFallbackError("invalid fallback atom address")
        if self.e3fp_row is not None and self.e3fp_row < 0:
            raise LosslessFallbackError("invalid E3FP row")


@dataclass(frozen=True)
class CompactAtomAddress:
    fragment_index: int
    fragment_local_atom_index: int
    carrier_token_index: int
    e3fp_row: int


@dataclass(frozen=True)
class LosslessFallbackSurface:
    schema_version: str
    canonical_stereo_free_smiles: str
    tokens: tuple[str, ...]
    roles: tuple[str, ...]
    atom_addresses: tuple[FallbackAtomAddress, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LosslessFallbackError("unexpected fallback schema")
        if not self.tokens:
            raise LosslessFallbackError("fallback molecule surface is empty")
        if len(self.tokens) != len(self.roles):
            raise LosslessFallbackError("fallback token/role arrays disagree")
        if decode_fallback_smiles(self.tokens) != self.canonical_stereo_free_smiles:
            raise LosslessFallbackError("fallback surface is not an exact fixed point")


@dataclass(frozen=True)
class RoutedFragSmilesSurface:
    schema_version: str
    mode: str
    tokens: tuple[str, ...]
    compact_surface: Optional[CompactStereoSurface]
    fallback_surface: Optional[LosslessFallbackSurface]
    compact_atom_addresses: tuple[CompactAtomAddress, ...]
    fallback_reason_type: Optional[str]
    fallback_reason: Optional[str]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LosslessFallbackError("unexpected routed surface schema")
        if self.mode == "compact":
            if self.compact_surface is None or self.fallback_surface is not None:
                raise LosslessFallbackError("compact routed surface is inconsistent")
            if self.tokens != self.compact_surface.tokens:
                raise LosslessFallbackError("compact routed tokens drifted")
            if self.fallback_reason_type is not None or self.fallback_reason is not None:
                raise LosslessFallbackError("compact surface carries a fallback reason")
        elif self.mode == "whole_molecule_fallback":
            if self.compact_surface is not None or self.fallback_surface is None:
                raise LosslessFallbackError("fallback routed surface is inconsistent")
            if self.tokens != self.fallback_surface.tokens:
                raise LosslessFallbackError("fallback routed tokens drifted")
            if not self.fallback_reason_type or not self.fallback_reason:
                raise LosslessFallbackError("fallback reason is missing")
            if self.compact_atom_addresses:
                raise LosslessFallbackError("fallback surface has compact atom addresses")
        else:
            raise LosslessFallbackError("unknown routed surface mode")


def fallback_token_universe() -> tuple[str, ...]:
    """Return the shared collision-free SMILES token surfaces."""

    return tuple(surface for _glyph, surface in smiles_glyph_token_map())


def decode_fallback_smiles(tokens: Sequence[str]) -> str:
    if not tokens:
        raise LosslessFallbackError("fallback molecule payload is empty")
    try:
        surface_to_glyph = {surface: glyph for glyph, surface in smiles_glyph_token_map()}
        return decode_smiles_glyphs(tuple(surface_to_glyph[token] for token in tokens))
    except (KeyError, SmirkSmilesVocabularyError) as exc:
        raise LosslessFallbackError("fallback payload contains an unknown SMILES token") from exc


def _smiles_atom_spans(smiles: str) -> tuple[tuple[int, int], ...]:
    """Return atom-lexeme spans in SMILES encounter order.

    RDKit canonical SMILES is ASCII.  Bracket atoms are one atom regardless of
    their isotope/chirality/charge fields; unbracketed atoms use the organic
    subset and ``*``.  Ring digits, bond symbols and connector punctuation are
    deliberately not atoms.
    """

    if not smiles or not smiles.isascii():
        raise LosslessFallbackError("canonical RDKit SMILES must be nonempty ASCII")
    spans = []
    index = 0
    while index < len(smiles):
        character = smiles[index]
        if character == "[":
            stop = smiles.find("]", index + 1)
            if stop < 0 or "[" in smiles[index + 1 : stop]:
                raise LosslessFallbackError("canonical SMILES has a malformed bracket atom")
            spans.append((index, stop + 1))
            index = stop + 1
            continue
        if smiles.startswith("Cl", index) or smiles.startswith("Br", index):
            spans.append((index, index + 2))
            index += 2
            continue
        if character in "BCNOPSFIbcnops*":
            spans.append((index, index + 1))
        index += 1
    return tuple(spans)


def _project_with_source_indices(source_mol: Chem.Mol) -> Chem.Mol:
    indexed = Chem.Mol(source_mol)
    for atom in indexed.GetAtoms():
        atom.SetIntProp(_SOURCE_INDEX_PROP, atom.GetIdx())
    # Formal Phase-I callers already pass the frozen hydrogen projection.  Do
    # not perform a second, stereo-dependent RemoveHs here: a raw compatibility
    # caller may contain explicit lexical hydrogens, and whether RDKit retains
    # a stereo-defining H would otherwise change after stereochemistry is
    # intentionally removed.  Such H atoms remain addressable but receive no
    # E3FP row.
    projected = Chem.Mol(indexed)
    if projected.GetNumAtoms() == 0:
        raise LosslessFallbackError("fallback source has no serializable atoms")
    e3fp_row = 0
    for atom in projected.GetAtoms():
        atom.SetIntProp(_PROJECTED_INDEX_PROP, atom.GetIdx())
        if atom.GetAtomicNum() != 1:
            atom.SetIntProp(_E3FP_ROW_PROP, e3fp_row)
            e3fp_row += 1
    # The model language is intentionally stereo-free.  Preserve isotope,
    # charge, radical, aromaticity, bond order and component identity, while
    # leaving discrete stereochemistry and conformational state to the aligned
    # E3FP sidecar rather than duplicating it in the fallback token stream.
    Chem.RemoveStereochemistry(projected)
    return projected


def _read_output_order(mol: Chem.Mol) -> tuple[int, ...]:
    try:
        return tuple(
            int(value)
            for value in ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder"))
        )
    except (KeyError, SyntaxError, TypeError, ValueError) as exc:
        raise LosslessFallbackError("RDKit canonical atom-output order is invalid") from exc


def _parse_smiles_preserving_hydrogens(smiles: str) -> Optional[Chem.Mol]:
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    return Chem.MolFromSmiles(smiles, parameters)


def _canonical_cycle_member(
    projected: Chem.Mol,
) -> tuple[str, Chem.Mol, tuple[int, ...]]:
    """Choose a deterministic member of RDKit's SMILES serialization cycle.

    A few directional ring systems alternate between two equivalent slash
    surfaces.  Selecting the minimum *cycle member* makes encoding invariant to
    which member entered the orbit.  Source/projected row properties are
    propagated through every parse boundary using the emitted atom order.
    """

    current = Chem.Mol(projected)
    seen_at: dict[str, int] = {}
    orbit: list[tuple[str, Chem.Mol, tuple[int, ...]]] = []
    for _ in range(16):
        smiles = Chem.MolToSmiles(current, canonical=True, isomericSmiles=True)
        order = _read_output_order(current)
        if len(order) != current.GetNumAtoms():
            raise LosslessFallbackError("RDKit atom-output order has wrong length")
        cycle_start = seen_at.get(smiles)
        if cycle_start is not None:
            return min(orbit[cycle_start:], key=lambda item: item[0])
        seen_at[smiles] = len(orbit)
        orbit.append((smiles, Chem.Mol(current), order))

        parsed = _parse_smiles_preserving_hydrogens(smiles)
        if parsed is None or parsed.GetNumAtoms() != len(order):
            raise LosslessFallbackError("canonical SMILES cannot be reparsed")
        for parsed_index, current_index in enumerate(order):
            source_atom = current.GetAtomWithIdx(current_index)
            target_atom = parsed.GetAtomWithIdx(parsed_index)
            if not (
                source_atom.HasProp(_SOURCE_INDEX_PROP)
                and source_atom.HasProp(_PROJECTED_INDEX_PROP)
            ):
                raise LosslessFallbackError("fallback atom mapping was lost in orbit")
            target_atom.SetIntProp(
                _SOURCE_INDEX_PROP, source_atom.GetIntProp(_SOURCE_INDEX_PROP)
            )
            target_atom.SetIntProp(
                _PROJECTED_INDEX_PROP, source_atom.GetIntProp(_PROJECTED_INDEX_PROP)
            )
            if source_atom.HasProp(_E3FP_ROW_PROP):
                target_atom.SetIntProp(
                    _E3FP_ROW_PROP, source_atom.GetIntProp(_E3FP_ROW_PROP)
                )
        current = parsed
    raise LosslessFallbackError("fallback canonicalization orbit did not close")


def _encode_lossless_fallback_once(source_mol: Chem.Mol) -> LosslessFallbackSurface:
    projected = _project_with_source_indices(source_mol)
    canonical, selected_mol, output_order = _canonical_cycle_member(projected)
    spans = _smiles_atom_spans(canonical)
    if len(spans) != len(output_order) or len(output_order) != selected_mol.GetNumAtoms():
        raise LosslessFallbackError("fallback atom lexemes do not cover the molecule")

    glyph_encoding = encode_smiles_glyphs(canonical)
    glyph_to_surface = dict(smiles_glyph_token_map())
    tokens = tuple(glyph_to_surface[glyph] for glyph in glyph_encoding.glyphs)
    glyph_roles = ["syntax_glyph"] * len(tokens)
    addresses = []
    for ordinal, ((start, stop), projected_index) in enumerate(zip(spans, output_order)):
        if projected_index < 0 or projected_index >= selected_mol.GetNumAtoms():
            raise LosslessFallbackError("fallback atom-output index is out of range")
        glyph_indices = tuple(
            glyph_index
            for glyph_index, (glyph_start, glyph_stop) in enumerate(
                glyph_encoding.character_spans
            )
            if glyph_start >= start and glyph_stop <= stop
        )
        if not glyph_indices or glyph_indices != tuple(
            range(glyph_indices[0], glyph_indices[-1] + 1)
        ):
            raise LosslessFallbackError("atom SMILES span does not map to contiguous glyphs")
        for glyph_index in glyph_indices:
            glyph_roles[glyph_index] = "atom_glyph"
        atom = selected_mol.GetAtomWithIdx(projected_index)
        if not (
            atom.HasProp(_SOURCE_INDEX_PROP) and atom.HasProp(_PROJECTED_INDEX_PROP)
        ):
            raise LosslessFallbackError("fallback source atom mapping was lost")
        original_projected_index = atom.GetIntProp(_PROJECTED_INDEX_PROP)
        addresses.append(
            FallbackAtomAddress(
                smiles_atom_ordinal=ordinal,
                token_start=glyph_indices[0],
                token_stop=glyph_indices[-1] + 1,
                projected_atom_index=original_projected_index,
                source_atom_index=atom.GetIntProp(_SOURCE_INDEX_PROP),
                e3fp_row=(
                    atom.GetIntProp(_E3FP_ROW_PROP)
                    if atom.HasProp(_E3FP_ROW_PROP)
                    else None
                ),
            )
        )
    surface = LosslessFallbackSurface(
        schema_version=SCHEMA_VERSION,
        canonical_stereo_free_smiles=canonical,
        tokens=tokens,
        roles=tuple(glyph_roles),
        atom_addresses=tuple(addresses),
    )
    return surface


def encode_lossless_fallback(source_mol: Chem.Mol) -> LosslessFallbackSurface:
    surface = _encode_lossless_fallback_once(source_mol)
    restored = decode_lossless_fallback_mol(surface.tokens)
    replay_smiles, _, _ = _canonical_cycle_member(
        _project_with_source_indices(restored)
    )
    if replay_smiles != surface.canonical_stereo_free_smiles:
        raise LosslessFallbackError("fallback stereo-free round trip failed")
    replay = _encode_lossless_fallback_once(restored)
    if replay.tokens != surface.tokens:
        raise LosslessFallbackError("fallback decode/re-encode fixed point failed")
    return surface


def decode_lossless_fallback_mol(tokens: Sequence[str]) -> Chem.Mol:
    smiles = decode_fallback_smiles(tokens)
    mol = _parse_smiles_preserving_hydrogens(smiles)
    if mol is None:
        raise LosslessFallbackError("fallback payload is not a valid molecule")
    canonical, selected, _ = _canonical_cycle_member(
        _project_with_source_indices(mol)
    )
    if canonical != smiles:
        raise LosslessFallbackError("fallback payload is not canonical")
    # Return the exact orbit member whose serialization is the accepted
    # canonical payload.  Returning the first parse boundary instead is wrong
    # for directional ring systems whose RDKit SMILES traversal alternates:
    # callers would immediately re-enter the orbit from a different member and
    # a valid surface could fail its own strict replay gate.
    return selected


def _compact_atom_addresses(
    source_mol: Chem.Mol,
    compact_surface: CompactStereoSurface,
    *,
    chemicalgof_root: Path,
) -> tuple[CompactAtomAddress, ...]:
    direct = encode_with_sidecar(
        source_mol,
        chemicalgof_root=chemicalgof_root,
        stereo_policy="connectivity_only",
    )
    if direct.tokens != compact_surface.connectivity_record.tokens:
        raise LosslessFallbackError("direct and compact connectivity surfaces disagree")
    carrier_positions = []
    for token_index, token in enumerate(compact_surface.tokens):
        if _is_fragment_token(token):
            carrier_positions.append(token_index)
    if len(carrier_positions) != len(direct.fragments):
        raise LosslessFallbackError("compact fragment carrier count drifted")
    rows = []
    seen_e3fp = set()
    for fragment, carrier in zip(direct.fragments, carrier_positions):
        for local_index, e3fp_row in enumerate(fragment.source_atom_indices):
            if e3fp_row in seen_e3fp:
                raise LosslessFallbackError("compact E3FP row occurs in multiple fragments")
            seen_e3fp.add(e3fp_row)
            rows.append(
                CompactAtomAddress(
                    fragment_index=fragment.sequence_fragment_index,
                    fragment_local_atom_index=local_index,
                    carrier_token_index=carrier,
                    e3fp_row=e3fp_row,
                )
            )
    expected_rows = Chem.RemoveHs(Chem.Mol(source_mol), sanitize=True).GetNumAtoms()
    if seen_e3fp != set(range(expected_rows)):
        raise LosslessFallbackError("compact E3FP rows do not partition the projection")
    return tuple(rows)


def encode_main_or_fallback(
    source_mol: Chem.Mol, *, chemicalgof_root: Path
) -> RoutedFragSmilesSurface:
    """Use compact fragSMILES when strict; otherwise use universal fallback."""

    try:
        compact = strict_round_trip(source_mol, chemicalgof_root=chemicalgof_root)
        # A strict chemistry codec is not yet a usable model surface unless
        # every fragment is representable by the frozen finite lexer.  Macro
        # selection happens later and cannot be relied on for open-world
        # coverage, so fail over at the whole-record boundary here.
        for fragment in compact.connectivity_record.fragments:
            require_stereo_free_fragment(fragment.fragment_smiles)
        addresses = _compact_atom_addresses(
            source_mol, compact, chemicalgof_root=chemicalgof_root
        )
        return RoutedFragSmilesSurface(
            schema_version=SCHEMA_VERSION,
            mode="compact",
            tokens=compact.tokens,
            compact_surface=compact,
            fallback_surface=None,
            compact_atom_addresses=addresses,
            fallback_reason_type=None,
            fallback_reason=None,
        )
    except (
        CompactStereoCodecError,
        FragSmilesAuditError,
        LosslessFallbackError,
        SmirkSmilesVocabularyError,
        KeyError,
        IndexError,
        RuntimeError,
        TimeoutError,
    ) as exc:
        fallback = encode_lossless_fallback(source_mol)
        return RoutedFragSmilesSurface(
            schema_version=SCHEMA_VERSION,
            mode="whole_molecule_fallback",
            tokens=fallback.tokens,
            compact_surface=None,
            fallback_surface=fallback,
            compact_atom_addresses=(),
            fallback_reason_type=type(exc).__name__,
            fallback_reason=str(exc),
        )


__all__ = [
    "CompactAtomAddress",
    "FallbackAtomAddress",
    "LosslessFallbackError",
    "LosslessFallbackSurface",
    "RoutedFragSmilesSurface",
    "SCHEMA_VERSION",
    "decode_fallback_smiles",
    "decode_lossless_fallback_mol",
    "encode_lossless_fallback",
    "encode_main_or_fallback",
    "fallback_token_universe",
]
