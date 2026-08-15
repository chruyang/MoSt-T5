"""Macro plus Smirk-glyph model surface for compact fragSMILES.

Official fragSMILES already tokenizes connector ``<n>`` records separately
from fragment SMILES.  A registered fragment is therefore one macro token; an
unregistered fragment is the pinned Smirk-derived glyph expansion followed by
one suffix.  No raw fragment string or connector number becomes an open
tokenizer entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import (
    CompactStereoSurface,
    _is_fragment_token,
    decode_compact_stereo_surface,
    strict_round_trip,
)
from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import (
    CompactAtomAddress,
    LosslessFallbackError,
    _compact_atom_addresses,
)
from most_t5_next.r1.tokenizer.smirk_smiles_vocabulary_v1 import (
    decode_smiles_glyphs,
    require_stereo_free_fragment,
    smiles_glyph_token_map,
)


SCHEMA_VERSION = "most-t5-next/fragsmiles-macro-fallback-surface/v1"
FRAGMENT_FALLBACK_END = "<MOST:FS:FRAG_END>"
CONNECTOR_PREFIX = "<MOST:FS:CONN>"
CONNECTOR_END = "<MOST:FS:CONN_END>"
BRANCH_OPEN = "<MOST:FS:BRANCH_OPEN>"
BRANCH_CLOSE = "<MOST:FS:BRANCH_CLOSE>"
COMPONENT = "<MOST:FS:COMP>"
_CONNECTOR_RE = re.compile(r"^<([0-9]+)>$")


class FragSmilesModelSurfaceError(ValueError):
    pass


@dataclass(frozen=True)
class FragmentPhrase:
    fragment_index: int
    fragment_smiles: str
    token_start: int
    token_stop: int
    carrier_token_index: int
    macro_used: bool


@dataclass(frozen=True)
class ModelAtomAddress:
    fragment_index: int
    fragment_local_atom_index: int
    carrier_token_index: int
    e3fp_row: int


@dataclass(frozen=True)
class FragSmilesModelSurface:
    schema_version: str
    tokens: tuple[str, ...]
    compact_tokens: tuple[str, ...]
    fragment_phrases: tuple[FragmentPhrase, ...]
    atom_addresses: tuple[ModelAtomAddress, ...]
    macro_used: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise FragSmilesModelSurfaceError("unexpected model-surface schema")
        if not self.tokens or not self.compact_tokens or not self.fragment_phrases:
            raise FragSmilesModelSurfaceError("model surface cannot be empty")
        if self.macro_used != tuple(row.macro_used for row in self.fragment_phrases):
            raise FragSmilesModelSurfaceError("macro flags disagree with phrases")


def _macro_maps(
    macro_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], dict[str, str]]:
    identity_to_token = {}
    token_to_identity = {}
    for row in macro_rows:
        identity = row.get("fragment_smiles")
        token = row.get("surface_token")
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(token, str)
            or not token.startswith("<MOST:FM:")
            or identity in identity_to_token
            or token in token_to_identity
        ):
            raise FragSmilesModelSurfaceError("macro registry is malformed")
        identity_to_token[identity] = token
        token_to_identity[token] = identity
    return identity_to_token, token_to_identity


def _smiles_maps() -> tuple[dict[str, str], dict[str, str]]:
    glyph_to_surface = dict(smiles_glyph_token_map())
    surface_to_glyph = {surface: glyph for glyph, surface in glyph_to_surface.items()}
    if len(surface_to_glyph) != len(glyph_to_surface):
        raise FragSmilesModelSurfaceError("SMILES glyph token map is not bijective")
    return glyph_to_surface, surface_to_glyph


def fixed_surface_tokens() -> tuple[str, ...]:
    return (
        FRAGMENT_FALLBACK_END,
        CONNECTOR_PREFIX,
        CONNECTOR_END,
        BRANCH_OPEN,
        BRANCH_CLOSE,
        COMPONENT,
    )


def encode_compact_model_surface(
    source_mol,
    compact_surface: CompactStereoSurface,
    macro_rows: Sequence[Mapping[str, object]],
    *,
    chemicalgof_root: Path,
) -> FragSmilesModelSurface:
    identity_to_macro, _ = _macro_maps(macro_rows)
    glyph_to_surface, _ = _smiles_maps()
    output = []
    phrases = []
    fragment_index = -1
    for compact_token in compact_surface.tokens:
        if _is_fragment_token(compact_token):
            fragment_index += 1
            start = len(output)
            macro = identity_to_macro.get(compact_token)
            if macro is not None:
                output.append(macro)
                used_macro = True
            else:
                lexed = require_stereo_free_fragment(compact_token)
                try:
                    output.extend(glyph_to_surface[glyph] for glyph in lexed.glyphs)
                except KeyError as exc:
                    raise FragSmilesModelSurfaceError(
                        "SMILES front end emitted an unregistered glyph"
                    ) from exc
                output.append(FRAGMENT_FALLBACK_END)
                used_macro = False
            stop = len(output)
            phrases.append(
                FragmentPhrase(
                    fragment_index=fragment_index,
                    fragment_smiles=compact_token,
                    token_start=start,
                    token_stop=stop,
                    carrier_token_index=stop - 1,
                    macro_used=used_macro,
                )
            )
            continue
        connector = _CONNECTOR_RE.fullmatch(compact_token)
        if connector is not None:
            output.append(CONNECTOR_PREFIX)
            output.extend(glyph_to_surface[digit] for digit in connector.group(1))
            output.append(CONNECTOR_END)
        elif compact_token == "(":
            output.append(BRANCH_OPEN)
        elif compact_token == ")":
            output.append(BRANCH_CLOSE)
        elif compact_token == "<COMP>":
            output.append(COMPONENT)
        elif compact_token.startswith("<ST:"):
            output.append(compact_token)
        elif compact_token in set("0123456789"):
            output.append(compact_token)
        else:
            raise FragSmilesModelSurfaceError(
                f"unsupported compact control token: {compact_token}"
            )
    if fragment_index + 1 != len(compact_surface.connectivity_record.fragments):
        raise FragSmilesModelSurfaceError("fragment traversal count drifted")

    compact_addresses = _compact_atom_addresses(
        source_mol, compact_surface, chemicalgof_root=chemicalgof_root
    )
    carrier_by_fragment = {row.fragment_index: row.carrier_token_index for row in phrases}
    atom_addresses = tuple(
        ModelAtomAddress(
            fragment_index=row.fragment_index,
            fragment_local_atom_index=row.fragment_local_atom_index,
            carrier_token_index=carrier_by_fragment[row.fragment_index],
            e3fp_row=row.e3fp_row,
        )
        for row in compact_addresses
    )
    surface = FragSmilesModelSurface(
        schema_version=SCHEMA_VERSION,
        tokens=tuple(output),
        compact_tokens=compact_surface.tokens,
        fragment_phrases=tuple(phrases),
        atom_addresses=atom_addresses,
        macro_used=tuple(row.macro_used for row in phrases),
    )
    if decode_model_tokens(surface.tokens, macro_rows) != compact_surface.tokens:
        raise FragSmilesModelSurfaceError("model surface is not exactly reversible")
    return surface


def decode_model_tokens(
    tokens: Sequence[str], macro_rows: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    _, token_to_identity = _macro_maps(macro_rows)
    _, surface_to_glyph = _smiles_maps()
    output = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in token_to_identity:
            output.append(token_to_identity[token])
            index += 1
            continue
        if token.startswith("<ST:A:"):
            stop = index + 4
            if stop > len(tokens) or any(item not in set("0123456789") for item in tokens[index + 1 : stop]):
                raise FragSmilesModelSurfaceError("atom stereo record has invalid digits")
            output.extend(tokens[index:stop])
            index = stop
            continue
        if token.startswith("<ST:B:"):
            stop = index + 7
            if stop > len(tokens) or any(item not in set("0123456789") for item in tokens[index + 1 : stop]):
                raise FragSmilesModelSurfaceError("bond stereo record has invalid digits")
            output.extend(tokens[index:stop])
            index = stop
            continue
        if token in surface_to_glyph:
            glyphs = []
            while index < len(tokens) and tokens[index] in surface_to_glyph:
                glyphs.append(surface_to_glyph[tokens[index]])
                index += 1
            if index >= len(tokens) or tokens[index] != FRAGMENT_FALLBACK_END:
                raise FragSmilesModelSurfaceError("fragment fallback suffix is missing")
            output.append(decode_smiles_glyphs(glyphs))
            index += 1
            continue
        if token == CONNECTOR_PREFIX:
            index += 1
            digits = []
            while index < len(tokens) and tokens[index] != CONNECTOR_END:
                glyph = surface_to_glyph.get(tokens[index])
                if glyph not in set("0123456789"):
                    raise FragSmilesModelSurfaceError("connector contains a non-digit")
                digits.append(glyph)
                index += 1
            if not digits or index >= len(tokens):
                raise FragSmilesModelSurfaceError("connector terminator is missing")
            if len(digits) > 1 and digits[0] == "0":
                raise FragSmilesModelSurfaceError("connector has a leading zero")
            output.append("<" + "".join(digits) + ">")
            index += 1
            continue
        fixed = {
            BRANCH_OPEN: "(",
            BRANCH_CLOSE: ")",
            COMPONENT: "<COMP>",
        }.get(token)
        if fixed is not None:
            output.append(fixed)
            index += 1
            continue
        raise FragSmilesModelSurfaceError(f"unknown model-surface token: {token}")
    if not output:
        raise FragSmilesModelSurfaceError("cannot decode an empty model surface")
    return tuple(output)


def decode_model_surface_mol(
    tokens: Sequence[str],
    macro_rows: Sequence[Mapping[str, object]],
    *,
    chemicalgof_root: Path,
):
    compact_tokens = decode_model_tokens(tokens, macro_rows)
    try:
        mol = decode_compact_stereo_surface(
            compact_tokens, chemicalgof_root=chemicalgof_root
        )
    except LosslessFallbackError:
        raise
    except Exception as exc:
        raise FragSmilesModelSurfaceError("decoded compact surface is invalid") from exc
    replay_compact = strict_round_trip(mol, chemicalgof_root=chemicalgof_root)
    replay_surface = encode_compact_model_surface(
        mol, replay_compact, macro_rows, chemicalgof_root=chemicalgof_root
    )
    if replay_surface.tokens != tuple(tokens):
        raise FragSmilesModelSurfaceError("model surface is not canonical")
    return mol


__all__ = [
    "BRANCH_CLOSE",
    "BRANCH_OPEN",
    "COMPONENT",
    "CONNECTOR_END",
    "CONNECTOR_PREFIX",
    "FRAGMENT_FALLBACK_END",
    "FragSmilesModelSurface",
    "FragSmilesModelSurfaceError",
    "FragmentPhrase",
    "ModelAtomAddress",
    "SCHEMA_VERSION",
    "decode_model_surface_mol",
    "decode_model_tokens",
    "encode_compact_model_surface",
    "fixed_surface_tokens",
]
