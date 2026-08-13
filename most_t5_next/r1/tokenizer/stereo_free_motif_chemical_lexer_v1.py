"""Finite, reversible lexer for stereo-free pure-motif identities.

This is a tokenizer-independent lexical floor, not a learned vocabulary.  It
keeps common chemical units (element symbols, bonds, branches and ring digits)
intact, represents every deleted-anchor branch ``()`` as one ``<SLOT>`` token,
and decomposes bracket atoms into a finite set of grammar primitives.  Macro or
BPE tokens may later compress this stream, but they are never required for
coverage.

The design deliberately follows the atom-wise SMILES regex used by MolBART /
FineMolTex while tightening its open ``[^]]+`` bracket token into a bounded,
validated stereo-free grammar.  The T5 SentencePiece vocabulary is therefore a
comparison candidate, not the lossless chemical fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
    AnchoredMotifSurfaceError,
    parse_anchor_token,
)


LEXER_SCHEMA_VERSION = "most-t5-next/stereo-free-motif-chemical-lexer/v1"
SLOT_TOKEN = "<SLOT>"

# Frozen periodic-table spellings.  Keeping this table local avoids allowing a
# dependency upgrade to alter lexical boundaries.
ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
# ``si`` and ``te`` are bracket-only aromatic spellings emitted by the pinned
# RDKit producer on the frozen Phase-I/II corpora.  They stay outside the
# unbracketed organic subset, but belong to the finite bracket-atom grammar.
AROMATIC_SYMBOLS = frozenset(
    {"b", "c", "n", "o", "p", "s", "se", "as", "si", "te"}
)
ORGANIC_SUBSET = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
    | {"b", "c", "n", "o", "p", "s"}
)
OUTSIDE_PUNCTUATION = frozenset({"(", ")", ".", "=", "#", "-", "+", ":", "~", "?", ">", "*", "$"})
STEREO_MARKERS = frozenset({"@", "/", "\\"})
_BRACKET_ATOM_RE = re.compile(
    r"^(?P<isotope>[0-9]*)(?P<symbol>[A-Z][a-z]?|se|as|si|te|[bcnops]|\*)"
    r"(?P<hcount>H[0-9]*)?"
    r"(?P<charge>(?:[+]{1,3}|[-]{1,3}|[+][0-9]+|[-][0-9]+))?"
    r"(?P<atom_class>:[0-9]+)?$"
)


class StereoFreeMotifLexerError(ValueError):
    """A pure motif lies outside the frozen stereo-free lexical grammar."""


@dataclass(frozen=True)
class LexedPureMotif:
    schema_version: str
    pure_motif: str
    tokens: tuple[str, ...]
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != LEXER_SCHEMA_VERSION:
            raise StereoFreeMotifLexerError("unexpected lexer schema")
        if not self.tokens or len(self.tokens) != len(self.roles):
            raise StereoFreeMotifLexerError("lexer token and role arrays disagree")
        if any(not isinstance(value, str) or not value for value in self.tokens):
            raise StereoFreeMotifLexerError("lexer emitted an invalid token")
        if decode_pure_motif(self.tokens) != self.pure_motif:
            raise StereoFreeMotifLexerError("lexer output is not an exact fixed point")


def _append_digits(
    tokens: list[str], roles: list[str], value: str, role: str
) -> None:
    for character in value:
        if character not in "0123456789":
            raise StereoFreeMotifLexerError("non-decimal character in numeric field")
        tokens.append(character)
        roles.append(role)


def _lex_bracket_atom(content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if any(marker in content for marker in STEREO_MARKERS):
        raise StereoFreeMotifLexerError("stereochemistry leaked into a pure motif")
    match = _BRACKET_ATOM_RE.fullmatch(content)
    if match is None:
        raise StereoFreeMotifLexerError(
            "bracket atom is outside the frozen stereo-free grammar"
        )
    symbol = match.group("symbol")
    if symbol != "*" and symbol not in ELEMENT_SYMBOLS and symbol not in AROMATIC_SYMBOLS:
        raise StereoFreeMotifLexerError("bracket atom uses an unknown element symbol")
    tokens = ["["]
    roles = ["bracket_open"]
    _append_digits(tokens, roles, match.group("isotope"), "isotope_digit")
    tokens.append(symbol)
    roles.append("atom_symbol")
    hcount = match.group("hcount")
    if hcount:
        tokens.append("H")
        roles.append("hydrogen_marker")
        _append_digits(tokens, roles, hcount[1:], "hydrogen_count_digit")
    charge = match.group("charge")
    if charge:
        for character in charge:
            tokens.append(character)
            roles.append("charge_sign" if character in "+-" else "charge_digit")
    atom_class = match.group("atom_class")
    if atom_class:
        tokens.append(":")
        roles.append("atom_class_marker")
        _append_digits(tokens, roles, atom_class[1:], "atom_class_digit")
    tokens.append("]")
    roles.append("bracket_close")
    return tuple(tokens), tuple(roles)


def lex_pure_motif(pure_motif: str) -> LexedPureMotif:
    """Lex one ``[core]`` pure motif into a finite, exact token stream."""

    if not isinstance(pure_motif, str) or len(pure_motif) < 3:
        raise StereoFreeMotifLexerError("pure motif must use the outer [core] form")
    if not (pure_motif.startswith("[") and pure_motif.endswith("]")):
        raise StereoFreeMotifLexerError("pure motif must use the outer [core] form")
    if any(marker in pure_motif for marker in STEREO_MARKERS):
        raise StereoFreeMotifLexerError("stereochemistry leaked into a pure motif")
    if "<" in pure_motif or ">" in pure_motif:
        raise StereoFreeMotifLexerError("anchor or control text leaked into a pure motif")

    core = pure_motif[1:-1]
    tokens: list[str] = []
    roles: list[str] = []
    cursor = 0
    while cursor < len(core):
        if core.startswith("()", cursor):
            tokens.append(SLOT_TOKEN)
            roles.append("anchor_slot")
            cursor += 2
            continue
        character = core[cursor]
        if character == "[":
            stop = core.find("]", cursor + 1)
            if stop < 0 or "[" in core[cursor + 1 : stop]:
                raise StereoFreeMotifLexerError("bracket atom is unclosed or nested")
            bracket_tokens, bracket_roles = _lex_bracket_atom(core[cursor + 1 : stop])
            tokens.extend(bracket_tokens)
            roles.extend(bracket_roles)
            cursor = stop + 1
            continue
        if character == "]":
            raise StereoFreeMotifLexerError("stray bracket close in pure motif")
        atom = None
        for width in (2, 1):
            candidate = core[cursor : cursor + width]
            if candidate in ORGANIC_SUBSET:
                atom = candidate
                break
        if atom is not None:
            tokens.append(atom)
            roles.append("atom_symbol")
            cursor += len(atom)
            continue
        if character == "%":
            ring = core[cursor + 1 : cursor + 3]
            if len(ring) != 2 or not ring.isdigit():
                raise StereoFreeMotifLexerError("percent ring closure is malformed")
            tokens.append("%")
            roles.append("ring_percent")
            _append_digits(tokens, roles, ring, "ring_digit")
            cursor += 3
            continue
        if character.isdigit() and character.isascii():
            tokens.append(character)
            roles.append("ring_digit")
            cursor += 1
            continue
        if character in OUTSIDE_PUNCTUATION:
            tokens.append(character)
            if character in "()":
                role = "branch"
            elif character == ".":
                role = "component_separator"
            elif character in "=#-:~":
                role = "bond"
            else:
                role = "syntax"
            roles.append(role)
            cursor += 1
            continue
        raise StereoFreeMotifLexerError(
            f"unsupported pure-motif character at offset {cursor}: {character!r}"
        )
    if not tokens:
        raise StereoFreeMotifLexerError("pure motif core cannot be empty")
    return LexedPureMotif(
        schema_version=LEXER_SCHEMA_VERSION,
        pure_motif=pure_motif,
        tokens=tuple(tokens),
        roles=tuple(roles),
    )


def decode_pure_motif(tokens: Sequence[str]) -> str:
    """Decode lexer tokens exactly; callers must not insert whitespace."""

    if not tokens:
        raise StereoFreeMotifLexerError("cannot decode an empty token stream")
    core_parts = []
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise StereoFreeMotifLexerError("decoder received an invalid token")
        if token == SLOT_TOKEN:
            core_parts.append("()")
        else:
            # Anchor/control strings are never part of this lexer domain.
            if token.startswith("<") or token.endswith(">"):
                raise StereoFreeMotifLexerError("decoder received an unknown control token")
            core_parts.append(token)
    return "[" + "".join(core_parts) + "]"


def byte_fallback_tokens(pure_motif: str) -> tuple[str, ...]:
    """Return a universal UTF-8 byte fallback for diagnostic comparison."""

    if not isinstance(pure_motif, str) or not pure_motif:
        raise StereoFreeMotifLexerError("byte fallback input must be nonempty")
    return tuple(f"<0x{value:02X}>" for value in pure_motif.encode("utf-8"))


def decode_byte_fallback(tokens: Sequence[str]) -> str:
    values = []
    for token in tokens:
        if not isinstance(token, str) or re.fullmatch(r"<0x[0-9A-F]{2}>", token) is None:
            raise StereoFreeMotifLexerError("malformed byte fallback token")
        values.append(int(token[3:5], 16))
    try:
        return bytes(values).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StereoFreeMotifLexerError("byte fallback is not valid UTF-8") from exc


def chemical_token_universe() -> tuple[str, ...]:
    """Return the finite logical-token domain in deterministic UTF-8 order."""

    values = (
        set(ELEMENT_SYMBOLS)
        | set(AROMATIC_SYMBOLS)
        | set(ORGANIC_SUBSET)
        | set(OUTSIDE_PUNCTUATION)
        | {"*", "[", "]", "%", SLOT_TOKEN}
        | set("0123456789")
    )
    return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


def opaque_chemical_token_map() -> tuple[tuple[str, str], ...]:
    """Map logical chemistry units to ordinary opaque tokenizer additions.

    Registering raw punctuation or element strings with Hugging Face
    ``add_tokens`` can change tokenization of ordinary natural language.  The
    model surface therefore uses opaque names while the sidecar retains each
    exact logical chemistry unit.
    """

    return tuple(
        (logical, f"<MOST:CHEM:{rank:03d}>")
        for rank, logical in enumerate(chemical_token_universe())
    )


def validate_anchor_token_outside_lexer(token: str) -> int:
    """Document the adjacent anchor namespace without mixing it into chemistry."""

    try:
        return parse_anchor_token(token)
    except AnchoredMotifSurfaceError as exc:
        raise StereoFreeMotifLexerError("invalid adjacent anchor token") from exc


__all__ = [
    "ELEMENT_SYMBOLS",
    "LEXER_SCHEMA_VERSION",
    "LexedPureMotif",
    "SLOT_TOKEN",
    "StereoFreeMotifLexerError",
    "byte_fallback_tokens",
    "chemical_token_universe",
    "decode_byte_fallback",
    "decode_pure_motif",
    "lex_pure_motif",
    "opaque_chemical_token_map",
    "validate_anchor_token_outside_lexer",
]
