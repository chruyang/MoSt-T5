"""Pinned Smirk-derived molecular glyph vocabulary.

The molecular front end and the natural-language T5 tokenizer deliberately
share one embedding table but not one segmentation algorithm.  Text continues
to use the base T5 SentencePiece model.  Inside an explicit molecular span,
SMILES is split into the atomically complete glyphs introduced by Smirk and
each glyph is mapped to one collision-free ordinary T5 added-token row.

The ordered core below is Smirk 0.3.0's ``vocab_smiles.json`` model vocabulary,
excluding its ``[UNK]`` row.  ``si`` and ``te`` are the only extensions: pinned
RDKit 2024.03.5 emits those lower-case aromatic spellings, whereas upstream
Smirk covers ``se`` and ``as`` but not these two RDKit extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


SCHEMA_VERSION = "most-t5-next/smirk-smiles-vocabulary/v1"
UPSTREAM_PROJECT = "BattModels/smirk"
UPSTREAM_DISTRIBUTION_VERSION = "0.3.0"
UPSTREAM_SOURCE_COMMIT = "5b8210612cdecb57e1cbc1aaa8cf38a081c1453e"

_CORE_GLYPHS = (
    "#", "$", "%", "(", ")", "*", "+", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ":", "=", "@", "@@", "@AL", "@OH", "@SP", "@TB", "@TH",
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba",
    "Be", "Bh", "Bi", "Bk", "Br", "C", "Ca", "Cd", "Ce", "Cf",
    "Cl", "Cm", "Cn", "Co", "Cr", "Cs", "Cu", "Db", "Ds", "Dy",
    "Er", "Es", "Eu", "F", "Fe", "Fl", "Fm", "Fr", "Ga", "Gd",
    "Ge", "H", "He", "Hf", "Hg", "Ho", "Hs", "I", "In", "Ir",
    "K", "Kr", "La", "Li", "Lr", "Lu", "Lv", "Mc", "Md", "Mg",
    "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Nh", "Ni",
    "No", "Np", "O", "Og", "Os", "P", "Pa", "Pb", "Pd", "Pm",
    "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rg", "Rh",
    "Rn", "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn",
    "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "Ts",
    "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr", "[", "\\", "]",
    "as", "b", "c", "n", "o", "p", "s", "se",
)
_RDKIT_EXTENSIONS = ("si", "te")

# These expressions are a Python transcription of Smirk 0.3.0's
# src/pre_tokenizers/split_smiles.rs.  The two RDKit aromatic extensions are
# added as complete bracket symbols; all numeric fields are decomposed to
# digits exactly as in Smirk.
_ORGANIC = ("Br", "Cl", "B", "C", "F", "I", "N", "O", "P", "S", "b", "c", "n", "o", "p", "s", "*")
_ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
_BRACKET_SYMBOLS = tuple(sorted(set(_ELEMENTS) | {"as", "b", "c", "n", "o", "p", "s", "se", "si", "te", "*"}, key=lambda x: (-len(x), x)))
_CHIRAL = ("@@", "@AL", "@OH", "@SP", "@TB", "@TH", "@")
_OUTSIDE_SINGLE = frozenset(".-=#$:/\\%()0123456789")
_BRACKET_RE = re.compile(
    r"^(?P<isotope>\d+)?"
    r"(?P<symbol>" + "|".join(re.escape(x) for x in _BRACKET_SYMBOLS) + r")"
    r"(?:(?P<chiral>@@|@AL|@OH|@SP|@TB|@TH|@)(?P<chiral_index>\d{1,2})?)?"
    r"(?:(?P<h>H)(?P<hcount>\d)?)?"
    r"(?:(?P<charge>[+-]{1,2})(?P<charge_count>\d{0,2}))?"
    r"(?:(?P<class_marker>:)(?P<atom_class>\d+))?$"
)


class SmirkSmilesVocabularyError(ValueError):
    """A string lies outside the pinned molecular glyph contract."""


@dataclass(frozen=True)
class SmirkSmilesEncoding:
    schema_version: str
    smiles: str
    glyphs: tuple[str, ...]
    character_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.glyphs:
            raise SmirkSmilesVocabularyError("invalid Smirk encoding")
        if len(self.glyphs) != len(self.character_spans):
            raise SmirkSmilesVocabularyError("glyph and span arrays disagree")
        if "".join(self.glyphs) != self.smiles:
            raise SmirkSmilesVocabularyError("glyphs are not an exact fixed point")


def smiles_glyph_universe() -> tuple[str, ...]:
    glyphs = _CORE_GLYPHS + _RDKIT_EXTENSIONS
    if len(_CORE_GLYPHS) != 158 or len(glyphs) != 160 or len(set(glyphs)) != len(glyphs):
        raise SmirkSmilesVocabularyError("pinned Smirk glyph universe drifted")
    return glyphs


_DECIMAL_DIGITS = frozenset("0123456789")


def glyph_surface_token(glyph_index: int) -> str:
    if glyph_index < 0 or glyph_index >= len(smiles_glyph_universe()):
        raise SmirkSmilesVocabularyError("glyph index is out of range")
    glyph = smiles_glyph_universe()[glyph_index]
    # Match the pinned 3D-MolT5 tokenizer contract: molecular numeric fields
    # reuse the existing T5 digit rows and are emitted one digit at a time.
    if glyph in _DECIMAL_DIGITS:
        return glyph
    return f"<MOST:SMI:{glyph_index:03d}>"


def smiles_glyph_token_map() -> tuple[tuple[str, str], ...]:
    return tuple(
        (glyph, glyph_surface_token(index))
        for index, glyph in enumerate(smiles_glyph_universe())
    )


def smiles_added_token_universe() -> tuple[str, ...]:
    """Return only glyph surfaces that require new tokenizer rows."""

    return tuple(
        surface
        for glyph, surface in smiles_glyph_token_map()
        if glyph not in _DECIMAL_DIGITS
    )


def _append_value(
    glyphs: list[str], spans: list[tuple[int, int]], value: str | None, start: int
) -> int:
    if not value:
        return start
    for char in value:
        glyphs.append(char)
        spans.append((start, start + 1))
        start += 1
    return start


def _tokenize_bracket(content: str, offset: int) -> tuple[list[str], list[tuple[int, int]]]:
    match = _BRACKET_RE.fullmatch(content)
    if match is None:
        raise SmirkSmilesVocabularyError("bracket atom is outside pinned Smirk/RDKit grammar")
    glyphs: list[str] = ["["]
    spans: list[tuple[int, int]] = [(offset, offset + 1)]
    cursor = offset + 1
    cursor = _append_value(glyphs, spans, match.group("isotope"), cursor)
    for group in ("symbol", "chiral"):
        value = match.group(group)
        if value:
            glyphs.append(value)
            spans.append((cursor, cursor + len(value)))
            cursor += len(value)
    cursor = _append_value(glyphs, spans, match.group("chiral_index"), cursor)
    for group in ("h",):
        value = match.group(group)
        if value:
            glyphs.append(value)
            spans.append((cursor, cursor + len(value)))
            cursor += len(value)
    cursor = _append_value(glyphs, spans, match.group("hcount"), cursor)
    value = match.group("charge")
    if value:
        # Smirk normalizes ++/-- to +2/-2 before tokenization.  RDKit canonical
        # SMILES normally emits the latter; accept the former with the same
        # deterministic normalization only at the caller boundary.
        for sign in value:
            glyphs.append(sign)
            spans.append((cursor, cursor + 1))
            cursor += 1
    cursor = _append_value(glyphs, spans, match.group("charge_count"), cursor)
    value = match.group("class_marker")
    if value:
        glyphs.append(value)
        spans.append((cursor, cursor + 1))
        cursor += 1
    cursor = _append_value(glyphs, spans, match.group("atom_class"), cursor)
    if cursor != offset + 1 + len(content):
        raise SmirkSmilesVocabularyError("bracket atom span accounting drifted")
    glyphs.append("]")
    spans.append((cursor, cursor + 1))
    return glyphs, spans


def encode_smiles_glyphs(smiles: str) -> SmirkSmilesEncoding:
    if not isinstance(smiles, str) or not smiles or not smiles.isascii():
        raise SmirkSmilesVocabularyError("SMILES must be nonempty ASCII")
    glyphs: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(smiles):
        if smiles[index] == "[":
            stop = smiles.find("]", index + 1)
            if stop < 0 or "[" in smiles[index + 1 : stop]:
                raise SmirkSmilesVocabularyError("bracket atom is unclosed or nested")
            inner_glyphs, inner_spans = _tokenize_bracket(smiles[index + 1 : stop], index)
            glyphs.extend(inner_glyphs)
            spans.extend(inner_spans)
            index = stop + 1
            continue
        token = next((x for x in _ORGANIC if smiles.startswith(x, index)), None)
        if token is None and smiles[index] in _OUTSIDE_SINGLE:
            token = smiles[index]
        if token is None:
            raise SmirkSmilesVocabularyError(
                f"unsupported SMILES text at character {index}: {smiles[index]!r}"
            )
        glyphs.append(token)
        spans.append((index, index + len(token)))
        index += len(token)
    universe = set(smiles_glyph_universe())
    if any(glyph not in universe for glyph in glyphs):
        raise SmirkSmilesVocabularyError("tokenizer emitted an unregistered glyph")
    return SmirkSmilesEncoding(SCHEMA_VERSION, smiles, tuple(glyphs), tuple(spans))


def decode_smiles_glyphs(glyphs: Sequence[str]) -> str:
    if not glyphs or any(glyph not in set(smiles_glyph_universe()) for glyph in glyphs):
        raise SmirkSmilesVocabularyError("decoder received an unknown glyph")
    return "".join(glyphs)


def require_stereo_free_fragment(fragment_smiles: str) -> SmirkSmilesEncoding:
    """Tokenize one ordinary motif identity and reject leaked stereo syntax.

    This is a representation-policy boundary, not a second lexer: the exact
    same Smirk vocabulary parses ordinary fragments and whole-molecule
    fallbacks.  Stereo glyphs remain available for the latter and for explicit
    reaction-SMILES paths, while compact motif identity stays non-stereochemical.
    """

    encoding = encode_smiles_glyphs(fragment_smiles)
    if any(glyph in {"/", "\\"} or glyph.startswith("@") for glyph in encoding.glyphs):
        raise SmirkSmilesVocabularyError(
            "stereochemistry leaked into an ordinary fragment identity"
        )
    return encoding


__all__ = [
    "SCHEMA_VERSION",
    "SmirkSmilesEncoding",
    "SmirkSmilesVocabularyError",
    "UPSTREAM_DISTRIBUTION_VERSION",
    "UPSTREAM_PROJECT",
    "UPSTREAM_SOURCE_COMMIT",
    "decode_smiles_glyphs",
    "encode_smiles_glyphs",
    "glyph_surface_token",
    "require_stereo_free_fragment",
    "smiles_added_token_universe",
    "smiles_glyph_token_map",
    "smiles_glyph_universe",
]
