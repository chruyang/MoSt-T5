"""Reversible model-token surface for stereo-free anchored motif phrases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
    ANCHOR_RE,
    EXPLICIT_MOTIF_BOUNDARY,
    anchor_token,
    parse_anchor_token,
)
from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
    decode_pure_motif,
    lex_pure_motif,
    opaque_chemical_token_map,
)


SCHEMA_VERSION = "most-t5-next/anchored-motif-model-surface/v1"
FALLBACK_MOTIF_PREFIX = "<MOST:FALLBACK>"
FALLBACK_MOTIF_SUFFIX = "<MOST:FALLBACK_END>"
FROZEN_GENERATIVE_BOUNDARY_MODE = "fallback_single_suffix"
FORMAL_BOUNDARY_MODES = (FROZEN_GENERATIVE_BOUNDARY_MODE,)
DIAGNOSTIC_BOUNDARY_MODES = (
    "fallback_single_prefix",
    "explicit_single_prefix",
    "implicit_sidecar",
)
BOUNDARY_MODES = FORMAL_BOUNDARY_MODES + DIAGNOSTIC_BOUNDARY_MODES


class AnchoredMotifModelSurfaceError(ValueError):
    """The macro/lexer/anchor model surface is not exactly reversible."""


@dataclass(frozen=True)
class EncodedAnchoredMotifSequence:
    schema_version: str
    boundary_mode: str
    tokens: tuple[str, ...]
    phrase_spans: tuple[tuple[int, int], ...]
    identity_spans: tuple[tuple[int, int], ...]
    anchor_token_positions: tuple[tuple[int, ...], ...]
    macro_used: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.boundary_mode not in BOUNDARY_MODES:
            raise AnchoredMotifModelSurfaceError("invalid model-surface schema or mode")
        sizes = {
            len(self.phrase_spans),
            len(self.identity_spans),
            len(self.anchor_token_positions),
            len(self.macro_used),
        }
        if len(sizes) != 1 or not self.phrase_spans:
            raise AnchoredMotifModelSurfaceError("model-surface phrase arrays disagree")


def _macro_maps(
    macro_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], dict[str, str]]:
    identity_to_token: dict[str, str] = {}
    token_to_identity: dict[str, str] = {}
    for row in macro_rows:
        pure = row.get("pure_motif")
        token = row.get("surface_token")
        if (
            not isinstance(pure, str)
            or not pure
            or not isinstance(token, str)
            or not token
            or pure in identity_to_token
            or token in token_to_identity
        ):
            raise AnchoredMotifModelSurfaceError("macro registry is malformed")
        identity_to_token[pure] = token
        token_to_identity[token] = pure
    return identity_to_token, token_to_identity


def _chemical_maps() -> tuple[dict[str, str], dict[str, str]]:
    logical_to_surface = dict(opaque_chemical_token_map())
    surface_to_logical = {surface: logical for logical, surface in logical_to_surface.items()}
    if len(logical_to_surface) != len(surface_to_logical):
        raise AnchoredMotifModelSurfaceError("chemical token mapping is not bijective")
    return logical_to_surface, surface_to_logical


def encode_phrases(
    phrases: Sequence[Mapping[str, object]],
    macro_rows: Sequence[Mapping[str, object]],
    *,
    boundary_mode: str,
) -> EncodedAnchoredMotifSequence:
    if boundary_mode not in BOUNDARY_MODES:
        raise AnchoredMotifModelSurfaceError("unsupported boundary mode")
    identity_to_macro, _ = _macro_maps(macro_rows)
    logical_to_chemical, _ = _chemical_maps()
    tokens: list[str] = []
    phrase_spans = []
    identity_spans = []
    anchor_positions = []
    macro_flags = []
    for phrase in phrases:
        pure = phrase.get("pure_motif")
        anchors = phrase.get("anchors")
        if not isinstance(pure, str) or not isinstance(anchors, list):
            raise AnchoredMotifModelSurfaceError("phrase row is malformed")
        start = len(tokens)
        if boundary_mode == "explicit_single_prefix":
            tokens.append(EXPLICIT_MOTIF_BOUNDARY)
        positions = []
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                raise AnchoredMotifModelSurfaceError("anchor row is malformed")
            anchor_id = anchor.get("anchor_id")
            if isinstance(anchor_id, bool) or not isinstance(anchor_id, int):
                raise AnchoredMotifModelSurfaceError("anchor ID is malformed")
            positions.append(len(tokens))
            tokens.append(anchor_token(anchor_id))
        identity_start = len(tokens)
        macro = identity_to_macro.get(pure)
        if macro is not None:
            tokens.append(macro)
            macro_flags.append(True)
        else:
            if boundary_mode == "fallback_single_prefix":
                tokens.append(FALLBACK_MOTIF_PREFIX)
                identity_start = len(tokens)
            lexed = lex_pure_motif(pure)
            try:
                tokens.extend(logical_to_chemical[token] for token in lexed.tokens)
            except KeyError as exc:
                raise AnchoredMotifModelSurfaceError("lexer emitted an unregistered token") from exc
            macro_flags.append(False)
            if boundary_mode == "fallback_single_suffix":
                tokens.append(FALLBACK_MOTIF_SUFFIX)
        identity_stop = len(tokens)
        phrase_spans.append((start, identity_stop))
        identity_spans.append((identity_start, identity_stop))
        anchor_positions.append(tuple(positions))
    if not tokens:
        raise AnchoredMotifModelSurfaceError("cannot encode an empty molecule")
    return EncodedAnchoredMotifSequence(
        schema_version=SCHEMA_VERSION,
        boundary_mode=boundary_mode,
        tokens=tuple(tokens),
        phrase_spans=tuple(phrase_spans),
        identity_spans=tuple(identity_spans),
        anchor_token_positions=tuple(anchor_positions),
        macro_used=tuple(macro_flags),
    )


def encode_frozen_phrases(
    phrases: Sequence[Mapping[str, object]],
    macro_rows: Sequence[Mapping[str, object]],
) -> EncodedAnchoredMotifSequence:
    """Encode the only model-facing generative grammar admitted after Stage 3."""

    return encode_phrases(
        phrases,
        macro_rows,
        boundary_mode=FROZEN_GENERATIVE_BOUNDARY_MODE,
    )


def frozen_grammar_contract() -> dict[str, object]:
    """Return the JSON-safe semantic contract bound into tokenizer artifacts."""

    return {
        "model_surface_schema": SCHEMA_VERSION,
        "boundary_mode": FROZEN_GENERATIVE_BOUNDARY_MODE,
        "macro_phrase": "anchors* + one_macro_token",
        "fallback_phrase": "anchors* + chemical_tokens+ + fallback_suffix",
        "fallback_suffix": FALLBACK_MOTIF_SUFFIX,
        "fallback_suffix_is_ordinary_token": True,
        "macro_is_self_delimiting": True,
        "macro_carrier": "macro_token",
        "fallback_carrier": "fallback_suffix",
        "identity_masking_unit": "complete_motif_identity_phrase",
        "anchors_are_outside_identity_span": True,
        "implicit_sidecar_is_not_a_generative_grammar": True,
        "prefix_and_double_boundary_are_not_training_candidates": True,
    }


def _decode_phrase(
    tokens: Sequence[str], token_to_identity: Mapping[str, str]
) -> tuple[str, tuple[int, ...]]:
    index = 0
    anchors = []
    while index < len(tokens) and ANCHOR_RE.fullmatch(tokens[index] or ""):
        anchors.append(parse_anchor_token(tokens[index]))
        index += 1
    identity_tokens = tuple(tokens[index:])
    if not identity_tokens:
        raise AnchoredMotifModelSurfaceError("phrase has no motif identity")
    if len(identity_tokens) == 1 and identity_tokens[0] in token_to_identity:
        pure = token_to_identity[identity_tokens[0]]
    else:
        _, surface_to_logical = _chemical_maps()
        try:
            logical = tuple(surface_to_logical[token] for token in identity_tokens)
        except KeyError as exc:
            raise AnchoredMotifModelSurfaceError("unknown chemical or macro token") from exc
        pure = decode_pure_motif(logical)
    return pure, tuple(anchors)


def decode_explicit_sequence(
    tokens: Sequence[str], macro_rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Decode a standalone generative sequence without any sidecar."""

    _, token_to_identity = _macro_maps(macro_rows)
    result = []
    index = 0
    while index < len(tokens):
        if tokens[index] != EXPLICIT_MOTIF_BOUNDARY:
            raise AnchoredMotifModelSurfaceError("explicit phrase is missing its prefix")
        stop = index + 1
        while stop < len(tokens) and tokens[stop] != EXPLICIT_MOTIF_BOUNDARY:
            stop += 1
        result.append(_decode_phrase(tokens[index + 1 : stop], token_to_identity))
        index = stop
    if not result:
        raise AnchoredMotifModelSurfaceError("cannot decode an empty sequence")
    return tuple(result)


def decode_fallback_prefixed_sequence(
    tokens: Sequence[str], macro_rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Decode the compact standalone grammar that delimits only fallbacks."""

    _, token_to_identity = _macro_maps(macro_rows)
    _, surface_to_logical = _chemical_maps()
    macro_tokens = set(token_to_identity)
    result = []
    index = 0
    while index < len(tokens):
        anchors = []
        while index < len(tokens) and ANCHOR_RE.fullmatch(tokens[index] or ""):
            anchors.append(parse_anchor_token(tokens[index]))
            index += 1
        if index >= len(tokens):
            raise AnchoredMotifModelSurfaceError("phrase has anchors but no motif identity")
        token = tokens[index]
        if token in macro_tokens:
            result.append((token_to_identity[token], tuple(anchors)))
            index += 1
            continue
        if token != FALLBACK_MOTIF_PREFIX:
            raise AnchoredMotifModelSurfaceError("fallback phrase is missing its prefix")
        index += 1
        logical = []
        while index < len(tokens):
            token = tokens[index]
            if (
                ANCHOR_RE.fullmatch(token or "")
                or token in macro_tokens
                or token == FALLBACK_MOTIF_PREFIX
            ):
                break
            try:
                logical.append(surface_to_logical[token])
            except KeyError as exc:
                raise AnchoredMotifModelSurfaceError("unknown fallback chemical token") from exc
            index += 1
        if not logical:
            raise AnchoredMotifModelSurfaceError("fallback phrase has no chemical identity")
        result.append((decode_pure_motif(logical), tuple(anchors)))
    if not result:
        raise AnchoredMotifModelSurfaceError("cannot decode an empty sequence")
    return tuple(result)


def decode_fallback_suffixed_sequence(
    tokens: Sequence[str], macro_rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Decode the old-carrier-compatible grammar delimited only at fallback end."""

    _, token_to_identity = _macro_maps(macro_rows)
    _, surface_to_logical = _chemical_maps()
    macro_tokens = set(token_to_identity)
    result = []
    index = 0
    while index < len(tokens):
        anchors = []
        while index < len(tokens) and ANCHOR_RE.fullmatch(tokens[index] or ""):
            anchors.append(parse_anchor_token(tokens[index]))
            index += 1
        if index >= len(tokens):
            raise AnchoredMotifModelSurfaceError("phrase has anchors but no motif identity")
        if tokens[index] in macro_tokens:
            result.append((token_to_identity[tokens[index]], tuple(anchors)))
            index += 1
            continue
        logical = []
        while index < len(tokens) and tokens[index] != FALLBACK_MOTIF_SUFFIX:
            token = tokens[index]
            if ANCHOR_RE.fullmatch(token or "") or token in macro_tokens:
                raise AnchoredMotifModelSurfaceError("fallback phrase is missing its suffix")
            try:
                logical.append(surface_to_logical[token])
            except KeyError as exc:
                raise AnchoredMotifModelSurfaceError("unknown fallback chemical token") from exc
            index += 1
        if index >= len(tokens) or tokens[index] != FALLBACK_MOTIF_SUFFIX:
            raise AnchoredMotifModelSurfaceError("fallback phrase is missing its suffix")
        if not logical:
            raise AnchoredMotifModelSurfaceError("fallback phrase has no chemical identity")
        index += 1
        result.append((decode_pure_motif(logical), tuple(anchors)))
    if not result:
        raise AnchoredMotifModelSurfaceError("cannot decode an empty sequence")
    return tuple(result)


def decode_implicit_with_sidecar(
    tokens: Sequence[str],
    phrase_spans: Sequence[Sequence[int]],
    macro_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Decode the encoder-only compact view using its mandatory phrase spans."""

    _, token_to_identity = _macro_maps(macro_rows)
    result = []
    cursor = 0
    for span in phrase_spans:
        if len(span) != 2 or span[0] != cursor or span[1] <= span[0] or span[1] > len(tokens):
            raise AnchoredMotifModelSurfaceError("implicit phrase spans are not dense")
        result.append(_decode_phrase(tokens[span[0] : span[1]], token_to_identity))
        cursor = span[1]
    if cursor != len(tokens) or not result:
        raise AnchoredMotifModelSurfaceError("implicit phrase spans do not cover the sequence")
    return tuple(result)


__all__ = [
    "AnchoredMotifModelSurfaceError",
    "BOUNDARY_MODES",
    "DIAGNOSTIC_BOUNDARY_MODES",
    "EncodedAnchoredMotifSequence",
    "FALLBACK_MOTIF_PREFIX",
    "FALLBACK_MOTIF_SUFFIX",
    "FORMAL_BOUNDARY_MODES",
    "FROZEN_GENERATIVE_BOUNDARY_MODE",
    "SCHEMA_VERSION",
    "decode_explicit_sequence",
    "decode_fallback_prefixed_sequence",
    "decode_fallback_suffixed_sequence",
    "decode_implicit_with_sidecar",
    "encode_frozen_phrases",
    "encode_phrases",
    "frozen_grammar_contract",
]
