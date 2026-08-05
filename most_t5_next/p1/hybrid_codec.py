"""Minimal hybrid motif identity and connection schema.

This module is a candidate contract, not a production chemical tokenizer.  It
implements only the deterministic mechanics needed by synthetic fixtures:

* a logical motif identity contains ordered local identity lexemes and the
  positions of its unlabeled attachment slots;
* a frequent identity may use one macro token, while the same identity can be
  forced through a framed fallback surface;
* both surfaces decode to the same canonical payload and SHA-256 digest; and
* molecule-local edge labels live in a separate connection schema.

The fallback lexemes are intentionally treated as already canonical.  A later
chemistry-aware codec must replace that assumption and prove graph round-trip
on its declared support domain before this path can admit training data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable, Sequence


CODEC_SCHEMA_VERSION = "most-t5-next/p1-hybrid-motif-codec-synthetic/v1"
FALLBACK_BEGIN = "<FALLBACK_BEGIN>"
FALLBACK_END = "<FALLBACK_END>"
_MACRO_PREFIX = "<MOTIF_"
_LEXEME_TOKEN_PREFIX = "<FB_LEX:"
_SLOT_TOKEN_PREFIX = "<FB_SLOT:"
_LEXEME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:+./=@#-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_BOND_TYPES = frozenset(
    {"single", "double", "triple", "aromatic", "dative", "other"}
)


class CodecContractError(ValueError):
    """Raised when a synthetic codec/schema invariant is violated."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True)
class LogicalMotifIdentity:
    """Canonical local motif identity used by the synthetic codec skeleton.

    ``canonical_lexemes`` stand in for the future atom/bond/branch/stereo
    grammar.  ``slot_atom_positions`` retain each attachment slot's position
    in the motif-local, sorted real-atom list.  Edge-pair IDs are deliberately
    absent: they belong to :class:`CrossMotifConnection`.
    """

    canonical_lexemes: tuple[str, ...]
    slot_atom_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_lexemes:
            raise CodecContractError("a logical motif identity needs at least one lexeme")
        for lexeme in self.canonical_lexemes:
            if not isinstance(lexeme, str) or not _LEXEME_RE.fullmatch(lexeme):
                raise CodecContractError(
                    "fallback lexemes must match the closed synthetic token alphabet"
                )
        for position in self.slot_atom_positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise CodecContractError("slot atom positions must be nonnegative integers")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "canonical_lexemes": list(self.canonical_lexemes),
            "schema_version": CODEC_SCHEMA_VERSION,
            "slot_atom_positions": list(self.slot_atom_positions),
        }

    @property
    def exact_identity_digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_payload)).hexdigest()


@dataclass(frozen=True)
class SurfaceEncoding:
    """One surface realization of exactly one logical motif."""

    mode: str
    tokens: tuple[str, ...]
    carrier_offset: int
    exact_identity_digest: str

    def __post_init__(self) -> None:
        if self.mode not in {"macro", "fallback"}:
            raise CodecContractError("surface mode must be macro or fallback")
        if not self.tokens:
            raise CodecContractError("a surface encoding cannot be empty")
        if self.carrier_offset != 0:
            raise CodecContractError("v1 uses the first identity token as its sole carrier")
        if not _SHA256_RE.fullmatch(self.exact_identity_digest):
            raise CodecContractError("exact identity digest must be lowercase SHA-256")


class HybridMotifCodec:
    """Deterministic macro/fallback codec for declared synthetic identities.

    Macro IDs are assigned by sorted identity digest, never by insertion order.
    Any valid :class:`LogicalMotifIdentity` can still use fallback, including an
    identity not present in the macro table.  No ``<unk>`` route exists.
    """

    def __init__(self, macro_identities: Iterable[LogicalMotifIdentity]) -> None:
        identities_by_digest: dict[str, LogicalMotifIdentity] = {}
        for identity in macro_identities:
            if not isinstance(identity, LogicalMotifIdentity):
                raise TypeError("macro_identities must contain LogicalMotifIdentity values")
            digest = identity.exact_identity_digest
            prior = identities_by_digest.get(digest)
            if prior is not None and prior != identity:
                raise CodecContractError("identity digest collision in macro registry")
            identities_by_digest[digest] = identity

        self._identity_by_digest = dict(sorted(identities_by_digest.items()))
        self._macro_by_digest = {
            digest: f"{_MACRO_PREFIX}{ordinal:06d}>"
            for ordinal, digest in enumerate(self._identity_by_digest)
        }
        self._digest_by_macro = {
            token: digest for digest, token in self._macro_by_digest.items()
        }

    @property
    def macro_tokens(self) -> tuple[str, ...]:
        return tuple(self._digest_by_macro)

    def encode(
        self, identity: LogicalMotifIdentity, *, force_fallback: bool = False
    ) -> SurfaceEncoding:
        if not isinstance(identity, LogicalMotifIdentity):
            raise TypeError("identity must be a LogicalMotifIdentity")
        digest = identity.exact_identity_digest
        if not force_fallback and digest in self._macro_by_digest:
            tokens = (self._macro_by_digest[digest],)
            mode = "macro"
        else:
            tokens = self._fallback_tokens(identity)
            mode = "fallback"
        return SurfaceEncoding(mode, tokens, 0, digest)

    def decode(self, tokens: Sequence[str]) -> LogicalMotifIdentity:
        surface = tuple(tokens)
        if len(surface) == 1 and surface[0] in self._digest_by_macro:
            return self._identity_by_digest[self._digest_by_macro[surface[0]]]
        return self._decode_fallback(surface)

    def verify_round_trip(
        self, identity: LogicalMotifIdentity, *, force_fallback: bool = False
    ) -> SurfaceEncoding:
        encoded = self.encode(identity, force_fallback=force_fallback)
        decoded = self.decode(encoded.tokens)
        if decoded != identity or decoded.exact_identity_digest != encoded.exact_identity_digest:
            raise CodecContractError("identity surface failed deterministic round-trip")
        return encoded

    @staticmethod
    def _fallback_tokens(identity: LogicalMotifIdentity) -> tuple[str, ...]:
        lexemes = tuple(f"{_LEXEME_TOKEN_PREFIX}{item}>" for item in identity.canonical_lexemes)
        slots = tuple(f"{_SLOT_TOKEN_PREFIX}{item}>" for item in identity.slot_atom_positions)
        return (FALLBACK_BEGIN, *lexemes, *slots, FALLBACK_END)

    @staticmethod
    def _decode_fallback(surface: tuple[str, ...]) -> LogicalMotifIdentity:
        if len(surface) < 3 or surface[0] != FALLBACK_BEGIN or surface[-1] != FALLBACK_END:
            raise CodecContractError("fallback surface must have intact begin/end boundaries")
        if FALLBACK_BEGIN in surface[1:] or FALLBACK_END in surface[:-1]:
            raise CodecContractError("nested or premature fallback boundary")

        lexemes: list[str] = []
        slot_positions: list[int] = []
        reached_slots = False
        for token in surface[1:-1]:
            if token.startswith(_LEXEME_TOKEN_PREFIX) and token.endswith(">"):
                if reached_slots:
                    raise CodecContractError("fallback lexemes must precede slot positions")
                lexemes.append(token[len(_LEXEME_TOKEN_PREFIX) : -1])
                continue
            if token.startswith(_SLOT_TOKEN_PREFIX) and token.endswith(">"):
                reached_slots = True
                raw_position = token[len(_SLOT_TOKEN_PREFIX) : -1]
                if not raw_position.isdecimal():
                    raise CodecContractError("fallback slot position is not decimal")
                slot_positions.append(int(raw_position))
                continue
            raise CodecContractError("unknown token in fallback surface")
        return LogicalMotifIdentity(tuple(lexemes), tuple(slot_positions))


@dataclass(frozen=True)
class LogicalMotif:
    """One logical motif in the record atom domain."""

    logical_motif_id: int
    identity: LogicalMotifIdentity
    atom_indices: tuple[int, ...]
    geometry_valid: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.logical_motif_id, bool)
            or not isinstance(self.logical_motif_id, int)
            or self.logical_motif_id < 0
        ):
            raise CodecContractError("logical motif ID must be a nonnegative integer")
        if not self.atom_indices:
            raise CodecContractError("a logical motif must contain at least one real atom")
        if tuple(sorted(set(self.atom_indices))) != self.atom_indices:
            raise CodecContractError("motif atom indices must be strictly ascending")
        if any(isinstance(atom, bool) or not isinstance(atom, int) or atom < 0 for atom in self.atom_indices):
            raise CodecContractError("motif atom indices must be nonnegative integers")
        if not isinstance(self.geometry_valid, bool):
            raise CodecContractError("geometry_valid must be bool")
        if any(position >= len(self.atom_indices) for position in self.identity.slot_atom_positions):
            raise CodecContractError("identity slot position is outside the motif atom list")

    @property
    def slot_atom_indices(self) -> tuple[int, ...]:
        return tuple(self.atom_indices[position] for position in self.identity.slot_atom_positions)


@dataclass(frozen=True, order=True)
class ConnectionEndpoint:
    """One endpoint of a molecule-local cross-motif edge."""

    logical_motif_id: int
    slot_id: int
    atom_index: int

    def __post_init__(self) -> None:
        values = (self.logical_motif_id, self.slot_id, self.atom_index)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise CodecContractError("connection endpoint values must be nonnegative integers")


@dataclass(frozen=True)
class CrossMotifConnection:
    """A canonical two-ended edge kept outside motif identity lexemes."""

    edge_id: int
    endpoint_a: ConnectionEndpoint
    endpoint_b: ConnectionEndpoint
    bond_type: str

    def __post_init__(self) -> None:
        if isinstance(self.edge_id, bool) or not isinstance(self.edge_id, int) or self.edge_id < 0:
            raise CodecContractError("edge ID must be a nonnegative integer")
        if self.endpoint_a >= self.endpoint_b:
            raise CodecContractError("connection endpoints must be strictly canonical-ordered")
        if self.endpoint_a.logical_motif_id == self.endpoint_b.logical_motif_id:
            raise CodecContractError("a cross-motif connection cannot stay within one motif")
        if self.bond_type not in ALLOWED_BOND_TYPES:
            raise CodecContractError(
                "bond type must use the lowercase vNext closed set: "
                f"{sorted(ALLOWED_BOND_TYPES)}"
            )

    @classmethod
    def canonical(
        cls,
        edge_id: int,
        left: ConnectionEndpoint,
        right: ConnectionEndpoint,
        bond_type: str,
    ) -> "CrossMotifConnection":
        endpoint_a, endpoint_b = sorted((left, right))
        return cls(edge_id, endpoint_a, endpoint_b, bond_type)


@dataclass(frozen=True)
class LogicalMoleculeSchema:
    """Closed synthetic molecule schema linking motif, connection and atoms."""

    motifs: tuple[LogicalMotif, ...]
    connections: tuple[CrossMotifConnection, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def atom_count(self) -> int:
        return sum(len(motif.atom_indices) for motif in self.motifs)

    def validate(self) -> None:
        if not self.motifs:
            raise CodecContractError("logical molecule schema needs at least one motif")
        motif_ids = tuple(motif.logical_motif_id for motif in self.motifs)
        if motif_ids != tuple(range(len(self.motifs))):
            raise CodecContractError("motifs must be ordered and densely indexed from zero")

        flattened_atoms = [atom for motif in self.motifs for atom in motif.atom_indices]
        if sorted(flattened_atoms) != list(range(len(flattened_atoms))):
            raise CodecContractError(
                "motif atom groups must be disjoint and cover the dense atom domain"
            )

        edge_ids = tuple(connection.edge_id for connection in self.connections)
        if edge_ids != tuple(range(len(self.connections))):
            raise CodecContractError("connections must be ordered and densely indexed from zero")
        edge_keys = tuple(
            (connection.endpoint_a, connection.endpoint_b, connection.bond_type)
            for connection in self.connections
        )
        if edge_keys != tuple(sorted(edge_keys)):
            raise CodecContractError(
                "connections must follow canonical endpoint/bond ordering"
            )

        observed_slots: set[tuple[int, int]] = set()
        for connection in self.connections:
            for endpoint in (connection.endpoint_a, connection.endpoint_b):
                if endpoint.logical_motif_id >= len(self.motifs):
                    raise CodecContractError("connection references an unknown logical motif")
                motif = self.motifs[endpoint.logical_motif_id]
                if endpoint.slot_id >= len(motif.slot_atom_indices):
                    raise CodecContractError("connection references an unknown motif slot")
                if endpoint.atom_index != motif.slot_atom_indices[endpoint.slot_id]:
                    raise CodecContractError("connection endpoint atom does not match its slot")
                slot_key = (endpoint.logical_motif_id, endpoint.slot_id)
                if slot_key in observed_slots:
                    raise CodecContractError("a motif slot is used by more than one edge")
                observed_slots.add(slot_key)

        expected_slots = {
            (motif.logical_motif_id, slot_id)
            for motif in self.motifs
            for slot_id in range(len(motif.slot_atom_indices))
        }
        if observed_slots != expected_slots:
            raise CodecContractError(
                "closed synthetic schema requires every declared slot exactly once"
            )
