"""Strict atom/SELFIES alignment for the A0/A1 production boundary.

This module derives only the immutable token/atom alignment.  It deliberately
does not build a release record, write LMDB, or recompute geometry.  The input
``projected_mol`` is treated as the already-frozen geometry row axis and is
never renumbered or mutated.

The alignment is narrower than the historical 3D-MolT5 helper:

* RDKit's actual canonical-SMILES output order is retained as an explicit
  canonical-position -> model-atom permutation;
* SELFIES encoding is strict and decoder attribution must prove one
  atom-producing SELFIES symbol per model atom;
* decoding must preserve the ordered graph and stereochemistry; and
* the frozen union tokenizer must encode every boundary/SELFIES symbol as one
  exact, reversible, non-UNK token, including in the complete surface.

Any ambiguity is a record-level rejection.  There is no positional guess,
``strict=False`` fallback, or all-padding substitute.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import re
from typing import Any, Iterable

from most_t5_next.p1.bound_record import Span


SELFIES_DISTRIBUTION_VERSION = "2.1.1"
MOLECULE_BEGIN = "<bom>"
MOLECULE_END = "<eom>"
SELFIES_SEPARATOR_TOKEN = "<MOST:A:DOT>"
ATOM_IDENTITY_ROLE = "atom_identity"
BOUNDARY_ROLE = "boundary"
BRANCH_ROLE = "branch"
RING_ROLE = "ring"
SEPARATOR_ROLE = "separator"
ALLOWED_SYMBOL_ROLES = frozenset(
    {ATOM_IDENTITY_ROLE, BRANCH_ROLE, RING_ROLE, SEPARATOR_ROLE}
)
ALLOWED_TOKEN_ROLES = frozenset((*ALLOWED_SYMBOL_ROLES, BOUNDARY_ROLE))

# ``selfies==2.1.1`` exposes ``__version__ == '2.1.0'`` in its wheel.  The
# distribution metadata is therefore the authoritative version boundary.
DEFAULT_SELFIES_211_CONSTRAINTS = {
    "H": 1,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "B": 3,
    "B+1": 2,
    "B-1": 4,
    "O": 2,
    "O+1": 3,
    "O-1": 1,
    "N": 3,
    "N+1": 4,
    "N-1": 2,
    "C": 4,
    "C+1": 5,
    "C-1": 3,
    "P": 5,
    "P+1": 6,
    "P-1": 4,
    "S": 6,
    "S+1": 7,
    "S-1": 5,
    "?": 8,
}

_BRANCH_SYMBOL_RE = re.compile(r"^\[(?:[-=#/\\])?Branch([123])\]$")
_RING_SYMBOL_RE = re.compile(r"^\[(?:[-=#/\\])?Ring([123])\]$")
_BRACKET_ATOM_RE = re.compile(r"^\[[^\[\]]+\]$")
_ORGANIC_ATOM_TOKENS = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "b", "c", "n", "o", "p", "s"}
)
_DECODER_NONATOM_TOKENS = frozenset({"-", "=", "#", ":", "/", "\\", "."})


class AtomSelfiesAlignmentError(ValueError):
    """A molecule cannot cross the strict atom/SELFIES alignment boundary."""

    def __init__(self, reason_code: str, stage: str) -> None:
        self.reason_code = reason_code
        self.stage = stage
        super().__init__(f"{reason_code} [{stage}]")


def _reject(reason_code: str, stage: str) -> None:
    raise AtomSelfiesAlignmentError(reason_code, stage)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class AtomSelfiesSurface:
    """Tokenizer-independent strict SELFIES surface on the model atom axis."""

    canonical_isomeric_smiles: str
    canonical_position_to_model_atom: tuple[int, ...]
    selfies: str
    selfies_symbols: tuple[str, ...]
    symbol_to_model_atom: tuple[int, ...]
    symbol_role: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.canonical_isomeric_smiles or not self.selfies:
            _reject("SURFACE_INVARIANT_INVALID", "dataclass")
        atom_count = len(self.canonical_position_to_model_atom)
        if atom_count == 0 or sorted(self.canonical_position_to_model_atom) != list(
            range(atom_count)
        ):
            _reject("SURFACE_INVARIANT_INVALID", "dataclass")
        if (
            not self.selfies_symbols
            or "".join(self.selfies_symbols) != self.selfies
            or len(self.symbol_to_model_atom) != len(self.selfies_symbols)
            or len(self.symbol_role) != len(self.selfies_symbols)
        ):
            _reject("SURFACE_INVARIANT_INVALID", "dataclass")
        atoms = tuple(value for value in self.symbol_to_model_atom if value >= 0)
        if sorted(atoms) != list(range(atom_count)):
            _reject("SURFACE_INVARIANT_INVALID", "dataclass")
        for atom_id, role in zip(self.symbol_to_model_atom, self.symbol_role):
            if (
                not _is_plain_int(atom_id)
                or atom_id < -1
                or atom_id >= atom_count
                or role not in ALLOWED_SYMBOL_ROLES
                or ((atom_id >= 0) != (role == ATOM_IDENTITY_ROLE))
            ):
                _reject("SURFACE_INVARIANT_INVALID", "dataclass")


@dataclass(frozen=True)
class AtomSelfiesAlignment:
    """Complete one-token-per-symbol alignment, indexed back to model atoms.

    ``canonical_position_to_model_atom[k]`` is the model-row atom serialized
    as canonical SMILES atom ``k``.  ``atom_identity_spans`` and
    ``atom_to_carrier`` are instead indexed by model atom ID, which is the row
    domain consumed by the geometry sidecar.
    """

    canonical_isomeric_smiles: str
    canonical_position_to_model_atom: tuple[int, ...]
    selfies: str
    selfies_symbols: tuple[str, ...]
    symbol_to_model_atom: tuple[int, ...]
    symbol_role: tuple[str, ...]
    input_ids: tuple[int, ...]
    token_to_atom: tuple[int, ...]
    token_role: tuple[str, ...]
    atom_identity_spans: tuple[Span, ...]
    atom_to_carrier: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_isomeric_smiles, str) or not self.canonical_isomeric_smiles:
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if not isinstance(self.selfies, str) or not self.selfies:
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        tuple_fields = (
            "canonical_position_to_model_atom",
            "selfies_symbols",
            "symbol_to_model_atom",
            "symbol_role",
            "input_ids",
            "token_to_atom",
            "token_role",
            "atom_identity_spans",
            "atom_to_carrier",
        )
        if any(not isinstance(getattr(self, field), tuple) for field in tuple_fields):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")

        atom_count = len(self.canonical_position_to_model_atom)
        if atom_count == 0 or sorted(self.canonical_position_to_model_atom) != list(range(atom_count)):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        symbol_count = len(self.selfies_symbols)
        if symbol_count == 0 or "".join(self.selfies_symbols) != self.selfies:
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if len(self.symbol_to_model_atom) != symbol_count or len(self.symbol_role) != symbol_count:
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if any(role not in ALLOWED_SYMBOL_ROLES for role in self.symbol_role):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if any(
            not _is_plain_int(atom_id) or atom_id < -1 or atom_id >= atom_count
            for atom_id in self.symbol_to_model_atom
        ):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        atom_symbols = tuple(atom_id for atom_id in self.symbol_to_model_atom if atom_id >= 0)
        if sorted(atom_symbols) != list(range(atom_count)):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        for atom_id, role in zip(self.symbol_to_model_atom, self.symbol_role):
            if (atom_id >= 0) != (role == ATOM_IDENTITY_ROLE):
                _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")

        token_count = symbol_count + 2
        if not (
            len(self.input_ids)
            == len(self.token_to_atom)
            == len(self.token_role)
            == token_count
        ):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if any(not _is_plain_int(token_id) or token_id < 0 for token_id in self.input_ids):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if self.token_to_atom != (-1, *self.symbol_to_model_atom, -1):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if self.token_role != (BOUNDARY_ROLE, *self.symbol_role, BOUNDARY_ROLE):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if any(role not in ALLOWED_TOKEN_ROLES for role in self.token_role):
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")

        if len(self.atom_identity_spans) != atom_count or len(self.atom_to_carrier) != atom_count:
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        if len(set(self.atom_to_carrier)) != atom_count:
            _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
        for atom_id, span in enumerate(self.atom_identity_spans):
            if not isinstance(span, Span) or span.stop != span.start + 1:
                _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
            if self.atom_to_carrier[atom_id] != span.start:
                _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
            if not 1 <= span.start <= symbol_count:
                _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")
            if self.token_to_atom[span.start] != atom_id or self.token_role[span.start] != ATOM_IDENTITY_ROLE:
                _reject("ALIGNMENT_INVARIANT_INVALID", "dataclass")


def _require_selfies_runtime(sf: Any) -> None:
    try:
        distribution_version = importlib_metadata.version("selfies")
    except importlib_metadata.PackageNotFoundError:
        _reject("SELFIES_VERSION_MISMATCH", "selfies_runtime")
    if distribution_version != SELFIES_DISTRIBUTION_VERSION:
        _reject("SELFIES_VERSION_MISMATCH", "selfies_runtime")
    try:
        constraints = dict(sf.get_semantic_constraints())
    except Exception as exc:
        raise AtomSelfiesAlignmentError(
            "SELFIES_CONSTRAINTS_UNAVAILABLE", "selfies_runtime"
        ) from exc
    if constraints != DEFAULT_SELFIES_211_CONSTRAINTS:
        _reject("SELFIES_CONSTRAINTS_MISMATCH", "selfies_runtime")


def _canonical_smiles_and_order(Chem: Any, projected_mol: Any) -> tuple[Any, str, tuple[int, ...]]:
    if projected_mol is None:
        _reject("PROJECTED_MOL_MISSING", "canonical_smiles")
    try:
        probe = Chem.Mol(projected_mol)
        Chem.SanitizeMol(probe)
        Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
        atom_count = int(probe.GetNumAtoms())
    except Exception as exc:
        raise AtomSelfiesAlignmentError("PROJECTED_MOL_INVALID", "canonical_smiles") from exc
    if atom_count <= 0:
        _reject("PROJECTED_MOL_EMPTY", "canonical_smiles")
    for atom in probe.GetAtoms():
        if int(atom.GetAtomicNum()) <= 1:
            _reject("PROJECTED_MOL_NON_HEAVY_ATOM", "canonical_smiles")
        if int(atom.GetAtomMapNum()) != 0:
            _reject("PROJECTED_MOL_ATOM_MAP_NOT_ALLOWED", "canonical_smiles")
    try:
        canonical = Chem.MolToSmiles(
            probe,
            canonical=True,
            isomericSmiles=True,
            kekuleSmiles=False,
        )
    except Exception as exc:
        raise AtomSelfiesAlignmentError("CANONICAL_SMILES_FAILED", "canonical_smiles") from exc
    if not isinstance(canonical, str) or not canonical:
        _reject("CANONICAL_SMILES_FAILED", "canonical_smiles")
    if not probe.HasProp("_smilesAtomOutputOrder"):
        _reject("SMILES_OUTPUT_ORDER_MISSING", "canonical_smiles")
    try:
        raw_order = ast.literal_eval(probe.GetProp("_smilesAtomOutputOrder"))
        order = tuple(int(value) for value in raw_order)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise AtomSelfiesAlignmentError("SMILES_OUTPUT_ORDER_INVALID", "canonical_smiles") from exc
    if len(order) != atom_count or sorted(order) != list(range(atom_count)):
        _reject("SMILES_OUTPUT_ORDER_INVALID", "canonical_smiles")
    return probe, canonical, order


def _call_selfies(
    sf: Any, canonical_smiles: str
) -> tuple[str, tuple[str, ...], str, tuple[Any, ...]]:
    try:
        selfies = sf.encoder(canonical_smiles, strict=True)
    except Exception as exc:
        raise AtomSelfiesAlignmentError("SELFIES_STRICT_ENCODE_FAILED", "selfies_encode") from exc
    if not isinstance(selfies, str) or not selfies:
        _reject("SELFIES_STRICT_ENCODE_FAILED", "selfies_encode")
    try:
        symbols = tuple(sf.split_selfies(selfies))
    except Exception as exc:
        raise AtomSelfiesAlignmentError("SELFIES_SPLIT_FAILED", "selfies_encode") from exc
    if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        _reject("SELFIES_SPLIT_FAILED", "selfies_encode")
    if "".join(symbols) != selfies:
        _reject("SELFIES_SPLIT_NOT_EXACT", "selfies_encode")
    try:
        decoded = sf.decoder(selfies, attribute=True)
    except Exception as exc:
        raise AtomSelfiesAlignmentError("SELFIES_DECODE_FAILED", "selfies_decode") from exc
    if not isinstance(decoded, tuple) or len(decoded) != 2:
        _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "selfies_decode")
    decoded_smiles, decoder_attribution = decoded
    if not isinstance(decoded_smiles, str) or not decoded_smiles or not isinstance(decoder_attribution, list):
        _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "selfies_decode")
    return (
        selfies,
        symbols,
        decoded_smiles,
        tuple(decoder_attribution),
    )


def _is_smiles_atom_token(token: Any) -> bool:
    if not isinstance(token, str) or not token:
        return False
    return token in _ORGANIC_ATOM_TOKENS or bool(_BRACKET_ATOM_RE.fullmatch(token))


def _iter_attribution(value: Any, reason_code: str, stage: str) -> tuple[Any, ...]:
    attribution = getattr(value, "attribution", None)
    if attribution is None:
        return ()
    if not isinstance(attribution, list):
        _reject(reason_code, stage)
    return tuple(attribution)


def _single_atom_selfies_symbol(Chem: Any, sf: Any, symbol: str) -> bool:
    if symbol == "." or _BRANCH_SYMBOL_RE.fullmatch(symbol) or _RING_SYMBOL_RE.fullmatch(symbol):
        return False
    try:
        decoded = sf.decoder(symbol)
    except Exception:
        return False
    if not isinstance(decoded, str) or not decoded or "." in decoded:
        return False
    try:
        atom_mol = Chem.MolFromSmiles(decoded)
    except Exception:
        return False
    return atom_mol is not None and atom_mol.GetNumAtoms() == 1 and atom_mol.GetNumBonds() == 0


def _attribution_atom_symbol_indices(
    Chem: Any,
    sf: Any,
    canonical_order: tuple[int, ...],
    symbols: tuple[str, ...],
    decoded_smiles: str,
    decoder_attribution: tuple[Any, ...],
) -> tuple[int, ...]:
    # SELFIES 2.1.1 decoder attribution indexes omit ``.`` separators and can
    # attribute branch/ring structure together with the atom-producing symbol.
    # Map that compressed domain back to the literal SELFIES surface, then
    # require exactly one chemically atom-producing symbol per decoded atom.
    compressed_symbol_indices: list[int] = []
    compressed_to_symbol_index: dict[int, int] = {}
    separators_seen = 0
    for symbol_index, symbol in enumerate(symbols):
        compressed_symbol_indices.append(symbol_index - separators_seen)
        if symbol == ".":
            separators_seen += 1
        else:
            compressed_to_symbol_index[compressed_symbol_indices[-1]] = symbol_index

    atom_decoder_maps: list[Any] = []
    previous_output_index = -1
    decoded_search_start = 0
    for item in decoder_attribution:
        output_index = getattr(item, "index", None)
        output_token = getattr(item, "token", None)
        if not _is_plain_int(output_index) or output_index < 0 or not isinstance(output_token, str) or not output_token:
            _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution")
        # Decoder attribution offsets omit inter-component dots in 2.1.1.
        # Locate output tokens monotonically in the actual decoded surface and
        # accept only the literal or dot-compressed end offset.
        actual_start = decoded_smiles.find(output_token, decoded_search_start)
        if actual_start < 0:
            _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution")
        actual_end = actual_start + len(output_token) - 1
        compressed_end = actual_end - decoded_smiles[: actual_end + 1].count(".")
        if output_index not in (actual_end, compressed_end):
            _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution")
        decoded_search_start = actual_end + 1
        if output_index <= previous_output_index:
            _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution")
        previous_output_index = output_index
        if _is_smiles_atom_token(output_token):
            atom_decoder_maps.append(item)
        elif output_token not in _DECODER_NONATOM_TOKENS:
            _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution")

    if len(atom_decoder_maps) != len(canonical_order):
        _reject("ATTRIBUTION_ATOM_COUNT_MISMATCH", "attribution")

    selected: list[int] = []
    for atom_map in atom_decoder_maps:
        candidates: list[int] = []
        for source in _iter_attribution(
            atom_map, "SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution"
        ):
            source_index = getattr(source, "index", None)
            source_token = getattr(source, "token", None)
            actual_symbol_index = (
                compressed_to_symbol_index.get(source_index)
                if _is_plain_int(source_index)
                else None
            )
            if (
                not _is_plain_int(source_index)
                or source_index < 0
                or actual_symbol_index is None
                or source_token != symbols[actual_symbol_index]
            ):
                _reject("SELFIES_DECODER_ATTRIBUTION_MALFORMED", "attribution")
            if _single_atom_selfies_symbol(Chem, sf, symbols[actual_symbol_index]):
                candidates.append(actual_symbol_index)
        if len(candidates) != 1:
            _reject("ATTRIBUTION_NOT_BIJECTIVE", "attribution")
        symbol_index = candidates[0]
        if symbol_index in selected:
            _reject("ATTRIBUTION_NOT_BIJECTIVE", "attribution")

        selected.append(symbol_index)

    if selected != sorted(selected):
        _reject("ATTRIBUTION_NOT_BIJECTIVE", "attribution")
    return tuple(selected)


def _structure_roles(symbols: tuple[str, ...], atom_symbol_indices: Iterable[int]) -> tuple[str, ...]:
    atom_indices = set(atom_symbol_indices)
    roles: list[str | None] = [None] * len(symbols)
    position = 0
    while position < len(symbols):
        if roles[position] is not None:
            position += 1
            continue
        if position in atom_indices:
            roles[position] = ATOM_IDENTITY_ROLE
            position += 1
            continue
        symbol = symbols[position]
        if symbol == ".":
            roles[position] = SEPARATOR_ROLE
            position += 1
            continue
        branch_match = _BRANCH_SYMBOL_RE.fullmatch(symbol)
        ring_match = _RING_SYMBOL_RE.fullmatch(symbol)
        if branch_match is not None:
            role = BRANCH_ROLE
            argument_count = int(branch_match.group(1))
        elif ring_match is not None:
            role = RING_ROLE
            argument_count = int(ring_match.group(1))
        else:
            _reject("SELFIES_STRUCTURE_SYMBOL_UNCLASSIFIED", "structure_roles")
        roles[position] = role
        argument_stop = position + 1 + argument_count
        if argument_stop > len(symbols):
            _reject("SELFIES_STRUCTURE_ARGUMENT_TRUNCATED", "structure_roles")
        for argument_position in range(position + 1, argument_stop):
            if argument_position in atom_indices or roles[argument_position] is not None:
                _reject("ATTRIBUTION_NOT_BIJECTIVE", "structure_roles")
            roles[argument_position] = role
        position += 1
    if any(role is None for role in roles):
        _reject("SELFIES_STRUCTURE_SYMBOL_UNCLASSIFIED", "structure_roles")
    return tuple(str(role) for role in roles)


def _atom_signature(atom: Any) -> tuple[Any, ...]:
    cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
    return (
        int(atom.GetAtomicNum()),
        int(atom.GetIsotope()),
        int(atom.GetFormalCharge()),
        int(atom.GetNumRadicalElectrons()),
        bool(atom.GetIsAromatic()),
        int(atom.GetNumExplicitHs()),
        int(atom.GetTotalNumHs(includeNeighbors=True)),
        cip,
    )


def _bond_table(mol: Any) -> dict[tuple[int, int], tuple[Any, ...]]:
    result: dict[tuple[int, int], tuple[Any, ...]] = {}
    for bond in mol.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        key = (min(left, right), max(left, right))
        result[key] = (
            str(bond.GetBondType()),
            bool(bond.GetIsAromatic()),
            str(bond.GetStereo()),
        )
    return result


def _validate_round_trip(
    Chem: Any,
    sf: Any,
    probe: Any,
    canonical_smiles: str,
    canonical_order: tuple[int, ...],
    selfies: str,
    decoded_smiles: str,
) -> None:
    try:
        decoded_mol = Chem.MolFromSmiles(decoded_smiles)
        if decoded_mol is None:
            _reject("SELFIES_DECODED_SMILES_INVALID", "round_trip")
        Chem.AssignStereochemistry(decoded_mol, cleanIt=True, force=True)
        roundtrip_canonical = Chem.MolToSmiles(
            decoded_mol,
            canonical=True,
            isomericSmiles=True,
            kekuleSmiles=False,
        )
    except AtomSelfiesAlignmentError:
        raise
    except Exception as exc:
        raise AtomSelfiesAlignmentError("SELFIES_DECODED_SMILES_INVALID", "round_trip") from exc
    if roundtrip_canonical != canonical_smiles:
        _reject("STRICT_ISOMERIC_ROUNDTRIP_MISMATCH", "round_trip")

    try:
        canonical_ordered = Chem.RenumberAtoms(probe, list(canonical_order))
        Chem.AssignStereochemistry(canonical_ordered, cleanIt=True, force=True)
        ordered_atom_signatures = tuple(_atom_signature(atom) for atom in canonical_ordered.GetAtoms())
        decoded_atom_signatures = tuple(_atom_signature(atom) for atom in decoded_mol.GetAtoms())
    except Exception as exc:
        raise AtomSelfiesAlignmentError("ORDERED_GRAPH_STEREO_MISMATCH", "round_trip") from exc
    if ordered_atom_signatures != decoded_atom_signatures or _bond_table(canonical_ordered) != _bond_table(decoded_mol):
        _reject("ORDERED_GRAPH_STEREO_MISMATCH", "round_trip")

    try:
        reencoded = sf.encoder(roundtrip_canonical, strict=True)
    except Exception as exc:
        raise AtomSelfiesAlignmentError("SELFIES_REENCODE_FAILED", "round_trip") from exc
    if reencoded != selfies:
        _reject("SELFIES_REENCODE_MISMATCH", "round_trip")


def _exact_token_id(tokenizer: Any, token: str, unk_token_id: int) -> int:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
    except Exception as exc:
        raise AtomSelfiesAlignmentError("UNION_TOKENIZER_API_FAILED", "tokenizer") from exc
    if not _is_plain_int(token_id) or token_id < 0 or token_id == unk_token_id:
        _reject("UNION_TOKENIZER_UNK", "tokenizer")
    if not isinstance(encoded, (list, tuple)) or tuple(encoded) != (token_id,):
        _reject("UNION_TOKENIZER_SYMBOL_NOT_EXACT", "tokenizer")
    try:
        reverse = tokenizer.convert_ids_to_tokens(token_id)
    except Exception as exc:
        raise AtomSelfiesAlignmentError("UNION_TOKENIZER_API_FAILED", "tokenizer") from exc
    if str(reverse) != token:
        _reject("UNION_TOKENIZER_SYMBOL_NOT_REVERSIBLE", "tokenizer")
    return token_id


def tokenizer_surface_for_selfies_symbol(symbol: str) -> str:
    """Return the frozen-tokenizer surface for one validated SELFIES symbol.

    SELFIES uses a raw ``.`` between disconnected components.  Registering
    that punctuation as an AddedToken would change ordinary T5 text
    tokenization, so the atom stream uses one opaque content token while the
    discovered chemistry surface retains the literal separator.
    """

    if not isinstance(symbol, str) or not symbol:
        _reject("SELFIES_SYMBOL_INVALID", "tokenizer_surface")
    return SELFIES_SEPARATOR_TOKEN if symbol == "." else symbol


def _tokenize_exact(tokenizer: Any, symbols: tuple[str, ...]) -> tuple[int, ...]:
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if not _is_plain_int(unk_token_id) or unk_token_id < 0:
        _reject("UNION_TOKENIZER_UNK_CONTRACT_MISSING", "tokenizer")
    molecular_surfaces = tuple(
        tokenizer_surface_for_selfies_symbol(symbol) for symbol in symbols
    )
    surfaces = (MOLECULE_BEGIN, *molecular_surfaces, MOLECULE_END)
    token_ids = tuple(_exact_token_id(tokenizer, token, unk_token_id) for token in surfaces)
    complete_surface = "".join(surfaces)
    try:
        complete_ids = tokenizer.encode(complete_surface, add_special_tokens=False)
    except Exception as exc:
        raise AtomSelfiesAlignmentError("UNION_TOKENIZER_API_FAILED", "tokenizer") from exc
    if not isinstance(complete_ids, (list, tuple)) or tuple(complete_ids) != token_ids:
        _reject("UNION_TOKENIZER_WHOLE_SURFACE_NOT_EXACT", "tokenizer")
    if unk_token_id in token_ids:
        _reject("UNION_TOKENIZER_UNK", "tokenizer")
    return token_ids


def discover_atom_selfies_surface(
    Chem: Any,
    sf: Any,
    projected_mol: Any,
) -> AtomSelfiesSurface:
    """Discover and validate the strict chemistry surface before vocab freeze.

    This is the first pass used by the bounded paired builder: it exposes only
    validated symbols and atom attribution, and performs no token lookup.
    """

    _require_selfies_runtime(sf)
    probe, canonical_smiles, canonical_order = _canonical_smiles_and_order(Chem, projected_mol)
    (
        selfies,
        symbols,
        decoded_smiles,
        decoder_attribution,
    ) = _call_selfies(sf, canonical_smiles)
    atom_symbol_indices = _attribution_atom_symbol_indices(
        Chem,
        sf,
        canonical_order,
        symbols,
        decoded_smiles,
        decoder_attribution,
    )
    roles = _structure_roles(symbols, atom_symbol_indices)
    _validate_round_trip(
        Chem,
        sf,
        probe,
        canonical_smiles,
        canonical_order,
        selfies,
        decoded_smiles,
    )
    symbol_to_model = [-1] * len(symbols)
    for canonical_position, symbol_index in enumerate(atom_symbol_indices):
        symbol_to_model[symbol_index] = canonical_order[canonical_position]
    symbol_to_model_tuple = tuple(symbol_to_model)

    atom_count = len(canonical_order)
    atom_symbol_position = [-1] * atom_count
    for symbol_index, model_atom_id in enumerate(symbol_to_model_tuple):
        if model_atom_id >= 0:
            if atom_symbol_position[model_atom_id] != -1:
                _reject("ATTRIBUTION_NOT_BIJECTIVE", "alignment_build")
            atom_symbol_position[model_atom_id] = symbol_index
    if any(position < 0 for position in atom_symbol_position):
        _reject("ATTRIBUTION_NOT_BIJECTIVE", "alignment_build")
    return AtomSelfiesSurface(
        canonical_isomeric_smiles=canonical_smiles,
        canonical_position_to_model_atom=canonical_order,
        selfies=selfies,
        selfies_symbols=symbols,
        symbol_to_model_atom=symbol_to_model_tuple,
        symbol_role=roles,
    )


def derive_atom_selfies_alignment(
    Chem: Any,
    sf: Any,
    projected_mol: Any,
    union_tokenizer: Any,
) -> AtomSelfiesAlignment:
    """Bind one discovered strict SELFIES surface to the frozen tokenizer."""

    surface = discover_atom_selfies_surface(Chem, sf, projected_mol)
    return bind_atom_selfies_surface(surface, union_tokenizer)


def bind_atom_selfies_surface(
    surface: AtomSelfiesSurface,
    union_tokenizer: Any,
) -> AtomSelfiesAlignment:
    """Bind a previously validated chemistry surface to one frozen tokenizer."""

    if not isinstance(surface, AtomSelfiesSurface):
        _reject("SURFACE_TYPE_INVALID", "tokenizer")
    input_ids = _tokenize_exact(union_tokenizer, surface.selfies_symbols)
    atom_count = len(surface.canonical_position_to_model_atom)
    atom_symbol_position = [-1] * atom_count
    for symbol_index, model_atom_id in enumerate(surface.symbol_to_model_atom):
        if model_atom_id >= 0:
            atom_symbol_position[model_atom_id] = symbol_index
    carriers = tuple(position + 1 for position in atom_symbol_position)
    spans = tuple(Span(carrier, carrier + 1) for carrier in carriers)

    return AtomSelfiesAlignment(
        canonical_isomeric_smiles=surface.canonical_isomeric_smiles,
        canonical_position_to_model_atom=surface.canonical_position_to_model_atom,
        selfies=surface.selfies,
        selfies_symbols=surface.selfies_symbols,
        symbol_to_model_atom=surface.symbol_to_model_atom,
        symbol_role=surface.symbol_role,
        input_ids=input_ids,
        token_to_atom=(-1, *surface.symbol_to_model_atom, -1),
        token_role=(BOUNDARY_ROLE, *surface.symbol_role, BOUNDARY_ROLE),
        atom_identity_spans=spans,
        atom_to_carrier=carriers,
    )


__all__ = [
    "ALLOWED_SYMBOL_ROLES",
    "ATOM_IDENTITY_ROLE",
    "AtomSelfiesAlignment",
    "AtomSelfiesAlignmentError",
    "AtomSelfiesSurface",
    "BOUNDARY_ROLE",
    "BRANCH_ROLE",
    "MOLECULE_BEGIN",
    "MOLECULE_END",
    "RING_ROLE",
    "SEPARATOR_ROLE",
    "SELFIES_SEPARATOR_TOKEN",
    "SELFIES_DISTRIBUTION_VERSION",
    "bind_atom_selfies_surface",
    "discover_atom_selfies_surface",
    "derive_atom_selfies_alignment",
    "tokenizer_surface_for_selfies_symbol",
]
