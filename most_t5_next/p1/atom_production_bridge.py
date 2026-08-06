"""Strict atom/SELFIES production boundary for the A0/A1 grid cells.

3D-MolT5 motivates the atom-level baseline: SELFIES supplies an atom-aligned
identity carrier and the released E3FP rows supply atom geometry.  This module
does not reuse or reinterpret a motif record.  Instead it requires the
tokenizer-bound atom/SELFIES mapping to be explicit and sends A1 through the
same :mod:`shared_geometry_fusion` sidecar used by M1.

The still-missing chemistry producer must prove topology -> SELFIES -> shared
union-tokenizer IDs and populate :class:`ProductionAtomSelfiesRecord`.  This
module deliberately cannot infer that mapping from a SMILES, motif record, or
token sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

from .bound_record import Span
from .ce_collator import IDENTITY_RECOVERY_OBJECTIVE, _select_logical_motifs
from .experiment_grid import (
    FourGridContractError,
    GeometryBatchSidecar,
    P1ConditionBatch,
    get_p1_condition_spec,
)
from .production_bridge import ProductionTokenizerRuntime
from .runtime_bridge import PaddedCEBatch, pad_ce_first_batch


ATOM_IDENTITY_ROLE = "atom_identity"
IDENTITY_SENTINEL_ROLE = "identity_sentinel"
ATOM_SELFIES_RECORD_SCHEMA = "most-t5-next/p1-atom-selfies-production-record/v1"
ATOM_MASK_DECISION_SCHEMA = "most-t5-next/atom-identity-mask-decision/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# SELFIES control symbols are retained rather than treated as atom identity.
# ``other_structure`` is explicit, not an implicit fallback for unknown roles.
ALLOWED_UNCORRUPTED_TOKEN_ROLES = frozenset(
    {
        ATOM_IDENTITY_ROLE,
        "branch",
        "ring",
        "bond",
        "stereo",
        "boundary",
        "separator",
        "other_structure",
    }
)


class AtomProductionBridgeError(ValueError):
    """An atom/SELFIES record cannot enter the A0/A1 comparison."""


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AtomProductionBridgeError(field + " must be a lower-case SHA-256")


def _require_nonempty_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AtomProductionBridgeError(field + " must be nonempty")


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class ProductionAtomSelfiesRecord:
    """Once-validated atom/SELFIES row bound to the shared union tokenizer.

    ``atom_identity_spans[atom_id]`` is the complete union-tokenizer span for
    one SELFIES atom identity.  It may contain multiple sub-tokens.  Each atom
    has exactly one unique carrier, fixed to the first token of that span.
    Branch, ring, boundary and other declared structure roles lie outside all
    identity spans and survive CE corruption unchanged.

    Atom ID is also the post-hydrogen-projection geometry model-row ID.
    ``model_to_source_atom_index[atom_id]`` must be copied from the same PCQM
    geometry record as ``full_e3fp_ids[atom_id]``.  The future producer must
    prove that SELFIES carrier order follows this mapping; this bridge never
    guesses the correspondence from heavy-atom count or token position.

    ``selfies`` is retained for audit/provenance.  Verifying that it tokenizes
    to ``input_ids`` belongs to the future topology-to-SELFIES producer and is
    intentionally not simulated here.
    """

    record_artifact_sha256: str
    record_id: str
    storage_key: str
    release_id: str
    geometry_record_content_sha256: str
    union_tokenizer_contract_sha256: str
    union_tokenizer_snapshot_sha256: str
    selfies: str
    input_ids: tuple[int, ...]
    token_to_atom: tuple[int, ...]
    token_role: tuple[str, ...]
    atom_identity_spans: tuple[Span, ...]
    atom_to_carrier: tuple[int, ...]
    source_atom_count: int
    model_to_source_atom_index: tuple[int, ...]
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    atom_valid_mask: tuple[bool, ...]
    schema_version: str = ATOM_SELFIES_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ATOM_SELFIES_RECORD_SCHEMA:
            raise AtomProductionBridgeError("unknown atom/SELFIES record schema")
        for field in (
            "record_artifact_sha256",
            "geometry_record_content_sha256",
            "union_tokenizer_contract_sha256",
            "union_tokenizer_snapshot_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        for field in ("record_id", "storage_key", "release_id", "selfies"):
            _require_nonempty_text(getattr(self, field), field)

        tuple_fields = (
            "input_ids",
            "token_to_atom",
            "token_role",
            "atom_identity_spans",
            "atom_to_carrier",
            "model_to_source_atom_index",
            "full_e3fp_ids",
            "atom_valid_mask",
        )
        if any(not isinstance(getattr(self, field), tuple) for field in tuple_fields):
            raise AtomProductionBridgeError(
                "production atom arrays must be immutable tuples"
            )
        if any(not isinstance(row, tuple) for row in self.full_e3fp_ids):
            raise AtomProductionBridgeError("E3FP atom rows must be immutable tuples")

        token_count = len(self.input_ids)
        if token_count == 0:
            raise AtomProductionBridgeError("input_ids cannot be empty")
        if any(not _is_nonnegative_int(token_id) for token_id in self.input_ids):
            raise AtomProductionBridgeError("input_ids must be nonnegative integers")
        if len(self.token_to_atom) != token_count or len(self.token_role) != token_count:
            raise AtomProductionBridgeError("atom token-domain arrays disagree")
        if any(
            not isinstance(role, str) or role not in ALLOWED_UNCORRUPTED_TOKEN_ROLES
            for role in self.token_role
        ):
            raise AtomProductionBridgeError("token_role contains an undeclared role")

        atom_count = len(self.atom_identity_spans)
        if atom_count == 0:
            raise AtomProductionBridgeError("an atom/SELFIES row needs at least one atom")
        parallel_atom_lengths = (
            len(self.atom_to_carrier),
            len(self.model_to_source_atom_index),
            len(self.full_e3fp_ids),
            len(self.atom_valid_mask),
        )
        if any(length != atom_count for length in parallel_atom_lengths):
            raise AtomProductionBridgeError("atom-domain arrays disagree")
        if any(type(active) is not bool for active in self.atom_valid_mask):
            raise AtomProductionBridgeError("atom_valid_mask must contain Boolean values")
        if self.atom_valid_mask != (True,) * atom_count:
            raise AtomProductionBridgeError(
                "narrow A0/A1 requires geometry for every real atom"
            )
        if any(not _is_nonnegative_int(carrier) for carrier in self.atom_to_carrier):
            raise AtomProductionBridgeError("atom carriers must be nonnegative integers")
        if len(set(self.atom_to_carrier)) != atom_count:
            raise AtomProductionBridgeError("every atom must have a unique carrier token")
        if not _is_nonnegative_int(self.source_atom_count) or self.source_atom_count == 0:
            raise AtomProductionBridgeError("source_atom_count must be a positive integer")
        if (
            any(
                not _is_nonnegative_int(source_index)
                or source_index >= self.source_atom_count
                for source_index in self.model_to_source_atom_index
            )
            or self.model_to_source_atom_index
            != tuple(sorted(set(self.model_to_source_atom_index)))
        ):
            raise AtomProductionBridgeError(
                "model_to_source_atom_index must be strictly increasing and in source range"
            )
        if any(
            isinstance(atom_id, bool)
            or not isinstance(atom_id, int)
            or atom_id < -1
            or atom_id >= atom_count
            for atom_id in self.token_to_atom
        ):
            raise AtomProductionBridgeError("token_to_atom is outside the atom domain")

        covered_positions: set[int] = set()
        for atom_id, span in enumerate(self.atom_identity_spans):
            if not isinstance(span, Span):
                raise AtomProductionBridgeError("atom_identity_spans must contain Span values")
            if not 0 <= span.start < span.stop <= token_count:
                raise AtomProductionBridgeError("atom identity span is outside input_ids")
            if self.atom_to_carrier[atom_id] != span.start:
                raise AtomProductionBridgeError(
                    "atom carrier must be the first token of its identity span"
                )
            positions = set(range(span.start, span.stop))
            if covered_positions.intersection(positions):
                raise AtomProductionBridgeError("atom identity spans overlap")
            covered_positions.update(positions)
            for position in positions:
                if self.token_to_atom[position] != atom_id:
                    raise AtomProductionBridgeError(
                        "identity-span token maps to another atom"
                    )
                if self.token_role[position] != ATOM_IDENTITY_ROLE:
                    raise AtomProductionBridgeError(
                        "identity span contains a non-identity structure token"
                    )

        for position in range(token_count):
            if position in covered_positions:
                continue
            if self.token_to_atom[position] != -1:
                raise AtomProductionBridgeError(
                    "tokens outside identity spans cannot act as atom carriers"
                )
            if self.token_role[position] == ATOM_IDENTITY_ROLE:
                raise AtomProductionBridgeError(
                    "every atom-identity token must belong to exactly one span"
                )

        level_counts = {len(row) for row in self.full_e3fp_ids}
        if len(level_counts) != 1 or 0 in level_counts:
            raise AtomProductionBridgeError("E3FP must be rectangular [atom, level]")
        for row in self.full_e3fp_ids:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -1
                or value > 4095
                for value in row
            ) or row[0] < 0:
                raise AtomProductionBridgeError("active E3FP row is outside the narrow domain")


@dataclass(frozen=True)
class ProductionAtomCEExample:
    """One epoch-keyed atom-identity CE realization for both A0 and A1."""

    record_id: str
    storage_key: str
    objective: str
    seed: int
    epoch: int
    mask_probability: float
    mask_decision_sha256: str
    geometry_record_content_sha256: str
    union_tokenizer_snapshot_sha256: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    input_token_to_atom: tuple[int, ...]
    input_token_role: tuple[str, ...]
    identity_recovery_mask: tuple[bool, ...]
    selected_atom_ids_in_input_order: tuple[int, ...]
    atom_identity_input_spans: tuple[Span, ...]
    atom_to_carrier: tuple[int, ...]
    model_to_source_atom_index: tuple[int, ...]
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    atom_valid_mask: tuple[bool, ...]


def _validate_runtime_key(seed: int, epoch: int, mask_probability: float) -> None:
    if not _is_nonnegative_int(seed) or not _is_nonnegative_int(epoch):
        raise AtomProductionBridgeError("seed and epoch must be nonnegative integers")
    if (
        isinstance(mask_probability, bool)
        or not isinstance(mask_probability, (int, float))
        or not math.isfinite(float(mask_probability))
        or not 0.0 < float(mask_probability) <= 1.0
    ):
        raise AtomProductionBridgeError("mask_probability must be finite and in (0, 1]")


def _atom_mask_decision_sha256(
    *,
    seed: int,
    epoch: int,
    record_id: str,
    objective: str,
    mask_probability: float,
    selected_atom_ids: Sequence[int],
) -> str:
    payload = {
        "epoch": epoch,
        "mask_probability": mask_probability,
        "objective": objective,
        "record_id": record_id,
        "schema": ATOM_MASK_DECISION_SCHEMA,
        "seed": seed,
        "selected_atom_ids": list(selected_atom_ids),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def collate_production_atom_record(
    record: ProductionAtomSelfiesRecord,
    *,
    tokenizer: ProductionTokenizerRuntime,
    seed: int,
    epoch: int,
    mask_probability: float = 0.15,
) -> ProductionAtomCEExample:
    """Corrupt complete atom identity spans with the shared stateless rule."""

    if not isinstance(record, ProductionAtomSelfiesRecord):
        raise AtomProductionBridgeError(
            "record must be an independently validated ProductionAtomSelfiesRecord"
        )
    if not isinstance(tokenizer, ProductionTokenizerRuntime):
        raise AtomProductionBridgeError("tokenizer must be a ProductionTokenizerRuntime")
    _validate_runtime_key(seed, epoch, mask_probability)
    if record.union_tokenizer_contract_sha256 != tokenizer.tokenizer_contract_sha256:
        raise AtomProductionBridgeError("union tokenizer contract hash differs from the record")
    if record.union_tokenizer_snapshot_sha256 != tokenizer.tokenizer_snapshot_sha256:
        raise AtomProductionBridgeError("union tokenizer snapshot hash differs from the record")
    if any(token_id >= tokenizer.vocab_size for token_id in record.input_ids):
        raise AtomProductionBridgeError("atom/SELFIES input ID is outside the union vocabulary")
    if set(record.input_ids).intersection(tokenizer.sentinel_token_ids):
        raise AtomProductionBridgeError("sentinel occurs in uncorrupted atom/SELFIES input")

    # Reuse the exact seed/epoch/record/objective/unit-ID score rule and the
    # deterministic nonempty gate used by motif CE.  Here the identity unit ID
    # is an atom ID rather than a logical motif ID.
    mask = _select_logical_motifs(
        seed=seed,
        epoch=epoch,
        record_id=record.record_id,
        objective=IDENTITY_RECOVERY_OBJECTIVE,
        mask_probability=float(mask_probability),
        motif_count=len(record.atom_identity_spans),
    )
    selected_in_input_order = tuple(
        sorted(
            (atom_id for atom_id, selected in enumerate(mask) if selected),
            key=lambda atom_id: record.atom_identity_spans[atom_id].start,
        )
    )
    sentinels = tokenizer.sentinel_token_ids
    if len(sentinels) < len(selected_in_input_order) + 1:
        raise AtomProductionBridgeError(
            "selected atom count plus one terminal sentinel is required"
        )

    selected_by_start = {
        record.atom_identity_spans[atom_id].start: (target_index, atom_id)
        for target_index, atom_id in enumerate(selected_in_input_order)
    }
    corrupted_ids: list[int] = []
    corrupted_to_atom: list[int] = []
    corrupted_roles: list[str] = []
    original_position = 0
    while original_position < len(record.input_ids):
        selected = selected_by_start.get(original_position)
        if selected is None:
            corrupted_ids.append(record.input_ids[original_position])
            corrupted_to_atom.append(record.token_to_atom[original_position])
            corrupted_roles.append(record.token_role[original_position])
            original_position += 1
            continue
        target_index, atom_id = selected
        corrupted_ids.append(sentinels[target_index])
        corrupted_to_atom.append(atom_id)
        corrupted_roles.append(IDENTITY_SENTINEL_ROLE)
        original_position = record.atom_identity_spans[atom_id].stop

    reductions = tuple(
        (
            record.atom_identity_spans[atom_id].stop,
            record.atom_identity_spans[atom_id].stop
            - record.atom_identity_spans[atom_id].start
            - 1,
        )
        for atom_id in selected_in_input_order
    )

    def transform_boundary(boundary: int) -> int:
        return boundary - sum(
            reduction for stop, reduction in reductions if stop <= boundary
        )

    transformed_spans: list[Span] = []
    atom_to_carrier: list[int] = []
    for atom_id, span in enumerate(record.atom_identity_spans):
        start = transform_boundary(span.start)
        stop = start + 1 if mask[atom_id] else transform_boundary(span.stop)
        transformed_spans.append(Span(start, stop))
        atom_to_carrier.append(start)

    labels: list[int] = []
    for target_index, atom_id in enumerate(selected_in_input_order):
        span = record.atom_identity_spans[atom_id]
        labels.extend((sentinels[target_index], *record.input_ids[span.start : span.stop]))
    labels.extend((sentinels[len(selected_in_input_order)], tokenizer.eos_token_id))

    selected_atom_ids = tuple(atom_id for atom_id, selected in enumerate(mask) if selected)
    example = ProductionAtomCEExample(
        record_id=record.record_id,
        storage_key=record.storage_key,
        objective=IDENTITY_RECOVERY_OBJECTIVE,
        seed=seed,
        epoch=epoch,
        mask_probability=float(mask_probability),
        mask_decision_sha256=_atom_mask_decision_sha256(
            seed=seed,
            epoch=epoch,
            record_id=record.record_id,
            objective=IDENTITY_RECOVERY_OBJECTIVE,
            mask_probability=float(mask_probability),
            selected_atom_ids=selected_atom_ids,
        ),
        geometry_record_content_sha256=record.geometry_record_content_sha256,
        union_tokenizer_snapshot_sha256=record.union_tokenizer_snapshot_sha256,
        input_ids=tuple(corrupted_ids),
        labels=tuple(labels),
        input_token_to_atom=tuple(corrupted_to_atom),
        input_token_role=tuple(corrupted_roles),
        identity_recovery_mask=mask,
        selected_atom_ids_in_input_order=selected_in_input_order,
        atom_identity_input_spans=tuple(transformed_spans),
        atom_to_carrier=tuple(atom_to_carrier),
        model_to_source_atom_index=record.model_to_source_atom_index,
        full_e3fp_ids=record.full_e3fp_ids,
        atom_valid_mask=record.atom_valid_mask,
    )
    _validate_atom_example(example, record, sentinels, tokenizer.eos_token_id)
    return example


def _validate_atom_example(
    example: ProductionAtomCEExample,
    record: ProductionAtomSelfiesRecord,
    sentinels: Sequence[int],
    eos_token_id: int,
) -> None:
    if not example.labels or example.labels[-1] != eos_token_id:
        raise AtomProductionBridgeError("CE labels must terminate with EOS")
    terminal_index = len(example.selected_atom_ids_in_input_order)
    if example.labels[-2] != sentinels[terminal_index]:
        raise AtomProductionBridgeError("CE labels lack the terminal sentinel")
    if not (
        len(example.input_ids)
        == len(example.input_token_to_atom)
        == len(example.input_token_role)
    ):
        raise AtomProductionBridgeError("corrupted token-domain arrays disagree")
    if len(set(example.atom_to_carrier)) != len(example.atom_to_carrier):
        raise AtomProductionBridgeError("A1 requires one atom per carrier")
    if len(example.model_to_source_atom_index) != len(example.full_e3fp_ids):
        raise AtomProductionBridgeError(
            "SELFIES carriers and post-projection geometry rows disagree"
        )
    for atom_id, span in enumerate(example.atom_identity_input_spans):
        carrier = example.atom_to_carrier[atom_id]
        if carrier != span.start or example.input_token_to_atom[carrier] != atom_id:
            raise AtomProductionBridgeError("post-corruption atom carrier is inconsistent")
        expected_role = (
            IDENTITY_SENTINEL_ROLE
            if example.identity_recovery_mask[atom_id]
            else ATOM_IDENTITY_ROLE
        )
        if example.input_token_role[carrier] != expected_role:
            raise AtomProductionBridgeError("post-corruption carrier role is inconsistent")

    # Every non-identity token is transformed positionally but preserved in
    # value and role.  This is the executable branch/ring/boundary guarantee.
    reductions = tuple(
        (
            record.atom_identity_spans[atom_id].stop,
            record.atom_identity_spans[atom_id].stop
            - record.atom_identity_spans[atom_id].start
            - 1,
        )
        for atom_id in example.selected_atom_ids_in_input_order
    )
    for original_position, role in enumerate(record.token_role):
        if role == ATOM_IDENTITY_ROLE:
            continue
        transformed_position = original_position - sum(
            reduction for stop, reduction in reductions if stop <= original_position
        )
        if (
            example.input_ids[transformed_position] != record.input_ids[original_position]
            or example.input_token_role[transformed_position] != role
            or example.input_token_to_atom[transformed_position] != -1
        ):
            raise AtomProductionBridgeError("SELFIES structure token changed during corruption")


def _build_atom_geometry_sidecar(
    examples: Sequence[ProductionAtomCEExample],
    ce_batch: PaddedCEBatch,
) -> GeometryBatchSidecar:
    atom_width = max(len(example.full_e3fp_ids) for example in examples)
    level_counts = {len(example.full_e3fp_ids[0]) for example in examples}
    if len(level_counts) != 1:
        raise AtomProductionBridgeError("E3FP level count must be common within a batch")
    level_count = next(iter(level_counts))
    padded_ids = []
    padded_masks = []
    padded_carriers = []
    padded_source_indices = []
    atom_lengths = []
    for example in examples:
        atom_length = len(example.full_e3fp_ids)
        pad_count = atom_width - atom_length
        atom_lengths.append(atom_length)
        padded_ids.append(example.full_e3fp_ids + ((-1,) * level_count,) * pad_count)
        padded_masks.append(example.atom_valid_mask + (False,) * pad_count)
        padded_carriers.append(example.atom_to_carrier + (-1,) * pad_count)
        padded_source_indices.append(
            example.model_to_source_atom_index + (-1,) * pad_count
        )
    try:
        return GeometryBatchSidecar(
            record_ids=tuple(example.record_id for example in examples),
            e3fp_ids=tuple(padded_ids),
            e3fp_atom_mask=tuple(padded_masks),
            e3fp_atom_to_token=tuple(padded_carriers),
            model_to_source_atom_index=tuple(padded_source_indices),
            atom_lengths=tuple(atom_lengths),
            e3fp_level_count=level_count,
            token_width=len(ce_batch.input_ids[0]),
        )
    except FourGridContractError as exc:
        raise AtomProductionBridgeError("atom geometry sidecar is inconsistent") from exc


def collate_production_atom_batch(
    records: Sequence[ProductionAtomSelfiesRecord],
    *,
    condition_id: str,
    tokenizer: ProductionTokenizerRuntime,
    seed: int,
    epoch: int,
    mask_probability: float = 0.15,
) -> P1ConditionBatch:
    """Build A0 or A1; CE is identical and only A1 exposes geometry."""

    try:
        spec = get_p1_condition_spec(condition_id)
    except FourGridContractError as exc:
        raise AtomProductionBridgeError("unknown four-grid condition") from exc
    if spec.record_family != "atom_selfies":
        raise AtomProductionBridgeError(
            "M0/M1 require logical-motif records; atom records cannot be reinterpreted"
        )
    rows = tuple(records)
    if not rows:
        raise AtomProductionBridgeError("an atom production batch cannot be empty")
    examples = tuple(
        collate_production_atom_record(
            record,
            tokenizer=tokenizer,
            seed=seed,
            epoch=epoch,
            mask_probability=mask_probability,
        )
        for record in rows
    )
    ce_batch = pad_ce_first_batch(examples, pad_token_id=tokenizer.pad_token_id)
    geometry = _build_atom_geometry_sidecar(examples, ce_batch) if spec.uses_geometry else None
    try:
        return P1ConditionBatch(condition_id, ce_batch, geometry)
    except FourGridContractError as exc:
        raise AtomProductionBridgeError("atom condition batch is inconsistent") from exc
