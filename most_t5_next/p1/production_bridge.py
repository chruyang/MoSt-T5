"""Thin production-record to CE-batch bridge for the motif grid cells.

The input is the existing hash-bound
``p1-logical-motif-training-record/vnext1`` audit document.  Unlike the
synthetic codec path, this bridge never tokenizes a molecule or infers an
atom/motif/token mapping.  It validates the closed production contract, applies
its declared whole-identity mask, and emits the same ordinary T5 CE triplet
used by the smoke path.  M1 additionally receives an explicit E3FP atom-to-
carrier sidecar; M0 drops geometry before the model boundary.

The sidecar also retains the validated post-hydrogen-projection
``model_to_source_atom_index`` beside the E3FP rows.  It is audit provenance,
not a model input, and permits an explicit A1/M1 atom-subset parity gate.

A0/A1 require an atom/SELFIES-bound production record and are intentionally
not fabricated from the logical-motif contract here.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from most_t5_next.r1.gates import validate_p1_logical_motif_vnext as validator

from .bound_record import Span
from .ce_collator import (
    IDENTITY_RECOVERY_OBJECTIVE,
    _mask_decision_sha256,
    _select_logical_motifs,
)
from .experiment_grid import (
    FourGridContractError,
    GeometryBatchSidecar,
    P1ConditionBatch,
    get_p1_condition_spec,
)
from .runtime_bridge import PaddedCEBatch, pad_ce_first_batch


IDENTITY_SENTINEL_ROLE = "identity_sentinel"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProductionBridgeError(ValueError):
    """A tokenizer-bound production record cannot enter the CE mainline."""


@dataclass(frozen=True)
class ProductionTokenizerRuntime:
    """Frozen tokenizer identity plus the special IDs used by CE corruption."""

    tokenizer_contract_sha256: str
    tokenizer_snapshot_sha256: str
    vocab_size: int
    pad_token_id: int
    eos_token_id: int
    sentinel_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for field in ("tokenizer_contract_sha256", "tokenizer_snapshot_sha256"):
            if not isinstance(getattr(self, field), str) or not SHA256_RE.fullmatch(
                getattr(self, field)
            ):
                raise ProductionBridgeError(field + " must be a lower-case SHA-256")
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ProductionBridgeError("vocab_size must be a positive integer")
        sentinels = _validate_sentinel_contract(self.sentinel_token_ids, self.eos_token_id)
        object.__setattr__(self, "sentinel_token_ids", sentinels)
        if (
            isinstance(self.pad_token_id, bool)
            or not isinstance(self.pad_token_id, int)
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ProductionBridgeError("pad_token_id must be inside the tokenizer vocabulary")
        if self.eos_token_id >= self.vocab_size or any(
            token_id >= self.vocab_size for token_id in sentinels
        ):
            raise ProductionBridgeError("EOS and sentinel IDs must be inside the tokenizer vocabulary")
        if self.pad_token_id == self.eos_token_id or self.pad_token_id in sentinels:
            raise ProductionBridgeError("padding must be distinct from EOS and sentinels")


@dataclass(frozen=True)
class ProductionCEExample:
    """One corrupted motif example plus the mappings needed by M1."""

    record_id: str
    storage_key: str
    objective: str
    seed: int
    epoch: int
    mask_probability: float
    mask_decision_sha256: str
    geometry_record_content_sha256: str
    tokenizer_snapshot_sha256: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    input_token_to_logical_motif: tuple[int, ...]
    input_token_role: tuple[str, ...]
    identity_recovery_mask: tuple[bool, ...]
    selected_logical_motif_ids_in_input_order: tuple[int, ...]
    identity_input_spans: tuple[Span, ...]
    connection_input_indices: tuple[tuple[int, ...], ...]
    logical_to_carrier: tuple[int, ...]
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    atom_valid_mask: tuple[bool, ...]
    model_to_source_atom_index: tuple[int, ...]
    atom_to_logical_motif: tuple[int, ...]
    atom_to_carrier: tuple[int, ...]
    atom_is_attachment: tuple[bool, ...] = ()


@dataclass(frozen=True)
class ProductionMotifRecord:
    """Once-validated, immutable Dataset row used across training epochs."""

    record_artifact_sha256: str
    record_id: str
    storage_key: str
    release_id: str
    geometry_record_content_sha256: str
    tokenizer_contract_sha256: str
    tokenizer_snapshot_sha256: str
    input_ids: tuple[int, ...]
    token_to_logical_motif: tuple[int, ...]
    token_role: tuple[str, ...]
    identity_spans: tuple[Span, ...]
    connection_token_indices: tuple[tuple[int, ...], ...]
    logical_to_carrier: tuple[int, ...]
    exact_identity_sha256: tuple[str, ...]
    source_atom_count: int
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    atom_valid_mask: tuple[bool, ...]
    model_to_source_atom_index: tuple[int, ...]
    atom_to_logical_motif: tuple[int, ...]
    atom_is_attachment: tuple[bool, ...] = ()


def _validate_sentinel_contract(
    sentinel_token_ids: Sequence[int], eos_token_id: int
) -> tuple[int, ...]:
    sentinels = tuple(sentinel_token_ids)
    if not sentinels:
        raise ProductionBridgeError("sentinel_token_ids cannot be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in sentinels
    ):
        raise ProductionBridgeError("sentinel IDs must be nonnegative integers")
    if len(set(sentinels)) != len(sentinels):
        raise ProductionBridgeError("sentinel IDs must be unique")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ProductionBridgeError("eos_token_id must be a nonnegative integer")
    if eos_token_id in sentinels:
        raise ProductionBridgeError("EOS must be distinct from sentinel IDs")
    return sentinels


def _raise_failed_record(report: Mapping[str, object]) -> None:
    errors = report.get("errors", [])
    preview = []
    if isinstance(errors, list):
        for error in errors[:3]:
            if isinstance(error, dict):
                preview.append(
                    "{}: {}".format(error.get("path", "?"), error.get("message", "invalid"))
                )
    suffix = "; ".join(preview) if preview else "unknown contract error"
    raise ProductionBridgeError("production training record failed validation: " + suffix)


def load_production_motif_record(
    document: Mapping[str, object],
) -> ProductionMotifRecord:
    """Validate once at the Dataset boundary and extract immutable model data."""

    if not isinstance(document, dict):
        raise ProductionBridgeError("production training record must be one JSON object")
    report = validator.validate_training_record(document)
    if report.get("pass") is not True:
        _raise_failed_record(report)
    if document["training_profile"] != validator.CE_PROFILE:
        raise ProductionBridgeError("the production mainline accepts ce_first only")

    token_domain = document["token_domain"]
    motif_domain = document["logical_motif_domain"]
    atom_domain = document["atom_domain"]
    return ProductionMotifRecord(
        record_artifact_sha256=report["artifact_sha256"],
        record_id=document["member"]["member_id"],
        storage_key=document["member"]["storage_key"],
        release_id=document["bindings"]["release_id"],
        geometry_record_content_sha256=document["bindings"]["geometry_record_content_sha256"],
        tokenizer_contract_sha256=document["bindings"]["tokenizer_contract_sha256"],
        tokenizer_snapshot_sha256=document["bindings"]["tokenizer_snapshot_sha256"],
        input_ids=tuple(token_domain["input_ids"]),
        token_to_logical_motif=tuple(token_domain["token_to_logical_motif"]),
        token_role=tuple(token_domain["token_role"]),
        identity_spans=tuple(Span(*row) for row in motif_domain["identity_spans"]),
        connection_token_indices=tuple(
            tuple(row) for row in motif_domain["connection_token_indices"]
        ),
        logical_to_carrier=tuple(motif_domain["logical_to_carrier"]),
        exact_identity_sha256=tuple(motif_domain["exact_identity_sha256"]),
        source_atom_count=document["dimensions"]["source_atom_count"],
        full_e3fp_ids=tuple(tuple(row) for row in atom_domain["full_e3fp_ids"]),
        atom_valid_mask=tuple(atom_domain["atom_valid_mask"]),
        model_to_source_atom_index=tuple(atom_domain["model_to_source_atom_index"]),
        atom_to_logical_motif=tuple(atom_domain["atom_to_logical_motif"]),
        atom_is_attachment=tuple(atom_domain["atom_is_attachment"]),
    )


def collate_production_motif_record(
    record: ProductionMotifRecord,
    *,
    tokenizer: ProductionTokenizerRuntime,
    seed: int,
    epoch: int,
    mask_probability: float = 0.15,
) -> ProductionCEExample:
    """Apply only the cheap stateless mask to one already validated row."""

    if not isinstance(record, ProductionMotifRecord):
        raise ProductionBridgeError("record must be a once-validated ProductionMotifRecord")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ProductionBridgeError("seed and epoch must be nonnegative integers")
    if (
        isinstance(mask_probability, bool)
        or not isinstance(mask_probability, (int, float))
        or not 0.0 < float(mask_probability) <= 1.0
    ):
        raise ProductionBridgeError("mask_probability must be in (0, 1]")

    if not isinstance(tokenizer, ProductionTokenizerRuntime):
        raise ProductionBridgeError("tokenizer must be a ProductionTokenizerRuntime")
    if record.tokenizer_contract_sha256 != tokenizer.tokenizer_contract_sha256:
        raise ProductionBridgeError("tokenizer contract hash differs from the production record")
    if record.tokenizer_snapshot_sha256 != tokenizer.tokenizer_snapshot_sha256:
        raise ProductionBridgeError("tokenizer snapshot hash differs from the production record")
    sentinels = tokenizer.sentinel_token_ids
    eos_token_id = tokenizer.eos_token_id
    original_ids = record.input_ids
    if any(token_id >= tokenizer.vocab_size for token_id in original_ids):
        raise ProductionBridgeError("production input ID is outside the tokenizer vocabulary")
    if set(original_ids).intersection(sentinels):
        raise ProductionBridgeError("sentinel IDs occur in the uncorrupted production input")

    mask = _select_logical_motifs(
        seed=seed,
        epoch=epoch,
        record_id=record.record_id,
        objective=IDENTITY_RECOVERY_OBJECTIVE,
        mask_probability=float(mask_probability),
        motif_count=len(record.identity_spans),
    )
    original_spans = record.identity_spans
    selected_in_input_order = tuple(
        sorted(
            (motif_id for motif_id, selected in enumerate(mask) if selected),
            key=lambda motif_id: original_spans[motif_id].start,
        )
    )
    if len(sentinels) < len(selected_in_input_order) + 1:
        raise ProductionBridgeError("selected identity count plus one terminal sentinel is required")

    selected_by_start = {
        original_spans[motif_id].start: (target_index, motif_id)
        for target_index, motif_id in enumerate(selected_in_input_order)
    }
    corrupted_ids: list[int] = []
    corrupted_to_motif: list[int] = []
    corrupted_roles: list[str] = []
    original_position = 0
    while original_position < len(original_ids):
        selected = selected_by_start.get(original_position)
        if selected is None:
            corrupted_ids.append(original_ids[original_position])
            corrupted_to_motif.append(record.token_to_logical_motif[original_position])
            corrupted_roles.append(record.token_role[original_position])
            original_position += 1
            continue
        target_index, motif_id = selected
        corrupted_ids.append(sentinels[target_index])
        corrupted_to_motif.append(motif_id)
        corrupted_roles.append(IDENTITY_SENTINEL_ROLE)
        original_position = original_spans[motif_id].stop

    reductions = tuple(
        (original_spans[motif_id].stop, original_spans[motif_id].stop - original_spans[motif_id].start - 1)
        for motif_id in selected_in_input_order
    )

    def transform_boundary(boundary: int) -> int:
        return boundary - sum(
            reduction for stop, reduction in reductions if stop <= boundary
        )

    identity_input_spans: list[Span] = []
    logical_to_carrier: list[int] = []
    for motif_id, original_span in enumerate(original_spans):
        start = transform_boundary(original_span.start)
        stop = start + 1 if mask[motif_id] else transform_boundary(original_span.stop)
        identity_input_spans.append(Span(start, stop))
        logical_to_carrier.append(start)

    connection_input_indices = tuple(
        tuple(transform_boundary(position) for position in row)
        for row in record.connection_token_indices
    )
    labels: list[int] = []
    for target_index, motif_id in enumerate(selected_in_input_order):
        span = original_spans[motif_id]
        labels.extend((sentinels[target_index], *original_ids[span.start : span.stop]))
    labels.extend((sentinels[len(selected_in_input_order)], eos_token_id))

    decision_sha = _mask_decision_sha256(
        seed=seed,
        epoch=epoch,
        record_id=record.record_id,
        objective=IDENTITY_RECOVERY_OBJECTIVE,
        mask_probability=float(mask_probability),
        selected_logical_motif_ids=tuple(
            motif_id for motif_id, selected in enumerate(mask) if selected
        ),
    )
    atom_to_logical_motif = record.atom_to_logical_motif
    atom_to_carrier = tuple(
        logical_to_carrier[motif_id] for motif_id in atom_to_logical_motif
    )
    example = ProductionCEExample(
        record_id=record.record_id,
        storage_key=record.storage_key,
        objective=IDENTITY_RECOVERY_OBJECTIVE,
        seed=seed,
        epoch=epoch,
        mask_probability=float(mask_probability),
        mask_decision_sha256=decision_sha,
        geometry_record_content_sha256=record.geometry_record_content_sha256,
        tokenizer_snapshot_sha256=record.tokenizer_snapshot_sha256,
        input_ids=tuple(corrupted_ids),
        labels=tuple(labels),
        input_token_to_logical_motif=tuple(corrupted_to_motif),
        input_token_role=tuple(corrupted_roles),
        identity_recovery_mask=mask,
        selected_logical_motif_ids_in_input_order=selected_in_input_order,
        identity_input_spans=tuple(identity_input_spans),
        connection_input_indices=connection_input_indices,
        logical_to_carrier=tuple(logical_to_carrier),
        full_e3fp_ids=record.full_e3fp_ids,
        atom_valid_mask=record.atom_valid_mask,
        model_to_source_atom_index=record.model_to_source_atom_index,
        atom_to_logical_motif=atom_to_logical_motif,
        atom_to_carrier=atom_to_carrier,
        atom_is_attachment=record.atom_is_attachment,
    )
    _validate_collated_example(example, record, sentinels, eos_token_id)
    return example


def collate_production_training_record(
    document: Mapping[str, object],
    *,
    tokenizer: ProductionTokenizerRuntime,
) -> ProductionCEExample:
    """Validate and replay the mask declared by one audit document.

    This convenience function is for fixtures and sampled audit replay.  The
    training Dataset should call :func:`load_production_motif_record` once and
    reuse :func:`collate_production_motif_record` across epochs.
    """

    record = load_production_motif_record(document)
    decision = document["mask_decision"]
    example = collate_production_motif_record(
        record,
        tokenizer=tokenizer,
        seed=decision["seed"],
        epoch=decision["epoch"],
        mask_probability=float(decision["mask_probability"]),
    )
    if example.identity_recovery_mask != tuple(document["masks"]["identity_recovery_mask"]):
        raise ProductionBridgeError("runtime mask differs from the audited production mask")
    if example.mask_decision_sha256 != decision["decision_sha256"]:
        raise ProductionBridgeError("runtime mask digest differs from the audited decision")
    return example


def _validate_collated_example(
    example: ProductionCEExample,
    record: ProductionMotifRecord,
    sentinels: Sequence[int],
    eos_token_id: int,
) -> None:
    """Check the small set of post-corruption invariants not in the record gate."""

    if not example.labels or example.labels[-1] != eos_token_id:
        raise ProductionBridgeError("CE labels must terminate with EOS")
    if example.labels[-2] != sentinels[len(example.selected_logical_motif_ids_in_input_order)]:
        raise ProductionBridgeError("CE labels lack the terminal sentinel")
    if len(example.input_ids) != len(example.input_token_role) or len(example.input_ids) != len(
        example.input_token_to_logical_motif
    ):
        raise ProductionBridgeError("corrupted token-domain arrays disagree")
    for motif_id, span in enumerate(example.identity_input_spans):
        if example.logical_to_carrier[motif_id] != span.start:
            raise ProductionBridgeError("logical carrier changed after corruption")
        if example.input_token_to_logical_motif[span.start] != motif_id:
            raise ProductionBridgeError("logical carrier maps to another motif")
        expected_role = IDENTITY_SENTINEL_ROLE if example.identity_recovery_mask[motif_id] else "identity"
        if example.input_token_role[span.start] != expected_role:
            raise ProductionBridgeError("identity carrier role changed after corruption")

    original_ids = record.input_ids
    original_connections = record.connection_token_indices
    for motif_id, transformed in enumerate(example.connection_input_indices):
        if any(example.input_token_role[position] != "connection" for position in transformed):
            raise ProductionBridgeError("connection token was hidden by identity corruption")
        if tuple(example.input_ids[position] for position in transformed) != tuple(
            original_ids[position] for position in original_connections[motif_id]
        ):
            raise ProductionBridgeError("connection token value changed during corruption")
    if len(example.atom_to_carrier) != len(example.full_e3fp_ids):
        raise ProductionBridgeError("atom geometry and carrier mappings disagree")
    if len(example.model_to_source_atom_index) != len(example.full_e3fp_ids):
        raise ProductionBridgeError("source-atom provenance and E3FP rows disagree")


def _build_geometry_sidecar(
    examples: Sequence[ProductionCEExample],
    ce_batch: PaddedCEBatch,
) -> GeometryBatchSidecar:
    atom_width = max(len(example.full_e3fp_ids) for example in examples)
    level_counts = {len(example.full_e3fp_ids[0]) for example in examples}
    if len(level_counts) != 1:
        raise ProductionBridgeError("E3FP level count must be common within a batch")
    level_count = next(iter(level_counts))
    padded_ids = []
    padded_masks = []
    padded_carriers = []
    padded_source_indices = []
    padded_attachment_roles = []
    atom_lengths = []
    for example in examples:
        atom_length = len(example.full_e3fp_ids)
        pad_count = atom_width - atom_length
        atom_lengths.append(atom_length)
        padded_ids.append(
            example.full_e3fp_ids + ((-1,) * level_count,) * pad_count
        )
        padded_masks.append(example.atom_valid_mask + (False,) * pad_count)
        padded_carriers.append(example.atom_to_carrier + (-1,) * pad_count)
        padded_source_indices.append(
            example.model_to_source_atom_index + (-1,) * pad_count
        )
        if example.atom_is_attachment:
            if len(example.atom_is_attachment) != atom_length:
                raise ProductionBridgeError(
                    "atom attachment roles and E3FP rows disagree"
                )
            padded_attachment_roles.append(
                example.atom_is_attachment + (False,) * pad_count
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
            e3fp_atom_is_attachment=(
                tuple(padded_attachment_roles)
                if len(padded_attachment_roles) == len(examples)
                else None
            ),
        )
    except FourGridContractError as exc:
        raise ProductionBridgeError("production geometry sidecar is inconsistent") from exc


def collate_production_batch(
    records: Sequence[ProductionMotifRecord],
    *,
    condition_id: str,
    tokenizer: ProductionTokenizerRuntime,
    seed: int,
    epoch: int,
    mask_probability: float = 0.15,
) -> P1ConditionBatch:
    """Collate M0 or M1 from the same once-validated Dataset rows."""

    try:
        spec = get_p1_condition_spec(condition_id)
    except FourGridContractError as exc:
        raise ProductionBridgeError("unknown four-grid condition") from exc
    if not spec.uses_logical_motifs:
        raise ProductionBridgeError(
            "A0/A1 require an atom/SELFIES-bound production record; "
            "logical-motif records cannot be reinterpreted as that baseline"
        )
    rows = tuple(records)
    if not rows:
        raise ProductionBridgeError("a production batch must contain at least one record")
    examples = tuple(
        collate_production_motif_record(
            record,
            tokenizer=tokenizer,
            seed=seed,
            epoch=epoch,
            mask_probability=mask_probability,
        )
        for record in rows
    )
    ce_batch = pad_ce_first_batch(examples, pad_token_id=tokenizer.pad_token_id)
    geometry = _build_geometry_sidecar(examples, ce_batch) if spec.uses_geometry else None
    try:
        return P1ConditionBatch(condition_id, ce_batch, geometry)
    except FourGridContractError as exc:
        raise ProductionBridgeError("condition batch is inconsistent") from exc
