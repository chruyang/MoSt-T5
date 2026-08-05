"""Pure-Python CE-first whole-logical-motif corruption.

This synthetic collator implements only T5-style identity recovery.  Mask
decisions are stateless and keyed by ``(seed, epoch, record_id, objective)``;
the per-motif score adds the logical motif ID to that key.  A selected motif's
entire identity span is replaced by one sentinel, while its connection span
and all atom/E3FP state remain visible.

There is deliberately no state-prediction mask, latent target, C1-R, C3, torch
tensor conversion, padding, or production tokenizer binding in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from .bound_record import BoundRecord, BoundRecordInvariantError, Span
from .hybrid_codec import HybridMotifCodec


IDENTITY_RECOVERY_OBJECTIVE = "identity_recovery_ce"
CONNECTION_VISIBILITY_POLICY = "always_visible_unmodified"
GEOMETRY_VISIBILITY_POLICY = "e3fp_visible_unmodified"


class CollatorContractError(ValueError):
    """Raised when the synthetic CE-first contract cannot be satisfied."""


@dataclass(frozen=True)
class MaskedIdentityTarget:
    """Audit metadata for one whole-span identity target."""

    logical_motif_id: int
    original_span: Span
    corrupted_span: Span
    sentinel_id: int
    original_input_ids: tuple[int, ...]


@dataclass(frozen=True)
class CEFirstExample:
    """One unpadded, pure-Python encoder/decoder training example."""

    objective: str
    record_id: str
    seed: int
    epoch: int
    mask_probability: float
    mask_decision_sha256: str
    eos_token_id: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    input_token_to_logical_motif: tuple[int, ...]
    input_token_role: tuple[str, ...]
    identity_recovery_mask: tuple[bool, ...]
    identity_input_spans: tuple[Span, ...]
    connection_input_spans: tuple[Span, ...]
    logical_to_carrier: tuple[int, ...]
    masked_identity_targets: tuple[MaskedIdentityTarget, ...]
    connection_visibility_policy: str
    connection_span_visible: tuple[bool, ...]
    geometry_visibility_policy: str
    geometry_corruption_mask: tuple[bool, ...]
    motif_geometry_valid: tuple[bool, ...]
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    state_prediction_enabled: bool

    def validate_against(
        self,
        record: BoundRecord,
        sentinel_token_ids: Sequence[int],
        eos_token_id: int,
    ) -> None:
        """Validate corruption without inferring semantics from token strings."""

        if self.objective != IDENTITY_RECOVERY_OBJECTIVE:
            raise CollatorContractError("only identity-recovery CE is permitted")
        if self.state_prediction_enabled is not False:
            raise CollatorContractError("state prediction/C3 is forbidden in CE-first v1")
        if self.record_id != record.record_id:
            raise CollatorContractError("collated record ID mismatch")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or self.epoch < 0
            or isinstance(self.mask_probability, bool)
            or not isinstance(self.mask_probability, (int, float))
            or not 0.0 < float(self.mask_probability) <= 1.0
        ):
            raise CollatorContractError("invalid stateless mask-decision inputs")
        if self.eos_token_id != eos_token_id:
            raise CollatorContractError("decoder EOS token changed")
        if not self.labels:
            raise CollatorContractError("CE decoder target cannot be empty")

        motif_count = len(record.identity_spans)
        motif_lengths = (
            len(self.identity_recovery_mask),
            len(self.identity_input_spans),
            len(self.connection_input_spans),
            len(self.logical_to_carrier),
            len(self.connection_span_visible),
            len(self.geometry_corruption_mask),
            len(self.motif_geometry_valid),
        )
        if any(length != motif_count for length in motif_lengths):
            raise CollatorContractError("collated motif-domain arrays have unequal lengths")
        if not any(self.identity_recovery_mask):
            raise CollatorContractError("at least one logical motif must be selected")
        if self.connection_visibility_policy != CONNECTION_VISIBILITY_POLICY:
            raise CollatorContractError("connection visibility policy changed")
        if not all(self.connection_span_visible):
            raise CollatorContractError("every connection span must remain visible")
        if self.geometry_visibility_policy != GEOMETRY_VISIBILITY_POLICY:
            raise CollatorContractError("geometry visibility policy changed")
        if any(self.geometry_corruption_mask):
            raise CollatorContractError("CE-first identity masking cannot corrupt geometry")
        if self.full_e3fp_ids != record.full_e3fp_ids:
            raise CollatorContractError("E3FP state changed during identity corruption")
        if self.motif_geometry_valid != record.motif_geometry_valid:
            raise CollatorContractError("motif geometry-valid flags changed")

        input_length = len(self.input_ids)
        if (
            len(self.input_token_to_logical_motif) != input_length
            or len(self.input_token_role) != input_length
        ):
            raise CollatorContractError("collated input-domain arrays have unequal lengths")

        selected_ids = tuple(
            motif_id
            for motif_id, selected in enumerate(self.identity_recovery_mask)
            if selected
        )
        expected_mask = _select_logical_motifs(
            seed=self.seed,
            epoch=self.epoch,
            record_id=self.record_id,
            objective=self.objective,
            mask_probability=float(self.mask_probability),
            motif_count=motif_count,
        )
        if self.identity_recovery_mask != expected_mask:
            raise CollatorContractError("identity mask disagrees with its stateless key")
        expected_decision_sha = _mask_decision_sha256(
            seed=self.seed,
            epoch=self.epoch,
            record_id=self.record_id,
            objective=self.objective,
            mask_probability=float(self.mask_probability),
            selected_logical_motif_ids=selected_ids,
        )
        if self.mask_decision_sha256 != expected_decision_sha:
            raise CollatorContractError("mask decision SHA-256 does not match its payload")
        if tuple(target.logical_motif_id for target in self.masked_identity_targets) != selected_ids:
            raise CollatorContractError("masked targets are not in logical motif order")
        if len(sentinel_token_ids) < len(selected_ids) + 1:
            raise CollatorContractError("one sentinel per span plus a terminal sentinel is required")

        selected_by_start = {
            record.identity_spans[motif_id].start: (target_index, motif_id)
            for target_index, motif_id in enumerate(selected_ids)
        }
        expected_input_ids: list[int] = []
        expected_to_motif: list[int] = []
        expected_roles: list[str] = []
        expected_targets: list[MaskedIdentityTarget] = []
        original_position = 0
        while original_position < len(record.input_ids):
            selected_entry = selected_by_start.get(original_position)
            if selected_entry is None:
                expected_input_ids.append(record.input_ids[original_position])
                expected_to_motif.append(record.token_to_logical_motif[original_position])
                expected_roles.append(record.token_role[original_position])
                original_position += 1
                continue
            target_index, motif_id = selected_entry
            original_span = record.identity_spans[motif_id]
            corrupted_start = len(expected_input_ids)
            sentinel_id = sentinel_token_ids[target_index]
            expected_input_ids.append(sentinel_id)
            expected_to_motif.append(motif_id)
            expected_roles.append("identity_sentinel")
            expected_targets.append(
                MaskedIdentityTarget(
                    logical_motif_id=motif_id,
                    original_span=original_span,
                    corrupted_span=Span(corrupted_start, corrupted_start + 1),
                    sentinel_id=sentinel_id,
                    original_input_ids=record.input_ids[
                        original_span.start : original_span.stop
                    ],
                )
            )
            original_position = original_span.stop

        if self.input_ids != tuple(expected_input_ids):
            raise CollatorContractError(
                "corrupted input differs from the complete canonical reconstruction"
            )
        if self.input_token_to_logical_motif != tuple(expected_to_motif):
            raise CollatorContractError("corrupted token-to-motif mapping changed")
        if self.input_token_role != tuple(expected_roles):
            raise CollatorContractError("corrupted token roles changed")
        if self.masked_identity_targets != tuple(expected_targets):
            raise CollatorContractError("masked target audit records changed")

        expected_labels: list[int] = []
        for target in expected_targets:
            expected_labels.extend((target.sentinel_id, *target.original_input_ids))
        expected_labels.append(sentinel_token_ids[len(selected_ids)])
        expected_labels.append(eos_token_id)
        if self.labels != tuple(expected_labels):
            raise CollatorContractError(
                "decoder labels must end with terminal sentinel then EOS"
            )

        reductions = tuple(
            (
                record.identity_spans[motif_id].stop,
                record.identity_spans[motif_id].stop
                - record.identity_spans[motif_id].start
                - 1,
            )
            for motif_id in selected_ids
        )

        def transform_boundary(boundary: int) -> int:
            return boundary - sum(
                reduction for stop, reduction in reductions if stop <= boundary
            )

        expected_identity_spans = []
        expected_connection_spans = []
        expected_carriers = []
        for motif_id in range(motif_count):
            original_identity = record.identity_spans[motif_id]
            identity_start = transform_boundary(original_identity.start)
            identity_stop = (
                identity_start + 1
                if self.identity_recovery_mask[motif_id]
                else transform_boundary(original_identity.stop)
            )
            expected_identity_spans.append(Span(identity_start, identity_stop))
            expected_carriers.append(identity_start)
            original_connection = record.connection_spans[motif_id]
            expected_connection_spans.append(
                Span(
                    transform_boundary(original_connection.start),
                    transform_boundary(original_connection.stop),
                )
            )
        if self.identity_input_spans != tuple(expected_identity_spans):
            raise CollatorContractError("collated identity spans changed")
        if self.connection_input_spans != tuple(expected_connection_spans):
            raise CollatorContractError("collated connection spans changed")
        if self.logical_to_carrier != tuple(expected_carriers):
            raise CollatorContractError("collated logical carriers changed")

        for motif_id in range(motif_count):
            identity_span = self.identity_input_spans[motif_id]
            connection_span = self.connection_input_spans[motif_id]
            if (
                identity_span.stop > input_length
                or connection_span.stop > input_length
                or identity_span.stop > connection_span.start
            ):
                raise CollatorContractError("collated motif spans are invalid")
            carrier = self.logical_to_carrier[motif_id]
            if carrier != identity_span.start:
                raise CollatorContractError("logical carrier must remain the identity-span start")
            if self.input_token_to_logical_motif[carrier] != motif_id:
                raise CollatorContractError("collated carrier maps to the wrong logical motif")

            if self.identity_recovery_mask[motif_id]:
                if self.input_token_role[carrier] != "identity_sentinel":
                    raise CollatorContractError("masked carrier must have identity_sentinel role")
            else:
                original = record.identity_spans[motif_id]
                if self.input_ids[identity_span.start : identity_span.stop] != record.input_ids[
                    original.start : original.stop
                ]:
                    raise CollatorContractError("unselected identity span changed")
                if any(
                    role != "identity"
                    for role in self.input_token_role[identity_span.start : identity_span.stop]
                ):
                    raise CollatorContractError("unselected identity token role changed")

            original_connection = record.connection_spans[motif_id]
            if self.input_ids[connection_span.start : connection_span.stop] != record.input_ids[
                original_connection.start : original_connection.stop
            ]:
                raise CollatorContractError("connection span was modified or hidden")
            if any(
                role != "connection"
                for role in self.input_token_role[connection_span.start : connection_span.stop]
            ):
                raise CollatorContractError("connection token role changed")


class SyntheticCEFirstCollator:
    """Stateless whole-logical-motif corruption for synthetic records."""

    def __init__(
        self,
        *,
        codec: HybridMotifCodec,
        token_to_id: Mapping[str, int],
        sentinel_token_ids: Sequence[int],
        eos_token_id: int,
        seed: int,
        mask_probability: float = 0.15,
        objective: str = IDENTITY_RECOVERY_OBJECTIVE,
    ) -> None:
        if objective != IDENTITY_RECOVERY_OBJECTIVE:
            raise CollatorContractError(
                "CE-first v1 forbids state prediction, C3 and other objectives"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CollatorContractError("seed must be a nonnegative integer")
        if (
            isinstance(mask_probability, bool)
            or not isinstance(mask_probability, (int, float))
            or not 0.0 < float(mask_probability) <= 1.0
        ):
            raise CollatorContractError("mask_probability must be in (0, 1]")
        sentinels = tuple(sentinel_token_ids)
        if not sentinels:
            raise CollatorContractError("sentinel_token_ids cannot be empty")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in sentinels
        ):
            raise CollatorContractError("sentinel IDs must be nonnegative integers")
        if len(set(sentinels)) != len(sentinels):
            raise CollatorContractError("sentinel IDs must be unique")
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise CollatorContractError("eos_token_id must be a nonnegative integer")
        if eos_token_id in sentinels:
            raise CollatorContractError("EOS token ID must be distinct from sentinels")

        self._codec = codec
        self._token_to_id = dict(token_to_id)
        self._sentinel_token_ids = sentinels
        self._eos_token_id = eos_token_id
        self._seed = seed
        self._mask_probability = float(mask_probability)
        self._objective = objective

    def __call__(
        self, records: Sequence[BoundRecord], *, epoch: int
    ) -> tuple[CEFirstExample, ...]:
        if not records:
            raise CollatorContractError("collator requires at least one BoundRecord")
        return tuple(self.collate_record(record, epoch=epoch) for record in records)

    def collate_record(self, record: BoundRecord, *, epoch: int) -> CEFirstExample:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise CollatorContractError("epoch must be a nonnegative integer")
        try:
            record.validate(self._codec, self._token_to_id)
        except BoundRecordInvariantError as exc:
            raise CollatorContractError("BoundRecord failed validation before collation") from exc
        if set(record.input_ids).intersection(self._sentinel_token_ids):
            raise CollatorContractError("sentinel IDs must not occur in uncorrupted input")

        motif_count = len(record.identity_spans)
        selected = list(
            _select_logical_motifs(
                seed=self._seed,
                epoch=epoch,
                record_id=record.record_id,
                objective=self._objective,
                mask_probability=self._mask_probability,
                motif_count=motif_count,
            )
        )
        selected_ids = tuple(motif_id for motif_id, value in enumerate(selected) if value)
        if len(self._sentinel_token_ids) < len(selected_ids) + 1:
            raise CollatorContractError(
                "selected_count + 1 sentinel IDs are required"
            )

        selected_by_start = {
            record.identity_spans[motif_id].start: (target_index, motif_id)
            for target_index, motif_id in enumerate(selected_ids)
        }
        corrupted_ids: list[int] = []
        corrupted_to_motif: list[int] = []
        corrupted_roles: list[str] = []
        masked_targets: list[MaskedIdentityTarget] = []
        original_position = 0
        while original_position < len(record.input_ids):
            selected_entry = selected_by_start.get(original_position)
            if selected_entry is None:
                corrupted_ids.append(record.input_ids[original_position])
                corrupted_to_motif.append(record.token_to_logical_motif[original_position])
                corrupted_roles.append(record.token_role[original_position])
                original_position += 1
                continue

            target_index, motif_id = selected_entry
            original_span = record.identity_spans[motif_id]
            sentinel_id = self._sentinel_token_ids[target_index]
            corrupted_start = len(corrupted_ids)
            corrupted_ids.append(sentinel_id)
            corrupted_to_motif.append(motif_id)
            corrupted_roles.append("identity_sentinel")
            masked_targets.append(
                MaskedIdentityTarget(
                    logical_motif_id=motif_id,
                    original_span=original_span,
                    corrupted_span=Span(corrupted_start, corrupted_start + 1),
                    sentinel_id=sentinel_id,
                    original_input_ids=record.input_ids[original_span.start : original_span.stop],
                )
            )
            original_position = original_span.stop

        reductions = tuple(
            (
                record.identity_spans[motif_id].stop,
                record.identity_spans[motif_id].stop
                - record.identity_spans[motif_id].start
                - 1,
            )
            for motif_id in selected_ids
        )

        def transform_boundary(boundary: int) -> int:
            return boundary - sum(
                reduction for stop, reduction in reductions if stop <= boundary
            )

        identity_input_spans: list[Span] = []
        connection_input_spans: list[Span] = []
        logical_to_carrier: list[int] = []
        for motif_id in range(motif_count):
            original_identity = record.identity_spans[motif_id]
            identity_start = transform_boundary(original_identity.start)
            identity_stop = (
                identity_start + 1
                if selected[motif_id]
                else transform_boundary(original_identity.stop)
            )
            identity_input_spans.append(Span(identity_start, identity_stop))
            logical_to_carrier.append(identity_start)

            original_connection = record.connection_spans[motif_id]
            connection_input_spans.append(
                Span(
                    transform_boundary(original_connection.start),
                    transform_boundary(original_connection.stop),
                )
            )

        labels: list[int] = []
        for target_index, target in enumerate(masked_targets):
            labels.extend((self._sentinel_token_ids[target_index], *target.original_input_ids))
        labels.append(self._sentinel_token_ids[len(masked_targets)])
        labels.append(self._eos_token_id)

        decision_sha = _mask_decision_sha256(
            seed=self._seed,
            epoch=epoch,
            record_id=record.record_id,
            objective=self._objective,
            mask_probability=self._mask_probability,
            selected_logical_motif_ids=selected_ids,
        )

        example = CEFirstExample(
            objective=self._objective,
            record_id=record.record_id,
            seed=self._seed,
            epoch=epoch,
            mask_probability=self._mask_probability,
            mask_decision_sha256=decision_sha,
            eos_token_id=self._eos_token_id,
            input_ids=tuple(corrupted_ids),
            labels=tuple(labels),
            input_token_to_logical_motif=tuple(corrupted_to_motif),
            input_token_role=tuple(corrupted_roles),
            identity_recovery_mask=tuple(selected),
            identity_input_spans=tuple(identity_input_spans),
            connection_input_spans=tuple(connection_input_spans),
            logical_to_carrier=tuple(logical_to_carrier),
            masked_identity_targets=tuple(masked_targets),
            connection_visibility_policy=CONNECTION_VISIBILITY_POLICY,
            connection_span_visible=tuple(True for _ in range(motif_count)),
            geometry_visibility_policy=GEOMETRY_VISIBILITY_POLICY,
            geometry_corruption_mask=tuple(False for _ in range(motif_count)),
            motif_geometry_valid=record.motif_geometry_valid,
            full_e3fp_ids=record.full_e3fp_ids,
            state_prediction_enabled=False,
        )
        example.validate_against(record, self._sentinel_token_ids, self._eos_token_id)
        return example


def _stateless_motif_score(
    *,
    seed: int,
    epoch: int,
    record_id: str,
    objective: str,
    logical_motif_id: int,
) -> float:
    """Map the declared stateless key and motif ID to ``[0, 1)``."""

    preimage = json.dumps(
        [seed, epoch, record_id, objective, logical_motif_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")
    return integer / float(1 << 64)


def _select_logical_motifs(
    *,
    seed: int,
    epoch: int,
    record_id: str,
    objective: str,
    mask_probability: float,
    motif_count: int,
) -> tuple[bool, ...]:
    """Apply the stateless score rule and its deterministic nonempty gate."""

    scores = tuple(
        _stateless_motif_score(
            seed=seed,
            epoch=epoch,
            record_id=record_id,
            objective=objective,
            logical_motif_id=motif_id,
        )
        for motif_id in range(motif_count)
    )
    selected = [score < mask_probability for score in scores]
    if not any(selected):
        selected[min(range(motif_count), key=lambda motif_id: (scores[motif_id], motif_id))] = True
    return tuple(selected)


def _mask_decision_sha256(
    *,
    seed: int,
    epoch: int,
    record_id: str,
    objective: str,
    mask_probability: float,
    selected_logical_motif_ids: Sequence[int],
) -> str:
    payload = {
        "epoch": epoch,
        "mask_probability": mask_probability,
        "objective": objective,
        "record_id": record_id,
        "schema": "most-t5-next/ce-first-mask-decision/v1",
        "seed": seed,
        "selected_logical_motif_ids": list(selected_logical_motif_ids),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
