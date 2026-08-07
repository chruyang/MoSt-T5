#!/usr/bin/env python3
"""Thin PF-1 A0/A1/M0/M1 training runner.

The small ``PF1RecordReader`` protocol keeps the scientific training loop
independent of the published LMDB directory layout while reusing the existing
paired-record, collator, wrapper and tensor boundaries.  The CLI binds that
protocol to ``PF1PairedReleaseReader`` over the passed, verified PF-1 paired
release (the current run3 artifact is one such release).

This module defines the frozen 1% screening run; importing it does not start
training.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from most_t5_next.p1.atom_production_bridge import collate_production_atom_batch
from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_four_grid_wrapper,
)
from most_t5_next.p1.build_pf1_paired_release_v1 import (
    TOKENIZER_DIRECTORY,
    PF1PairedReleaseReader,
)
from most_t5_next.p1.experiment_grid import P1ConditionBatch
from most_t5_next.p1.pf1_optimization import (
    FROZEN_PF1_PROTOCOL,
    PF1LearningRateSchedule,
    PF1OptimizationProtocol,
    build_pf1_optimizer,
    clip_pf1_gradients,
)
from most_t5_next.p1.production_bridge import collate_production_batch
from most_t5_next.p1.training_adapter import (
    select_four_grid_forward_inputs,
    to_four_grid_batch_encoding,
)
from most_t5_next.r1.adapter.paired_record_wire_v1 import (
    LoadedPairedTrainingRecord,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p1/pf1-four-grid-training/v1"
CHECKPOINT_SCHEMA = "most-t5-p1/pf1-four-grid-checkpoint/v1"
FOUR_GRID_MANIFEST_NAME = "pf1_training_manifest.json"
CONDITION_MANIFEST_NAME = "pf1_condition_manifest.json"
CONDITION_ORDER = ("A0", "A1", "M0", "M1")
TRAIN_CORRUPTION_SEED = 0
DEV_CORRUPTION_SEED = 1
DEV_CORRUPTION_EPOCH = 0
MASK_PROBABILITY = 0.15
FORWARD_SEED = 20260807
EVALUATION_UPDATES = (0, 250, 500, 750, 1000)
CHECKPOINT_UPDATES = (500, 1000)
GEOMETRY_PERTURBATION_UPDATE = 1000
GEOMETRY_DERANGEMENT_SEED = 20260809
GEOMETRY_PAIRED_FORWARD_SEED = 20260810
TRAIN_PREFETCH_DEPTH = 2


class PF1TrainingError(RuntimeError):
    """The frozen PF-1 screen could not complete."""


@dataclass(frozen=True)
class PF1GeometryDerangementPlan:
    """Eligible dev recipients, their donors, and explicit singleton exclusions."""

    eligible_indices: tuple[int, ...]
    donor_indices: tuple[int, ...]
    excluded_singleton_indices: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedTrainUpdate:
    """One ordered optimizer update and its exact consumed-data cursor."""

    batches: tuple[P1ConditionBatch, ...]
    committed_cursor_state: dict[str, int]


class PF1RecordReader(Protocol):
    """Reader boundary for one frozen, group-disjoint PF-1 membership.

    ``iter_train_epoch`` must replay the same member order for every cell at a
    given epoch.  Like the standard PyTorch ``drop_last=False`` contract, its
    final micro-batch may be shorter but must be non-empty.  ``iter_dev`` must
    replay one fixed dev order and may also end in a shorter batch.  Both
    methods return the already-decoded paired A/M record used by the existing
    production collators.
    """

    train_member_count: int
    dev_member_count: int

    def iter_train_epoch(
        self, *, epoch: int, batch_size: int
    ) -> Iterator[Sequence[LoadedPairedTrainingRecord]]: ...

    def iter_dev(
        self, *, batch_size: int
    ) -> Iterator[Sequence[LoadedPairedTrainingRecord]]: ...


def collate_pf1_condition(
    records: Sequence[LoadedPairedTrainingRecord],
    *,
    condition_id: str,
    tokenizer_runtime: Any,
    seed: int,
    epoch: int,
) -> P1ConditionBatch:
    """Reuse the production collators without reinterpreting paired records."""

    rows = tuple(records)
    if condition_id in ("A0", "A1"):
        return collate_production_atom_batch(
            tuple(row.atom_record for row in rows),
            condition_id=condition_id,
            tokenizer=tokenizer_runtime,
            seed=seed,
            epoch=epoch,
            mask_probability=MASK_PROBABILITY,
        )
    if condition_id in ("M0", "M1"):
        return collate_production_batch(
            tuple(row.motif_record for row in rows),
            condition_id=condition_id,
            tokenizer=tokenizer_runtime,
            seed=seed,
            epoch=epoch,
            mask_probability=MASK_PROBABILITY,
        )
    raise PF1TrainingError("unknown PF-1 condition")


class _TrainCursor:
    def __init__(self, reader: PF1RecordReader, micro_batch_size: int) -> None:
        self.reader = reader
        self.micro_batch_size = micro_batch_size
        self.epoch = 0
        self.batch_in_epoch = 0
        self._iterator = iter(
            reader.iter_train_epoch(epoch=0, batch_size=micro_batch_size)
        )

    def next(self) -> tuple[int, tuple[LoadedPairedTrainingRecord, ...]]:
        while True:
            try:
                rows = tuple(next(self._iterator))
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self._iterator = iter(
                    self.reader.iter_train_epoch(
                        epoch=self.epoch, batch_size=self.micro_batch_size
                    )
                )
                continue
            if not rows or len(rows) > self.micro_batch_size:
                raise PF1TrainingError(
                    "PF-1 train reader yielded an empty or oversized micro-batch"
                )
            epoch = self.epoch
            self.batch_in_epoch += 1
            return epoch, rows

    def state_dict(self) -> dict[str, int]:
        return {
            "next_epoch": self.epoch,
            "next_batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore the exact next batch without replaying it to the trainer.

        ``batch_in_epoch`` counts batches already consumed in ``epoch``.  The
        iterator is therefore rebuilt at the beginning of that epoch and the
        consumed prefix is skipped once.  A state positioned exactly at an
        epoch boundary remains valid: the next call advances to the following
        epoch through the ordinary ``StopIteration`` path.
        """

        epoch = state.get("next_epoch")
        batch_in_epoch = state.get("next_batch_in_epoch")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(batch_in_epoch, bool)
            or not isinstance(batch_in_epoch, int)
            or batch_in_epoch < 0
        ):
            raise PF1TrainingError("invalid PF-1 train cursor state")

        iterator = iter(
            self.reader.iter_train_epoch(
                epoch=epoch,
                batch_size=self.micro_batch_size,
            )
        )
        for _ in range(batch_in_epoch):
            try:
                skipped = tuple(next(iterator))
            except StopIteration as exc:
                raise PF1TrainingError(
                    "PF-1 train cursor exceeds the frozen epoch"
                ) from exc
            if not skipped or len(skipped) > self.micro_batch_size:
                raise PF1TrainingError(
                    "PF-1 train reader yielded an empty or oversized micro-batch"
                )
        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        self._iterator = iterator


def _prepare_train_update(
    *,
    cursor: _TrainCursor,
    gradient_accumulation_steps: int,
    condition_id: str,
    tokenizer_runtime: Any,
    data_lock: Any | None,
) -> _PreparedTrainUpdate:
    """Decode and collate one update without changing its ordered semantics.

    The cursor is owned by the single producer while prefetch is active.  Its
    state is copied into the payload only after every micro-batch in the update
    has been consumed.  The training thread therefore checkpoints this
    committed state rather than the producer's potentially prefetched state.
    """

    guard = data_lock if data_lock is not None else nullcontext()
    with guard:
        batches: list[P1ConditionBatch] = []
        for _ in range(gradient_accumulation_steps):
            epoch, records = cursor.next()
            batches.append(
                collate_pf1_condition(
                    records,
                    condition_id=condition_id,
                    tokenizer_runtime=tokenizer_runtime,
                    seed=TRAIN_CORRUPTION_SEED,
                    epoch=epoch,
                )
            )
        return _PreparedTrainUpdate(
            batches=tuple(batches),
            committed_cursor_state=dict(cursor.state_dict()),
        )


class _OrderedTrainPrefetch:
    """Bounded single-producer prefetch with strict in-order delivery."""

    def __init__(
        self,
        *,
        cursor: _TrainCursor,
        total_updates: int,
        depth: int,
        gradient_accumulation_steps: int,
        condition_id: str,
        tokenizer_runtime: Any,
        data_lock: Any,
    ) -> None:
        if total_updates < 0 or depth <= 0:
            raise PF1TrainingError("invalid PF-1 train prefetch bounds")
        self._cursor = cursor
        self._total_updates = total_updates
        self._depth = depth
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._condition_id = condition_id
        self._tokenizer_runtime = tokenizer_runtime
        self._data_lock = data_lock
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pf1-ordered-prefetch",
        )
        self._futures: deque[Future[_PreparedTrainUpdate]] = deque()
        self._submitted = 0
        self._delivered = 0
        self.closed = False
        for _ in range(min(depth, total_updates)):
            self._submit_one()

    def _submit_one(self) -> None:
        self._futures.append(
            self._executor.submit(
                _prepare_train_update,
                cursor=self._cursor,
                gradient_accumulation_steps=self._gradient_accumulation_steps,
                condition_id=self._condition_id,
                tokenizer_runtime=self._tokenizer_runtime,
                data_lock=self._data_lock,
            )
        )
        self._submitted += 1

    def __iter__(self) -> "_OrderedTrainPrefetch":
        return self

    def __next__(self) -> _PreparedTrainUpdate:
        if self._delivered >= self._total_updates:
            raise StopIteration
        future = self._futures.popleft()
        try:
            prepared = future.result()
        except BaseException:
            self.close()
            raise
        self._delivered += 1
        if self._submitted < self._total_updates:
            self._submit_one()
        return prepared

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self._futures:
            future.cancel()
        self._executor.shutdown(wait=True)
        self._futures.clear()

    def __enter__(self) -> "_OrderedTrainPrefetch":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _autocast(torch_module: Any, use_bf16: bool) -> Any:
    if not use_bf16:
        return nullcontext()
    return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)


def _finite_loss(torch_module: Any, value: Any, condition_id: str) -> float:
    if value is None or value.ndim != 0 or not bool(torch_module.isfinite(value).item()):
        raise PF1TrainingError(condition_id + " produced a non-finite CE loss")
    return float(value.detach().float().cpu().item())


def evaluate_pf1_condition(
    model: Any,
    *,
    condition_id: str,
    reader: PF1RecordReader,
    tokenizer_runtime: Any,
    device: Any,
    use_bf16: bool,
    protocol: PF1OptimizationProtocol,
    torch_module: Any,
    data_lock: Any | None = None,
) -> dict[str, object]:
    """Evaluate token-weighted NLL and teacher-forced masked-token accuracy."""

    model.eval()
    nll_sum = 0.0
    correct_tokens = 0
    target_tokens = 0
    encoder_tokens = 0
    members = 0
    # The same concrete reader and tokenizer runtime are shared with the
    # producer.  Evaluation takes the data lock for its complete fixed replay,
    # preventing an LMDB/tokenizer call from overlapping across the two
    # threads; GPU execution can still overlap with already prepared batches.
    guard = data_lock if data_lock is not None else nullcontext()
    with guard:
        with torch_module.no_grad():
            for records in reader.iter_dev(batch_size=protocol.micro_batch_size):
                rows = tuple(records)
                if not rows:
                    continue
                batch = collate_pf1_condition(
                    rows,
                    condition_id=condition_id,
                    tokenizer_runtime=tokenizer_runtime,
                    seed=DEV_CORRUPTION_SEED,
                    epoch=DEV_CORRUPTION_EPOCH,
                )
                encoded = to_four_grid_batch_encoding(batch, device=device)
                forward_inputs = select_four_grid_forward_inputs(encoded)
                with _autocast(torch_module, use_bf16):
                    outputs = model(
                        **forward_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                loss_value = _finite_loss(torch_module, outputs.loss, condition_id)
                labels = forward_inputs["labels"]
                active = labels.ne(-100)
                count = int(active.sum().item())
                predictions = outputs.logits.argmax(dim=-1)
                correct_tokens += int((predictions.eq(labels) & active).sum().item())
                target_tokens += count
                nll_sum += loss_value * count
                encoder_tokens += sum(batch.ce_batch.input_lengths)
                members += len(rows)
    if target_tokens == 0:
        raise PF1TrainingError("PF-1 dev reader produced no supervised tokens")
    return {
        "members": members,
        "encoder_nonpadding_tokens": encoder_tokens,
        "supervised_target_tokens": target_tokens,
        "token_weighted_nll": nll_sum / target_tokens,
        "masked_token_accuracy": correct_tokens / target_tokens,
    }


def build_pf1_geometry_derangement(
    records: Sequence[LoadedPairedTrainingRecord],
    *,
    seed: int = GEOMETRY_DERANGEMENT_SEED,
) -> PF1GeometryDerangementPlan:
    """Map every dev recipient to a different, same-size geometry donor.

    Buckets use the post-projection model-atom count consumed by the wrapper.
    Within each bucket, the frozen dev order is made explicit by sorting on the
    persisted schedule index and record ID, then applying one non-zero cyclic
    shift derived from ``seed``.  Buckets of size one are explicitly returned
    as ineligible because a same-size, no-self donor does not exist.  The
    remaining mapping is a deterministic bijection: it neither samples, crops
    nor duplicates E3FP rows.
    """

    rows = tuple(records)
    if not rows:
        raise PF1TrainingError("geometry perturbation requires the complete dev set")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PF1TrainingError("geometry derangement seed must be nonnegative")

    identities: list[tuple[int, str]] = []
    atom_counts: list[int] = []
    buckets: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        record_id = row.atom_record.record_id
        identity = (row.schedule_index, record_id)
        identities.append(identity)
        atom_count = len(row.atom_record.full_e3fp_ids)
        if atom_count <= 0:
            raise PF1TrainingError("geometry donor has no model-atom E3FP rows")
        atom_counts.append(atom_count)
        buckets.setdefault(atom_count, []).append(index)
    if len(set(identities)) != len(identities):
        raise PF1TrainingError("geometry perturbation dev identities are not unique")

    eligible_indices: list[int] = []
    donor_indices: list[int] = []
    excluded_singletons: list[int] = []
    for atom_count in sorted(buckets):
        members = sorted(buckets[atom_count], key=lambda index: identities[index])
        if len(members) < 2:
            excluded_singletons.extend(members)
            continue
        shift = 1 + ((seed + atom_count) % (len(members) - 1))
        for position, recipient_index in enumerate(members):
            eligible_indices.append(recipient_index)
            donor_indices.append(members[(position + shift) % len(members)])

    paired = sorted(zip(eligible_indices, donor_indices))
    eligible = tuple(recipient for recipient, _donor in paired)
    mapping = tuple(donor for _recipient, donor in paired)
    if not eligible:
        raise PF1TrainingError("geometry perturbation has no derangeable dev members")
    if sorted(mapping) != list(eligible):
        raise PF1TrainingError("geometry donor mapping is not an eligible-set bijection")
    if any(
        donor_index == recipient_index or atom_counts[donor_index] != atom_counts[recipient_index]
        for recipient_index, donor_index in zip(eligible, mapping)
    ):
        raise PF1TrainingError(
            "geometry donor mapping violates no-self or model-atom-count parity"
        )
    return PF1GeometryDerangementPlan(
        eligible_indices=eligible,
        donor_indices=mapping,
        excluded_singleton_indices=tuple(sorted(excluded_singletons)),
    )


def _replace_with_donor_e3fp(
    aligned_batch: P1ConditionBatch,
    donor_records: Sequence[LoadedPairedTrainingRecord],
) -> P1ConditionBatch:
    """Replace only active E3FP rows; keep every recipient-side field fixed."""

    if aligned_batch.condition_id not in ("A1", "M1") or aligned_batch.geometry is None:
        raise PF1TrainingError("E3FP perturbation is defined only for A1 and M1")
    donors = tuple(donor_records)
    geometry = aligned_batch.geometry
    if len(donors) != len(geometry.record_ids):
        raise PF1TrainingError("geometry donor batch size differs from recipients")

    atom_width = len(geometry.e3fp_ids[0])
    level_count = geometry.e3fp_level_count
    padded_donor_rows: list[tuple[tuple[int, ...], ...]] = []
    for batch_index, donor in enumerate(donors):
        active_rows = tuple(tuple(levels) for levels in donor.atom_record.full_e3fp_ids)
        expected_count = geometry.atom_lengths[batch_index]
        if len(active_rows) != expected_count:
            raise PF1TrainingError(
                "geometry donor model atom count differs from its recipient"
            )
        if any(len(levels) != level_count for levels in active_rows):
            raise PF1TrainingError("geometry donor E3FP level width differs")
        # This is only the recipient batch's existing rectangular padding; no
        # donor atom is cropped, fabricated or repeated.
        padded_donor_rows.append(
            active_rows + ((-1,) * level_count,) * (atom_width - expected_count)
        )

    shuffled_geometry = replace(
        geometry,
        e3fp_ids=tuple(padded_donor_rows),
    )
    shuffled = replace(aligned_batch, geometry=shuffled_geometry)
    fixed_fields = (
        "record_ids",
        "e3fp_atom_mask",
        "e3fp_atom_to_token",
        "model_to_source_atom_index",
        "atom_lengths",
        "e3fp_level_count",
        "token_width",
    )
    if shuffled.ce_batch is not aligned_batch.ce_batch or any(
        getattr(shuffled_geometry, field) != getattr(geometry, field)
        for field in fixed_fields
    ):
        raise PF1TrainingError("recipient CE or geometry mapping changed during shuffle")
    return shuffled


def _seed_geometry_paired_forward(
    torch_module: Any,
    *,
    device: Any,
) -> None:
    torch_module.manual_seed(GEOMETRY_PAIRED_FORWARD_SEED)
    if device.type == "cuda":
        torch_module.cuda.manual_seed_all(GEOMETRY_PAIRED_FORWARD_SEED)


def evaluate_pf1_geometry_sensitivity(
    model: Any,
    *,
    condition_id: str,
    reader: PF1RecordReader,
    tokenizer_runtime: Any,
    device: Any,
    use_bf16: bool,
    protocol: PF1OptimizationProtocol,
    torch_module: Any,
) -> dict[str, object]:
    """Paired final-dev NLL under aligned and same-size shuffled E3FP.

    This diagnostic changes no parameter and is not a causal test.  Both
    forwards share recipient text, masks, labels, carriers and the frozen
    forward seed; the shuffled call substitutes only donor ``e3fp_ids``.
    """

    if condition_id not in ("A1", "M1"):
        raise PF1TrainingError("geometry sensitivity runs only for A1 and M1")
    dev_rows = tuple(
        row
        for batch in reader.iter_dev(batch_size=protocol.micro_batch_size)
        for row in batch
    )
    if len(dev_rows) != reader.dev_member_count:
        raise PF1TrainingError(
            "geometry sensitivity must consume the complete fixed dev membership"
        )
    plan = build_pf1_geometry_derangement(dev_rows)
    atom_counts = tuple(len(row.atom_record.full_e3fp_ids) for row in dev_rows)

    model.eval()
    aligned_nll_sum = 0.0
    shuffled_nll_sum = 0.0
    target_tokens = 0
    encoder_tokens = 0
    members = 0
    cpu_rng_state = torch_module.random.get_rng_state()
    cuda_rng_state = (
        torch_module.cuda.get_rng_state_all() if device.type == "cuda" else None
    )
    try:
        with torch_module.no_grad():
            for start in range(
                0, len(plan.eligible_indices), protocol.micro_batch_size
            ):
                recipient_indices = plan.eligible_indices[
                    start : start + protocol.micro_batch_size
                ]
                donor_indices = plan.donor_indices[
                    start : start + len(recipient_indices)
                ]
                rows = tuple(dev_rows[index] for index in recipient_indices)
                donor_rows = tuple(
                    dev_rows[index] for index in donor_indices
                )
                aligned_batch = collate_pf1_condition(
                    rows,
                    condition_id=condition_id,
                    tokenizer_runtime=tokenizer_runtime,
                    seed=DEV_CORRUPTION_SEED,
                    epoch=DEV_CORRUPTION_EPOCH,
                )
                shuffled_batch = _replace_with_donor_e3fp(
                    aligned_batch,
                    donor_rows,
                )
                aligned_encoded = to_four_grid_batch_encoding(
                    aligned_batch,
                    device=device,
                )
                aligned_inputs = select_four_grid_forward_inputs(aligned_encoded)
                shuffled_inputs = dict(aligned_inputs)
                shuffled_ids = torch_module.as_tensor(
                    shuffled_batch.geometry.e3fp_ids,
                    dtype=aligned_inputs["e3fp_ids"].dtype,
                    device=device,
                )
                if shuffled_ids.shape != aligned_inputs["e3fp_ids"].shape:
                    raise PF1TrainingError("shuffled E3FP tensor shape differs")
                shuffled_inputs["e3fp_ids"] = shuffled_ids

                _seed_geometry_paired_forward(torch_module, device=device)
                with _autocast(torch_module, use_bf16):
                    aligned_outputs = model(
                        **aligned_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                _seed_geometry_paired_forward(torch_module, device=device)
                with _autocast(torch_module, use_bf16):
                    shuffled_outputs = model(
                        **shuffled_inputs,
                        use_cache=False,
                        return_dict=True,
                    )

                aligned_loss = _finite_loss(
                    torch_module, aligned_outputs.loss, condition_id
                )
                shuffled_loss = _finite_loss(
                    torch_module, shuffled_outputs.loss, condition_id
                )
                labels = aligned_inputs["labels"]
                active = labels.ne(-100)
                count = int(active.sum().item())
                target_tokens += count
                aligned_nll_sum += aligned_loss * count
                shuffled_nll_sum += shuffled_loss * count
                encoder_tokens += sum(aligned_batch.ce_batch.input_lengths)
                members += len(rows)
    finally:
        torch_module.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch_module.cuda.set_rng_state_all(cuda_rng_state)

    if target_tokens == 0:
        raise PF1TrainingError("PF-1 dev reader produced no supervised tokens")
    aligned_nll = aligned_nll_sum / target_tokens
    shuffled_nll = shuffled_nll_sum / target_tokens
    bucket_sizes = [
        {
            "model_atom_count": atom_count,
            "members": atom_counts.count(atom_count),
        }
        for atom_count in sorted(set(atom_counts))
    ]
    excluded_singletons = [
        {
            "schedule_index": dev_rows[index].schedule_index,
            "record_id": dev_rows[index].atom_record.record_id,
            "model_atom_count": atom_counts[index],
        }
        for index in plan.excluded_singleton_indices
    ]
    return {
        "update": GEOMETRY_PERTURBATION_UPDATE,
        "dev_members": len(dev_rows),
        "eligible_members": members,
        "coverage_fraction": members / len(dev_rows),
        "excluded_singletons": excluded_singletons,
        "encoder_nonpadding_tokens": encoder_tokens,
        "supervised_target_tokens": target_tokens,
        "aligned_nll": aligned_nll,
        "shuffled_nll": shuffled_nll,
        "delta_nll": shuffled_nll - aligned_nll,
        "delta_definition": "shuffled_nll_minus_aligned_nll",
        "derangement_seed": GEOMETRY_DERANGEMENT_SEED,
        "paired_forward_seed": GEOMETRY_PAIRED_FORWARD_SEED,
        "self_pairs": 0,
        "no_self_pairing": all(
            donor != recipient
            for recipient, donor in zip(
                plan.eligible_indices, plan.donor_indices
            )
        ),
        "atom_count_parity_pairs": sum(
            atom_counts[recipient] == atom_counts[donor]
            for recipient, donor in zip(
                plan.eligible_indices, plan.donor_indices
            )
        ),
        "atom_count_mismatches": 0,
        "model_atom_count_buckets": bucket_sizes,
        "recipient_ce_mask_carrier_source_preserved": True,
        "causal_effect_claim": False,
    }


def write_pf1_checkpoint(
    *,
    output_dir: Path,
    condition_id: str,
    update: int,
    model: Any,
    optimizer: Any,
    scheduler: PF1LearningRateSchedule,
    cursor_state: dict[str, int],
    torch_module: Any,
    training_progress: Mapping[str, object] | None = None,
) -> str:
    """Write one complete, versioned wrapper-and-training recovery state."""

    if update not in CHECKPOINT_UPDATES:
        raise PF1TrainingError("PF-1 checkpoints are restricted to steps 500 and 1000")
    if scheduler.completed_updates != update:
        raise PF1TrainingError(
            "checkpoint update differs from the learning-rate schedule state"
        )

    checkpoint_dir = output_dir / condition_id / f"step-{update:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    cuda_rng_state = None
    if torch_module.cuda.is_available():
        cuda_rng_state = torch_module.cuda.get_rng_state_all()
    torch_module.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "condition_id": condition_id,
            "completed_updates": update,
            "optimization_protocol": asdict(scheduler.protocol),
            # FourGridT5Wrapper is an nn.Module, not a PreTrainedModel.  Its
            # state dict deliberately includes both the complete T5 and the
            # shared geometry-fusion module for every condition.
            "wrapper_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "data_cursor": cursor_state,
            "rng_state": {
                "cpu": torch_module.random.get_rng_state(),
                "cuda_all": cuda_rng_state,
            },
            "training_progress": dict(training_progress or {}),
        },
        checkpoint_dir / "training_state.pt",
    )
    return str(checkpoint_dir)


def load_pf1_checkpoint(
    *,
    checkpoint_dir: Path,
    condition_id: str,
    model: Any,
    optimizer: Any,
    scheduler: PF1LearningRateSchedule,
    cursor: _TrainCursor,
    torch_module: Any,
) -> dict[str, object]:
    """Restore a checkpoint at the next-batch/next-update boundary.

    RNG restoration is intentionally last: model/optimizer construction and
    state loading cannot perturb the random stream used by the next training
    update.  ``torch.load`` maps tensors to CPU; ``load_state_dict`` then moves
    optimizer slots to the device of their corresponding model parameters.
    """

    checkpoint_dir = Path(checkpoint_dir)
    payload = torch_module.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise PF1TrainingError("PF-1 checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PF1TrainingError("unsupported PF-1 checkpoint schema")
    if payload.get("condition_id") != condition_id:
        raise PF1TrainingError("PF-1 checkpoint condition does not match the model")
    if payload.get("optimization_protocol") != asdict(scheduler.protocol):
        raise PF1TrainingError(
            "PF-1 checkpoint optimization protocol differs from the frozen run"
        )

    completed_updates = payload.get("completed_updates")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates not in CHECKPOINT_UPDATES
    ):
        raise PF1TrainingError("invalid PF-1 checkpoint completed update")
    scheduler_state = payload.get("scheduler_state_dict")
    data_cursor = payload.get("data_cursor")
    rng_state = payload.get("rng_state")
    if not isinstance(scheduler_state, Mapping):
        raise PF1TrainingError("PF-1 checkpoint lacks scheduler state")
    if scheduler_state.get("completed_updates") != completed_updates:
        raise PF1TrainingError("checkpoint and scheduler update counts differ")
    if not isinstance(data_cursor, Mapping):
        raise PF1TrainingError("PF-1 checkpoint lacks data cursor state")
    if not isinstance(rng_state, Mapping) or rng_state.get("cpu") is None:
        raise PF1TrainingError("PF-1 checkpoint lacks RNG state")

    model.load_state_dict(payload["wrapper_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(dict(scheduler_state))
    cursor.load_state_dict(data_cursor)
    torch_module.random.set_rng_state(rng_state["cpu"])
    cuda_rng_state = rng_state.get("cuda_all")
    if cuda_rng_state is not None:
        if not torch_module.cuda.is_available():
            raise PF1TrainingError(
                "CUDA RNG state cannot be restored without a CUDA runtime"
            )
        torch_module.cuda.set_rng_state_all(cuda_rng_state)

    progress = payload.get("training_progress", {})
    if not isinstance(progress, Mapping):
        raise PF1TrainingError("PF-1 checkpoint training progress is invalid")
    return {
        "completed_updates": completed_updates,
        "training_progress": dict(progress),
        "checkpoint_dir": str(checkpoint_dir),
    }


def _train_one_condition(
    *,
    condition_id: str,
    reader: PF1RecordReader,
    tokenizer_runtime: Any,
    model: Any,
    device: Any,
    use_bf16: bool,
    output_dir: Path,
    protocol: PF1OptimizationProtocol,
    torch_module: Any,
    checkpoint_writer: Callable[..., str],
    resume_checkpoint: Path | None = None,
    train_prefetch_depth: int = TRAIN_PREFETCH_DEPTH,
) -> dict[str, object]:
    model.to(device)
    optimizer = build_pf1_optimizer(model, protocol)
    scheduler = PF1LearningRateSchedule(optimizer, protocol)
    cursor = _TrainCursor(reader, protocol.micro_batch_size)
    if train_prefetch_depth < 0:
        raise PF1TrainingError("PF-1 train prefetch depth must be nonnegative")
    data_lock = threading.Lock() if train_prefetch_depth else None
    evaluations: list[dict[str, object]] = []
    checkpoints: list[str] = []
    preclip_norms: list[float] = []
    clipped_updates = 0
    train_nll_sum = 0.0
    encoder_tokens = 0
    target_tokens = 0
    members_seen = 0
    microbatch_member_counts: list[int] = []
    update_member_counts: list[int] = []
    elapsed_before_resume = 0.0
    completed_updates = 0
    learning_rate = float(optimizer.param_groups[0]["lr"])
    geometry_sensitivity: dict[str, object] | None = None

    model.train()
    if device.type == "cuda":
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats(device)
    if resume_checkpoint is not None:
        restored = load_pf1_checkpoint(
            checkpoint_dir=resume_checkpoint,
            condition_id=condition_id,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
            torch_module=torch_module,
        )
        completed_updates = int(restored["completed_updates"])
        progress = restored["training_progress"]
        assert isinstance(progress, dict)
        evaluations = list(progress.get("evaluations", ()))
        checkpoints = list(progress.get("checkpoints", ()))
        if str(resume_checkpoint) not in checkpoints:
            checkpoints.append(str(resume_checkpoint))
        preclip_norms = [float(value) for value in progress.get("preclip_norms", ())]
        clipped_updates = int(progress.get("clipped_updates", 0))
        train_nll_sum = float(progress.get("train_nll_sum", 0.0))
        encoder_tokens = int(progress.get("encoder_tokens", 0))
        target_tokens = int(progress.get("target_tokens", 0))
        members_seen = int(progress.get("members_seen", 0))
        restored_microbatch_counts = progress.get("microbatch_member_counts")
        restored_update_counts = progress.get("update_member_counts")
        if restored_microbatch_counts is None or restored_update_counts is None:
            # Checkpoints predating the drop_last=False contract could only
            # contain full micro-batches, so their exposure is reconstructible.
            microbatch_member_counts = [protocol.micro_batch_size] * (
                completed_updates * protocol.gradient_accumulation_steps
            )
            update_member_counts = [protocol.effective_batch_size] * completed_updates
        else:
            microbatch_member_counts = [
                int(value) for value in restored_microbatch_counts
            ]
            update_member_counts = [int(value) for value in restored_update_counts]
        elapsed_before_resume = float(progress.get("wall_seconds", 0.0))
        learning_rate = float(
            progress.get("last_update_learning_rate", learning_rate)
        )
        restored_geometry_sensitivity = progress.get("geometry_sensitivity")
        if restored_geometry_sensitivity is not None:
            if not isinstance(restored_geometry_sensitivity, Mapping):
                raise PF1TrainingError(
                    "checkpoint geometry sensitivity diagnostic is invalid"
                )
            geometry_sensitivity = dict(restored_geometry_sensitivity)
    else:
        evaluations.append(
            {
                "update": 0,
                **evaluate_pf1_condition(
                    model,
                    condition_id=condition_id,
                    reader=reader,
                    tokenizer_runtime=tokenizer_runtime,
                    device=device,
                    use_bf16=use_bf16,
                    protocol=protocol,
                    torch_module=torch_module,
                ),
            }
        )
    model.train()
    started = time.perf_counter()

    remaining_updates = protocol.total_updates - completed_updates
    committed_cursor_state = dict(cursor.state_dict())
    if train_prefetch_depth:
        update_stream_context: Any = _OrderedTrainPrefetch(
            cursor=cursor,
            total_updates=remaining_updates,
            depth=train_prefetch_depth,
            gradient_accumulation_steps=protocol.gradient_accumulation_steps,
            condition_id=condition_id,
            tokenizer_runtime=tokenizer_runtime,
            data_lock=data_lock,
        )
    else:
        synchronous_stream = (
            _prepare_train_update(
                cursor=cursor,
                gradient_accumulation_steps=protocol.gradient_accumulation_steps,
                condition_id=condition_id,
                tokenizer_runtime=tokenizer_runtime,
                data_lock=None,
            )
            for _ in range(remaining_updates)
        )
        update_stream_context = nullcontext(synchronous_stream)

    with update_stream_context as update_stream:
        for update, prepared in zip(
            range(completed_updates + 1, protocol.total_updates + 1),
            update_stream,
        ):
            batches = prepared.batches
            current_microbatch_counts = [
                len(batch.ce_batch.record_ids) for batch in batches
            ]
            if any(
                count <= 0 or count > protocol.micro_batch_size
                for count in current_microbatch_counts
            ):
                raise PF1TrainingError(
                    "PF-1 collator produced an empty or oversized micro-batch"
                )
            microbatch_member_counts.extend(current_microbatch_counts)
            update_member_counts.append(sum(current_microbatch_counts))

            update_target_tokens = sum(
                sum(batch.ce_batch.target_lengths) for batch in batches
            )
            optimizer.zero_grad(set_to_none=True)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for batch in batches:
                batch_target_tokens = sum(batch.ce_batch.target_lengths)
                encoded = to_four_grid_batch_encoding(batch, device=device)
                forward_inputs = select_four_grid_forward_inputs(encoded)
                with _autocast(torch_module, use_bf16):
                    outputs = model(
                        **forward_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                    loss = outputs.loss
                loss_value = _finite_loss(torch_module, loss, condition_id)
                (loss * (batch_target_tokens / update_target_tokens)).backward()
                train_nll_sum += loss_value * batch_target_tokens
                target_tokens += batch_target_tokens
                encoder_tokens += sum(batch.ce_batch.input_lengths)
                members_seen += len(batch.ce_batch.record_ids)

            preclip_norm = clip_pf1_gradients(model, protocol)
            if not math.isfinite(preclip_norm):
                raise PF1TrainingError(
                    condition_id + " produced a non-finite gradient norm"
                )
            preclip_norms.append(preclip_norm)
            clipped_updates += int(preclip_norm > protocol.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            # Commit only after the optimizer update has succeeded.  The
            # producer may already be decoding later updates, but those reads
            # must never move a checkpoint's resume boundary.
            committed_cursor_state = dict(prepared.committed_cursor_state)

            if update in EVALUATION_UPDATES:
                model.train(False)
                evaluations.append(
                    {
                        "update": update,
                        **evaluate_pf1_condition(
                            model,
                            condition_id=condition_id,
                            reader=reader,
                            tokenizer_runtime=tokenizer_runtime,
                            device=device,
                            use_bf16=use_bf16,
                            protocol=protocol,
                            torch_module=torch_module,
                            data_lock=data_lock,
                        ),
                    }
                )
                if (
                    update == GEOMETRY_PERTURBATION_UPDATE
                    and condition_id in ("A1", "M1")
                ):
                    geometry_guard = (
                        data_lock if data_lock is not None else nullcontext()
                    )
                    with geometry_guard:
                        geometry_sensitivity = evaluate_pf1_geometry_sensitivity(
                            model,
                            condition_id=condition_id,
                            reader=reader,
                            tokenizer_runtime=tokenizer_runtime,
                            device=device,
                            use_bf16=use_bf16,
                            protocol=protocol,
                            torch_module=torch_module,
                        )
                model.train()
            if update in CHECKPOINT_UPDATES:
                checkpoint_path = checkpoint_writer(
                    output_dir=output_dir,
                    condition_id=condition_id,
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cursor_state=committed_cursor_state,
                    torch_module=torch_module,
                    training_progress={
                        "evaluations": evaluations,
                        "checkpoints": checkpoints,
                        "preclip_norms": preclip_norms,
                        "clipped_updates": clipped_updates,
                        "train_nll_sum": train_nll_sum,
                        "encoder_tokens": encoder_tokens,
                        "target_tokens": target_tokens,
                        "members_seen": members_seen,
                        "microbatch_member_counts": microbatch_member_counts,
                        "update_member_counts": update_member_counts,
                        "wall_seconds": elapsed_before_resume
                        + (time.perf_counter() - started),
                        "last_update_learning_rate": learning_rate,
                        "geometry_sensitivity": geometry_sensitivity,
                    },
                )
                checkpoints.append(checkpoint_path)
    elapsed = elapsed_before_resume + (time.perf_counter() - started)
    peak_memory = (
        int(torch_module.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    if sum(update_member_counts) != members_seen:
        raise PF1TrainingError("PF-1 member exposure accounting is inconsistent")
    return {
        "condition": condition_id,
        "optimizer_updates": protocol.total_updates,
        "train_prefetch_depth": train_prefetch_depth,
        "members_seen": members_seen,
        "nominal_effective_batch_size": protocol.effective_batch_size,
        "short_microbatches": sum(
            count < protocol.micro_batch_size for count in microbatch_member_counts
        ),
        "min_microbatch_members": min(microbatch_member_counts),
        "max_microbatch_members": max(microbatch_member_counts),
        "mean_microbatch_members": statistics.fmean(microbatch_member_counts),
        "min_members_per_update": min(update_member_counts),
        "max_members_per_update": max(update_member_counts),
        "mean_members_per_update": statistics.fmean(update_member_counts),
        "train_encoder_nonpadding_tokens": encoder_tokens,
        "train_supervised_target_tokens": target_tokens,
        "train_token_weighted_nll": train_nll_sum / target_tokens,
        "wall_seconds": elapsed,
        "members_per_second": members_seen / elapsed,
        "encoder_tokens_per_second": encoder_tokens / elapsed,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clipped_updates": clipped_updates,
        "clip_rate": clipped_updates / protocol.total_updates,
        "peak_gpu_memory_bytes": peak_memory,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "final_data_cursor": committed_cursor_state,
        "last_update_learning_rate": learning_rate,
        "final_e3fp_shuffle_diagnostic": geometry_sensitivity,
    }


def execute_pf1_four_grid(
    *,
    reader: PF1RecordReader,
    tokenizer_runtime: Any,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    union_tokenizer_dir: Path,
    union_init_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
    expected_vocab_size: int,
    output_dir: Path,
    device: Any,
    use_bf16: bool,
    torch_module: Any,
    wrapper_loader: Callable[..., Any] = load_verified_four_grid_wrapper,
    checkpoint_writer: Callable[..., str] = write_pf1_checkpoint,
    protocol: PF1OptimizationProtocol = FROZEN_PF1_PROTOCOL,
    resume_checkpoints: Mapping[str, Path] | None = None,
    condition_ids: Sequence[str] = CONDITION_ORDER,
) -> dict[str, object]:
    """Train the requested cells with one shared frozen PF-1 contract.

    The default remains the original sequential four-grid run.  Passing one
    condition supports one-process-per-GPU execution without duplicating the
    scientific training loop.
    """

    output_dir = Path(output_dir)
    requested_conditions = tuple(condition_ids)
    if not (
        requested_conditions == CONDITION_ORDER
        or (
            len(requested_conditions) == 1
            and requested_conditions[0] in CONDITION_ORDER
        )
    ):
        raise PF1TrainingError(
            "PF-1 execution requires the full grid or exactly one condition"
        )
    resume_by_condition = dict(resume_checkpoints or {})
    if len(resume_by_condition) > 1 or any(
        condition_id not in requested_conditions
        for condition_id in resume_by_condition
    ):
        raise PF1TrainingError(
            "PF-1 execution accepts at most one selected-condition checkpoint"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    conditions: list[dict[str, object]] = []
    for condition_id in requested_conditions:
        torch_module.manual_seed(FORWARD_SEED)
        if device.type == "cuda":
            torch_module.cuda.manual_seed_all(FORWARD_SEED)
        model = wrapper_loader(
            condition_id=condition_id,
            base_model_snapshot=base_model_snapshot,
            base_tokenizer_snapshot=base_tokenizer_snapshot,
            union_tokenizer_dir=union_tokenizer_dir,
            output_dir=union_init_dir,
            geometry_fusion_seed=geometry_fusion_seed,
            num_e3fp_embeddings=num_e3fp_embeddings,
        )
        if int(model.config.vocab_size) != expected_vocab_size:
            raise PF1TrainingError("wrapper vocabulary differs from the PF-1 tokenizer")
        try:
            conditions.append(
                _train_one_condition(
                    condition_id=condition_id,
                    reader=reader,
                    tokenizer_runtime=tokenizer_runtime,
                    model=model,
                    device=device,
                    use_bf16=use_bf16,
                    output_dir=output_dir,
                    protocol=protocol,
                    torch_module=torch_module,
                    checkpoint_writer=checkpoint_writer,
                    resume_checkpoint=resume_by_condition.get(condition_id),
                )
            )
        finally:
            model.zero_grad(set_to_none=True)
            del model
            gc.collect()
            if device.type == "cuda":
                torch_module.cuda.empty_cache()

    optimization = asdict(protocol)
    optimization["effective_batch_size"] = protocol.effective_batch_size
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "pf1_one_percent_failure_screen_only",
        "interpretation": {
            "single_paired_seed": True,
            "architecture_superiority_claim": False,
            "statistical_significance_claim": False,
            "raw_atom_motif_nll_directly_comparable": False,
            "final_update_used_for_comparison": protocol.total_updates,
        },
        "comparison_contract": {
            "A0_A1_same_members_order_and_corruption": True,
            "M0_M1_same_members_order_and_corruption": True,
            "all_cells_same_initialization_optimizer_and_update_budget": True,
            "atom_motif_mask_units_differ": True,
        },
        "data": {
            "train_members": reader.train_member_count,
            "dev_members": reader.dev_member_count,
            "train_corruption_seed": TRAIN_CORRUPTION_SEED,
            "train_corruption_changes_by_epoch": True,
            "dev_corruption_seed": DEV_CORRUPTION_SEED,
            "dev_corruption_epoch": DEV_CORRUPTION_EPOCH,
            "mask_probability": MASK_PROBABILITY,
        },
        "optimization": optimization,
        "precision": "bf16_autocast" if use_bf16 else "test_or_debug_precision",
        "evaluation_updates": list(EVALUATION_UPDATES),
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "resumed_condition": next(iter(resume_by_condition), None),
        "execution": {
            "requested_conditions": list(requested_conditions),
            "complete_four_grid": requested_conditions == CONDITION_ORDER,
            "parallelizable_one_condition_per_process": True,
            "forward_seed": FORWARD_SEED,
            "geometry_fusion_seed": geometry_fusion_seed,
            "num_e3fp_embeddings": num_e3fp_embeddings,
            "expected_vocab_size": expected_vocab_size,
            "base_model_snapshot": str(Path(base_model_snapshot).resolve()),
            "base_tokenizer_snapshot": str(
                Path(base_tokenizer_snapshot).resolve()
            ),
            "union_tokenizer_dir": str(Path(union_tokenizer_dir).resolve()),
            "union_init_dir": str(Path(union_init_dir).resolve()),
        },
        "conditions": conditions,
    }
    manifest_name = (
        FOUR_GRID_MANIFEST_NAME
        if requested_conditions == CONDITION_ORDER
        else CONDITION_MANIFEST_NAME
    )
    (output_dir / manifest_name).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-protocol PF-1 training CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    parser.add_argument(
        "--condition-id",
        choices=CONDITION_ORDER,
        help="run exactly one cell; launch four such processes on four GPUs",
    )
    parser.add_argument("--resume-condition", choices=CONDITION_ORDER)
    parser.add_argument("--resume-checkpoint")
    return parser


def run(
    args: argparse.Namespace,
    *,
    torch_module: Any | None = None,
    reader_factory: Callable[[Path], PF1RecordReader] = PF1PairedReleaseReader,
    tokenizer_loader: Callable[..., Any] = load_verified_canary_union_tokenizer,
    executor: Callable[..., dict[str, object]] = execute_pf1_four_grid,
) -> dict[str, object]:
    """Load one verified PF-1 release and execute the frozen GPU protocol."""

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime boundary
            raise PF1TrainingError("PyTorch is required for PF-1 training") from exc
    if not torch_module.cuda.is_available():
        raise PF1TrainingError("PF-1 training requires one CUDA GPU")
    if not torch_module.cuda.is_bf16_supported():
        raise PF1TrainingError("PF-1 training requires CUDA BF16 support")

    paired_release = Path(args.paired_release).expanduser().resolve()
    base_model_snapshot = Path(args.base_model_snapshot).expanduser().resolve()
    base_tokenizer_snapshot = Path(args.base_tokenizer_snapshot).expanduser().resolve()
    union_init_dir = Path(args.union_init_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise PF1TrainingError("output_dir must be a new path")

    resume_condition = args.resume_condition
    resume_checkpoint = args.resume_checkpoint
    if (resume_condition is None) != (resume_checkpoint is None):
        raise PF1TrainingError(
            "resume-condition and resume-checkpoint must be provided together"
        )
    resume_checkpoints: dict[str, Path] = {}
    if resume_condition is not None:
        if resume_condition not in CONDITION_ORDER:
            raise PF1TrainingError("resume-condition is not a PF-1 condition")
        if args.condition_id is not None and resume_condition != args.condition_id:
            raise PF1TrainingError(
                "resume-condition must equal the selected condition-id"
            )
        resume_checkpoints[resume_condition] = (
            Path(resume_checkpoint).expanduser().resolve()
        )

    union_tokenizer_dir = paired_release / TOKENIZER_DIRECTORY
    tokenizer_build = tokenizer_loader(
        base_snapshot=base_tokenizer_snapshot,
        output_dir=union_tokenizer_dir,
    )
    reader = reader_factory(paired_release)
    return executor(
        reader=reader,
        tokenizer_runtime=tokenizer_build.runtime,
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=union_tokenizer_dir,
        union_init_dir=union_init_dir,
        geometry_fusion_seed=args.geometry_fusion_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
        expected_vocab_size=tokenizer_build.runtime.vocab_size,
        output_dir=output_dir,
        device=torch_module.device("cuda", 0),
        use_bf16=True,
        torch_module=torch_module,
        resume_checkpoints=resume_checkpoints,
        condition_ids=(args.condition_id,) if args.condition_id else CONDITION_ORDER,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PF1TrainingError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_UPDATES",
    "CONDITION_ORDER",
    "EVALUATION_UPDATES",
    "PF1RecordReader",
    "PF1TrainingError",
    "build_parser",
    "collate_pf1_condition",
    "evaluate_pf1_condition",
    "execute_pf1_four_grid",
    "load_pf1_checkpoint",
    "main",
    "run",
    "write_pf1_checkpoint",
]
