"""Formal PF-10 state-imputation stage for the B2D/F3D paired cells.

The GPU smoke and the formal run intentionally remain separate artifacts.  This
runner owns the preregistered S-stage only: T5 is frozen in evaluation mode,
the motif-state adapter is trainable, and each sampled motif contributes at
most one masked atom row while retaining at least one visible peer atom.

Operational cache worker counts are exposed because they do not change member
order, masks, optimizer updates, or model state.  Both cells can run in one
process and reuse one strictly decoded release cache.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from most_t5_next.p1.build_pf1_paired_release_v1 import PF1PairedReleaseReader
from most_t5_next.p1.pf1_optimization import (
    PF1LearningRateSchedule,
    PF1OptimizationProtocol,
    build_pf1_optimizer,
    clip_pf1_gradients,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .build_pf10_morgan_overlay_v1 import MorganAtomStateProvider
from .factorized_model_init_v1 import load_deterministic_factorized_model
from .factorized_motif_t5_v1 import FactorizedMotifT5V1
from .factorized_view_collator_v1 import (
    AtomStateProvider,
    FactorizedMotifViewBatch,
    collate_factorized_motif_view,
)
from .run_pf10_factorized_smoke_cli_v1 import (
    ADAPTER_SEED,
    NUM_E3FP_EMBEDDINGS,
    UNION_GEOMETRY_FUSION_SEED,
)


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-state-training/v1"
CHECKPOINT_SCHEMA = "most-t5-p2/pf10-factorized-state-checkpoint/v1"
TRAIN_SEED = 20260809
DEV_SEED = 20260810
DEV_MASK_EPOCH = 0
STATE_MASK_PROBABILITY = 0.15
FORMAL_STATE_MASKING = "motif_atom_row"
EVALUATION_UPDATES = (0, 625, 1250, 1875, 2500)
CHECKPOINT_UPDATES = (1250, 2500)
S_PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=1.0e-3,
    warmup_updates=250,
    total_updates=2500,
    final_learning_rate=1.0e-5,
    warmup_start_factor=0.1,
    gradient_clip_norm=1.0,
    weight_decay=0.0,
    micro_batch_size=64,
    gradient_accumulation_steps=2,
    beta1=0.9,
    beta2=0.999,
    epsilon=1.0e-6,
)


class PF10StateTrainingError(RuntimeError):
    """The formal PF-10 state stage violated its frozen contract."""


def _load_eligible_indices(path: Path, *, expected_split: str) -> tuple[int, ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                split_index = int(row["split_index"])
                motifs = tuple(row["eligible_motifs"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise PF10StateTrainingError(
                    f"malformed {expected_split} eligible membership line {line_number}"
                ) from exc
            if split_index < 0 or not motifs:
                raise PF10StateTrainingError("eligible membership contains an invalid row")
            rows.append(split_index)
    indices = tuple(rows)
    if not indices or tuple(sorted(set(indices))) != indices:
        raise PF10StateTrainingError(
            f"{expected_split} eligible split indices must be unique and increasing"
        )
    return indices


class _EligibleReader:
    """Ordered subset view over a validated PF-10 paired release."""

    def __init__(
        self,
        source: PF1PairedReleaseReader,
        *,
        train_indices: Sequence[int],
        dev_indices: Sequence[int],
    ) -> None:
        self.source = source
        self.train_indices = tuple(train_indices)
        self.dev_indices = tuple(dev_indices)
        if (
            not self.train_indices
            or not self.dev_indices
            or tuple(sorted(set(self.train_indices))) != self.train_indices
            or tuple(sorted(set(self.dev_indices))) != self.dev_indices
        ):
            raise PF10StateTrainingError(
                "eligible split indices must be nonempty, unique and increasing"
            )
        if self.train_indices[-1] >= source.train_member_count:
            raise PF10StateTrainingError("eligible train index exceeds paired membership")
        if self.dev_indices[-1] >= source.dev_member_count:
            raise PF10StateTrainingError("eligible dev index exceeds paired membership")
        self.train_member_count = len(self.train_indices)
        self.dev_member_count = len(self.dev_indices)

    def iter_train_epoch(self, *, epoch: int, batch_size: int) -> Iterator[tuple[Any, ...]]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise PF10StateTrainingError("epoch must be a non-negative integer")
        yield from self.source.iter_selected_split_indices(
            split="train",
            split_indices=self.train_indices,
            batch_size=batch_size,
        )

    def iter_dev(self, *, batch_size: int) -> Iterator[tuple[Any, ...]]:
        yield from self.source.iter_selected_split_indices(
            split="dev",
            split_indices=self.dev_indices,
            batch_size=batch_size,
        )


class _TrainCursor:
    def __init__(self, reader: _EligibleReader, micro_batch_size: int) -> None:
        self.reader = reader
        self.micro_batch_size = int(micro_batch_size)
        self.epoch = 0
        self.batch_in_epoch = 0
        self._iterator = iter(reader.iter_train_epoch(epoch=0, batch_size=micro_batch_size))

    def next(self) -> tuple[int, tuple[Any, ...]]:
        while True:
            try:
                rows = tuple(next(self._iterator))
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self._iterator = iter(
                    self.reader.iter_train_epoch(
                        epoch=self.epoch,
                        batch_size=self.micro_batch_size,
                    )
                )
                continue
            if not rows or len(rows) > self.micro_batch_size:
                raise PF10StateTrainingError("eligible reader yielded an invalid micro-batch")
            epoch = self.epoch
            self.batch_in_epoch += 1
            return epoch, rows

    def state_dict(self) -> dict[str, int]:
        return {"next_epoch": self.epoch, "next_batch_in_epoch": self.batch_in_epoch}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        epoch = state.get("next_epoch")
        batch = state.get("next_batch_in_epoch")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(batch, bool)
            or not isinstance(batch, int)
            or batch < 0
        ):
            raise PF10StateTrainingError("invalid state-stage cursor checkpoint")
        iterator = iter(
            self.reader.iter_train_epoch(epoch=epoch, batch_size=self.micro_batch_size)
        )
        for _ in range(batch):
            try:
                rows = tuple(next(iterator))
            except StopIteration as exc:
                raise PF10StateTrainingError("state-stage cursor exceeds its epoch") from exc
            if not rows or len(rows) > self.micro_batch_size:
                raise PF10StateTrainingError("invalid replayed state micro-batch")
        self.epoch = epoch
        self.batch_in_epoch = batch
        self._iterator = iterator


def _autocast(device: torch.device, use_bf16: bool) -> Any:
    if not use_bf16:
        return nullcontext()
    if device.type != "cuda":
        raise PF10StateTrainingError("BF16 formal state training requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _motif_records(rows: Sequence[Any]) -> tuple[Any, ...]:
    records = tuple(getattr(row, "motif_record", None) for row in rows)
    if any(record is None for record in records):
        raise PF10StateTrainingError("paired reader row lacks a motif record")
    return records


def _collate_state(
    rows: Sequence[Any],
    *,
    tokenizer: Any,
    provider: AtomStateProvider | None,
    seed: int,
    epoch: int,
    device: torch.device,
) -> FactorizedMotifViewBatch:
    return collate_factorized_motif_view(
        _motif_records(rows),
        tokenizer=tokenizer,
        objective_mode="state",
        seed=seed,
        epoch=epoch,
        state_mask_probability=STATE_MASK_PROBABILITY,
        state_masking_strategy=FORMAL_STATE_MASKING,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        atom_state_provider=provider,
        device=device,
    )


def _target_counts(batch: FactorizedMotifViewBatch) -> dict[int, int]:
    if batch.state_target_mask is None:
        raise PF10StateTrainingError("state batch lacks target mask")
    return {
        level: int(batch.state_target_mask[..., level].sum().item())
        for level in (1, 2)
    }


def evaluate_state_stage(
    model: FactorizedMotifT5V1,
    *,
    reader: _EligibleReader,
    tokenizer: Any,
    provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, object]:
    model.eval()
    level_nll_sums = {1: 0.0, 2: 0.0}
    level_correct = {1: 0, 2: 0}
    level_counts = {1: 0, 2: 0}
    members = 0
    with torch.no_grad():
        for rows in reader.iter_dev(batch_size=S_PROTOCOL.micro_batch_size):
            batch = _collate_state(
                rows,
                tokenizer=tokenizer,
                provider=provider,
                seed=DEV_SEED,
                epoch=DEV_MASK_EPOCH,
                device=device,
            )
            with _autocast(device, use_bf16):
                output = model(**batch.model_inputs())
            if output.state_logits is None or batch.state_target_mask is None or batch.state_target_ids is None:
                raise PF10StateTrainingError("state evaluation omitted logits or targets")
            for external_level, logit_index in ((1, 0), (2, 1)):
                mask = batch.state_target_mask[..., external_level]
                count = int(mask.sum().item())
                loss = output.state_level_losses.get(external_level)
                if count <= 0 or not isinstance(loss, Tensor) or not bool(torch.isfinite(loss).item()):
                    raise PF10StateTrainingError("state evaluation produced an invalid level loss")
                logits = output.state_logits[..., logit_index, :][mask]
                targets = batch.state_target_ids[..., external_level][mask]
                level_nll_sums[external_level] += float(loss.float().item()) * count
                level_correct[external_level] += int((logits.argmax(dim=-1) == targets).sum().item())
                level_counts[external_level] += count
            members += len(rows)
    if members != reader.dev_member_count or any(value <= 0 for value in level_counts.values()):
        raise PF10StateTrainingError("state evaluation did not exhaust its frozen dev domain")
    means = {level: level_nll_sums[level] / level_counts[level] for level in (1, 2)}
    return {
        "members": members,
        "level_target_counts": {str(level): level_counts[level] for level in (1, 2)},
        "level_nll": {str(level): means[level] for level in (1, 2)},
        "level_accuracy": {
            str(level): level_correct[level] / level_counts[level] for level in (1, 2)
        },
        "weighted_state_loss": means[1] + model.state_level2_weight * means[2],
        "mask_seed": DEV_SEED,
        "mask_epoch": DEV_MASK_EPOCH,
    }


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    cuda = state.get("torch_cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)  # type: ignore[arg-type]


def _write_checkpoint(
    *,
    path: Path,
    cell: str,
    state_kind: str,
    update: int,
    model: FactorizedMotifT5V1,
    optimizer: Any,
    scheduler: PF1LearningRateSchedule,
    cursor: _TrainCursor,
    progress: Mapping[str, object],
) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    destination = path / "training_state.pt"
    staging = path / "training_state.pt.tmp"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "cell": cell,
            "stage": "S",
            "state_kind": state_kind,
            "completed_updates": update,
            "protocol": asdict(S_PROTOCOL),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cursor_state_dict": cursor.state_dict(),
            "rng_state": _rng_state(),
            "progress": dict(progress),
        },
        staging,
    )
    staging.replace(destination)
    return path


def _load_checkpoint(
    path: Path,
    *,
    cell: str,
    state_kind: str,
    model: FactorizedMotifT5V1,
    optimizer: Any,
    scheduler: PF1LearningRateSchedule,
    cursor: _TrainCursor,
) -> tuple[int, dict[str, object]]:
    payload = torch.load(Path(path) / "training_state.pt", map_location="cpu")
    if not isinstance(payload, Mapping) or not (
        payload.get("schema_version") == CHECKPOINT_SCHEMA
        and payload.get("cell") == cell
        and payload.get("stage") == "S"
        and payload.get("state_kind") == state_kind
        and payload.get("protocol") == asdict(S_PROTOCOL)
    ):
        raise PF10StateTrainingError("resume checkpoint differs from the formal S contract")
    completed = int(payload["completed_updates"])
    model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
    optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
    scheduler.load_state_dict(payload["scheduler_state_dict"])  # type: ignore[arg-type]
    cursor.load_state_dict(payload["cursor_state_dict"])  # type: ignore[arg-type]
    _restore_rng(payload["rng_state"])  # type: ignore[arg-type]
    progress = payload.get("progress")
    if not isinstance(progress, Mapping):
        raise PF10StateTrainingError("resume checkpoint lacks training progress")
    return completed, dict(progress)


def run_state_cell(
    *,
    cell: str,
    reader: _EligibleReader,
    tokenizer: Any,
    model: FactorizedMotifT5V1,
    output_dir: Path,
    provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
    resume_checkpoint: Path | None = None,
) -> dict[str, object]:
    if cell not in {"B2D", "F3D"}:
        raise PF10StateTrainingError("formal S-stage admits B2D and F3D only")
    state_kind = "e3fp" if provider is None else str(provider.state_kind)
    if cell == "B2D" and (provider is None or state_kind == "e3fp"):
        raise PF10StateTrainingError("B2D requires its Morgan atom-state provider")
    if cell == "F3D" and provider is not None:
        raise PF10StateTrainingError("F3D consumes aligned E3FP from the paired release")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.requires_grad_(False)
    model.adapter.requires_grad_(True)
    model.to(device)
    optimizer = build_pf1_optimizer(model, S_PROTOCOL)
    scheduler = PF1LearningRateSchedule(optimizer, S_PROTOCOL)
    cursor = _TrainCursor(reader, S_PROTOCOL.micro_batch_size)
    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    completed = 0
    evaluations: list[dict[str, object]] = []
    preclip_norms: list[float] = []
    clipped_updates = 0
    members_seen = 0
    target_counts_total = {1: 0, 2: 0}
    short_microbatches = 0
    checkpoints: list[str] = []
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        completed, progress = _load_checkpoint(
            Path(resume_checkpoint),
            cell=cell,
            state_kind=state_kind,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=cursor,
        )
        evaluations = list(progress.get("evaluations", ()))
        preclip_norms = [float(value) for value in progress.get("preclip_norms", ())]
        clipped_updates = int(progress.get("clipped_updates", 0))
        members_seen = int(progress.get("members_seen", 0))
        restored_counts = progress.get("target_counts", {})
        target_counts_total = {level: int(restored_counts.get(str(level), 0)) for level in (1, 2)}
        short_microbatches = int(progress.get("short_microbatches", 0))
        checkpoints = [str(value) for value in progress.get("checkpoints", ())]
        elapsed_before = float(progress.get("wall_seconds", 0.0))
    else:
        evaluations.append(
            {"update": 0, **evaluate_state_stage(
                model,
                reader=reader,
                tokenizer=tokenizer,
                provider=provider,
                device=device,
                use_bf16=use_bf16,
            )}
        )

    started = time.perf_counter()
    for update in range(completed + 1, S_PROTOCOL.total_updates + 1):
        model.train()
        model.t5.eval()
        model.adapter.train()
        collated: list[FactorizedMotifViewBatch] = []
        member_count = 0
        for _ in range(S_PROTOCOL.gradient_accumulation_steps):
            epoch, rows = cursor.next()
            short_microbatches += int(len(rows) < S_PROTOCOL.micro_batch_size)
            member_count += len(rows)
            collated.append(
                _collate_state(
                    rows,
                    tokenizer=tokenizer,
                    provider=provider,
                    seed=TRAIN_SEED,
                    epoch=epoch,
                    device=device,
                )
            )
        counts = [_target_counts(batch) for batch in collated]
        totals = {level: sum(row[level] for row in counts) for level in (1, 2)}
        if any(total <= 0 for total in totals.values()):
            raise PF10StateTrainingError("formal S update lacks level-1/2 targets")
        optimizer.zero_grad(set_to_none=True)
        for batch, batch_counts in zip(collated, counts):
            with _autocast(device, use_bf16):
                output = model(**batch.model_inputs())
                contribution = None
                for level in (1, 2):
                    level_loss = output.state_level_losses.get(level)
                    if not isinstance(level_loss, Tensor):
                        raise PF10StateTrainingError("formal S forward omitted a level loss")
                    weight = 1.0 if level == 1 else model.state_level2_weight
                    term = level_loss * weight * (batch_counts[level] / totals[level])
                    contribution = term if contribution is None else contribution + term
            assert contribution is not None
            contribution.backward()
        preclip = clip_pf1_gradients(model, S_PROTOCOL)
        if not math.isfinite(preclip):
            raise PF10StateTrainingError("formal S gradient norm is non-finite")
        preclip_norms.append(preclip)
        clipped_updates += int(preclip > S_PROTOCOL.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        members_seen += member_count
        for level in (1, 2):
            target_counts_total[level] += totals[level]

        if update in EVALUATION_UPDATES:
            evaluations.append(
                {"update": update, **evaluate_state_stage(
                    model,
                    reader=reader,
                    tokenizer=tokenizer,
                    provider=provider,
                    device=device,
                    use_bf16=use_bf16,
                )}
            )
        if update in CHECKPOINT_UPDATES:
            wall = elapsed_before + (time.perf_counter() - started)
            progress = {
                "evaluations": evaluations,
                "preclip_norms": preclip_norms,
                "clipped_updates": clipped_updates,
                "members_seen": members_seen,
                "target_counts": {str(level): target_counts_total[level] for level in (1, 2)},
                "short_microbatches": short_microbatches,
                "checkpoints": checkpoints,
                "wall_seconds": wall,
            }
            checkpoint = _write_checkpoint(
                path=output_dir / f"step-{update:04d}",
                cell=cell,
                state_kind=state_kind,
                update=update,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=cursor,
                progress=progress,
            )
            checkpoints.append(str(checkpoint))

    elapsed = elapsed_before + (time.perf_counter() - started)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cell": cell,
        "stage": "S",
        "state_kind": state_kind,
        "protocol": asdict(S_PROTOCOL),
        "formal_state_masking": FORMAL_STATE_MASKING,
        "train_eligible_members": reader.train_member_count,
        "dev_eligible_members": reader.dev_member_count,
        "optimizer_updates": S_PROTOCOL.total_updates,
        "members_seen": members_seen,
        "target_counts": {str(level): target_counts_total[level] for level in (1, 2)},
        "short_microbatches": short_microbatches,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clipped_updates": clipped_updates,
        "clip_rate": clipped_updates / S_PROTOCOL.total_updates,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "wall_seconds": elapsed,
        "members_per_second": members_seen / elapsed,
        "precision": "bf16_autocast" if use_bf16 else "test_precision",
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "scientific_boundary": "state learnability only; B2D/F3D grammar effects are not ranked here",
    }
    (output_dir / "state_training_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    paired_release = Path(args.paired_release).expanduser().resolve()
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired_release / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired_release)
    cache_report = reader.warm_decoded_record_cache(
        workers=args.cache_workers,
        max_pending=args.cache_max_pending,
    )
    support = Path(args.support_census)
    eligible_reader = _EligibleReader(
        reader,
        train_indices=_load_eligible_indices(
            support / "train_state_eligible_membership.jsonl", expected_split="train"
        ),
        dev_indices=_load_eligible_indices(
            support / "dev_state_eligible_membership.jsonl", expected_split="dev"
        ),
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PF10StateTrainingError("one CUDA BF16 device is required")
    device = torch.device("cuda:0")
    requested = ("B2D", "F3D") if args.cell == "both" else (args.cell,)
    results: dict[str, object] = {}
    provider = MorganAtomStateProvider(Path(args.morgan_overlay))
    try:
        for cell in requested:
            model = load_deterministic_factorized_model(
                base_model_snapshot=Path(args.base_model_snapshot),
                base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
                union_tokenizer_dir=paired_release / "union_tokenizer",
                union_init_dir=Path(args.union_init_dir),
                union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
                adapter_seed=ADAPTER_SEED,
                num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            )
            results[cell] = run_state_cell(
                cell=cell,
                reader=eligible_reader,
                tokenizer=tokenizer.runtime,
                model=model,
                output_dir=Path(args.output_dir) / cell,
                provider=provider if cell == "B2D" else None,
                device=device,
                use_bf16=True,
                resume_checkpoint=(
                    Path(args.resume_checkpoint)
                    if args.resume_checkpoint and len(requested) == 1
                    else None
                ),
            )
            del model
            torch.cuda.empty_cache()
    finally:
        provider.close()
    combined = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cells": results,
        "decoded_cache_warmup": cache_report,
        "decoded_cache_final": reader.decoded_record_cache_stats(),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "state_pair_manifest.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return combined


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D", "both"), default="both")
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--support-census", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--cache-workers", type=int, default=12)
    parser.add_argument("--cache-max-pending", type=int, default=48)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = _parser().parse_args(argv)
    if args.cache_workers <= 0 or args.cache_max_pending < args.cache_workers:
        raise SystemExit("cache worker bounds are invalid")
    if args.cell == "both" and args.resume_checkpoint:
        raise SystemExit("resume checkpoint requires one selected cell")
    report = run_cli(args)
    print(json.dumps({"status": report["status"], "cells": list(report["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PF10StateTrainingError",
    "SCHEMA_VERSION",
    "S_PROTOCOL",
    "evaluate_state_stage",
    "run_cli",
    "run_state_cell",
]
