"""Formal PF-10 V2 state-pretraining and frozen-encoder bridge stages.

One invocation runs one cell and one stage.  ``S`` trains only the V2 adapter
on canonical-addressed atom-state imputation.  ``B`` loads the completed S
checkpoint, freezes the adapter and T5 encoder in evaluation mode, and trains
only the decoder/LM head on the identity-masked, state-visible cross view.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DONOR_ATOM_MAP_NAME,
    PF1PairedReleaseReader,
)
from most_t5_next.p1.pf1_optimization import (
    PF1LearningRateSchedule,
    build_pf1_optimizer,
    clip_pf1_gradients,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .build_pf10_morgan_overlay_v1 import MorganAtomStateProvider
from .factorized_model_init_v2 import load_deterministic_factorized_model_v2
from .factorized_motif_t5_v2 import FACTORISATION_ID, FactorizedMotifT5V2
from .factorized_view_collator_v1 import AtomStateProvider
from .factorized_view_collator_v2 import (
    CanonicalAtomAddressProvider,
    FactorizedMotifViewBatchV2,
    GraphPortsCanonicalAtomAddressProvider,
    collate_factorized_motif_view_v2,
)
from .motif_geometry_adapter_v2 import ADAPTER_ID
from .run_pf10_factorized_smoke_cli_v1 import (
    ADAPTER_SEED,
    NUM_E3FP_EMBEDDINGS,
    UNION_GEOMETRY_FUSION_SEED,
)
from .run_pf10_factorized_state_v1 import (
    S_PROTOCOL as V1_S_PROTOCOL,
    _EligibleReader,
    _TrainCursor,
    _autocast,
    _load_eligible_indices,
    _restore_rng,
    _rng_state,
)
from .run_pf10_factorized_grammar_v1 import G_PROTOCOL as V1_G_PROTOCOL


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-v2-staged-training/v1"
CHECKPOINT_SCHEMA = "most-t5-p2/pf10-factorized-v2-staged-checkpoint/v1"
ADDRESS_CONTRACT = "graphports-canonical-local-atom-address/v1"
TRAIN_SEED = 20260810
DEV_SEED = 20260811
DEV_MASK_EPOCH = 0
STATE_MASK_PROBABILITY = 0.15
IDENTITY_MASK_PROBABILITY = 0.15
S_PROTOCOL = V1_S_PROTOCOL
B_PROTOCOL = replace(
    V1_G_PROTOCOL,
    base_learning_rate=1.0e-4,
    micro_batch_size=64,
    gradient_accumulation_steps=2,
)
S_EVALUATIONS = (0, 625, 1250, 1875, 2500)
S_CHECKPOINTS = (1250, 2500)
B_EVALUATIONS = (0, 2500, 5000, 7500, 10000)
B_CHECKPOINTS = (5000, 10000)
COUNTERFACTUAL_B_PROTOCOL = replace(
    B_PROTOCOL,
    total_updates=2500,
    warmup_updates=250,
    micro_batch_size=32,
    gradient_accumulation_steps=4,
)
COUNTERFACTUAL_B_EVALUATIONS = (0, 625, 1250, 1875, 2500)
COUNTERFACTUAL_B_CHECKPOINTS = (1250, 2500)


class PF10V2StagedTrainingError(RuntimeError):
    """A formal V2 stage violated its frozen scientific contract."""


def _records(rows: Sequence[Any]) -> tuple[Any, ...]:
    records = tuple(getattr(row, "motif_record", None) for row in rows)
    if any(record is None for record in records):
        raise PF10V2StagedTrainingError("paired row lacks a motif record")
    return records


def _collate(
    rows: Sequence[Any],
    *,
    objective: str,
    tokenizer: Any,
    address_provider: CanonicalAtomAddressProvider,
    state_provider: AtomStateProvider | None,
    seed: int,
    epoch: int,
    device: torch.device,
) -> FactorizedMotifViewBatchV2:
    return collate_factorized_motif_view_v2(
        _records(rows),
        tokenizer=tokenizer,
        objective_mode=objective,
        seed=seed,
        epoch=epoch,
        identity_mask_probability=IDENTITY_MASK_PROBABILITY,
        state_mask_probability=STATE_MASK_PROBABILITY,
        state_masking_strategy="motif_atom_row",
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        atom_state_provider=state_provider,
        atom_address_provider=address_provider,
        device=device,
    )


def _state_counts(batch: FactorizedMotifViewBatchV2) -> dict[int, int]:
    mask = batch.state_target_mask
    if mask is None:
        raise PF10V2StagedTrainingError("state batch lacks targets")
    return {level: int(mask[..., level].sum()) for level in (1, 2)}


def _identity_count(batch: FactorizedMotifViewBatchV2) -> int:
    if batch.labels is None:
        raise PF10V2StagedTrainingError("bridge batch lacks labels")
    return int((batch.labels != -100).sum())


def _counterfactual_token_mask(
    batch: FactorizedMotifViewBatchV2,
    shuffle_provider: Any,
) -> Tensor:
    """Select decoder identity tokens owned by actually replaced motifs."""

    labels = batch.labels
    owners = batch.label_to_motif
    if labels is None or owners is None or labels.shape != owners.shape:
        raise PF10V2StagedTrainingError(
            "counterfactual bridge requires decoder label ownership"
        )
    mask = torch.zeros_like(labels, dtype=torch.bool)
    changed_getter = getattr(shuffle_provider, "changed_motif_indices", None)
    if not callable(changed_getter):
        raise PF10V2StagedTrainingError(
            "counterfactual bridge requires changed-motif metadata"
        )
    for row_index, record_id in enumerate(batch.record_ids):
        changed = tuple(changed_getter(record_id))
        if not changed:
            continue
        changed_tensor = torch.as_tensor(
            changed,
            dtype=owners.dtype,
            device=owners.device,
        )
        mask[row_index] = (owners[row_index, :, None] == changed_tensor).any(-1)
    mask &= labels != -100
    return mask


def _concatenate_counterfactual_inputs(
    aligned: FactorizedMotifViewBatchV2,
    shuffled: FactorizedMotifViewBatchV2,
) -> dict[str, object]:
    """Build one aligned+shuffle forward without retaining two CUDA graphs."""

    if aligned.record_ids != shuffled.record_ids:
        raise PF10V2StagedTrainingError("counterfactual record order differs")
    if aligned.labels is None or shuffled.labels is None or not torch.equal(
        aligned.labels, shuffled.labels
    ):
        raise PF10V2StagedTrainingError("counterfactual decoder labels differ")
    left = aligned.model_inputs()
    right = shuffled.model_inputs()
    if set(left) != set(right):
        raise PF10V2StagedTrainingError("counterfactual model input keys differ")
    combined: dict[str, object] = {}
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, Tensor) and isinstance(right_value, Tensor):
            if left_value.shape[1:] != right_value.shape[1:]:
                raise PF10V2StagedTrainingError(
                    f"counterfactual tensor shape differs for {key}"
                )
            combined[key] = torch.cat((left_value, right_value), dim=0)
        elif left_value != right_value:
            raise PF10V2StagedTrainingError(
                f"counterfactual scalar input differs for {key}"
            )
        else:
            combined[key] = left_value
    return combined


def _token_nll(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise PF10V2StagedTrainingError("decoder logits and labels disagree")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)


def _set_stage_boundary(model: FactorizedMotifT5V2, stage: str) -> tuple[str, ...]:
    model.requires_grad_(False)
    if stage == "S":
        model.adapter.requires_grad_(True)
        return ("adapter",)
    if stage != "B":
        raise PF10V2StagedTrainingError("stage must be S or B")
    decoder = getattr(model.t5, "decoder", None)
    lm_head = getattr(model.t5, "lm_head", None)
    if not isinstance(decoder, nn.Module) or not isinstance(lm_head, nn.Module):
        raise PF10V2StagedTrainingError("bridge requires T5 decoder and LM head")
    decoder.requires_grad_(True)
    lm_head.requires_grad_(True)
    # Hugging Face T5 exposes the shared input embedding through both encoder
    # and decoder.embed_tokens.  Re-freeze it after opening the decoder or the
    # apparent decoder-only bridge would silently update the encoder's input
    # vocabulary as well.
    shared = getattr(model.t5, "shared", None)
    if isinstance(shared, nn.Module):
        shared.requires_grad_(False)
    decoder_embedding = getattr(decoder, "embed_tokens", None)
    if isinstance(decoder_embedding, nn.Module):
        decoder_embedding.requires_grad_(False)
    return ("t5.decoder_nonembedding", "t5.lm_head")


def _set_train_modes(model: FactorizedMotifT5V2, stage: str) -> None:
    if stage == "S":
        model.train()
        model.t5.eval()
        model.adapter.train()
    else:
        model.eval()
        model.t5.decoder.train()
        model.t5.lm_head.train()


def _evaluate_state(
    model: FactorizedMotifT5V2,
    *,
    reader: _EligibleReader,
    tokenizer: Any,
    address_provider: CanonicalAtomAddressProvider,
    state_provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
    memory_mode: str,
) -> dict[str, object]:
    model.eval()
    sums = {1: 0.0, 2: 0.0}
    counts = {1: 0, 2: 0}
    correct = {1: 0, 2: 0}
    members = 0
    with torch.no_grad():
        for rows in reader.iter_dev(batch_size=S_PROTOCOL.micro_batch_size):
            batch = _collate(
                rows,
                objective="state",
                tokenizer=tokenizer,
                address_provider=address_provider,
                state_provider=state_provider,
                seed=DEV_SEED,
                epoch=DEV_MASK_EPOCH,
                device=device,
            )
            inputs = batch.model_inputs()
            inputs["state_memory_mode"] = memory_mode
            with _autocast(device, use_bf16):
                output = model(**inputs)
            if output.state_logits is None or batch.state_target_ids is None:
                raise PF10V2StagedTrainingError("state evaluation omitted logits")
            for level, logit_index in ((1, 0), (2, 1)):
                mask = batch.state_target_mask[..., level]
                count = int(mask.sum())
                loss = output.state_level_losses.get(level)
                if count <= 0 or not isinstance(loss, Tensor):
                    raise PF10V2StagedTrainingError("invalid state evaluation target")
                logits = output.state_logits[..., logit_index, :][mask]
                targets = batch.state_target_ids[..., level][mask]
                sums[level] += float(loss) * count
                counts[level] += count
                correct[level] += int((logits.argmax(-1) == targets).sum())
            members += len(rows)
    if members != reader.dev_member_count:
        raise PF10V2StagedTrainingError("state evaluation did not exhaust dev")
    means = {level: sums[level] / counts[level] for level in (1, 2)}
    return {
        "members": members,
        "state_memory_mode": memory_mode,
        "level_target_counts": {str(k): counts[k] for k in (1, 2)},
        "level_nll": {str(k): means[k] for k in (1, 2)},
        "level_accuracy": {str(k): correct[k] / counts[k] for k in (1, 2)},
        "weighted_state_loss": means[1] + model.state_level2_weight * means[2],
    }


def _evaluate_identity(
    model: FactorizedMotifT5V2,
    *,
    reader: PF1PairedReleaseReader,
    tokenizer: Any,
    address_provider: CanonicalAtomAddressProvider,
    state_provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
    memory_mode: str,
) -> dict[str, object]:
    model.eval()
    nll_sum = 0.0
    targets = 0
    correct = 0
    members = 0
    with torch.no_grad():
        for rows in reader.iter_dev(batch_size=B_PROTOCOL.micro_batch_size):
            batch = _collate(
                rows,
                objective="cross_view",
                tokenizer=tokenizer,
                address_provider=address_provider,
                state_provider=state_provider,
                seed=DEV_SEED,
                epoch=DEV_MASK_EPOCH,
                device=device,
            )
            count = _identity_count(batch)
            inputs = batch.model_inputs()
            inputs["state_memory_mode"] = memory_mode
            with _autocast(device, use_bf16):
                output = model(**inputs)
            logits = getattr(output.t5_output, "logits", None)
            if not isinstance(logits, Tensor) or batch.labels is None:
                raise PF10V2StagedTrainingError("bridge evaluation omitted logits")
            mask = batch.labels != -100
            correct += int((logits.argmax(-1)[mask] == batch.labels[mask]).sum())
            nll_sum += float(output.loss) * count
            targets += count
            members += len(rows)
    if members != reader.dev_member_count or targets <= 0:
        raise PF10V2StagedTrainingError("bridge evaluation did not exhaust dev")
    return {
        "members": members,
        "target_tokens": targets,
        "token_weighted_nll": nll_sum / targets,
        "masked_token_accuracy": correct / targets,
        "state_memory_mode": memory_mode,
        "state_kind": "e3fp" if state_provider is None else str(state_provider.state_kind),
    }


def _checkpoint_payload(
    *, model: FactorizedMotifT5V2, optimizer: Any, scheduler: Any,
    cursor: _TrainCursor, stage: str, cell: str, state_kind: str,
    update: int, protocol: Any, progress: Mapping[str, object],
    trainable_modules: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "factorisation_id": FACTORISATION_ID,
        "adapter_id": ADAPTER_ID,
        "address_contract": ADDRESS_CONTRACT,
        "stage": stage,
        "cell": cell,
        "state_kind": state_kind,
        "completed_updates": update,
        "protocol": asdict(protocol),
        "trainable_modules": list(trainable_modules),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cursor_state_dict": cursor.state_dict(),
        "rng_state": _rng_state(),
        "progress": dict(progress),
    }


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    temporary = path / "training_state.pt.tmp"
    torch.save(dict(payload), temporary)
    temporary.replace(path / "training_state.pt")


def _load_checkpoint(
    path: Path, *, model: FactorizedMotifT5V2, optimizer: Any | None,
    scheduler: Any | None, cursor: _TrainCursor | None, stage: str,
    cell: str, state_kind: str, protocol: Any, restore_training: bool,
) -> tuple[int, dict[str, object]]:
    payload = torch.load(Path(path) / "training_state.pt", map_location="cpu")
    expected_modules = (
        ["adapter"]
        if stage == "S"
        else ["t5.decoder_nonembedding", "t5.lm_head"]
    )
    if not isinstance(payload, Mapping) or not all((
        payload.get("schema_version") == CHECKPOINT_SCHEMA,
        payload.get("factorisation_id") == FACTORISATION_ID,
        payload.get("adapter_id") == ADAPTER_ID,
        payload.get("address_contract") == ADDRESS_CONTRACT,
        payload.get("stage") == stage,
        payload.get("cell") == cell,
        payload.get("state_kind") == state_kind,
        payload.get("protocol") == asdict(protocol),
        payload.get("trainable_modules") == expected_modules,
    )):
        raise PF10V2StagedTrainingError("checkpoint differs from the V2 stage contract")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if restore_training:
        assert optimizer is not None and scheduler is not None and cursor is not None
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        cursor.load_state_dict(payload["cursor_state_dict"])
        _restore_rng(payload["rng_state"])
    progress = payload.get("progress", {})
    if not isinstance(progress, Mapping):
        raise PF10V2StagedTrainingError("checkpoint progress is malformed")
    return int(payload["completed_updates"]), dict(progress)


def run_stage(
    *, stage: str, cell: str, full_reader: PF1PairedReleaseReader,
    eligible_reader: _EligibleReader, tokenizer: Any,
    model: FactorizedMotifT5V2, address_provider: CanonicalAtomAddressProvider,
    output_dir: Path, state_provider: AtomStateProvider | None,
    device: torch.device, use_bf16: bool,
    s_checkpoint: Path | None = None,
    shuffle_provider: AtomStateProvider | None = None,
    train_shuffle_provider: AtomStateProvider | None = None,
    resume_checkpoint: Path | None = None,
    counterfactual_weight: float = 0.0,
    counterfactual_margin: float = 0.05,
) -> dict[str, object]:
    if stage not in {"S", "B"} or cell not in {"B2D", "F3D"}:
        raise PF10V2StagedTrainingError("invalid stage/cell")
    state_kind = "e3fp" if state_provider is None else str(state_provider.state_kind)
    if cell == "B2D" and state_provider is None:
        raise PF10V2StagedTrainingError("B2D requires Morgan state")
    if cell == "F3D" and state_provider is not None:
        raise PF10V2StagedTrainingError("F3D consumes persisted E3FP")
    if stage == "S" and s_checkpoint is not None:
        raise PF10V2StagedTrainingError("S cannot load an S checkpoint")
    if stage == "B" and s_checkpoint is None:
        raise PF10V2StagedTrainingError("B requires its completed S checkpoint")
    if (
        isinstance(counterfactual_weight, bool)
        or not isinstance(counterfactual_weight, (int, float))
        or float(counterfactual_weight) < 0.0
        or isinstance(counterfactual_margin, bool)
        or not isinstance(counterfactual_margin, (int, float))
        or float(counterfactual_margin) <= 0.0
    ):
        raise PF10V2StagedTrainingError("invalid counterfactual bridge contract")
    counterfactual = float(counterfactual_weight) > 0.0
    if counterfactual and (
        stage != "B"
        or cell != "F3D"
        or shuffle_provider is None
        or train_shuffle_provider is None
        or resume_checkpoint is not None
    ):
        raise PF10V2StagedTrainingError(
            "counterfactual bridge is a fresh F3D B run with matched state"
        )

    protocol = (
        S_PROTOCOL
        if stage == "S"
        else COUNTERFACTUAL_B_PROTOCOL if counterfactual else B_PROTOCOL
    )
    evaluation_updates = (
        S_EVALUATIONS
        if stage == "S"
        else COUNTERFACTUAL_B_EVALUATIONS if counterfactual else B_EVALUATIONS
    )
    checkpoint_updates = (
        S_CHECKPOINTS
        if stage == "S"
        else COUNTERFACTUAL_B_CHECKPOINTS if counterfactual else B_CHECKPOINTS
    )
    train_reader = eligible_reader if stage == "S" else full_reader
    trainable_modules = _set_stage_boundary(model, stage)
    if stage == "B":
        completed_s, _ = _load_checkpoint(
            Path(s_checkpoint), model=model, optimizer=None, scheduler=None,
            cursor=None, stage="S", cell=cell, state_kind=state_kind,
            protocol=S_PROTOCOL, restore_training=False,
        )
        if completed_s != S_PROTOCOL.total_updates:
            raise PF10V2StagedTrainingError("bridge requires completed S")
        trainable_modules = _set_stage_boundary(model, stage)
    model.to(device)
    optimizer = build_pf1_optimizer(model, protocol)
    scheduler = PF1LearningRateSchedule(optimizer, protocol)
    cursor = _TrainCursor(train_reader, protocol.micro_batch_size)
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    completed = 0
    evaluations: list[dict[str, object]] = []
    norms: list[float] = []
    clipped = members_seen = supervised = short_batches = 0
    checkpoints: list[str] = []
    pair_tokens = 0
    pair_aligned_nll_sum = 0.0
    pair_shuffled_nll_sum = 0.0
    pair_hinge_weighted_sum = 0.0
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        completed, progress = _load_checkpoint(
            Path(resume_checkpoint), model=model, optimizer=optimizer,
            scheduler=scheduler, cursor=cursor, stage=stage, cell=cell,
            state_kind=state_kind, protocol=protocol, restore_training=True,
        )
        evaluations = list(progress.get("evaluations", ()))
        norms = [float(v) for v in progress.get("preclip_norms", ())]
        clipped = int(progress.get("clipped_updates", 0))
        members_seen = int(progress.get("members_seen", 0))
        supervised = int(progress.get("supervised_targets", 0))
        short_batches = int(progress.get("short_microbatches", 0))
        checkpoints = [str(v) for v in progress.get("checkpoints", ())]
        elapsed_before = float(progress.get("wall_seconds", 0.0))

    def evaluate(update: int) -> dict[str, object]:
        # Intermediate evaluations track optimization.  The expensive causal
        # counterfactuals are needed only before training and at the frozen
        # final decision point; replaying them at every milestone cannot affect
        # training and adds several complete dev passes.
        include_causal = update in {0, protocol.total_updates}
        aligned_state = _evaluate_state(
            model, reader=eligible_reader, tokenizer=tokenizer,
            address_provider=address_provider, state_provider=state_provider,
            device=device, use_bf16=use_bf16, memory_mode="aligned",
        )
        result: dict[str, object] = {
            "state_aligned": aligned_state,
            "causal_diagnostics_included": include_causal,
        }
        if include_causal:
            zero_state = _evaluate_state(
                model, reader=eligible_reader, tokenizer=tokenizer,
                address_provider=address_provider, state_provider=state_provider,
                device=device, use_bf16=use_bf16, memory_mode="zero",
            )
            result.update({
                "state_zero": zero_state,
                "state_zero_minus_aligned": float(zero_state["weighted_state_loss"])
                - float(aligned_state["weighted_state_loss"]),
            })
        if stage == "B":
            aligned = _evaluate_identity(
                model, reader=full_reader, tokenizer=tokenizer,
                address_provider=address_provider, state_provider=state_provider,
                device=device, use_bf16=use_bf16, memory_mode="aligned",
            )
            result["identity_aligned"] = aligned
            if include_causal:
                zero = _evaluate_identity(
                    model, reader=full_reader, tokenizer=tokenizer,
                    address_provider=address_provider, state_provider=state_provider,
                    device=device, use_bf16=use_bf16, memory_mode="zero",
                )
                result.update({
                    "identity_zero": zero,
                    "identity_zero_minus_aligned": float(zero["token_weighted_nll"])
                    - float(aligned["token_weighted_nll"]),
                })
            if include_causal and cell == "F3D" and shuffle_provider is not None:
                shuffled = _evaluate_identity(
                    model, reader=full_reader, tokenizer=tokenizer,
                    address_provider=address_provider, state_provider=shuffle_provider,
                    device=device, use_bf16=use_bf16, memory_mode="aligned",
                )
                result["identity_matched_shuffle"] = shuffled
                result["identity_shuffle_minus_aligned"] = float(
                    shuffled["token_weighted_nll"]
                ) - float(aligned["token_weighted_nll"])
        return result

    if not evaluations:
        evaluations.append({"update": 0, **evaluate(0)})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for update in range(completed + 1, protocol.total_updates + 1):
        _set_train_modes(model, stage)
        batches = []
        shuffle_batches = []
        pair_masks = []
        batch_counts = []
        update_members = 0
        for _ in range(protocol.gradient_accumulation_steps):
            epoch, rows = cursor.next()
            short_batches += int(len(rows) < protocol.micro_batch_size)
            update_members += len(rows)
            batch = _collate(
                rows, objective="state" if stage == "S" else "cross_view",
                tokenizer=tokenizer, address_provider=address_provider,
                state_provider=state_provider, seed=TRAIN_SEED, epoch=epoch,
                device=device,
            )
            batches.append(batch)
            batch_counts.append(_state_counts(batch) if stage == "S" else _identity_count(batch))
            if counterfactual:
                shuffled_batch = _collate(
                    rows, objective="cross_view",
                    tokenizer=tokenizer, address_provider=address_provider,
                    state_provider=train_shuffle_provider, seed=TRAIN_SEED, epoch=epoch,
                    device=device,
                )
                shuffle_batches.append(shuffled_batch)
                pair_masks.append(
                    _counterfactual_token_mask(batch, train_shuffle_provider)
                )
        optimizer.zero_grad(set_to_none=True)
        if stage == "S":
            totals = {level: sum(row[level] for row in batch_counts) for level in (1, 2)}
            if any(value <= 0 for value in totals.values()):
                raise PF10V2StagedTrainingError("S update lacks L1/L2 targets")
            supervised_now = sum(totals.values())
            for batch, counts in zip(batches, batch_counts):
                with _autocast(device, use_bf16):
                    output = model(**batch.model_inputs())
                    loss = sum(
                        output.state_level_losses[level]
                        * (1.0 if level == 1 else model.state_level2_weight)
                        * (counts[level] / totals[level])
                        for level in (1, 2)
                    )
                loss.backward()
        else:
            total = sum(batch_counts)
            if total <= 0:
                raise PF10V2StagedTrainingError("bridge update lacks targets")
            supervised_now = total
            if not counterfactual:
                for batch, count in zip(batches, batch_counts):
                    with _autocast(device, use_bf16):
                        loss = model(**batch.model_inputs()).loss * (count / total)
                    loss.backward()
            else:
                update_pair_total = sum(int(mask.sum()) for mask in pair_masks)
                if update_pair_total <= 0:
                    raise PF10V2StagedTrainingError(
                        "counterfactual update lacks changed masked motif tokens"
                    )
                for batch, shuffled_batch, mask in zip(
                    batches, shuffle_batches, pair_masks
                ):
                    labels = batch.labels
                    if labels is None:
                        raise PF10V2StagedTrainingError(
                            "counterfactual batch lacks labels"
                        )
                    with _autocast(device, use_bf16):
                        output = model(
                            **_concatenate_counterfactual_inputs(
                                batch, shuffled_batch
                            )
                        )
                        t5_output = output.t5_output
                        logits = getattr(t5_output, "logits", None)
                        if not isinstance(logits, Tensor):
                            raise PF10V2StagedTrainingError(
                                "counterfactual bridge lacks decoder logits"
                            )
                        batch_size = labels.shape[0]
                        aligned_nll = _token_nll(logits[:batch_size], labels)
                        shuffled_nll = _token_nll(logits[batch_size:], labels)
                        active = labels != -100
                        local_pair_count = int(mask.sum())
                        aligned_pair = aligned_nll[mask].mean()
                        shuffled_pair = shuffled_nll[mask].mean()
                        pair_delta = shuffled_pair - aligned_pair
                        hinge = torch.relu(
                            float(counterfactual_margin) - pair_delta
                        )
                        loss = (
                            aligned_nll[active].sum() / total
                            + float(counterfactual_weight)
                            * hinge
                            * (local_pair_count / update_pair_total)
                        )
                    loss.backward()
                    pair_tokens += local_pair_count
                    pair_aligned_nll_sum += float(
                        aligned_nll[mask].detach().sum().float()
                    )
                    pair_shuffled_nll_sum += float(
                        shuffled_nll[mask].detach().sum().float()
                    )
                    pair_hinge_weighted_sum += float(
                        hinge.detach().float()
                    ) * local_pair_count
        preclip = clip_pf1_gradients(model, protocol)
        if not math.isfinite(preclip):
            raise PF10V2StagedTrainingError("non-finite gradient")
        norms.append(preclip)
        clipped += int(preclip > protocol.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        members_seen += update_members
        supervised += supervised_now
        if update in evaluation_updates:
            evaluations.append({"update": update, **evaluate(update)})
        if update in checkpoint_updates:
            wall = elapsed_before + time.perf_counter() - started
            path = output_dir / f"step-{update:05d}"
            progress = {
                "evaluations": evaluations, "preclip_norms": norms,
                "clipped_updates": clipped, "members_seen": members_seen,
                "supervised_targets": supervised,
                "short_microbatches": short_batches,
                "checkpoints": checkpoints + [str(path)], "wall_seconds": wall,
            }
            _write_checkpoint(path, _checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler,
                cursor=cursor, stage=stage, cell=cell, state_kind=state_kind,
                update=update, protocol=protocol, progress=progress,
                trainable_modules=trainable_modules,
            ))
            checkpoints.append(str(path))

    elapsed = elapsed_before + time.perf_counter() - started
    report = {
        "schema_version": SCHEMA_VERSION, "status": "pass",
        "factorisation_id": FACTORISATION_ID, "adapter_id": ADAPTER_ID,
        "address_contract": ADDRESS_CONTRACT, "stage": stage, "cell": cell,
        "state_kind": state_kind, "protocol": asdict(protocol),
        "trainable_modules": list(trainable_modules),
        "frozen_modules": ["t5"] if stage == "S" else ["adapter", "t5.encoder", "t5.shared"],
        "objective": (
            "state" if stage == "S" else
            "cross_view_identity_with_matched_ranking" if counterfactual else
            "cross_view_identity"
        ),
        "optimizer_updates": protocol.total_updates, "members_seen": members_seen,
        "supervised_targets": supervised, "short_microbatches": short_batches,
        "mean_preclip_gradient_norm": statistics.fmean(norms),
        "max_preclip_gradient_norm": max(norms), "clipped_updates": clipped,
        "clip_rate": clipped / protocol.total_updates,
        "evaluations": evaluations, "checkpoints": checkpoints,
        "causal_diagnostic_updates": [0, protocol.total_updates],
        "wall_seconds": elapsed, "members_per_second": members_seen / elapsed,
        "precision": "bf16_autocast" if use_bf16 else "test_precision",
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "scientific_boundary": "PF-10 failure screen; not a full-pretraining or downstream result",
    }
    report["counterfactual_bridge"] = {
        "enabled": counterfactual,
        "weight": float(counterfactual_weight) if counterfactual else 0.0,
        "margin_nll": float(counterfactual_margin) if counterfactual else None,
        "pair_token_scope": (
            "masked identity tokens owned by actually replaced logical motifs"
            if counterfactual else None
        ),
        "pair_tokens": pair_tokens,
        "train_shuffle_minus_aligned_nll": (
            (pair_shuffled_nll_sum - pair_aligned_nll_sum) / pair_tokens
            if pair_tokens else None
        ),
        "mean_weighted_hinge": (
            pair_hinge_weighted_sum / pair_tokens if pair_tokens else None
        ),
        "unique_members_per_update": protocol.effective_batch_size,
        "paired_forward_rows_per_update": (
            protocol.effective_batch_size * 2 if counterfactual
            else protocol.effective_batch_size
        ),
    }
    (output_dir / f"{stage.lower()}_training_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    paired = Path(args.paired_release).expanduser().resolve()
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired)
    cache = reader.warm_decoded_record_cache(
        workers=args.cache_workers, max_pending=args.cache_max_pending
    )
    support = Path(args.support_census)
    eligible = _EligibleReader(
        reader,
        train_indices=_load_eligible_indices(
            support / "train_state_eligible_membership.jsonl", expected_split="train"
        ),
        dev_indices=_load_eligible_indices(
            support / "dev_state_eligible_membership.jsonl", expected_split="dev"
        ),
    )
    addresses = GraphPortsCanonicalAtomAddressProvider(paired / DONOR_ATOM_MAP_NAME)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PF10V2StagedTrainingError("one CUDA BF16 device is required")
    provider = None
    shuffle = None
    train_shuffle = None
    try:
        if args.cell == "B2D":
            if args.morgan_overlay is None:
                raise PF10V2StagedTrainingError("B2D requires --morgan-overlay")
            provider = MorganAtomStateProvider(Path(args.morgan_overlay))
        elif args.morgan_overlay is not None:
            raise PF10V2StagedTrainingError("F3D rejects --morgan-overlay")
        if args.stage in {"B", "both"} and args.cell == "F3D":
            if args.shuffle_overlay is None:
                raise PF10V2StagedTrainingError("F3D bridge requires --shuffle-overlay")
            from .build_pf10_matched_motif_overlay_v1 import MatchedMotifStateProvider
            shuffle = MatchedMotifStateProvider(Path(args.shuffle_overlay))
            if args.counterfactual_weight > 0.0:
                if args.train_shuffle_overlay is None:
                    raise PF10V2StagedTrainingError(
                        "counterfactual bridge requires a train matched overlay"
                    )
                train_shuffle = MatchedMotifStateProvider(
                    Path(args.train_shuffle_overlay)
                )
        model = load_deterministic_factorized_model_v2(
            base_model_snapshot=Path(args.base_model_snapshot),
            base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
            union_tokenizer_dir=paired / "union_tokenizer",
            union_init_dir=Path(args.union_init_dir),
            union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
            adapter_seed=ADAPTER_SEED,
            num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        )
        output_root = Path(args.output_dir)
        if args.stage == "both":
            s_report = run_stage(
                stage="S", cell=args.cell, full_reader=reader,
                eligible_reader=eligible, tokenizer=tokenizer.runtime, model=model,
                address_provider=addresses, output_dir=output_root / "S",
                state_provider=provider, device=torch.device("cuda:0"), use_bf16=True,
            )
            s_checkpoint = output_root / "S" / f"step-{S_PROTOCOL.total_updates:05d}"
            b_report = run_stage(
                stage="B", cell=args.cell, full_reader=reader,
                eligible_reader=eligible, tokenizer=tokenizer.runtime, model=model,
                address_provider=addresses, output_dir=output_root / "B",
                state_provider=provider, device=torch.device("cuda:0"), use_bf16=True,
                s_checkpoint=s_checkpoint, shuffle_provider=shuffle,
                train_shuffle_provider=train_shuffle,
                counterfactual_weight=args.counterfactual_weight,
                counterfactual_margin=args.counterfactual_margin,
            )
            report: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "status": "pass",
                "cell": args.cell,
                "stage_order": ["S", "B"],
                "stages": {"S": s_report, "B": b_report},
            }
        else:
            report = run_stage(
                stage=args.stage, cell=args.cell, full_reader=reader,
                eligible_reader=eligible, tokenizer=tokenizer.runtime, model=model,
                address_provider=addresses, output_dir=output_root,
                state_provider=provider, device=torch.device("cuda:0"), use_bf16=True,
                s_checkpoint=Path(args.s_checkpoint) if args.s_checkpoint else None,
                shuffle_provider=shuffle,
                train_shuffle_provider=train_shuffle,
                resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
                counterfactual_weight=args.counterfactual_weight,
                counterfactual_margin=args.counterfactual_margin,
            )
        report["artifact_binding"] = {
            "paired_release": str(paired), "support_census": str(support.resolve()),
            "canonical_address_records": addresses.record_count,
            "decoded_cache_warmup": cache,
            "decoded_cache_final": reader.decoded_record_cache_stats(),
        }
        manifest_name = (
            "staged_training_manifest.json"
            if args.stage == "both"
            else f"{args.stage.lower()}_training_manifest.json"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / manifest_name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if provider is not None:
            provider.close()
        if shuffle is not None:
            shuffle.close()
        if train_shuffle is not None:
            train_shuffle.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("S", "B", "both"), required=True)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--support-census", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path)
    parser.add_argument("--shuffle-overlay", type=Path)
    parser.add_argument("--train-shuffle-overlay", type=Path)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--s-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=12)
    parser.add_argument("--cache-max-pending", type=int, default=48)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = _parser().parse_args(argv)
    if args.cache_workers <= 0 or args.cache_max_pending < args.cache_workers:
        raise SystemExit("invalid cache worker bounds")
    if args.stage == "B" and args.s_checkpoint is None:
        raise SystemExit("bridge requires --s-checkpoint")
    if args.stage in {"S", "both"} and args.s_checkpoint is not None:
        raise SystemExit("S/both reject --s-checkpoint")
    if args.stage == "both" and args.resume_checkpoint is not None:
        raise SystemExit("both-stage launch rejects --resume-checkpoint")
    if args.counterfactual_weight > 0.0 and not (
        args.stage == "B" and args.cell == "F3D"
        and args.shuffle_overlay is not None
        and args.train_shuffle_overlay is not None
    ):
        raise SystemExit(
            "counterfactual bridge requires F3D B with matched shuffle overlay"
        )
    report = run_cli(args)
    print(json.dumps({"status": report["status"], "cell": args.cell, "stage": args.stage}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADDRESS_CONTRACT", "B_PROTOCOL", "CHECKPOINT_SCHEMA",
    "COUNTERFACTUAL_B_PROTOCOL",
    "PF10V2StagedTrainingError", "SCHEMA_VERSION", "S_PROTOCOL",
    "run_cli", "run_stage",
]
