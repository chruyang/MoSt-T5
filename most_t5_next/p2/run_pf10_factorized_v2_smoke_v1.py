"""CPU/GPU mechanism smoke for carrier-only factorized T5 V2.

The existing B0 result is architecture-independent and is not rerun here.
Each V2 geometry cell executes an adapter-only state stage followed by a
decoder bridge stage.  The bridge masks motif identity while retaining aligned
L1/L2 state, freezes the adapter and T5 encoder, and trains only the T5 decoder
and language-model head.  This is the smallest schedule that both preserves the
carrier state route and lets identity reconstruction learn to consume it.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)

from .factorized_motif_t5_v2 import FACTORISATION_ID, FactorizedMotifT5V2
from .factorized_view_collator_v1 import AtomStateProvider
from .factorized_view_collator_v2 import (
    CanonicalAtomAddressProvider,
    collate_factorized_motif_view_v2,
)
from .motif_geometry_adapter_v2 import ADAPTER_ID


SMOKE_SCHEMA = "most-t5-p2/pf10-factorized-v2-mechanism-smoke/v1"
CHECKPOINT_SCHEMA = "most-t5-p2/pf10-factorized-v2-mechanism-checkpoint/v1"
SMOKE_RECORD_COUNT = 128
MICRO_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2
S_UPDATES = 3
B_UPDATES = 4
B_OBJECTIVE_SCHEDULE = ("cross_view",) * B_UPDATES
LEARNING_RATE = 1.0e-4
MAX_GRAD_NORM = 1.0
SEED = 20260810


class PF10FactorizedV2SmokeError(RuntimeError):
    """The fixed V2 mechanism smoke violated its execution contract."""


def _validate_cell_and_provider(
    cell: str,
    provider: AtomStateProvider | None,
) -> str:
    if cell not in {"B2D", "F3D"}:
        raise PF10FactorizedV2SmokeError("V2 smoke admits B2D and F3D only")
    if cell == "B2D":
        if provider is None or str(provider.state_kind) in {"", "e3fp"}:
            raise PF10FactorizedV2SmokeError(
                "B2D requires a non-E3FP atom-state provider"
            )
        return str(provider.state_kind)
    if provider is not None:
        raise PF10FactorizedV2SmokeError(
            "F3D consumes persisted E3FP and cannot receive another provider"
        )
    return "e3fp"


def _autocast(device: torch.device, use_bf16: bool):
    if not use_bf16:
        return nullcontext()
    if device.type != "cuda":
        raise PF10FactorizedV2SmokeError("BF16 smoke requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _checkpoint_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f"{stage.lower()}_stage_checkpoint.pt"


def _write_checkpoint(
    *,
    model: FactorizedMotifT5V2,
    output_dir: Path,
    cell: str,
    stage: str,
    state_kind: str,
    updates: int,
) -> Path:
    path = _checkpoint_path(output_dir, stage)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "factorisation_id": FACTORISATION_ID,
            "adapter_id": ADAPTER_ID,
            "cell": cell,
            "stage": stage,
            "state_kind": state_kind,
            "completed_updates": updates,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _load_s_checkpoint(
    *,
    model: FactorizedMotifT5V2,
    checkpoint: Path,
    cell: str,
    state_kind: str,
) -> None:
    try:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(Path(checkpoint), map_location="cpu")
    if not isinstance(payload, Mapping) or not all(
        (
            payload.get("schema_version") == CHECKPOINT_SCHEMA,
            payload.get("factorisation_id") == FACTORISATION_ID,
            payload.get("adapter_id") == ADAPTER_ID,
            payload.get("cell") == cell,
            payload.get("stage") == "S",
            payload.get("state_kind") == state_kind,
            payload.get("completed_updates") == S_UPDATES,
        )
    ):
        raise PF10FactorizedV2SmokeError(
            "bridge stage requires its own completed V2 S checkpoint"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise PF10FactorizedV2SmokeError("V2 S checkpoint lacks model state")
    model.load_state_dict(dict(state), strict=True)


def _collate_microbatches(
    records: tuple[ProductionMotifRecord, ...],
    *,
    tokenizer: ProductionTokenizerRuntime,
    objective: str,
    epoch: int,
    address_provider: CanonicalAtomAddressProvider,
    state_provider: AtomStateProvider | None,
    device: torch.device,
):
    batches = []
    for start in range(0, SMOKE_RECORD_COUNT, MICRO_BATCH_SIZE):
        batches.append(
            collate_factorized_motif_view_v2(
                records[start : start + MICRO_BATCH_SIZE],
                tokenizer=tokenizer,
                objective_mode=objective,
                seed=SEED,
                epoch=epoch,
                atom_address_provider=address_provider,
                atom_state_provider=state_provider,
                num_e3fp_embeddings=4096,
                device=device,
            )
        )
    if len(batches) != GRADIENT_ACCUMULATION_STEPS:
        raise PF10FactorizedV2SmokeError(
            "V2 smoke must produce two 64-record microbatches"
        )
    return tuple(batches)


def _backward_update(
    *,
    model: FactorizedMotifT5V2,
    batches: Sequence[Any],
    objective: str,
    device: torch.device,
    use_bf16: bool,
) -> tuple[float, int]:
    if objective in {"grammar", "cross_view"}:
        counts = [int((batch.labels != -100).sum()) for batch in batches]
        total = sum(counts)
        if total <= 0:
            raise PF10FactorizedV2SmokeError("grammar update has no target tokens")
        loss_value = 0.0
        for batch, count in zip(batches, counts):
            with _autocast(device, use_bf16):
                loss = model(**batch.model_inputs()).loss
            (loss * (count / total)).backward()
            loss_value += float(loss.detach()) * count / total
        return loss_value, total

    level_counts = {
        level: sum(int(batch.state_target_mask[..., level].sum()) for batch in batches)
        for level in (1, 2)
    }
    if any(value <= 0 for value in level_counts.values()):
        raise PF10FactorizedV2SmokeError("state update lacks L1 or L2 targets")
    loss_value = 0.0
    for batch in batches:
        with _autocast(device, use_bf16):
            output = model(**batch.model_inputs())
            terms = []
            for level in (1, 2):
                count = int(output.state_level_counts.get(level, 0))
                if count:
                    weight = 1.0 if level == 1 else model.state_level2_weight
                    terms.append(
                        output.state_level_losses[level]
                        * weight
                        * (count / level_counts[level])
                    )
            loss = torch.stack(terms).sum()
        loss.backward()
        loss_value += float(loss.detach())
    return loss_value, sum(level_counts.values())


def _state_evaluation_loss(
    *,
    model: FactorizedMotifT5V2,
    batches: Sequence[Any],
    state_memory_mode: str,
    device: torch.device,
    use_bf16: bool,
) -> tuple[float, dict[int, int]]:
    counts = {
        level: sum(int(batch.state_target_mask[..., level].sum()) for batch in batches)
        for level in (1, 2)
    }
    sums = {1: 0.0, 2: 0.0}
    with torch.no_grad():
        for batch in batches:
            inputs = batch.model_inputs()
            inputs["state_memory_mode"] = state_memory_mode
            with _autocast(device, use_bf16):
                output = model(**inputs)
            for level in (1, 2):
                count = int(output.state_level_counts.get(level, 0))
                if count:
                    sums[level] += float(output.state_level_losses[level]) * count
    loss = sums[1] / counts[1]
    loss += model.state_level2_weight * sums[2] / counts[2]
    return loss, counts


def _evaluate_aligned_zero(
    *,
    model: FactorizedMotifT5V2,
    records: tuple[ProductionMotifRecord, ...],
    tokenizer: ProductionTokenizerRuntime,
    address_provider: CanonicalAtomAddressProvider,
    state_provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    batches = _collate_microbatches(
        records,
        tokenizer=tokenizer,
        objective="state",
        epoch=991,
        address_provider=address_provider,
        state_provider=state_provider,
        device=device,
    )
    aligned, counts = _state_evaluation_loss(
        model=model,
        batches=batches,
        state_memory_mode="aligned",
        device=device,
        use_bf16=use_bf16,
    )
    zero, zero_counts = _state_evaluation_loss(
        model=model,
        batches=batches,
        state_memory_mode="zero",
        device=device,
        use_bf16=use_bf16,
    )
    if counts != zero_counts:
        raise PF10FactorizedV2SmokeError(
            "aligned and zero diagnostics changed the state target domain"
        )
    if was_training:
        model.train()
    return {
        "aligned_loss": aligned,
        "zero_loss": zero,
        "zero_minus_aligned": zero - aligned,
        "target_counts": {str(level): counts[level] for level in (1, 2)},
        "same_targets": True,
        "interpretation": "wiring diagnostic only; four updates do not establish effect size",
    }


def _evaluate_identity_aligned_zero(
    *,
    model: FactorizedMotifT5V2,
    records: tuple[ProductionMotifRecord, ...],
    tokenizer: ProductionTokenizerRuntime,
    address_provider: CanonicalAtomAddressProvider,
    state_provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, object]:
    model.eval()
    batches = _collate_microbatches(
        records,
        tokenizer=tokenizer,
        objective="cross_view",
        epoch=992,
        address_provider=address_provider,
        state_provider=state_provider,
        device=device,
    )
    counts = [int((batch.labels != -100).sum()) for batch in batches]
    total = sum(counts)
    if total <= 0:
        raise PF10FactorizedV2SmokeError("identity diagnostic has no targets")

    def evaluate(memory_mode: str) -> float:
        value = 0.0
        with torch.no_grad():
            for batch, count in zip(batches, counts):
                inputs = batch.model_inputs()
                inputs["state_memory_mode"] = memory_mode
                with _autocast(device, use_bf16):
                    loss = model(**inputs).loss
                value += float(loss.detach()) * count / total
        return value

    aligned = evaluate("aligned")
    zero = evaluate("zero")
    return {
        "aligned_loss": aligned,
        "zero_loss": zero,
        "zero_minus_aligned": zero - aligned,
        "target_count": total,
        "same_targets": True,
    }


def run_pf10_factorized_v2_smoke_stage(
    *,
    cell: str,
    stage: str,
    records: Sequence[ProductionMotifRecord],
    tokenizer: ProductionTokenizerRuntime,
    model: FactorizedMotifT5V2,
    atom_address_provider: CanonicalAtomAddressProvider,
    output_dir: Path,
    atom_state_provider: AtomStateProvider | None = None,
    s_checkpoint: Path | None = None,
    device: torch.device | str = "cpu",
    use_bf16: bool = False,
) -> dict[str, object]:
    """Run one V2 state-pretraining or frozen-encoder bridge smoke stage."""

    if stage not in {"S", "B"}:
        raise PF10FactorizedV2SmokeError("stage must be S or B")
    if not isinstance(model, FactorizedMotifT5V2):
        raise PF10FactorizedV2SmokeError("V2 smoke requires FactorizedMotifT5V2")
    rows = tuple(records)
    if len(rows) != SMOKE_RECORD_COUNT or len({row.record_id for row in rows}) != len(rows):
        raise PF10FactorizedV2SmokeError("V2 smoke requires 128 unique records")
    state_kind = _validate_cell_and_provider(cell, atom_state_provider)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)

    if stage == "B":
        if s_checkpoint is None:
            raise PF10FactorizedV2SmokeError("V2 bridge stage requires an S checkpoint")
        _load_s_checkpoint(
            model=model,
            checkpoint=Path(s_checkpoint),
            cell=cell,
            state_kind=state_kind,
        )
    elif s_checkpoint is not None:
        raise PF10FactorizedV2SmokeError("V2 S stage cannot load an S checkpoint")

    model.requires_grad_(False)
    if stage == "S":
        model.adapter.requires_grad_(True)
    else:
        decoder = getattr(model.t5, "decoder", None)
        lm_head = getattr(model.t5, "lm_head", None)
        if not isinstance(decoder, torch.nn.Module) or not isinstance(
            lm_head, torch.nn.Module
        ):
            raise PF10FactorizedV2SmokeError(
                "V2 bridge requires explicit T5 decoder and LM head modules"
            )
        decoder.requires_grad_(True)
        lm_head.requires_grad_(True)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    torch.manual_seed(SEED)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)

    objectives = ("state",) * S_UPDATES if stage == "S" else B_OBJECTIVE_SCHEDULE
    started = time.perf_counter()
    updates = []
    model.train()
    if stage == "S":
        model.t5.eval()
        model.adapter.train()
    else:
        model.eval()
        model.t5.decoder.train()
        model.t5.lm_head.train()
    for update_index, objective in enumerate(objectives):
        batches = _collate_microbatches(
            rows,
            tokenizer=tokenizer,
            objective=objective,
            epoch=update_index,
            address_provider=atom_address_provider,
            state_provider=atom_state_provider,
            device=resolved_device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, targets = _backward_update(
            model=model,
            batches=batches,
            objective=objective,
            device=resolved_device,
            use_bf16=use_bf16,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise PF10FactorizedV2SmokeError("V2 smoke gradient is not finite")
        optimizer.step()
        updates.append(
            {
                "update": update_index + 1,
                "objective": objective,
                "loss": loss,
                "target_count": targets,
                "preclip_grad_norm": float(torch.as_tensor(grad_norm)),
            }
        )

    checkpoint = _write_checkpoint(
        model=model,
        output_dir=output_root,
        cell=cell,
        stage=stage,
        state_kind=state_kind,
        updates=len(objectives),
    )
    causal_diagnostic = _evaluate_aligned_zero(
        model=model,
        records=rows,
        tokenizer=tokenizer,
        address_provider=atom_address_provider,
        state_provider=atom_state_provider,
        device=resolved_device,
        use_bf16=use_bf16,
    )
    identity_diagnostic = _evaluate_identity_aligned_zero(
        model=model,
        records=rows,
        tokenizer=tokenizer,
        address_provider=atom_address_provider,
        state_provider=atom_state_provider,
        device=resolved_device,
        use_bf16=use_bf16,
    )
    gates = model.adapter.geometry_gate_values().detach().float().cpu()
    report: dict[str, object] = {
        "schema_version": SMOKE_SCHEMA,
        "status": "pass",
        "factorisation_id": FACTORISATION_ID,
        "adapter_id": ADAPTER_ID,
        "cell": cell,
        "stage": stage,
        "state_kind": state_kind,
        "record_count": len(rows),
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "optimizer_updates": len(objectives),
        "objective_schedule": list(objectives),
        "loaded_s_checkpoint": str(s_checkpoint) if s_checkpoint else None,
        "written_checkpoint": str(checkpoint),
        "geometry_gate": {
            "minimum": float(gates.min()),
            "mean": float(gates.mean()),
            "maximum": float(gates.max()),
        },
        "state_causal_diagnostic": causal_diagnostic,
        "identity_causal_diagnostic": identity_diagnostic,
        "updates": updates,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(resolved_device),
        "precision": "bf16_autocast" if use_bf16 else "float32",
    }
    if resolved_device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(resolved_device)
        )
        report["peak_cuda_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(resolved_device)
        )
    (output_root / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "CHECKPOINT_SCHEMA",
    "B_OBJECTIVE_SCHEDULE",
    "PF10FactorizedV2SmokeError",
    "SMOKE_SCHEMA",
    "run_pf10_factorized_v2_smoke_stage",
]
