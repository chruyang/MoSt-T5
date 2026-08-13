"""Three-update PF-10 factorized GPU smoke contract.

This is an independent custom-loop runner.  It does not modify or call the
historical PF-1 four-grid runner and importing it never starts training.

One invocation owns exactly one ``(cell, stage)``:

* S-stage admits B2D/F3D only, freezes the complete T5, and trains the shared
  adapter with the formal ``motif_atom_row`` categorical-state objective.
* G-stage trains GraphPorts grammar.  B0 calls raw T5; B2D/F3D call the
  factorized wrapper and must explicitly load their own completed S checkpoint.

The fixed smoke budget is 128 records, two micro-batches of 64 per optimizer
update, and three updates.  It is a memory/throughput and wiring probe, not an
effect-size experiment.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn

from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)

from .factorized_motif_t5_v1 import FactorizedMotifT5V1
from .factorized_view_collator_v1 import (
    AtomStateProvider,
    FactorizedMotifViewBatch,
    collate_factorized_motif_view,
)


SMOKE_SCHEMA = "most-t5-p2/pf10-factorized-smoke/v1"
SMOKE_CHECKPOINT_SCHEMA = "most-t5-p2/pf10-factorized-smoke-checkpoint/v1"
SMOKE_RECORD_COUNT = 128
SMOKE_MICRO_BATCH_SIZE = 64
SMOKE_GRADIENT_ACCUMULATION = 2
SMOKE_UPDATES = 3
SMOKE_LEARNING_RATE = 1.0e-4
SMOKE_MAX_GRAD_NORM = 1.0
SMOKE_SEED = 20260808
STATE_MASK_PROBABILITY = 0.15
IDENTITY_MASK_PROBABILITY = 0.15
FORMAL_STATE_MASKING = "motif_atom_row"
CELLS = ("B0", "B2D", "F3D")
STAGES = ("S", "G")


class PF10FactorizedSmokeError(RuntimeError):
    """The fixed smoke-stage contract could not be executed."""


class RawT5Output(Protocol):
    loss: Tensor


@dataclass(frozen=True)
class SmokeStageSpec:
    cell: str
    stage: str
    objective_mode: str
    model_path: str
    t5_trainable: bool
    adapter_trainable: bool
    requires_s_checkpoint: bool


def get_smoke_stage_spec(cell: str, stage: str) -> SmokeStageSpec:
    """Return the only admitted model/objective pairing for one invocation."""

    if cell not in CELLS or stage not in STAGES:
        raise PF10FactorizedSmokeError("cell/stage must belong to the frozen smoke grid")
    if stage == "S":
        if cell == "B0":
            raise PF10FactorizedSmokeError("S-stage admits B2D and F3D only")
        return SmokeStageSpec(
            cell=cell,
            stage=stage,
            objective_mode="state",
            model_path="factorized_t5",
            t5_trainable=False,
            adapter_trainable=True,
            requires_s_checkpoint=False,
        )
    if cell == "B0":
        return SmokeStageSpec(
            cell=cell,
            stage=stage,
            objective_mode="grammar",
            model_path="raw_t5",
            t5_trainable=True,
            adapter_trainable=False,
            requires_s_checkpoint=False,
        )
    return SmokeStageSpec(
        cell=cell,
        stage=stage,
        objective_mode="grammar",
        model_path="factorized_t5",
        t5_trainable=True,
        adapter_trainable=True,
        requires_s_checkpoint=True,
    )


def assert_same_factorized_initialization(
    b2d_model: FactorizedMotifT5V1,
    f3d_model: FactorizedMotifT5V1,
) -> None:
    """Require exact pre-S parameter equality for the two state-source cells."""

    if not isinstance(b2d_model, FactorizedMotifT5V1) or not isinstance(
        f3d_model, FactorizedMotifT5V1
    ):
        raise PF10FactorizedSmokeError("B2D/F3D initialization check needs factorized models")
    left = b2d_model.state_dict()
    right = f3d_model.state_dict()
    if tuple(left) != tuple(right):
        raise PF10FactorizedSmokeError("B2D/F3D state-dict schemas differ at initialization")
    for name in left:
        if left[name].shape != right[name].shape or not torch.equal(left[name], right[name]):
            raise PF10FactorizedSmokeError(
                f"B2D/F3D parameter {name} differs at initialization"
            )


def _validate_records(
    records: Sequence[ProductionMotifRecord],
) -> tuple[ProductionMotifRecord, ...]:
    rows = tuple(records)
    if len(rows) != SMOKE_RECORD_COUNT:
        raise PF10FactorizedSmokeError(
            f"smoke runner requires exactly {SMOKE_RECORD_COUNT} records"
        )
    if any(not isinstance(row, ProductionMotifRecord) for row in rows):
        raise PF10FactorizedSmokeError("smoke records must be validated motif records")
    record_ids = tuple(row.record_id for row in rows)
    if len(set(record_ids)) != len(record_ids):
        raise PF10FactorizedSmokeError("smoke records must have unique record IDs")
    return rows


def _validate_cell_provider(
    cell: str,
    provider: AtomStateProvider | None,
) -> str:
    if cell == "B2D":
        if provider is None:
            raise PF10FactorizedSmokeError("B2D requires an aligned 2D atom-state provider")
        state_kind = str(provider.state_kind)
        if not state_kind or state_kind == "e3fp":
            raise PF10FactorizedSmokeError("B2D provider must declare a non-E3FP state kind")
        return state_kind
    if provider is not None:
        raise PF10FactorizedSmokeError("B0/F3D cannot receive an alternate state provider")
    return "none" if cell == "B0" else "e3fp"


def _apply_trainability(model: nn.Module, spec: SmokeStageSpec) -> tuple[str, ...]:
    """Apply the stage boundary and return the exact trainable parameter names."""

    if spec.model_path == "factorized_t5":
        if not isinstance(model, FactorizedMotifT5V1):
            raise PF10FactorizedSmokeError("B2D/F3D require FactorizedMotifT5V1")
        model.requires_grad_(False)
        model.t5.requires_grad_(spec.t5_trainable)
        model.adapter.requires_grad_(spec.adapter_trainable)
    else:
        if isinstance(model, FactorizedMotifT5V1):
            raise PF10FactorizedSmokeError("B0 G-stage requires the raw T5 module")
        model.requires_grad_(True)
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not names:
        raise PF10FactorizedSmokeError("smoke stage exposes no trainable parameters")
    if spec.stage == "S" and any(name.startswith("t5.") for name in names):
        raise PF10FactorizedSmokeError("S-stage must freeze every T5 parameter")
    return names


def _checkpoint_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f"{stage.lower()}_stage_checkpoint.pt"


def _write_checkpoint(
    *,
    model: nn.Module,
    spec: SmokeStageSpec,
    output_dir: Path,
    state_kind: str,
) -> Path:
    path = _checkpoint_path(output_dir, spec.stage)
    torch.save(
        {
            "schema_version": SMOKE_CHECKPOINT_SCHEMA,
            "cell": spec.cell,
            "stage": spec.stage,
            "completed_updates": SMOKE_UPDATES,
            "state_kind": state_kind,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _load_s_checkpoint(
    *,
    model: FactorizedMotifT5V1,
    cell: str,
    state_kind: str,
    checkpoint_path: Path,
) -> None:
    path = Path(checkpoint_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise PF10FactorizedSmokeError("G-stage S checkpoint is absent or empty")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch before the weights-only argument still loads this
        # runner-owned, schema-checked checkpoint.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not (
        payload.get("schema_version") == SMOKE_CHECKPOINT_SCHEMA
        and payload.get("cell") == cell
        and payload.get("stage") == "S"
        and payload.get("completed_updates") == SMOKE_UPDATES
        and payload.get("state_kind") == state_kind
    ):
        raise PF10FactorizedSmokeError("G-stage checkpoint differs from its own S cell")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise PF10FactorizedSmokeError("S checkpoint lacks a model state")
    model.load_state_dict(dict(state), strict=True)


def _autocast(device: torch.device, use_bf16: bool) -> Any:
    if not use_bf16:
        return nullcontext()
    if device.type != "cuda":
        raise PF10FactorizedSmokeError("BF16 smoke mode requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _collate_update(
    records: Sequence[ProductionMotifRecord],
    *,
    tokenizer: ProductionTokenizerRuntime,
    spec: SmokeStageSpec,
    provider: AtomStateProvider | None,
    epoch: int,
    device: torch.device,
) -> tuple[FactorizedMotifViewBatch, ...]:
    batches = []
    for start in range(0, SMOKE_RECORD_COUNT, SMOKE_MICRO_BATCH_SIZE):
        batch = collate_factorized_motif_view(
            records[start : start + SMOKE_MICRO_BATCH_SIZE],
            tokenizer=tokenizer,
            objective_mode=spec.objective_mode,
            seed=SMOKE_SEED,
            epoch=epoch,
            identity_mask_probability=IDENTITY_MASK_PROBABILITY,
            state_mask_probability=STATE_MASK_PROBABILITY,
            state_masking_strategy=FORMAL_STATE_MASKING,
            num_e3fp_embeddings=4096,
            atom_state_provider=provider,
            device=device,
        )
        batches.append(batch)
    if len(batches) != SMOKE_GRADIENT_ACCUMULATION:
        raise PF10FactorizedSmokeError("smoke update must contain exactly two micro-batches")
    return tuple(batches)


def _target_tokens(batch: FactorizedMotifViewBatch) -> int:
    if batch.labels is None:
        raise PF10FactorizedSmokeError("grammar batch lacks labels")
    return int((batch.labels != -100).sum().item())


def _state_counts(batch: FactorizedMotifViewBatch) -> dict[int, int]:
    if batch.state_target_mask is None:
        raise PF10FactorizedSmokeError("state batch lacks target mask")
    return {
        1: int(batch.state_target_mask[..., 1].sum().item()),
        2: int(batch.state_target_mask[..., 2].sum().item()),
    }


def _raw_t5_loss(model: nn.Module, batch: FactorizedMotifViewBatch) -> Tensor:
    if batch.labels is None:
        raise PF10FactorizedSmokeError("raw grammar path requires labels")
    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        return_dict=True,
    )
    loss = getattr(output, "loss", None)
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise PF10FactorizedSmokeError("raw T5 must return one scalar grammar loss")
    return loss


def run_pf10_factorized_smoke_stage(
    *,
    cell: str,
    stage: str,
    records: Sequence[ProductionMotifRecord],
    tokenizer: ProductionTokenizerRuntime,
    model: nn.Module,
    output_dir: Path,
    atom_state_provider: AtomStateProvider | None = None,
    s_checkpoint: Path | None = None,
    device: torch.device | str = "cpu",
    use_bf16: bool = False,
) -> dict[str, object]:
    """Run one fixed three-update smoke stage and write its checkpoint/report."""

    spec = get_smoke_stage_spec(cell, stage)
    rows = _validate_records(records)
    if not isinstance(tokenizer, ProductionTokenizerRuntime):
        raise PF10FactorizedSmokeError("tokenizer must be a production runtime")
    state_kind = _validate_cell_provider(cell, atom_state_provider)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)

    if spec.requires_s_checkpoint:
        if s_checkpoint is None or not isinstance(model, FactorizedMotifT5V1):
            raise PF10FactorizedSmokeError("B2D/F3D G-stage requires its S checkpoint")
        _load_s_checkpoint(
            model=model,
            cell=cell,
            state_kind=state_kind,
            checkpoint_path=Path(s_checkpoint),
        )
    elif s_checkpoint is not None:
        raise PF10FactorizedSmokeError("this smoke stage does not accept an S checkpoint")

    trainable_names = _apply_trainability(model, spec)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=SMOKE_LEARNING_RATE,
    )
    torch.manual_seed(SMOKE_SEED)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(SMOKE_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)

    started = time.perf_counter()
    update_reports: list[dict[str, object]] = []
    model.train()
    if spec.stage == "S":
        # Frozen means a stable pretrained language backbone in this stage,
        # including disabled T5 dropout.  The adapter remains in train mode.
        assert isinstance(model, FactorizedMotifT5V1)
        model.t5.eval()
        model.adapter.train()
    for update_index in range(SMOKE_UPDATES):
        batches = _collate_update(
            rows,
            tokenizer=tokenizer,
            spec=spec,
            provider=atom_state_provider,
            epoch=update_index,
            device=resolved_device,
        )
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        update_target_count = 0

        if spec.stage == "S":
            per_batch_counts = tuple(_state_counts(batch) for batch in batches)
            total_counts = {
                level: sum(counts[level] for counts in per_batch_counts)
                for level in (1, 2)
            }
            if any(count <= 0 for count in total_counts.values()):
                raise PF10FactorizedSmokeError("S-stage needs level-1 and level-2 targets")
            for batch, counts in zip(batches, per_batch_counts):
                with _autocast(resolved_device, use_bf16):
                    output = model(**batch.model_inputs())
                    contribution = None
                    for level in (1, 2):
                        level_loss = output.state_level_losses.get(level)
                        if not isinstance(level_loss, Tensor):
                            raise PF10FactorizedSmokeError(
                                "factorized S-stage omitted a selected level loss"
                            )
                        level_weight = 1.0 if level == 1 else model.state_level2_weight
                        term = level_loss * level_weight * (
                            counts[level] / total_counts[level]
                        )
                        contribution = term if contribution is None else contribution + term
                assert contribution is not None
                contribution.backward()
                update_loss += float(contribution.detach().float().item())
            update_target_count = sum(total_counts.values())
        else:
            token_counts = tuple(_target_tokens(batch) for batch in batches)
            total_tokens = sum(token_counts)
            if total_tokens <= 0:
                raise PF10FactorizedSmokeError("G-stage needs decoder target tokens")
            for batch, token_count in zip(batches, token_counts):
                with _autocast(resolved_device, use_bf16):
                    if spec.model_path == "raw_t5":
                        loss = _raw_t5_loss(model, batch)
                    else:
                        loss = model(**batch.model_inputs()).grammar_loss
                        if not isinstance(loss, Tensor):
                            raise PF10FactorizedSmokeError(
                                "factorized G-stage omitted grammar loss"
                            )
                    contribution = loss * (token_count / total_tokens)
                contribution.backward()
                update_loss += float(contribution.detach().float().item())
            update_target_count = total_tokens

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            SMOKE_MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
            raise PF10FactorizedSmokeError("smoke gradient norm is not finite")
        optimizer.step()
        update_reports.append(
            {
                "update": update_index + 1,
                "loss": update_loss,
                "target_count": update_target_count,
                "preclip_grad_norm": float(torch.as_tensor(grad_norm).item()),
            }
        )

    elapsed = time.perf_counter() - started
    checkpoint = _write_checkpoint(
        model=model,
        spec=spec,
        output_dir=output_root,
        state_kind=state_kind,
    )
    report: dict[str, object] = {
        "schema_version": SMOKE_SCHEMA,
        "status": "pass",
        "cell": cell,
        "stage": stage,
        "objective_mode": spec.objective_mode,
        "model_path": spec.model_path,
        "state_kind": state_kind,
        "formal_state_masking": FORMAL_STATE_MASKING if stage == "S" else None,
        "record_count": len(rows),
        "micro_batch_size": SMOKE_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": SMOKE_GRADIENT_ACCUMULATION,
        "optimizer_updates": SMOKE_UPDATES,
        "learning_rate": SMOKE_LEARNING_RATE,
        "t5_trainable": spec.t5_trainable,
        "adapter_trainable": spec.adapter_trainable,
        "trainable_parameter_names": list(trainable_names),
        "loaded_s_checkpoint": str(Path(s_checkpoint)) if s_checkpoint is not None else None,
        "written_checkpoint": str(checkpoint),
        "elapsed_seconds": elapsed,
        "members_seen": SMOKE_RECORD_COUNT * SMOKE_UPDATES,
        "updates": update_reports,
        "device": str(resolved_device),
        "precision": "bf16_autocast" if use_bf16 else "float32_or_test_precision",
    }
    if resolved_device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(resolved_device)
        )
        report["peak_cuda_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(resolved_device)
        )
    with (output_root / "smoke_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


__all__ = [
    "CELLS",
    "FORMAL_STATE_MASKING",
    "IDENTITY_MASK_PROBABILITY",
    "PF10FactorizedSmokeError",
    "SMOKE_GRADIENT_ACCUMULATION",
    "SMOKE_MICRO_BATCH_SIZE",
    "SMOKE_RECORD_COUNT",
    "SMOKE_SCHEMA",
    "SMOKE_UPDATES",
    "STAGES",
    "STATE_MASK_PROBABILITY",
    "SmokeStageSpec",
    "assert_same_factorized_initialization",
    "get_smoke_stage_spec",
    "run_pf10_factorized_smoke_stage",
]
