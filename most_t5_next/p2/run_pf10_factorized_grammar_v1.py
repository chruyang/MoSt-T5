"""Formal single-cell PF-10 GraphPorts grammar stage.

Each invocation owns exactly one B0, B2D, or F3D cell.  This makes the three
scientifically independent runs directly launchable on three GPUs while also
supporting sequential single-GPU execution.  B2D/F3D must load their own
completed formal S-stage checkpoint; B0 starts from the common union-init T5.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from most_t5_next.p1.build_pf1_paired_release_v1 import PF1PairedReleaseReader
from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_union_init_checkpoint,
)
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
from .run_pf10_factorized_state_v1 import (
    CHECKPOINT_SCHEMA as S_CHECKPOINT_SCHEMA,
    S_PROTOCOL,
    _TrainCursor,
    _autocast,
    _restore_rng,
    _rng_state,
)


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-grammar-training/v1"
CHECKPOINT_SCHEMA = "most-t5-p2/pf10-factorized-grammar-checkpoint/v1"
TRAIN_SEED = 20260809
DEV_SEED = 20260810
DEV_MASK_EPOCH = 0
IDENTITY_MASK_PROBABILITY = 0.15
EVALUATION_UPDATES = (0, 2500, 5000, 7500, 10000)
CHECKPOINT_UPDATES = (5000, 10000)
G_PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=1.0e-3,
    warmup_updates=1000,
    total_updates=10000,
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
EVAL_BATCH_SIZE = 64


def _cell_protocol(cell: str) -> PF1OptimizationProtocol:
    """Keep effective batch 128 within the measured 24 GiB GPU ceiling."""

    if cell not in {"B0", "B2D", "F3D"}:
        raise PF10GrammarTrainingError("unknown grammar cell")
    return replace(
        G_PROTOCOL,
        micro_batch_size=G_PROTOCOL.micro_batch_size // 2,
        gradient_accumulation_steps=G_PROTOCOL.gradient_accumulation_steps * 2,
    )


class PF10GrammarTrainingError(RuntimeError):
    """The formal grammar-stage contract could not be executed."""


def _motif_records(rows: Sequence[Any]) -> tuple[Any, ...]:
    records = tuple(getattr(row, "motif_record", None) for row in rows)
    if any(record is None for record in records):
        raise PF10GrammarTrainingError("paired reader row lacks a motif record")
    return records


def _collate_grammar(
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
        objective_mode="grammar",
        seed=seed,
        epoch=epoch,
        identity_mask_probability=IDENTITY_MASK_PROBABILITY,
        state_masking_strategy="motif_atom_row",
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        atom_state_provider=provider,
        device=device,
    )


def _target_tokens(batch: FactorizedMotifViewBatch) -> int:
    if batch.labels is None:
        raise PF10GrammarTrainingError("grammar batch lacks labels")
    return int((batch.labels != -100).sum().item())


def _forward_grammar(
    model: nn.Module,
    batch: FactorizedMotifViewBatch,
    *,
    raw_t5: bool,
    state_memory_mode: str = "aligned",
) -> tuple[Tensor, Tensor]:
    if batch.labels is None:
        raise PF10GrammarTrainingError("grammar batch lacks labels")
    if raw_t5:
        if state_memory_mode != "aligned":
            raise PF10GrammarTrainingError("B0 has no state-memory ablation")
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
            return_dict=True,
        )
        loss = getattr(output, "loss", None)
        logits = getattr(output, "logits", None)
    else:
        inputs = batch.model_inputs()
        inputs["state_memory_mode"] = state_memory_mode
        output = model(**inputs)
        loss = output.grammar_loss
        t5_output = output.t5_output
        logits = getattr(t5_output, "logits", None)
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise PF10GrammarTrainingError("grammar forward produced an invalid CE loss")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise PF10GrammarTrainingError("grammar forward omitted decoder logits")
    return loss, logits


def evaluate_grammar_stage(
    model: nn.Module,
    *,
    reader: PF1PairedReleaseReader,
    tokenizer: Any,
    provider: AtomStateProvider | None,
    raw_t5: bool,
    device: torch.device,
    use_bf16: bool,
    state_memory_mode: str = "aligned",
) -> dict[str, object]:
    model.eval()
    nll_sum = 0.0
    target_tokens = 0
    correct = 0
    members = 0
    with torch.no_grad():
        for rows in reader.iter_dev(batch_size=EVAL_BATCH_SIZE):
            batch = _collate_grammar(
                rows,
                tokenizer=tokenizer,
                provider=provider,
                seed=DEV_SEED,
                epoch=DEV_MASK_EPOCH,
                device=device,
            )
            count = _target_tokens(batch)
            with _autocast(device, use_bf16):
                loss, logits = _forward_grammar(
                    model,
                    batch,
                    raw_t5=raw_t5,
                    state_memory_mode=state_memory_mode,
                )
            assert batch.labels is not None
            mask = batch.labels != -100
            correct += int((logits.argmax(dim=-1)[mask] == batch.labels[mask]).sum().item())
            nll_sum += float(loss.float().item()) * count
            target_tokens += count
            members += len(rows)
    if members != reader.dev_member_count or target_tokens <= 0:
        raise PF10GrammarTrainingError("grammar evaluation did not exhaust dev")
    return {
        "members": members,
        "target_tokens": target_tokens,
        "token_weighted_nll": nll_sum / target_tokens,
        "masked_token_accuracy": correct / target_tokens,
        "mask_seed": DEV_SEED,
        "mask_epoch": DEV_MASK_EPOCH,
        "state_memory_mode": state_memory_mode,
        "state_kind": "e3fp" if provider is None else str(provider.state_kind),
    }


def _load_formal_s_checkpoint(
    model: FactorizedMotifT5V1,
    *,
    checkpoint_dir: Path,
    cell: str,
    state_kind: str,
) -> None:
    path = Path(checkpoint_dir) / "training_state.pt"
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not (
        payload.get("schema_version") == S_CHECKPOINT_SCHEMA
        and payload.get("cell") == cell
        and payload.get("stage") == "S"
        and payload.get("state_kind") == state_kind
        and payload.get("completed_updates") == S_PROTOCOL.total_updates
        and payload.get("protocol") == asdict(S_PROTOCOL)
    ):
        raise PF10GrammarTrainingError("G-stage S checkpoint differs from its cell")
    model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]


def _write_checkpoint(
    *,
    path: Path,
    cell: str,
    state_kind: str,
    update: int,
    model: nn.Module,
    optimizer: Any,
    scheduler: PF1LearningRateSchedule,
    cursor: _TrainCursor,
    progress: Mapping[str, object],
    protocol: PF1OptimizationProtocol,
) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    staging = path / "training_state.pt.tmp"
    destination = path / "training_state.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "cell": cell,
            "stage": "G",
            "state_kind": state_kind,
            "completed_updates": update,
            "protocol": asdict(protocol),
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
    model: nn.Module,
    optimizer: Any,
    scheduler: PF1LearningRateSchedule,
    cursor: _TrainCursor,
    protocol: PF1OptimizationProtocol,
) -> tuple[int, dict[str, object]]:
    payload = torch.load(Path(path) / "training_state.pt", map_location="cpu")
    if not isinstance(payload, Mapping) or not (
        payload.get("schema_version") == CHECKPOINT_SCHEMA
        and payload.get("cell") == cell
        and payload.get("stage") == "G"
        and payload.get("state_kind") == state_kind
        and payload.get("protocol") == asdict(protocol)
    ):
        raise PF10GrammarTrainingError("resume checkpoint differs from formal G")
    completed = int(payload["completed_updates"])
    model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
    optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
    scheduler.load_state_dict(payload["scheduler_state_dict"])  # type: ignore[arg-type]
    cursor.load_state_dict(payload["cursor_state_dict"])  # type: ignore[arg-type]
    _restore_rng(payload["rng_state"])  # type: ignore[arg-type]
    progress = payload.get("progress")
    if not isinstance(progress, Mapping):
        raise PF10GrammarTrainingError("resume checkpoint lacks progress")
    return completed, dict(progress)


def run_grammar_cell(
    *,
    cell: str,
    reader: PF1PairedReleaseReader,
    tokenizer: Any,
    model: nn.Module,
    output_dir: Path,
    provider: AtomStateProvider | None,
    device: torch.device,
    use_bf16: bool,
    s_checkpoint: Path | None = None,
    shuffle_provider: AtomStateProvider | None = None,
    resume_checkpoint: Path | None = None,
) -> dict[str, object]:
    if cell not in {"B0", "B2D", "F3D"}:
        raise PF10GrammarTrainingError("cell must be B0, B2D or F3D")
    raw_t5 = cell == "B0"
    protocol = _cell_protocol(cell)
    if (
        protocol.micro_batch_size * protocol.gradient_accumulation_steps
        != G_PROTOCOL.micro_batch_size * G_PROTOCOL.gradient_accumulation_steps
    ):
        raise PF10GrammarTrainingError("grammar cells must keep one effective batch")
    state_kind = "none" if raw_t5 else "e3fp" if provider is None else str(provider.state_kind)
    if cell == "B2D" and (provider is None or state_kind == "e3fp"):
        raise PF10GrammarTrainingError("B2D requires Morgan atom state")
    if cell != "B2D" and provider is not None:
        raise PF10GrammarTrainingError("only B2D accepts an alternate training provider")
    if raw_t5:
        if s_checkpoint is not None or shuffle_provider is not None:
            raise PF10GrammarTrainingError("B0 cannot receive state checkpoints/providers")
    else:
        if not isinstance(model, FactorizedMotifT5V1) or s_checkpoint is None:
            raise PF10GrammarTrainingError("B2D/F3D require their formal S checkpoint")
        _load_formal_s_checkpoint(
            model,
            checkpoint_dir=Path(s_checkpoint),
            cell=cell,
            state_kind=state_kind,
        )
    if cell == "F3D" and shuffle_provider is None:
        raise PF10GrammarTrainingError("F3D requires its matched-shuffle dev provider")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.requires_grad_(True)
    model.to(device)
    optimizer = build_pf1_optimizer(model, protocol)
    scheduler = PF1LearningRateSchedule(optimizer, protocol)
    cursor = _TrainCursor(reader, protocol.micro_batch_size)
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
    supervised_tokens = 0
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
            protocol=protocol,
        )
        evaluations = list(progress.get("evaluations", ()))
        preclip_norms = [float(value) for value in progress.get("preclip_norms", ())]
        clipped_updates = int(progress.get("clipped_updates", 0))
        members_seen = int(progress.get("members_seen", 0))
        supervised_tokens = int(progress.get("supervised_tokens", 0))
        short_microbatches = int(progress.get("short_microbatches", 0))
        checkpoints = [str(value) for value in progress.get("checkpoints", ())]
        elapsed_before = float(progress.get("wall_seconds", 0.0))
    else:
        evaluations.append(
            {"update": 0, **evaluate_grammar_stage(
                model,
                reader=reader,
                tokenizer=tokenizer,
                provider=provider,
                raw_t5=raw_t5,
                device=device,
                use_bf16=use_bf16,
            )}
        )

    started = time.perf_counter()
    for update in range(completed + 1, protocol.total_updates + 1):
        model.train()
        batches: list[FactorizedMotifViewBatch] = []
        counts: list[int] = []
        update_members = 0
        for _ in range(protocol.gradient_accumulation_steps):
            epoch, rows = cursor.next()
            short_microbatches += int(len(rows) < protocol.micro_batch_size)
            update_members += len(rows)
            batch = _collate_grammar(
                rows,
                tokenizer=tokenizer,
                provider=provider,
                seed=TRAIN_SEED,
                epoch=epoch,
                device=device,
            )
            batches.append(batch)
            counts.append(_target_tokens(batch))
        total = sum(counts)
        if total <= 0:
            raise PF10GrammarTrainingError("grammar update lacks target tokens")
        optimizer.zero_grad(set_to_none=True)
        for batch, count in zip(batches, counts):
            with _autocast(device, use_bf16):
                loss, _logits = _forward_grammar(model, batch, raw_t5=raw_t5)
                contribution = loss * (count / total)
            contribution.backward()
        preclip = clip_pf1_gradients(model, protocol)
        if not math.isfinite(preclip):
            raise PF10GrammarTrainingError("grammar gradient norm is non-finite")
        preclip_norms.append(preclip)
        clipped_updates += int(preclip > protocol.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        members_seen += update_members
        supervised_tokens += total

        if update in EVALUATION_UPDATES:
            evaluations.append(
                {"update": update, **evaluate_grammar_stage(
                    model,
                    reader=reader,
                    tokenizer=tokenizer,
                    provider=provider,
                    raw_t5=raw_t5,
                    device=device,
                    use_bf16=use_bf16,
                )}
            )
        if update in CHECKPOINT_UPDATES:
            wall = elapsed_before + (time.perf_counter() - started)
            checkpoint_path = output_dir / f"step-{update:05d}"
            progress = {
                "evaluations": evaluations,
                "preclip_norms": preclip_norms,
                "clipped_updates": clipped_updates,
                "members_seen": members_seen,
                "supervised_tokens": supervised_tokens,
                "short_microbatches": short_microbatches,
                "checkpoints": checkpoints + [str(checkpoint_path)],
                "wall_seconds": wall,
            }
            checkpoint = _write_checkpoint(
                path=checkpoint_path,
                cell=cell,
                state_kind=state_kind,
                update=update,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=cursor,
                progress=progress,
                protocol=protocol,
            )
            checkpoints.append(str(checkpoint))

    aligned_final = evaluations[-1]
    diagnostics: dict[str, object] | None = None
    if cell == "F3D":
        assert shuffle_provider is not None
        zero = evaluate_grammar_stage(
            model,
            reader=reader,
            tokenizer=tokenizer,
            provider=None,
            raw_t5=False,
            device=device,
            use_bf16=use_bf16,
            state_memory_mode="zero",
        )
        shuffled = evaluate_grammar_stage(
            model,
            reader=reader,
            tokenizer=tokenizer,
            provider=shuffle_provider,
            raw_t5=False,
            device=device,
            use_bf16=use_bf16,
        )
        aligned_nll = float(aligned_final["token_weighted_nll"])
        diagnostics = {
            "aligned": aligned_final,
            "zero": zero,
            "matched_shuffle": shuffled,
            "zero_minus_aligned_delta_nll": float(zero["token_weighted_nll"]) - aligned_nll,
            "shuffle_minus_aligned_delta_nll": float(shuffled["token_weighted_nll"]) - aligned_nll,
        }

    elapsed = elapsed_before + (time.perf_counter() - started)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cell": cell,
        "stage": "G",
        "state_kind": state_kind,
        "protocol": asdict(protocol),
        "effective_members_per_update": (
            protocol.micro_batch_size * protocol.gradient_accumulation_steps
        ),
        "optimizer_updates": protocol.total_updates,
        "members_seen": members_seen,
        "supervised_target_tokens": supervised_tokens,
        "short_microbatches": short_microbatches,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clipped_updates": clipped_updates,
        "clip_rate": clipped_updates / protocol.total_updates,
        "evaluations": evaluations,
        "f3d_state_diagnostics": diagnostics,
        "checkpoints": checkpoints,
        "wall_seconds": elapsed,
        "members_per_second": members_seen / elapsed,
        "precision": "bf16_autocast" if use_bf16 else "test_precision",
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }
    (output_dir / "grammar_training_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    paired_release = Path(args.paired_release).expanduser().resolve()
    verified_tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired_release / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired_release)
    cache = reader.warm_decoded_record_cache(
        workers=args.cache_workers,
        max_pending=args.cache_max_pending,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PF10GrammarTrainingError("one CUDA BF16 device is required")
    provider = None
    shuffle_provider = None
    try:
        if args.cell == "B0":
            verified = load_verified_union_init_checkpoint(
                base_model_snapshot=Path(args.base_model_snapshot),
                base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
                union_tokenizer_dir=paired_release / "union_tokenizer",
                output_dir=Path(args.union_init_dir),
                geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
                num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            )
            model = verified.model
        else:
            model = load_deterministic_factorized_model(
                base_model_snapshot=Path(args.base_model_snapshot),
                base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
                union_tokenizer_dir=paired_release / "union_tokenizer",
                union_init_dir=Path(args.union_init_dir),
                union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
                adapter_seed=ADAPTER_SEED,
                num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            )
        if args.cell == "B2D":
            provider = MorganAtomStateProvider(Path(args.morgan_overlay))
        if args.cell == "F3D":
            if args.shuffle_overlay is None:
                raise PF10GrammarTrainingError("F3D requires --shuffle-overlay")
            from .build_pf10_matched_motif_overlay_v1 import MatchedMotifStateProvider

            shuffle_provider = MatchedMotifStateProvider(Path(args.shuffle_overlay))
        report = run_grammar_cell(
            cell=args.cell,
            reader=reader,
            tokenizer=verified_tokenizer.runtime,
            model=model,
            output_dir=Path(args.output_dir),
            provider=provider,
            device=torch.device("cuda:0"),
            use_bf16=True,
            s_checkpoint=Path(args.s_checkpoint) if args.s_checkpoint else None,
            shuffle_provider=shuffle_provider,
            resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        )
        report["data_contract"] = {
            "paired_release": str(paired_release),
            "paired_release_schema_version": reader.manifest.get("schema_version"),
            "train_members": reader.train_member_count,
            "dev_members": reader.dev_member_count,
            "tokenizer_contract_sha256": verified_tokenizer.runtime.tokenizer_contract_sha256,
            "tokenizer_snapshot_sha256": verified_tokenizer.runtime.tokenizer_snapshot_sha256,
            "union_init_dir": str(Path(args.union_init_dir).expanduser().resolve()),
            "train_seed": TRAIN_SEED,
            "dev_seed": DEV_SEED,
            "dev_mask_epoch": DEV_MASK_EPOCH,
            "identity_mask_probability": IDENTITY_MASK_PROBABILITY,
        }
        report["state_artifacts"] = {
            "morgan_overlay": str(Path(args.morgan_overlay).expanduser().resolve())
            if args.cell == "B2D"
            else None,
            "s_checkpoint": str(Path(args.s_checkpoint).expanduser().resolve())
            if args.s_checkpoint is not None
            else None,
            "matched_shuffle_overlay": str(Path(args.shuffle_overlay).expanduser().resolve())
            if args.cell == "F3D"
            else None,
            "matched_shuffle_coverage": dict(shuffle_provider.manifest["coverage"])
            if shuffle_provider is not None
            else None,
        }
        report["decoded_cache_warmup"] = cache
        report["decoded_cache_final"] = reader.decoded_record_cache_stats()
        (Path(args.output_dir) / "grammar_training_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if provider is not None:
            provider.close()
        if shuffle_provider is not None:
            shuffle_provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B0", "B2D", "F3D"), required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path)
    parser.add_argument("--shuffle-overlay", type=Path)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--s-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--cache-max-pending", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = _parser().parse_args(argv)
    if args.cache_workers <= 0 or args.cache_max_pending < args.cache_workers:
        raise SystemExit("cache worker bounds are invalid")
    if args.cell == "B2D" and args.morgan_overlay is None:
        raise SystemExit("B2D requires --morgan-overlay")
    report = run_cli(args)
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "G_PROTOCOL",
    "PF10GrammarTrainingError",
    "SCHEMA_VERSION",
    "evaluate_grammar_stage",
    "run_grammar_cell",
]
