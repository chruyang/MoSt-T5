"""Single-GPU paired short screen for the 3D-MotifT5 V3 interface.

The screen is deliberately smaller than formal pretraining.  B2D and F3D
share one deterministic four-update view cycle (M+G, G-only, M-only, M+G),
the same records, optimizer, and initialization.  B0 consumes the same record
stream but disables geometry and trains only the M-only view.  Evaluation uses
one fixed 512-member dev prefix and reports aligned-versus-zero geometry NLL.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DONOR_ATOM_MAP_NAME,
    PF1PairedReleaseReader,
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
from .factorized_model_init_v3 import (
    factorized_initialization_contract_v3,
    load_deterministic_factorized_model_v3,
)
from .factorized_view_collator_v2 import GraphPortsCanonicalAtomAddressProvider
from .run_pf10_factorized_state_v1 import _TrainCursor
from .three_d_motif_training_views_v3 import (
    TRAINING_VIEW_ID,
    collate_3d_motif_training_view_v3,
)


SCHEMA_VERSION = "most-t5-p2/pf10-3d-motif-v3-short-screen/v1"
TRAIN_SEED = 20260813
DEV_SEED = 20260814
ADAPTER_SEED = 20260812
UNION_GEOMETRY_FUSION_SEED = 20260808
NUM_E3FP_EMBEDDINGS = 4096
DEV_RECORD_COUNT = 512
EVALUATION_UPDATES = (0, 500, 1000)
VIEW_CYCLE = ("m_plus_g", "g_only", "m_only", "m_plus_g")
PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=1.0e-3,
    warmup_updates=100,
    total_updates=1000,
    final_learning_rate=1.0e-5,
    warmup_start_factor=0.1,
    gradient_clip_norm=1.0,
    weight_decay=0.0,
    micro_batch_size=32,
    gradient_accumulation_steps=4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1.0e-6,
)


class ThreeDMotifV3ShortScreenError(RuntimeError):
    """The paired V3 short-screen contract failed."""


def view_for_update(cell: str, update: int) -> str:
    if cell not in {"B0", "B2D", "F3D"}:
        raise ThreeDMotifV3ShortScreenError("cell must be B0, B2D or F3D")
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise ThreeDMotifV3ShortScreenError("update must be a positive integer")
    if cell == "B0":
        return "m_only"
    return VIEW_CYCLE[(update - 1) % len(VIEW_CYCLE)]


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _motif_records(rows: Sequence[Any]) -> tuple[Any, ...]:
    records = tuple(getattr(row, "motif_record", None) for row in rows)
    if any(record is None for record in records):
        raise ThreeDMotifV3ShortScreenError("paired row lacks a motif record")
    return records


def _collate(
    rows: Sequence[Any],
    *,
    view_id: str,
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    seed: int,
    epoch: int,
    device: torch.device,
):
    return collate_3d_motif_training_view_v3(
        _motif_records(rows),
        view_id=view_id,
        tokenizer=tokenizer,
        seed=seed,
        epoch=epoch,
        atom_address_provider=addresses,
        atom_state_provider=provider,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        device=device,
    )


def _forward(model: Any, batch: Any, *, memory_mode: str) -> tuple[Any, Any]:
    inputs = batch.model_inputs()
    inputs["state_memory_mode"] = memory_mode
    inputs["use_cache"] = False
    output = model(**inputs)
    loss = getattr(output, "loss", None)
    t5_output = getattr(output, "t5_output", None)
    logits = getattr(t5_output, "logits", None)
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not bool(torch.isfinite(loss))
        or not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
    ):
        raise ThreeDMotifV3ShortScreenError("V3 forward omitted finite CE/logits")
    return loss, logits


def _load_dev_prefix(reader: PF1PairedReleaseReader) -> tuple[Any, ...]:
    rows: list[Any] = []
    for batch in reader.iter_dev(batch_size=PROTOCOL.micro_batch_size):
        remaining = DEV_RECORD_COUNT - len(rows)
        rows.extend(batch[:remaining])
        if len(rows) == DEV_RECORD_COUNT:
            break
    if len(rows) != DEV_RECORD_COUNT:
        raise ThreeDMotifV3ShortScreenError("paired release lacks fixed dev prefix")
    return tuple(rows)


def _evaluate_one_view(
    model: Any,
    *,
    rows: Sequence[Any],
    view_id: str,
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    device: torch.device,
    memory_mode: str,
) -> dict[str, object]:
    model.eval()
    nll_sum = 0.0
    target_tokens = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(rows), PROTOCOL.micro_batch_size):
            batch = _collate(
                rows[start : start + PROTOCOL.micro_batch_size],
                view_id=view_id,
                tokenizer=tokenizer,
                addresses=addresses,
                provider=provider,
                seed=DEV_SEED,
                epoch=0,
                device=device,
            )
            labels = batch.labels
            if labels is None:
                raise ThreeDMotifV3ShortScreenError("dev view lacks CE labels")
            count = int((labels != -100).sum().item())
            if count <= 0:
                raise ThreeDMotifV3ShortScreenError("dev batch lacks targets")
            with _autocast(device):
                loss, logits = _forward(model, batch, memory_mode=memory_mode)
            mask = labels != -100
            correct += int((logits.argmax(-1)[mask] == labels[mask]).sum().item())
            nll_sum += float(loss.detach().float().cpu().item()) * count
            target_tokens += count
    return {
        "view_id": view_id,
        "state_memory_mode": memory_mode,
        "members": len(rows),
        "target_tokens": target_tokens,
        "token_weighted_nll": nll_sum / target_tokens,
        "masked_token_accuracy": correct / target_tokens,
    }


def evaluate(
    model: Any,
    *,
    cell: str,
    rows: Sequence[Any],
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    device: torch.device,
) -> dict[str, object]:
    views = ("m_only",) if cell == "B0" else ("m_only", "m_plus_g", "g_only")
    reports: list[dict[str, object]] = []
    for view_id in views:
        aligned = _evaluate_one_view(
            model,
            rows=rows,
            view_id=view_id,
            tokenizer=tokenizer,
            addresses=addresses,
            provider=provider,
            device=device,
            memory_mode="zero" if view_id == "m_only" else "aligned",
        )
        reports.append(aligned)
        if view_id != "m_only":
            zero = _evaluate_one_view(
                model,
                rows=rows,
                view_id=view_id,
                tokenizer=tokenizer,
                addresses=addresses,
                provider=provider,
                device=device,
                memory_mode="zero",
            )
            zero["zero_minus_aligned_delta_nll"] = (
                float(zero["token_weighted_nll"])
                - float(aligned["token_weighted_nll"])
            )
            reports.append(zero)
    return {"dev_records": len(rows), "views": reports}


def run_cell(
    *,
    cell: str,
    reader: PF1PairedReleaseReader,
    tokenizer: Any,
    addresses: Any,
    provider: Any,
    model: Any,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model.to(device)
    model.train()
    optimizer = build_pf1_optimizer(model, PROTOCOL)
    scheduler = PF1LearningRateSchedule(optimizer, PROTOCOL)
    cursor = _TrainCursor(reader, PROTOCOL.micro_batch_size)
    dev_rows = _load_dev_prefix(reader)
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    evaluations = [
        {"update": 0, **evaluate(
            model,
            cell=cell,
            rows=dev_rows,
            tokenizer=tokenizer,
            addresses=addresses,
            provider=provider,
            device=device,
        )}
    ]
    view_updates = {"m_only": 0, "m_plus_g": 0, "g_only": 0}
    view_target_tokens = {"m_only": 0, "m_plus_g": 0, "g_only": 0}
    preclip_norms: list[float] = []
    clipped_updates = 0
    members_seen = 0
    started = time.perf_counter()

    for update in range(1, PROTOCOL.total_updates + 1):
        view_id = view_for_update(cell, update)
        model.train()
        batches = []
        counts = []
        update_members = 0
        for _ in range(PROTOCOL.gradient_accumulation_steps):
            epoch, rows = cursor.next()
            update_members += len(rows)
            batch = _collate(
                rows,
                view_id=view_id,
                tokenizer=tokenizer,
                addresses=addresses,
                provider=provider,
                seed=TRAIN_SEED,
                epoch=epoch,
                device=device,
            )
            if batch.labels is None:
                raise ThreeDMotifV3ShortScreenError("train view lacks labels")
            count = int((batch.labels != -100).sum().item())
            if count <= 0:
                raise ThreeDMotifV3ShortScreenError("train batch lacks targets")
            batches.append(batch)
            counts.append(count)
        total = sum(counts)
        optimizer.zero_grad(set_to_none=True)
        for batch, count in zip(batches, counts):
            with _autocast(device):
                loss, _logits = _forward(
                    model,
                    batch,
                    memory_mode="zero" if view_id == "m_only" else "aligned",
                )
                contribution = loss * (count / total)
            contribution.backward()
        preclip = clip_pf1_gradients(model, PROTOCOL)
        if not math.isfinite(preclip):
            raise ThreeDMotifV3ShortScreenError("gradient norm is non-finite")
        preclip_norms.append(preclip)
        clipped_updates += int(preclip > PROTOCOL.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        members_seen += update_members
        view_updates[view_id] += 1
        view_target_tokens[view_id] += total

        if update in EVALUATION_UPDATES:
            evaluations.append(
                {"update": update, **evaluate(
                    model,
                    cell=cell,
                    rows=dev_rows,
                    tokenizer=tokenizer,
                    addresses=addresses,
                    provider=provider,
                    device=device,
                )}
            )

    wall_seconds = time.perf_counter() - started
    checkpoint = output_dir / "model_state.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "cell": cell,
            "completed_updates": PROTOCOL.total_updates,
            "protocol": asdict(PROTOCOL),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "paired_short_screen_not_formal_pretraining",
        "cell": cell,
        "state_kind": "none" if cell == "B0" else (
            "e3fp" if provider is None else str(provider.state_kind)
        ),
        "training_view_contract": TRAINING_VIEW_ID,
        "view_cycle": ["m_only"] if cell == "B0" else list(VIEW_CYCLE),
        "protocol": asdict(PROTOCOL),
        "view_updates": view_updates,
        "view_target_tokens": view_target_tokens,
        "members_seen": members_seen,
        "evaluations": evaluations,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clip_rate": clipped_updates / PROTOCOL.total_updates,
        "wall_seconds": wall_seconds,
        "members_per_second": members_seen / wall_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint": str(checkpoint),
        "scientific_boundary": (
            "F3D_vs_B2D_is_the_primary_state_comparison; "
            "B0_has_no_geometry-required_view"
        ),
    }
    (output_dir / "short_screen_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ThreeDMotifV3ShortScreenError("one CUDA BF16 device is required")
    paired = Path(args.paired_release).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ThreeDMotifV3ShortScreenError("output directory already exists")
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired)
    cache = reader.warm_decoded_record_cache(
        workers=int(args.cache_workers),
        max_pending=int(args.cache_workers) * 4,
    )
    addresses = GraphPortsCanonicalAtomAddressProvider(
        paired / DONOR_ATOM_MAP_NAME
    )
    provider = None
    if args.cell == "B2D":
        if args.morgan_overlay is None:
            raise ThreeDMotifV3ShortScreenError("B2D requires --morgan-overlay")
        provider = MorganAtomStateProvider(Path(args.morgan_overlay))
    elif args.morgan_overlay is not None:
        raise ThreeDMotifV3ShortScreenError("only B2D accepts --morgan-overlay")
    contract = factorized_initialization_contract_v3(
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        state_level2_weight=0.25,
        state_embedding_dim=64,
        atom_memory_dim=128,
        max_identity_span_length=128,
        max_atoms_per_motif=128,
        geometry_fraction=0.5,
    )
    try:
        model = load_deterministic_factorized_model_v3(
            base_model_snapshot=Path(args.base_model_snapshot),
            base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
            union_tokenizer_dir=paired / "union_tokenizer",
            union_init_dir=Path(args.union_init_dir),
            union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
            adapter_seed=ADAPTER_SEED,
            num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        )
        report = run_cell(
            cell=args.cell,
            reader=reader,
            tokenizer=tokenizer.runtime,
            addresses=addresses,
            provider=provider,
            model=model,
            output_dir=output_dir,
            device=torch.device("cuda:0"),
        )
        report["initialization_contract"] = contract
        report["decoded_record_cache_warmup"] = cache
        (output_dir / "short_screen_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if provider is not None:
            provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B0", "B2D", "F3D"), required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run_cli(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_UPDATES",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "ThreeDMotifV3ShortScreenError",
    "VIEW_CYCLE",
    "evaluate",
    "run_cell",
    "run_cli",
    "view_for_update",
]
