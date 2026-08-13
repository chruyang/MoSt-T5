#!/usr/bin/env python3
"""Run one paired PF1-scale anchored V4 shell-fusion screen cell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Callable, Mapping, Sequence

import torch

from most_t5_next.p1.build_anchored_candidate_tokenizer_v1 import (
    load_verified_anchored_candidate_tokenizer,
)
from most_t5_next.p1.pf1_optimization import (
    G_CODEC_PF1_PROTOCOL,
    PF1LearningRateSchedule,
    clip_pf1_gradients,
)

from .factorized_model_init_v4 import (
    factorized_initialization_contract_v4,
    load_deterministic_factorized_model_v4,
)
from .build_anchored_v4_matched_state_overlay_v1 import (
    AnchoredV4MatchedStateProvider,
)
from .motif_geometry_adapter_v4 import SHELL_FUSION_MODES
from .pf10_training_tensor_cache_v1 import (
    CacheTrainingSample,
    CachedV3Batch,
    CachedV3Collator,
    IndexedPF10TrainingTensorCache,
    build_v3_cache_dataloader,
)


SCHEMA_VERSION = "most-t5-p2/anchored-v4-shell-screen/v1"
TRAIN_SEED = 20260807
DEV_SEED = 20260817
ADAPTER_SEED = 20260809
GEOMETRY_FUSION_SEED = 20260808
EVALUATION_UPDATES = (0, 500, 1000)
COMPONENT_MODES = ("both", "carrier_only", "endpoint_only")


class AnchoredV4ShellScreenError(RuntimeError):
    """One paired shell-screen cell cannot be interpreted safely."""


def cell_contract(cell: str, shell_fusion_mode: str) -> dict[str, str]:
    if cell == "B0":
        return {
            "cell": cell,
            "state_kind": "none",
            "view_id": "m_only",
            "memory_mode": "zero",
            "shell_fusion_mode": shell_fusion_mode,
        }
    if cell == "B2D":
        return {
            "cell": cell,
            "state_kind": "coordinate_blind_morgan",
            "view_id": "m_plus_g",
            "memory_mode": "aligned",
            "shell_fusion_mode": shell_fusion_mode,
        }
    if cell == "F3D":
        if shell_fusion_mode not in SHELL_FUSION_MODES:
            raise AnchoredV4ShellScreenError("F3D shell mode is not frozen")
        return {
            "cell": cell,
            "state_kind": "e3fp",
            "view_id": "m_plus_g",
            "memory_mode": "aligned",
            "shell_fusion_mode": shell_fusion_mode,
        }
    raise AnchoredV4ShellScreenError("cell must be B0, B2D or F3D")


def _fused_adamw(
    model: torch.nn.Module, protocol: object = G_CODEC_PF1_PROTOCOL
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise AnchoredV4ShellScreenError("model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=protocol.base_learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        eps=protocol.epsilon,
        weight_decay=protocol.weight_decay,
        fused=True,
    )


def _dev_batches(
    *,
    cache_root: Path,
    tokenizer: object,
    cell: str,
    view_id: str,
    micro_batch_size: int = G_CODEC_PF1_PROTOCOL.micro_batch_size,
) -> tuple[CachedV3Batch, ...]:
    dataset = IndexedPF10TrainingTensorCache(cache_root)
    collator = CachedV3Collator(
        cache=dataset,
        tokenizer=tokenizer,
        cell="B2D" if cell == "B2D" else "F3D",
        seed=DEV_SEED,
    )
    indices = dataset.split_indices("dev")
    batches = []
    size = micro_batch_size
    for start in range(0, len(indices), size):
        rows = tuple(
            CacheTrainingSample(
                record=dataset[index],
                epoch=0,
                view_id=view_id,
            )
            for index in indices[start : start + size]
        )
        batches.append(collator(rows))
    if sum(len(batch.record_ids) for batch in batches) != len(indices):
        raise AnchoredV4ShellScreenError("dev cache replay is incomplete")
    return tuple(batches)


def _forward_inputs(
    batch: CachedV3Batch,
    *,
    device: torch.device,
    memory_mode: str,
    component_mode: str,
) -> tuple[dict[str, object], torch.Tensor]:
    moved = batch.to(device)
    values = dict(moved.inputs)
    values["state_memory_mode"] = memory_mode
    values["geometry_component_mode"] = component_mode
    values["use_cache"] = False
    labels = values.get("labels")
    if not isinstance(labels, torch.Tensor):
        raise AnchoredV4ShellScreenError("CE labels are absent")
    return values, labels


def _replace_state_batches(
    batches: Sequence[CachedV3Batch],
    *,
    provider: object,
) -> tuple[CachedV3Batch, ...]:
    """Replace only the unpadded atom-state rows of already-collated dev batches."""

    replaced = []
    for batch in batches:
        values = dict(batch.inputs)
        states = values.get("e3fp_input_ids")
        atom_mask = values.get("atom_mask")
        if not isinstance(states, torch.Tensor) or not isinstance(atom_mask, torch.Tensor):
            raise AnchoredV4ShellScreenError("dev batch lacks atom-state tensors")
        states = states.clone()
        for row_index, record_id in enumerate(batch.record_ids):
            rows = tuple(tuple(int(value) for value in row) for row in provider.get(record_id))
            atom_count = int(atom_mask[row_index].sum().item())
            if len(rows) != atom_count or any(len(row) != 4 for row in rows):
                raise AnchoredV4ShellScreenError("matched state shape differs from dev batch")
            states[row_index, :atom_count] = torch.as_tensor(
                rows, dtype=states.dtype, device=states.device
            )
        values["e3fp_input_ids"] = states
        replaced.append(
            CachedV3Batch(
                view_id=batch.view_id,
                epoch=batch.epoch,
                record_ids=batch.record_ids,
                exact_identity_sha256=batch.exact_identity_sha256,
                inputs=values,
                labels=batch.labels,
            )
        )
    return tuple(replaced)


def _evaluate(
    model: torch.nn.Module,
    *,
    batches: Sequence[CachedV3Batch],
    cell: str,
    device: torch.device,
    matched_batches: Sequence[CachedV3Batch] | None = None,
) -> dict[str, object]:
    model.eval()
    conditions = (
        (("zero", "both"),)
        if cell == "B0"
        else (
            ("aligned", "both"),
            ("aligned", "carrier_only"),
            ("aligned", "endpoint_only"),
            ("zero", "both"),
        )
    )
    specifications = [
        (memory_mode, memory_mode, component_mode, batches)
        for memory_mode, component_mode in conditions
    ]
    if matched_batches is not None:
        if cell == "B0":
            raise AnchoredV4ShellScreenError("B0 cannot consume a matched overlay")
        specifications.append(("matched_donor", "aligned", "both", matched_batches))
    reports = []
    with torch.no_grad():
        for report_memory_mode, forward_memory_mode, component_mode, condition_batches in specifications:
            nll_sum = 0.0
            target_tokens = 0
            correct = 0
            for batch in condition_batches:
                values, labels = _forward_inputs(
                    batch,
                    device=device,
                    memory_mode=forward_memory_mode,
                    component_mode=component_mode,
                )
                count = int((labels != -100).sum().item())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**values)
                loss = output.loss
                logits = getattr(output.t5_output, "logits", None)
                if (
                    not isinstance(loss, torch.Tensor)
                    or not bool(torch.isfinite(loss).item())
                    or not isinstance(logits, torch.Tensor)
                ):
                    raise AnchoredV4ShellScreenError("dev forward is non-finite")
                mask = labels != -100
                correct += int((logits.argmax(-1)[mask] == labels[mask]).sum().item())
                nll_sum += float(loss.float().item()) * count
                target_tokens += count
            reports.append(
                {
                    "state_memory_mode": report_memory_mode,
                    "geometry_component_mode": component_mode,
                    "members": sum(len(batch.record_ids) for batch in condition_batches),
                    "target_tokens": target_tokens,
                    "token_weighted_nll": nll_sum / target_tokens,
                    "masked_token_accuracy": correct / target_tokens,
                }
            )
    if cell != "B0":
        aligned = reports[0]["token_weighted_nll"]
        zero = reports[-1]["token_weighted_nll"]
        if matched_batches is not None:
            zero = reports[-2]["token_weighted_nll"]
            reports[-1]["matched_minus_aligned_delta_nll"] = (
                float(reports[-1]["token_weighted_nll"]) - float(aligned)
            )
        reports[-1]["zero_minus_aligned_delta_nll"] = float(zero) - float(aligned)
        if matched_batches is not None:
            reports[-2]["zero_minus_aligned_delta_nll"] = float(zero) - float(aligned)
            reports[-1].pop("zero_minus_aligned_delta_nll", None)
    return {"conditions": reports}


def run(
    args: argparse.Namespace,
    *,
    model_loader: Callable[..., torch.nn.Module] = load_deterministic_factorized_model_v4,
    initialization_contract_builder: Callable[..., dict[str, object]] = (
        factorized_initialization_contract_v4
    ),
    contract_builder: Callable[[str, str], dict[str, str]] = cell_contract,
    schema_version: str = SCHEMA_VERSION,
    scope: str = "pf1_candidate_shell_selection_not_formal_pretraining",
    manifest_filename: str = "shell_screen_manifest.json",
    training_component_mode: str = "both",
    optimization_protocol: object = G_CODEC_PF1_PROTOCOL,
    evaluation_updates: Sequence[int] = EVALUATION_UPDATES,
) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise AnchoredV4ShellScreenError("one CUDA BF16 device is required")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise AnchoredV4ShellScreenError("output directory must be new")
    output_dir.mkdir(parents=True)
    if training_component_mode not in COMPONENT_MODES:
        raise AnchoredV4ShellScreenError("training component mode is invalid")
    if (
        tuple(evaluation_updates)[0] != 0
        or tuple(evaluation_updates)[-1] != optimization_protocol.total_updates
    ):
        raise AnchoredV4ShellScreenError("evaluation schedule must span update 0 to final")
    contract = contract_builder(args.cell, args.shell_fusion_mode)
    tokenizer = load_verified_anchored_candidate_tokenizer(
        base_snapshot=args.base_tokenizer_snapshot,
        output_dir=args.anchored_tokenizer_dir,
        semantic_plan_sha256=args.semantic_plan_sha256,
    )
    model = model_loader(
        base_model_snapshot=args.base_model_snapshot,
        base_tokenizer_snapshot=args.base_tokenizer_snapshot,
        anchored_tokenizer_dir=args.anchored_tokenizer_dir,
        semantic_plan_sha256=args.semantic_plan_sha256,
        union_init_dir=args.union_init_dir,
        union_geometry_fusion_seed=GEOMETRY_FUSION_SEED,
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=4096,
        state_level2_weight=0.25,
        state_embedding_dim=64,
        atom_memory_dim=128,
        max_identity_span_length=128,
        max_atoms_per_motif=128,
        geometry_fraction=0.5,
        shell_fusion_mode=args.shell_fusion_mode,
    )
    device = torch.device("cuda", 0)
    model.to(device)
    optimizer = _fused_adamw(model, optimization_protocol)
    scheduler = PF1LearningRateSchedule(optimizer, optimization_protocol)
    train_loader = build_v3_cache_dataloader(
        cache_root=args.cache_root,
        tokenizer=tokenizer.runtime,
        cell="B2D" if args.cell == "B2D" else "F3D",
        seed=TRAIN_SEED,
        micro_batch_size=optimization_protocol.micro_batch_size,
        gradient_accumulation_steps=optimization_protocol.gradient_accumulation_steps,
        total_updates=optimization_protocol.total_updates,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle_seed=TRAIN_SEED,
        fixed_view_id=contract["view_id"],
    )
    dev_batches = _dev_batches(
        cache_root=args.cache_root,
        tokenizer=tokenizer.runtime,
        cell=args.cell,
        view_id=contract["view_id"],
        micro_batch_size=optimization_protocol.micro_batch_size,
    )
    matched_provider = None
    matched_batches = None
    if args.matched_overlay is not None:
        if args.cell == "B0":
            raise AnchoredV4ShellScreenError("matched overlay is only valid for B2D/F3D")
        matched_provider = AnchoredV4MatchedStateProvider(
            args.matched_overlay,
            state_kind="morgan" if args.cell == "B2D" else "e3fp",
        )
        matched_batches = _replace_state_batches(
            dev_batches,
            provider=matched_provider,
        )

    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    evaluations = [{"update": 0, **_evaluate(
        model, batches=dev_batches, cell=args.cell, device=device
    )}]
    iterator = iter(train_loader)
    member_counts = []
    target_tokens_seen = 0
    preclip_norms = []
    clipped_updates = 0
    started = time.perf_counter()
    for update in range(1, optimization_protocol.total_updates + 1):
        model.train()
        microbatches = [
            next(iterator)
            for _ in range(optimization_protocol.gradient_accumulation_steps)
        ]
        counts = [int((batch.labels != -100).sum().item()) for batch in microbatches]
        total_targets = sum(counts)
        optimizer.zero_grad(set_to_none=True)
        for batch, count in zip(microbatches, counts):
            values, _labels = _forward_inputs(
                batch,
                device=device,
                memory_mode=contract["memory_mode"],
                component_mode=training_component_mode,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**values).loss * (count / total_targets)
            loss.backward()
        preclip = clip_pf1_gradients(model, optimization_protocol)
        if not math.isfinite(preclip):
            raise AnchoredV4ShellScreenError("gradient norm is non-finite")
        preclip_norms.append(preclip)
        clipped_updates += int(preclip > optimization_protocol.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        members = sum(len(batch.record_ids) for batch in microbatches)
        member_counts.append(members)
        target_tokens_seen += total_targets

        if update in tuple(evaluation_updates)[1:]:
            evaluations.append({"update": update, **_evaluate(
                model,
                batches=dev_batches,
                cell=args.cell,
                device=device,
                matched_batches=(
                    matched_batches
                    if update == optimization_protocol.total_updates
                    else None
                ),
            )})
        if update % 100 == 0:
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "schema_version": schema_version,
                        "status": "running",
                        "cell": contract,
                        "completed_updates": update,
                        "total_updates": optimization_protocol.total_updates,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    wall_seconds = time.perf_counter() - started
    initialization = initialization_contract_builder(
        semantic_plan_sha256=args.semantic_plan_sha256,
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=4096,
        state_level2_weight=0.25,
        state_embedding_dim=64,
        atom_memory_dim=128,
        max_identity_span_length=128,
        max_atoms_per_motif=128,
        geometry_fraction=0.5,
        shell_fusion_mode=args.shell_fusion_mode,
    )
    checkpoint = None
    if args.save_final_checkpoint:
        checkpoint_path = output_dir / "final_model_state.pt"
        torch.save(
            {
                "schema_version": schema_version,
                "completed_updates": optimization_protocol.total_updates,
                "cell": contract,
                "initialization": initialization,
                "model_state_dict": {
                    key: (
                        value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
        checkpoint = {
            "file": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "model_only": True,
            "optimizer_or_scheduler_saved": False,
        }
    report = {
        "schema_version": schema_version,
        "status": "pass",
        "scope": scope,
        "cell": contract,
        "initialization": initialization,
        "optimization": {
            **asdict(optimization_protocol),
            "optimizer": "torch.optim.AdamW",
            "fused": True,
            "schedule": "linear_warmup_then_cosine",
        },
        "data": {
            "cache_root": str(Path(args.cache_root).resolve()),
            "train_seed": TRAIN_SEED,
            "dev_seed": DEV_SEED,
            "num_workers": args.num_workers,
            "dynamic_corruption": True,
            "drop_last": False,
            "members_seen": sum(member_counts),
            "min_members_per_update": min(member_counts),
            "max_members_per_update": max(member_counts),
            "mean_members_per_update": statistics.fmean(member_counts),
            "target_tokens_seen": target_tokens_seen,
            "matched_overlay": (
                str(Path(args.matched_overlay).resolve())
                if args.matched_overlay is not None
                else None
            ),
            "training_geometry_component_mode": training_component_mode,
        },
        "evaluations": evaluations,
        "training": {
            "wall_seconds": wall_seconds,
            "members_per_second": sum(member_counts) / wall_seconds,
            "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
            "max_preclip_gradient_norm": max(preclip_norms),
            "clip_rate": clipped_updates / len(preclip_norms),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "checkpoint_saved": checkpoint is not None,
        "checkpoint": checkpoint,
        "scientific_boundary": {
            "shell_quality_inferred_from_smoke": False,
            "selection_uses_fixed_dev_after_equal_training": True,
            "three_dimensional_gain_requires_b2d_and_perturbation_controls": True,
            "g_only_identity_reconstruction_used": False,
        },
    }
    (output_dir / manifest_filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "COMPLETE").write_text("pass\n", encoding="ascii")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B0", "B2D", "F3D"), required=True)
    parser.add_argument("--shell-fusion-mode", choices=SHELL_FUSION_MODES, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--matched-overlay", type=Path)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
