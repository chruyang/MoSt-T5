"""Leakage-free PF-10 held-out atom-state mechanism screen.

The stock T5 is warm-started from one shared B0 grammar checkpoint and frozen.
For every eligible motif, exactly one atom's L1/L2 state is hidden while a
same-motif peer remains visible.  Only the adapter and a disposable matching
head train; the target atom memory is detached from the adapter graph.
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
from torch import Tensor, nn

from most_t5_next.p1.pf1_optimization import (
    PF1LearningRateSchedule,
    PF1OptimizationProtocol,
    build_pf1_optimizer,
    clip_pf1_gradients,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .factorized_model_init_v3 import load_deterministic_factorized_model_v3
from .held_out_motif_state_v4 import (
    HELD_OUT_MATCHING_ID,
    HeldOutAtomStateMatchingHeadV4,
    HeldOutMotifStateError,
    HeldOutMotifStatePlan,
    build_held_out_motif_state_plan,
    eligible_held_out_anchor_count,
)
from .motif_state_matching_v3 import MotifStateMatchingOutput
from .pf10_training_tensor_cache_v1 import (
    CacheTrainingSample,
    CachedV3Batch,
    CachedV3Collator,
    IndexedPF10TrainingTensorCache,
    build_v3_cache_dataloader,
)
from .run_pf10_3d_motif_v3_matching_only_v1 import (
    COMPONENT_MODES,
    _summarize_matching,
)
from .run_pf10_3d_motif_v3_matching_v1 import (
    ADAPTER_SEED,
    DEV_SEED,
    MATCHING_HEAD_SEED,
    NUM_E3FP_EMBEDDINGS,
    TRAIN_SEED,
    UNION_GEOMETRY_FUSION_SEED,
)
from .run_pf10_3d_motif_v3_short_v1 import SCHEMA_VERSION as B0_SCHEMA_VERSION


SCHEMA_VERSION = "most-t5-p2/pf10-3d-motif-v4-held-out/v1"
GEOMETRY_FRACTION = 0.5
TRAIN_COMPONENT_MODE = "carrier_only"
EVALUATION_UPDATES = (0, 250, 500, 1000)
DEV_RECORD_LIMIT = 128
PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=3.0e-4,
    warmup_updates=100,
    total_updates=1000,
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


class V4HeldOutRunnerError(RuntimeError):
    """The held-out V4 execution contract was violated."""


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _tensor(inputs: Mapping[str, object], name: str) -> Tensor:
    value = inputs.get(name)
    if not isinstance(value, Tensor):
        raise V4HeldOutRunnerError(f"cached V4 batch lacks tensor {name}")
    return value


def load_shared_b0_warm_start(model: nn.Module, checkpoint_path: Path) -> Mapping[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != B0_SCHEMA_VERSION
        or payload.get("cell") != "B0"
    ):
        raise V4HeldOutRunnerError("shared B0 checkpoint contract mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise V4HeldOutRunnerError("shared B0 checkpoint omits model state")
    model.load_state_dict(state, strict=True)
    return payload


def freeze_t5_for_held_out(model: nn.Module, head: nn.Module) -> nn.Module:
    for parameter in model.t5.parameters():
        parameter.requires_grad_(False)
    for parameter in model.adapter.parameters():
        parameter.requires_grad_(True)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    model.t5.eval()
    model.adapter.train()
    head.train()
    return nn.ModuleDict({"adapter": model.adapter, "held_out_head": head})


def build_plan(batch: CachedV3Batch, *, seed: int) -> HeldOutMotifStatePlan:
    inputs = batch.inputs
    try:
        return build_held_out_motif_state_plan(
            e3fp_input_ids=_tensor(inputs, "e3fp_input_ids"),
            atom_mask=_tensor(inputs, "atom_mask"),
            atom_to_motif=_tensor(inputs, "atom_to_motif"),
            atom_local_positions=_tensor(inputs, "atom_local_positions"),
            motif_mask=_tensor(inputs, "motif_mask"),
            record_ids=batch.record_ids,
            exact_identity_sha256=batch.exact_identity_sha256,
            mask_token_id=int(inputs["e3fp_mask_token_id"]),
            seed=seed,
            epoch=batch.epoch,
        )
    except (KeyError, TypeError, ValueError, HeldOutMotifStateError) as exc:
        raise V4HeldOutRunnerError("held-out plan construction failed") from exc


def forward_held_out_matching(
    model: nn.Module,
    head: HeldOutAtomStateMatchingHeadV4,
    batch: CachedV3Batch,
    plan: HeldOutMotifStatePlan,
    *,
    component_mode: str,
) -> MotifStateMatchingOutput:
    if component_mode not in COMPONENT_MODES:
        raise V4HeldOutRunnerError("unknown V4 component mode")
    inputs = batch.inputs
    input_ids = _tensor(inputs, "input_ids")
    attention_mask = _tensor(inputs, "attention_mask")
    atom_mask = _tensor(inputs, "atom_mask")
    atom_roles = _tensor(inputs, "atom_is_attachment")
    embeddings = model.t5.get_input_embeddings()(input_ids)
    with torch.no_grad():
        original_atom_memory = model.adapter.encode_atom_memory(
            _tensor(inputs, "e3fp_input_ids"),
            atom_mask,
            atom_roles,
        ).detach()
    encoded = model.adapter.encode(
        embeddings,
        attention_mask=attention_mask.to(torch.bool),
        e3fp_input_ids=plan.corrupted_e3fp_input_ids,
        atom_mask=atom_mask,
        atom_to_motif=_tensor(inputs, "atom_to_motif"),
        motif_mask=_tensor(inputs, "motif_mask"),
        motif_to_carrier=_tensor(inputs, "motif_to_carrier"),
        identity_span_bounds=_tensor(inputs, "identity_span_bounds"),
        endpoint_token_to_atom=_tensor(inputs, "endpoint_token_to_atom"),
        atom_is_attachment=atom_roles,
        state_memory_mode="aligned",
        geometry_component_mode=component_mode,
    )
    encoder_output = model.t5.encoder(
        inputs_embeds=encoded.fused_embeddings,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = getattr(encoder_output, "last_hidden_state", None)
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise V4HeldOutRunnerError("T5 encoder omitted hidden states")
    return head(
        encoder_hidden=hidden,
        original_atom_memory=original_atom_memory,
        motif_mask=_tensor(inputs, "motif_mask"),
        motif_to_carrier=_tensor(inputs, "motif_to_carrier"),
        target_atom_indices=plan.target_atom_indices,
        target_local_positions=plan.target_local_positions,
        target_motif_mask=plan.target_motif_mask,
        exact_identity_sha256=batch.exact_identity_sha256,
    )


def _new_head(hidden_size: int, atom_memory_dim: int) -> HeldOutAtomStateMatchingHeadV4:
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(MATCHING_HEAD_SEED)
        return HeldOutAtomStateMatchingHeadV4(
            hidden_size=hidden_size,
            atom_memory_dim=atom_memory_dim,
            projection_dim=128,
            temperature=0.1,
        )
    finally:
        torch.random.set_rng_state(state)


def evaluate_components(
    model: nn.Module,
    head: HeldOutAtomStateMatchingHeadV4,
    *,
    cache: IndexedPF10TrainingTensorCache,
    tokenizer: Any,
    cell: str,
    device: torch.device,
) -> dict[str, object]:
    model.t5.eval(); model.adapter.eval(); head.eval()
    indices = cache.split_indices("dev")[:DEV_RECORD_LIMIT]
    collator = CachedV3Collator(cache=cache, tokenizer=tokenizer, cell=cell, seed=DEV_SEED)
    by_mode: dict[str, list[MotifStateMatchingOutput]] = {
        mode: [] for mode in COMPONENT_MODES
    }
    selected_targets = 0
    visible_peers = 0
    with torch.no_grad():
        for start in range(0, len(indices), PROTOCOL.micro_batch_size):
            samples = tuple(
                CacheTrainingSample(cache[index], 0, "m_plus_g")
                for index in indices[start : start + PROTOCOL.micro_batch_size]
            )
            cpu_batch = collator(samples)
            cpu_plan = build_plan(cpu_batch, seed=DEV_SEED)
            selected_targets += cpu_plan.selected_targets
            visible_peers += int(cpu_plan.visible_peer_counts.sum().item())
            batch = cpu_batch.to(device)
            plan = cpu_plan.to(device)
            for mode in COMPONENT_MODES:
                with _autocast(device):
                    by_mode[mode].append(
                        forward_held_out_matching(
                            model, head, batch, plan, component_mode=mode
                        )
                    )
    return {
        "components": {
            mode: _summarize_matching(outputs)
            for mode, outputs in by_mode.items()
        },
        "selected_targets": selected_targets,
        "visible_peer_count": visible_peers,
        "target_has_visible_peer": True,
    }


def run_held_out_screen(
    *,
    cell: str,
    cache_root: Path,
    tokenizer: Any,
    model: nn.Module,
    head: HeldOutAtomStateMatchingHeadV4,
    b0_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
) -> dict[str, object]:
    if cell not in {"B2D", "F3D"}:
        raise V4HeldOutRunnerError("held-out cell must be B2D or F3D")
    if output_dir.exists():
        raise V4HeldOutRunnerError("held-out output already exists")
    output_dir.mkdir(parents=True)
    load_shared_b0_warm_start(model, b0_checkpoint)
    model.to(device); head.to(device)
    trainable = freeze_t5_for_held_out(model, head)
    optimizer = build_pf1_optimizer(trainable, PROTOCOL)
    scheduler = PF1LearningRateSchedule(optimizer, PROTOCOL)
    loader = build_v3_cache_dataloader(
        cache_root=cache_root,
        tokenizer=tokenizer,
        cell=cell,
        seed=TRAIN_SEED,
        micro_batch_size=PROTOCOL.micro_batch_size,
        gradient_accumulation_steps=PROTOCOL.gradient_accumulation_steps,
        total_updates=PROTOCOL.total_updates,
        num_workers=num_workers,
        prefetch_factor=4,
        shuffle_seed=TRAIN_SEED,
        fixed_view_id="m_plus_g",
    )
    dev_cache = IndexedPF10TrainingTensorCache(cache_root)
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    evaluations = [{
        "update": 0,
        **evaluate_components(
            model, head, cache=dev_cache, tokenizer=tokenizer, cell=cell, device=device
        ),
    }]
    iterator = iter(loader)
    preclip_norms: list[float] = []
    clipped = 0
    members_seen = 0
    anchors_seen = 0
    selected_targets = 0
    started = time.perf_counter()
    try:
        for update in range(1, PROTOCOL.total_updates + 1):
            model.t5.eval(); model.adapter.train(); head.train()
            batches = [next(iterator) for _ in range(PROTOCOL.gradient_accumulation_steps)]
            plans = [build_plan(batch, seed=TRAIN_SEED) for batch in batches]
            counts = [
                eligible_held_out_anchor_count(
                    plan.target_motif_mask,
                    plan.target_local_positions,
                    batch.exact_identity_sha256,
                )
                for plan, batch in zip(plans, batches)
            ]
            total = sum(counts)
            if total <= 0:
                raise V4HeldOutRunnerError("held-out update has no cross-record anchors")
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch, cpu_plan, count in zip(batches, plans, counts):
                batch = cpu_batch.to(device)
                plan = cpu_plan.to(device)
                with _autocast(device):
                    result = forward_held_out_matching(
                        model,
                        head,
                        batch,
                        plan,
                        component_mode=TRAIN_COMPONENT_MODE,
                    )
                    if result.eligible_anchors != count:
                        raise V4HeldOutRunnerError("held-out anchor count drifted")
                    loss = result.loss * (count / total)
                loss.backward()
                members_seen += len(batch.record_ids)
                selected_targets += cpu_plan.selected_targets
            preclip = clip_pf1_gradients(trainable, PROTOCOL)
            if not math.isfinite(preclip):
                raise V4HeldOutRunnerError("held-out gradient is non-finite")
            preclip_norms.append(preclip)
            clipped += int(preclip > PROTOCOL.gradient_clip_norm)
            optimizer.step(); scheduler.step()
            anchors_seen += total
            if update in EVALUATION_UPDATES:
                evaluations.append({
                    "update": update,
                    **evaluate_components(
                        model,
                        head,
                        cache=dev_cache,
                        tokenizer=tokenizer,
                        cell=cell,
                        device=device,
                    ),
                })
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        loader.dataset.close(); dev_cache.close()

    wall = time.perf_counter() - started
    checkpoint = output_dir / "adapter_and_held_out_head.pt"
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "completed_updates": PROTOCOL.total_updates,
        "geometry_fraction": GEOMETRY_FRACTION,
        "training_component_mode": TRAIN_COMPONENT_MODE,
        "adapter_state_dict": model.adapter.state_dict(),
        "held_out_head_state_dict": head.state_dict(),
    }, checkpoint)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "held_out_atom_state_mechanism_screen_not_pretraining",
        "cell": cell,
        "held_out_contract": HELD_OUT_MATCHING_ID,
        "protocol": asdict(PROTOCOL),
        "geometry_fraction": GEOMETRY_FRACTION,
        "training_component_mode": TRAIN_COMPONENT_MODE,
        "b0_warm_start_checkpoint": str(b0_checkpoint),
        "t5_frozen_and_eval": True,
        "target_adapter_memory_detached": True,
        "one_target_atom_per_eligible_motif": True,
        "visible_same_motif_state_peer_required": True,
        "members_seen": members_seen,
        "selected_targets_seen": selected_targets,
        "eligible_anchors_seen": anchors_seen,
        "evaluations": evaluations,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clip_rate": clipped / PROTOCOL.total_updates,
        "wall_seconds": wall,
        "members_per_second": members_seen / wall,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint": str(checkpoint),
    }
    (output_dir / "held_out_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--b0-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise V4HeldOutRunnerError("one CUDA BF16 device is required")
    paired = Path(args.paired_release).expanduser().resolve()
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired / "union_tokenizer",
    )
    model = load_deterministic_factorized_model_v3(
        base_model_snapshot=Path(args.base_model_snapshot),
        base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
        union_tokenizer_dir=paired / "union_tokenizer",
        union_init_dir=Path(args.union_init_dir),
        union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
        adapter_seed=ADAPTER_SEED,
        num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        geometry_fraction=GEOMETRY_FRACTION,
    )
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    atom_memory_dim = int(model.adapter.atom_memory_dim)
    return run_held_out_screen(
        cell=args.cell,
        cache_root=Path(args.cache_root),
        tokenizer=tokenizer.runtime,
        model=model,
        head=_new_head(hidden_size, atom_memory_dim),
        b0_checkpoint=Path(args.b0_checkpoint),
        output_dir=Path(args.output_dir),
        device=torch.device("cuda:0"),
        num_workers=int(args.workers),
    )


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
    "GEOMETRY_FRACTION",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "TRAIN_COMPONENT_MODE",
    "V4HeldOutRunnerError",
    "build_plan",
    "evaluate_components",
    "forward_held_out_matching",
    "freeze_t5_for_held_out",
    "load_shared_b0_warm_start",
    "run_held_out_screen",
]
