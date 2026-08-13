"""Block-coordinate PF-10 V3 geometry/grammar screen.

The runner starts from a completed matching-only adapter checkpoint.  It then
alternates four equally exposed update kinds:

1. M+G same-identity matching (T5 frozen; adapter/head train),
2. G-only grammar CE (adapter/head frozen; T5 trains),
3. M-only grammar CE (adapter/head frozen; T5 trains), and
4. M+G grammar CE (adapter/head frozen; T5 trains).

Separating the blocks prevents the much larger token CE gradient from erasing
the geometry route in the same optimizer update.  V2 gives the two parameter
families independent optimizer clocks: 400 adapter/head matching updates and
1,200 T5 grammar updates.  A frozen family no longer consumes the other
family's warmup or cosine schedule.
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
from .motif_state_matching_v3 import MATCHING_ID, MotifStateMatchingHeadV3
from .pf10_training_tensor_cache_v1 import (
    CacheTrainingSample,
    CachedV3Batch,
    CachedV3Collator,
    IndexedPF10TrainingTensorCache,
    build_v3_cache_dataloader,
)
from .run_pf10_3d_motif_v3_matching_only_v1 import (
    COMPONENT_MODES,
    GEOMETRY_FRACTION,
    SCHEMA_VERSION as S_SCHEMA_VERSION,
    _new_matching_head,
    evaluate_components,
    forward_matching_only,
)
from .run_pf10_3d_motif_v3_matching_v1 import (
    ADAPTER_SEED,
    DEV_SEED,
    NUM_E3FP_EMBEDDINGS,
    TRAIN_SEED,
    UNION_GEOMETRY_FUSION_SEED,
    _eligible_anchor_count,
)


SCHEMA_VERSION = "most-t5-p2/pf10-3d-motif-v3-block-joint/v2"
BLOCK_CYCLE = ("matching_m_plus_g", "grammar_g_only", "grammar_m_only", "grammar_m_plus_g")
TOTAL_UPDATES = 1600
EVALUATION_UPDATES = (0, 800, 1600)
DEV_RECORD_LIMIT = 128
MATCHING_PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=3.0e-4,
    warmup_updates=40,
    total_updates=400,
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
GRAMMAR_PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=1.0e-3,
    warmup_updates=120,
    total_updates=1200,
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


class V3BlockJointError(RuntimeError):
    """The block-coordinate V3 execution contract was violated."""


def block_for_update(update: int) -> str:
    if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
        raise V3BlockJointError("update must be a positive integer")
    return BLOCK_CYCLE[(update - 1) % len(BLOCK_CYCLE)]


def optimization_family_for_block(block: str) -> str:
    if block == "matching_m_plus_g":
        return "matching"
    if block.startswith("grammar_") and block in BLOCK_CYCLE:
        return "grammar"
    raise V3BlockJointError("unknown block-coordinate update kind")


def _protocol_for_family(family: str) -> PF1OptimizationProtocol:
    if family == "matching":
        return MATCHING_PROTOCOL
    if family == "grammar":
        return GRAMMAR_PROTOCOL
    raise V3BlockJointError("unknown optimization family")


def _set_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def configure_block(model: nn.Module, head: nn.Module, block: str) -> None:
    if block == "matching_m_plus_g":
        _set_trainable(model.t5, False)
        _set_trainable(model.adapter, True)
        _set_trainable(head, True)
        model.t5.eval()
        model.adapter.train()
        head.train()
        return
    if block.startswith("grammar_"):
        _set_trainable(model.t5, True)
        _set_trainable(model.adapter, False)
        _set_trainable(head, False)
        model.t5.train()
        model.adapter.eval()
        head.eval()
        return
    raise V3BlockJointError("unknown block-coordinate update kind")


def load_matching_only_state(
    model: nn.Module,
    head: nn.Module,
    *,
    checkpoint_path: Path,
    cell: str,
    expected_geometry_fraction: float = GEOMETRY_FRACTION,
) -> Mapping[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise V3BlockJointError("matching-only checkpoint is not a mapping")
    if payload.get("schema_version") != S_SCHEMA_VERSION or payload.get("cell") != cell:
        raise V3BlockJointError("matching-only checkpoint contract/cell mismatch")
    checkpoint_fraction = payload.get("geometry_fraction")
    if checkpoint_fraction is None:
        manifest_path = checkpoint_path.with_name("matching_only_manifest.json")
        if not manifest_path.is_file():
            raise V3BlockJointError("matching-only geometry fraction is unbound")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_fraction = manifest.get("geometry_fraction")
    if (
        isinstance(checkpoint_fraction, bool)
        or not isinstance(checkpoint_fraction, (int, float))
        or float(checkpoint_fraction) != float(expected_geometry_fraction)
    ):
        raise V3BlockJointError("matching-only geometry fraction mismatch")
    adapter_state = payload.get("adapter_state_dict")
    head_state = payload.get("matching_head_state_dict")
    if not isinstance(adapter_state, Mapping) or not isinstance(head_state, Mapping):
        raise V3BlockJointError("matching-only checkpoint omits adapter/head state")
    model.adapter.load_state_dict(adapter_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    return payload


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _tensor(inputs: Mapping[str, object], name: str) -> Tensor:
    value = inputs.get(name)
    if not isinstance(value, Tensor):
        raise V3BlockJointError(f"cached V3 batch lacks tensor {name}")
    return value


def _grammar_forward(model: nn.Module, batch: CachedV3Batch, *, memory_mode: str):
    inputs = dict(batch.inputs)
    if memory_mode == "zero":
        inputs["state_memory_mode"] = "zero"
        inputs["geometry_component_mode"] = "zero"
    elif memory_mode in {"aligned", "both", "carrier_only", "endpoint_only"}:
        inputs["state_memory_mode"] = "aligned"
        inputs["geometry_component_mode"] = (
            "both" if memory_mode == "aligned" else memory_mode
        )
    else:
        raise V3BlockJointError("unknown grammar geometry component mode")
    inputs["use_cache"] = False
    output = model(**inputs)
    loss = getattr(output, "loss", None)
    logits = getattr(getattr(output, "t5_output", None), "logits", None)
    if (
        not isinstance(loss, Tensor)
        or loss.ndim != 0
        or not bool(torch.isfinite(loss))
        or not isinstance(logits, Tensor)
        or logits.ndim != 3
    ):
        raise V3BlockJointError("grammar forward omitted finite CE/logits")
    return output


def evaluate_grammar(
    model: nn.Module,
    *,
    cache: IndexedPF10TrainingTensorCache,
    tokenizer: Any,
    cell: str,
    device: torch.device,
) -> dict[str, object]:
    model.t5.eval()
    model.adapter.eval()
    indices = cache.split_indices("dev")[:DEV_RECORD_LIMIT]
    collator = CachedV3Collator(cache=cache, tokenizer=tokenizer, cell=cell, seed=DEV_SEED)
    reports = []
    with torch.no_grad():
        for view in ("m_only", "m_plus_g", "g_only"):
            modes = (
                ("zero",)
                if view == "m_only"
                else ("both", "carrier_only", "endpoint_only", "zero")
            )
            for mode in modes:
                nll_sum = 0.0
                targets = 0
                correct = 0
                for start in range(0, len(indices), GRAMMAR_PROTOCOL.micro_batch_size):
                    samples = tuple(
                        CacheTrainingSample(cache[index], 0, view)
                        for index in indices[start : start + GRAMMAR_PROTOCOL.micro_batch_size]
                    )
                    batch = collator(samples).to(device)
                    labels = batch.labels
                    count = int((labels != -100).sum().item())
                    with _autocast(device):
                        output = _grammar_forward(model, batch, memory_mode=mode)
                    logits = output.t5_output.logits
                    mask = labels != -100
                    nll_sum += float(output.loss.detach().float().cpu().item()) * count
                    targets += count
                    correct += int((logits.argmax(-1)[mask] == labels[mask]).sum().item())
                reports.append({
                    "view_id": view,
                    "state_memory_mode": "zero" if mode == "zero" else "aligned",
                    "geometry_component_mode": mode,
                    "records": len(indices),
                    "target_tokens": targets,
                    "token_weighted_nll": nll_sum / targets,
                    "masked_token_accuracy": correct / targets,
                })
    return {"views": reports}


def evaluate_all(
    model: nn.Module,
    head: nn.Module,
    *,
    cache: IndexedPF10TrainingTensorCache,
    tokenizer: Any,
    cell: str,
    device: torch.device,
) -> dict[str, object]:
    grammar = evaluate_grammar(
        model, cache=cache, tokenizer=tokenizer, cell=cell, device=device
    )
    matching = evaluate_components(
        model,
        head,
        cache=cache,
        tokenizer=tokenizer,
        cell=cell,
        device=device,
        max_records=DEV_RECORD_LIMIT,
    )
    return {"grammar": grammar, "matching": matching}


def run_block_joint(
    *,
    cell: str,
    cache_root: Path,
    tokenizer: Any,
    model: nn.Module,
    matching_head: nn.Module,
    s_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    num_workers: int,
) -> dict[str, object]:
    if cell not in {"B2D", "F3D"}:
        raise V3BlockJointError("block-joint cell must be B2D or F3D")
    if output_dir.exists():
        raise V3BlockJointError("block-joint output already exists")
    output_dir.mkdir(parents=True)
    load_matching_only_state(
        model,
        matching_head,
        checkpoint_path=s_checkpoint,
        cell=cell,
        expected_geometry_fraction=GEOMETRY_FRACTION,
    )
    model.to(device); matching_head.to(device)
    _set_trainable(model, True); _set_trainable(matching_head, True)
    matching_trainable = nn.ModuleDict({
        "adapter": model.adapter,
        "matching_head": matching_head,
    })
    grammar_trainable = model.t5
    optimizers = {
        "matching": build_pf1_optimizer(matching_trainable, MATCHING_PROTOCOL),
        "grammar": build_pf1_optimizer(grammar_trainable, GRAMMAR_PROTOCOL),
    }
    schedulers = {
        "matching": PF1LearningRateSchedule(optimizers["matching"], MATCHING_PROTOCOL),
        "grammar": PF1LearningRateSchedule(optimizers["grammar"], GRAMMAR_PROTOCOL),
    }
    trainable_by_family = {
        "matching": matching_trainable,
        "grammar": grammar_trainable,
    }
    loader = build_v3_cache_dataloader(
        cache_root=cache_root,
        tokenizer=tokenizer,
        cell=cell,
        seed=TRAIN_SEED,
        micro_batch_size=GRAMMAR_PROTOCOL.micro_batch_size,
        gradient_accumulation_steps=GRAMMAR_PROTOCOL.gradient_accumulation_steps,
        total_updates=TOTAL_UPDATES,
        num_workers=num_workers,
        prefetch_factor=4,
        shuffle_seed=TRAIN_SEED,
        fixed_view_id=None,
    )
    dev_cache = IndexedPF10TrainingTensorCache(cache_root)
    torch.manual_seed(TRAIN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    evaluations = [{"update": 0, **evaluate_all(
        model, matching_head, cache=dev_cache, tokenizer=tokenizer, cell=cell, device=device
    )}]
    iterator = iter(loader)
    update_counts = {name: 0 for name in BLOCK_CYCLE}
    block_stats = {
        name: {
            "losses": [],
            "learning_rates": [],
            "preclip_norms": [],
            "clipped": 0,
        }
        for name in BLOCK_CYCLE
    }
    members_seen = 0
    matching_anchors = 0
    started = time.perf_counter()
    try:
        for update in range(1, TOTAL_UPDATES + 1):
            block = block_for_update(update)
            family = optimization_family_for_block(block)
            protocol = _protocol_for_family(family)
            optimizer = optimizers[family]
            scheduler = schedulers[family]
            trainable = trainable_by_family[family]
            configure_block(model, matching_head, block)
            batches = [
                next(iterator)
                for _ in range(protocol.gradient_accumulation_steps)
            ]
            expected_view = (
                "m_plus_g"
                if block == "matching_m_plus_g"
                else block[len("grammar_") :]
            )
            if any(batch.view_id != expected_view for batch in batches):
                raise V3BlockJointError("cache view schedule drifted from block cycle")
            optimizer.zero_grad(set_to_none=True)
            update_loss: Tensor | None = None
            if block == "matching_m_plus_g":
                counts = [_eligible_anchor_count(batch.exact_identity_sha256) for batch in batches]
                total = sum(counts)
                if total <= 0:
                    raise V3BlockJointError("matching update has no eligible anchors")
                for cpu_batch, count in zip(batches, counts):
                    batch = cpu_batch.to(device)
                    with _autocast(device):
                        result = forward_matching_only(
                            model, matching_head, batch, component_mode="both"
                        )
                        loss = result.loss * (count / total)
                    weighted_loss = result.loss.detach() * (count / total)
                    update_loss = (
                        weighted_loss
                        if update_loss is None
                        else update_loss + weighted_loss
                    )
                    loss.backward()
                    members_seen += len(batch.record_ids)
                matching_anchors += total
            else:
                counts = [int((batch.labels != -100).sum().item()) for batch in batches]
                total = sum(counts)
                if total <= 0:
                    raise V3BlockJointError("grammar update has no target tokens")
                for cpu_batch, count in zip(batches, counts):
                    batch = cpu_batch.to(device)
                    with _autocast(device):
                        result = _grammar_forward(
                            model,
                            batch,
                            memory_mode="zero" if batch.view_id == "m_only" else "aligned",
                        )
                        loss = result.loss * (count / total)
                    weighted_loss = result.loss.detach() * (count / total)
                    update_loss = (
                        weighted_loss
                        if update_loss is None
                        else update_loss + weighted_loss
                    )
                    loss.backward()
                    members_seen += len(batch.record_ids)
            learning_rate = scheduler.learning_rate_for_next_update()
            preclip = clip_pf1_gradients(trainable, protocol)
            if not math.isfinite(preclip):
                raise V3BlockJointError("block update gradient is non-finite")
            if update_loss is None:
                raise V3BlockJointError("block update omitted loss")
            update_loss_value = float(update_loss.detach().float().cpu().item())
            if not math.isfinite(update_loss_value):
                raise V3BlockJointError("block update loss is non-finite")
            stats = block_stats[block]
            stats["losses"].append(update_loss_value)
            stats["learning_rates"].append(learning_rate)
            stats["preclip_norms"].append(preclip)
            stats["clipped"] += int(preclip > protocol.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            update_counts[block] += 1
            if update in EVALUATION_UPDATES:
                evaluations.append({"update": update, **evaluate_all(
                    model, matching_head, cache=dev_cache, tokenizer=tokenizer,
                    cell=cell, device=device,
                )})
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        loader.dataset.close(); dev_cache.close()

    if schedulers["matching"].completed_updates != MATCHING_PROTOCOL.total_updates:
        raise V3BlockJointError("matching optimizer clock did not complete")
    if schedulers["grammar"].completed_updates != GRAMMAR_PROTOCOL.total_updates:
        raise V3BlockJointError("grammar optimizer clock did not complete")

    wall = time.perf_counter() - started
    block_telemetry = {}
    for block, values in block_stats.items():
        protocol = _protocol_for_family(optimization_family_for_block(block))
        count = update_counts[block]
        block_telemetry[block] = {
            "updates": count,
            "mean_loss": statistics.fmean(values["losses"]),
            "initial_learning_rate": values["learning_rates"][0],
            "final_learning_rate_used": values["learning_rates"][-1],
            "mean_learning_rate": statistics.fmean(values["learning_rates"]),
            "mean_preclip_gradient_norm": statistics.fmean(values["preclip_norms"]),
            "max_preclip_gradient_norm": max(values["preclip_norms"]),
            "clip_rate": values["clipped"] / count,
            "gradient_clip_norm": protocol.gradient_clip_norm,
        }
    checkpoint = output_dir / "model_adapter_and_matching_state.pt"
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "completed_updates": TOTAL_UPDATES,
        "geometry_fraction": GEOMETRY_FRACTION,
        "matching_protocol": asdict(MATCHING_PROTOCOL),
        "grammar_protocol": asdict(GRAMMAR_PROTOCOL),
        "model_state_dict": model.state_dict(),
        "matching_head_state_dict": matching_head.state_dict(),
        "matching_optimizer_state_dict": optimizers["matching"].state_dict(),
        "grammar_optimizer_state_dict": optimizers["grammar"].state_dict(),
        "matching_scheduler_state_dict": schedulers["matching"].state_dict(),
        "grammar_scheduler_state_dict": schedulers["grammar"].state_dict(),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
    }, checkpoint)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "block_coordinate_geometry_preservation_screen",
        "cell": cell,
        "total_global_updates": TOTAL_UPDATES,
        "optimization_protocols": {
            "matching_adapter_and_head": asdict(MATCHING_PROTOCOL),
            "grammar_t5": asdict(GRAMMAR_PROTOCOL),
        },
        "independent_optimizer_clocks": True,
        "block_cycle": list(BLOCK_CYCLE),
        "block_update_counts": update_counts,
        "block_telemetry": block_telemetry,
        "matching_contract": MATCHING_ID,
        "geometry_fraction": GEOMETRY_FRACTION,
        "component_modes": list(COMPONENT_MODES),
        "matching_only_checkpoint": str(s_checkpoint),
        "members_seen": members_seen,
        "matching_eligible_anchors_train": matching_anchors,
        "evaluations": evaluations,
        "wall_seconds": wall,
        "members_per_second": members_seen / wall,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint": str(checkpoint),
    }
    (output_dir / "block_joint_manifest.json").write_text(
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
    parser.add_argument("--s-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise V3BlockJointError("one CUDA BF16 device is required")
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
    return run_block_joint(
        cell=args.cell,
        cache_root=Path(args.cache_root),
        tokenizer=tokenizer.runtime,
        model=model,
        matching_head=_new_matching_head(hidden_size),
        s_checkpoint=Path(args.s_checkpoint),
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
    "BLOCK_CYCLE",
    "EVALUATION_UPDATES",
    "GRAMMAR_PROTOCOL",
    "MATCHING_PROTOCOL",
    "SCHEMA_VERSION",
    "TOTAL_UPDATES",
    "V3BlockJointError",
    "block_for_update",
    "configure_block",
    "load_matching_only_state",
    "optimization_family_for_block",
    "run_block_joint",
    "run_cli",
]
