"""Frozen-T5 learnability gate for the V3 motif geometry route.

This runner deliberately excludes grammar CE.  It asks whether the atom-state
adapter and disposable matching head can beat the exact per-candidate uniform
baseline while gradients pass through a frozen, evaluation-mode T5 encoder.
The same checkpoint is evaluated with both, carrier-only, endpoint-only, and
zero geometry injection, so component attribution does not require retraining.
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
from .run_pf10_3d_motif_v3_matching_v1 import (
    ADAPTER_SEED,
    DEV_SEED,
    MATCHING_HEAD_SEED,
    NUM_E3FP_EMBEDDINGS,
    TRAIN_SEED,
    UNION_GEOMETRY_FUSION_SEED,
    _eligible_anchor_count,
)


SCHEMA_VERSION = "most-t5-p2/pf10-3d-motif-v3-matching-only/v1"
COMPONENT_MODES = ("both", "carrier_only", "endpoint_only", "zero")
GEOMETRY_FRACTION = 0.15
EVALUATION_UPDATES = (0, 250, 500, 1000)
DEV_RECORD_LIMIT = 512
PROTOCOL = PF1OptimizationProtocol(
    base_learning_rate=3.0e-4,
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


class V3MatchingOnlyError(RuntimeError):
    """The frozen-backbone matching gate violated its compact contract."""


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def freeze_t5_for_matching_only(model: nn.Module) -> nn.Module:
    t5 = getattr(model, "t5", None)
    adapter = getattr(model, "adapter", None)
    if not isinstance(t5, nn.Module) or not isinstance(adapter, nn.Module):
        raise V3MatchingOnlyError("matching-only model lacks T5 or V3 adapter")
    for parameter in t5.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    t5.eval()
    adapter.train()
    return adapter


def _tensor(inputs: Mapping[str, object], name: str) -> Tensor:
    value = inputs.get(name)
    if not isinstance(value, Tensor):
        raise V3MatchingOnlyError(f"cached V3 batch lacks tensor {name}")
    return value


def forward_matching_only(
    model: nn.Module,
    head: MotifStateMatchingHeadV3,
    batch: CachedV3Batch,
    *,
    component_mode: str,
) -> Any:
    if component_mode not in COMPONENT_MODES:
        raise V3MatchingOnlyError("unknown component ablation mode")
    t5 = getattr(model, "t5", None)
    adapter = getattr(model, "adapter", None)
    if not isinstance(t5, nn.Module) or not isinstance(adapter, nn.Module):
        raise V3MatchingOnlyError("matching-only model lacks T5 or adapter")
    inputs = batch.inputs
    input_ids = _tensor(inputs, "input_ids")
    attention_mask = _tensor(inputs, "attention_mask")
    embeddings = t5.get_input_embeddings()(input_ids)
    encoded = adapter.encode(
        embeddings,
        attention_mask=attention_mask.to(torch.bool),
        e3fp_input_ids=_tensor(inputs, "e3fp_input_ids"),
        atom_mask=_tensor(inputs, "atom_mask"),
        atom_to_motif=_tensor(inputs, "atom_to_motif"),
        motif_mask=_tensor(inputs, "motif_mask"),
        motif_to_carrier=_tensor(inputs, "motif_to_carrier"),
        identity_span_bounds=_tensor(inputs, "identity_span_bounds"),
        endpoint_token_to_atom=_tensor(inputs, "endpoint_token_to_atom"),
        atom_is_attachment=_tensor(inputs, "atom_is_attachment"),
        state_memory_mode="aligned",
        geometry_component_mode=component_mode,
    )
    encoder = getattr(t5, "encoder", None)
    if not callable(encoder):
        raise V3MatchingOnlyError("T5 encoder is unavailable")
    encoder_output = encoder(
        inputs_embeds=encoded.fused_embeddings,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = getattr(encoder_output, "last_hidden_state", None)
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise V3MatchingOnlyError("T5 encoder omitted hidden states")
    return head(
        encoder_hidden=hidden,
        motif_state=encoded.pre_t5_motif_context,
        motif_mask=_tensor(inputs, "motif_mask"),
        motif_to_carrier=_tensor(inputs, "motif_to_carrier"),
        exact_identity_sha256=batch.exact_identity_sha256,
    )


def _summarize_matching(outputs: Sequence[Any]) -> dict[str, object]:
    anchors = sum(int(output.eligible_anchors) for output in outputs)
    if anchors == 0:
        raise V3MatchingOnlyError("matching evaluation has no eligible anchors")
    loss_sum = sum(
        float(output.loss.detach().float().cpu().item()) * output.eligible_anchors
        for output in outputs
    )
    uniform_sum = sum(float(output.uniform_loss_sum) for output in outputs)
    return {
        "eligible_anchors": anchors,
        "matching_loss": loss_sum / anchors,
        "uniform_loss": uniform_sum / anchors,
        "excess_loss_over_uniform": (loss_sum - uniform_sum) / anchors,
        "tie_aware_top1_accuracy": sum(
            float(output.tie_aware_top1_credit_sum) for output in outputs
        )
        / anchors,
        "mean_positive_probability": sum(
            float(output.positive_probability_sum) for output in outputs
        )
        / anchors,
    }


def evaluate_components(
    model: nn.Module,
    head: MotifStateMatchingHeadV3,
    *,
    cache: IndexedPF10TrainingTensorCache,
    tokenizer: Any,
    cell: str,
    device: torch.device,
    max_records: int = DEV_RECORD_LIMIT,
) -> dict[str, object]:
    model.t5.eval()
    model.adapter.eval()
    head.eval()
    indices = cache.split_indices("dev")[:max_records]
    if not indices:
        raise V3MatchingOnlyError("cache dev split is empty")
    collator = CachedV3Collator(
        cache=cache,
        tokenizer=tokenizer,
        cell=cell,
        seed=DEV_SEED,
    )
    by_mode: dict[str, list[Any]] = {mode: [] for mode in COMPONENT_MODES}
    with torch.no_grad():
        for start in range(0, len(indices), PROTOCOL.micro_batch_size):
            samples = tuple(
                CacheTrainingSample(cache[index], 0, "m_plus_g")
                for index in indices[start : start + PROTOCOL.micro_batch_size]
            )
            batch = collator(samples).to(device)
            for mode in COMPONENT_MODES:
                with _autocast(device):
                    by_mode[mode].append(
                        forward_matching_only(
                            model,
                            head,
                            batch,
                            component_mode=mode,
                        )
                    )
    return {
        "dev_records": len(indices),
        "components": {
            mode: _summarize_matching(outputs)
            for mode, outputs in by_mode.items()
        },
    }


def run_matching_only(
    *,
    cell: str,
    cache_root: Path,
    tokenizer: Any,
    model: nn.Module,
    matching_head: MotifStateMatchingHeadV3,
    output_dir: Path,
    device: torch.device,
    num_workers: int = 8,
) -> dict[str, object]:
    if cell not in {"B2D", "F3D"}:
        raise V3MatchingOnlyError("matching-only cell must be B2D or F3D")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise V3MatchingOnlyError("matching-only output already exists")
    output_dir.mkdir(parents=True)
    model.to(device)
    adapter = freeze_t5_for_matching_only(model)
    matching_head.to(device)
    trainable = nn.ModuleDict({"adapter": adapter, "matching_head": matching_head})
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
            model,
            matching_head,
            cache=dev_cache,
            tokenizer=tokenizer,
            cell=cell,
            device=device,
        ),
    }]
    iterator = iter(loader)
    preclip_norms: list[float] = []
    clipped = 0
    anchors_seen = 0
    members_seen = 0
    started = time.perf_counter()
    try:
        for update in range(1, PROTOCOL.total_updates + 1):
            model.t5.eval()
            model.adapter.train()
            matching_head.train()
            batches = [next(iterator) for _ in range(PROTOCOL.gradient_accumulation_steps)]
            anchor_counts = [
                _eligible_anchor_count(batch.exact_identity_sha256)
                for batch in batches
            ]
            total_anchors = sum(anchor_counts)
            if total_anchors <= 0:
                raise V3MatchingOnlyError("one update has no matching anchors")
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch, anchor_count in zip(batches, anchor_counts):
                batch = cpu_batch.to(device)
                with _autocast(device):
                    matched = forward_matching_only(
                        model,
                        matching_head,
                        batch,
                        component_mode="both",
                    )
                    if matched.eligible_anchors != anchor_count:
                        raise V3MatchingOnlyError("matching coverage changed in forward")
                    loss = matched.loss * (anchor_count / total_anchors)
                loss.backward()
                members_seen += len(batch.record_ids)
            preclip = clip_pf1_gradients(trainable, PROTOCOL)
            if not math.isfinite(preclip):
                raise V3MatchingOnlyError("matching-only gradient is non-finite")
            preclip_norms.append(preclip)
            clipped += int(preclip > PROTOCOL.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            anchors_seen += total_anchors
            if update in EVALUATION_UPDATES:
                evaluations.append({
                    "update": update,
                    **evaluate_components(
                        model,
                        matching_head,
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
        loader.dataset.close()
        dev_cache.close()

    wall = time.perf_counter() - started
    checkpoint = output_dir / "adapter_and_matching_state.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "cell": cell,
            "completed_updates": PROTOCOL.total_updates,
            "geometry_fraction": GEOMETRY_FRACTION,
            "adapter_state_dict": model.adapter.state_dict(),
            "matching_head_state_dict": matching_head.state_dict(),
        },
        checkpoint,
    )
    final_both = evaluations[-1]["components"]["both"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cell": cell,
        "state_kind": (
            "most-t5-p2/coordinate-blind-morgan-atom-state/r3-fp4096-v1"
            if cell == "B2D"
            else "e3fp"
        ),
        "scope": "frozen_t5_geometry_route_learnability_gate",
        "protocol": asdict(PROTOCOL),
        "matching_contract": MATCHING_ID,
        "geometry_fraction": GEOMETRY_FRACTION,
        "component_modes": list(COMPONENT_MODES),
        "t5_frozen_and_eval": True,
        "grammar_ce_used": False,
        "members_seen": members_seen,
        "eligible_anchors_seen": anchors_seen,
        "evaluations": evaluations,
        "mean_preclip_gradient_norm": statistics.fmean(preclip_norms),
        "max_preclip_gradient_norm": max(preclip_norms),
        "clip_rate": clipped / PROTOCOL.total_updates,
        "wall_seconds": wall,
        "members_per_second": members_seen / wall,
        "checkpoint": str(checkpoint),
        "gate": {
            "both_beats_uniform": float(final_both["excess_loss_over_uniform"]) < 0.0,
        },
    }
    (output_dir / "matching_only_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _new_matching_head(hidden_size: int) -> MotifStateMatchingHeadV3:
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(MATCHING_HEAD_SEED)
        return MotifStateMatchingHeadV3(
            hidden_size=hidden_size,
            projection_dim=128,
            temperature=0.1,
        )
    finally:
        torch.random.set_rng_state(state)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise V3MatchingOnlyError("one CUDA BF16 device is required")
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
    return run_matching_only(
        cell=args.cell,
        cache_root=Path(args.cache_root),
        tokenizer=tokenizer.runtime,
        model=model,
        matching_head=_new_matching_head(hidden_size),
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
    "COMPONENT_MODES",
    "GEOMETRY_FRACTION",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "V3MatchingOnlyError",
    "evaluate_components",
    "forward_matching_only",
    "freeze_t5_for_matching_only",
    "run_cli",
    "run_matching_only",
]
