"""Dedicated, non-adjudicating PF-1 GPU phase profiler.

This diagnostic loads a fresh verified condition, performs a short discarded
optimization trajectory, and records data wait, tensor materialization,
forward, loss synchronization, backward, clipping, and optimizer timings.  It
never writes a checkpoint and its measurements must not be mixed into the
GraphPorts codec quality gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Mapping, Sequence

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    PF1PairedReleaseReader,
    TOKENIZER_DIRECTORY,
)
from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_four_grid_wrapper,
)
from most_t5_next.p1.pf1_optimization import (
    G_CODEC_PF1_PROTOCOL,
    G_CODEC_PROTOCOL_ID,
    PF1LearningRateSchedule,
    build_pf1_optimizer,
    clip_pf1_gradients,
)
from most_t5_next.p1 import run_pf1_four_grid_v1 as training
from most_t5_next.p1.training_adapter import (
    select_four_grid_forward_inputs,
    to_four_grid_batch_encoding,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p1/pf1-gpu-pipeline-profile/v1"
DEFAULT_WARMUP_UPDATES = 3
DEFAULT_PROFILE_UPDATES = 10


class PF1GPUPipelineProfileError(RuntimeError):
    """Raised when the diagnostic cannot produce a valid phase profile."""


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise PF1GPUPipelineProfileError("timing population is empty")
    index = int(round((len(sorted_values) - 1) * probability))
    return float(sorted_values[index])


def _distribution(values: Sequence[float]) -> dict[str, int | float]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise PF1GPUPipelineProfileError("timings must be finite and non-negative")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _event_pair(torch_module: Any) -> tuple[Any, Any]:
    return (
        torch_module.cuda.Event(enable_timing=True),
        torch_module.cuda.Event(enable_timing=True),
    )


def _elapsed_ms(pairs: Sequence[tuple[Any, Any]]) -> float:
    return sum(float(start.elapsed_time(end)) for start, end in pairs)


def profile_pipeline(
    *,
    reader: PF1PairedReleaseReader,
    tokenizer_runtime: Any,
    model: Any,
    condition_id: str,
    device: Any,
    torch_module: Any,
    use_bf16: bool,
    warmup_updates: int,
    profile_updates: int,
) -> dict[str, Any]:
    if condition_id not in training.CONDITION_ORDER:
        raise PF1GPUPipelineProfileError("condition id is invalid")
    if warmup_updates < 0 or profile_updates <= 0:
        raise PF1GPUPipelineProfileError("profile update counts are invalid")
    if device.type != "cuda":
        raise PF1GPUPipelineProfileError("the phase profiler requires CUDA")
    protocol = G_CODEC_PF1_PROTOCOL
    model.to(device)
    model.train()
    optimizer = build_pf1_optimizer(model, protocol)
    scheduler = PF1LearningRateSchedule(optimizer, protocol)
    cursor = training._TrainCursor(reader, protocol.micro_batch_size)
    data_lock = threading.Lock()
    total_updates = warmup_updates + profile_updates
    phase_values: dict[str, list[float]] = {
        "update_wall_seconds": [],
        "prepared_data_wait_wall_seconds": [],
        "tensor_adapter_wall_seconds": [],
        "forward_call_wall_seconds": [],
        "finite_loss_sync_wall_seconds": [],
        "backward_call_wall_seconds": [],
        "clip_wall_seconds": [],
        "optimizer_wall_seconds": [],
        "adapter_h2d_cuda_event_ms": [],
        "forward_cuda_event_ms": [],
        "backward_cuda_event_ms": [],
        "optimizer_cuda_event_ms": [],
    }
    members_per_update: list[int] = []
    encoder_tokens_per_update: list[int] = []
    target_tokens_per_update: list[int] = []
    preclip_norms: list[float] = []
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(device)
    torch_module.manual_seed(training.FORWARD_SEED)
    torch_module.cuda.manual_seed_all(training.FORWARD_SEED)

    with training._OrderedTrainPrefetch(
        cursor=cursor,
        total_updates=total_updates,
        depth=training.TRAIN_PREFETCH_DEPTH,
        gradient_accumulation_steps=protocol.gradient_accumulation_steps,
        condition_id=condition_id,
        tokenizer_runtime=tokenizer_runtime,
        data_lock=data_lock,
    ) as stream:
        iterator = iter(stream)
        for zero_based_update in range(total_updates):
            update_started = time.perf_counter()
            data_wait_started = time.perf_counter()
            prepared = next(iterator)
            data_wait = time.perf_counter() - data_wait_started
            batches = prepared.batches
            update_target_tokens = sum(
                sum(batch.ce_batch.target_lengths) for batch in batches
            )
            if update_target_tokens <= 0:
                raise PF1GPUPipelineProfileError("profile update has no targets")
            optimizer.zero_grad(set_to_none=True)
            adapter_wall = 0.0
            forward_wall = 0.0
            loss_sync_wall = 0.0
            backward_wall = 0.0
            adapter_events: list[tuple[Any, Any]] = []
            forward_events: list[tuple[Any, Any]] = []
            backward_events: list[tuple[Any, Any]] = []
            update_members = 0
            update_encoder_tokens = 0

            for batch in batches:
                batch_target_tokens = sum(batch.ce_batch.target_lengths)
                adapter_event = _event_pair(torch_module)
                adapter_event[0].record()
                phase_started = time.perf_counter()
                encoded = to_four_grid_batch_encoding(batch, device=device)
                forward_inputs = select_four_grid_forward_inputs(encoded)
                adapter_wall += time.perf_counter() - phase_started
                adapter_event[1].record()
                adapter_events.append(adapter_event)

                forward_event = _event_pair(torch_module)
                forward_event[0].record()
                phase_started = time.perf_counter()
                with training._autocast(torch_module, use_bf16):
                    outputs = model(
                        **forward_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                    loss = outputs.loss
                forward_wall += time.perf_counter() - phase_started
                forward_event[1].record()
                forward_events.append(forward_event)

                phase_started = time.perf_counter()
                training._finite_loss(torch_module, loss, condition_id)
                loss_sync_wall += time.perf_counter() - phase_started

                backward_event = _event_pair(torch_module)
                backward_event[0].record()
                phase_started = time.perf_counter()
                (loss * (batch_target_tokens / update_target_tokens)).backward()
                backward_wall += time.perf_counter() - phase_started
                backward_event[1].record()
                backward_events.append(backward_event)
                update_members += len(batch.ce_batch.record_ids)
                update_encoder_tokens += sum(batch.ce_batch.input_lengths)

            clip_started = time.perf_counter()
            preclip_norm = clip_pf1_gradients(model, protocol)
            clip_wall = time.perf_counter() - clip_started
            if not math.isfinite(preclip_norm):
                raise PF1GPUPipelineProfileError("profile gradient norm is non-finite")
            optimizer_event = _event_pair(torch_module)
            optimizer_event[0].record()
            optimizer_started = time.perf_counter()
            optimizer.step()
            optimizer_wall = time.perf_counter() - optimizer_started
            optimizer_event[1].record()
            scheduler.step()
            torch_module.cuda.synchronize(device)
            update_wall = time.perf_counter() - update_started

            if zero_based_update >= warmup_updates:
                phase_values["update_wall_seconds"].append(update_wall)
                phase_values["prepared_data_wait_wall_seconds"].append(data_wait)
                phase_values["tensor_adapter_wall_seconds"].append(adapter_wall)
                phase_values["forward_call_wall_seconds"].append(forward_wall)
                phase_values["finite_loss_sync_wall_seconds"].append(loss_sync_wall)
                phase_values["backward_call_wall_seconds"].append(backward_wall)
                phase_values["clip_wall_seconds"].append(clip_wall)
                phase_values["optimizer_wall_seconds"].append(optimizer_wall)
                phase_values["adapter_h2d_cuda_event_ms"].append(
                    _elapsed_ms(adapter_events)
                )
                phase_values["forward_cuda_event_ms"].append(
                    _elapsed_ms(forward_events)
                )
                phase_values["backward_cuda_event_ms"].append(
                    _elapsed_ms(backward_events)
                )
                phase_values["optimizer_cuda_event_ms"].append(
                    float(optimizer_event[0].elapsed_time(optimizer_event[1]))
                )
                members_per_update.append(update_members)
                encoder_tokens_per_update.append(update_encoder_tokens)
                target_tokens_per_update.append(update_target_tokens)
                preclip_norms.append(preclip_norm)

    phase_distributions = {
        name: _distribution(values) for name, values in phase_values.items()
    }
    member_distribution = _distribution(members_per_update)
    encoder_token_distribution = _distribution(encoder_tokens_per_update)
    target_token_distribution = _distribution(target_tokens_per_update)
    mean_update = phase_distributions["update_wall_seconds"]["mean"]
    assert isinstance(mean_update, float)
    wall_phase_fraction_of_update = {
        name: distribution["mean"] / mean_update
        for name, distribution in phase_distributions.items()
        if name.endswith("_wall_seconds") and name != "update_wall_seconds"
    }
    return {
        "condition": condition_id,
        "warmup_updates": warmup_updates,
        "profile_updates": profile_updates,
        "discarded_optimization_trajectory": True,
        "checkpoint_written": False,
        "phase_distributions": phase_distributions,
        "wall_phase_fraction_of_update": wall_phase_fraction_of_update,
        "members_per_update": member_distribution,
        "encoder_tokens_per_update": encoder_token_distribution,
        "target_tokens_per_update": target_token_distribution,
        "throughput": {
            "members_per_second": member_distribution["mean"] / mean_update,
            "encoder_tokens_per_second": encoder_token_distribution["mean"]
            / mean_update,
            "target_tokens_per_second": target_token_distribution["mean"]
            / mean_update,
        },
        "preclip_gradient_norm": _distribution(preclip_norms),
        "peak_gpu_memory_bytes": int(torch_module.cuda.max_memory_allocated(device)),
        "peak_gpu_reserved_bytes": int(torch_module.cuda.max_memory_reserved(device)),
        "timing_semantics": {
            "wall_and_cuda_event_phases_are_not_additive": True,
            "loss_item_sync_is_reported_separately": True,
            "clip_wall_may_wait_for_queued_backward": True,
            "optimizer_wall_includes_reference_adamwscale_host_sync": True,
            "profiling_overhead_means_not_for_throughput_claims": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    parser.add_argument("--condition-id", choices=training.CONDITION_ORDER, default="M0")
    parser.add_argument("--warmup-updates", type=int, default=DEFAULT_WARMUP_UPDATES)
    parser.add_argument("--profile-updates", type=int, default=DEFAULT_PROFILE_UPDATES)
    return parser


def run(args: argparse.Namespace, *, torch_module: Any | None = None) -> dict[str, Any]:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:  # pragma: no cover - runtime boundary
            raise PF1GPUPipelineProfileError("PyTorch is required") from exc
    if not torch_module.cuda.is_available() or not torch_module.cuda.is_bf16_supported():
        raise PF1GPUPipelineProfileError("one BF16 CUDA GPU is required")
    paired_release = Path(args.paired_release).expanduser().resolve()
    base_model_snapshot = Path(args.base_model_snapshot).expanduser().resolve()
    base_tokenizer_snapshot = Path(args.base_tokenizer_snapshot).expanduser().resolve()
    union_init_dir = Path(args.union_init_dir).expanduser().resolve()
    output_report = Path(args.output_report).expanduser().resolve()
    if output_report.exists():
        raise PF1GPUPipelineProfileError("output report must be a new path")
    tokenizer_build = load_verified_canary_union_tokenizer(
        base_snapshot=base_tokenizer_snapshot,
        output_dir=paired_release / TOKENIZER_DIRECTORY,
    )
    reader = PF1PairedReleaseReader(paired_release)
    reader.enable_decoded_record_cache()
    cache_warmup = reader.warm_decoded_record_cache(
        workers=training.TRAIN_DECODE_CACHE_WORKERS,
        max_pending=training.TRAIN_DECODE_CACHE_MAX_PENDING,
    )
    model = load_verified_four_grid_wrapper(
        condition_id=args.condition_id,
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=paired_release / TOKENIZER_DIRECTORY,
        output_dir=union_init_dir,
        geometry_fusion_seed=args.geometry_fusion_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
    )
    if int(model.config.vocab_size) != tokenizer_build.runtime.vocab_size:
        raise PF1GPUPipelineProfileError("wrapper and tokenizer vocabularies differ")
    try:
        result = profile_pipeline(
            reader=reader,
            tokenizer_runtime=tokenizer_build.runtime,
            model=model,
            condition_id=args.condition_id,
            device=torch_module.device("cuda", 0),
            torch_module=torch_module,
            use_bf16=True,
            warmup_updates=args.warmup_updates,
            profile_updates=args.profile_updates,
        )
    finally:
        model.zero_grad(set_to_none=True)
        del model
        torch_module.cuda.empty_cache()
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "dedicated_performance_diagnostic_only",
        "training_admission": False,
        "scientific_gate_input": False,
        "paired_release": str(paired_release),
        "runtime": {
            "torch": str(torch_module.__version__),
            "cuda_runtime": str(torch_module.version.cuda),
            "cuda_device": str(torch_module.cuda.get_device_name(0)),
            "cuda_total_memory_bytes": int(
                torch_module.cuda.get_device_properties(0).total_memory
            ),
            "bf16": True,
        },
        "optimization_protocol_id": G_CODEC_PROTOCOL_ID,
        "optimization_protocol": asdict(G_CODEC_PF1_PROTOCOL),
        "validated_cache_warmup": cache_warmup,
        "profile": result,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report["profile"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROFILE_UPDATES",
    "DEFAULT_WARMUP_UPDATES",
    "PF1GPUPipelineProfileError",
    "REPORT_SCHEMA",
    "build_parser",
    "main",
    "profile_pipeline",
    "run",
]
