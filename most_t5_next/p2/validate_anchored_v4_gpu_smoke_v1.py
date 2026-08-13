#!/usr/bin/env python3
"""Run one 128-record BF16 forward/backward smoke for every V4 shell mode."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Sequence

import torch

from most_t5_next.p1.build_anchored_candidate_tokenizer_v1 import (
    load_verified_anchored_candidate_tokenizer,
)

from .factorized_model_init_v4 import (
    factorized_initialization_contract_v4,
    load_deterministic_factorized_model_v4,
)
from .motif_geometry_adapter_v4 import SHELL_FUSION_MODES
from .pf10_training_tensor_cache_v1 import build_v3_cache_dataloader


SCHEMA_VERSION = "most-t5-p2/anchored-v4-gpu-smoke/v1"


class AnchoredV4GpuSmokeError(RuntimeError):
    """The minimal real-device V4 execution contract failed."""


def _gradient_summary(model: torch.nn.Module) -> dict[str, object]:
    total_sq = 0.0
    adapter_sq = 0.0
    t5_sq = 0.0
    tensor_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            raise AnchoredV4GpuSmokeError(f"non-finite gradient: {name}")
        value = float(gradient.detach().float().norm().item())
        total_sq += value * value
        if name.startswith("adapter."):
            adapter_sq += value * value
        elif name.startswith("t5."):
            t5_sq += value * value
        tensor_count += 1
    result = {
        "all_l2": math.sqrt(total_sq),
        "adapter_l2": math.sqrt(adapter_sq),
        "t5_l2": math.sqrt(t5_sq),
        "gradient_tensor_count": tensor_count,
    }
    if result["all_l2"] <= 0.0 or result["adapter_l2"] <= 0.0 or result["t5_l2"] <= 0.0:
        raise AnchoredV4GpuSmokeError("T5 and adapter must both receive nonzero gradients")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise AnchoredV4GpuSmokeError("CUDA is required")
    if torch.cuda.device_count() < 1:
        raise AnchoredV4GpuSmokeError("cuda:0 is absent")
    if args.records != args.micro_batch_size * args.gradient_accumulation_steps:
        raise AnchoredV4GpuSmokeError(
            "records must equal micro_batch_size * gradient_accumulation_steps"
        )
    report_path = Path(args.output_report).expanduser().resolve()
    if report_path.exists():
        raise AnchoredV4GpuSmokeError("output report must be a new path")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    verified_tokenizer = load_verified_anchored_candidate_tokenizer(
        base_snapshot=args.base_tokenizer_snapshot,
        output_dir=args.anchored_tokenizer_dir,
        semantic_plan_sha256=args.semantic_plan_sha256,
    )
    device = torch.device("cuda", 0)
    results = []
    reference_record_ids = None
    for mode in SHELL_FUSION_MODES:
        model = load_deterministic_factorized_model_v4(
            base_model_snapshot=args.base_model_snapshot,
            base_tokenizer_snapshot=args.base_tokenizer_snapshot,
            anchored_tokenizer_dir=args.anchored_tokenizer_dir,
            semantic_plan_sha256=args.semantic_plan_sha256,
            union_init_dir=args.union_init_dir,
            union_geometry_fusion_seed=args.geometry_fusion_seed,
            adapter_seed=args.adapter_seed,
            num_e3fp_embeddings=args.num_e3fp_embeddings,
            state_level2_weight=args.state_level2_weight,
            state_embedding_dim=args.state_embedding_dim,
            atom_memory_dim=args.atom_memory_dim,
            max_identity_span_length=args.max_identity_span_length,
            max_atoms_per_motif=args.max_atoms_per_motif,
            geometry_fraction=args.geometry_fraction,
            shell_fusion_mode=mode,
        )
        model.to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        loader = build_v3_cache_dataloader(
            cache_root=args.cache_root,
            tokenizer=verified_tokenizer.runtime,
            cell="F3D",
            seed=args.data_seed,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_updates=1,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            fixed_view_id="m_plus_g",
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        record_ids = []
        losses = []
        endpoint_count = 0
        input_tokens = 0
        target_tokens = 0
        for batch in loader:
            record_ids.extend(batch.record_ids)
            cuda_batch = batch.to(device)
            endpoint_count += int(
                (cuda_batch.inputs["endpoint_token_to_atom"] >= 0).sum().item()
            )
            input_tokens += int(cuda_batch.inputs["attention_mask"].sum().item())
            target_tokens += int((cuda_batch.inputs["labels"] != -100).sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**cuda_batch.inputs)
                loss = output.loss / args.gradient_accumulation_steps
            if not bool(torch.isfinite(loss).item()):
                raise AnchoredV4GpuSmokeError(f"non-finite loss for {mode}")
            loss.backward()
            losses.append(float(loss.detach().item()))
        torch.cuda.synchronize(device)
        if len(record_ids) != args.records or len(set(record_ids)) != args.records:
            raise AnchoredV4GpuSmokeError("smoke did not consume exactly 128 unique records")
        if reference_record_ids is None:
            reference_record_ids = tuple(record_ids)
        elif tuple(record_ids) != reference_record_ids:
            raise AnchoredV4GpuSmokeError("shell candidates consumed different records")
        gradients = _gradient_summary(model)
        attention_gradient = model.adapter.shell_attention_score.weight.grad
        if mode == "l0_shell_attention_l123":
            if attention_gradient is None or float(attention_gradient.float().norm().item()) <= 0.0:
                raise AnchoredV4GpuSmokeError("shell attention did not receive gradient")
        elif attention_gradient is not None:
            raise AnchoredV4GpuSmokeError("inactive shell attention unexpectedly received gradient")
        results.append(
            {
                "shell_fusion_mode": mode,
                "records": len(record_ids),
                "microbatches": len(losses),
                "loss": sum(losses),
                "input_tokens": input_tokens,
                "target_tokens": target_tokens,
                "endpoint_tokens": endpoint_count,
                "wall_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "gradients": gradients,
                "shell_attention_active": attention_gradient is not None,
            }
        )
        del loader, model
        gc.collect()
        torch.cuda.empty_cache()

    contract = factorized_initialization_contract_v4(
        semantic_plan_sha256=args.semantic_plan_sha256,
        adapter_seed=args.adapter_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
        state_level2_weight=args.state_level2_weight,
        state_embedding_dim=args.state_embedding_dim,
        atom_memory_dim=args.atom_memory_dim,
        max_identity_span_length=args.max_identity_span_length,
        max_atoms_per_motif=args.max_atoms_per_motif,
        geometry_fraction=args.geometry_fraction,
        shell_fusion_mode="l0_shell_attention_l123",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "forward_backward_runtime_smoke_not_architecture_selection",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "bf16": True,
        "optimizer_step": False,
        "records": args.records,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "tokenizer_contract_sha256": verified_tokenizer.runtime.tokenizer_contract_sha256,
        "tokenizer_snapshot_sha256": verified_tokenizer.runtime.tokenizer_snapshot_sha256,
        "factorized_contract_common": {
            key: value for key, value in contract.items() if key != "shell_fusion_mode"
        },
        "results": results,
        "scientific_claims": {
            "runtime_boundary_verified": True,
            "shell_candidate_quality_inferred": False,
            "three_dimensional_gain_inferred": False,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=20260807)
    parser.add_argument("--geometry-fusion-seed", type=int, default=20260808)
    parser.add_argument("--adapter-seed", type=int, default=20260809)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    parser.add_argument("--state-level2-weight", type=float, default=0.25)
    parser.add_argument("--state-embedding-dim", type=int, default=64)
    parser.add_argument("--atom-memory-dim", type=int, default=128)
    parser.add_argument("--max-identity-span-length", type=int, default=128)
    parser.add_argument("--max-atoms-per-motif", type=int, default=128)
    parser.add_argument("--geometry-fraction", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "results": report["results"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
