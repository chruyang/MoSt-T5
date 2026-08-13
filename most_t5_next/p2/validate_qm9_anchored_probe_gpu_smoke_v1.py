#!/usr/bin/env python3
"""Run one BF16 forward/backward for each anchored QM9 probe cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from most_t5_next.p1.build_anchored_candidate_tokenizer_v1 import (
    load_verified_anchored_candidate_tokenizer,
)
from most_t5_next.p2.factorized_model_init_v4 import (
    load_deterministic_factorized_model_v4,
)
from most_t5_next.p2.run_qm9_anchored_probe_v1 import (
    ADAPTER_SEED,
    GEOMETRY_FUSION_SEED,
    AnchoredQM9Regressor,
    ProbeCollator,
    QM9ProbeDataset,
    training_target_statistics,
)


SCHEMA_VERSION = "most-t5-p2/qm9-anchored-property-probe-gpu-smoke/v1"
CELL_SPECS = (
    ("B0", "l0_l12_mean", "zero"),
    ("B2D", "l0_l12_mean", "aligned"),
    ("F3D", "l0_l12_mean", "aligned"),
    ("F3D", "l0_l123_mean", "aligned"),
)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one CUDA BF16 device is required")
    tokenizer = load_verified_anchored_candidate_tokenizer(
        base_snapshot=args.base_tokenizer_snapshot,
        output_dir=args.anchored_tokenizer_dir,
        semantic_plan_sha256=args.semantic_plan_sha256,
    )
    dataset = QM9ProbeDataset(args.cache_root, split="train")
    means, stds = training_target_statistics(dataset)
    rows = [dataset[index] for index in range(min(args.records, len(dataset)))]
    device = torch.device("cuda", 0)
    reports = []
    for cell, shell_mode, memory_mode in CELL_SPECS:
        torch.manual_seed(20260810)
        torch.cuda.manual_seed_all(20260810)
        backbone = load_deterministic_factorized_model_v4(
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
            shell_fusion_mode=shell_mode,
        )
        hidden = int(backbone.get_input_embeddings().weight.shape[1])
        model = AnchoredQM9Regressor(backbone, hidden_size=hidden).to(device)
        batch = ProbeCollator(
            pad_token_id=tokenizer.runtime.pad_token_id,
            cell=cell,
        )(rows[: args.micro_batch_size]).to(device)
        target = (batch.targets - means.to(device)) / stds.to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(
                batch.inputs,
                state_memory_mode=memory_mode,
                geometry_component_mode="both",
            )
            loss = (prediction - target).square()[batch.target_mask].mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if (
            not bool(torch.isfinite(loss).item())
            or not gradients
            or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)
        ):
            raise RuntimeError("QM9 probe smoke produced non-finite loss or gradients")
        reports.append({
            "cell": cell,
            "shell_fusion_mode": shell_mode,
            "records": len(batch.record_ids),
            "loss": float(loss.detach().float().item()),
            "gradient_tensors": len(gradients),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        })
        del batch, model, backbone, loss, prediction
        torch.cuda.empty_cache()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cells": reports,
        "scientific_result": False,
    }
    if args.output_report is not None:
        Path(args.output_report).write_text(
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
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--output-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

