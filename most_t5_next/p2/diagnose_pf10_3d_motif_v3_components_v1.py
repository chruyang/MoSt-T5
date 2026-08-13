"""Evaluate one completed V3 matching checkpoint by geometry component."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .factorized_model_init_v3 import load_deterministic_factorized_model_v3
from .pf10_training_tensor_cache_v1 import IndexedPF10TrainingTensorCache
from .run_pf10_3d_motif_v3_matching_only_v1 import (
    GEOMETRY_FRACTION,
    _new_matching_head,
    evaluate_components,
)
from .run_pf10_3d_motif_v3_matching_v1 import (
    ADAPTER_SEED,
    NUM_E3FP_EMBEDDINGS,
    UNION_GEOMETRY_FUSION_SEED,
)


SCHEMA_VERSION = "most-t5-p2/pf10-3d-motif-v3-component-diagnostic/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one CUDA BF16 device is required")
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
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
    head = _new_matching_head(hidden_size)
    payload = torch.load(Path(args.checkpoint), map_location="cpu")
    if (
        payload.get("cell") != args.cell
        or payload.get("completed_updates") != 1000
    ):
        raise RuntimeError("checkpoint cell or completed update differs")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    head.load_state_dict(payload["matching_head_state_dict"], strict=True)
    device = torch.device("cuda:0")
    model.to(device)
    head.to(device)
    cache = IndexedPF10TrainingTensorCache(Path(args.cache_root))
    try:
        evaluation = evaluate_components(
            model,
            head,
            cache=cache,
            tokenizer=tokenizer.runtime,
            cell=args.cell,
            device=device,
        )
    finally:
        cache.close()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "cell": args.cell,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_completed_updates": 1000,
        "training_updates_added": 0,
        "evaluation": evaluation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = run_cli(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cell": report["cell"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
