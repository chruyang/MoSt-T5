#!/usr/bin/env python3
"""Build the raw-T5 initializer for one explicit anchored tokenizer plan."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from most_t5_next.p1.build_anchored_candidate_tokenizer_v1 import (
    load_verified_anchored_candidate_tokenizer,
)
from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    VerifiedUnionInitCheckpoint,
    build_union_init_checkpoint,
    load_verified_union_init_checkpoint,
)


SCHEMA_VERSION = "most-t5-p1/anchored-union-init-bridge/v1"


def anchored_tokenizer_loader(semantic_plan_sha256: str) -> Callable[..., Any]:
    """Bind the shared token surface to exactly one macro-semantic plan."""

    return partial(
        load_verified_anchored_candidate_tokenizer,
        semantic_plan_sha256=semantic_plan_sha256,
    )


def build_anchored_union_init_checkpoint(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    anchored_tokenizer_dir: Path,
    semantic_plan_sha256: str,
    output_dir: Path,
    seed: int,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
) -> VerifiedUnionInitCheckpoint:
    return build_union_init_checkpoint(
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=anchored_tokenizer_dir,
        output_dir=output_dir,
        seed=seed,
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        verified_tokenizer_loader=anchored_tokenizer_loader(semantic_plan_sha256),
    )


def load_verified_anchored_union_init_checkpoint(
    *,
    base_model_snapshot: Path,
    base_tokenizer_snapshot: Path,
    anchored_tokenizer_dir: Path,
    semantic_plan_sha256: str,
    output_dir: Path,
    geometry_fusion_seed: int,
    num_e3fp_embeddings: int,
) -> VerifiedUnionInitCheckpoint:
    return load_verified_union_init_checkpoint(
        base_model_snapshot=base_model_snapshot,
        base_tokenizer_snapshot=base_tokenizer_snapshot,
        union_tokenizer_dir=anchored_tokenizer_dir,
        output_dir=output_dir,
        geometry_fusion_seed=geometry_fusion_seed,
        num_e3fp_embeddings=num_e3fp_embeddings,
        verified_tokenizer_loader=anchored_tokenizer_loader(semantic_plan_sha256),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    built = build_anchored_union_init_checkpoint(
        base_model_snapshot=args.base_model_snapshot,
        base_tokenizer_snapshot=args.base_tokenizer_snapshot,
        anchored_tokenizer_dir=args.anchored_tokenizer_dir,
        semantic_plan_sha256=args.semantic_plan_sha256,
        output_dir=args.output_dir,
        seed=args.seed,
        geometry_fusion_seed=args.geometry_fusion_seed,
        num_e3fp_embeddings=args.num_e3fp_embeddings,
    )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "pass",
                "checkpoint_path": str(built.checkpoint_path),
                "tokenizer_contract_sha256": built.manifest["tokenizer"][
                    "tokenizer_contract_sha256"
                ],
                "tokenizer_snapshot_sha256": built.manifest["tokenizer"][
                    "tokenizer_snapshot_sha256"
                ],
                "base_vocab_size": built.manifest["tokenizer"]["base_vocab_size"],
                "union_vocab_size": built.manifest["tokenizer"]["union_vocab_size"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "anchored_tokenizer_loader",
    "build_anchored_union_init_checkpoint",
    "load_verified_anchored_union_init_checkpoint",
]
