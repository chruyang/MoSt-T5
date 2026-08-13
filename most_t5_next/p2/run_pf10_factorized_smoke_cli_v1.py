"""CLI loader for the fixed PF-10 B0/B2D/F3D factorized GPU smoke.

This module only binds published artifacts to the already frozen single-stage
runner.  It exposes no batch-size, update-count, masking or optimizer knobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import PF1PairedReleaseReader
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)
from most_t5_next.p1.build_union_init_checkpoint_v1 import (
    load_verified_union_init_checkpoint,
)

from .build_pf10_morgan_overlay_v1 import MorganAtomStateProvider
from .factorized_model_init_v1 import load_deterministic_factorized_model
from .freeze_pf10_factorized_smoke128_v1 import (
    MANIFEST_NAME as SMOKE_MANIFEST_NAME,
    MEMBERSHIP_NAME as SMOKE_MEMBERSHIP_NAME,
    SCHEMA_VERSION as SMOKE_MEMBERSHIP_SCHEMA,
    SMOKE_COUNT,
)
from .run_pf10_factorized_smoke_v1 import (
    PF10FactorizedSmokeError,
    get_smoke_stage_spec,
    run_pf10_factorized_smoke_stage,
)


ADAPTER_SEED = 20260809
UNION_GEOMETRY_FUSION_SEED = 20260808
NUM_E3FP_EMBEDDINGS = 4096


def _load_smoke_records(
    *,
    reader: PF1PairedReleaseReader,
    smoke_membership: Path,
):
    root = Path(smoke_membership).expanduser().resolve()
    manifest = json.loads((root / SMOKE_MANIFEST_NAME).read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SMOKE_MEMBERSHIP_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("counts", {}).get("smoke_records") != SMOKE_COUNT
    ):
        raise PF10FactorizedSmokeError("smoke membership is not a passed 128 release")
    rows = [
        json.loads(line)
        for line in (root / SMOKE_MEMBERSHIP_NAME).read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != SMOKE_COUNT or [row.get("smoke_index") for row in rows] != list(range(SMOKE_COUNT)):
        raise PF10FactorizedSmokeError("smoke membership order or count changed")
    indices = tuple(int(row["split_index"]) for row in rows)
    loaded = tuple(
        item
        for batch in reader.iter_selected_split_indices(
            split="train",
            split_indices=indices,
            batch_size=SMOKE_COUNT,
        )
        for item in batch
    )
    if len(loaded) != SMOKE_COUNT:
        raise PF10FactorizedSmokeError("smoke reader did not return 128 records")
    for source, item in zip(rows, loaded):
        record = item.motif_record
        if not (
            record.record_id == source["record_id"]
            and record.storage_key == source["storage_key"]
        ):
            raise PF10FactorizedSmokeError("smoke record differs from frozen membership")
    return tuple(item.motif_record for item in loaded)


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PF10FactorizedSmokeError("one CUDA BF16 device is required")
    spec = get_smoke_stage_spec(args.cell, args.stage)
    paired_release = Path(args.paired_release).expanduser().resolve()
    union_tokenizer_dir = paired_release / "union_tokenizer"
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=union_tokenizer_dir,
    )
    reader = PF1PairedReleaseReader(paired_release)
    records = _load_smoke_records(
        reader=reader,
        smoke_membership=Path(args.smoke_membership),
    )

    provider = None
    if args.cell == "B2D":
        provider = MorganAtomStateProvider(Path(args.morgan_overlay))
    try:
        if spec.model_path == "raw_t5":
            verified = load_verified_union_init_checkpoint(
                base_model_snapshot=Path(args.base_model_snapshot),
                base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
                union_tokenizer_dir=union_tokenizer_dir,
                output_dir=Path(args.union_init_dir),
                geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
                num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            )
            model = verified.model
        else:
            model = load_deterministic_factorized_model(
                base_model_snapshot=Path(args.base_model_snapshot),
                base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
                union_tokenizer_dir=union_tokenizer_dir,
                union_init_dir=Path(args.union_init_dir),
                union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
                adapter_seed=ADAPTER_SEED,
                num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
            )
        return run_pf10_factorized_smoke_stage(
            cell=args.cell,
            stage=args.stage,
            records=records,
            tokenizer=tokenizer.runtime,
            model=model,
            output_dir=Path(args.output_dir),
            atom_state_provider=provider,
            s_checkpoint=Path(args.s_checkpoint) if args.s_checkpoint else None,
            device="cuda:0",
            use_bf16=True,
        )
    finally:
        if provider is not None:
            provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B0", "B2D", "F3D"), required=True)
    parser.add_argument("--stage", choices=("S", "G"), required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path, required=True)
    parser.add_argument("--smoke-membership", type=Path, required=True)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--s-checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = _parser().parse_args(argv)
    report = run_cli(args)
    print(json.dumps({"status": report["status"], "cell": report["cell"], "stage": report["stage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ADAPTER_SEED", "NUM_E3FP_EMBEDDINGS", "UNION_GEOMETRY_FUSION_SEED", "run_cli"]
