"""Artifact-binding CLI for the one-card PF-10 V2 mechanism smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from most_t5_next.p1.build_pf1_paired_release_v1 import (
    DONOR_ATOM_MAP_NAME,
    PF1PairedReleaseReader,
)
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)

from .build_pf10_morgan_overlay_v1 import MorganAtomStateProvider
from .factorized_model_init_v2 import load_deterministic_factorized_model_v2
from .factorized_view_collator_v2 import (
    GraphPortsCanonicalAtomAddressProvider,
)
from .run_pf10_factorized_smoke_cli_v1 import (
    ADAPTER_SEED,
    NUM_E3FP_EMBEDDINGS,
    UNION_GEOMETRY_FUSION_SEED,
    _load_smoke_records,
)
from .run_pf10_factorized_v2_smoke_v1 import (
    PF10FactorizedV2SmokeError,
    run_pf10_factorized_v2_smoke_stage,
)


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise PF10FactorizedV2SmokeError("one CUDA BF16 device is required")
    paired_release = Path(args.paired_release).expanduser().resolve()
    tokenizer = load_verified_canary_union_tokenizer(
        base_snapshot=Path(args.base_tokenizer_snapshot),
        output_dir=paired_release / "union_tokenizer",
    )
    reader = PF1PairedReleaseReader(paired_release)
    loaded = _load_smoke_records(
        reader=reader,
        smoke_membership=Path(args.smoke_membership),
    )
    records = tuple(loaded)
    addresses = GraphPortsCanonicalAtomAddressProvider(
        paired_release / DONOR_ATOM_MAP_NAME,
        required_record_ids=tuple(record.record_id for record in records),
    )
    provider = None
    if args.cell == "B2D":
        if args.morgan_overlay is None:
            raise PF10FactorizedV2SmokeError("B2D requires --morgan-overlay")
        provider = MorganAtomStateProvider(Path(args.morgan_overlay))
    elif args.morgan_overlay is not None:
        raise PF10FactorizedV2SmokeError("F3D does not accept --morgan-overlay")
    try:
        model = load_deterministic_factorized_model_v2(
            base_model_snapshot=Path(args.base_model_snapshot),
            base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
            union_tokenizer_dir=paired_release / "union_tokenizer",
            union_init_dir=Path(args.union_init_dir),
            union_geometry_fusion_seed=UNION_GEOMETRY_FUSION_SEED,
            adapter_seed=ADAPTER_SEED,
            num_e3fp_embeddings=NUM_E3FP_EMBEDDINGS,
        )
        report = run_pf10_factorized_v2_smoke_stage(
            cell=args.cell,
            stage=args.stage,
            records=records,
            tokenizer=tokenizer.runtime,
            model=model,
            atom_address_provider=addresses,
            atom_state_provider=provider,
            output_dir=Path(args.output_dir),
            s_checkpoint=Path(args.s_checkpoint) if args.s_checkpoint else None,
            device=torch.device("cuda:0"),
            use_bf16=True,
        )
        report["artifact_binding"] = {
            "paired_release": str(paired_release),
            "smoke_membership": str(Path(args.smoke_membership).resolve()),
            "canonical_address_records": addresses.record_count,
            "base_model_snapshot": str(Path(args.base_model_snapshot).resolve()),
            "union_init_dir": str(Path(args.union_init_dir).resolve()),
        }
        (Path(args.output_dir) / "smoke_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if provider is not None:
            provider.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("B2D", "F3D"), required=True)
    parser.add_argument("--stage", choices=("S", "B"), required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--smoke-membership", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--s-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = _parser().parse_args(argv)
    if args.stage == "B" and args.s_checkpoint is None:
        raise SystemExit("V2 bridge stage requires --s-checkpoint")
    if args.stage == "S" and args.s_checkpoint is not None:
        raise SystemExit("V2 S stage does not accept --s-checkpoint")
    report = run_cli(args)
    print(json.dumps({"status": report["status"], "cell": args.cell, "stage": args.stage}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
