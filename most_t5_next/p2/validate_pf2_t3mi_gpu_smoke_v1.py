#!/usr/bin/env python3
"""Validate the real run3 T3MI boundary with one discarded GPU update."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Sequence

from most_t5_next.p2.run_pf2_t3mi_v1 import (
    T3MI_MASK_PROBABILITY,
    T3MI_OBJECTIVE_CONTRACT,
    T3MI_PROTOCOL,
)
from most_t5_next.p2.validate_pf2_gated_fusion_gpu_smoke_v1 import (
    FGateSmokeError,
    run_smoke as run_gated_smoke,
)


REPORT_SCHEMA = "most-t5-p2/t3mi-gpu-smoke/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--base-model-snapshot", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--union-init-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-fusion-seed", type=int, required=True)
    parser.add_argument("--num-e3fp-embeddings", type=int, default=4096)
    return parser


def run_smoke(
    args: argparse.Namespace,
    *,
    runner: Callable[..., dict[str, object]] = run_gated_smoke,
) -> dict[str, object]:
    return runner(
        args,
        mask_probability=T3MI_MASK_PROBABILITY,
        protocol=T3MI_PROTOCOL,
        report_schema=REPORT_SCHEMA,
        scope="real_run3_t3mi_all_identity_one_discarded_optimizer_step",
        objective_contract=T3MI_OBJECTIVE_CONTRACT,
        require_all_motif_identities_masked=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_smoke(args)
    except (FGateSmokeError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "build_parser", "main", "run_smoke"]
