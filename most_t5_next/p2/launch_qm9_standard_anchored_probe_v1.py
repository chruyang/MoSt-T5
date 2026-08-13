#!/usr/bin/env python3
"""Launch the three standard-QM9 anchored representation-probe cells.

The matrix deliberately omits the already-screened ``F3D-l0_l12_mean`` cell.
It compares one state-free baseline, one coordinate-blind atom-state control,
and the surviving E3FP shell candidate on the same joined records and targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


SCHEMA_VERSION = "most-t5-p2/qm9-standard-anchored-probe-launcher/v1"
PROPERTY_NAMES = ("mu", "alpha", "r2", "u0", "u0_atom")
CELLS = (
    ("B0", "B0", "l0_l123_mean"),
    ("B2D", "B2D", "l0_l12_mean"),
    ("F3D-l0_l123_mean", "F3D", "l0_l123_mean"),
)


def cell_command(
    args: argparse.Namespace,
    *,
    name: str,
    cell: str,
    shell: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "most_t5_next.p2.run_qm9_anchored_probe_v1",
        "--cell", cell,
        "--shell-fusion-mode", shell,
        "--base-model-snapshot", str(args.base_model_snapshot),
        "--base-tokenizer-snapshot", str(args.base_tokenizer_snapshot),
        "--anchored-tokenizer-dir", str(args.anchored_tokenizer_dir),
        "--semantic-plan-sha256", args.semantic_plan_sha256,
        "--union-init-dir", str(args.union_init_dir),
        "--cache-root", str(args.cache_root),
        "--target-overlay-dir", str(args.target_overlay_dir),
        "--property-names", *PROPERTY_NAMES,
        "--output-dir", str(Path(args.output_root) / name),
        "--epochs", str(args.epochs),
        "--micro-batch-size", str(args.micro_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--learning-rate", str(args.learning_rate),
        "--warmup-updates", str(args.warmup_updates),
        "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise RuntimeError("launcher output root must be new")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    started = time.time()
    reports: list[dict[str, object]] = []
    for name, cell, shell in CELLS:
        command = cell_command(args, name=name, cell=cell, shell=shell)
        cell_started = time.time()
        with (logs / f"{name}.log").open("wb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        reports.append({
            "name": name,
            "cell": cell,
            "shell_fusion_mode": shell,
            "started_unix": cell_started,
            "finished_unix": time.time(),
            "returncode": result.returncode,
            "status": "pass" if result.returncode == 0 else "failed",
        })
        status = {
            "schema_version": SCHEMA_VERSION,
            "status": "running" if result.returncode == 0 else "failed",
            "started_unix": started,
            "properties": list(PROPERTY_NAMES),
            "cells": reports,
        }
        (output / "launcher_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result.returncode != 0:
            return status
    final = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "started_unix": started,
        "finished_unix": time.time(),
        "properties": list(PROPERTY_NAMES),
        "cells": reports,
    }
    (output / "launcher_status.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--target-overlay-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cells": len(report["cells"])}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

