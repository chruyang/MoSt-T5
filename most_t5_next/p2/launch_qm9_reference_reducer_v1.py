#!/usr/bin/env python3
"""Run the paired QM9 fixed-reference versus adaptive-L0 reducer screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


SCHEMA_VERSION = "most-t5-p2/qm9-reference-reducer-launcher/v1"
PROPERTY_NAMES = ("mu", "alpha", "r2", "u0", "u0_atom")
CELLS = (
    ("B2D-reference", "B2D", "reference_fixed_four_mean"),
    ("B2D-adaptive", "B2D", "adaptive_l0_high"),
    ("F3D-reference", "F3D", "reference_fixed_four_mean"),
    ("F3D-adaptive", "F3D", "adaptive_l0_high"),
    ("B2D-linear", "B2D", "linear_l0_high"),
    ("F3D-linear", "F3D", "linear_l0_high"),
    ("B2D-minimal-phi", "B2D", "minimal_phi_l0_high"),
    ("F3D-minimal-phi", "F3D", "minimal_phi_l0_high"),
    ("B2D-level-aware-phi", "B2D", "level_aware_phi_l0_high"),
    ("F3D-level-aware-phi", "F3D", "level_aware_phi_l0_high"),
)


def cell_command(
    args: argparse.Namespace, *, name: str, cell: str, shell: str
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
        "--train-seed", str(args.train_seed),
        "--adapter-seed", str(args.adapter_seed),
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
    selected_names = set(args.cell_names or (name for name, _, _ in CELLS))
    selected_cells = [row for row in CELLS if row[0] in selected_names]
    if len(selected_cells) != len(selected_names):
        raise RuntimeError("requested reducer cell name is invalid")
    for name, cell, shell in selected_cells:
        command = cell_command(args, name=name, cell=cell, shell=shell)
        cell_started = time.time()
        with (logs / f"{name}.log").open("wb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        reports.append({
            "name": name,
            "cell": cell,
            "shell_reducer_mode": shell,
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
    parser.add_argument(
        "--cell-names",
        nargs="+",
        choices=[name for name, _, _ in CELLS],
    )
    parser.add_argument("--train-seed", type=int, default=20260810)
    parser.add_argument("--adapter-seed", type=int, default=20260809)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cells": len(report["cells"])}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
