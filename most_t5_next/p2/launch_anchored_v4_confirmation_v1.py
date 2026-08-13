#!/usr/bin/env python3
"""Run the paired B2D/F3D anchored V4 confirmation on one GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


SCHEMA_VERSION = "most-t5-p2/anchored-v4-confirmation-launcher/v1"
MATRIX = (
    ("B2D", "l0_l12_mean", "B2D"),
    ("F3D", "l0_l12_mean", "F3D-l0_l12_mean"),
    ("F3D", "l0_l123_mean", "F3D-l0_l123_mean"),
)


def cell_command(
    args: argparse.Namespace, cell: str, mode: str, name: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "most_t5_next.p2.run_anchored_v4_shell_screen_v1",
        "--cell",
        cell,
        "--shell-fusion-mode",
        mode,
        "--base-model-snapshot",
        str(args.base_model_snapshot),
        "--base-tokenizer-snapshot",
        str(args.base_tokenizer_snapshot),
        "--anchored-tokenizer-dir",
        str(args.anchored_tokenizer_dir),
        "--semantic-plan-sha256",
        args.semantic_plan_sha256,
        "--union-init-dir",
        str(args.union_init_dir),
        "--cache-root",
        str(args.cache_root),
        "--matched-overlay",
        str(args.matched_overlay),
        "--output-dir",
        str(Path(args.output_root) / name),
        "--num-workers",
        str(args.num_workers),
        "--save-final-checkpoint",
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_root).expanduser().resolve()
    if root.exists():
        raise RuntimeError("output root must be new")
    logs = root / "logs"
    logs.mkdir(parents=True)
    status_path = root / "launcher_status.json"
    started = time.time()
    rows = []
    for cell, mode, name in MATRIX:
        row = {
            "cell": cell,
            "name": name,
            "shell_fusion_mode": mode,
            "status": "running",
            "started_unix": time.time(),
        }
        rows.append(row)
        status_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running",
                    "started_unix": started,
                    "cells": rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with (logs / f"{name}.log").open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                cell_command(args, cell, mode, name),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
                check=False,
            )
        row["finished_unix"] = time.time()
        row["returncode"] = completed.returncode
        row["status"] = "pass" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            break
    status = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if len(rows) == len(MATRIX) and all(row["status"] == "pass" for row in rows) else "failed",
        "started_unix": started,
        "finished_unix": time.time(),
        "cells": rows,
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--matched-overlay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cells": report["cells"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MATRIX", "SCHEMA_VERSION", "cell_command", "run"]
