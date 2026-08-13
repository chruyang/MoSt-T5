#!/usr/bin/env python3
"""Launch the final six-cell PF1 atom-encoder elimination screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

SCHEMA_VERSION = "most-t5-p2/anchored-atom-encoder-screen-launcher/v1"
CANDIDATES = (
    "reference_fixed_four_mean",
    "l0_high_minimal_phi",
    "l0_high_level_aware_phi",
)
MATRIX = tuple(
    (cell, candidate, f"{cell}-{candidate}")
    for candidate in CANDIDATES
    for cell in ("B2D", "F3D")
)


def cell_command(
    args: argparse.Namespace, cell: str, candidate: str, name: str
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "most_t5_next.p2.run_anchored_atom_encoder_screen_v1",
        "--cell",
        cell,
        "--atom-encoder-candidate",
        candidate,
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
        "--output-dir",
        str(Path(args.output_root) / name),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
    ]
    if args.matched_overlay is not None:
        command.extend(("--matched-overlay", str(args.matched_overlay)))
    return command


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_root).expanduser().resolve()
    if root.exists():
        raise RuntimeError("output root must be new")
    logs = root / "logs"
    logs.mkdir(parents=True)
    status_path = root / "launcher_status.json"
    started = time.time()
    rows: list[dict[str, object]] = []
    for cell, candidate, name in MATRIX:
        command = cell_command(args, cell, candidate, name)
        row: dict[str, object] = {
            "cell": cell,
            "atom_encoder_candidate": candidate,
            "name": name,
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
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
                check=False,
            )
        row.update(
            {
                "finished_unix": time.time(),
                "returncode": completed.returncode,
                "status": "pass" if completed.returncode == 0 else "failed",
            }
        )
        if completed.returncode != 0:
            break
    status = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if len(rows) == len(MATRIX) and all(
            row["status"] == "pass" for row in rows
        ) else "failed",
        "started_unix": started,
        "finished_unix": time.time(),
        "cells": rows,
        "selection_boundary": {
            "one_percent_is_elimination_only": True,
            "ten_percent_finalists": 2,
            "new_candidates_after_this_matrix": False,
        },
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--matched-overlay", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cells": report["cells"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
