"""Launch PF-10 G cells over one or more visible GPUs.

Cells are independent processes.  With three GPU IDs they run concurrently;
with one or two IDs the same frozen cells are queued without changing their
training contract.  Each child sees exactly one device as ``cuda:0``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


SCHEMA_VERSION = "most-t5-p2/pf10-factorized-grammar-launch/v1"
CELLS = ("B0", "B2D", "F3D")


class PF10GrammarLaunchError(RuntimeError):
    """The requested GPU/cell launch matrix is invalid."""


def _csv(value: str) -> tuple[str, ...]:
    rows = tuple(item.strip() for item in value.split(",") if item.strip())
    if not rows or len(rows) != len(set(rows)):
        raise PF10GrammarLaunchError("comma-separated values must be nonempty and unique")
    return rows


def _cell_command(args: argparse.Namespace, cell: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "most_t5_next.p2.run_pf10_factorized_grammar_v1",
        "--cell",
        cell,
        "--paired-release",
        str(args.paired_release),
        "--base-model-snapshot",
        str(args.base_model_snapshot),
        "--base-tokenizer-snapshot",
        str(args.base_tokenizer_snapshot),
        "--union-init-dir",
        str(args.union_init_dir),
        "--output-dir",
        str(output),
        "--cache-workers",
        str(args.cache_workers),
        "--cache-max-pending",
        str(args.cache_max_pending),
    ]
    if cell == "B2D":
        command.extend(
            [
                "--morgan-overlay",
                str(args.morgan_overlay),
                "--s-checkpoint",
                str(Path(args.s_stage_root) / "B2D" / "step-2500"),
            ]
        )
    elif cell == "F3D":
        command.extend(
            [
                "--shuffle-overlay",
                str(args.shuffle_overlay),
                "--s-checkpoint",
                str(Path(args.s_stage_root) / "F3D" / "step-2500"),
            ]
        )
    return command


def launch_matrix(args: argparse.Namespace) -> dict[str, object]:
    cells = _csv(args.cells)
    gpu_ids = _csv(args.gpu_ids)
    if any(cell not in CELLS for cell in cells):
        raise PF10GrammarLaunchError("cells must be drawn from B0,B2D,F3D")
    if args.cache_workers <= 0 or args.cache_max_pending < args.cache_workers:
        raise PF10GrammarLaunchError("cache worker bounds are invalid")
    if "B2D" in cells and args.morgan_overlay is None:
        raise PF10GrammarLaunchError("B2D requires --morgan-overlay")
    if "F3D" in cells and (args.shuffle_overlay is None or args.s_stage_root is None):
        raise PF10GrammarLaunchError("F3D requires shuffle overlay and S-stage root")
    if "B2D" in cells and args.s_stage_root is None:
        raise PF10GrammarLaunchError("B2D requires S-stage root")

    output_root = Path(args.output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    pending = list(cells)
    available = list(gpu_ids)
    running: dict[str, tuple[subprocess.Popen, object, str, float]] = {}
    results: list[dict[str, object]] = []
    started = time.time()
    while pending or running:
        while pending and available:
            cell = pending.pop(0)
            gpu = available.pop(0)
            cell_output = output_root / cell
            log_path = output_root / f"{cell}.log"
            log_handle = log_path.open("x", encoding="utf-8", newline="\n")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment["HF_HUB_OFFLINE"] = "1"
            process = subprocess.Popen(
                _cell_command(args, cell, cell_output),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            running[cell] = (process, log_handle, gpu, time.time())
        finished: list[str] = []
        for cell, (process, log_handle, gpu, cell_started) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()  # type: ignore[attr-defined]
            results.append(
                {
                    "cell": cell,
                    "gpu_id": gpu,
                    "return_code": return_code,
                    "seconds": time.time() - cell_started,
                    "output_dir": str(output_root / cell),
                    "log": str(output_root / f"{cell}.log"),
                }
            )
            available.append(gpu)
            finished.append(cell)
        for cell in finished:
            del running[cell]
        if running and not finished:
            time.sleep(2.0)

    results.sort(key=lambda row: cells.index(str(row["cell"])))
    passed = all(row["return_code"] == 0 for row in results)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed else "failed",
        "cells": list(cells),
        "gpu_ids": list(gpu_ids),
        "scheduling": "one_process_per_cell_bounded_by_gpu_count",
        "cell_results": results,
        "wall_seconds": time.time() - started,
        "scientific_contract_unchanged_by_gpu_count": True,
    }
    (output_root / "launcher_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", default="B0,B2D,F3D")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path)
    parser.add_argument("--shuffle-overlay", type=Path)
    parser.add_argument("--s-stage-root", type=Path)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--cache-max-pending", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = launch_matrix(_parser().parse_args(argv))
    print(json.dumps({"status": report["status"], "cells": report["cells"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PF10GrammarLaunchError", "SCHEMA_VERSION", "launch_matrix"]
