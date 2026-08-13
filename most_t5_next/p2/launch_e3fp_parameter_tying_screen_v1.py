#!/usr/bin/env python3
"""Launch and close the six-cell E3FP parameter-tying matrix.

With three visible GPUs, one candidate is assigned per GPU and its B2D/F3D
pair runs sequentially.  With fewer GPUs, candidates run in deterministic
waves.  Each subprocess sees exactly one CUDA device as ``cuda:0``.
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

from .run_e3fp_parameter_tying_screen_v1 import CANDIDATES, MANIFEST_FILENAME


SCHEMA_VERSION = "most-t5-p2/e3fp-parameter-tying-launcher/v1"


def cell_command(args: argparse.Namespace, *, cell: str, candidate: str) -> list[str]:
    name = f"{candidate}-{cell}"
    command = [
        sys.executable,
        "-m",
        "most_t5_next.p2.run_e3fp_parameter_tying_screen_v1",
        "--cell",
        cell,
        "--parameter-tying",
        candidate,
        "--base-model-snapshot",
        str(args.base_model_snapshot),
        "--base-tokenizer-snapshot",
        str(args.base_tokenizer_snapshot),
        "--anchored-tokenizer-dir",
        str(args.anchored_tokenizer_dir),
        "--semantic-plan-sha256",
        str(args.semantic_plan_sha256),
        "--union-init-dir",
        str(args.union_init_dir),
        "--cache-root",
        str(args.cache_root),
        "--output-dir",
        str(Path(args.output_root) / name),
        "--code-commit",
        str(args.code_commit),
        "--num-workers",
        str(args.workers_per_cell),
        "--prefetch-factor",
        str(args.prefetch_factor),
    ]
    overlay = (
        args.matched_b2d_overlay
        if cell == "B2D"
        else args.matched_f3d_overlay
    )
    if overlay is not None:
        command.extend(("--matched-overlay", str(overlay)))
    if args.save_final_checkpoints:
        command.append("--save-final-checkpoint")
    return command


def pair_command(args: argparse.Namespace, *, candidate: str, gpu_id: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "most_t5_next.p2.launch_e3fp_parameter_tying_screen_v1",
        "--base-model-snapshot",
        str(args.base_model_snapshot),
        "--base-tokenizer-snapshot",
        str(args.base_tokenizer_snapshot),
        "--anchored-tokenizer-dir",
        str(args.anchored_tokenizer_dir),
        "--semantic-plan-sha256",
        str(args.semantic_plan_sha256),
        "--union-init-dir",
        str(args.union_init_dir),
        "--cache-root",
        str(args.cache_root),
        "--output-root",
        str(args.output_root),
        "--code-commit",
        str(args.code_commit),
        "--workers-per-cell",
        str(args.workers_per_cell),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--pair-candidate",
        candidate,
        "--pair-gpu-id",
        str(gpu_id),
    ]
    if args.matched_b2d_overlay is not None:
        command.extend(("--matched-b2d-overlay", str(args.matched_b2d_overlay)))
    if args.matched_f3d_overlay is not None:
        command.extend(("--matched-f3d-overlay", str(args.matched_f3d_overlay)))
    if args.save_final_checkpoints:
        command.append("--save-final-checkpoints")
    return command


def _worker_environment(gpu_id: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _candidate_pair(args: argparse.Namespace, candidate: str, gpu_id: int) -> dict[str, object]:
    root = Path(args.output_root)
    rows = []
    for cell in ("B2D", "F3D"):
        name = f"{candidate}-{cell}"
        log_path = root / "logs" / f"{name}.log"
        command = cell_command(args, cell=cell, candidate=candidate)
        started = time.time()
        with log_path.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=_worker_environment(gpu_id),
                check=False,
            )
        row = {
            "cell": cell,
            "candidate": candidate,
            "gpu_id": gpu_id,
            "command": command,
            "started_unix": started,
            "finished_unix": time.time(),
            "returncode": completed.returncode,
            "status": "pass" if completed.returncode == 0 else "failed",
        }
        rows.append(row)
        if completed.returncode != 0:
            break
    return {
        "candidate": candidate,
        "gpu_id": gpu_id,
        "status": "pass" if len(rows) == 2 and all(row["status"] == "pass" for row in rows) else "failed",
        "cells": rows,
    }


def _update_zero_signature(manifest: dict[str, object]) -> str:
    evaluations = manifest.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations or evaluations[0].get("update") != 0:
        raise RuntimeError("cell manifest lacks update-zero evaluation")
    return json.dumps(evaluations[0], sort_keys=True, separators=(",", ":"))


def merge_completed_matrix(root: Path) -> dict[str, object]:
    manifests = {}
    for candidate in CANDIDATES:
        for cell in ("B2D", "F3D"):
            path = root / f"{candidate}-{cell}" / MANIFEST_FILENAME
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "pass":
                raise RuntimeError(f"cell did not pass: {candidate}/{cell}")
            manifests[f"{candidate}-{cell}"] = payload
    source_controls = {
        json.dumps(payload.get("source_control"), sort_keys=True, separators=(",", ":"))
        for payload in manifests.values()
    }
    if len(source_controls) != 1:
        raise RuntimeError("source-control contract drifted across matrix cells")
    for cell in ("B2D", "F3D"):
        signatures = {
            _update_zero_signature(manifests[f"{candidate}-{cell}"])
            for candidate in CANDIDATES
        }
        if len(signatures) != 1:
            raise RuntimeError(f"update-zero evaluation drifted across {cell} candidates")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "matrix_cells": sorted(manifests),
        "update_zero_evaluation_equal_within_state_kind": True,
        "candidate_count": len(CANDIDATES),
        "cell_count": len(manifests),
        "source_control": next(iter(manifests.values()))["source_control"],
        "scientific_boundary": {
            "identity_ce_alone_selects_winner": False,
            "qm9_or_other_3d_sensitive_probe_required": True,
            "endpoint_site_not_selected_here": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.output_root).expanduser().resolve()
    if root.exists():
        raise RuntimeError("output root must be new")
    (root / "logs").mkdir(parents=True)
    if args.gpu_count <= 0 or args.gpu_count > len(CANDIDATES):
        raise RuntimeError("gpu_count must be between one and three")
    if args.workers_per_cell <= 0 or args.prefetch_factor <= 0:
        raise RuntimeError("worker and prefetch counts must be positive")

    started = time.time()
    pairs = []
    status_path = root / "launcher_status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "started_unix": started,
                "gpu_count": args.gpu_count,
                "candidate_pairs": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # A wave contains at most one candidate pair per visible GPU.  The parent
    # waits only between waves, never between candidates in the same wave.
    for wave_start in range(0, len(CANDIDATES), args.gpu_count):
        wave = CANDIDATES[wave_start : wave_start + args.gpu_count]
        processes = []
        for gpu_id, candidate in enumerate(wave):
            command = pair_command(args, candidate=candidate, gpu_id=gpu_id)
            log = (root / "logs" / f"pair-{candidate}.json").open("x", encoding="utf-8")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
            processes.append((candidate, gpu_id, process, log))
        wave_failed = False
        for candidate, gpu_id, process, log in processes:
            returncode = process.wait()
            log.close()
            payload_path = root / "logs" / f"pair-{candidate}.json"
            try:
                lines = [line for line in payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                payload = json.loads(lines[-1])
            except Exception:
                payload = {"candidate": candidate, "gpu_id": gpu_id, "status": "failed", "cells": []}
            pairs.append(payload)
            wave_failed |= returncode != 0 or payload.get("status") != "pass"
            status_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "running" if not wave_failed else "failed",
                        "started_unix": started,
                        "gpu_count": args.gpu_count,
                        "candidate_pairs": pairs,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if wave_failed:
            break

    status = "pass" if len(pairs) == len(CANDIDATES) and all(row.get("status") == "pass" for row in pairs) else "failed"
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_unix": started,
        "finished_unix": time.time(),
        "gpu_count": args.gpu_count,
        "workers_per_cell": args.workers_per_cell,
        "candidate_pairs": pairs,
    }
    if status == "pass":
        report["matrix_closure"] = merge_completed_matrix(root)
    (root / "launcher_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-snapshot", type=Path, required=True)
    parser.add_argument("--base-tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--anchored-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-sha256", required=True)
    parser.add_argument("--union-init-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--matched-b2d-overlay", type=Path)
    parser.add_argument("--matched-f3d-overlay", type=Path)
    parser.add_argument("--gpu-count", type=int, default=3)
    parser.add_argument("--workers-per-cell", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=5)
    parser.add_argument("--save-final-checkpoints", action="store_true")
    parser.add_argument("--pair-candidate", choices=CANDIDATES)
    parser.add_argument("--pair-gpu-id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pair_candidate is not None:
        if args.pair_gpu_id is None:
            raise SystemExit("pair mode requires --pair-gpu-id")
        report = _candidate_pair(args, args.pair_candidate, args.pair_gpu_id)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    if args.pair_gpu_id is not None:
        raise SystemExit("--pair-gpu-id requires --pair-candidate")
    report = run(args)
    print(json.dumps({"status": report["status"], "pairs": len(report["candidate_pairs"])}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "cell_command", "merge_completed_matrix", "pair_command", "run"]
