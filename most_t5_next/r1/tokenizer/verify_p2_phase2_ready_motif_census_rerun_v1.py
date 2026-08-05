#!/usr/bin/env python3
"""Verify an independent fresh-process rerun of the P2 census extractor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional


RECEIPT_SCHEMA = "most-t5-r1/p2-phase2-ready-motif-census-receipt/v1"
REPORT_SCHEMA = "most-t5-r1/p2-phase2-ready-motif-census-rerun-verification/v1"
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_TRUSTED_PICKLE_CAN_EXECUTE_CODE"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_NAMES = (
    "membership.jsonl",
    "reject_ledger.jsonl",
    "record_projection.jsonl",
    "motif_census.jsonl",
    "pure_motif_census.jsonl",
    "anchor_summary.json",
)


class VerificationError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise VerificationError("{} must be a regular non-symlink file".format(label))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError("{} must be a JSON object".format(label))
    return value


def validate_receipt(root: Path) -> dict[str, Any]:
    receipt = load_json(root / "derivation_receipt.json", "P2 derivation receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise VerificationError("P2 derivation receipt schema mismatch")
    payload_hash = receipt.get("receipt_payload_sha256")
    if not isinstance(payload_hash, str) or SHA256_RE.fullmatch(payload_hash) is None:
        raise VerificationError("receipt payload SHA-256 is invalid")
    unhashed = dict(receipt)
    del unhashed["receipt_payload_sha256"]
    if sha256_bytes(canonical_json_bytes(unhashed)) != payload_hash:
        raise VerificationError("receipt self-hash mismatch")
    if receipt.get("release_status") != "candidate_p2_motif_census_non_release":
        raise VerificationError("P2 derivation receipt has an invalid status")
    if receipt.get("training_admission") is not False or receipt.get("p1_p2_union_decision_permitted") is not False:
        raise VerificationError("P2 receipt overclaims admission or a union decision")
    artifacts = receipt.get("artifacts")
    expected_keys = {name.rsplit(".", 1)[0] for name in ARTIFACT_NAMES}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_keys:
        raise VerificationError("P2 receipt artifact set is not closed")
    for name in ARTIFACT_NAMES:
        key = name.rsplit(".", 1)[0]
        row = artifacts[key]
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise VerificationError("P2 artifact observation fields are not closed")
        if row["relative_path"] != name:
            raise VerificationError("P2 artifact relative path mismatch")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise VerificationError("P2 artifact is absent or not regular: {}".format(name))
        if int(path.stat().st_size) != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise VerificationError("P2 artifact byte/hash mismatch: {}".format(name))
    return receipt


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_block = left_handle.read(8 * 1024 * 1024)
            right_block = right_handle.read(8 * 1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def verify_rerun(args: argparse.Namespace) -> dict[str, Any]:
    baseline = args.baseline_release.expanduser().resolve()
    rerun = args.rerun_output.expanduser().resolve()
    extractor = args.extractor.expanduser().resolve()
    source = args.source_lmdb.expanduser().resolve()
    source_lock = args.source_lock.expanduser().resolve()
    contract = args.contract.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if rerun.exists():
        raise FileExistsError("--rerun-output must be new")
    if report_path.exists():
        raise FileExistsError("--report must be new")
    baseline_receipt = validate_receipt(baseline)
    if sha256_file(extractor) != baseline_receipt.get("extractor", {}).get("sha256"):
        raise VerificationError("rerun extractor differs from baseline receipt")
    if sha256_file(source_lock) != baseline_receipt.get("source_lock", {}).get("sha256"):
        raise VerificationError("rerun source lock differs from baseline receipt")
    if sha256_file(contract) != baseline_receipt.get("contract", {}).get("sha256"):
        raise VerificationError("rerun contract differs from baseline receipt")
    if sha256_file(source) != baseline_receipt.get("source", {}).get("sha256"):
        raise VerificationError("rerun source differs from baseline receipt")
    command = [
        sys.executable,
        str(extractor),
        "--source-lmdb", str(source),
        "--source-lock", str(source_lock),
        "--contract", str(contract),
        "--output-dir", str(rerun),
        "--legacy-pickle-acknowledgement", ACKNOWLEDGEMENT,
    ]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(args.pythonhashseed)
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
    if completed.returncode != 0:
        raise VerificationError("fresh-process extractor failed: {}".format(completed.stderr[-4000:]))
    rerun_receipt = validate_receipt(rerun)
    comparisons = []
    for name in ARTIFACT_NAMES:
        baseline_sha = sha256_file(baseline / name)
        rerun_sha = sha256_file(rerun / name)
        comparisons.append({
            "relative_path": name,
            "baseline_sha256": baseline_sha,
            "rerun_sha256": rerun_sha,
            "byte_identical": baseline_sha == rerun_sha and files_equal(baseline / name, rerun / name),
        })
    passed = (
        all(item["byte_identical"] for item in comparisons)
        and baseline_receipt["logical_derivation_sha256"] == rerun_receipt["logical_derivation_sha256"]
        and baseline_receipt["counts"] == rerun_receipt["counts"]
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": utc_now(),
        "status": "pass" if passed else "fail",
        "baseline_release": str(baseline),
        "rerun_release": str(rerun),
        "pythonhashseed": str(args.pythonhashseed),
        "artifact_comparisons": comparisons,
        "baseline_logical_derivation_sha256": baseline_receipt["logical_derivation_sha256"],
        "rerun_logical_derivation_sha256": rerun_receipt["logical_derivation_sha256"],
        "logical_derivation_equal": baseline_receipt["logical_derivation_sha256"] == rerun_receipt["logical_derivation_sha256"],
        "byte_identical_rerun": passed,
        "p1_p2_union_decision_permitted": False,
        "training_admission": False,
    }
    report["report_payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    write_json_new(report_path, report)
    if not passed:
        raise VerificationError("independent rerun was not byte-identical")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-release", required=True, type=Path)
    parser.add_argument("--rerun-output", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--source-lmdb", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--pythonhashseed", default="271828")
    parser.add_argument("--legacy-pickle-acknowledgement", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.legacy_pickle_acknowledgement != ACKNOWLEDGEMENT:
        raise VerificationError("exact legacy-pickle acknowledgement literal is required")
    report = verify_rerun(args)
    print(json.dumps({
        "status": report["status"],
        "byte_identical_rerun": report["byte_identical_rerun"],
        "report": str(args.report.expanduser().resolve()),
        "training_admission": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
