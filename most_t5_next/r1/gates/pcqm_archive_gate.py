#!/usr/bin/env python3
"""Verify a frozen PCQM4Mv2 archive without extracting it.

This R1 gate reads the archive bytes only to calculate checksums.  It never
extracts SDF content, writes a dataset, or changes the archive.  Output is one
small JSON report in a caller-provided new sidecar location.
"""

from __future__ import print_function

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path


OFFICIAL_OGB_MD5 = "fd72bce606e7ddf36c2a832badeec6ab"
OFFICIAL_OGB_TRAIN_RECORDS = 3378606


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def hash_file(path, chunk_size=8 * 1024 * 1024):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    bytes_read = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            md5.update(block)
            sha256.update(block)
            bytes_read += len(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest(), "bytes_read": bytes_read}


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def load_json_if_present(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True, help="new sidecar JSON report path")
    parser.add_argument("--source-contract", default="")
    parser.add_argument("--expected-md5", default=OFFICIAL_OGB_MD5)
    parser.add_argument("--expected-bytes", type=int, default=1559712928)
    parser.add_argument("--expected-records", type=int, default=OFFICIAL_OGB_TRAIN_RECORDS)
    parser.add_argument("--label", default="pcqm4mv2-train-3d")
    args = parser.parse_args()

    archive = Path(args.archive)
    errors = []
    if not archive.is_file():
        errors.append("archive is not a regular file")
        observed = {"path": str(archive), "exists": archive.exists()}
    else:
        stat = archive.stat()
        observed = {
            "path": str(archive.resolve()),
            "exists": True,
            "bytes": int(stat.st_size),
            "modified_utc": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat(),
        }
        observed.update(hash_file(archive))
        if observed["bytes"] != args.expected_bytes:
            errors.append("archive byte count differs from frozen expected value")
        if observed["md5"].lower() != args.expected_md5.lower():
            errors.append("archive MD5 differs from frozen expected value")

    contract = load_json_if_present(args.source_contract)
    if contract is not None:
        source = contract.get("source", {})
        if source.get("official_md5", "").lower() != args.expected_md5.lower():
            errors.append("source contract official MD5 conflicts with gate argument")
        if source.get("official_train_sdf_records") != args.expected_records:
            errors.append("source contract expected record count conflicts with gate argument")

    report = {
        "schema_version": "most-t5-r1/pcqm-archive-gate/v1",
        "created_utc": utc_now(),
        "label": args.label,
        "scope": {
            "archive_extracted": False,
            "dataset_records_written": 0,
            "local_data_transfer": False,
        },
        "expected": {
            "official_md5": args.expected_md5,
            "archive_bytes": args.expected_bytes,
            "official_train_sdf_records": args.expected_records,
        },
        "observed": observed,
        "source_contract_path": str(Path(args.source_contract).resolve()) if args.source_contract else None,
        "pass": not errors,
        "errors": errors,
        "next_gate": "pcqm_stream_smoke.py; passing this archive check does not admit PCQM data to P1",
    }
    write_json_atomic(args.output, report)
    print(json.dumps({"pass": report["pass"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
