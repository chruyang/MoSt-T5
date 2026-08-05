#!/usr/bin/env python3
"""CLI gate for a cross-region PCQM4Mv2 CPU staging receipt."""

from __future__ import print_function

import argparse
import importlib.util
import json
import sys
from pathlib import Path


R1_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = R1_ROOT / "adapter" / "pcqm_staging_receipt.py"
DEFAULT_CONTRACT = R1_ROOT / "contracts" / "pcqm4mv2_staging_receipt_contract.json"
DEFAULT_SOURCE_CONTRACT = R1_ROOT / "contracts" / "pcqm4mv2_source_contract.json"


def import_adapter():
    spec = importlib.util.spec_from_file_location("r1_pcqm_staging_receipt", str(ADAPTER_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import staging receipt adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="absolute canonical staging receipt path")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--source-contract", default=str(DEFAULT_SOURCE_CONTRACT))
    parser.add_argument("--output", help="optional JSON verification-report path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    adapter = import_adapter()
    try:
        verified = adapter.verify_staging_receipt(
            Path(args.contract).resolve(),
            Path(args.source_contract).resolve(),
            Path(args.receipt).resolve(),
        )
        report = verified.report()
    except Exception as exc:
        failure = {
            "schema_version": adapter.VERIFICATION_REPORT_SCHEMA,
            "pass": False,
            "p1_training_admitted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    encoded = adapter.canonical_json_bytes(report) + b"\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    print(encoded.decode("utf-8").rstrip("\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
