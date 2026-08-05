#!/usr/bin/env python3
"""Structurally validate a canonical CPU runtime attestation sidecar."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

try:
    from . import capture_cpu_runtime_attestation as collector
except ImportError:
    import capture_cpu_runtime_attestation as collector


def _validate_file_array(value, label, errors):
    if not isinstance(value, list) or not value:
        errors.append("{} file array is empty or invalid".format(label))
        return
    paths = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append("{} file {} is not an object".format(label, index))
            continue
        relative = item.get("relative_path")
        byte_count = item.get("bytes")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            errors.append("{} file {} has no relative path".format(label, index))
        else:
            paths.append(relative)
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            errors.append("{} file {} has invalid byte count".format(label, index))
        if not collector._is_sha256(digest):
            errors.append("{} file {} has invalid SHA-256".format(label, index))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        errors.append("{} file paths are not sorted and unique".format(label))


def validate_attestation(attestation_path, contract_path):
    report_path = collector.require_regular_nonsymlink_file(attestation_path, "CPU runtime attestation")
    raw = report_path.read_bytes()
    try:
        report = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return ["attestation JSON cannot be parsed: {}".format(type(exc).__name__)]
    errors = []
    try:
        if raw != collector.canonical_json_bytes(report):
            errors.append("attestation is not exact canonical JSON")
    except (TypeError, ValueError) as exc:
        errors.append("attestation contains a non-canonical value: {}".format(type(exc).__name__))
    if not isinstance(report, dict) or report.get("schema_version") != collector.ATTESTATION_SCHEMA:
        return errors + ["attestation schema is not the expected v1"]

    _, contract, contract_observation = collector.load_contract(contract_path)
    if report.get("contract") != contract_observation:
        errors.append("attestation contract lock differs from the supplied contract")

    claimed_payload = report.get("attestation_payload_sha256")
    unsigned = dict(report)
    unsigned.pop("attestation_payload_sha256", None)
    if not collector._is_sha256(claimed_payload) or claimed_payload != collector.sha256_bytes(
        collector.canonical_json_bytes(unsigned)
    ):
        errors.append("attestation payload SHA-256 is invalid")

    runtime = report.get("runtime")
    if not isinstance(runtime, dict):
        return errors + ["attestation runtime is absent"]
    dependencies = runtime.get("dependencies")
    if not isinstance(dependencies, dict) or tuple(dependencies) != tuple(sorted(collector.REQUIRED_DEPENDENCY_ROLES)):
        errors.append("attestation dependency role set differs from the required six roles")
    else:
        for role in collector.REQUIRED_DEPENDENCY_ROLES:
            item = dependencies[role]
            if not isinstance(item, dict) or item.get("status") != "ok":
                errors.append("dependency {} is not attested as ok".format(role))
                continue
            if not item.get("module_versions") and not item.get("distributions"):
                errors.append("dependency {} has no exact version observation".format(role))

    thread = runtime.get("thread_environment", {})
    observed_thread = thread.get("observed") if isinstance(thread, dict) else None
    required_thread = contract["thread_environment"]["required_exact"]
    if not isinstance(observed_thread, dict):
        errors.append("thread environment observation is absent")
    else:
        for key, expected in required_thread.items():
            if observed_thread.get(key) != expected:
                errors.append("thread environment {} differs from contract".format(key))

    cpu = runtime.get("cpu")
    if not isinstance(cpu, dict) or not isinstance(cpu.get("cgroup"), dict):
        errors.append("CPU/cgroup observation is absent")
    platform_observation = runtime.get("platform")
    if not isinstance(platform_observation, dict):
        errors.append("platform/libc observation is absent")
    elif platform_observation.get("operating_system", {}).get("system") == "Linux" and not platform_observation.get(
        "libc", {}
    ).get("resolved_glibc_versions"):
        errors.append("Linux glibc observation is absent")

    for key, label in (("e3fp_source_closure", "E3FP"), ("bundle_file_lock", "bundle")):
        lock = report.get(key)
        if not isinstance(lock, dict):
            errors.append("{} lock is absent".format(label))
            continue
        files = lock.get("files")
        _validate_file_array(files, label, errors)
        if isinstance(files, list):
            expected_digest = collector.sha256_bytes(collector.canonical_json_bytes(files))
            if lock.get("closure_sha256") != expected_digest:
                errors.append("{} closure SHA-256 is invalid".format(label))
            if lock.get("file_count") != len(files):
                errors.append("{} file count is invalid".format(label))
            expected_bytes = sum(
                item.get("bytes", 0) for item in files if isinstance(item, dict) and isinstance(item.get("bytes"), int)
            )
            if lock.get("total_bytes") != expected_bytes:
                errors.append("{} total byte count is invalid".format(label))
        if key == "e3fp_source_closure" and isinstance(files, list):
            names = {item.get("relative_path") for item in files if isinstance(item, dict)}
            missing = sorted(set(contract["e3fp_source_closure"]["required_relative_files"]) - names)
            if missing:
                errors.append("E3FP closure lacks required anchors")

    reported_errors = report.get("errors")
    if not isinstance(reported_errors, list) or reported_errors:
        errors.append("attestation contains collection errors")
    if report.get("pass") is not True:
        errors.append("attestation pass flag is not true")
    return sorted(set(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--contract", default=str(collector.default_contract_path()))
    args = parser.parse_args(argv)
    errors = validate_attestation(args.attestation, args.contract)
    print(json.dumps({"pass": not errors, "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
