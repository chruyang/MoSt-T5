"""Fail-closed gate for byte-identical PCQM4Mv2 CPU staging copies.

The canonical R1 source contract names files on the original AutoDL region.
A CPU worker in another region cannot satisfy ``samefile`` against those
paths, so it must not call the canonical-path gate with substituted paths.
Instead, this module verifies a staging receipt whose every role is derived
from, and byte/hash-equal to, the immutable v3 source contract.

Only the Python standard library is imported.  A staging command may call
``generate_and_verify_staging_receipt`` after copying files.  A production
runner may call ``verify_staging_receipt`` and consume paths exclusively
through the returned ``VerifiedPCQMStagingInputs.work_path`` method.  The
result is an input lock, not P1 admission evidence.
"""

from __future__ import print_function

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


STAGING_CONTRACT_SCHEMA = "most-t5-r1/pcqm4mv2-staging-receipt-contract/v1"
RECEIPT_SCHEMA = "most-t5-r1/pcqm4mv2-staging-receipt/v1"
SOURCE_CONTRACT_SCHEMA = "most-t5-r1/pcqm4mv2-source-contract/v3"
VERIFICATION_REPORT_SCHEMA = "most-t5-r1/pcqm4mv2-staging-verification/v1"

REQUIRED_ROLES = (
    "train_3d_sdf_archive",
    "companion_archive",
    "companion_data_csv_gz",
    "companion_split_dict_pt",
    "train_sdf_source_manifest",
    "train_sdf_member_hash_report",
    "companion_source_manifest",
    "companion_content_validation",
)
ARTIFACT_FIELDS = ("source_path", "work_path", "bytes", "sha256")
TRANSFER_FIELDS = (
    "transfer_id",
    "method",
    "source_endpoint",
    "destination_endpoint",
    "started_at_utc",
    "completed_at_utc",
    "status",
    "verification",
)
ALLOWED_TRANSFER_METHODS = (
    "rsync_over_ssh",
    "scp_over_ssh",
    "official_redownload",
    "object_storage_transfer",
)
CPU_RUNTIME_FIELDS = (
    "captured_at_utc",
    "hostname",
    "platform_system",
    "platform_release",
    "machine",
    "python_implementation",
    "python_version",
    "python_executable",
    "cpu_model",
    "logical_cpu_count",
    "affinity_cpu_count",
    "memory_bytes",
    "working_root",
    "environment",
    "packages",
    "snapshot_sha256",
)
CPU_ENVIRONMENT_FIELDS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
CPU_PACKAGES = ("python", "numpy", "scipy", "rdkit", "e3fp", "lmdb", "torch")
FILE_POLICY = {
    "source_contract_and_receipt": "absolute_regular_nonsymlink_files",
    "work_files": "canonical_absolute_regular_nonsymlink_files_below_working_root",
    "work_paths_must_be_distinct": True,
    "source_paths_must_equal_v3_contract": True,
    "bytes_and_sha256_must_equal_v3_contract": True,
    "receipt_serialization": "utf8_canonical_json_plus_lf",
}

_HEX64 = frozenset("0123456789abcdef")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_POSITIVE_DECIMAL = re.compile(r"^[1-9][0-9]*$")


def canonical_json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    observed_bytes = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            observed_bytes += len(block)
            digest.update(block)
    return observed_bytes, digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _require_object(value, label):
    if not isinstance(value, dict):
        raise RuntimeError("{} must be an object".format(label))
    return value


def _require_exact_keys(value, fields, label):
    value = _require_object(value, label)
    expected = set(fields)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            "{} fields are not exact (missing={}, extra={})".format(label, missing, extra)
        )
    return value


def _require_nonempty_text(value, label, maximum=512):
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise RuntimeError("{} must be non-empty text of at most {} characters".format(label, maximum))
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RuntimeError("{} contains a control character".format(label))
    return value


def _require_nonnegative_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError("{} must be a non-negative integer".format(label))
    return value


def _require_positive_int(value, label):
    value = _require_nonnegative_int(value, label)
    if value == 0:
        raise RuntimeError("{} must be positive".format(label))
    return value


def _absolute_path(value, label):
    if not isinstance(value, (str, Path)) or not str(value):
        raise RuntimeError("{} must be an absolute path string".format(label))
    target = Path(value)
    if not target.is_absolute():
        raise RuntimeError("{} must be an absolute path".format(label))
    return target


def _regular_nonsymlink_file(value, label, require_canonical=False):
    target = _absolute_path(value, label)
    if target.is_symlink():
        raise RuntimeError("{} must not be a symlink: {}".format(label, target))
    if not target.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, target))
    resolved = target.resolve(strict=True)
    if require_canonical and target != resolved:
        raise RuntimeError("{} must be a canonical absolute path".format(label))
    return resolved


def _regular_nonsymlink_directory(value, label):
    target = _absolute_path(value, label)
    if target.is_symlink():
        raise RuntimeError("{} must not be a symlink".format(label))
    if not target.is_dir():
        raise FileNotFoundError("{} is not a directory".format(label))
    resolved = target.resolve(strict=True)
    if target != resolved:
        raise RuntimeError("{} must be a canonical absolute path".format(label))
    return resolved


def _strict_json_object(raw, label):
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError("{} contains duplicate JSON key {}".format(label, key))
            result[key] = value
        return result

    def reject_nonfinite(token):
        raise RuntimeError("{} contains non-finite JSON token {}".format(label, token))

    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=reject_nonfinite)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("{} is not valid UTF-8 JSON".format(label)) from exc
    return _require_object(value, label)


def _load_json_file(path, label, require_canonical=False):
    target = _regular_nonsymlink_file(path, label)
    raw = target.read_bytes()
    value = _strict_json_object(raw, label)
    if require_canonical and raw != canonical_json_bytes(value) + b"\n":
        raise RuntimeError("{} is not canonical UTF-8 JSON plus LF".format(label))
    return target, value, raw


def _parse_utc(value, label):
    _require_nonempty_text(value, label, maximum=20)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise RuntimeError("{} must use YYYY-MM-DDTHH:MM:SSZ".format(label)) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _nested(value, keys, label):
    current = value
    for key in keys:
        current = _require_object(current, label).get(key)
    return current


def _derive_source_specs(source_contract):
    if source_contract.get("schema_version") != SOURCE_CONTRACT_SCHEMA:
        raise RuntimeError("source contract is not the frozen PCQM4Mv2 v3 schema")
    source = _require_object(source_contract.get("source"), "source contract source")
    companion = _require_object(
        source_contract.get("official_companion"), "source contract official_companion"
    )
    raw_specs = {
        "train_3d_sdf_archive": {
            "path": source.get("remote_archive_path"),
            "bytes": source.get("remote_archive_bytes"),
            "sha256": source.get("remote_archive_sha256"),
        },
        "companion_archive": companion.get("official_archive"),
        "companion_data_csv_gz": companion.get("data_csv"),
        "companion_split_dict_pt": companion.get("split_dict"),
        "train_sdf_source_manifest": source.get("remote_source_manifest"),
        "train_sdf_member_hash_report": source.get("remote_sdf_member_hash_report"),
        "companion_source_manifest": companion.get("remote_source_manifest"),
        "companion_content_validation": companion.get("remote_content_validation"),
    }
    specs = {}
    for role in REQUIRED_ROLES:
        spec = _require_object(raw_specs.get(role), "source contract role {}".format(role))
        source_path = spec.get("path")
        _absolute_path(source_path, "source contract {} path".format(role))
        expected_bytes = _require_nonnegative_int(
            spec.get("bytes"), "source contract {} bytes".format(role)
        )
        expected_sha256 = spec.get("sha256")
        if not _is_sha256(expected_sha256):
            raise RuntimeError("source contract {} SHA-256 is invalid".format(role))
        specs[role] = {
            "source_path": source_path,
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    if len({spec["source_path"] for spec in specs.values()}) != len(REQUIRED_ROLES):
        raise RuntimeError("source contract canonical role paths are not distinct")
    return specs


def _validate_staging_contract(contract, source_contract_path, source_contract_raw):
    required_top = {
        "schema_version",
        "purpose",
        "receipt_schema_version",
        "pinned_source_contract",
        "required_roles",
        "required_artifact_fields",
        "required_transfer_fields",
        "allowed_transfer_methods",
        "required_cpu_runtime_fields",
        "required_cpu_environment_fields",
        "required_cpu_packages",
        "file_policy",
    }
    _require_exact_keys(contract, required_top, "staging receipt contract")
    if contract["schema_version"] != STAGING_CONTRACT_SCHEMA:
        raise RuntimeError("staging receipt contract schema is invalid")
    if contract["receipt_schema_version"] != RECEIPT_SCHEMA:
        raise RuntimeError("staging receipt contract names an invalid receipt schema")
    expected_lists = (
        ("required_roles", REQUIRED_ROLES),
        ("required_artifact_fields", ARTIFACT_FIELDS),
        ("required_transfer_fields", TRANSFER_FIELDS),
        ("allowed_transfer_methods", ALLOWED_TRANSFER_METHODS),
        ("required_cpu_runtime_fields", CPU_RUNTIME_FIELDS),
        ("required_cpu_environment_fields", CPU_ENVIRONMENT_FIELDS),
        ("required_cpu_packages", CPU_PACKAGES),
    )
    for key, expected in expected_lists:
        if contract.get(key) != list(expected):
            raise RuntimeError("staging receipt contract {} is not the frozen list".format(key))
    if contract.get("file_policy") != FILE_POLICY:
        raise RuntimeError("staging receipt contract file policy differs from the frozen policy")

    pin = _require_exact_keys(
        contract.get("pinned_source_contract"),
        ("repository_relative_path", "filename", "schema_version", "bytes", "sha256"),
        "pinned source contract",
    )
    if pin["repository_relative_path"] != "contracts/pcqm4mv2_source_contract.json":
        raise RuntimeError("staging contract pins an unexpected repository source-contract path")
    if pin["filename"] != "pcqm4mv2_source_contract.json":
        raise RuntimeError("staging contract pins an unexpected source-contract filename")
    if pin["schema_version"] != SOURCE_CONTRACT_SCHEMA:
        raise RuntimeError("staging contract does not pin source contract v3")
    if source_contract_path.name != pin["filename"]:
        raise RuntimeError("provided source contract filename differs from the staging contract")
    if _require_nonnegative_int(pin["bytes"], "pinned source contract bytes") != len(source_contract_raw):
        raise RuntimeError("source contract byte count differs from the staging pin")
    if not _is_sha256(pin["sha256"]):
        raise RuntimeError("pinned source contract SHA-256 is invalid")
    if sha256_bytes(source_contract_raw) != pin["sha256"]:
        raise RuntimeError("source contract SHA-256 differs from the staging pin")
    return pin


def cpu_runtime_snapshot_sha256(cpu_runtime):
    """Hash a runtime object without its self-referential ``snapshot_sha256``."""
    runtime = dict(_require_object(cpu_runtime, "cpu_runtime"))
    runtime.pop("snapshot_sha256", None)
    return sha256_bytes(canonical_json_bytes(runtime))


def create_staging_receipt_document(
    staging_contract_path,
    source_contract_path,
    work_paths,
    transfer,
    cpu_runtime,
    receipt_created_at_utc,
):
    """Create a receipt from already-copied files, hashing every working role.

    Claims for source paths, byte counts, and SHA-256 values are always copied
    from the pinned v3 source contract; callers cannot supply replacements.
    ``cpu_runtime`` may omit ``snapshot_sha256`` and this function will add it.
    No receipt is written by this function.
    """
    staging_path, staging_contract, _ = _load_json_file(
        staging_contract_path, "staging receipt contract"
    )
    source_path, source_contract, source_raw = _load_json_file(
        source_contract_path, "source contract"
    )
    _validate_staging_contract(staging_contract, source_path, source_raw)
    source_specs = _derive_source_specs(source_contract)
    if not isinstance(work_paths, dict) or set(work_paths) != set(REQUIRED_ROLES):
        raise RuntimeError("work_paths role set is not exact")

    receipt_created = _parse_utc(receipt_created_at_utc, "receipt_created_at_utc")
    normalized_transfer = dict(_require_object(transfer, "transfer"))
    _validate_transfer(normalized_transfer, receipt_created)
    normalized_runtime = dict(_require_object(cpu_runtime, "cpu_runtime"))
    if "snapshot_sha256" not in normalized_runtime:
        normalized_runtime["snapshot_sha256"] = cpu_runtime_snapshot_sha256(normalized_runtime)
    working_root, _ = _validate_runtime(normalized_runtime, receipt_created)

    artifacts = {}
    seen_work_paths = set()
    for role in REQUIRED_ROLES:
        expected = source_specs[role]
        work_path = _regular_nonsymlink_file(
            work_paths[role], "work_paths.{}".format(role), require_canonical=True
        )
        try:
            work_path.relative_to(working_root)
        except ValueError as exc:
            raise RuntimeError("work_paths.{} is outside CPU working_root".format(role)) from exc
        work_key = str(work_path)
        if work_key in seen_work_paths:
            raise RuntimeError("staging work paths are not distinct")
        seen_work_paths.add(work_key)
        observed_bytes, observed_sha256 = sha256_file(work_path)
        if observed_bytes != expected["bytes"]:
            raise RuntimeError("work_paths.{} byte count differs from source contract".format(role))
        if observed_sha256 != expected["sha256"]:
            raise RuntimeError("work_paths.{} SHA-256 differs from source contract".format(role))
        artifacts[role] = {
            "source_path": expected["source_path"],
            "work_path": work_key,
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        }

    return {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_created_at_utc": receipt_created_at_utc,
        "source_contract": {
            "schema_version": SOURCE_CONTRACT_SCHEMA,
            "path": str(source_path),
            "bytes": len(source_raw),
            "sha256": sha256_bytes(source_raw),
        },
        "transfer": normalized_transfer,
        "cpu_runtime": normalized_runtime,
        "artifacts": artifacts,
    }


def write_staging_receipt(receipt_path, receipt):
    """Write one canonical receipt to an explicit non-symlink path."""
    target = _absolute_path(receipt_path, "receipt output path")
    parent = _regular_nonsymlink_directory(target.parent, "receipt output parent")
    if target.parent != parent:
        raise RuntimeError("receipt output parent must be a canonical absolute path")
    if target.is_symlink():
        raise RuntimeError("receipt output path must not be a symlink")
    if target.exists() and not target.is_file():
        raise RuntimeError("receipt output path exists and is not a regular file")
    target.write_bytes(canonical_json_bytes(_require_object(receipt, "receipt")) + b"\n")
    return target.resolve(strict=True)


def generate_and_verify_staging_receipt(
    staging_contract_path,
    source_contract_path,
    receipt_path,
    work_paths,
    transfer,
    cpu_runtime,
    receipt_created_at_utc,
):
    """Generate a canonical receipt, then independently re-read and verify it."""
    receipt = create_staging_receipt_document(
        staging_contract_path,
        source_contract_path,
        work_paths,
        transfer,
        cpu_runtime,
        receipt_created_at_utc,
    )
    persisted = write_staging_receipt(receipt_path, receipt)
    return verify_staging_receipt(staging_contract_path, source_contract_path, persisted)


def _validate_runtime(runtime, receipt_created_at):
    runtime = _require_exact_keys(runtime, CPU_RUNTIME_FIELDS, "cpu_runtime")
    captured = _parse_utc(runtime["captured_at_utc"], "cpu_runtime.captured_at_utc")
    if captured > receipt_created_at:
        raise RuntimeError("CPU runtime capture is later than receipt creation")
    for key in (
        "hostname",
        "platform_release",
        "python_implementation",
        "python_version",
        "cpu_model",
    ):
        _require_nonempty_text(runtime[key], "cpu_runtime.{}".format(key))
    if runtime["platform_system"] != "Linux":
        raise RuntimeError("CPU staging runtime must report platform_system=Linux")
    if str(runtime["machine"]).lower() not in ("x86_64", "amd64"):
        raise RuntimeError("CPU staging runtime must report an x86_64 machine")
    _absolute_path(runtime["python_executable"], "cpu_runtime.python_executable")
    logical = _require_positive_int(runtime["logical_cpu_count"], "cpu_runtime.logical_cpu_count")
    affinity = _require_positive_int(runtime["affinity_cpu_count"], "cpu_runtime.affinity_cpu_count")
    if affinity > logical:
        raise RuntimeError("CPU affinity count cannot exceed logical CPU count")
    _require_positive_int(runtime["memory_bytes"], "cpu_runtime.memory_bytes")
    working_root = _regular_nonsymlink_directory(runtime["working_root"], "cpu_runtime.working_root")

    environment = _require_exact_keys(
        runtime["environment"], CPU_ENVIRONMENT_FIELDS, "cpu_runtime.environment"
    )
    if environment["CUDA_VISIBLE_DEVICES"] != "-1":
        raise RuntimeError("CPU staging requires CUDA_VISIBLE_DEVICES=-1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = environment[key]
        if not isinstance(value, str) or not _POSITIVE_DECIMAL.match(value):
            raise RuntimeError("cpu_runtime.environment.{} must be a positive decimal string".format(key))
        if int(value) > affinity:
            raise RuntimeError("cpu_runtime.environment.{} exceeds CPU affinity".format(key))

    packages = _require_exact_keys(runtime["packages"], CPU_PACKAGES, "cpu_runtime.packages")
    for key in CPU_PACKAGES:
        _require_nonempty_text(packages[key], "cpu_runtime.packages.{}".format(key), maximum=128)
    if not _is_sha256(runtime["snapshot_sha256"]):
        raise RuntimeError("CPU runtime snapshot SHA-256 is invalid")
    observed_snapshot = cpu_runtime_snapshot_sha256(runtime)
    if runtime["snapshot_sha256"] != observed_snapshot:
        raise RuntimeError("CPU runtime snapshot SHA-256 mismatch")
    return working_root, observed_snapshot


def _validate_transfer(transfer, receipt_created_at):
    transfer = _require_exact_keys(transfer, TRANSFER_FIELDS, "transfer")
    transfer_id = transfer["transfer_id"]
    if not isinstance(transfer_id, str) or not _SAFE_ID.match(transfer_id):
        raise RuntimeError("transfer.transfer_id is not a safe identifier")
    if transfer["method"] not in ALLOWED_TRANSFER_METHODS:
        raise RuntimeError("transfer.method is not allowed")
    _require_nonempty_text(transfer["source_endpoint"], "transfer.source_endpoint")
    _require_nonempty_text(transfer["destination_endpoint"], "transfer.destination_endpoint")
    started = _parse_utc(transfer["started_at_utc"], "transfer.started_at_utc")
    completed = _parse_utc(transfer["completed_at_utc"], "transfer.completed_at_utc")
    if completed < started:
        raise RuntimeError("transfer completion precedes transfer start")
    if completed > receipt_created_at:
        raise RuntimeError("transfer completion is later than receipt creation")
    if transfer["status"] != "completed":
        raise RuntimeError("transfer.status must be completed")
    if transfer["verification"] != "post_transfer_bytes_and_sha256":
        raise RuntimeError("transfer.verification must be post_transfer_bytes_and_sha256")


@dataclass(frozen=True)
class VerifiedPCQMStagingInputs:
    staging_contract_path: str
    staging_contract_sha256: str
    receipt_path: str
    receipt_sha256: str
    source_contract_path: str
    source_contract_sha256: str
    cpu_runtime_sha256: str
    artifact_observations: tuple
    transfer: tuple

    def artifact(self, role):
        for candidate_role, observation in self.artifact_observations:
            if candidate_role == role:
                return dict(observation)
        raise KeyError(role)

    def work_path(self, role):
        """Return a verified CPU path; runners should not consume raw receipt paths."""
        return self.artifact(role)["work_path"]

    def report(self):
        return {
            "schema_version": VERIFICATION_REPORT_SCHEMA,
            "pass": True,
            "admission_scope": "cross_region_input_integrity_only",
            "p1_training_admitted": False,
            "staging_contract_path": self.staging_contract_path,
            "staging_contract_sha256": self.staging_contract_sha256,
            "receipt_path": self.receipt_path,
            "receipt_sha256": self.receipt_sha256,
            "source_contract_path": self.source_contract_path,
            "source_contract_sha256": self.source_contract_sha256,
            "cpu_runtime_sha256": self.cpu_runtime_sha256,
            "artifacts": {key: dict(value) for key, value in self.artifact_observations},
            "transfer": dict(self.transfer),
        }


def verify_staging_receipt(staging_contract_path, source_contract_path, receipt_path):
    """Verify a cross-region CPU copy against the exact canonical v3 contract.

    The source files named by v3 need not be mounted on the CPU machine.  Their
    canonical paths remain immutable receipt evidence, while each working file
    is re-hashed locally before a verified path is returned.
    """
    staging_path, staging_contract, staging_raw = _load_json_file(
        staging_contract_path, "staging receipt contract"
    )
    source_path, source_contract, source_raw = _load_json_file(
        source_contract_path, "source contract"
    )
    _validate_staging_contract(staging_contract, source_path, source_raw)
    source_specs = _derive_source_specs(source_contract)

    receipt_file, receipt, receipt_raw = _load_json_file(
        receipt_path, "staging receipt", require_canonical=True
    )
    _require_exact_keys(
        receipt,
        (
            "schema_version",
            "receipt_created_at_utc",
            "source_contract",
            "transfer",
            "cpu_runtime",
            "artifacts",
        ),
        "staging receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        raise RuntimeError("staging receipt schema is invalid")
    receipt_created = _parse_utc(receipt["receipt_created_at_utc"], "receipt_created_at_utc")

    receipt_source = _require_exact_keys(
        receipt["source_contract"],
        ("schema_version", "path", "bytes", "sha256"),
        "receipt source_contract",
    )
    if receipt_source["schema_version"] != SOURCE_CONTRACT_SCHEMA:
        raise RuntimeError("receipt does not bind source contract v3")
    receipt_source_path = _regular_nonsymlink_file(
        receipt_source["path"], "receipt source_contract path", require_canonical=True
    )
    if not receipt_source_path.samefile(source_path):
        raise RuntimeError("receipt source-contract path differs from the verified source contract")
    if receipt_source["bytes"] != len(source_raw):
        raise RuntimeError("receipt source-contract byte count mismatch")
    source_sha256 = sha256_bytes(source_raw)
    if receipt_source["sha256"] != source_sha256:
        raise RuntimeError("receipt source-contract SHA-256 mismatch")

    _validate_transfer(receipt["transfer"], receipt_created)
    working_root, runtime_sha256 = _validate_runtime(receipt["cpu_runtime"], receipt_created)

    artifacts = _require_object(receipt["artifacts"], "receipt artifacts")
    if set(artifacts) != set(REQUIRED_ROLES):
        raise RuntimeError("receipt artifact role set is not exact")
    observations = []
    seen_work_paths = set()
    for role in REQUIRED_ROLES:
        entry = _require_exact_keys(artifacts[role], ARTIFACT_FIELDS, "artifact {}".format(role))
        expected = source_specs[role]
        if entry["source_path"] != expected["source_path"]:
            raise RuntimeError("artifact {} source path differs from source contract".format(role))
        if entry["bytes"] != expected["bytes"]:
            raise RuntimeError("artifact {} byte claim differs from source contract".format(role))
        if entry["sha256"] != expected["sha256"]:
            raise RuntimeError("artifact {} SHA-256 claim differs from source contract".format(role))

        work_path = _regular_nonsymlink_file(
            entry["work_path"], "artifact {} work_path".format(role), require_canonical=True
        )
        try:
            work_path.relative_to(working_root)
        except ValueError as exc:
            raise RuntimeError("artifact {} is outside CPU working_root".format(role)) from exc
        work_key = str(work_path)
        if work_key in seen_work_paths:
            raise RuntimeError("staging work paths are not distinct")
        seen_work_paths.add(work_key)
        observed_bytes, observed_sha256 = sha256_file(work_path)
        if observed_bytes != expected["bytes"]:
            raise RuntimeError("artifact {} working byte count mismatch".format(role))
        if observed_sha256 != expected["sha256"]:
            raise RuntimeError("artifact {} working SHA-256 mismatch".format(role))
        observations.append(
            (
                role,
                {
                    "source_path": expected["source_path"],
                    "work_path": work_key,
                    "bytes": observed_bytes,
                    "sha256": observed_sha256,
                },
            )
        )

    return VerifiedPCQMStagingInputs(
        staging_contract_path=str(staging_path),
        staging_contract_sha256=sha256_bytes(staging_raw),
        receipt_path=str(receipt_file),
        receipt_sha256=sha256_bytes(receipt_raw),
        source_contract_path=str(source_path),
        source_contract_sha256=source_sha256,
        cpu_runtime_sha256=runtime_sha256,
        artifact_observations=tuple(observations),
        transfer=tuple((key, receipt["transfer"][key]) for key in TRANSFER_FIELDS),
    )
