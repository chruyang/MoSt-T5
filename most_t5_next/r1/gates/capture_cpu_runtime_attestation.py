#!/usr/bin/env python3
"""Capture a fail-closed CPU runtime and transferred-code attestation.

The collector is deliberately data-independent: it never opens PCQM4Mv2,
never imports the builder, and never writes anywhere except one caller-named
new JSON report.  The report binds the executable runtime, dependency origins,
CPU/cgroup limits, worker thread environment, the complete vendored E3FP source
closure, and an explicit list of transferred bundle files.

An attestation is an observation, not a compatibility proof.  A new CPU host
must additionally reproduce the separately frozen 128-record golden result.
"""

from __future__ import print_function

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path, PurePosixPath

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - retained for older supported Python.
    import importlib_metadata  # type: ignore


CONTRACT_SCHEMA = "most-t5-r1/cpu-runtime-attestation-contract/v1"
ATTESTATION_SCHEMA = "most-t5-r1/cpu-runtime-attestation/v1"
REQUIRED_DEPENDENCY_ROLES = ("numpy", "scipy", "rdkit", "lmdb", "mmh3", "torch")
REQUIRED_THREAD_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
HEX64 = frozenset("0123456789abcdef")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_canonical_value(value, location="$"):
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        raise TypeError("floating-point value is forbidden in canonical attestation JSON at {}".format(location))
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_value(item, "{}[{}]".format(location, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("non-string object key in canonical attestation JSON at {}".format(location))
            _validate_canonical_value(item, "{}.{}".format(location, key))
        return
    raise TypeError("unsupported canonical attestation JSON type {} at {}".format(type(value).__name__, location))


def canonical_json_bytes(value):
    """Return the contract's deterministic, float-free UTF-8 JSON encoding."""
    _validate_canonical_value(value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encoded + b"\n"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    target = Path(path)
    before = target.stat()
    digest = hashlib.sha256()
    with open(str(target), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = target.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("file changed while it was being attested: {}".format(target))
    return int(after.st_size), digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def require_regular_nonsymlink_file(path, label):
    target = Path(path).expanduser()
    if target.is_symlink():
        raise RuntimeError("{} must not be a symlink: {}".format(label, target))
    if not target.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, target))
    return target.resolve(strict=True)


def require_nonsymlink_directory(path, label):
    target = Path(path).expanduser()
    if target.is_symlink():
        raise RuntimeError("{} must not be a symlink: {}".format(label, target))
    if not target.is_dir():
        raise NotADirectoryError("{} is not a directory: {}".format(label, target))
    return target.resolve(strict=True)


def file_observation(path, include_path=True):
    target = require_regular_nonsymlink_file(path, "attested file")
    byte_count, digest = sha256_file(target)
    result = {"bytes": byte_count, "sha256": digest}
    if include_path:
        result["path"] = str(target)
    return result


def load_contract(path):
    target = require_regular_nonsymlink_file(path, "runtime attestation contract")
    try:
        with open(str(target), "r", encoding="utf-8") as handle:
            contract = json.load(handle)
    except Exception as exc:
        raise RuntimeError("cannot parse runtime attestation contract: {}".format(type(exc).__name__)) from exc
    validate_contract(contract)
    byte_count, digest = sha256_file(target)
    return target, contract, {"path": str(target), "bytes": byte_count, "sha256": digest}


def validate_contract(contract):
    if not isinstance(contract, dict) or contract.get("schema_version") != CONTRACT_SCHEMA:
        raise RuntimeError("runtime attestation contract schema is not the expected v1")
    if contract.get("attestation_schema_version") != ATTESTATION_SCHEMA:
        raise RuntimeError("runtime attestation output schema is not the expected v1")

    dependencies = contract.get("dependencies")
    if not isinstance(dependencies, list):
        raise RuntimeError("runtime contract dependencies must be an array")
    roles = [item.get("role") for item in dependencies if isinstance(item, dict)]
    if tuple(roles) != REQUIRED_DEPENDENCY_ROLES:
        raise RuntimeError("runtime contract dependency roles or order differ from the required six-role lock")
    for item in dependencies:
        if not isinstance(item.get("import_name"), str) or not item["import_name"]:
            raise RuntimeError("dependency {} has no import_name".format(item.get("role")))
        names = item.get("distribution_names")
        attributes = item.get("version_attributes")
        if not isinstance(names, list) or not names or not all(isinstance(value, str) and value for value in names):
            raise RuntimeError("dependency {} has no distribution candidates".format(item["role"]))
        if not isinstance(attributes, list) or not attributes:
            raise RuntimeError("dependency {} has no version attributes".format(item["role"]))

    thread = contract.get("thread_environment", {})
    exact = thread.get("required_exact") if isinstance(thread, dict) else None
    if not isinstance(exact, dict) or tuple(sorted(exact)) != tuple(sorted(REQUIRED_THREAD_KEYS)):
        raise RuntimeError("runtime contract must pin exactly the four required worker thread variables")
    if any(exact[key] != "1" for key in REQUIRED_THREAD_KEYS):
        raise RuntimeError("all required worker thread variables must be pinned to one")

    closure = contract.get("e3fp_source_closure", {})
    anchors = closure.get("required_relative_files") if isinstance(closure, dict) else None
    if not isinstance(anchors, list) or not anchors:
        raise RuntimeError("runtime contract has no E3FP source anchors")
    if "pipeline.py" not in anchors or "fingerprint/fprinter.py" not in anchors or "config/defaults.cfg" not in anchors:
        raise RuntimeError("runtime contract lacks a required E3FP semantic anchor")
    if closure.get("generated_directory_exclusions") != ["__pycache__"]:
        raise RuntimeError("runtime contract E3FP directory exclusion is not the locked generated-cache policy")
    if closure.get("generated_file_suffix_exclusions") != [".pyc", ".pyo"]:
        raise RuntimeError("runtime contract E3FP file exclusion is not the locked bytecode policy")
    minimum = contract.get("bundle_lock", {}).get("minimum_explicit_files")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise RuntimeError("runtime contract bundle minimum is invalid")


def write_new_canonical_json(path, value):
    """Exclusively create one canonical report; partial evidence is never overwritten."""
    target = Path(path).expanduser()
    if target.exists():
        raise FileExistsError("refusing to replace an existing attestation: {}".format(target))
    if not target.parent.is_dir():
        raise NotADirectoryError("attestation parent directory does not exist: {}".format(target.parent))
    payload = canonical_json_bytes(value)
    with open(str(target), "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return target.resolve(strict=True)


def _safe_executable_observation(path):
    supplied = Path(path).expanduser()
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("Python executable does not resolve to a regular file: {}".format(supplied))
    byte_count, digest = sha256_file(resolved)
    return {
        "supplied_path": str(supplied),
        "resolved_path": str(resolved),
        "supplied_path_is_symlink": bool(supplied.is_symlink()),
        "bytes": byte_count,
        "sha256": digest,
    }


def collect_python_observation():
    version = sys.version_info
    return {
        "implementation": platform.python_implementation(),
        "version": "{}.{}.{}".format(version.major, version.minor, version.micro),
        "version_info": {
            "major": int(version.major),
            "minor": int(version.minor),
            "micro": int(version.micro),
            "releaselevel": str(version.releaselevel),
            "serial": int(version.serial),
        },
        "compiler": platform.python_compiler(),
        "build": list(platform.python_build()),
        "cache_tag": getattr(sys.implementation, "cache_tag", None),
        "abi_flags": getattr(sys, "abiflags", ""),
        "byteorder": sys.byteorder,
        "maxsize": int(sys.maxsize),
        "executable": _safe_executable_observation(sys.executable),
    }


def _glibc_from_ctypes():
    try:
        symbol = ctypes.CDLL(None).gnu_get_libc_version
        symbol.restype = ctypes.c_char_p
        value = symbol()
        return value.decode("ascii") if value else None
    except Exception:
        return None


def collect_platform_observation():
    libc_name, libc_version = platform.libc_ver()
    try:
        confstr_value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        confstr_value = None
    confstr_version = None
    if isinstance(confstr_value, str):
        match = re.fullmatch(r"glibc\s+(.+)", confstr_value.strip())
        confstr_version = match.group(1) if match else None
    ctypes_version = _glibc_from_ctypes()
    glibc_candidates = [value for value in (ctypes_version, confstr_version) if value]
    if libc_name.lower() in ("glibc", "gnu libc") and libc_version:
        glibc_candidates.append(libc_version)
    glibc_versions = sorted(set(glibc_candidates))
    errors = []
    if platform.system() == "Linux" and not glibc_versions:
        errors.append("Linux runtime exposes no concrete glibc version")
    if len(glibc_versions) > 1:
        errors.append("glibc probes disagree: {}".format(",".join(glibc_versions)))
    observation = {
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": list(platform.architecture()),
            "platform": platform.platform(aliased=False, terse=False),
        },
        "libc": {
            "platform_libc_name": libc_name or None,
            "platform_libc_version": libc_version or None,
            "gnu_confstr": confstr_value,
            "gnu_get_libc_version": ctypes_version,
            "resolved_glibc_versions": glibc_versions,
        },
    }
    return observation, errors


def _nested_attribute(module, attribute):
    value = module
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def _distribution_file_observation(distribution, relative_name):
    candidates = []
    for entry in distribution.files or ():
        if str(entry).replace("\\", "/").rsplit("/", 1)[-1] == relative_name:
            candidates.append(entry)
    if len(candidates) != 1:
        return None
    try:
        candidate = Path(distribution.locate_file(candidates[0]))
    except Exception:
        return None
    if candidate.is_symlink() or not candidate.is_file():
        return None
    byte_count, digest = sha256_file(candidate)
    return {
        "relative_path": str(candidates[0]).replace("\\", "/"),
        "bytes": byte_count,
        "sha256": digest,
    }


def _distribution_observations(names):
    found = []
    for name in names:
        try:
            distribution = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        metadata_name = distribution.metadata.get("Name") or name
        files = []
        for relative_name in ("METADATA", "RECORD"):
            observation = _distribution_file_observation(distribution, relative_name)
            if observation is not None:
                files.append(observation)
        found.append(
            {
                "requested_name": name,
                "metadata_name": str(metadata_name),
                "version": str(distribution.version),
                "metadata_lock_files": sorted(files, key=lambda item: item["relative_path"]),
            }
        )
    return found


def _module_origin_observation(module):
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError("imported module exposes no concrete __file__")
    supplied = Path(origin)
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("imported module origin is not a regular file")
    byte_count, digest = sha256_file(resolved)
    return {
        "supplied_path": str(supplied),
        "resolved_path": str(resolved),
        "supplied_path_is_symlink": bool(supplied.is_symlink()),
        "bytes": byte_count,
        "sha256": digest,
    }


def collect_dependency_observations(specifications):
    results = {}
    errors = []
    for specification in specifications:
        role = specification["role"]
        item = {
            "status": "error",
            "import_name": specification["import_name"],
            "module_origin": None,
            "module_versions": {},
            "distributions": [],
            "runtime_details": {},
        }
        try:
            module = importlib.import_module(specification["import_name"])
            item["module_origin"] = _module_origin_observation(module)
            for version_spec in specification["version_attributes"]:
                version_module = importlib.import_module(version_spec["module"])
                try:
                    value = _nested_attribute(version_module, version_spec["attribute"])
                except AttributeError:
                    continue
                if value is not None:
                    item["module_versions"][
                        "{}:{}".format(version_spec["module"], version_spec["attribute"])
                    ] = str(value)
            item["distributions"] = _distribution_observations(specification["distribution_names"])
            if specification.get("forbid_multiple_installed_distributions") and len(item["distributions"]) > 1:
                errors.append("dependency {} has multiple candidate distributions installed".format(role))
            if not item["module_versions"] and not item["distributions"]:
                errors.append("dependency {} exposes no exact version observation".format(role))
            if role == "torch":
                item["runtime_details"] = {
                    "cuda_build_version": str(getattr(getattr(module, "version", None), "cuda", None)),
                    "cuda_available": bool(module.cuda.is_available()),
                    "intraop_threads": int(module.get_num_threads()),
                    "interop_threads": int(module.get_num_interop_threads()),
                }
            item["status"] = "ok"
        except Exception as exc:
            item["error_type"] = type(exc).__name__
            errors.append("dependency {} import/inspection failed: {}".format(role, type(exc).__name__))
        results[role] = item
    return results, errors


def parse_cpu_set(value):
    """Parse Linux cpuset syntax and return a sorted, duplicate-free tuple."""
    if not isinstance(value, str) or not value.strip():
        return tuple()
    cpus = set()
    for token in value.strip().split(","):
        token = token.strip()
        if not token:
            raise ValueError("empty cpuset token")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError("invalid cpuset range: {}".format(token))
            first, last = (int(part) for part in parts)
            if first > last:
                raise ValueError("descending cpuset range: {}".format(token))
            cpus.update(range(first, last + 1))
        elif token.isdigit():
            cpus.add(int(token))
        else:
            raise ValueError("invalid cpuset token: {}".format(token))
    return tuple(sorted(cpus))


def encode_cpu_set(cpus):
    values = sorted(set(int(value) for value in cpus))
    if not values:
        return ""
    ranges = []
    first = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(first) if first == previous else "{}-{}".format(first, previous))
        first = previous = value
    ranges.append(str(first) if first == previous else "{}-{}".format(first, previous))
    return ",".join(ranges)


def _read_small_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(1024 * 1024).strip()
    except OSError:
        return None


def _parse_v2_cpu_max(value):
    if not value:
        return None
    parts = value.split()
    if len(parts) != 2 or not parts[1].isdigit() or int(parts[1]) <= 0:
        raise ValueError("invalid cgroup v2 cpu.max")
    if parts[0] == "max":
        return {"limited": False, "quota_us": None, "period_us": int(parts[1])}
    if not parts[0].isdigit() or int(parts[0]) <= 0:
        raise ValueError("invalid cgroup v2 CPU quota")
    return {"limited": True, "quota_us": int(parts[0]), "period_us": int(parts[1])}


def _parse_v1_cpu_quota(quota_value, period_value):
    if quota_value is None or period_value is None:
        return None
    quota = int(quota_value)
    period = int(period_value)
    if period <= 0 or quota == 0 or quota < -1:
        raise ValueError("invalid cgroup v1 CPU quota")
    return {"limited": quota > 0, "quota_us": quota if quota > 0 else None, "period_us": period}


def _cpu_model_observation():
    text = _read_small_text("/proc/cpuinfo")
    if text is None:
        return {
            "processor_count_in_proc": None,
            "model_names": [platform.processor()] if platform.processor() else [],
            "vendor_ids": [],
        }
    processors = 0
    models = set()
    vendors = set()
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "processor":
            processors += 1
        elif key in ("model name", "hardware") and value:
            models.add(value)
        elif key in ("vendor_id", "cpu implementer") and value:
            vendors.add(value)
    return {
        "processor_count_in_proc": processors,
        "model_names": sorted(models),
        "vendor_ids": sorted(vendors),
    }


def collect_cpu_observation():
    logical = os.cpu_count()
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = tuple(sorted(os.sched_getaffinity(0)))
        except OSError:
            affinity = None

    v2_max_raw = _read_small_text("/sys/fs/cgroup/cpu.max")
    v2_cpuset_raw = _read_small_text("/sys/fs/cgroup/cpuset.cpus.effective")
    v1_quota_raw = _read_small_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    v1_period_raw = _read_small_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    v1_cpuset_raw = _read_small_text("/sys/fs/cgroup/cpuset/cpuset.cpus")
    errors = []
    quota = None
    cgroup_version = None
    try:
        if v2_max_raw is not None:
            quota = _parse_v2_cpu_max(v2_max_raw)
            cgroup_version = 2
        elif v1_quota_raw is not None or v1_period_raw is not None:
            quota = _parse_v1_cpu_quota(v1_quota_raw, v1_period_raw)
            cgroup_version = 1
    except (TypeError, ValueError) as exc:
        errors.append("cannot parse CPU cgroup quota: {}".format(type(exc).__name__))

    cpuset_raw = v2_cpuset_raw if v2_cpuset_raw is not None else v1_cpuset_raw
    cpuset = None
    if cpuset_raw is not None:
        try:
            cpuset = parse_cpu_set(cpuset_raw)
        except ValueError:
            errors.append("cannot parse effective CPU cgroup cpuset")

    capacity_candidates = []
    if isinstance(logical, int) and logical > 0:
        capacity_candidates.append(("os_cpu_count", Fraction(logical, 1)))
    if affinity:
        capacity_candidates.append(("process_affinity", Fraction(len(affinity), 1)))
    if cpuset:
        capacity_candidates.append(("cgroup_cpuset", Fraction(len(cpuset), 1)))
    if quota and quota["limited"]:
        capacity_candidates.append(("cgroup_quota", Fraction(quota["quota_us"], quota["period_us"])))
    if not capacity_candidates:
        errors.append("no CPU capacity observation is available")
        effective = None
    else:
        limiting_role, limiting_value = min(capacity_candidates, key=lambda item: item[1])
        effective = {
            "limiting_role": limiting_role,
            "numerator": int(limiting_value.numerator),
            "denominator": int(limiting_value.denominator),
        }
    if platform.system() == "Linux" and cgroup_version is None:
        errors.append("Linux runtime exposes no readable CPU cgroup quota interface")

    observation = {
        "logical_cpu_count": int(logical) if isinstance(logical, int) else None,
        "process_affinity": {
            "cpu_set": encode_cpu_set(affinity) if affinity is not None else None,
            "count": len(affinity) if affinity is not None else None,
        },
        "cpu_model": _cpu_model_observation(),
        "cgroup": {
            "version": cgroup_version,
            "cpu_max_raw": v2_max_raw,
            "v1_quota_raw": v1_quota_raw,
            "v1_period_raw": v1_period_raw,
            "quota": quota,
            "cpuset_raw": cpuset_raw,
            "cpuset_count": len(cpuset) if cpuset is not None else None,
            "cpuset_normalized": encode_cpu_set(cpuset) if cpuset is not None else None,
        },
        "effective_cpu_capacity": effective,
    }
    return observation, errors


def collect_thread_environment(contract):
    thread_contract = contract["thread_environment"]
    required = thread_contract["required_exact"]
    keys = sorted(set(required) | set(thread_contract.get("also_observe", [])))
    observed = {key: os.environ.get(key) for key in keys}
    errors = []
    for key in sorted(required):
        if observed[key] != required[key]:
            errors.append("thread environment {} must equal {} (observed {})".format(key, required[key], observed[key]))
    return {"required_exact": dict(sorted(required.items())), "observed": observed}, errors


def resolve_e3fp_package_root(path):
    supplied = require_nonsymlink_directory(path, "E3FP source")
    if (supplied / "e3fp" / "pipeline.py").is_file():
        package_root = supplied / "e3fp"
    elif supplied.name == "e3fp" and (supplied / "pipeline.py").is_file():
        package_root = supplied
    else:
        raise FileNotFoundError("E3FP source must be a 3d_tokenization root or its e3fp package")
    if package_root.is_symlink():
        raise RuntimeError("E3FP package root must not be a symlink")
    return package_root.resolve(strict=True)


def collect_e3fp_source_closure(source_path, contract):
    package_root = resolve_e3fp_package_root(source_path)
    policy = contract["e3fp_source_closure"]
    excluded_directories = set(policy["generated_directory_exclusions"])
    excluded_suffixes = tuple(policy["generated_file_suffix_exclusions"])
    observations = []
    excluded_generated = []
    for current_root, directory_names, file_names in os.walk(str(package_root), topdown=True, followlinks=False):
        current = Path(current_root)
        kept_directories = []
        for directory_name in sorted(directory_names):
            candidate = current / directory_name
            relative = candidate.relative_to(package_root).as_posix()
            if candidate.is_symlink():
                raise RuntimeError("symlink in E3FP source closure: {}".format(relative))
            if directory_name in excluded_directories:
                excluded_generated.append(relative + "/")
            else:
                kept_directories.append(directory_name)
        directory_names[:] = kept_directories
        for file_name in sorted(file_names):
            candidate = current / file_name
            relative = candidate.relative_to(package_root).as_posix()
            if candidate.is_symlink():
                raise RuntimeError("symlink in E3FP source closure: {}".format(relative))
            if candidate.suffix in excluded_suffixes:
                excluded_generated.append(relative)
                continue
            if not candidate.is_file():
                raise RuntimeError("non-regular path in E3FP source closure: {}".format(relative))
            byte_count, digest = sha256_file(candidate)
            observations.append({"relative_path": relative, "bytes": byte_count, "sha256": digest})
    observations.sort(key=lambda item: item["relative_path"])
    observed_names = {item["relative_path"] for item in observations}
    missing = sorted(set(policy["required_relative_files"]) - observed_names)
    if missing:
        raise RuntimeError("E3FP source closure lacks required anchors: {}".format(",".join(missing)))
    if not observations:
        raise RuntimeError("E3FP source closure is empty")
    return {
        "package_root": str(package_root),
        "file_count": len(observations),
        "total_bytes": sum(item["bytes"] for item in observations),
        "files": observations,
        "closure_sha256": sha256_bytes(canonical_json_bytes(observations)),
        "excluded_generated_paths": sorted(excluded_generated),
        "exclusion_policy": {
            "directories": sorted(excluded_directories),
            "file_suffixes": list(excluded_suffixes),
        },
    }


def _canonical_bundle_relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("bundle paths must be non-empty forward-slash paths")
    pure = PurePosixPath(value)
    parts = value.split("/")
    if pure.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError("bundle paths must be normalized relative paths")
    if ":" in parts[0]:
        raise ValueError("bundle paths must not contain a drive prefix")
    return pure.as_posix()


def _reject_symlink_components(root, relative):
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError("bundle path contains a symlink: {}".format(relative))


def collect_bundle_file_lock(bundle_root_path, relative_paths, contract):
    root = require_nonsymlink_directory(bundle_root_path, "bundle root")
    minimum = int(contract["bundle_lock"]["minimum_explicit_files"])
    if len(relative_paths) < minimum:
        raise RuntimeError("at least {} explicit bundle file is required".format(minimum))
    canonical_paths = [_canonical_bundle_relative_path(value) for value in relative_paths]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise RuntimeError("duplicate explicit bundle file path")
    observations = []
    for relative in sorted(canonical_paths):
        _reject_symlink_components(root, relative)
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("bundle file escapes the declared bundle root: {}".format(relative)) from exc
        if not candidate.is_file():
            raise FileNotFoundError("bundle path is not a regular file: {}".format(relative))
        byte_count, digest = sha256_file(candidate)
        observations.append({"relative_path": relative, "bytes": byte_count, "sha256": digest})
    return {
        "bundle_root": str(root),
        "file_count": len(observations),
        "total_bytes": sum(item["bytes"] for item in observations),
        "files": observations,
        "closure_sha256": sha256_bytes(canonical_json_bytes(observations)),
    }


def build_attestation(contract_path, e3fp_source, bundle_root, bundle_files, created_utc=None):
    _, contract, contract_observation = load_contract(contract_path)
    errors = []
    python_observation = collect_python_observation()
    platform_observation, platform_errors = collect_platform_observation()
    dependencies, dependency_errors = collect_dependency_observations(contract["dependencies"])
    cpu, cpu_errors = collect_cpu_observation()
    thread_environment, thread_errors = collect_thread_environment(contract)
    errors.extend(platform_errors)
    errors.extend(dependency_errors)
    errors.extend(cpu_errors)
    errors.extend(thread_errors)
    e3fp_closure = collect_e3fp_source_closure(e3fp_source, contract)
    bundle_lock = collect_bundle_file_lock(bundle_root, bundle_files, contract)

    report = {
        "schema_version": ATTESTATION_SCHEMA,
        "created_utc": created_utc or utc_now(),
        "contract": contract_observation,
        "runtime": {
            "python": python_observation,
            "platform": platform_observation,
            "dependencies": dependencies,
            "cpu": cpu,
            "thread_environment": thread_environment,
        },
        "e3fp_source_closure": e3fp_closure,
        "bundle_file_lock": bundle_lock,
        "pass": not errors,
        "errors": sorted(set(errors)),
        "interpretation": {
            "pass_meaning": "All required runtime observations and exact transferred-code hashes were captured under the v1 contract.",
            "non_claim": "This attestation alone proves neither cross-host numerical equivalence nor PCQM/P1 admission.",
            "next_gate": "Reproduce the frozen 128-record golden result before any full CPU release run.",
        },
    }
    report["attestation_payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def default_contract_path():
    return Path(__file__).resolve().parents[1] / "contracts" / "cpu_runtime_attestation_contract.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(default_contract_path()))
    parser.add_argument("--e3fp-source", required=True, help="3d_tokenization root or its e3fp child")
    parser.add_argument("--bundle-root", required=True, help="root of the transferred CPU bundle")
    parser.add_argument(
        "--bundle-file",
        action="append",
        required=True,
        help="one explicit forward-slash relative file below --bundle-root; repeat for every transferred file",
    )
    parser.add_argument("--output", required=True, help="new canonical JSON attestation path")
    args = parser.parse_args(argv)

    report = build_attestation(
        contract_path=args.contract,
        e3fp_source=args.e3fp_source,
        bundle_root=args.bundle_root,
        bundle_files=args.bundle_file,
    )
    output = write_new_canonical_json(args.output, report)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "output": str(output),
                "attestation_payload_sha256": report["attestation_payload_sha256"],
                "e3fp_closure_sha256": report["e3fp_source_closure"]["closure_sha256"],
                "bundle_closure_sha256": report["bundle_file_lock"]["closure_sha256"],
                "errors": report["errors"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
