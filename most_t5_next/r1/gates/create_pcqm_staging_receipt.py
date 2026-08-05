#!/usr/bin/env python3
"""Create and immediately verify a cross-region PCQM4Mv2 staging receipt.

Run this on the CPU worker after all eight canonical roles have been copied.
The command records the actual Linux CPU/runtime environment, obtains package
versions without importing heavy packages when distribution metadata is
available, and delegates all provenance and file checks to
``adapter/pcqm_staging_receipt.py``.
"""

from __future__ import print_function

import argparse
import ast
import importlib
import importlib.util
import json
import os
import platform
import socket
import sys
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python 3.7 compatibility only
    import importlib_metadata


R1_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = R1_ROOT / "adapter" / "pcqm_staging_receipt.py"
DEFAULT_CONTRACT = R1_ROOT / "contracts" / "pcqm4mv2_staging_receipt_contract.json"
DEFAULT_SOURCE_CONTRACT = R1_ROOT / "contracts" / "pcqm4mv2_source_contract.json"

ROLE_FLAGS = (
    ("train_3d_sdf_archive", "--train-3d-sdf-archive"),
    ("companion_archive", "--companion-archive"),
    ("companion_data_csv_gz", "--companion-data-csv-gz"),
    ("companion_split_dict_pt", "--companion-split-dict-pt"),
    ("train_sdf_source_manifest", "--train-sdf-source-manifest"),
    ("train_sdf_member_hash_report", "--train-sdf-member-hash-report"),
    ("companion_source_manifest", "--companion-source-manifest"),
    ("companion_content_validation", "--companion-content-validation"),
)
TRANSFER_METHODS = (
    "rsync_over_ssh",
    "scp_over_ssh",
    "official_redownload",
    "object_storage_transfer",
)
THREAD_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
_VERSION_TOKEN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,63}$")


def import_adapter():
    spec = importlib.util.spec_from_file_location("r1_pcqm_staging_receipt_create", str(ADAPTER_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import staging receipt adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file() and not cpuinfo.is_symlink():
        try:
            with cpuinfo.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    key, separator, value = line.partition(":")
                    if separator and key.strip().lower() in ("model name", "hardware"):
                        if value.strip():
                            return value.strip()
        except OSError:
            pass
    value = platform.processor().strip()
    if not value:
        raise RuntimeError("cannot determine CPU model")
    return value


def _read_memory_bytes():
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file() and not meminfo.is_symlink():
        try:
            with meminfo.open("r", encoding="ascii", errors="strict") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        fields = line.split()
                        if len(fields) == 3 and fields[2] == "kB":
                            value = int(fields[1]) * 1024
                            if value > 0:
                                return value
        except (OSError, ValueError, UnicodeError):
            pass
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        value = pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        value = 0
    if value <= 0:
        raise RuntimeError("cannot determine positive physical memory bytes")
    return value


def _installed_version(distributions, module_names=()):
    for distribution in distributions:
        try:
            value = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attribute in ("__version__", "version"):
            value = getattr(module, attribute, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise RuntimeError(
        "cannot determine installed version for {}".format("/".join(distributions))
    )


def _resolve_e3fp_package_root(source_path):
    supplied = Path(source_path).expanduser()
    if not supplied.is_absolute():
        raise RuntimeError("e3fp_source must be an absolute path")
    if supplied.is_symlink() or not supplied.is_dir():
        raise RuntimeError("e3fp_source must be an existing non-symlink directory")
    resolved = supplied.resolve(strict=True)
    if supplied != resolved:
        raise RuntimeError("e3fp_source must be a canonical path without symlink components")
    if (resolved / "e3fp" / "pipeline.py").is_file():
        package_root = resolved / "e3fp"
    elif resolved.name == "e3fp" and (resolved / "pipeline.py").is_file():
        package_root = resolved
    else:
        raise RuntimeError("e3fp_source must be a 3d_tokenization root or its e3fp directory")
    if package_root.is_symlink() or package_root.resolve(strict=True) != package_root:
        raise RuntimeError("vendored e3fp package root must not use symlink components")
    return package_root


def _vendored_e3fp_version(source_path):
    """Read vendored ``__version__`` statically without executing source code."""
    package_root = _resolve_e3fp_package_root(source_path)
    init_path = package_root / "__init__.py"
    if init_path.is_symlink() or not init_path.is_file():
        raise RuntimeError("vendored e3fp __init__.py must be a regular non-symlink file")
    before = init_path.stat()
    if before.st_size <= 0 or before.st_size > 65536:
        raise RuntimeError("vendored e3fp __init__.py has an invalid byte count")
    raw = init_path.read_bytes()
    after = init_path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("vendored e3fp __init__.py changed while being read")
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(init_path))
    except (SyntaxError, UnicodeError) as exc:
        raise RuntimeError("cannot safely parse vendored e3fp __init__.py") from exc

    version_info = None
    direct_version = None
    version_alias = None
    has_version_join = False
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if name == "version_info":
            try:
                candidate = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                candidate = None
            if isinstance(candidate, (tuple, list)) and candidate:
                if all(isinstance(item, (int, str)) and not isinstance(item, bool) for item in candidate):
                    version_info = tuple(str(item) for item in candidate)
        elif name == "version":
            try:
                candidate = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                candidate = None
            if isinstance(candidate, str):
                direct_version = candidate
            elif isinstance(statement.value, ast.Call):
                # The locked E3FP source uses: ".".join(str(c) for c in version_info).
                function = statement.value.func
                has_version_join = (
                    isinstance(function, ast.Attribute)
                    and function.attr == "join"
                    and isinstance(function.value, ast.Constant)
                    and function.value.value == "."
                )
        elif name == "__version__":
            try:
                candidate = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                candidate = None
            if isinstance(candidate, str):
                direct_version = candidate
                version_alias = "literal"
            elif isinstance(statement.value, ast.Name):
                version_alias = statement.value.id

    if version_alias == "literal":
        version = direct_version
    elif version_alias == "version" and direct_version is not None:
        version = direct_version
    elif version_alias == "version" and has_version_join and version_info is not None:
        version = ".".join(version_info)
    elif version_alias == "version_info" and version_info is not None:
        version = ".".join(version_info)
    else:
        raise RuntimeError("vendored e3fp __version__ is not statically resolvable")
    if not isinstance(version, str) or not _VERSION_TOKEN.fullmatch(version):
        raise RuntimeError("vendored e3fp __version__ is invalid")
    return version


def collect_cpu_runtime(working_root, e3fp_source, captured_at_utc=None):
    """Collect the exact runtime fields required by the staging contract."""
    if platform.system() != "Linux":
        raise RuntimeError("staging receipt creation must run on the Linux CPU worker")
    root = Path(working_root).expanduser()
    if not root.is_absolute():
        raise RuntimeError("working_root must be absolute")
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("working_root must be an existing non-symlink directory")
    root = root.resolve(strict=True)

    environment = {}
    for key in THREAD_ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if value is None or not value:
            raise RuntimeError("required CPU environment variable {} is unset".format(key))
        environment[key] = value
    if environment["CUDA_VISIBLE_DEVICES"] != "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal -1 for CPU staging")

    logical_cpu_count = os.cpu_count()
    if not isinstance(logical_cpu_count, int) or logical_cpu_count <= 0:
        raise RuntimeError("cannot determine logical CPU count")
    try:
        affinity_cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpu_count = logical_cpu_count
    if affinity_cpu_count <= 0:
        raise RuntimeError("CPU affinity set is empty")

    packages = {
        "python": platform.python_version(),
        "numpy": _installed_version(("numpy",), ("numpy",)),
        "scipy": _installed_version(("scipy",), ("scipy",)),
        "rdkit": _installed_version(("rdkit", "rdkit-pypi"), ("rdkit",)),
        "e3fp": _vendored_e3fp_version(e3fp_source),
        "lmdb": _installed_version(("lmdb",), ("lmdb",)),
        "torch": _installed_version(("torch",), ("torch",)),
    }
    return {
        "captured_at_utc": captured_at_utc or utc_now(),
        "hostname": socket.gethostname(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "cpu_model": _read_cpu_model(),
        "logical_cpu_count": logical_cpu_count,
        "affinity_cpu_count": affinity_cpu_count,
        "memory_bytes": _read_memory_bytes(),
        "working_root": str(root),
        "environment": environment,
        "packages": packages,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--source-contract", default=str(DEFAULT_SOURCE_CONTRACT))
    parser.add_argument(
        "--e3fp-source",
        required=True,
        help="absolute 3d_tokenization root or its vendored e3fp directory",
    )
    parser.add_argument("--output", required=True, help="absolute output receipt JSON path")
    parser.add_argument(
        "--working-root",
        help="CPU staging root; defaults to the common parent of all eight work paths",
    )
    for role, flag in ROLE_FLAGS:
        parser.add_argument(flag, dest=role, required=True)
    parser.add_argument("--transfer-id", required=True)
    parser.add_argument("--transfer-method", required=True, choices=TRANSFER_METHODS)
    parser.add_argument("--source-endpoint", required=True)
    parser.add_argument("--destination-endpoint", required=True)
    parser.add_argument("--transfer-started-at-utc", required=True)
    parser.add_argument("--transfer-completed-at-utc", required=True)
    parser.add_argument(
        "--receipt-created-at-utc",
        help="test/replay override; defaults to the current UTC second",
    )
    return parser.parse_args(argv)


def _infer_working_root(work_paths):
    parent_paths = [str(Path(path).expanduser().resolve().parent) for path in work_paths.values()]
    try:
        common = os.path.commonpath(parent_paths)
    except ValueError as exc:
        raise RuntimeError("the eight work paths have no common filesystem root") from exc
    return Path(common).resolve()


def main(argv=None):
    args = parse_args(argv)
    adapter = import_adapter()
    try:
        work_paths = {
            role: str(Path(getattr(args, role)).expanduser().resolve())
            for role, _ in ROLE_FLAGS
        }
        working_root = (
            Path(args.working_root).expanduser().resolve()
            if args.working_root
            else _infer_working_root(work_paths)
        )
        receipt_created_at_utc = args.receipt_created_at_utc or utc_now()
        runtime = collect_cpu_runtime(
            working_root,
            args.e3fp_source,
            captured_at_utc=receipt_created_at_utc,
        )
        transfer = {
            "transfer_id": args.transfer_id,
            "method": args.transfer_method,
            "source_endpoint": args.source_endpoint,
            "destination_endpoint": args.destination_endpoint,
            "started_at_utc": args.transfer_started_at_utc,
            "completed_at_utc": args.transfer_completed_at_utc,
            "status": "completed",
            "verification": "post_transfer_bytes_and_sha256",
        }
        verified = adapter.generate_and_verify_staging_receipt(
            Path(args.contract).expanduser().resolve(),
            Path(args.source_contract).expanduser().resolve(),
            Path(args.output).expanduser().resolve(),
            work_paths,
            transfer,
            runtime,
            receipt_created_at_utc,
        )
    except Exception as exc:
        failure = {
            "schema_version": getattr(
                adapter,
                "VERIFICATION_REPORT_SCHEMA",
                "most-t5-r1/pcqm4mv2-staging-verification/v1",
            ),
            "pass": False,
            "p1_training_admitted": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(adapter.canonical_json_bytes(verified.report()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
