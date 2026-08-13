"""Stream a chemistry domain through the compact fragSMILES stereo codec.

This is an offline CPU preflight, not a training data loader.  It supports the
source formats currently needed by the project and keeps only a bounded number
of records in flight while worker processes perform strict codec round trips.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from itertools import islice
import multiprocessing
from pathlib import Path
import platform
import signal
import tarfile
import time
from typing import Iterator

from rdkit import Chem, rdBase

from most_t5_next.p1.fragsmiles_compact_stereo_codec_v1 import strict_round_trip


SCHEMA_VERSION = "most-t5-next/fragsmiles-compact-stereo-domain-audit/v1"


def _iter_jsonl(path: Path, field: str) -> Iterator[tuple[int, str | None, str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                value = json.loads(line)[field]
                if not isinstance(value, str) or not value:
                    raise ValueError("SMILES field is not a non-empty string")
                yield index, value, None
            except Exception as exc:
                yield index, None, f"source JSONL parse: {type(exc).__name__}: {exc}"


def _iter_parquet(path: Path, field: str) -> Iterator[tuple[int, str | None, str | None]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise RuntimeError("parquet input requires pyarrow") from exc
    source_index = 0
    for batch in parquet.ParquetFile(path).iter_batches(columns=[field], batch_size=4096):
        for value in batch.column(0).to_pylist():
            if isinstance(value, str) and value:
                yield source_index, value, None
            else:
                yield source_index, None, "source Parquet SMILES is empty"
            source_index += 1


def _iter_sdf_supplier(supplier) -> Iterator[tuple[int, str | None, str | None]]:
    for index, mol in enumerate(supplier):
        if mol is None:
            yield index, None, "source SDF parse failure"
            continue
        try:
            yield index, Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            ), None
        except Exception as exc:
            yield index, None, f"source SDF normalization: {type(exc).__name__}: {exc}"


def _iter_sdf(path: Path) -> Iterator[tuple[int, str | None, str | None]]:
    yield from _iter_sdf_supplier(Chem.SDMolSupplier(str(path), removeHs=False))


def _iter_sdf_tar(path: Path) -> Iterator[tuple[int, str | None, str | None]]:
    # Stream mode is essential here.  ``getmembers()`` on a compressed tar
    # consumes the complete multi-gigabyte SDF merely to build an index before
    # the first molecule can be read.
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.name.endswith(".sdf"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("could not open SDF archive member")
            yield from _iter_sdf_supplier(
                Chem.ForwardSDMolSupplier(handle, removeHs=False)
            )
            return
    raise RuntimeError("SDF archive contains no .sdf member")


def _iter_source(
    path: Path, input_format: str, field: str
) -> Iterator[tuple[int, str | None, str | None]]:
    if input_format == "jsonl":
        yield from _iter_jsonl(path, field)
    elif input_format == "parquet":
        yield from _iter_parquet(path, field)
    elif input_format == "sdf":
        yield from _iter_sdf(path)
    elif input_format == "sdf-tar":
        yield from _iter_sdf_tar(path)
    else:  # pragma: no cover - argparse closes this
        raise RuntimeError(f"unsupported input format: {input_format}")


def _audit_one(task: tuple[int, str | None, str | None, str, int | None]) -> dict:
    index, smiles, source_error, chemicalgof_root, timeout_seconds = task
    if source_error is not None:
        return {"source_index": index, "status": "reject", "error": source_error}
    previous_handler = None
    if timeout_seconds is not None and hasattr(signal, "SIGALRM"):
        def _timeout(_signum, _frame):
            raise TimeoutError(
                f"compact stereo codec exceeded {timeout_seconds} seconds"
            )

        previous_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(timeout_seconds)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit MolFromSmiles returned None")
        surface = strict_round_trip(
            mol, chemicalgof_root=Path(chemicalgof_root)
        )
        return {
            "source_index": index,
            "status": "pass",
            "atom_records": len(surface.atom_records),
            "bond_records": len(surface.bond_records),
            "extra_tokens": len(surface.tokens) - len(surface.connectivity_record.tokens),
            "surface_tokens": len(surface.tokens),
            "fragments": len(surface.connectivity_record.fragments),
        }
    except Exception as exc:
        return {
            "source_index": index,
            "status": "reject",
            "smiles": smiles,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if previous_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)


def _bounded_results(
    tasks: Iterator[tuple[int, str | None, str | None, str, int | None]],
    *,
    workers: int,
    max_pending: int,
) -> Iterator[dict]:
    if workers == 1:
        for task in tasks:
            yield _audit_one(task)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    pending.add(pool.submit(_audit_one, next(tasks)))
                except StopIteration:
                    exhausted = True
            if pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()


def run_audit(
    *,
    input_path: Path,
    input_format: str,
    smiles_field: str,
    chemicalgof_root: Path,
    output_dir: Path,
    workers: int,
    max_pending: int,
    max_records: int | None,
    progress_every: int,
    record_timeout_seconds: int | None,
) -> dict:
    if workers <= 0 or max_pending < workers:
        raise ValueError("workers must be positive and max_pending >= workers")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    if record_timeout_seconds is not None and record_timeout_seconds <= 0:
        raise ValueError("record_timeout_seconds must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    source = _iter_source(input_path, input_format, smiles_field)
    if max_records is not None:
        source = islice(source, max_records)
    root_text = str(chemicalgof_root.resolve())
    tasks = (
        (index, smiles, error, root_text, record_timeout_seconds)
        for index, smiles, error in source
    )

    counts = Counter()
    sums = Counter()
    maxima = Counter()
    start = time.time()
    rejects_path = output_dir / "rejects.jsonl"
    with rejects_path.open("w", encoding="utf-8") as rejects:
        for processed, result in enumerate(
            _bounded_results(tasks, workers=workers, max_pending=max_pending),
            start=1,
        ):
            counts["input"] += 1
            if result["status"] == "pass":
                counts["pass"] += 1
                for field in (
                    "atom_records",
                    "bond_records",
                    "extra_tokens",
                    "surface_tokens",
                    "fragments",
                ):
                    value = int(result[field])
                    sums[field] += value
                    maxima[field] = max(maxima[field], value)
            else:
                counts["reject"] += 1
                error_key = result.get("error_type", "source_parse_failure")
                counts[f"reject:{error_key}"] += 1
                rejects.write(json.dumps(result, sort_keys=True) + "\n")
            if progress_every and processed % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "pass": counts["pass"],
                            "reject": counts["reject"],
                            "wall_seconds": round(time.time() - start, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    passed = counts["pass"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if counts["reject"] == 0 else "completed_with_rejects",
        "training_admission": False,
        "source": {
            "path": str(input_path.resolve()),
            "format": input_format,
            "smiles_field": smiles_field,
            "size_bytes": input_path.stat().st_size,
            "max_records": max_records,
        },
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "workers": workers,
            "max_pending": max_pending,
            "multiprocessing_start_method": "spawn",
            "result_collection_order": "worker_completion; source_index retained",
            "record_timeout_seconds": record_timeout_seconds,
            "wall_seconds": time.time() - start,
        },
        "counts": dict(counts),
        "sums": dict(sums),
        "means_on_pass": {
            field: (sums[field] / passed if passed else None)
            for field in sums
        },
        "maxima_on_pass": dict(maxima),
        "artifacts": {"rejects": rejects_path.name},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--input-format", choices=("jsonl", "parquet", "sdf", "sdf-tar"), required=True
    )
    parser.add_argument("--smiles-field", default="smiles")
    parser.add_argument("--chemicalgof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--record-timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    report = run_audit(
        input_path=args.input,
        input_format=args.input_format,
        smiles_field=args.smiles_field,
        chemicalgof_root=args.chemicalgof_root,
        output_dir=args.output_dir,
        workers=args.workers,
        max_pending=args.max_pending,
        max_records=args.max_records,
        progress_every=args.progress_every,
        record_timeout_seconds=args.record_timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
