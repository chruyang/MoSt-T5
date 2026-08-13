"""Merge contiguous fragSMILES census shards without re-running chemistry."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Iterable, Iterator, TextIO

from most_t5_next.p1.build_fragsmiles_fragment_census_v1 import SCHEMA_VERSION


class FragSmilesCensusMergeError(RuntimeError):
    """Shard artifacts do not form one exact, contiguous census."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


@contextmanager
def _deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline="\n"
            ) as text:
                yield text


def _load_shard(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FragSmilesCensusMergeError(f"shard manifest is absent: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise FragSmilesCensusMergeError("shard schema differs")
    record_range = manifest.get("selected_record_range")
    if not isinstance(record_range, dict):
        raise FragSmilesCensusMergeError("shard record range is absent")
    start = record_range.get("start_inclusive")
    stop = record_range.get("stop_exclusive")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise FragSmilesCensusMergeError("shard record range is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FragSmilesCensusMergeError("shard artifact lock is absent")
    for name in ("fragment_census.jsonl", "molecule_fragments.jsonl.gz", "rejects.jsonl"):
        path = root / name
        lock = artifacts.get(name)
        if (
            not path.is_file()
            or not isinstance(lock, dict)
            or lock.get("bytes") != path.stat().st_size
            or lock.get("sha256") != _sha256_file(path)
        ):
            raise FragSmilesCensusMergeError(f"shard artifact differs: {path}")
    return manifest


def _counter_from_registry(path: Path) -> tuple[Counter[str], Counter[str]]:
    occurrences: Counter[str] = Counter()
    molecules: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for expected_rank, line in enumerate(handle):
            row = json.loads(line)
            identity = row.get("fragment_identity")
            if (
                row.get("rank") != expected_rank
                or not isinstance(identity, str)
                or not identity
                or row.get("fragment_identity_sha256")
                != hashlib.sha256(identity.encode("utf-8")).hexdigest()
            ):
                raise FragSmilesCensusMergeError("shard registry row is invalid")
            occurrences[identity] = int(row["occurrences"])
            molecules[identity] = int(row["molecule_occurrences"])
    return occurrences, molecules


def merge_census_shards(
    *,
    shard_dirs: Iterable[Path],
    output_dir: Path,
    expected_records: int,
) -> dict[str, object]:
    roots = tuple(Path(path).resolve() for path in shard_dirs)
    if len(roots) < 2:
        raise FragSmilesCensusMergeError("at least two shards are required")
    if expected_records <= 0:
        raise ValueError("expected_records must be positive")
    output_dir = output_dir.resolve()
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise FileExistsError(output_dir if output_dir.exists() else staging)

    loaded = [(root, _load_shard(root)) for root in roots]
    loaded.sort(key=lambda item: item[1]["selected_record_range"]["start_inclusive"])
    first_manifest = loaded[0][1]
    cursor = 0
    for _root, manifest in loaded:
        record_range = manifest["selected_record_range"]
        start = record_range["start_inclusive"]
        stop = record_range["stop_exclusive"]
        if start != cursor:
            raise FragSmilesCensusMergeError("shard ranges are not contiguous")
        cursor = stop
        for key in ("source", "membership", "contracts", "schema_version"):
            if manifest.get(key) != first_manifest.get(key):
                raise FragSmilesCensusMergeError(f"shard {key} contract differs")
    if cursor != expected_records:
        raise FragSmilesCensusMergeError("shards do not cover expected records")

    staging.mkdir(parents=False)
    try:
        merged_occurrences: Counter[str] = Counter()
        merged_molecules: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()
        fallback_reasons: Counter[str] = Counter()
        projection_modes: Counter[str] = Counter()
        total_processed = total_rejected = total_ineligible = 0
        shard_rows: list[dict[str, object]] = []
        cache_path = staging / "molecule_fragments.jsonl.gz"
        rejects_path = staging / "rejects.jsonl"
        with _deterministic_gzip_text(cache_path) as cache_out, rejects_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as rejects_out:
            for root, manifest in loaded:
                record_range = manifest["selected_record_range"]
                start = int(record_range["start_inclusive"])
                stop = int(record_range["stop_exclusive"])
                observed = bytearray(stop - start)
                local_occurrences: Counter[str] = Counter()
                local_molecules: Counter[str] = Counter()
                local_modes: Counter[str] = Counter()
                local_projection_modes: Counter[str] = Counter()
                local_ineligible = 0
                pass_rows = 0
                last_selection = start - 1
                with gzip.open(
                    root / "molecule_fragments.jsonl.gz", "rt", encoding="utf-8"
                ) as handle:
                    for line in handle:
                        row = json.loads(line)
                        selection = row.get("selection_index")
                        if (
                            isinstance(selection, bool)
                            or not isinstance(selection, int)
                            or not start <= selection < stop
                            or selection <= last_selection
                        ):
                            raise FragSmilesCensusMergeError("cache order/range differs")
                        last_selection = selection
                        offset = selection - start
                        if observed[offset]:
                            raise FragSmilesCensusMergeError("duplicate shard selection")
                        observed[offset] = 1
                        identities = tuple(row.get("fragment_identities", ()))
                        eligible = tuple(row.get("fragment_macro_eligible", ()))
                        if len(identities) != len(eligible):
                            raise FragSmilesCensusMergeError("cache eligibility differs")
                        admitted = tuple(
                            identity
                            for identity, keep in zip(identities, eligible)
                            if keep
                        )
                        local_occurrences.update(admitted)
                        local_molecules.update(set(admitted))
                        local_ineligible += len(identities) - len(admitted)
                        local_modes[str(row["mode"])] += 1
                        local_projection_modes[str(row["projection_mode"])] += 1
                        pass_rows += 1
                        cache_out.write(
                            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                        )

                reject_rows = 0
                last_reject = start - 1
                with (root / "rejects.jsonl").open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        selection = row.get("selection_index")
                        if (
                            isinstance(selection, bool)
                            or not isinstance(selection, int)
                            or not start <= selection < stop
                            or selection <= last_reject
                        ):
                            raise FragSmilesCensusMergeError("reject order/range differs")
                        last_reject = selection
                        offset = selection - start
                        if observed[offset]:
                            raise FragSmilesCensusMergeError("selection is pass and reject")
                        observed[offset] = 1
                        reject_rows += 1
                        rejects_out.write(
                            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                        )
                if not all(observed):
                    raise FragSmilesCensusMergeError("shard selection coverage has gaps")

                counts = manifest.get("counts", {})
                registry_occurrences, registry_molecules = _counter_from_registry(
                    root / "fragment_census.jsonl"
                )
                if (
                    pass_rows + reject_rows != stop - start
                    or counts.get("processed_records") != stop - start
                    or counts.get("admitted_records") != pass_rows
                    or counts.get("rejected_records") != reject_rows
                    or counts.get("modes") != dict(sorted(local_modes.items()))
                    or counts.get("projection_modes")
                    != dict(sorted(local_projection_modes.items()))
                    or counts.get("fragment_occurrences") != sum(local_occurrences.values())
                    or counts.get("unique_fragment_identities") != len(local_occurrences)
                    or counts.get(
                        "noncanonical_fragment_surfaces_forced_to_semantic_fallback"
                    )
                    != local_ineligible
                    or registry_occurrences != local_occurrences
                    or registry_molecules != local_molecules
                ):
                    raise FragSmilesCensusMergeError("shard manifest/counts differ")
                merged_occurrences.update(local_occurrences)
                merged_molecules.update(local_molecules)
                mode_counts.update(local_modes)
                projection_modes.update(local_projection_modes)
                fallback_reasons.update(counts.get("fallback_reasons", {}))
                total_processed += stop - start
                total_rejected += reject_rows
                total_ineligible += local_ineligible
                shard_rows.append(
                    {
                        "path": str(root),
                        "manifest_sha256": _sha256_file(root / "manifest.json"),
                        "selected_record_range": record_range,
                        "status": manifest.get("status"),
                        "runtime": manifest.get("runtime"),
                    }
                )

        registry_path = staging / "fragment_census.jsonl"
        ranking = sorted(
            merged_occurrences,
            key=lambda identity: (-merged_occurrences[identity], identity.encode("utf-8")),
        )
        with registry_path.open("x", encoding="utf-8", newline="\n") as handle:
            for rank, identity in enumerate(ranking):
                handle.write(
                    json.dumps(
                        {
                            "rank": rank,
                            "fragment_identity": identity,
                            "fragment_identity_sha256": hashlib.sha256(
                                identity.encode("utf-8")
                            ).hexdigest(),
                            "occurrences": merged_occurrences[identity],
                            "molecule_occurrences": merged_molecules[identity],
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass" if total_rejected == 0 else "failed",
            "training_admission": False,
            "source": first_manifest["source"],
            "membership": first_manifest.get("membership"),
            "selected_record_range": {
                "start_inclusive": 0,
                "stop_exclusive": total_processed,
            },
            "counts": {
                "processed_records": total_processed,
                "admitted_records": total_processed - total_rejected,
                "rejected_records": total_rejected,
                "modes": dict(sorted(mode_counts.items())),
                "fallback_reasons": dict(sorted(fallback_reasons.items())),
                "projection_modes": dict(sorted(projection_modes.items())),
                "fragment_occurrences": sum(merged_occurrences.values()),
                "unique_fragment_identities": len(merged_occurrences),
                "noncanonical_fragment_surfaces_forced_to_semantic_fallback": total_ineligible,
            },
            "runtime": {
                "mode": "merge_contiguous_completed_shards",
                "shard_count": len(shard_rows),
                "sum_shard_wall_seconds": sum(
                    float(row["runtime"]["wall_seconds"]) for row in shard_rows
                ),
            },
            "contracts": {
                **first_manifest["contracts"],
                "merged_from_contiguous_completed_shards": True,
            },
            "shards": shard_rows,
            "artifacts": {
                path.name: _artifact(path)
                for path in (registry_path, cache_path, rejects_path)
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = merge_census_shards(
        shard_dirs=[Path(path) for path in args.shard_dir],
        output_dir=Path(args.output_dir),
        expected_records=args.expected_records,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FragSmilesCensusMergeError", "merge_census_shards"]
