#!/usr/bin/env python3
"""Benchmark PF-1 chemistry materialization on the frozen ~1024-member prefix.

This is a bounded performance run, not a training release.  It opens the
production geometry release read-only, scans the original SDF archive once and
executes the real PF-1 chemistry path for every benchmark member:

``hydrogen projection -> explicit inherited E3FP -> motif linearization ->
atom SELFIES and graph/ports surface discovery``.

Workers return only compact counts and phase timings.  No molecule payload,
overlay or paired record is published by this benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from most_t5_next.p1 import freeze_pf1_connectivity_sample_v1 as selection
from most_t5_next.r1.adapter import build_p1_inherited_e3fp_overlay_v1 as overlay
from most_t5_next.r1.adapter import build_p1_paired_canary_v1 as paired_builder
from most_t5_next.r1.adapter import build_pcqm_p1_geometry_production_v1 as production_builder
from most_t5_next.r1.adapter import graphports_donor_atom_map_sidecar_v1 as donor_atom_map
from most_t5_next.r1.adapter import mol_linearizer
from most_t5_next.r1.adapter import p1_topology_augmentation_v1 as topology
from most_t5_next.r1.adapter import production_paired_identity_records_v1 as paired
from most_t5_next.r1.adapter import run_p1_topology_canary_v1 as release_reader
from most_t5_next.r1.gates import pcqm_e3fp_preflight as projection
from most_t5_next.r1.semantic import e3fp_duplicate_inheritance_v1 as inheritance


SCHEMA_VERSION = "most-t5-p1/pf1-materialization-benchmark-manifest/v1"
DEFAULT_WORKERS = 8
DEFAULT_MAX_PENDING = 24
PF1_TARGET_MEMBERS = selection.TARGET_MEMBERS
DONOR_ATOM_MAP_NAME = "donor_atom_maps.jsonl"
_WORKER_STATE: dict[str, Any] = {}


class PF1MaterializationBenchmarkError(RuntimeError):
    """The bounded materialization benchmark did not complete exactly."""


def load_benchmark_membership(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    seen_ordinals: set[int] = set()
    seen_members: set[str] = set()
    for line_number, row in selection._read_jsonl(path):
        required = {
            "schema_version",
            "benchmark_index",
            "pf1_selection_index",
            "group_order_index",
            "member_id",
            "sdf_record_index",
            "connectivity_identity_sha256",
            "split",
        }
        if row.get("schema_version") != selection.BENCHMARK_SCHEMA or set(row) != required:
            raise PF1MaterializationBenchmarkError(
                f"benchmark membership line {line_number} has an unexpected schema"
            )
        index = row.get("benchmark_index")
        ordinal = row.get("sdf_record_index")
        member_id = row.get("member_id")
        if index != len(rows):
            raise PF1MaterializationBenchmarkError(
                "benchmark indices must be dense and ordered"
            )
        if (
            not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal < 0
            or selection.member_ordinal(member_id) != ordinal
        ):
            raise PF1MaterializationBenchmarkError(
                f"benchmark line {line_number} has an invalid member/ordinal"
            )
        if ordinal in seen_ordinals or member_id in seen_members:
            raise PF1MaterializationBenchmarkError("benchmark membership repeats a member")
        if row.get("split") not in {"train", "dev"}:
            raise PF1MaterializationBenchmarkError("benchmark split must be train or dev")
        group_id = row.get("connectivity_identity_sha256")
        if not isinstance(group_id, str) or not group_id:
            raise PF1MaterializationBenchmarkError(
                "benchmark connectivity group is absent"
            )
        seen_ordinals.add(ordinal)
        seen_members.add(member_id)
        rows.append(row)
    if not rows:
        raise PF1MaterializationBenchmarkError("benchmark membership is empty")
    return tuple(rows)


def _init_worker(e3fp_source: str, linearizer_sha256: str) -> None:
    import numpy as np
    import selfies as sf
    from rdkit import Chem

    import_root, package_root, _files = projection.resolve_e3fp_source(
        Path(e3fp_source)
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "Chem": Chem,
            "np": np,
            "sf": sf,
            "e3fp_api": projection.import_locked_e3fp(import_root, package_root),
            "linearizer_sha256": linearizer_sha256,
        }
    )


def _materialize_one(
    task: tuple[int, int, str, str, bytes, Mapping[str, object]]
) -> dict[str, object]:
    benchmark_index, ordinal, member_id, split, mol_binary, binding = task
    Chem = _WORKER_STATE["Chem"]
    np = _WORKER_STATE["np"]
    sf = _WORKER_STATE["sf"]
    e3fp_api = _WORKER_STATE["e3fp_api"]

    source_mol = Chem.Mol(mol_binary)
    if source_mol is None:
        raise PF1MaterializationBenchmarkError("worker could not restore source Mol")
    projection_seconds = 0.0
    e3fp_seconds = 0.0
    surface_seconds = 0.0
    stage = "base_binding"
    stage_started = time.perf_counter()
    try:
        base_record, base_membership = overlay.validate_base_binding(
            np, binding, ordinal
        )
        if base_membership["member_id"] != member_id:
            raise PF1MaterializationBenchmarkError(
                "benchmark member differs from production binding"
            )
        stage = "projection"
        stage_started = time.perf_counter()
        tagged, source_atom_count, _ = projection.tag_source_atoms(Chem, source_mol)
        projected_mol, model_to_source = projection.project_hydrogens(
            Chem, tagged, source_atom_count
        )
        atom_universe = base_record["atom_universe"]
        base_geometry = base_record["geometry"]
        mapping = np.ascontiguousarray(np.asarray(model_to_source, dtype=np.int32))
        coordinates = np.ascontiguousarray(
            np.asarray(projected_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        )
        if not (
            source_atom_count == atom_universe["source_atom_count"]
            and int(projected_mol.GetNumAtoms()) == atom_universe["model_atom_count"]
            and bool(
                np.array_equal(mapping, atom_universe["model_to_source_atom_index"])
            )
            and bool(np.array_equal(coordinates, base_geometry["coordinates"]))
        ):
            raise PF1MaterializationBenchmarkError(
                f"projection parity failed for ordinal {ordinal}"
            )
        projection_seconds = time.perf_counter() - stage_started

        stage = "inherited_e3fp"
        stage_started = time.perf_counter()
        raw, inherited_ids, duplicate_mask, summary, _resolved = (
            inheritance.generate_e3fp_projection_pair(
                np, e3fp_api, projected_mol, ordinal
            )
        )
        if not bool(np.array_equal(raw, base_geometry["e3fp"])):
            raise PF1MaterializationBenchmarkError(
                f"raw E3FP parity failed for ordinal {ordinal}"
            )
        if inherited_ids.shape != base_geometry["e3fp"].shape:
            raise PF1MaterializationBenchmarkError(
                f"inherited E3FP shape failed for ordinal {ordinal}"
            )
        e3fp_seconds = time.perf_counter() - stage_started

        stage = "atom_selfies_graph_ports"
        stage_started = time.perf_counter()
        linearization = mol_linearizer.linearize_mol(projected_mol)
        augmentation = topology.build_topology_augmentation(
            linearization_result=linearization,
            member_id=base_membership["member_id"],
            base_record_content_sha256=base_membership["record_content_sha256"],
            linearizer_spec_sha256=_WORKER_STATE["linearizer_sha256"],
            expected_motif_atom_indices=base_record["topology"]["motif_atom_indices"],
            expected_motif_lexeme_sha256=base_record["topology"]["motif_lexeme_sha256"],
            source_atom_count=source_atom_count,
            model_to_source_atom_index=model_to_source,
        )
        groups = tuple(
            tuple(row)
            for row in augmentation["logical_motif_domain"]["motif_atom_indices"]
        )
        cross_edges = paired_builder.cross_edges_from_augmentation(augmentation)
        surfaces = paired.discover_production_paired_identity_surfaces(
            Chem, sf, projected_mol, groups, cross_edges
        )
        surface_seconds = time.perf_counter() - stage_started
    except Exception as exc:
        elapsed = time.perf_counter() - stage_started
        if stage == "projection":
            projection_seconds = elapsed
        elif stage == "inherited_e3fp":
            e3fp_seconds = elapsed
        elif stage == "atom_selfies_graph_ports":
            surface_seconds = elapsed
        return {
            "status": "reject",
            "benchmark_index": benchmark_index,
            "sdf_record_index": ordinal,
            "stage": stage,
            "reason": f"{type(exc).__name__}: {exc}",
            "projection_seconds": projection_seconds,
            "e3fp_seconds": e3fp_seconds,
            "surface_seconds": surface_seconds,
        }

    motif_identity_utf8_bytes = sum(
        len(motif.identity_smiles.encode("utf-8"))
        for motif in surfaces.graph_encoding.motifs
    )
    planning_row = donor_atom_map.build_release_row(
        selection_index=benchmark_index,
        member_id=member_id,
        sdf_record_index=ordinal,
        split=split,
        storage_key=str(base_membership["record_storage_key"]),
        graph_encoding=surfaces.graph_encoding,
    )
    return {
        "status": "pass",
        "benchmark_index": benchmark_index,
        "sdf_record_index": ordinal,
        "atom_count": int(projected_mol.GetNumAtoms()),
        "motif_count": len(groups),
        "cross_edge_count": len(cross_edges),
        "selfies_symbol_count": len(surfaces.atom_surface.selfies_symbols),
        "motif_identity_utf8_bytes": motif_identity_utf8_bytes,
        "slots_populated": int(summary["slots_populated"]),
        "duplicate_slots": int(summary["duplicate_slots"]),
        "changed_token_slots": int(summary["changed_token_slots"]),
        "projection_seconds": projection_seconds,
        "e3fp_seconds": e3fp_seconds,
        "surface_seconds": surface_seconds,
        "donor_atom_map_row": planning_row,
    }


def _sum(results: Sequence[Mapping[str, object]], field: str) -> float:
    return float(sum(float(row[field]) for row in results))


def _integer_sum(results: Sequence[Mapping[str, object]], field: str) -> int:
    return int(sum(int(row.get(field, 0)) for row in results))


def summarize_results(
    results: Sequence[Mapping[str, object]],
    *,
    phase_seconds: Mapping[str, float],
    workers: int,
    max_pending: int,
    source_observation: Mapping[str, object],
    shard_count: int,
) -> dict[str, object]:
    count = len(results)
    if count <= 0:
        raise PF1MaterializationBenchmarkError("no benchmark result was produced")
    passed = [row for row in results if row.get("status", "pass") == "pass"]
    rejected = [row for row in results if row.get("status") == "reject"]
    if len(passed) + len(rejected) != count:
        raise PF1MaterializationBenchmarkError("worker result status is invalid")
    worker_wall = float(phase_seconds["worker_pool"])
    scalable_seconds = worker_wall * PF1_TARGET_MEMBERS / count
    two_scan_estimate = scalable_seconds + 2.0 * float(phase_seconds["sdf_scan"])
    split_counts = Counter(str(row["split"]) for row in source_observation["rows"])  # type: ignore[index]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not rejected else "complete_with_rejects",
        "purpose": "bounded_pf1_materialization_performance_estimate",
        "counts": {
            "benchmark_members": count,
            "passed_members": len(passed),
            "failed_members": len(rejected),
            "train_members": split_counts["train"],
            "dev_members": split_counts["dev"],
            "base_shards_opened": shard_count,
            "atoms": _integer_sum(passed, "atom_count"),
            "motifs": _integer_sum(passed, "motif_count"),
            "cross_motif_edges": _integer_sum(passed, "cross_edge_count"),
            "selfies_symbols": _integer_sum(passed, "selfies_symbol_count"),
            "motif_identity_utf8_bytes": _integer_sum(
                passed, "motif_identity_utf8_bytes"
            ),
            "e3fp_slots_populated": _integer_sum(passed, "slots_populated"),
            "e3fp_duplicate_slots": _integer_sum(passed, "duplicate_slots"),
            "e3fp_changed_token_slots": _integer_sum(
                passed, "changed_token_slots"
            ),
        },
        "rejects": {
            "by_stage": dict(
                sorted(Counter(str(row["stage"]) for row in rejected).items())
            ),
            "rows": [
                {
                    "benchmark_index": row["benchmark_index"],
                    "sdf_record_index": row["sdf_record_index"],
                    "stage": row["stage"],
                    "reason": row["reason"],
                }
                for row in rejected
            ],
        },
        "execution": {
            "workers": workers,
            "max_pending": max_pending,
            "worker_model": "ordered_bounded_ProcessPoolExecutor_parent_read_only",
            "gpu_used": False,
        },
        "timing_seconds": {
            **{key: round(float(value), 6) for key, value in phase_seconds.items()},
            "worker_cpu_projection_sum": round(_sum(results, "projection_seconds"), 6),
            "worker_cpu_e3fp_sum": round(_sum(results, "e3fp_seconds"), 6),
            "worker_cpu_atom_selfies_graph_ports_sum": round(
                _sum(results, "surface_seconds"), 6
            ),
        },
        "throughput": {
            "worker_pool_members_per_second": round(count / worker_wall, 6),
            "projected_pf1_target_members": PF1_TARGET_MEMBERS,
            "projected_worker_wall_seconds_at_same_throughput": round(
                scalable_seconds, 3
            ),
            "projected_two_full_sdf_scan_plus_worker_seconds": round(
                two_scan_estimate, 3
            ),
            "projection_excludes_tokenizer_and_lmdb_publication": True,
        },
        "source_sdf": {
            key: value for key, value in source_observation.items() if key != "rows"
        },
        "method_boundary": {
            "production_release_opened_read_only": True,
            "production_release_modified": False,
            "molecule_payload_copied": False,
            "explicit_inherited_e3fp_executed": True,
            "atom_selfies_and_graph_ports_surfaces_executed": True,
            "benchmark_publishes_training_records": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    benchmark_path = Path(args.benchmark_membership).expanduser().resolve()
    release_root = Path(args.release_root).expanduser().resolve()
    source_archive = Path(args.source_archive).expanduser().resolve()
    e3fp_source = Path(args.e3fp_source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not benchmark_path.is_file():
        raise PF1MaterializationBenchmarkError("benchmark membership is not a file")
    if not release_root.is_dir() or not source_archive.is_file() or not e3fp_source.exists():
        raise PF1MaterializationBenchmarkError(
            "release root, source archive or E3FP source is absent"
        )
    if output_dir.exists():
        raise PF1MaterializationBenchmarkError("--output-dir must be a new path")

    total_started = time.perf_counter()
    rows = load_benchmark_membership(benchmark_path)
    selection_seconds = time.perf_counter() - total_started
    ordinals = tuple(int(row["sdf_record_index"]) for row in rows)

    try:
        import lmdb
        import numpy as np
        from rdkit import Chem
    except ImportError as exc:
        raise PF1MaterializationBenchmarkError(
            "NumPy, RDKit and python-lmdb are required"
        ) from exc

    started = time.perf_counter()
    full_manifest_path = release_root / "full_release_manifest.json"
    candidate = release_reader.load_json(full_manifest_path, "production full manifest")
    release_selection = {
        "release": {
            "release_id": candidate.get("release_id"),
            "full_release_manifest_sha256": release_reader.sha256_file(
                full_manifest_path
            ),
            "logical_release_root_sha256": candidate.get(
                "logical_release_root_sha256"
            ),
        }
    }
    _manifest_path, release_manifest = release_reader.load_release_manifest(
        release_root, release_selection
    )
    configuration = release_manifest.get("configuration", {})
    source_record_count = configuration.get("source_record_count")
    locked_member = configuration.get("locked_sdf_member")
    archive_lock = configuration.get("staged_inputs", {}).get(
        "train_3d_sdf_archive"
    )
    if (
        not isinstance(source_record_count, int)
        or not isinstance(locked_member, dict)
        or not isinstance(archive_lock, dict)
        or source_archive.stat().st_size != archive_lock.get("bytes")
    ):
        raise PF1MaterializationBenchmarkError(
            "production release and source archive binding is incomplete"
        )
    items = tuple(
        release_reader.SelectionItem("pf1_benchmark", index, ordinal, ())
        for index, ordinal in enumerate(ordinals)
    )
    bound, shard_receipts = release_reader.load_bound_records(
        release_root,
        release_manifest,
        items,
        np,
        lmdb,
        record_validator=overlay.validate_overlay_release_record,
    )
    release_read_seconds = time.perf_counter() - started

    def report_progress(observed: int, expected: int) -> None:
        print(
            "[pf1-benchmark] scanned {:,}/{:,} SDF records".format(
                observed, expected
            ),
            file=sys.stderr,
            flush=True,
        )

    started = time.perf_counter()
    molecules, sdf_observation = release_reader.stream_selected_sdf(
        Chem,
        source_archive,
        locked_member,
        ordinals,
        source_record_count,
        progress_every=args.progress_every,
        progress=report_progress,
    )
    sdf_scan_seconds = time.perf_counter() - started

    linearizer_sha = release_reader.sha256_file(Path(mol_linearizer.__file__).resolve())
    tasks = (
        (
            int(row["benchmark_index"]),
            int(row["sdf_record_index"]),
            str(row["member_id"]),
            str(row["split"]),
            bytes(molecules[int(row["sdf_record_index"])].ToBinary()),
            bound[int(row["sdf_record_index"])],
        )
        for row in rows
    )
    started = time.perf_counter()
    results = list(
        production_builder.ordered_bounded_map(
            _materialize_one,
            tasks,
            args.workers,
            args.max_pending,
            initializer=_init_worker,
            initargs=(str(e3fp_source), linearizer_sha),
        )
    )
    worker_seconds = time.perf_counter() - started
    if [int(row["benchmark_index"]) for row in results] != list(range(len(rows))):
        raise PF1MaterializationBenchmarkError(
            "ordered worker results differ from benchmark membership"
        )
    total_seconds = time.perf_counter() - total_started
    observation: dict[str, object] = {
        "archive_bytes": archive_lock["bytes"],
        "sdf_records_scanned": sdf_observation["prefix_scan"]["sdf_records_scanned"],
        "maximum_selected_ordinal": max(ordinals),
        "rows": rows,
    }
    manifest = summarize_results(
        results,
        phase_seconds={
            "selection_load": selection_seconds,
            "release_lmdb_read": release_read_seconds,
            "sdf_scan": sdf_scan_seconds,
            "worker_pool": worker_seconds,
            "total": total_seconds,
        },
        workers=args.workers,
        max_pending=args.max_pending,
        source_observation=observation,
        shard_count=len(shard_receipts),
    )
    output_dir.mkdir(parents=True)
    donor_path = output_dir / DONOR_ATOM_MAP_NAME
    if not any(row.get("status") == "reject" for row in results):
        sidecar_bytes = 0
        with donor_path.open("x", encoding="utf-8", newline="\n") as handle:
            for result in results:
                sidecar_bytes += donor_atom_map.write_release_row(
                    handle, result["donor_atom_map_row"]  # type: ignore[arg-type]
                )
        sidecar_benchmark = donor_atom_map.benchmark_release_prefix(
            donor_path, max_rows=len(results)
        )
        manifest["donor_atom_map_artifact"] = {
            "relative_path": DONOR_ATOM_MAP_NAME,
            "schema_version": donor_atom_map.ROW_SCHEMA,
            "row_count": len(results),
            "payload_bytes": sidecar_bytes,
            "stream_replay": sidecar_benchmark,
            "enters_training_wire": False,
        }
    else:
        manifest["donor_atom_map_artifact"] = {
            "published": False,
            "reason": "benchmark chemistry rejects prevent a dense planning sidecar",
        }
    with (output_dir / "manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-membership", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-pending", type=int, default=DEFAULT_MAX_PENDING)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_pending < args.workers:
        parser.error("--max-pending must be at least --workers")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DONOR_ATOM_MAP_NAME",
    "PF1MaterializationBenchmarkError",
    "load_benchmark_membership",
    "summarize_results",
]
