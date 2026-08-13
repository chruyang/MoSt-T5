#!/usr/bin/env python3
"""Reprocess frozen PCQM stereo-divergence records into an SDF-authoritative supplement.

The original production-v2 schema requires exact strict-isomeric equality
between the official CSV and SDF.  The current model contract instead uses a
stereo-free 2D identity surface and treats the SDF/E3FP channel as state.  This
builder therefore creates a separate supplement; it never appends to or
rewrites the strict production release.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path


SCHEMA = "most-t5-r1/pcqm-stereo-recovery-supplement/v1"
RECORD_SCHEMA = "most-t5-r1/pcqm-stereo-recovery-record/v1"
MEMBERSHIP_SCHEMA = "most-t5-r1/pcqm-stereo-recovery-supplement-membership/v1"
_WORKER = None


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    return _sha256_bytes(_canonical_json(value))


def _json_line(value: dict) -> bytes:
    return _canonical_json(value) + b"\n"


def _import_file(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_rows(path: Path, policy: str) -> dict[int, dict]:
    rows = {}
    allowed = {
        "all_connectivity_equal": None,
        "conservative_recovery": {
            "candidate_representation_normalization",
            "candidate_stereo_free_identity_plus_sdf_state",
        },
    }[policy]
    with path.open("rb") as handle:
        for raw in handle:
            row = json.loads(raw)
            if allowed is not None and row["policy_action"] not in allowed:
                continue
            ordinal = int(row["sdf_record_index"])
            if ordinal in rows:
                raise RuntimeError("duplicate ordinal {}".format(ordinal))
            rows[ordinal] = row
    if not rows:
        raise RuntimeError("selected classification is empty")
    return rows


def _iter_selected_blocks(classifier, archive: Path, selected: set[int]):
    yield from classifier._iter_selected_sdf_blocks(archive, selected)


def _init_worker(config: dict) -> None:
    global _WORKER
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    token = str(os.getpid())
    import numpy as np
    from rdkit import Chem

    builder = _import_file(config["builder"], "recovery_builder_" + token)
    preflight = _import_file(config["preflight"], "recovery_preflight_" + token)
    linearizer = _import_file(config["linearizer"], "recovery_linearizer_" + token)
    codec = _import_file(config["codec"], "recovery_codec_" + token)
    import_root, package_root, _ = preflight.resolve_e3fp_source(config["e3fp_source"])
    e3fp_api = preflight.import_locked_e3fp(import_root, package_root)
    _WORKER = {
        "np": np,
        "Chem": Chem,
        "builder": builder,
        "preflight": preflight,
        "linearizer": linearizer,
        "codec": codec,
        "e3fp_api": e3fp_api,
        "config": config,
    }


def _worker(item):
    ordinal, block, classification = item
    state = _WORKER
    np = state["np"]
    Chem = state["Chem"]
    builder = state["builder"]
    preflight = state["preflight"]
    try:
        supplier = Chem.ForwardSDMolSupplier(io.BytesIO(block), sanitize=True, removeHs=False)
        source_mol = next(iter(supplier), None)
        if source_mol is None:
            return {"ordinal": ordinal, "status": "reject", "reason": "SDF_PARSE_FAILED"}
        # Match the frozen production worker IPC boundary exactly.  RDKit's
        # binary round-trip deterministically stores conformer coordinates as
        # float32; the immutable source/geometry identity hashes bind this
        # representation rather than the supplier's initial float64 array.
        source_mol = Chem.Mol(bytes(source_mol.ToBinary()))
        source_identity = builder.molecule_identity_sha256(Chem, np, source_mol)
        if source_identity != classification["source_mol_identity_sha256"]:
            raise RuntimeError("source identity drift at {}".format(ordinal))
        preflight.finite_single_conformer(source_mol, "sdf_parse")
        tagged, source_atom_count, _ = preflight.tag_source_atoms(Chem, source_mol)
        source_h_count = sum(atom.GetAtomicNum() == 1 for atom in tagged.GetAtoms())
        geometry_mol, model_to_source = preflight.project_hydrogens(Chem, tagged, source_atom_count)
        geometry_identity = builder.molecule_identity_sha256(Chem, np, geometry_mol)
        if geometry_identity != classification["geometry_mol_identity_sha256"]:
            raise RuntimeError("geometry identity drift at {}".format(ordinal))
        positions = np.ascontiguousarray(
            np.asarray(geometry_mol.GetConformer(0).GetPositions(), dtype=np.float32)
        )
        e3fp, _, resolved_e3fp = preflight.generate_e3fp(
            np, state["e3fp_api"], geometry_mol, ordinal
        )
        e3fp = np.ascontiguousarray(np.asarray(e3fp, dtype=np.int32))
        linearized = state["linearizer"].linearize_mol(geometry_mol)
        groups = builder.motif_arrays(np, linearized, int(geometry_mol.GetNumAtoms()))
        fragments = tuple(linearized.fragment_sequence)
        if len(fragments) != len(groups):
            raise RuntimeError("motif fragment/group mismatch at {}".format(ordinal))
        motif_lexeme_sha256 = [
            _sha256_bytes(fragment.encode("utf-8")) for fragment in fragments
        ]
        model_to_source = np.ascontiguousarray(np.asarray(model_to_source, dtype=np.int32))
        record = {
            "record_schema_version": RECORD_SCHEMA,
            "release": {
                "release_id": state["config"]["release_id"],
                "source_release_id": state["config"]["source_release_id"],
                "identity_policy": "stereo_free_connectivity_with_sdf_authoritative_state",
                "strict_production_release_mutated": False,
            },
            "member": {
                "member_id": classification["member_id"],
                "sdf_record_index": ordinal,
                "official_csv_row_index": ordinal,
                "source_address_sha256": classification["source_address_sha256"],
                "source_mol_identity_sha256": source_identity,
            },
            "identity": {
                "status": classification["classification"],
                "policy_action": classification["policy_action"],
                "canonical_connectivity_sha256": classification["connectivity_sha256"],
                "sdf_strict_smiles_sha256": classification["sdf_strict_sha256"],
                "official_strict_smiles_sha256": classification["official_strict_sha256"],
                "strict_hashes_expected_to_differ": True,
            },
            "atom_universe": {
                "source_atom_count": int(source_atom_count),
                "source_explicit_hydrogen_count": int(source_h_count),
                "model_atom_count": int(geometry_mol.GetNumAtoms()),
                "model_to_source_atom_index": model_to_source,
                "geometry_mol_identity_sha256": geometry_identity,
            },
            "topology": {
                "motif_count": len(groups),
                "motif_atom_indices": groups,
                "motif_atom_indices_sha256": _sha256_json([
                    {"dtype": str(group.dtype), "shape": list(group.shape), "sha256": _sha256_bytes(group.tobytes())}
                    for group in groups
                ]),
                "motif_lexeme_sha256": motif_lexeme_sha256,
            },
            "geometry": {
                "geometry_valid": True,
                "coordinates": positions,
                "coordinates_sha256": _sha256_bytes(positions.tobytes()),
                "e3fp": e3fp,
                "e3fp_shape": list(e3fp.shape),
                "e3fp_sha256": _sha256_bytes(e3fp.tobytes()),
                "e3fp_params_sha256": _sha256_json(resolved_e3fp),
            },
        }
        payload = state["codec"].encode_record(np, record)
        decoded, logical_sha = state["codec"].decode_record(np, payload)
        if decoded["member"]["sdf_record_index"] != ordinal:
            raise RuntimeError("wire replay ordinal mismatch")
        return {
            "ordinal": ordinal,
            "status": "admit",
            "payload": payload,
            "logical_sha256": logical_sha,
            "record_wire_sha256": _sha256_bytes(payload),
            "record_wire_bytes": len(payload),
            "motif_count": len(groups),
            "model_atom_count": int(geometry_mol.GetNumAtoms()),
        }
    except preflight.RecordRejected as exc:
        return {"ordinal": ordinal, "status": "reject", "reason": exc.reason_code}


def _ordered_map(function, iterable, workers: int, max_pending: int, initializer, initargs):
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=initializer, initargs=initargs
    ) as executor:
        iterator = iter(iterable)
        pending = collections.deque()
        try:
            for _ in range(max_pending):
                pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-rows", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--e3fp-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=("all_connectivity_equal", "conservative_recovery"), default="all_connectivity_equal")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-pending", type=int, default=84)
    parser.add_argument("--map-size-gib", type=int, default=2)
    parser.add_argument("--commit-every", type=int, default=512)
    args = parser.parse_args()
    if args.workers < 1 or args.max_pending < args.workers:
        parser.error("workers must be positive and max-pending >= workers")

    import collections
    globals()["collections"] = collections
    import lmdb

    root = Path(__file__).resolve().parents[1]
    paths = {
        "classifier": str(root / "semantic" / "classify_pcqm_stereo_2d3d_divergence_v1.py"),
        "builder": str(root / "adapter" / "build_pcqm_p1_geometry_sidecar.py"),
        "preflight": str(root / "gates" / "pcqm_e3fp_preflight.py"),
        "linearizer": str(root / "adapter" / "mol_linearizer.py"),
        "codec": str(root / "adapter" / "sidecar_v2_codec.py"),
    }
    classifier = _import_file(paths["classifier"], "recovery_classifier_parent")
    classification_path = Path(args.classification_rows).resolve()
    archive_path = Path(args.source_archive).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError("output directory already exists")
    output_dir.mkdir(parents=True)
    rows = _load_rows(classification_path, args.policy)
    selected = set(rows)
    release_id = "pcqm-stereo-recovery-supplement-v1"
    config = {
        **paths,
        "e3fp_source": str(Path(args.e3fp_source).resolve()),
        "release_id": release_id,
        "source_release_id": "pcqm-geometry-production-v2-20260805T0430Z",
    }

    def items():
        for ordinal, block in _iter_selected_blocks(classifier, archive_path, selected):
            yield ordinal, block, rows[ordinal]

    started = time.time()
    env = lmdb.open(
        str(output_dir / "records.lmdb"), map_size=args.map_size_gib * 1024 ** 3,
        subdir=True, readonly=False, lock=True, max_dbs=1,
    )
    membership_path = output_dir / "membership.jsonl"
    rejects_path = output_dir / "rejects.jsonl"
    counts = collections.Counter()
    wire_bytes = 0
    expected_index = 0
    txn = env.begin(write=True)
    try:
        with membership_path.open("wb") as membership, rejects_path.open("wb") as rejects:
            results = _ordered_map(
                _worker, items(), args.workers, args.max_pending, _init_worker, (config,)
            )
            for result in results:
                ordinal = int(result["ordinal"])
                classification = rows[ordinal]
                if result["status"] == "admit":
                    key = "ogb_pcqm4mv2_train_row_index:{:010d}".format(ordinal).encode("ascii")
                    if not txn.put(key, result["payload"], overwrite=False):
                        raise RuntimeError("duplicate LMDB key")
                    membership.write(_json_line({
                        "schema": MEMBERSHIP_SCHEMA,
                        "selection_index": expected_index,
                        "member_id": classification["member_id"],
                        "sdf_record_index": ordinal,
                        "record_storage_key": key.decode("ascii"),
                        "record_wire_sha256": result["record_wire_sha256"],
                        "record_wire_bytes": result["record_wire_bytes"],
                        "logical_record_sha256": result["logical_sha256"],
                        "classification": classification["classification"],
                        "policy_action": classification["policy_action"],
                        "model_atom_count": result["model_atom_count"],
                        "motif_count": result["motif_count"],
                    }))
                    wire_bytes += result["record_wire_bytes"]
                    counts["admitted"] += 1
                else:
                    rejects.write(_json_line({
                        "schema": MEMBERSHIP_SCHEMA,
                        "selection_index": expected_index,
                        "member_id": classification["member_id"],
                        "sdf_record_index": ordinal,
                        "reason": result["reason"],
                    }))
                    counts["rejected"] += 1
                expected_index += 1
                if expected_index % args.commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                if expected_index % 1000 == 0:
                    print("materialized {}/{}".format(expected_index, len(rows)), flush=True)
            txn.commit()
            txn = None
    finally:
        if txn is not None:
            txn.abort()
        env.sync()
        env.close()
    if expected_index != len(rows):
        raise RuntimeError("source scan did not close selected membership")
    manifest = {
        "schema": SCHEMA,
        "status": "pass" if counts["rejected"] == 0 else "pass_with_explicit_rejects",
        "identity_policy": "stereo_free_connectivity_with_sdf_authoritative_state",
        "strict_production_release_mutated": False,
        "requires_separate_reader_or_next_release_merge": True,
        "inputs": {
            "classification_rows": str(classification_path),
            "classification_rows_sha256": _sha256_file(classification_path),
            "source_archive": str(archive_path),
            "policy": args.policy,
        },
        "runtime": {
            "workers": args.workers,
            "max_pending": args.max_pending,
            "wall_seconds": time.time() - started,
        },
        "counts": {
            "selected": len(rows),
            "admitted": counts["admitted"],
            "rejected": counts["rejected"],
            "record_wire_bytes": wire_bytes,
        },
        "artifacts": {
            "membership": {"path": membership_path.name, "rows": counts["admitted"], "sha256": _sha256_file(membership_path)},
            "rejects": {"path": rejects_path.name, "rows": counts["rejected"], "sha256": _sha256_file(rejects_path)},
            "lmdb": {"path": "records.lmdb", "entries": counts["admitted"]},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    (output_dir / "COMPLETED").write_text(_sha256_file(manifest_path) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
