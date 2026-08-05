#!/usr/bin/env python3
"""Replay the molecule-native linearizer on a frozen PCQM topology canary.

The runner consumes an explicit 32-record smoke set and a disjoint 256-record
canary set.  For each selected SDF ordinal it reads the existing production-v2
payload, replays only the producer's RDKit molecule transport and hydrogen
projection, and asks :mod:`mol_linearizer` for motif topology.  The derived
augmentation must agree with the production record's motif atom groups and
ordered motif-lexeme digests.

No coordinate or E3FP feature is recomputed.  The production release is opened
read-only and the output is a separate JSONL plus a compact manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from most_t5_next.r1.adapter import mol_linearizer
from most_t5_next.r1.adapter import p1_topology_augmentation_v1 as topology
from most_t5_next.r1.adapter import sidecar_v2_codec
from most_t5_next.r1.gates import pcqm_e3fp_preflight as projection


SELECTION_SCHEMA = "most-t5-r1/p1-topology-canary-selection/v1"
ROW_SCHEMA = "most-t5-r1/p1-topology-canary-row/v1"
MANIFEST_SCHEMA = "most-t5-r1/p1-topology-canary-manifest/v1"
FULL_RELEASE_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-full-release/v2"
PRODUCTION_RECORD_SCHEMA = "most-t5-r1/p1-pcqm-geometry-production-pretokenizer-record/v2"
SMOKE_COUNT = 32
CANARY_COUNT = 256
GROUP_ORDER = ("smoke", "canary")


class TopologyCanaryError(RuntimeError):
    """One selected member could not be bound and replayed exactly."""


@dataclass(frozen=True)
class SelectionItem:
    group: str
    group_index: int
    sdf_record_index: int
    selection_tags: tuple[str, ...]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise TopologyCanaryError(f"{label} is not a file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TopologyCanaryError(f"{label} must contain one JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_selection(path: Path) -> tuple[dict, tuple[SelectionItem, ...]]:
    """Load the frozen selection and preserve its declared within-group order."""

    document = load_json(path, "topology canary selection")
    if document.get("schema_version") != SELECTION_SCHEMA:
        raise TopologyCanaryError("selection schema version mismatch")
    if not isinstance(document.get("selection_id"), str) or not document["selection_id"]:
        raise TopologyCanaryError("selection_id must be non-empty")
    release = document.get("release")
    if not isinstance(release, dict) or not all(
        isinstance(release.get(key), str) and release[key]
        for key in ("release_id", "full_release_manifest_sha256", "logical_release_root_sha256")
    ):
        raise TopologyCanaryError("selection release binding is incomplete")
    if not _is_sha256(release["full_release_manifest_sha256"]) or not _is_sha256(
        release["logical_release_root_sha256"]
    ):
        raise TopologyCanaryError("selection release hashes are malformed")

    groups = document.get("groups")
    expected_counts = {"smoke": SMOKE_COUNT, "canary": CANARY_COUNT}
    if not isinstance(groups, dict) or set(groups) != set(expected_counts):
        raise TopologyCanaryError("selection groups must be exactly smoke and canary")
    items: list[SelectionItem] = []
    seen: set[int] = set()
    for group in GROUP_ORDER:
        rows = groups[group]
        if not isinstance(rows, list) or len(rows) != expected_counts[group]:
            raise TopologyCanaryError(
                f"selection group {group} must contain exactly {expected_counts[group]} rows"
            )
        for group_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TopologyCanaryError(f"selection {group}[{group_index}] must be an object")
            ordinal = row.get("sdf_record_index")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
                raise TopologyCanaryError(f"selection {group}[{group_index}] has an invalid ordinal")
            if ordinal in seen:
                raise TopologyCanaryError(f"selection repeats SDF ordinal {ordinal}")
            tags = row.get("selection_tags", [])
            if (
                not isinstance(tags, list)
                or any(not isinstance(tag, str) or not tag for tag in tags)
                or tags != sorted(set(tags))
            ):
                raise TopologyCanaryError("selection_tags must be sorted unique non-empty strings")
            seen.add(ordinal)
            items.append(SelectionItem(group, group_index, ordinal, tuple(tags)))
    return document, tuple(items)


def load_release_manifest(release_root: Path, selection: Mapping[str, object]) -> tuple[Path, dict]:
    manifest_path = release_root / "full_release_manifest.json"
    manifest = load_json(manifest_path, "production-v2 full release manifest")
    release_binding = selection["release"]
    if sha256_file(manifest_path) != release_binding["full_release_manifest_sha256"]:
        raise TopologyCanaryError("selection points to different full-release-manifest bytes")
    if not (
        manifest.get("schema_version") == FULL_RELEASE_SCHEMA
        and manifest.get("release_status") == "complete"
        and manifest.get("release_id") == release_binding["release_id"]
        and manifest.get("logical_release_root_sha256")
        == release_binding["logical_release_root_sha256"]
    ):
        raise TopologyCanaryError("selection and production-v2 release do not describe the same release")
    return manifest_path, manifest


def _shard_for_ordinal(manifest: Mapping[str, object], ordinal: int) -> dict:
    for row in manifest.get("shards", []):
        if row.get("range_start") <= ordinal < row.get("range_end"):
            return row
    raise TopologyCanaryError(f"SDF ordinal {ordinal} is outside the production shard map")


def _read_selected_membership(
    path: Path, range_start: int, ordinals: Iterable[int]
) -> dict[int, dict]:
    wanted = set(ordinals)
    found: dict[int, dict] = {}
    max_offset = max(wanted) - range_start
    with path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            if offset > max_offset:
                break
            ordinal = range_start + offset
            if ordinal in wanted:
                row = json.loads(line)
                if row.get("sdf_record_index") != ordinal:
                    raise TopologyCanaryError("membership line position and SDF ordinal disagree")
                found[ordinal] = row
    if set(found) != wanted:
        raise TopologyCanaryError("selected membership rows are incomplete")
    return found


def validate_bound_record(record: Mapping[str, object], membership: Mapping[str, object], ordinal: int) -> None:
    member = record.get("member")
    atom_universe = record.get("atom_universe")
    record_topology = record.get("topology")
    if record.get("record_schema_version") != PRODUCTION_RECORD_SCHEMA:
        raise TopologyCanaryError("selected payload is not a production-v2 record")
    if not isinstance(member, dict) or not isinstance(atom_universe, dict) or not isinstance(record_topology, dict):
        raise TopologyCanaryError("selected production record lacks member/topology/atom-universe fields")
    if not (
        membership.get("disposition") == "admit"
        and membership.get("sdf_record_index") == ordinal
        and membership.get("record_storage_key") == f"{ordinal:09d}"
        and member.get("sdf_record_index") == ordinal
        and member.get("member_id") == membership.get("member_id")
        and member.get("storage_key") == membership.get("record_storage_key")
    ):
        raise TopologyCanaryError("selected membership and production payload are not bound")
    if record_topology.get("linearizer_spec_sha256") != sha256_file(Path(mol_linearizer.__file__)):
        raise TopologyCanaryError("production record was built by different mol_linearizer bytes")
    motif_count = record_topology.get("motif_count")
    if not isinstance(motif_count, int) or motif_count <= 0:
        raise TopologyCanaryError("production record motif count is invalid")
    if not (
        len(record_topology.get("motif_atom_indices", [])) == motif_count
        and len(record_topology.get("motif_lexeme_sha256", [])) == motif_count
    ):
        raise TopologyCanaryError("production motif arrays are not aligned")
    if not all(
        isinstance(atom_universe.get(key), int) and atom_universe[key] > 0
        for key in ("source_atom_count", "model_atom_count")
    ):
        raise TopologyCanaryError("production atom-universe counts are invalid")


def load_bound_records(
    release_root: Path,
    manifest: Mapping[str, object],
    items: Sequence[SelectionItem],
    np,
    lmdb_module,
) -> tuple[dict[int, dict], list[dict]]:
    """Read only selected membership lines and LMDB values from production-v2."""

    by_shard: dict[int, list[int]] = defaultdict(list)
    shard_entries: dict[int, dict] = {}
    for item in items:
        shard = _shard_for_ordinal(manifest, item.sdf_record_index)
        shard_index = int(shard["shard_index"])
        by_shard[shard_index].append(item.sdf_record_index)
        shard_entries[shard_index] = shard

    bound: dict[int, dict] = {}
    shard_receipts: list[dict] = []
    for shard_index in sorted(by_shard):
        top_entry = shard_entries[shard_index]
        shard_dir = release_root / f"shard-{shard_index:06d}"
        shard_manifest_path = shard_dir / "shard_manifest.json"
        shard_manifest = load_json(shard_manifest_path, "production shard manifest")
        if sha256_file(shard_manifest_path) != top_entry.get("shard_manifest_sha256"):
            raise TopologyCanaryError("production shard manifest hash mismatch")
        start, end = int(shard_manifest["range_start"]), int(shard_manifest["range_end"])
        ordinals = sorted(by_shard[shard_index])
        if any(not start <= ordinal < end for ordinal in ordinals):
            raise TopologyCanaryError("selected ordinal does not belong to its declared shard")
        memberships = _read_selected_membership(shard_dir / "membership.jsonl", start, ordinals)

        environment = lmdb_module.open(
            str(shard_dir / "geometry_records.lmdb"),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=8,
        )
        try:
            with environment.begin(write=False) as transaction:
                for ordinal in ordinals:
                    membership = memberships[ordinal]
                    if membership.get("disposition") != "admit":
                        raise TopologyCanaryError(f"selected ordinal {ordinal} is rejected in production-v2")
                    raw = transaction.get(membership["record_storage_key"].encode("ascii"))
                    if raw is None:
                        raise TopologyCanaryError(f"selected ordinal {ordinal} has no LMDB payload")
                    record, logical_hash = sidecar_v2_codec.decode_record(np, raw)
                    if logical_hash != membership.get("record_content_sha256"):
                        raise TopologyCanaryError("selected payload logical hash differs from membership")
                    validate_bound_record(record, membership, ordinal)
                    bound[ordinal] = {
                        "record": record,
                        "membership": membership,
                        "shard_index": shard_index,
                    }
        finally:
            environment.close()
        shard_receipts.append(
            {
                "shard_index": shard_index,
                "range_start": start,
                "range_end": end,
                "selected_record_count": len(ordinals),
                "shard_manifest_sha256": top_entry["shard_manifest_sha256"],
            }
        )
    return bound, shard_receipts


def _parse_selected_mol(Chem, block: bytes) -> object:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(block + b"$$$$\n"), sanitize=True, removeHs=False, strictParsing=True
    )
    molecule = next(iter(supplier), None)
    if molecule is None:
        raise TopologyCanaryError("selected SDF record did not parse")
    # Production serializes the parent-process Mol before its worker sees it.
    molecule = Chem.Mol(bytes(molecule.ToBinary()))
    if molecule is None:
        raise TopologyCanaryError("selected SDF molecule failed the production IPC round-trip")
    return molecule


def stream_selected_sdf(
    Chem,
    archive_path: Path,
    locked_member: Mapping[str, object],
    selected_ordinals: Iterable[int],
    expected_record_count: int,
    progress_every: int = 250_000,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[int, object], dict]:
    """Scan through the highest selected ordinal and parse only selected blocks."""

    targets = set(selected_ordinals)
    if not targets:
        raise TopologyCanaryError("selected SDF ordinal set is empty")
    max_target = max(targets)
    molecules: dict[int, object] = {}
    prefix_digest = hashlib.sha256()
    byte_count = 0
    ordinal = 0
    record_has_content = False
    buffer: bytearray | None = bytearray() if 0 in targets else None
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        member = next(
            (candidate for candidate in archive if candidate.name == locked_member.get("tar_member_name")),
            None,
        )
        if member is None or not member.isfile():
            raise TopologyCanaryError("locked SDF tar member is absent")
        if int(member.size) != locked_member.get("uncompressed_bytes"):
            raise TopologyCanaryError("locked SDF member size differs from production-v2")
        stream = archive.extractfile(member)
        if stream is None:
            raise TopologyCanaryError("locked SDF member cannot be opened")
        try:
            for line in stream:
                prefix_digest.update(line)
                byte_count += len(line)
                if line.rstrip(b"\r\n") == b"$$$$":
                    if ordinal in targets:
                        molecules[ordinal] = _parse_selected_mol(Chem, bytes(buffer or b""))
                    ordinal += 1
                    if progress is not None and progress_every > 0 and ordinal % progress_every == 0:
                        progress(ordinal, expected_record_count)
                    record_has_content = False
                    if ordinal > max_target:
                        break
                    buffer = bytearray() if ordinal in targets else None
                    continue
                record_has_content = True
                if buffer is not None:
                    buffer.extend(line)
            if record_has_content:
                if ordinal in targets:
                    molecules[ordinal] = _parse_selected_mol(Chem, bytes(buffer or b""))
                ordinal += 1
        finally:
            stream.close()

    if set(molecules) != targets:
        raise TopologyCanaryError("SDF stream did not resolve every selected ordinal")
    complete_member_rehash = byte_count == int(member.size) and ordinal == expected_record_count
    if complete_member_rehash and prefix_digest.hexdigest() != locked_member.get("sha256"):
        raise TopologyCanaryError("complete SDF member hash differs from the production-v2 source lock")
    observation = {
        "release_lock": {
            "tar_member_name": locked_member.get("tar_member_name"),
            "member_type": locked_member.get("member_type"),
            "uncompressed_bytes": locked_member.get("uncompressed_bytes"),
            "sha256": locked_member.get("sha256"),
        },
        "prefix_scan": {
            "maximum_selected_ordinal": max_target,
            "sdf_records_scanned": ordinal,
            "uncompressed_prefix_bytes": byte_count,
            "uncompressed_prefix_sha256": prefix_digest.hexdigest(),
            "stopped_after_maximum_selected_ordinal": ordinal == max_target + 1,
            "complete_member_rehashed": complete_member_rehash,
        },
    }
    return molecules, observation


def build_augmentation_row(
    Chem,
    np,
    item: SelectionItem,
    binding: Mapping[str, object],
    source_mol,
    linearizer_sha256: str,
) -> dict:
    """Replay projection/linearization and bind the result to one base payload."""

    record = binding["record"]
    membership = binding["membership"]
    tagged, source_atom_count, _ = projection.tag_source_atoms(Chem, source_mol)
    geometry_mol, model_to_source = projection.project_hydrogens(Chem, tagged, source_atom_count)
    atom_universe = record["atom_universe"]
    expected_mapping = [int(value) for value in atom_universe["model_to_source_atom_index"]]
    if not (
        source_atom_count == atom_universe["source_atom_count"]
        and geometry_mol.GetNumAtoms() == atom_universe["model_atom_count"]
        and list(model_to_source) == expected_mapping
    ):
        raise TopologyCanaryError("replayed source/model atom mapping differs from production-v2")

    linearization = mol_linearizer.linearize_mol(geometry_mol)
    record_topology = record["topology"]
    augmentation = topology.build_topology_augmentation(
        linearization_result=linearization,
        member_id=membership["member_id"],
        base_record_content_sha256=membership["record_content_sha256"],
        linearizer_spec_sha256=linearizer_sha256,
        expected_motif_atom_indices=record_topology["motif_atom_indices"],
        expected_motif_lexeme_sha256=record_topology["motif_lexeme_sha256"],
        source_atom_count=source_atom_count,
        model_to_source_atom_index=model_to_source,
    )
    return {
        "schema_version": ROW_SCHEMA,
        "selection": {
            "group": item.group,
            "group_index": item.group_index,
            "selection_tags": list(item.selection_tags),
        },
        "release": {
            "release_id": membership["sidecar_id"],
            "shard_index": binding["shard_index"],
        },
        "member": {
            "member_id": membership["member_id"],
            "sdf_record_index": item.sdf_record_index,
            "record_storage_key": membership["record_storage_key"],
            "base_record_content_sha256": membership["record_content_sha256"],
        },
        "augmentation_sha256": topology.augmentation_sha256(augmentation),
        "augmentation": augmentation,
    }


def _artifact(path: Path, row_count: int | None = None) -> dict:
    result = {
        "relative_path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def summarize_coverage(rows: Sequence[Mapping[str, object]]) -> dict:
    atom_counts: list[int] = []
    motif_counts: list[int] = []
    edge_counts: list[int] = []
    attachment_profiles: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    for row in rows:
        augmentation = row["augmentation"]
        atoms = augmentation["atom_universe"]
        motifs = augmentation["logical_motif_domain"]
        atom_counts.append(atoms["model_atom_count"])
        motif_counts.append(motifs["logical_motif_count"])
        edges = len(motifs["cross_motif_bonds"])
        edge_counts.append(edges)
        attached = sum(atoms["atom_is_attachment"])
        if attached == 0:
            attachment_profiles["no_attachment"] += 1
        elif attached == atoms["model_atom_count"]:
            attachment_profiles["attachment_only"] += 1
        else:
            attachment_profiles["attachment_and_core"] += 1
        tag_counts.update(row["selection"]["selection_tags"])

    def bounds(values: Sequence[int]) -> dict:
        return {"minimum": min(values), "maximum": max(values)}

    return {
        "model_atom_count": bounds(atom_counts),
        "logical_motif_count": bounds(motif_counts),
        "cross_motif_bond_count": bounds(edge_counts),
        "attachment_profile_counts": dict(sorted(attachment_profiles.items())),
        "selection_tag_counts": dict(sorted(tag_counts.items())),
    }


def write_outputs(output_dir: Path, rows: Sequence[dict], manifest: dict) -> dict:
    if output_dir.exists():
        raise TopologyCanaryError("--output-dir must be a new path")
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "topology_augmentation.jsonl"
    with rows_path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    manifest["artifacts"] = {"topology_augmentation": _artifact(rows_path, len(rows))}
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def run(args: argparse.Namespace) -> dict:
    selection_path = Path(args.selection).expanduser().resolve()
    release_root = Path(args.release_root).expanduser().resolve()
    archive_path = Path(args.source_archive).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not release_root.is_dir() or not archive_path.is_file():
        raise TopologyCanaryError("release root and source archive must exist")

    selection, items = load_selection(selection_path)
    manifest_path, release_manifest = load_release_manifest(release_root, selection)
    configuration = release_manifest.get("configuration", {})
    locked_member = configuration.get("locked_sdf_member")
    expected_record_count = configuration.get("source_record_count")
    if not isinstance(locked_member, dict) or not isinstance(expected_record_count, int):
        raise TopologyCanaryError("production-v2 source-member lock is incomplete")
    if max(item.sdf_record_index for item in items) >= expected_record_count:
        raise TopologyCanaryError("selection contains an ordinal outside the PCQM source range")
    archive_lock = configuration.get("staged_inputs", {}).get("train_3d_sdf_archive", {})
    if not isinstance(archive_lock, dict) or archive_path.stat().st_size != archive_lock.get("bytes"):
        raise TopologyCanaryError("source archive byte size differs from the production-v2 source lock")

    try:
        import lmdb
        import numpy as np
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise TopologyCanaryError("NumPy, RDKit, and python-lmdb are required") from exc

    bound, shard_receipts = load_bound_records(release_root, release_manifest, items, np, lmdb)

    def report_progress(observed: int, expected: int) -> None:
        print(
            f"[topology-canary] scanned {observed:,}/{expected:,} SDF records",
            file=sys.stderr,
            flush=True,
        )

    molecules, member_observation = stream_selected_sdf(
        Chem,
        archive_path,
        locked_member,
        (item.sdf_record_index for item in items),
        expected_record_count,
        progress_every=args.progress_every,
        progress=report_progress,
    )
    linearizer_sha = sha256_file(Path(mol_linearizer.__file__))
    rows = [
        build_augmentation_row(
            Chem, np, item, bound[item.sdf_record_index], molecules[item.sdf_record_index], linearizer_sha
        )
        for item in items
    ]
    group_counts = Counter(row["selection"]["group"] for row in rows)
    if group_counts != Counter({"smoke": SMOKE_COUNT, "canary": CANARY_COUNT}):
        raise TopologyCanaryError("output rows do not preserve the frozen group counts")

    code_files = {
        "runner": Path(__file__).resolve(),
        "mol_linearizer": Path(mol_linearizer.__file__).resolve(),
        "topology_augmentation": Path(topology.__file__).resolve(),
        "hydrogen_projection": Path(projection.__file__).resolve(),
        "payload_codec": Path(sidecar_v2_codec.__file__).resolve(),
    }
    compact_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "pass",
        "created_utc": utc_now(),
        "training_admission": False,
        "selection": {
            "selection_id": selection["selection_id"],
            "selection_manifest_sha256": sha256_file(selection_path),
            "ordered_selection_sha256": sha256_json(
                [
                    {
                        "group": item.group,
                        "group_index": item.group_index,
                        "sdf_record_index": item.sdf_record_index,
                        "selection_tags": list(item.selection_tags),
                    }
                    for item in items
                ]
            ),
            "counts": dict(sorted(group_counts.items())),
        },
        "production_release": {
            "release_id": release_manifest["release_id"],
            "full_release_manifest_sha256": sha256_file(manifest_path),
            "logical_release_root_sha256": release_manifest["logical_release_root_sha256"],
            "opened_read_only": True,
            "shards_read": shard_receipts,
        },
        "source_sdf": {
            "archive_release_observation": {
                "bytes": archive_lock["bytes"],
                "sha256": archive_lock["sha256"],
                "rehashed_by_this_runner": False,
            },
            "member": member_observation,
        },
        "code_sha256": {name: sha256_file(path) for name, path in sorted(code_files.items())},
        "runtime": {"python": sys.version.split()[0], "rdkit": rdBase.rdkitVersion},
        "counts": {"output_rows": len(rows), "failed_rows": 0},
        "coverage": summarize_coverage(rows),
        "method_boundary": {
            "linearizer_replayed": True,
            "production_worker_ipc_and_hydrogen_projection_replayed": True,
            "coordinates_recomputed": False,
            "e3fp_recomputed": False,
            "production_release_modified": False,
        },
    }
    return write_outputs(output_dir, rows, compact_manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="Frozen 32+256 selection JSON")
    parser.add_argument("--release-root", required=True, help="Completed production-v2 release")
    parser.add_argument("--source-archive", required=True, help="Official PCQM train-3D SDF tar.gz")
    parser.add_argument("--output-dir", required=True, help="New directory for JSONL and manifest")
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
