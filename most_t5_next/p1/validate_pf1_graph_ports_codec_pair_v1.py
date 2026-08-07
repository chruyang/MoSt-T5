#!/usr/bin/env python3
"""Validate a paired GraphPorts-v1/v2 PF-1 codec screen.

The two releases may differ only in the unmasked connection-token surface.
This gate replays every train member under every corruption epoch reached by
the 1,000-update PF-1 protocol (0--4) and every dev member under its frozen
view.  Motif selections and CE labels must be exactly equal; no sample is
replaced or truncated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from most_t5_next.p1 import build_pf1_graph_ports_v2_release_v1 as derive
from most_t5_next.p1.build_pf1_paired_release_v1 import (
    PF1PairedReleaseReader,
    TOKENIZER_DIRECTORY,
)
from most_t5_next.p1.production_bridge import collate_production_motif_record
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v1 as graph_v1
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v2 as graph_v2
from most_t5_next.r1.tokenizer.build_p1_canary_union_tokenizer_v1 import (
    load_verified_canary_union_tokenizer,
)


REPORT_SCHEMA = "most-t5-p1/pf1-graphports-codec-pair-gate/v1"
TRAIN_CORRUPTION_SEED = 0
TRAIN_CORRUPTION_EPOCHS = (0, 1, 2, 3, 4)
DEV_CORRUPTION_SEED = 1
DEV_CORRUPTION_EPOCH = 0
MASK_PROBABILITY = 0.15
DEFAULT_BATCH_SIZE = 64
TRAIN_GRADIENT_ACCUMULATION_STEPS = 2
TRAIN_OPTIMIZER_UPDATES = 1000


class PF1GraphPortsCodecPairError(RuntimeError):
    """The v1/v2 releases are not a controlled codec pair."""


@dataclass
class _PairCounts:
    records: int = 0
    corruption_views: int = 0
    selected_motifs: int = 0
    target_tokens: int = 0
    source_input_tokens: int = 0
    target_input_tokens: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        reduction = self.source_input_tokens - self.target_input_tokens
        return {
            "records": self.records,
            "corruption_views": self.corruption_views,
            "selected_motifs": self.selected_motifs,
            "target_tokens": self.target_tokens,
            "source_input_tokens": self.source_input_tokens,
            "target_input_tokens": self.target_input_tokens,
            "input_token_reduction": reduction,
            "input_fraction_reduction": (
                reduction / self.source_input_tokens
                if self.source_input_tokens
                else None
            ),
        }


def _identity_surfaces(record: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(record.input_ids[span.start : span.stop])
        for span in record.identity_spans
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_graph_document(
    document: Mapping[str, Any],
    *,
    tokenizer: Any,
    version: str,
) -> tuple[graph_v1.ConnectionRecord, ...]:
    motif = document.get("motif_training_document")
    if not isinstance(motif, dict):
        raise PF1GraphPortsCodecPairError("raw wire motif document is absent")
    dimensions = motif.get("dimensions")
    token_domain = motif.get("token_domain")
    logical = motif.get("logical_motif_domain")
    if not all(isinstance(value, dict) for value in (dimensions, token_domain, logical)):
        raise PF1GraphPortsCodecPairError("raw graph document domains are malformed")
    motif_count = int(dimensions["logical_motif_count"])
    expected = derive._connections_from_document(
        motif_count, logical["cross_motif_bonds"]
    )
    components = graph_v1._connected_component_motifs(motif_count, expected)
    graph_offset = derive._identity_graph_offset(motif)
    input_ids = token_domain.get("input_ids")
    roles = token_domain.get("token_role")
    owners = token_domain.get("token_to_logical_motif")
    connection_indices = logical.get("connection_token_indices")
    if not all(isinstance(value, list) for value in (input_ids, roles, owners, connection_indices)):
        raise PF1GraphPortsCodecPairError("raw graph token arrays are malformed")
    if not (len(input_ids) == len(roles) == len(owners)) or graph_offset >= len(input_ids) - 1:
        raise PF1GraphPortsCodecPairError("raw graph token-array lengths disagree")
    graph_ids = input_ids[graph_offset:-1]
    graph_roles = tuple(str(value) for value in roles[graph_offset:-1])
    graph_owners = tuple(int(value) for value in owners[graph_offset:-1])
    graph_tokens = tuple(tokenizer.convert_ids_to_tokens(int(value)) for value in graph_ids)
    if any(not isinstance(token, str) for token in graph_tokens):
        raise PF1GraphPortsCodecPairError("graph token ID does not reverse to one token")
    relative_indices = tuple(
        tuple(int(index) - graph_offset for index in row)
        for row in connection_indices
    )
    carrier_positions = tuple(
        index for index, role in enumerate(graph_roles) if role == "connection"
    )
    if len(carrier_positions) != 2 * len(expected):
        raise PF1GraphPortsCodecPairError("graph stream does not expose two owners per edge")
    endpoint_indices = tuple(
        ((carrier_positions[offset],), (carrier_positions[offset + 1],))
        for offset in range(0, len(carrier_positions), 2)
    )
    port_radix = max(
        (
            endpoint.port_id
            for connection in expected
            for endpoint in (connection.endpoint_a, connection.endpoint_b)
        ),
        default=0,
    ) + 1
    stream = graph_v1.ProductionGraphTokenStream(
        port_radix=port_radix,
        tokens=graph_tokens,
        token_roles=graph_roles,
        token_to_logical_motif=graph_owners,
        component_token_indices=tuple(() for _ in components),
        connection_endpoint_token_indices=endpoint_indices,
        connection_token_indices=relative_indices,
    )
    identity_mapping = tuple(range(motif_count))
    try:
        if version == graph_v1.FORMAT_VERSION:
            decoded = graph_v1._decode_graph_token_stream(
                stream, components, identity_mapping
            )
        elif version == graph_v2.FORMAT_VERSION:
            decoded = graph_v2._decode_endpoint_pair_graph_token_stream(
                stream, components, identity_mapping
            )
        else:
            raise PF1GraphPortsCodecPairError("unknown graph codec version")
    except graph_v1.GraphPortsContractError as exc:
        raise PF1GraphPortsCodecPairError("raw graph token decode failed") from exc
    if decoded != expected:
        raise PF1GraphPortsCodecPairError(
            "raw graph tokens do not decode to the persisted cross-motif bonds"
        )
    return expected


def _require_raw_graph_pair(
    source_document: Mapping[str, Any],
    target_document: Mapping[str, Any],
    *,
    tokenizer: Any,
) -> int:
    source = _decode_graph_document(
        source_document,
        tokenizer=tokenizer,
        version=graph_v1.FORMAT_VERSION,
    )
    target = _decode_graph_document(
        target_document,
        tokenizer=tokenizer,
        version=graph_v2.FORMAT_VERSION,
    )
    if source != target:
        raise PF1GraphPortsCodecPairError(
            "v1/v2 raw graph streams decode to different bond tables"
        )
    return len(source)


def _validate_raw_graph_release_pair(
    source_reader: PF1PairedReleaseReader,
    target_reader: PF1PairedReleaseReader,
    *,
    tokenizer: Any,
) -> dict[str, int]:
    source_rows = source_reader._train_rows + source_reader._dev_rows
    target_rows = target_reader._train_rows + target_reader._dev_rows
    if len(source_rows) != len(target_rows):
        raise PF1GraphPortsCodecPairError("raw release membership sizes differ")
    source_environment = source_reader.lmdb_module.open(
        str(source_reader.lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    target_environment = target_reader.lmdb_module.open(
        str(target_reader.lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    records = 0
    connections = 0
    try:
        with source_environment.begin(write=False) as source_transaction, target_environment.begin(
            write=False
        ) as target_transaction:
            for source_row, target_row in zip(source_rows, target_rows):
                membership_fields = (
                    "split",
                    "split_index",
                    "selection_index",
                    "sdf_record_index",
                    "member_id",
                    "storage_key",
                )
                if any(
                    source_row.get(field) != target_row.get(field)
                    for field in membership_fields
                ):
                    raise PF1GraphPortsCodecPairError(
                        "raw release membership order differs"
                    )
                storage_key = str(source_row["storage_key"])
                source_raw = source_transaction.get(storage_key.encode("ascii"))
                target_raw = target_transaction.get(storage_key.encode("ascii"))
                if source_raw is None or target_raw is None:
                    raise PF1GraphPortsCodecPairError("raw paired LMDB value is absent")
                try:
                    source_document = json.loads(bytes(source_raw))
                    target_document = json.loads(bytes(target_raw))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PF1GraphPortsCodecPairError(
                        "raw paired LMDB JSON is invalid"
                    ) from exc
                if not isinstance(source_document, dict) or not isinstance(
                    target_document, dict
                ):
                    raise PF1GraphPortsCodecPairError(
                        "raw paired LMDB root is not an object"
                    )
                if not (
                    source_document.get("schedule_index")
                    == target_document.get("schedule_index")
                    == source_row["selection_index"]
                    and source_document.get("sdf_record_index")
                    == target_document.get("sdf_record_index")
                    == source_row["sdf_record_index"]
                ):
                    raise PF1GraphPortsCodecPairError(
                        "raw wire schedule/source row differs from membership"
                    )
                connections += _require_raw_graph_pair(
                    source_document,
                    target_document,
                    tokenizer=tokenizer,
                )
                records += 1
    finally:
        source_environment.close()
        target_environment.close()
    if records != len(source_rows):
        raise PF1GraphPortsCodecPairError("raw graph replay did not close")
    return {
        "records": records,
        "cross_motif_connections": connections,
        "source_graph_decodes": records,
        "target_graph_decodes": records,
    }


def _require_release_provenance(
    *,
    source_release: Path,
    target_release: Path,
    target_manifest: Mapping[str, Any],
) -> dict[str, str]:
    source_manifest_sha256 = _sha256_file(source_release / "manifest.json")
    target_manifest_sha256 = _sha256_file(target_release / "manifest.json")
    source_binding = target_manifest.get("source_release")
    codec_binding = target_manifest.get("codec")
    if not isinstance(source_binding, Mapping) or not isinstance(
        codec_binding, Mapping
    ):
        raise PF1GraphPortsCodecPairError(
            "target release lacks its source/codec provenance"
        )
    observed_codec_sha256 = _sha256_file(Path(graph_v2.__file__).resolve())
    if not (
        source_binding.get("manifest_sha256") == source_manifest_sha256
        and codec_binding.get("source_format_version") == graph_v1.FORMAT_VERSION
        and codec_binding.get("target_format_version") == graph_v2.FORMAT_VERSION
        and codec_binding.get("target_source_sha256") == observed_codec_sha256
    ):
        raise PF1GraphPortsCodecPairError(
            "target source-release or codec provenance differs"
        )
    return {
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "target_codec_source_sha256": observed_codec_sha256,
        "source_format_version": graph_v1.FORMAT_VERSION,
        "target_format_version": graph_v2.FORMAT_VERSION,
    }


def _require_record_pair(source: Any, target: Any) -> None:
    if not (
        source.schedule_index == target.schedule_index
        and source.sdf_record_index == target.sdf_record_index
        and source.atom_record == target.atom_record
        and source.receipt == target.receipt
    ):
        raise PF1GraphPortsCodecPairError(
            "paired schedule, source row, atom record, or receipt differs"
        )
    left = source.motif_record
    right = target.motif_record
    shared_fields = (
        "record_id",
        "storage_key",
        "release_id",
        "geometry_record_content_sha256",
        "tokenizer_contract_sha256",
        "tokenizer_snapshot_sha256",
        "identity_spans",
        "logical_to_carrier",
        "exact_identity_sha256",
        "source_atom_count",
        "full_e3fp_ids",
        "atom_valid_mask",
        "model_to_source_atom_index",
        "atom_to_logical_motif",
    )
    if any(getattr(left, field) != getattr(right, field) for field in shared_fields):
        raise PF1GraphPortsCodecPairError(
            "a non-connection motif-record field differs across codecs"
        )
    if _identity_surfaces(left) != _identity_surfaces(right):
        raise PF1GraphPortsCodecPairError("motif identity token surfaces differ")
    if tuple(map(len, left.connection_token_indices)) != tuple(
        map(len, right.connection_token_indices)
    ):
        raise PF1GraphPortsCodecPairError(
            "connection ownership cardinality differs across codecs"
        )
    if len(right.input_ids) > len(left.input_ids):
        raise PF1GraphPortsCodecPairError("the candidate graph surface became longer")


def _require_corruption_pair(
    source_record: Any,
    target_record: Any,
    *,
    tokenizer_runtime: Any,
    seed: int,
    epoch: int,
) -> tuple[int, int, int, int]:
    source = collate_production_motif_record(
        source_record,
        tokenizer=tokenizer_runtime,
        seed=seed,
        epoch=epoch,
        mask_probability=MASK_PROBABILITY,
    )
    target = collate_production_motif_record(
        target_record,
        tokenizer=tokenizer_runtime,
        seed=seed,
        epoch=epoch,
        mask_probability=MASK_PROBABILITY,
    )
    shared_fields = (
        "record_id",
        "storage_key",
        "objective",
        "seed",
        "epoch",
        "mask_probability",
        "mask_decision_sha256",
        "geometry_record_content_sha256",
        "tokenizer_snapshot_sha256",
        "labels",
        "identity_recovery_mask",
        "selected_logical_motif_ids_in_input_order",
        "identity_input_spans",
        "logical_to_carrier",
        "full_e3fp_ids",
        "atom_valid_mask",
        "model_to_source_atom_index",
        "atom_to_logical_motif",
        "atom_to_carrier",
    )
    if any(getattr(source, field) != getattr(target, field) for field in shared_fields):
        raise PF1GraphPortsCodecPairError(
            "mask choice, CE target, identity carrier, or geometry differs"
        )
    if len(target.input_ids) > len(source.input_ids):
        raise PF1GraphPortsCodecPairError(
            "candidate corrupted input is longer than the source"
        )
    return (
        len(source.selected_logical_motif_ids_in_input_order),
        len(source.labels),
        len(source.input_ids),
        len(target.input_ids),
    )


def _paired_batches(
    source: Iterable[Sequence[Any]], target: Iterable[Sequence[Any]]
) -> Iterator[tuple[tuple[Any, ...], tuple[Any, ...]]]:
    missing = object()
    for source_batch, target_batch in zip_longest(source, target, fillvalue=missing):
        if source_batch is missing or target_batch is missing:
            raise PF1GraphPortsCodecPairError("release batch counts differ")
        left = tuple(source_batch)
        right = tuple(target_batch)
        if not left or len(left) != len(right):
            raise PF1GraphPortsCodecPairError("paired batch sizes differ")
        yield left, right


def _validate_split(
    *,
    source_batches: Iterable[Sequence[Any]],
    target_batches: Iterable[Sequence[Any]],
    tokenizer_runtime: Any,
    corruption_keys: Sequence[tuple[int, int]],
) -> _PairCounts:
    counts = _PairCounts()
    for source_batch, target_batch in _paired_batches(source_batches, target_batches):
        for source, target in zip(source_batch, target_batch):
            _require_record_pair(source, target)
            counts.records += 1
            for seed, epoch in corruption_keys:
                selected, target_tokens, source_tokens, candidate_tokens = (
                    _require_corruption_pair(
                        source.motif_record,
                        target.motif_record,
                        tokenizer_runtime=tokenizer_runtime,
                        seed=seed,
                        epoch=epoch,
                    )
                )
                counts.corruption_views += 1
                counts.selected_motifs += selected
                counts.target_tokens += target_tokens
                counts.source_input_tokens += source_tokens
                counts.target_input_tokens += candidate_tokens
    return counts


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise PF1GraphPortsCodecPairError("paired release manifest is absent")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("status") != "pass":
        raise PF1GraphPortsCodecPairError("paired release manifest is not passed")
    return document


def _require_manifest_pair(
    source_manifest: Mapping[str, Any], target_manifest: Mapping[str, Any]
) -> None:
    """Compare scientific bindings while allowing source-only census fields."""

    count_fields = (
        "scheduled_members",
        "paired_records",
        "train_members",
        "dev_members",
    )
    source_counts = source_manifest.get("counts", {})
    target_counts = target_manifest.get("counts", {})
    if not isinstance(source_counts, Mapping) or not isinstance(target_counts, Mapping):
        raise PF1GraphPortsCodecPairError("paired release counts are invalid")
    if any(source_counts.get(field) != target_counts.get(field) for field in count_fields):
        raise PF1GraphPortsCodecPairError("paired release counts differ")
    source_rejects = source_counts.get("rejects", source_counts.get("rejected_members"))
    target_rejects = target_counts.get("rejects", target_counts.get("rejected_members"))
    if source_rejects != 0 or target_rejects != 0:
        raise PF1GraphPortsCodecPairError("paired releases must both have zero rejects")
    source_tokenizer = source_manifest.get("artifacts", {}).get("union_tokenizer")
    target_tokenizer = target_manifest.get("artifacts", {}).get("union_tokenizer")
    tokenizer_fields = (
        "tokenizer_contract_sha256",
        "tokenizer_snapshot_sha256",
    )
    if not isinstance(source_tokenizer, Mapping) or not isinstance(
        target_tokenizer, Mapping
    ) or any(
        source_tokenizer.get(field) != target_tokenizer.get(field)
        for field in tokenizer_fields
    ):
        raise PF1GraphPortsCodecPairError("union tokenizer artifact bindings differ")


def validate_releases(
    *,
    source_release: Path,
    target_release: Path,
    base_tokenizer_snapshot: Path,
    output_report: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Replay the complete v1/v2 corruption domain and publish one report."""

    source_release = Path(source_release).expanduser().resolve()
    target_release = Path(target_release).expanduser().resolve()
    base_tokenizer_snapshot = Path(base_tokenizer_snapshot).expanduser().resolve()
    output_report = Path(output_report).expanduser().resolve()
    if output_report.exists():
        raise PF1GraphPortsCodecPairError("output_report must be a new path")
    if batch_size != DEFAULT_BATCH_SIZE:
        raise PF1GraphPortsCodecPairError(
            "the frozen codec gate requires microbatch size 64"
        )

    source_manifest = _read_manifest(source_release)
    target_manifest = _read_manifest(target_release)
    _require_manifest_pair(source_manifest, target_manifest)
    provenance = _require_release_provenance(
        source_release=source_release,
        target_release=target_release,
        target_manifest=target_manifest,
    )

    tokenizer_build = load_verified_canary_union_tokenizer(
        base_snapshot=base_tokenizer_snapshot,
        output_dir=source_release / TOKENIZER_DIRECTORY,
    )
    target_tokenizer_build = load_verified_canary_union_tokenizer(
        base_snapshot=base_tokenizer_snapshot,
        output_dir=target_release / TOKENIZER_DIRECTORY,
    )
    if tokenizer_build.runtime != target_tokenizer_build.runtime:
        raise PF1GraphPortsCodecPairError("verified tokenizer runtimes differ")
    source_reader = PF1PairedReleaseReader(source_release)
    target_reader = PF1PairedReleaseReader(target_release)
    if (
        source_reader.train_member_count != target_reader.train_member_count
        or source_reader.dev_member_count != target_reader.dev_member_count
    ):
        raise PF1GraphPortsCodecPairError("reader split sizes differ")
    batches_per_epoch = (
        source_reader.train_member_count + DEFAULT_BATCH_SIZE - 1
    ) // DEFAULT_BATCH_SIZE
    consumed_microbatches = (
        TRAIN_GRADIENT_ACCUMULATION_STEPS * TRAIN_OPTIMIZER_UPDATES
    )
    reached_epochs = tuple(
        range((consumed_microbatches - 1) // batches_per_epoch + 1)
    )
    if reached_epochs != TRAIN_CORRUPTION_EPOCHS:
        raise PF1GraphPortsCodecPairError(
            "train size no longer derives the preregistered corruption epochs"
        )
    raw_graph_replay = _validate_raw_graph_release_pair(
        source_reader,
        target_reader,
        tokenizer=tokenizer_build.tokenizer,
    )

    train = _validate_split(
        source_batches=source_reader.iter_train_epoch(epoch=0, batch_size=batch_size),
        target_batches=target_reader.iter_train_epoch(epoch=0, batch_size=batch_size),
        tokenizer_runtime=tokenizer_build.runtime,
        corruption_keys=tuple(
            (TRAIN_CORRUPTION_SEED, epoch) for epoch in TRAIN_CORRUPTION_EPOCHS
        ),
    )
    dev = _validate_split(
        source_batches=source_reader.iter_dev(batch_size=batch_size),
        target_batches=target_reader.iter_dev(batch_size=batch_size),
        tokenizer_runtime=tokenizer_build.runtime,
        corruption_keys=((DEV_CORRUPTION_SEED, DEV_CORRUPTION_EPOCH),),
    )
    if (
        train.records != source_reader.train_member_count
        or dev.records != source_reader.dev_member_count
    ):
        raise PF1GraphPortsCodecPairError("complete split replay did not close")

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "pf1_graphports_v1_v2_codec_pair_preflight",
        "training_admission": False,
        "source_release_scope": source_manifest.get("scope"),
        "target_release_scope": target_manifest.get("scope"),
        "protocol": {
            "train_corruption_seed": TRAIN_CORRUPTION_SEED,
            "train_corruption_epochs": list(TRAIN_CORRUPTION_EPOCHS),
            "train_micro_batch_size": DEFAULT_BATCH_SIZE,
            "train_gradient_accumulation_steps": (
                TRAIN_GRADIENT_ACCUMULATION_STEPS
            ),
            "train_optimizer_updates": TRAIN_OPTIMIZER_UPDATES,
            "train_microbatches_per_epoch": batches_per_epoch,
            "derived_reached_epochs": list(reached_epochs),
            "dev_corruption_seed": DEV_CORRUPTION_SEED,
            "dev_corruption_epoch": DEV_CORRUPTION_EPOCH,
            "mask_probability": MASK_PROBABILITY,
            "batch_size": batch_size,
            "no_replacement": True,
            "sequence_truncation": False,
        },
        "counts": {
            "train": train.as_dict(),
            "dev": dev.as_dict(),
            "total_records": train.records + dev.records,
            "total_corruption_views": train.corruption_views + dev.corruption_views,
        },
        "raw_graph_replay": raw_graph_replay,
        "artifact_bindings": provenance,
        "invariants": {
            "same_member_and_split_order": True,
            "same_atom_record_and_pair_receipt": True,
            "same_motif_identity_tokens": True,
            "same_geometry_and_e3fp": True,
            "same_mask_decision_every_view": True,
            "same_ce_labels_every_view": True,
            "candidate_input_never_longer": True,
            "same_union_tokenizer": True,
            "v1_v2_graph_tokens_decode_to_same_cross_motif_bonds": True,
            "source_manifest_and_target_codec_provenance_bound": True,
        },
        "decision_boundary": {
            "eligible_for_paired_m0_codec_screen": True,
            "automatic_mainline_promotion": False,
            "raw_atom_vs_motif_ce_comparison_authorized": False,
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    staging = output_report.with_name(output_report.name + ".staging")
    if staging.exists():
        raise PF1GraphPortsCodecPairError("output report staging path exists")
    with staging.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    staging.rename(output_report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", required=True)
    parser.add_argument("--target-release", required=True)
    parser.add_argument("--base-tokenizer-snapshot", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_releases(
        source_release=Path(args.source_release),
        target_release=Path(args.target_release),
        base_tokenizer_snapshot=Path(args.base_tokenizer_snapshot),
        output_report=Path(args.output_report),
        batch_size=args.batch_size,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PF1GraphPortsCodecPairError",
    "TRAIN_CORRUPTION_EPOCHS",
    "validate_releases",
]
