#!/usr/bin/env python3
"""Build an exact full-corpus anchored-motif vocabulary trade-off report.

The authoritative geometry release remains read-only.  A compact cache stores
only the final-v4 member ordinal, per-record pure-motif IDs, offsets and anchor
counts.  Random corruption, token padding and model inputs are deliberately not
materialized here.
"""

from __future__ import annotations

import argparse
from array import array
import concurrent.futures
from collections import Counter
import hashlib
import json
import multiprocessing
from pathlib import Path
import struct
import sys
import time
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "most-t5-next/full-anchored-vocab-analysis/v1"
MEMBER_PREFIX = "ogb_pcqm4mv2_train_row_index:"
PERMITTED_SCHEMA = "most-t5-r1/permitted-pretrain-member/v1"
DEFAULT_BUDGETS = "512,2048,4096,8192,12000,16000,24735,30080,32768"


class FullAnchoredVocabError(RuntimeError):
    """The declared release cannot support an exact vocabulary analysis."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_ordinal(member_id: object) -> int:
    if not isinstance(member_id, str) or not member_id.startswith(MEMBER_PREFIX):
        raise FullAnchoredVocabError("invalid permitted member ID")
    suffix = member_id[len(MEMBER_PREFIX) :]
    if not suffix.isdigit() or (len(suffix) > 1 and suffix.startswith("0")):
        raise FullAnchoredVocabError("noncanonical permitted member ordinal")
    return int(suffix)


def _load_permitted_mask(path: Path, expected: int | None) -> tuple[bytearray, int]:
    ordinals: list[int] = []
    maximum = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("schema_version") != PERMITTED_SCHEMA or set(row) != {
                "schema_version",
                "member_id",
            }:
                raise FullAnchoredVocabError(
                    f"invalid permitted membership at line {line_number}"
                )
            ordinal = _member_ordinal(row["member_id"])
            ordinals.append(ordinal)
            maximum = max(maximum, ordinal)
    if not ordinals or len(set(ordinals)) != len(ordinals):
        raise FullAnchoredVocabError("permitted membership is empty or duplicated")
    if expected is not None and len(ordinals) != expected:
        raise FullAnchoredVocabError("permitted membership count differs")
    mask = bytearray(maximum + 1)
    for ordinal in ordinals:
        mask[ordinal] = 1
    return mask, len(ordinals)


def _load_pure_registry(
    census_path: Path,
) -> tuple[dict[str, tuple[int, int]], tuple[str, ...]]:
    """Map exact-lexeme digest to (pure ID, anchor arity)."""

    from most_t5_next.r1.tokenizer.stereo_free_anchored_motif_surface_v1 import (
        canonicalize_legacy_fragment,
    )

    rows: list[tuple[str, str, int]] = []
    pure_values: set[str] = set()
    with census_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            digest = row.get("motif_lexeme_sha256")
            fragment = row.get("motif_fragment")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or not isinstance(fragment, str)
                or hashlib.sha256(fragment.encode("utf-8")).hexdigest() != digest
            ):
                raise FullAnchoredVocabError(
                    f"invalid global motif census row {line_number}"
                )
            pure, anchors = canonicalize_legacy_fragment(fragment)
            rows.append((digest, pure, len(anchors)))
            pure_values.add(pure)
    pure_registry = tuple(sorted(pure_values, key=lambda value: value.encode("utf-8")))
    pure_to_id = {value: index for index, value in enumerate(pure_registry)}
    mapping = {
        digest: (pure_to_id[pure], anchor_arity)
        for digest, pure, anchor_arity in rows
    }
    if len(mapping) != len(rows):
        raise FullAnchoredVocabError("global motif census repeats a digest")
    return mapping, pure_registry


def _extract_header_record(codec, payload: bytes) -> tuple[Mapping[str, object], str]:
    prefix = len(codec.MAGIC) + codec.HEADER_LENGTH_BYTES
    if len(payload) < prefix or payload[: len(codec.MAGIC)] != codec.MAGIC:
        raise FullAnchoredVocabError("production payload magic differs")
    header_size = struct.unpack(">I", payload[len(codec.MAGIC) : prefix])[0]
    if header_size < 2 or prefix + header_size > len(payload):
        raise FullAnchoredVocabError("production payload header length is invalid")
    header_bytes = payload[prefix : prefix + header_size]
    header = json.loads(header_bytes)
    if not isinstance(header, dict) or header.get("payload_schema_version") != codec.PAYLOAD_SCHEMA:
        raise FullAnchoredVocabError("production payload schema differs")
    record = header.get("record")
    logical_hash = header.get("logical_record_sha256")
    if not isinstance(record, dict) or not isinstance(logical_hash, str):
        raise FullAnchoredVocabError("production payload header is malformed")
    return record, logical_hash


_WORKER: dict[str, object] = {}


def _init_shard_worker(
    permitted_mask: bytes,
    digest_rows: Mapping[str, tuple[int, int]],
    cache_dir: str,
) -> None:
    _WORKER.clear()
    _WORKER.update(
        {
            "permitted": permitted_mask,
            "digest_rows": dict(digest_rows),
            "cache_dir": cache_dir,
        }
    )


def _process_shard(shard_path_text: str) -> dict[str, object]:
    import lmdb

    from most_t5_next.r1.adapter import sidecar_v2_codec as codec

    shard = Path(shard_path_text)
    permitted = _WORKER["permitted"]
    digest_rows = _WORKER["digest_rows"]
    cache_dir = Path(str(_WORKER["cache_dir"]))
    if not isinstance(permitted, bytes) or not isinstance(digest_rows, dict):
        raise FullAnchoredVocabError("shard worker was not initialized")

    motif_ids = array("I")
    offsets = array("Q", [0])
    ordinals = array("I")
    anchor_counts = array("H")
    environment = lmdb.open(
        str(shard / "geometry_records.lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=True,
    )
    try:
        with environment.begin(write=False, buffers=True) as transaction:
            with (shard / "membership.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    ordinal = int(row["sdf_record_index"])
                    if ordinal >= len(permitted) or not permitted[ordinal]:
                        continue
                    if row.get("disposition") != "admit":
                        raise FullAnchoredVocabError("permitted member is not admitted")
                    raw = transaction.get(str(row["record_storage_key"]).encode("ascii"))
                    if raw is None:
                        raise FullAnchoredVocabError("permitted LMDB record is absent")
                    record, logical_hash = _extract_header_record(codec, bytes(raw))
                    if logical_hash != row.get("record_content_sha256"):
                        raise FullAnchoredVocabError("membership and payload hashes differ")
                    topology = record.get("topology")
                    if not isinstance(topology, dict):
                        raise FullAnchoredVocabError("production topology is absent")
                    digests = topology.get("motif_lexeme_sha256")
                    if not isinstance(digests, list) or not digests:
                        raise FullAnchoredVocabError("production motif sequence is absent")
                    record_anchor_count = 0
                    for digest in digests:
                        try:
                            pure_id, arity = digest_rows[digest]
                        except (KeyError, TypeError) as exc:
                            raise FullAnchoredVocabError(
                                "record motif digest is absent from global census"
                            ) from exc
                        motif_ids.append(int(pure_id))
                        record_anchor_count += int(arity)
                    if record_anchor_count > 65535:
                        raise FullAnchoredVocabError("record anchor count exceeds uint16")
                    ordinals.append(ordinal)
                    anchor_counts.append(record_anchor_count)
                    offsets.append(len(motif_ids))
    finally:
        environment.close()

    stem = shard.name
    paths = {
        "motif_ids": cache_dir / f"{stem}.motif_ids.u32",
        "offsets": cache_dir / f"{stem}.offsets.u64",
        "ordinals": cache_dir / f"{stem}.ordinals.u32",
        "anchor_counts": cache_dir / f"{stem}.anchor_counts.u16",
    }
    for key, values in (
        ("motif_ids", motif_ids),
        ("offsets", offsets),
        ("ordinals", ordinals),
        ("anchor_counts", anchor_counts),
    ):
        with paths[key].open("xb") as output:
            values.tofile(output)
    return {
        "shard": stem,
        "records": len(ordinals),
        "motifs": len(motif_ids),
        "artifacts": {key: str(path) for key, path in paths.items()},
    }


def _candidate_budgets(text: str, type_count: int) -> tuple[int, ...]:
    values = sorted({int(item) for item in text.split(",")})
    if not values or values[0] <= 0:
        raise FullAnchoredVocabError("budgets must be positive")
    return tuple(sorted({min(value, type_count) for value in values}))


def _quantile(values, probability: float) -> int:
    import numpy as np

    if len(values) == 0:
        return 0
    return int(np.quantile(values, probability, method="higher"))


def _analyze_cache(
    cache_dir: Path,
    pure_registry: Sequence[str],
    budgets_text: str,
    hidden_size: int,
    tie_word_embeddings: bool,
) -> tuple[dict[str, object], list[int], list[int]]:
    import numpy as np

    count = np.zeros(len(pure_registry), dtype=np.uint64)
    shards = sorted(cache_dir.glob("shard-*.motif_ids.u32"))
    if not shards:
        raise FullAnchoredVocabError("compact motif cache is empty")
    total_records = 0
    total_motifs = 0
    for motif_path in shards:
        motif_ids = np.fromfile(motif_path, dtype="<u4")
        count += np.bincount(motif_ids, minlength=len(pure_registry)).astype(np.uint64)
        stem = motif_path.name[: -len(".motif_ids.u32")]
        offsets = np.fromfile(cache_dir / f"{stem}.offsets.u64", dtype="<u8")
        ordinals = np.fromfile(cache_dir / f"{stem}.ordinals.u32", dtype="<u4")
        anchors = np.fromfile(cache_dir / f"{stem}.anchor_counts.u16", dtype="<u2")
        if (
            len(offsets) != len(ordinals) + 1
            or len(anchors) != len(ordinals)
            or offsets[0] != 0
            or offsets[-1] != len(motif_ids)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise FullAnchoredVocabError("compact motif cache arrays are inconsistent")
        total_records += len(ordinals)
        total_motifs += len(motif_ids)
    if int(count.sum()) != total_motifs:
        raise FullAnchoredVocabError("compact motif cache counts do not balance")
    ranking = sorted(
        range(len(pure_registry)),
        key=lambda index: (-int(count[index]), pure_registry[index].encode("utf-8")),
    )
    rank = np.empty(len(pure_registry), dtype=np.uint32)
    rank[np.asarray(ranking, dtype=np.uint32)] = np.arange(
        len(ranking), dtype=np.uint32
    )
    from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
        lex_pure_motif,
    )

    lexical_lengths = np.asarray(
        [len(lex_pure_motif(pure).tokens) + 1 for pure in pure_registry],
        dtype=np.uint32,
    )
    budgets = _candidate_budgets(budgets_text, len(pure_registry))
    rows: list[dict[str, object]] = []
    total_occurrences = int(count.sum())
    for budget in budgets:
        selected_ids = np.asarray(ranking[:budget], dtype=np.uint32)
        selected_mask = np.zeros(len(pure_registry), dtype=np.bool_)
        selected_mask[selected_ids] = True
        covered_occurrences = int(count[selected_ids].sum())
        molecule_count = 0
        fully_macro = 0
        fallback_le = {1: 0, 2: 0, 5: 0}
        fallback_total = 0
        phrase_lengths: list[int] = []
        for motif_path in shards:
            stem = motif_path.name[: -len(".motif_ids.u32")]
            motif_ids = np.fromfile(motif_path, dtype="<u4")
            offsets = np.fromfile(cache_dir / f"{stem}.offsets.u64", dtype="<u8")
            anchors = np.fromfile(
                cache_dir / f"{stem}.anchor_counts.u16", dtype="<u2"
            ).astype(np.uint64)
            starts = offsets[:-1].astype(np.int64, copy=False)
            is_fallback = ~selected_mask[motif_ids]
            fallback_counts = np.add.reduceat(
                is_fallback.astype(np.uint32), starts
            )
            token_cost = np.where(
                is_fallback,
                lexical_lengths[motif_ids],
                np.uint32(1),
            )
            identity_lengths = np.add.reduceat(
                token_cost.astype(np.uint64), starts
            )
            identity_lengths += anchors
            molecule_count += len(fallback_counts)
            fully_macro += int(np.count_nonzero(fallback_counts == 0))
            fallback_total += int(fallback_counts.sum())
            for threshold in fallback_le:
                fallback_le[threshold] += int(
                    np.count_nonzero(fallback_counts <= threshold)
                )
            phrase_lengths.extend(int(value) for value in identity_lengths)
        selected_frequencies = count[selected_ids]
        output_rows = 1 if tie_word_embeddings else 2
        rows.append(
            {
                "requested_budget": budget,
                "selected_macro_count": len(selected_ids),
                "macro_occurrence_coverage": covered_occurrences / total_occurrences,
                "pure_motif_type_coverage": len(selected_ids) / len(pure_registry),
                "fully_macro_tokenized_molecules": fully_macro,
                "fully_macro_tokenized_molecule_rate": fully_macro / molecule_count,
                "molecules_with_at_most_1_fallback_rate": fallback_le[1] / molecule_count,
                "molecules_with_at_most_2_fallback_rate": fallback_le[2] / molecule_count,
                "molecules_with_at_most_5_fallback_rate": fallback_le[5] / molecule_count,
                "mean_fallback_motifs_per_molecule": fallback_total / molecule_count,
                "identity_tokens_excluding_component_separators": {
                    "mean": sum(phrase_lengths) / len(phrase_lengths),
                    "p50": _quantile(phrase_lengths, 0.50),
                    "p95": _quantile(phrase_lengths, 0.95),
                    "p99": _quantile(phrase_lengths, 0.99),
                    "max": max(phrase_lengths),
                },
                "selected_frequency": {
                    "minimum": int(selected_frequencies.min()),
                    "median": _quantile(selected_frequencies, 0.50),
                    "types_below_8": int(np.count_nonzero(selected_frequencies < 8)),
                    "types_below_32": int(np.count_nonzero(selected_frequencies < 32)),
                    "types_below_100": int(np.count_nonzero(selected_frequencies < 100)),
                },
                "additional_untied_vocab_parameters": len(selected_ids)
                * hidden_size
                * output_rows,
            }
        )
    return (
        {
            "records": total_records,
            "pure_motif_occurrences": total_motifs,
            "unique_pure_motifs": int(np.count_nonzero(count)),
            "registered_but_absent_pure_motifs": int(np.count_nonzero(count == 0)),
            "budget_rows": rows,
        },
        [int(value) for value in ranking],
        [int(value) for value in count],
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    release = Path(args.release_root).expanduser().resolve()
    permitted_path = Path(args.permitted_membership).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise FullAnchoredVocabError("output and sibling staging must be absent")
    if args.workers <= 0:
        raise FullAnchoredVocabError("workers must be positive")
    staging.mkdir(parents=False)
    cache_dir = staging / "compact_motif_cache"
    cache_dir.mkdir()
    started = time.perf_counter()
    permitted_mask, permitted_count = _load_permitted_mask(
        permitted_path, args.expected_members
    )
    digest_rows, pure_registry = _load_pure_registry(release / "motif_census.jsonl")
    shards = sorted(path for path in release.glob("shard-*") if path.is_dir())
    if not shards:
        raise FullAnchoredVocabError("production release contains no shards")
    context = multiprocessing.get_context("spawn")
    receipts = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_shard_worker,
        initargs=(bytes(permitted_mask), digest_rows, str(cache_dir)),
    ) as executor:
        for receipt in executor.map(
            _process_shard, (str(path) for path in shards), chunksize=1
        ):
            receipts.append(receipt)
            if args.progress_every_shards and len(receipts) % args.progress_every_shards == 0:
                print(f"full-vocab-cache {len(receipts)}/{len(shards)} shards", flush=True)
    if sum(int(row["records"]) for row in receipts) != permitted_count:
        raise FullAnchoredVocabError("compact cache record count differs from membership")
    analysis, ranking, counts = _analyze_cache(
        cache_dir,
        pure_registry,
        args.budgets,
        args.hidden_size,
        args.tie_word_embeddings,
    )
    registry_path = staging / "pure_motif_registry.jsonl"
    with registry_path.open("x", encoding="utf-8", newline="\n") as handle:
        for rank, pure_id in enumerate(ranking):
            pure = pure_registry[pure_id]
            if counts[pure_id] == 0:
                continue
            handle.write(
                _canonical_json(
                    {
                        "rank": rank,
                        "pure_motif_id": pure_id,
                        "pure_motif": pure,
                        "pure_motif_sha256": hashlib.sha256(
                            pure.encode("utf-8")
                        ).hexdigest(),
                        "occurrences": counts[pure_id],
                    }
                )
                + "\n"
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "full_final_v4_pretraining_support",
        "inputs": {
            "release_root": str(release),
            "release_manifest_sha256": _sha256_file(
                release / "full_release_manifest.json"
            ),
            "global_motif_census_sha256": _sha256_file(
                release / "motif_census.jsonl"
            ),
            "permitted_membership": str(permitted_path),
            "permitted_membership_sha256": _sha256_file(permitted_path),
        },
        "runtime": {
            "workers": args.workers,
            "shards": len(shards),
            "wall_seconds": time.perf_counter() - started,
        },
        "analysis": analysis,
        "model_cost": {
            "hidden_size": args.hidden_size,
            "tie_word_embeddings": args.tie_word_embeddings,
            "rows_per_token": 1 if args.tie_word_embeddings else 2,
        },
        "contracts": {
            "release_is_read_only": True,
            "sdf_scan_reexecuted": False,
            "e3fp_recomputed": False,
            "validation_or_test_influences_ranking": False,
            "ranking": "descending_full_pretraining_occurrence_then_utf8_identity",
            "chemical_lexer_preserves_lossless_representability": True,
            "fully_macro_tokenized_rate_is_distinct_from_lossless_representability": True,
            "training_admission": False,
        },
        "artifacts": {
            "pure_motif_registry": {
                "path": registry_path.name,
                "bytes": registry_path.stat().st_size,
                "sha256": _sha256_file(registry_path),
            }
        },
    }
    report_path = staging / "report.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    staging.rename(output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--permitted-membership", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-members", type=int, default=3_360_067)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-every-shards", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"full anchored vocab analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
