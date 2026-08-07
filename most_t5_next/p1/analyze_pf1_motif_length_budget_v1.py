"""Analyze PF-1 motif length, graph lower bounds, and train-only macro budgets.

The analysis reads one already-published paired release.  It never rebuilds
chemistry, changes tokenizer state, or consumes dev frequencies when ranking
motif macros.  Persisted identities are recovered losslessly from either the
opaque macro registry or the GraphPorts UTF-8 fallback surface.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPORT_SCHEMA = "most-t5-p1/pf1-motif-length-budget/v1"
DEFAULT_K_VALUES = (0, 32, 64, 128, 256, 512, 1024, 1536, 2150, 4096)
FALLBACK_BEGIN = "<GPORTS:FALLBACK:BEGIN>"
FALLBACK_END = "<GPORTS:FALLBACK:END>"
BYTE_PREFIX = "<GPORTS:B"


class PF1MotifLengthBudgetError(RuntimeError):
    """Raised when a published release cannot support the analysis."""


@dataclass(frozen=True)
class _Record:
    split: str
    member_id: str
    atom_tokens: int
    persisted_motif_tokens: int
    graph_tokens: int
    edge_count: int
    atom_count: int
    motif_count: int
    identities: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PF1MotifLengthBudgetError("cannot read {}".format(path)) from exc
    if not isinstance(value, dict):
        raise PF1MotifLengthBudgetError("{} is not a JSON object".format(path))
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise PF1MotifLengthBudgetError("cannot read {}".format(path)) from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PF1MotifLengthBudgetError(
                    "{}:{} is not JSON".format(path, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise PF1MotifLengthBudgetError(
                    "{}:{} is not an object".format(path, line_number)
                )
            yield row


def _quantile(sorted_values: Sequence[int], probability: float) -> int:
    if not sorted_values:
        raise PF1MotifLengthBudgetError("cannot summarize an empty population")
    index = int(round((len(sorted_values) - 1) * probability))
    return int(sorted_values[index])


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        raise PF1MotifLengthBudgetError("cannot summarize an empty population")
    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": int(ordered[-1]),
    }


def _decode_fallback(tokens: Sequence[str]) -> str:
    if len(tokens) < 3 or tokens[0] != FALLBACK_BEGIN or tokens[-1] != FALLBACK_END:
        raise PF1MotifLengthBudgetError("fallback identity framing is invalid")
    payload = bytearray()
    for token in tokens[1:-1]:
        if not (
            token.startswith(BYTE_PREFIX)
            and token.endswith(">")
            and len(token) == len(BYTE_PREFIX) + 3
        ):
            raise PF1MotifLengthBudgetError("fallback identity contains a non-byte token")
        try:
            payload.append(int(token[len(BYTE_PREFIX) : -1], 16))
        except ValueError as exc:
            raise PF1MotifLengthBudgetError("fallback byte token is invalid") from exc
    try:
        return bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PF1MotifLengthBudgetError("fallback identity is not UTF-8") from exc


def _decode_identity(
    input_ids: Sequence[int],
    span: Sequence[int],
    *,
    id_to_token: Mapping[int, str],
    macro_identity_by_token: Mapping[str, str],
) -> str:
    if (
        len(span) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in span)
        or not 0 <= span[0] < span[1] <= len(input_ids)
    ):
        raise PF1MotifLengthBudgetError("motif identity span is invalid")
    try:
        tokens = tuple(id_to_token[int(token_id)] for token_id in input_ids[span[0] : span[1]])
    except (KeyError, TypeError, ValueError) as exc:
        raise PF1MotifLengthBudgetError("identity token id is not declared") from exc
    if len(tokens) == 1 and tokens[0] in macro_identity_by_token:
        return macro_identity_by_token[tokens[0]]
    return _decode_fallback(tokens)


def _record_from_wire(
    document: Mapping[str, Any],
    membership: Mapping[str, Any],
    *,
    id_to_token: Mapping[int, str],
    macro_identity_by_token: Mapping[str, str],
) -> _Record:
    try:
        summary = document["surface_summary"]
        motif = document["motif_training_document"]
        token_domain = motif["token_domain"]
        logical = motif["logical_motif_domain"]
        dimensions = motif["dimensions"]
        input_ids = token_domain["input_ids"]
        spans = logical["identity_spans"]
        identity_digests = logical["exact_identity_sha256"]
        edges = logical["cross_motif_bonds"]
        persisted_identity_counts = summary["motif_identity_token_counts"]
    except (KeyError, TypeError) as exc:
        raise PF1MotifLengthBudgetError("paired wire lacks motif length fields") from exc
    identities = tuple(
        _decode_identity(
            input_ids,
            span,
            id_to_token=id_to_token,
            macro_identity_by_token=macro_identity_by_token,
        )
        for span in spans
    )
    if not (
        len(identities)
        == len(identity_digests)
        == len(persisted_identity_counts)
        == int(dimensions["logical_motif_count"])
    ):
        raise PF1MotifLengthBudgetError("motif identity domains differ in length")
    for identity, digest in zip(identities, identity_digests):
        observed = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if observed != digest:
            raise PF1MotifLengthBudgetError("decoded motif identity digest differs")
    observed_span_lengths = [int(span[1]) - int(span[0]) for span in spans]
    if observed_span_lengths != list(persisted_identity_counts):
        raise PF1MotifLengthBudgetError("identity span length differs from summary")
    edge_count = len(edges)
    graph_tokens = int(summary["graph_token_count"])
    if graph_tokens != 4 + 2 * edge_count:
        raise PF1MotifLengthBudgetError(
            "GraphPorts v2 is not at its declared 4+2E byte-grammar boundary"
        )
    persisted_motif_tokens = int(summary["motif_input_token_count"])
    if persisted_motif_tokens != 2 + graph_tokens + sum(observed_span_lengths):
        raise PF1MotifLengthBudgetError("motif input length does not close")
    split = str(membership.get("split"))
    member_id = str(membership.get("member_id"))
    if split not in {"train", "dev"} or not member_id:
        raise PF1MotifLengthBudgetError("membership identity or split is invalid")
    return _Record(
        split=split,
        member_id=member_id,
        atom_tokens=int(summary["atom_input_token_count"]),
        persisted_motif_tokens=persisted_motif_tokens,
        graph_tokens=graph_tokens,
        edge_count=edge_count,
        atom_count=int(dimensions["atom_count"]),
        motif_count=len(identities),
        identities=identities,
    )


def _fallback_length(identity: str) -> int:
    return 2 + len(identity.encode("utf-8"))


def _length_for_macros(record: _Record, macros: set[str]) -> tuple[int, int, int]:
    macro_occurrences = sum(identity in macros for identity in record.identities)
    identity_tokens = sum(
        1 if identity in macros else _fallback_length(identity)
        for identity in record.identities
    )
    return 2 + record.graph_tokens + identity_tokens, identity_tokens, macro_occurrences


def _summarize_budget(records: Sequence[_Record], macros: set[str]) -> dict[str, Any]:
    motif_lengths: list[int] = []
    identity_lengths: list[int] = []
    macro_occurrences = 0
    total_occurrences = 0
    motif_longer = 0
    squared_tokens: list[int] = []
    for record in records:
        motif_length, identity_length, record_macro_occurrences = _length_for_macros(
            record, macros
        )
        motif_lengths.append(motif_length)
        identity_lengths.append(identity_length)
        macro_occurrences += record_macro_occurrences
        total_occurrences += len(record.identities)
        motif_longer += int(motif_length > record.atom_tokens)
        squared_tokens.append(motif_length * motif_length)
    total_identity_tokens = sum(identity_lengths)
    fallback_identity_tokens = total_identity_tokens - macro_occurrences
    return {
        "records": len(records),
        "motif_input_tokens": _distribution(motif_lengths),
        "motif_identity_tokens": _distribution(identity_lengths),
        "mean_attention_length_squared": statistics.fmean(squared_tokens),
        "motif_longer_than_atom_records": motif_longer,
        "motif_longer_than_atom_fraction": motif_longer / len(records),
        "macro_occurrences": macro_occurrences,
        "motif_identity_occurrences": total_occurrences,
        "macro_occurrence_coverage": macro_occurrences / total_occurrences,
        "fallback_occurrences": total_occurrences - macro_occurrences,
        "total_motif_identity_tokens": total_identity_tokens,
        "macro_identity_tokens": macro_occurrences,
        "fallback_identity_tokens": fallback_identity_tokens,
        "fallback_identity_token_fraction": fallback_identity_tokens
        / total_identity_tokens,
    }


def _bucket_summary(
    records: Sequence[_Record],
    *,
    macros: set[str],
    buckets: Sequence[tuple[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for label, predicate in buckets:
        selected = [record for record in records if predicate(record)]
        if not selected:
            continue
        summary = _summarize_budget(selected, macros)
        rows.append(
            {
                "bucket": label,
                "records": len(selected),
                "mean_atom_tokens": statistics.fmean(
                    record.atom_tokens for record in selected
                ),
                "mean_motif_tokens": summary["motif_input_tokens"]["mean"],
                "motif_longer_than_atom_fraction": summary[
                    "motif_longer_than_atom_fraction"
                ],
            }
        )
    return rows


def analyze_release(
    release_root: Path,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    lmdb_module: Any | None = None,
) -> dict[str, Any]:
    release_root = Path(release_root).expanduser().resolve()
    manifest = _load_json(release_root / "manifest.json")
    if manifest.get("status") != "pass":
        raise PF1MotifLengthBudgetError("paired release is not passed")
    macro_document = _load_json(release_root / "macro_registry.json")
    macro_rows = macro_document.get("rows")
    if not isinstance(macro_rows, list):
        raise PF1MotifLengthBudgetError("macro registry rows are absent")
    macro_identity_by_token: dict[str, str] = {}
    persisted_macros: list[str] = []
    for expected_rank, row in enumerate(macro_rows):
        if not (
            isinstance(row, dict)
            and row.get("rank") == expected_rank
            and isinstance(row.get("identity"), str)
            and isinstance(row.get("token"), str)
        ):
            raise PF1MotifLengthBudgetError("macro registry order is invalid")
        identity = str(row["identity"])
        token = str(row["token"])
        macro_identity_by_token[token] = identity
        persisted_macros.append(identity)
    tokenizer_manifest = _load_json(release_root / "union_tokenizer" / "manifest.json")
    try:
        declared = tokenizer_manifest["token_ids"]["declared"]
        id_to_token = {int(token_id): str(token) for token, token_id in declared.items()}
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise PF1MotifLengthBudgetError("tokenizer declared-token map is invalid") from exc
    membership_rows = list(_iter_jsonl(release_root / "train_membership.jsonl"))
    membership_rows.extend(_iter_jsonl(release_root / "dev_membership.jsonl"))
    if lmdb_module is None:
        try:
            import lmdb as lmdb_module
        except ImportError as exc:
            raise PF1MotifLengthBudgetError("python-lmdb is required") from exc
    environment = lmdb_module.open(
        str(release_root / "paired_records.lmdb"),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=2,
    )
    records: list[_Record] = []
    try:
        with environment.begin(write=False) as transaction:
            for row in membership_rows:
                storage_key = row.get("storage_key")
                if not isinstance(storage_key, str):
                    raise PF1MotifLengthBudgetError("membership storage key is invalid")
                raw = transaction.get(storage_key.encode("ascii"))
                if raw is None:
                    raise PF1MotifLengthBudgetError("paired LMDB row is absent")
                try:
                    document = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PF1MotifLengthBudgetError("paired LMDB row is not JSON") from exc
                records.append(
                    _record_from_wire(
                        document,
                        row,
                        id_to_token=id_to_token,
                        macro_identity_by_token=macro_identity_by_token,
                    )
                )
    finally:
        environment.close()
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    counts = manifest.get("counts", {})
    if not (
        len(train_records) == counts.get("train_members")
        and len(dev_records) == counts.get("dev_members")
        and len(records) == counts.get("paired_records")
    ):
        raise PF1MotifLengthBudgetError("release counts differ from decoded rows")
    train_identity_counts = Counter(
        identity for record in train_records for identity in record.identities
    )
    ranked_identities = sorted(
        train_identity_counts,
        key=lambda identity: (-train_identity_counts[identity], identity.encode("utf-8")),
    )
    persisted_k = len(persisted_macros)
    if persisted_macros != ranked_identities[:persisted_k]:
        raise PF1MotifLengthBudgetError("persisted macros differ from train-only ranking")
    persisted_macro_set = set(persisted_macros)
    for record in records:
        recomputed, _identity_tokens, _macro_occurrences = _length_for_macros(
            record, persisted_macro_set
        )
        if recomputed != record.persisted_motif_tokens:
            raise PF1MotifLengthBudgetError("persisted motif surface differs from K policy")

    requested_k = {int(value) for value in k_values}
    if any(value < 0 for value in requested_k):
        raise PF1MotifLengthBudgetError("macro K values must be non-negative")
    requested_k.update({persisted_k, len(ranked_identities)})
    effective_k_values = sorted(min(value, len(ranked_identities)) for value in requested_k)
    effective_k_values = sorted(set(effective_k_values))
    budget_rows = []
    for k_value in effective_k_values:
        macros = set(ranked_identities[:k_value])
        budget_rows.append(
            {
                "k": k_value,
                "macro_vocabulary_rows": k_value,
                "train": _summarize_budget(train_records, macros),
                "dev": _summarize_budget(dev_records, macros),
                "all": _summarize_budget(records, macros),
            }
        )

    current_summary = _summarize_budget(records, persisted_macro_set)
    if any(record.edge_count >= record.motif_count for record in records):
        raise PF1MotifLengthBudgetError(
            "cross-motif connection graph is not a forest in the audited domain"
        )
    implied_component_counts = [
        record.motif_count - record.edge_count for record in records
    ]
    motif_without_graph = [
        2
        + sum(
            1 if identity in persisted_macro_set else _fallback_length(identity)
            for identity in record.identities
        )
        for record in records
    ]
    one_token_per_motif = [
        2 + record.graph_tokens + record.motif_count for record in records
    ]
    one_token_per_motif_without_graph = [
        2 + record.motif_count for record in records
    ]
    headerless_lengths = [record.persisted_motif_tokens - 4 for record in records]
    top_excess = sorted(
        records,
        key=lambda record: (
            record.persisted_motif_tokens - record.atom_tokens,
            record.member_id,
        ),
        reverse=True,
    )[:20]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "scope": "pf1_one_percent_length_and_vocabulary_diagnostic",
        "training_admission": False,
        "release": {
            "path": str(release_root),
            "records": len(records),
            "train_records": len(train_records),
            "dev_records": len(dev_records),
        },
        "observed": {
            "atom_input_tokens": _distribution([record.atom_tokens for record in records]),
            "motif_input_tokens": _distribution(
                [record.persisted_motif_tokens for record in records]
            ),
            "motif_identity_tokens": current_summary["motif_identity_tokens"],
            "graph_tokens": _distribution([record.graph_tokens for record in records]),
            "atom_count": _distribution([record.atom_count for record in records]),
            "motif_count": _distribution([record.motif_count for record in records]),
            "cross_motif_edges": _distribution([record.edge_count for record in records]),
            "implied_motif_graph_components": _distribution(implied_component_counts),
            "motif_graph_is_forest_for_every_record": True,
            "motif_longer_than_atom_fraction": current_summary[
                "motif_longer_than_atom_fraction"
            ],
            "persisted_macro_k": persisted_k,
            "persisted_macro_occurrence_coverage": current_summary[
                "macro_occurrence_coverage"
            ],
            "persisted_fallback_identity_token_fraction": current_summary[
                "fallback_identity_token_fraction"
            ],
            "train_unique_motif_identities": len(ranked_identities),
        },
        "graph_lower_bound": {
            "grammar": "4 + 2 * cross_motif_edges",
            "records_exactly_at_bound": len(records),
            "records_above_bound": 0,
            "fixed_header_tokens_per_record": 4,
            "endpoint_tokens_per_edge": 2,
            "headerless_unattainable_bound": {
                "motif_input_tokens": _distribution(headerless_lengths),
                "motif_longer_than_atom_fraction": sum(
                    length > record.atom_tokens
                    for length, record in zip(headerless_lengths, records)
                )
                / len(records),
            },
            "identity_only_no_graph_diagnostic": {
                "motif_input_tokens": _distribution(motif_without_graph),
                "motif_longer_than_atom_fraction": sum(
                    length > record.atom_tokens
                    for length, record in zip(motif_without_graph, records)
                )
                / len(records),
            },
            "one_token_per_motif_bound": {
                "motif_input_tokens": _distribution(one_token_per_motif),
                "motif_longer_than_atom_fraction": sum(
                    length > record.atom_tokens
                    for length, record in zip(one_token_per_motif, records)
                )
                / len(records),
            },
            "one_token_per_motif_no_graph_diagnostic": {
                "motif_input_tokens": _distribution(
                    one_token_per_motif_without_graph
                ),
                "motif_longer_than_atom_fraction": sum(
                    length > record.atom_tokens
                    for length, record in zip(
                        one_token_per_motif_without_graph, records
                    )
                )
                / len(records),
            },
        },
        "macro_budget": {
            "ranking_source": "train identity occurrence count, then UTF-8 identity",
            "dev_used_for_ranking": False,
            "rows": budget_rows,
        },
        "stratified_current_k": {
            "by_cross_motif_edges": _bucket_summary(
                records,
                macros=persisted_macro_set,
                buckets=(
                    ("0", lambda record: record.edge_count == 0),
                    ("1-2", lambda record: 1 <= record.edge_count <= 2),
                    ("3-4", lambda record: 3 <= record.edge_count <= 4),
                    ("5-8", lambda record: 5 <= record.edge_count <= 8),
                    ("9+", lambda record: record.edge_count >= 9),
                ),
            ),
            "by_motif_count": _bucket_summary(
                records,
                macros=persisted_macro_set,
                buckets=(
                    ("1", lambda record: record.motif_count == 1),
                    ("2-3", lambda record: 2 <= record.motif_count <= 3),
                    ("4-6", lambda record: 4 <= record.motif_count <= 6),
                    ("7-10", lambda record: 7 <= record.motif_count <= 10),
                    ("11+", lambda record: record.motif_count >= 11),
                ),
            ),
        },
        "largest_persisted_motif_minus_atom": [
            {
                "member_id": record.member_id,
                "split": record.split,
                "atom_tokens": record.atom_tokens,
                "motif_tokens": record.persisted_motif_tokens,
                "excess_tokens": record.persisted_motif_tokens - record.atom_tokens,
                "atom_count": record.atom_count,
                "motif_count": record.motif_count,
                "cross_motif_edges": record.edge_count,
                "fallback_motif_count": sum(
                    identity not in persisted_macro_set for identity in record.identities
                ),
            }
            for record in top_excess
        ],
        "decision_boundary": {
            "sample_bound": True,
            "final_pretraining_macro_k": False,
            "does_not_test_downstream_quality": True,
            "does_not_authorize_partition_change": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=list(DEFAULT_K_VALUES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_release(
        Path(args.paired_release),
        k_values=tuple(args.k_values),
    )
    output = Path(args.output_report).expanduser().resolve()
    if output.exists():
        raise PF1MotifLengthBudgetError("output report must be a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["observed"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "DEFAULT_K_VALUES",
    "PF1MotifLengthBudgetError",
    "REPORT_SCHEMA",
    "analyze_release",
    "build_parser",
    "main",
]
