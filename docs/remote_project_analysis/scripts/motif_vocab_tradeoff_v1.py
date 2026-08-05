#!/usr/bin/env python3
"""Read-only motif census analysis for vocabulary/sequence trade-offs.

The source census is never modified. Results are written to a fresh output
directory. The slot projection replaces inline ``<n*>`` anchors in place with
``<*>``; the deletion projection is retained only as a diagnostic comparator.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ANCHOR_RE = re.compile(r"<(\d+)\*>")
SMILES_TOKEN_RE = re.compile(
    r"<\*>|\[[^\]]+\]|Br|Cl|"
    r"[BCNOPSFIbcnosp]|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|"
    r"%[0-9]{2}|[0-9]"
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def lexical_length(template: str) -> Tuple[int, bool]:
    """Return constrained SMILES-token length and exact coverage flag."""
    position = 0
    length = 0
    while position < len(template):
        match = SMILES_TOKEN_RE.match(template, position)
        if match is None:
            return len(template), False
        position = match.end()
        length += 1
    return length, True


def counter_stats(counter: Dict[str, int], total_occurrences: int) -> dict:
    frequencies = list(counter.values())
    thresholds = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    threshold_rows = []
    for minimum in thresholds:
        kept = [value for value in frequencies if value >= minimum]
        mass = sum(kept)
        threshold_rows.append(
            {
                "min_frequency": minimum,
                "vocab_size": len(kept),
                "occurrence_coverage": mass / total_occurrences,
                "oov_occurrence_rate": 1.0 - (mass / total_occurrences),
            }
        )

    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    topk_values = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        200000,
        len(ordered),
    ]
    topk_rows = []
    running = 0
    cursor = 0
    for requested in sorted(set(min(value, len(ordered)) for value in topk_values)):
        while cursor < requested:
            running += ordered[cursor][1]
            cursor += 1
        topk_rows.append(
            {
                "top_k": requested,
                "occurrence_coverage": running / total_occurrences,
                "oov_occurrence_rate": 1.0 - (running / total_occurrences),
            }
        )

    return {
        "unique_types": len(counter),
        "singleton_types": sum(value == 1 for value in frequencies),
        "singleton_type_rate": sum(value == 1 for value in frequencies) / len(counter),
        "singleton_occurrence_rate": sum(value == 1 for value in frequencies)
        / total_occurrences,
        "frequency_thresholds": threshold_rows,
        "top_k": topk_rows,
    }


def memory_estimate(vocab_size: int, hidden_size: int) -> dict:
    parameters = vocab_size * hidden_size
    return {
        "additional_parameters": parameters,
        "bf16_weight_mib": parameters * 2 / (1024**2),
        "rough_dense_adam_training_state_gib_at_16_bytes_per_parameter": parameters
        * 16
        / (1024**3),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--hidden-size", type=int, default=768)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    source_sha256 = sha256_file(source)
    if source_sha256 != args.expected_sha256.lower():
        raise RuntimeError(
            f"source SHA-256 mismatch: {source_sha256} != {args.expected_sha256.lower()}"
        )

    exact: collections.Counter[str] = collections.Counter()
    deletion_core: collections.Counter[str] = collections.Counter()
    slot_template: collections.Counter[str] = collections.Counter()
    anchor_arity_occurrences: collections.Counter[int] = collections.Counter()
    row_count = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            fragment = record["motif_fragment"]
            count = int(record["count"])
            if count <= 0:
                raise ValueError(f"non-positive count at line {line_number}")
            exact[fragment] += count
            deletion_core[ANCHOR_RE.sub("", fragment)] += count
            slot_template[ANCHOR_RE.sub("<*>", fragment)] += count
            anchor_arity_occurrences[len(ANCHOR_RE.findall(fragment))] += count
            row_count += 1

    total_occurrences = sum(exact.values())
    anchor_occurrences = sum(
        arity * occurrences for arity, occurrences in anchor_arity_occurrences.items()
    )
    if row_count != len(exact):
        raise RuntimeError("source census contains duplicate exact motif lexemes")

    lexical_lengths: Dict[str, int] = {}
    lexical_failures: List[str] = []
    for template in slot_template:
        length, covered = lexical_length(template)
        lexical_lengths[template] = length
        if not covered and len(lexical_failures) < 50:
            lexical_failures.append(template)

    ordered_slots = sorted(slot_template.items(), key=lambda item: (-item[1], item[0]))
    requested_topk = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        200000,
        len(ordered_slots),
    ]
    hybrid_rows = []
    for top_k in sorted(set(min(value, len(ordered_slots)) for value in requested_topk)):
        common = ordered_slots[:top_k]
        rare = ordered_slots[top_k:]
        common_occurrences = sum(count for _, count in common)
        rare_occurrences = total_occurrences - common_occurrences
        template_segment_tokens = common_occurrences + sum(
            count * lexical_lengths[template] for template, count in rare
        )
        total_factorized_tokens = template_segment_tokens + anchor_occurrences
        hybrid_rows.append(
            {
                "top_k": top_k,
                "atomic_template_occurrence_coverage": common_occurrences
                / total_occurrences,
                "fallback_occurrence_rate": rare_occurrences / total_occurrences,
                "template_segment_tokens": template_segment_tokens,
                "anchor_id_tokens": anchor_occurrences,
                "total_factorized_tokens": total_factorized_tokens,
                "token_multiplier_vs_one_exact_motif_token": total_factorized_tokens
                / total_occurrences,
                **memory_estimate(top_k, args.hidden_size),
            }
        )

    report = {
        "schema": "motif-vocab-tradeoff-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
            "rows": row_count,
        },
        "projection": {
            "exact_unique": len(exact),
            "deletion_core_unique": len(deletion_core),
            "slot_template_unique": len(slot_template),
            "total_motif_occurrences": total_occurrences,
            "total_anchor_occurrences": anchor_occurrences,
            "all_atomic_slot_token_multiplier": (total_occurrences + anchor_occurrences)
            / total_occurrences,
            "anchor_arity_occurrences": dict(sorted(anchor_arity_occurrences.items())),
        },
        "exact_stats": counter_stats(exact, total_occurrences),
        "deletion_core_stats": counter_stats(deletion_core, total_occurrences),
        "slot_template_stats": counter_stats(slot_template, total_occurrences),
        "slot_lexical_fallback": {
            "tokenizer": "constrained-smiles-regex-v1",
            "failed_unique_templates": len(
                [template for template in slot_template if not lexical_length(template)[1]]
            ),
            "failure_examples": lexical_failures,
        },
        "hybrid_slot_topk": hybrid_rows,
        "embedding_assumption": {
            "hidden_size": args.hidden_size,
            "tied_shared_embedding": True,
            "note": "Memory excludes activations and the base tokenizer vocabulary; 16 bytes/parameter is an illustrative dense AdamW training-state estimate.",
        },
    }

    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.md"
    receipt_path = output_dir / "receipt.json"
    write_new(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    threshold_rows = report["slot_template_stats"]["frequency_thresholds"]
    summary_parts = [
        "# Motif vocabulary trade-off v1",
        "",
        f"- source SHA-256: `{source_sha256}`",
        f"- exact unique motifs: {len(exact):,}",
        f"- deletion-core unique motifs: {len(deletion_core):,}",
        f"- slot-template unique motifs: {len(slot_template):,}",
        f"- motif occurrences: {total_occurrences:,}",
        f"- anchor occurrences: {anchor_occurrences:,}",
        f"- all-atomic slot factorization multiplier: {(total_occurrences + anchor_occurrences) / total_occurrences:.4f}x",
        f"- lexical fallback failures: {report['slot_lexical_fallback']['failed_unique_templates']:,}",
        "",
        "## Slot-template minimum-frequency policy",
        "",
        markdown_table(
            ["min freq", "vocab", "occurrence coverage", "fallback rate"],
            (
                (
                    row["min_frequency"],
                    f"{row['vocab_size']:,}",
                    percent(row["occurrence_coverage"]),
                    percent(row["oov_occurrence_rate"]),
                )
                for row in threshold_rows
            ),
        ),
        "",
        "## Hybrid slot-template top-K policy",
        "",
        markdown_table(
            [
                "top-K",
                "atomic coverage",
                "fallback rate",
                "total token multiplier",
                "extra params",
                "BF16 MiB",
                "rough Adam state GiB",
            ],
            (
                (
                    f"{row['top_k']:,}",
                    percent(row["atomic_template_occurrence_coverage"]),
                    percent(row["fallback_occurrence_rate"]),
                    f"{row['token_multiplier_vs_one_exact_motif_token']:.4f}x",
                    f"{row['additional_parameters']:,}",
                    f"{row['bf16_weight_mib']:.2f}",
                    f"{row['rough_dense_adam_training_state_gib_at_16_bytes_per_parameter']:.3f}",
                )
                for row in hybrid_rows
            ),
        ),
        "",
        "Coverage is occurrence-weighted. Rare templates are never mapped to UNK in the hybrid estimate; they fall back to constrained SMILES lexical tokens while slot and anchor-ID tokens remain explicit.",
        "",
    ]
    write_new(summary_path, "\n".join(summary_parts))

    receipt = {
        "schema": "motif-vocab-tradeoff-receipt-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": source_sha256,
        "report_sha256": sha256_file(report_path),
        "summary_sha256": sha256_file(summary_path),
    }
    write_new(receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
