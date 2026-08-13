#!/usr/bin/env python3
"""Compare lossless lexer candidates on a published anchored PF-1 surface."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, Sequence

from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
    decode_pure_motif,
    lex_pure_motif,
)


SCHEMA_VERSION = "most-t5-next/pf1-anchored-lexer-analysis/v1"
FAILURES_NAME = "failures.jsonl"
REPORT_NAME = "report.json"
_ATOMWISE_RE = re.compile(
    r"\[[^\]]+\]|Br|Cl|[BCNOPSFIbcnosp]|"
    r"\(|\)|\.|=|#|-|\+|:|~|\?|>|\*|\$|%[0-9]{2}|[0-9]"
)


class PF1AnchoredLexerAnalysisError(RuntimeError):
    """The comparison cannot be completed exactly."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise PF1AnchoredLexerAnalysisError(
                    f"{path.name} line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise PF1AnchoredLexerAnalysisError(
                    f"{path.name} line {line_number} is not an object"
                )
            yield row


def _distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _atomwise_reference_tokens(pure_motif: str) -> tuple[str, ...]:
    core = pure_motif[1:-1]
    tokens = []
    cursor = 0
    while cursor < len(core):
        match = _ATOMWISE_RE.match(core, cursor)
        if match is None:
            raise PF1AnchoredLexerAnalysisError(
                f"atom-wise reference lexer stopped at {core[cursor:]!r}"
            )
        tokens.append(match.group(0))
        cursor = match.end()
    if "".join(tokens) != core:
        raise PF1AnchoredLexerAnalysisError("atom-wise reference lexer drifted")
    return tuple(tokens)


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def analyze(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import T5Tokenizer
    except ImportError as exc:
        raise PF1AnchoredLexerAnalysisError("transformers is required") from exc

    release = Path(args.surface_release).expanduser().resolve()
    base_snapshot = Path(args.base_tokenizer).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise PF1AnchoredLexerAnalysisError("output directory must be absent")
    source_manifest_path = release / "manifest.json"
    source_records_path = release / "surface_records.jsonl"
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if not isinstance(source_manifest, dict) or source_manifest.get("status") != "pass":
        raise PF1AnchoredLexerAnalysisError("anchored source release is not passed")
    tokenizer = T5Tokenizer.from_pretrained(
        str(base_snapshot), local_files_only=True, legacy=True
    )
    if tokenizer.unk_token_id is None:
        raise PF1AnchoredLexerAnalysisError("base tokenizer lacks an unknown token ID")

    output.mkdir(parents=False)
    failures_path = output / FAILURES_NAME
    cache: dict[str, dict[str, object]] = {}
    chemical_vocab: Counter[str] = Counter()
    atomwise_vocab: Counter[str] = Counter()
    motif_occurrences = 0
    t5_unknown_occurrences = 0
    t5_decode_mismatches = 0
    lexical_failures = 0
    member_lengths = {
        name: []
        for name in (
            "one_token_pure_motif",
            "bounded_chemical_lexer",
            "fine_moltex_atomwise_reference",
            "base_t5_sentencepiece",
            "utf8_byte_floor",
        )
    }
    member_count = 0
    with failures_path.open("x", encoding="utf-8", newline="\n") as failures_handle:
        for row in _iter_jsonl(source_records_path):
            surface = row.get("surface")
            if not isinstance(surface, dict):
                raise PF1AnchoredLexerAnalysisError("surface record is malformed")
            phrases = surface.get("phrases")
            component_ranges = surface.get("component_motif_ranges")
            if not isinstance(phrases, list) or not phrases or not isinstance(component_ranges, list):
                raise PF1AnchoredLexerAnalysisError("surface phrase arrays are malformed")
            anchor_count = 0
            totals = {name: 0 for name in member_lengths}
            for phrase in phrases:
                if not isinstance(phrase, dict):
                    raise PF1AnchoredLexerAnalysisError("surface phrase is malformed")
                pure = phrase.get("pure_motif")
                anchors = phrase.get("anchors")
                if not isinstance(pure, str) or not isinstance(anchors, list):
                    raise PF1AnchoredLexerAnalysisError("surface phrase identity is malformed")
                anchor_count += len(anchors)
                result = cache.get(pure)
                if result is None:
                    try:
                        chemical = lex_pure_motif(pure)
                        if decode_pure_motif(chemical.tokens) != pure:
                            raise PF1AnchoredLexerAnalysisError(
                                "bounded chemical lexer is not exact"
                            )
                        atomwise = _atomwise_reference_tokens(pure)
                        core = pure[1:-1]
                        t5_ids = tuple(
                            int(value)
                            for value in tokenizer.encode(core, add_special_tokens=False)
                        )
                        t5_decoded = tokenizer.decode(
                            t5_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        result = {
                            "chemical_tokens": chemical.tokens,
                            "atomwise_tokens": atomwise,
                            "t5_ids": t5_ids,
                            "t5_exact": t5_decoded == core,
                            "t5_unknowns": sum(
                                value == tokenizer.unk_token_id for value in t5_ids
                            ),
                            "byte_length": len(core.encode("utf-8")),
                        }
                        cache[pure] = result
                    except Exception as exc:
                        lexical_failures += 1
                        failure = {
                            "selection_index": row.get("selection_index"),
                            "pure_motif": pure,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                        failures_handle.write(
                            _canonical_json_bytes(failure).decode("utf-8") + "\n"
                        )
                        continue
                motif_occurrences += 1
                chemical_tokens = result["chemical_tokens"]
                atomwise_tokens = result["atomwise_tokens"]
                assert isinstance(chemical_tokens, tuple) and isinstance(atomwise_tokens, tuple)
                chemical_vocab.update(chemical_tokens)
                atomwise_vocab.update(atomwise_tokens)
                t5_unknown_occurrences += int(result["t5_unknowns"])
                t5_decode_mismatches += int(not bool(result["t5_exact"]))
                totals["one_token_pure_motif"] += 1
                totals["bounded_chemical_lexer"] += len(chemical_tokens)
                totals["fine_moltex_atomwise_reference"] += len(atomwise_tokens)
                totals["base_t5_sentencepiece"] += len(result["t5_ids"])
                totals["utf8_byte_floor"] += int(result["byte_length"])
            separators = max(0, len(component_ranges) - 1)
            for name, value in totals.items():
                member_lengths[name].append(anchor_count + separators + value)
            member_count += 1
            if args.progress_every and member_count % args.progress_every == 0:
                print(f"anchored-lexer {member_count}", flush=True)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if lexical_failures == 0 else "failed",
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "surface_manifest_sha256": _sha256_file(source_manifest_path),
            "surface_records_sha256": _sha256_file(source_records_path),
            "base_tokenizer_snapshot": str(base_snapshot),
        },
        "counts": {
            "members": member_count,
            "motif_occurrences": motif_occurrences,
            "unique_pure_motifs": len(cache),
            "lexical_failures": lexical_failures,
            "base_t5_unknown_token_occurrences": t5_unknown_occurrences,
            "base_t5_decode_mismatch_occurrences": t5_decode_mismatches,
            "bounded_chemical_vocabulary_size": len(chemical_vocab),
            "atomwise_reference_vocabulary_size": len(atomwise_vocab),
        },
        "member_sequence_lengths": {
            name: _distribution(values) for name, values in member_lengths.items()
        },
        "vocabularies": {
            "bounded_chemical_tokens": sorted(chemical_vocab),
            "atomwise_reference_tokens": sorted(atomwise_vocab),
        },
        "contracts": {
            "anchors_counted_as_one_registered_token_each": True,
            "component_separator_counted_as_one_registered_token": True,
            "motif_phrase_boundaries_are_implicit_sidecar_spans": True,
            "base_t5_is_a_comparator_not_the_lossless_fallback": True,
            "fine_moltex_atomwise_bracket_atoms_are_open_types": True,
            "bounded_chemical_lexer_is_finite_and_reversible": lexical_failures == 0,
            "training_admission": False,
        },
    }
    _write_json_new(output / REPORT_NAME, report)
    if lexical_failures:
        raise PF1AnchoredLexerAnalysisError(
            f"anchored lexer rejected {lexical_failures} motif occurrence(s)"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-release", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = analyze(_parser().parse_args(argv))
    except Exception as exc:
        print(f"anchored lexer analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
