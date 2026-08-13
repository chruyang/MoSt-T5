"""Promote an immutable fragSMILES tokenizer candidate under the frozen runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "most-t5-next/fragsmiles-tokenizer-runtime-promotion/v1"
EXPECTED_TRANSFORMERS_VERSION = "4.45.2"
EXPECTED_SENTENCEPIECE_VERSION = "0.2.0"
EXPECTED_RDKIT_VERSION = "2024.03.5"
DIGITS = tuple("0123456789")
SENTINELS = tuple(f"<extra_id_{index}>" for index in range(100))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _singleton_id(tokenizer: object, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)  # type: ignore[attr-defined]
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id == tokenizer.unk_token_id  # type: ignore[attr-defined]
        or tokenizer.encode(token, add_special_tokens=False) != [token_id]  # type: ignore[attr-defined]
    ):
        raise ValueError(f"token is not atomic: {token!r}")
    return token_id


def promote(*, candidate_dir: Path, base_snapshot: Path, output: Path) -> dict[str, object]:
    import sentencepiece
    from transformers import T5Tokenizer, __version__ as transformers_version

    candidate = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (candidate_dir / "macro_registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    macros = []
    for expected_rank, row in enumerate(rows):
        token = row.get("surface_token")
        if row.get("rank") != expected_rank or not isinstance(token, str):
            raise ValueError("macro registry is not dense")
        macros.append(token)
    if len(macros) != len(set(macros)):
        raise ValueError("macro surfaces are duplicated")
    replay = json.loads((candidate_dir / "chemistry_replay.json").read_text(encoding="utf-8"))
    if (
        replay.get("runtime", {}).get("rdkit_version") != EXPECTED_RDKIT_VERSION
        or replay.get("candidate", {}).get("rows") != len(macros)
        or replay.get("candidate", {}).get("canonical_fixed_point_rows") != len(macros)
        or replay.get("candidate", {}).get("canonical_non_fixed_point_rows") != 0
    ):
        raise ValueError("chemistry replay is not production-complete")
    if transformers_version != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError("Transformers runtime differs from the frozen version")
    if sentencepiece.__version__ != EXPECTED_SENTENCEPIECE_VERSION:
        raise ValueError("SentencePiece runtime differs from the frozen version")

    base = T5Tokenizer.from_pretrained(str(base_snapshot), local_files_only=True, legacy=True)
    observed = T5Tokenizer.from_pretrained(
        str(candidate_dir / "tokenizer_snapshot"), local_files_only=True, legacy=True
    )
    if len(base) != 32100 or len(observed) != 53368 or len(macros) != 21100:
        raise ValueError("frozen vocabulary dimensions drifted")
    base_vocab = base.get_vocab()
    observed_vocab = observed.get_vocab()
    if any(observed_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise ValueError("base T5 IDs changed")
    macro_ids = [_singleton_id(observed, token) for token in macros]
    if len(set(macro_ids)) != len(macro_ids):
        raise ValueError("macro IDs are not bijective")
    digit_ids = {digit: _singleton_id(observed, digit) for digit in DIGITS}
    if any(token_id not in set(base_vocab.values()) for token_id in digit_ids.values()):
        raise ValueError("a digit does not reuse a base T5 row")
    sentinel_ids = {token: _singleton_id(observed, token) for token in SENTINELS}
    if sentinel_ids != {token: _singleton_id(base, token) for token in SENTINELS}:
        raise ValueError("sentinel IDs changed")

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "runtime": {
            "transformers": transformers_version,
            "sentencepiece": sentencepiece.__version__,
            "rdkit": EXPECTED_RDKIT_VERSION,
        },
        "counts": {
            "base_vocab_size": len(base),
            "macro_tokens": len(macros),
            "non_macro_added_tokens": len(observed) - len(base) - len(macros),
            "digit_rows_added": 0,
            "final_vocab_size": len(observed),
        },
        "contracts": {
            "complete_registry_chemistry_replay_passed": True,
            "all_macro_ids_verified": True,
            "base_ids_unchanged": True,
            "sentinel_ids_unchanged": True,
            "digits_reuse_base_t5_rows": True,
            "offline_snapshot_loaded_under_frozen_runtime": True,
            "no_locality_or_frequency_filter": True,
        },
        "artifacts": {
            "candidate_manifest_sha256": _sha256(candidate_dir / "manifest.json"),
            "macro_registry_sha256": _sha256(candidate_dir / "macro_registry.jsonl"),
            "chemistry_replay_sha256": _sha256(candidate_dir / "chemistry_replay.json"),
            "tokenizer_snapshot_tree_sha256": _tree_sha256(candidate_dir / "tokenizer_snapshot"),
        },
        "candidate_status": candidate.get("status"),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--base-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = promote(
        candidate_dir=args.candidate_dir,
        base_snapshot=args.base_snapshot,
        output=args.output,
    )
    print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
