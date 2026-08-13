"""Publish one immutable fragSMILES macro registry and T5 tokenizer snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA_VERSION = "most-t5-next/fragsmiles-tokenizer-release/v1"
EXPECTED_TRANSFORMERS_VERSION = "4.45.2"
EXPECTED_RDKIT_VERSION = "2024.03.5"


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def publish_tokenizer(
    *,
    base_snapshot: Path,
    macro_registry: Path,
    chemistry_replay: Path,
    output_dir: Path,
    expected_transformers_version: str = EXPECTED_TRANSFORMERS_VERSION,
    expected_rdkit_version: str = EXPECTED_RDKIT_VERSION,
) -> dict[str, object]:
    from transformers import AddedToken, T5Tokenizer, __version__ as transformers_version

    from most_t5_next.p1.build_fragsmiles_macro_registry_v1 import non_macro_token_universe
    from most_t5_next.p1.validate_fragsmiles_hf_tokenizer_v1 import (
        DIGITS,
        SENTINELS,
        _load_macro_surfaces,
        _singleton_id,
    )

    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = _read_jsonl(macro_registry)
    macros = _load_macro_surfaces(rows)
    replay = json.loads(chemistry_replay.read_text(encoding="utf-8"))
    candidate = replay.get("candidate")
    if (
        replay.get("status") != "pass"
        or not isinstance(candidate, dict)
        or candidate.get("rows") != len(rows)
        or candidate.get("canonical_fixed_point_rows") != len(rows)
        or candidate.get("canonical_non_fixed_point_rows") != 0
    ):
        raise ValueError("chemistry replay does not admit the complete registry")

    extras = ("<bom>", "<eom>") + non_macro_token_universe()
    if len(extras) != len(set(extras)) or set(extras).intersection(macros):
        raise ValueError("token domains overlap")
    tokenizer = T5Tokenizer.from_pretrained(str(base_snapshot), local_files_only=True, legacy=True)
    base_size = len(tokenizer)
    base_vocab = dict(tokenizer.get_vocab())
    sentinel_ids = {token: _singleton_id(tokenizer, token) for token in SENTINELS}
    requested = extras + macros
    added = tokenizer.add_tokens(
        [AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False) for token in requested],
        special_tokens=False,
    )
    if added != len(requested):
        raise ValueError("added token count drifted")
    ids_before_digits = {token: _singleton_id(tokenizer, token) for token in requested}
    if tokenizer.add_tokens(list(DIGITS), special_tokens=True) != 0:
        raise ValueError("digit registration added rows")
    digit_ids = {digit: _singleton_id(tokenizer, digit) for digit in DIGITS}
    if {token: _singleton_id(tokenizer, token) for token in requested} != ids_before_digits:
        raise ValueError("digit registration split an opaque token")

    output_dir.mkdir(parents=True)
    snapshot = output_dir / "tokenizer_snapshot"
    tokenizer.save_pretrained(str(snapshot))
    frozen_registry = output_dir / "macro_registry.jsonl"
    frozen_registry.write_bytes(macro_registry.read_bytes())
    frozen_replay = output_dir / "chemistry_replay.json"
    frozen_replay.write_bytes(chemistry_replay.read_bytes())

    reloaded = T5Tokenizer.from_pretrained(str(snapshot), local_files_only=True, legacy=True)
    if len(reloaded) != len(tokenizer):
        raise ValueError("offline reload changed vocabulary size")
    if {token: _singleton_id(reloaded, token) for token in requested} != ids_before_digits:
        raise ValueError("offline reload changed added-token IDs")
    if {digit: _singleton_id(reloaded, digit) for digit in DIGITS} != digit_ids:
        raise ValueError("offline reload changed digit IDs")
    if {token: _singleton_id(reloaded, token) for token in SENTINELS} != sentinel_ids:
        raise ValueError("offline reload changed sentinel IDs")
    observed_vocab = reloaded.get_vocab()
    if any(observed_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise ValueError("offline reload changed base vocabulary IDs")

    observed_rdkit = replay.get("runtime", {}).get("rdkit_version")
    runtime_matches = (
        transformers_version == expected_transformers_version
        and observed_rdkit == expected_rdkit_version
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if runtime_matches else "candidate_runtime_mismatch",
        "training_admission": runtime_matches,
        "runtime": {
            "transformers": transformers_version,
            "rdkit": observed_rdkit,
            "expected_transformers": expected_transformers_version,
            "expected_rdkit": expected_rdkit_version,
            "matches_frozen_training_runtime": runtime_matches,
        },
        "counts": {
            "base_vocab_size": base_size,
            "macro_tokens": len(macros),
            "non_macro_added_tokens": len(extras),
            "digit_rows_added": 0,
            "final_vocab_size": len(reloaded),
        },
        "contracts": {
            "complete_registry_chemistry_replay_passed": True,
            "no_locality_or_frequency_filter": True,
            "all_macro_names_atomic": True,
            "digits_reuse_base_t5_rows": True,
            "base_and_sentinel_ids_unchanged": True,
            "offline_save_reload_verified": True,
        },
        "artifacts": {
            "macro_registry": {"path": frozen_registry.name, "sha256": _sha256(frozen_registry)},
            "chemistry_replay": {"path": frozen_replay.name, "sha256": _sha256(frozen_replay)},
            "tokenizer_snapshot": {"path": snapshot.name, "tree_sha256": _tree_sha256(snapshot)},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-snapshot", required=True, type=Path)
    parser.add_argument("--macro-registry", required=True, type=Path)
    parser.add_argument("--chemistry-replay", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-transformers-version", default=EXPECTED_TRANSFORMERS_VERSION
    )
    parser.add_argument("--expected-rdkit-version", default=EXPECTED_RDKIT_VERSION)
    args = parser.parse_args(argv)
    manifest = publish_tokenizer(
        base_snapshot=args.base_snapshot,
        macro_registry=args.macro_registry,
        chemistry_replay=args.chemistry_replay,
        output_dir=args.output_dir,
        expected_transformers_version=args.expected_transformers_version,
        expected_rdkit_version=args.expected_rdkit_version,
    )
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
