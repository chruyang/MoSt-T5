"""Validate digit reuse and opaque fragSMILES macro AddedTokens on HF T5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import uuid
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "most-t5-next/fragsmiles-hf-tokenizer-compatibility/v1"
DIGITS = tuple("0123456789")
SENTINELS = tuple(f"<extra_id_{index}>" for index in range(100))


class FragSmilesTokenizerCompatibilityError(RuntimeError):
    pass


def _singleton_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id == tokenizer.unk_token_id
        or tokenizer.encode(token, add_special_tokens=False) != [token_id]
    ):
        raise FragSmilesTokenizerCompatibilityError(f"token is not atomic: {token!r}")
    return token_id


def _load_macro_surfaces(rows: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    result = []
    for expected_rank, row in enumerate(rows):
        token = row.get("surface_token")
        if row.get("rank") != expected_rank or not isinstance(token, str) or not token:
            raise FragSmilesTokenizerCompatibilityError("macro registry is malformed")
        result.append(token)
    if not result or len(result) != len(set(result)):
        raise FragSmilesTokenizerCompatibilityError("macro surfaces are empty or duplicate")
    return tuple(result)


def validate_tokenizer(
    *,
    base_snapshot: Path,
    macro_rows: Iterable[Mapping[str, object]],
    extra_added_tokens: Iterable[str],
) -> dict[str, object]:
    from transformers import AddedToken, T5Tokenizer, __version__ as transformers_version

    macros = _load_macro_surfaces(macro_rows)
    extras = tuple(extra_added_tokens)
    if len(extras) != len(set(extras)) or set(extras).intersection(macros):
        raise FragSmilesTokenizerCompatibilityError("added token domains overlap")
    tokenizer = T5Tokenizer.from_pretrained(
        str(base_snapshot), local_files_only=True, legacy=True
    )
    base_size = len(tokenizer)
    base_vocab = dict(tokenizer.get_vocab())
    sentinel_ids = {token: _singleton_id(tokenizer, token) for token in SENTINELS}
    digit_piece_ids = {
        digit: tuple(tokenizer.encode(digit, add_special_tokens=False)) for digit in DIGITS
    }

    requested = extras + macros
    added = tokenizer.add_tokens(
        [
            AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False)
            for token in requested
        ],
        special_tokens=False,
    )
    if added != len(requested) or len(tokenizer) != base_size + len(requested):
        raise FragSmilesTokenizerCompatibilityError("ordinary AddedToken count drifted")
    before_digits = {token: _singleton_id(tokenizer, token) for token in requested}
    size_before_digits = len(tokenizer)
    digit_added = tokenizer.add_tokens(list(DIGITS), special_tokens=True)
    if digit_added != 0 or len(tokenizer) != size_before_digits:
        raise FragSmilesTokenizerCompatibilityError("digit registration added vocabulary rows")
    digit_ids = {digit: _singleton_id(tokenizer, digit) for digit in DIGITS}
    if any(token_id not in set(base_vocab.values()) for token_id in digit_ids.values()):
        raise FragSmilesTokenizerCompatibilityError("a digit does not reuse a base T5 row")
    after_digits = {token: _singleton_id(tokenizer, token) for token in requested}
    if after_digits != before_digits:
        raise FragSmilesTokenizerCompatibilityError("digit registration split an opaque token")

    adversarial = (
        macros[0] + "7",
        "7" + macros[0],
        macros[0] + macros[min(1, len(macros) - 1)],
    )
    expected = (
        [after_digits[macros[0]], digit_ids["7"]],
        [digit_ids["7"], after_digits[macros[0]]],
        [after_digits[macros[0]], after_digits[macros[min(1, len(macros) - 1)]]],
    )
    if [tokenizer.encode(text, add_special_tokens=False) for text in adversarial] != list(expected):
        raise FragSmilesTokenizerCompatibilityError("adjacent opaque-token boundary drifted")

    # Some managed Windows environments deny nested writes below directories
    # created by tempfile. Let the caller pin a workspace-local scratch root
    # and create the directory directly without weakening the save/reload gate.
    scratch_root = os.environ.get("MOST_T5_TOKENIZER_GATE_TMPDIR")
    owned_scratch: Path | None = None
    if scratch_root:
        owned_scratch = Path(scratch_root) / f"gate-{uuid.uuid4().hex}"
        owned_scratch.mkdir(parents=True, exist_ok=False)
        raw = str(owned_scratch)
    else:
        raw = tempfile.mkdtemp()
    snapshot = Path(raw) / "tokenizer"
    tokenizer.save_pretrained(str(snapshot))
    reloaded = T5Tokenizer.from_pretrained(
        str(snapshot), local_files_only=True, legacy=True
    )
    if len(reloaded) != len(tokenizer):
        raise FragSmilesTokenizerCompatibilityError("reload changed vocabulary size")
    if {digit: _singleton_id(reloaded, digit) for digit in DIGITS} != digit_ids:
        raise FragSmilesTokenizerCompatibilityError("reload changed digit rows")
    if {token: _singleton_id(reloaded, token) for token in requested} != after_digits:
        raise FragSmilesTokenizerCompatibilityError("reload changed opaque token rows")
    if {token: _singleton_id(reloaded, token) for token in SENTINELS} != sentinel_ids:
        raise FragSmilesTokenizerCompatibilityError("reload changed T5 sentinels")
    observed_vocab = reloaded.get_vocab()
    if any(observed_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise FragSmilesTokenizerCompatibilityError("reload changed base vocabulary IDs")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": False,
        "runtime": {"transformers": transformers_version},
        "counts": {
            "base_vocab_size": base_size,
            "macro_tokens": len(macros),
            "non_macro_added_tokens": len(extras),
            "final_vocab_size": size_before_digits,
            "digit_rows_added": digit_added,
        },
        "digits": {
            "base_sentencepiece_encoding_before_atomic_registration": digit_piece_ids,
            "reused_base_token_ids": digit_ids,
            "one_digit_one_token": True,
        },
        "contracts": {
            "all_macros_atomic_before_and_after_digit_registration": True,
            "all_added_tokens_atomic_after_reload": True,
            "base_ids_unchanged": True,
            "sentinel_ids_unchanged": True,
            "adjacent_opaque_tokens_verified": True,
        },
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    from most_t5_next.p1.build_fragsmiles_macro_registry_v1 import non_macro_token_universe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-snapshot", required=True, type=Path)
    parser.add_argument("--macro-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = validate_tokenizer(
        base_snapshot=args.base_snapshot,
        macro_rows=_jsonl(args.macro_registry),
        extra_added_tokens=("<bom>", "<eom>") + non_macro_token_universe(),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
