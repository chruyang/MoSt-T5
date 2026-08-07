#!/usr/bin/env python3
"""Build the shared, sample-bound tokenizer used by the 128-record P1 canary.

This is deliberately a *candidate* tokenizer, not the final pretraining
vocabulary.  It freezes every symbol needed by the paired A/M canary before a
model is resized: T5 sentinels stay at their base IDs, SELFIES symbols remain
one token each (with an opaque separator instead of registering raw ``.``),
graph/port grammar tokens are fixed, and every selected opaque motif macro is
bound to its exact identity.  No token may be added once a canary checkpoint
exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from most_t5_next.p1.production_bridge import (
    ProductionTokenizerRuntime,
)
from most_t5_next.r1.tokenizer.production_atom_selfies_codec_v1 import (
    MOLECULE_BEGIN,
    MOLECULE_END,
    SELFIES_DISTRIBUTION_VERSION,
    tokenizer_surface_for_selfies_symbol,
)
from most_t5_next.r1.tokenizer.production_graph_ports_codec_v1 import (
    GPORTS_UNION_TOKENS,
)


SCHEMA_VERSION = "most-t5-r1/p1-canary-union-tokenizer/v1"
SCOPE = "paired_128_sample_bound_candidate"
T5_SENTINELS = tuple("<extra_id_{}>".format(index) for index in range(100))
CONTROL_SPECIAL_TOKENS = (
    MOLECULE_BEGIN,
    MOLECULE_END,
    "[MMM]:",
    "[Caption]:",
    "[Text2Mol]:",
    "[Denoise]:",
)
NATURAL_TEXT_REGRESSION_SURFACES = (
    "This molecule is aromatic.",
    "A. B.",
    "C.O",
    "CCO",
    "The HOMO-LUMO gap is 1.0 eV.",
)
MACRO_TOKEN_RE = re.compile(r"^<MOST:M:[0-9]{6}>$")
MANIFEST_NAME = "manifest.json"
SNAPSHOT_DIRECTORY = "tokenizer_snapshot"


class CanaryUnionTokenizerError(RuntimeError):
    """The shared canary vocabulary cannot be frozen exactly."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observe_snapshot(path: Path) -> dict[str, object]:
    rows = []
    for candidate in sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix().encode("utf-8"),
    ):
        rows.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "bytes": int(candidate.stat().st_size),
                "sha256": _sha256_file(candidate),
            }
        )
    if not rows:
        raise CanaryUnionTokenizerError("saved tokenizer snapshot is empty")
    return {"files": rows, "tree_sha256": _sha256_json(rows)}


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise CanaryUnionTokenizerError("token surfaces must be non-empty strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _sorted_utf8(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(_stable_unique(values), key=lambda value: value.encode("utf-8"))
    )


def _normalize_macro_registry(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Close the opaque macro names over their deterministic identities."""

    if isinstance(rows, (str, bytes, bytearray)):
        raise CanaryUnionTokenizerError("motif_macro_registry must be a row sequence")
    try:
        candidates = tuple(rows)
    except TypeError as exc:
        raise CanaryUnionTokenizerError(
            "motif_macro_registry must be a finite row sequence"
        ) from exc
    parsed: list[tuple[str, str, int]] = []
    identities: set[str] = set()
    tokens: set[str] = set()
    required = {"identity", "token", "occurrence_count"}
    for offset, row in enumerate(candidates):
        if not isinstance(row, Mapping) or set(row) != required:
            raise CanaryUnionTokenizerError(
                "macro row {} must contain exactly identity/token/occurrence_count".format(
                    offset
                )
            )
        identity = row["identity"]
        token = row["token"]
        occurrence_count = row["occurrence_count"]
        if not isinstance(identity, str) or not identity:
            raise CanaryUnionTokenizerError(
                "macro row {} identity must be a non-empty string".format(offset)
            )
        if not isinstance(token, str) or MACRO_TOKEN_RE.fullmatch(token) is None:
            raise CanaryUnionTokenizerError(
                "macro row {} token must use <MOST:M:000000>".format(offset)
            )
        if (
            isinstance(occurrence_count, bool)
            or not isinstance(occurrence_count, Integral)
            or int(occurrence_count) <= 0
        ):
            raise CanaryUnionTokenizerError(
                "macro row {} occurrence_count must be positive".format(offset)
            )
        if identity in identities or token in tokens:
            raise CanaryUnionTokenizerError(
                "macro identities and token surfaces must both be unique"
            )
        identities.add(identity)
        tokens.add(token)
        parsed.append((identity, token, int(occurrence_count)))

    parsed.sort(key=lambda row: (-row[2], row[0].encode("utf-8")))
    normalized = []
    for rank, (identity, token, occurrence_count) in enumerate(parsed):
        expected_token = "<MOST:M:{:06d}>".format(rank)
        if token != expected_token:
            raise CanaryUnionTokenizerError(
                "macro token {!r} disagrees with deterministic rank {}".format(
                    token, rank
                )
            )
        normalized.append(
            {
                "rank": rank,
                "identity": identity,
                "identity_sha256": _sha256_bytes(identity.encode("utf-8")),
                "token": token,
                "occurrence_count": occurrence_count,
            }
        )
    return tuple(normalized)


def _selfies_token_surfaces(symbols: Sequence[str]) -> tuple[str, ...]:
    surfaces = tuple(tokenizer_surface_for_selfies_symbol(symbol) for symbol in symbols)
    if len(set(surfaces)) != len(surfaces):
        raise CanaryUnionTokenizerError(
            "distinct SELFIES symbols collapse onto one tokenizer surface"
        )
    if "." in surfaces:
        raise CanaryUnionTokenizerError(
            "raw SELFIES dot must never be registered in the union tokenizer"
        )
    return surfaces


def _exact_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id == unk_id
    ):
        raise CanaryUnionTokenizerError("token is absent or maps to UNK: {!r}".format(token))
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if tuple(encoded) != (token_id,):
        raise CanaryUnionTokenizerError("token is not an exact one-ID surface: {!r}".format(token))
    if str(tokenizer.convert_ids_to_tokens(token_id)) != token:
        raise CanaryUnionTokenizerError("token ID is not reversible: {!r}".format(token))
    return token_id


def _validate_declared_tokens(
    tokenizer: Any,
    tokens: Sequence[str],
    expected_ids: dict[str, int] | None = None,
) -> dict[str, int]:
    token_ids = {token: _exact_token_id(tokenizer, token) for token in tokens}
    if len(set(token_ids.values())) != len(token_ids):
        raise CanaryUnionTokenizerError("declared token surfaces do not have unique IDs")
    if expected_ids is not None and token_ids != expected_ids:
        raise CanaryUnionTokenizerError("saved tokenizer changed the declared token mapping")
    # Do not concatenate the entire registry into an artificial mega-surface:
    # slow SentencePiece tokenizers can spend minutes resolving that input and
    # it is not a molecule the model will see.  The paired producer performs
    # the stronger whole-surface equality check on every real A and M record.
    return token_ids


def _registry_origins(
    base_tokenizer: Any,
    base_vocab: Mapping[str, int],
    declared_tokens: Sequence[str],
) -> dict[str, str]:
    origins = {}
    for token in declared_tokens:
        if token not in base_vocab:
            origins[token] = "new_row"
            continue
        try:
            token_id = _exact_token_id(base_tokenizer, token)
        except CanaryUnionTokenizerError as exc:
            raise CanaryUnionTokenizerError(
                "declared surface overlaps a non-exact base token: {!r}".format(token)
            ) from exc
        if token_id != int(base_vocab[token]):
            raise CanaryUnionTokenizerError(
                "declared surface disagrees with its base vocabulary ID: {!r}".format(
                    token
                )
            )
        origins[token] = "base_id_reused_exact"
    return origins


def _validate_token_classification(
    tokenizer: Any,
    controls: Sequence[str],
    ordinary_tokens: Sequence[str],
) -> None:
    special = {str(token) for token in (getattr(tokenizer, "all_special_tokens", ()) or ())}
    missing = set(controls).difference(special)
    promoted = set(ordinary_tokens).intersection(special)
    if missing:
        raise CanaryUnionTokenizerError(
            "control tokens are not special after reload: {}".format(sorted(missing))
        )
    if promoted:
        raise CanaryUnionTokenizerError(
            "ordinary molecular tokens became special: {}".format(sorted(promoted))
        )


@dataclass(frozen=True)
class CanaryUnionTokenizerBuild:
    tokenizer: Any
    manifest: dict[str, object]
    runtime: ProductionTokenizerRuntime
    snapshot_path: Path


class RegistryBackedTokenizer:
    """Delegate tokenizer with O(1) singleton checks for the frozen registry.

    The builder has already proven each declared surface against the real slow
    tokenizer and its offline reload.  A/M codecs still send every complete
    molecular surface through the underlying tokenizer; only repeated
    singleton calls are served from the frozen token-to-ID table.
    """

    def __init__(self, tokenizer: Any, exact_token_ids: dict[str, int]) -> None:
        self._tokenizer = tokenizer
        self._exact_token_ids = dict(exact_token_ids)

    def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens is False and surface in self._exact_token_ids:
            return [self._exact_token_ids[surface]]
        return list(
            self._tokenizer.encode(
                surface,
                add_special_tokens=add_special_tokens,
            )
        )

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._tokenizer.convert_tokens_to_ids(token))

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return str(self._tokenizer.convert_ids_to_tokens(token_id))

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def build_canary_union_tokenizer(
    *,
    base_snapshot: Path,
    output_dir: Path,
    selfies_distribution_version: str,
    robust_selfies_symbols: Iterable[str],
    observed_selfies_symbols: Iterable[str],
    motif_macro_registry: Sequence[Mapping[str, object]],
) -> CanaryUnionTokenizerBuild:
    """Freeze one offline-reloadable shared tokenizer before canary training."""

    base_snapshot = Path(base_snapshot).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not base_snapshot.is_dir():
        raise CanaryUnionTokenizerError("base_snapshot must be an existing directory")
    staging_dir = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists():
        raise CanaryUnionTokenizerError("output_dir must be a new path")
    if staging_dir.exists():
        raise CanaryUnionTokenizerError(
            "staging path already exists from another or failed build: {}".format(
                staging_dir
            )
        )
    if selfies_distribution_version != SELFIES_DISTRIBUTION_VERSION:
        raise CanaryUnionTokenizerError(
            "SELFIES distribution version must be exactly {}".format(
                SELFIES_DISTRIBUTION_VERSION
            )
        )

    robust = _sorted_utf8(robust_selfies_symbols)
    observed = _sorted_utf8(observed_selfies_symbols)
    robust_surfaces = _selfies_token_surfaces(robust)
    observed_surfaces = _selfies_token_surfaces(observed)
    combined_symbols = _sorted_utf8((*robust, *observed))
    combined_surfaces = _selfies_token_surfaces(combined_symbols)
    selfies_symbol_registry = tuple(
        {
            "selfies_symbol": symbol,
            "token_surface": surface,
        }
        for symbol, surface in zip(combined_symbols, combined_surfaces)
    )
    macro_registry = _normalize_macro_registry(motif_macro_registry)
    macros = tuple(str(row["token"]) for row in macro_registry)
    controls = CONTROL_SPECIAL_TOKENS
    if set(GPORTS_UNION_TOKENS).intersection(macros):
        raise CanaryUnionTokenizerError(
            "motif macro tokens overlap the graph/ports grammar"
        )
    if set(combined_surfaces).intersection((*GPORTS_UNION_TOKENS, *macros)):
        raise CanaryUnionTokenizerError(
            "SELFIES tokenizer surfaces overlap another molecular token domain"
        )
    ordinary_tokens = _stable_unique(
        (*GPORTS_UNION_TOKENS, *robust_surfaces, *observed_surfaces, *macros)
    )
    if set(controls).intersection(ordinary_tokens):
        raise CanaryUnionTokenizerError("control tokens overlap the ordinary token domain")
    declared_tokens = _stable_unique((*controls, *ordinary_tokens))

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from transformers import AddedToken, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_snapshot),
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )
    base_tokenizer_class = tokenizer.__class__.__name__
    base_vocab_size = len(tokenizer)
    base_vocab = {
        str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()
    }
    base_vocab_sha256 = _sha256_json(base_vocab)
    registry_origins = _registry_origins(tokenizer, base_vocab, declared_tokens)
    natural_text_before = {
        surface: tuple(tokenizer.encode(surface, add_special_tokens=False))
        for surface in NATURAL_TEXT_REGRESSION_SURFACES
    }
    base_sentinel_ids = {
        token: _exact_token_id(tokenizer, token) for token in T5_SENTINELS
    }

    existing_special_objects = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    existing_decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    by_content = {
        str(getattr(token, "content", token)): token for token in existing_decoder.values()
    }
    preserved_special_objects = [
        by_content.get(str(token), token) for token in existing_special_objects
    ]
    new_controls = [
        AddedToken(
            token,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=True,
        )
        for token in controls
        if token not in {str(value) for value in existing_special_objects}
    ]
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                *preserved_special_objects,
                *new_controls,
            ]
        }
    )
    tokenizer.add_tokens(
        [
            AddedToken(
                token,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=False,
            )
            for token in ordinary_tokens
        ],
        special_tokens=False,
    )
    _validate_token_classification(tokenizer, controls, ordinary_tokens)

    if {
        token: _exact_token_id(tokenizer, token) for token in T5_SENTINELS
    } != base_sentinel_ids:
        raise CanaryUnionTokenizerError("T5 sentinel IDs changed during vocabulary extension")
    extended_vocab = {
        str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()
    }
    if any(
        extended_vocab.get(token) != token_id
        for token, token_id in base_vocab.items()
    ):
        raise CanaryUnionTokenizerError("a base vocabulary token ID changed")
    natural_text_after = {
        surface: tuple(tokenizer.encode(surface, add_special_tokens=False))
        for surface in NATURAL_TEXT_REGRESSION_SURFACES
    }
    if natural_text_after != natural_text_before:
        raise CanaryUnionTokenizerError("ordinary natural-text tokenization changed")
    declared_ids = _validate_declared_tokens(tokenizer, declared_tokens)

    contract = {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "final_pretraining_vocabulary": False,
        "vocabulary_extension_after_checkpoint_forbidden": True,
        "base_tokenizer_class": base_tokenizer_class,
        "base_vocab_size": base_vocab_size,
        "selfies_distribution_version": selfies_distribution_version,
        "control_special_tokens": list(controls),
        "gports_tokens": list(GPORTS_UNION_TOKENS),
        "robust_selfies_symbols": list(robust),
        "observed_selfies_symbols": list(observed),
        "robust_selfies_token_surfaces": list(robust_surfaces),
        "observed_selfies_token_surfaces": list(observed_surfaces),
        "selfies_symbol_registry": list(selfies_symbol_registry),
        "motif_macro_registry": list(macro_registry),
        "motif_macro_tokens": list(macros),
        "declared_token_order": list(declared_tokens),
        "declared_token_ids": declared_ids,
        "registry_token_origin": registry_origins,
        "base_vocab_sha256": base_vocab_sha256,
        "natural_text_regression": {
            surface: list(ids) for surface, ids in natural_text_before.items()
        },
        "base_sentinel_token_ids": base_sentinel_ids,
    }
    contract_sha256 = _sha256_json(contract)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    staging_snapshot_path = staging_dir / SNAPSHOT_DIRECTORY
    tokenizer.save_pretrained(str(staging_snapshot_path))
    reloaded = AutoTokenizer.from_pretrained(
        str(staging_snapshot_path),
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )
    _validate_token_classification(reloaded, controls, ordinary_tokens)
    _validate_declared_tokens(reloaded, declared_tokens, declared_ids)
    reloaded_vocab = {
        str(token): int(token_id) for token, token_id in reloaded.get_vocab().items()
    }
    if any(reloaded_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise CanaryUnionTokenizerError("offline reload changed a base vocabulary token ID")
    if {
        surface: tuple(reloaded.encode(surface, add_special_tokens=False))
        for surface in NATURAL_TEXT_REGRESSION_SURFACES
    } != natural_text_before:
        raise CanaryUnionTokenizerError("offline reload changed ordinary natural-text tokenization")
    reloaded_sentinel_ids = {
        token: _exact_token_id(reloaded, token) for token in T5_SENTINELS
    }
    if reloaded_sentinel_ids != base_sentinel_ids:
        raise CanaryUnionTokenizerError("offline reload changed T5 sentinel IDs")

    snapshot = _observe_snapshot(staging_snapshot_path)
    snapshot_sha256 = str(snapshot["tree_sha256"])
    sentinel_ids = tuple(reloaded_sentinel_ids[token] for token in T5_SENTINELS)
    runtime = ProductionTokenizerRuntime(
        tokenizer_contract_sha256=contract_sha256,
        tokenizer_snapshot_sha256=snapshot_sha256,
        vocab_size=len(reloaded),
        pad_token_id=int(reloaded.pad_token_id),
        eos_token_id=int(reloaded.eos_token_id),
        sentinel_token_ids=sentinel_ids,
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "scope": SCOPE,
        "sample_bound": True,
        "training_admission": False,
        "tokenizer_contract_sha256": contract_sha256,
        "tokenizer_snapshot_sha256": snapshot_sha256,
        "contract": contract,
        "snapshot": snapshot,
        "counts": {
            "base_vocab_size": base_vocab_size,
            "final_vocab_size": len(reloaded),
            "added_vocab_size": len(reloaded) - base_vocab_size,
            "control_special_tokens": len(controls),
            "gports_tokens": len(GPORTS_UNION_TOKENS),
            "robust_selfies_symbols": len(robust),
            "observed_selfies_symbols": len(observed),
            "unique_selfies_token_surfaces": len(combined_surfaces),
            "motif_macro_tokens": len(macros),
        },
        "token_ids": {
            "pad": runtime.pad_token_id,
            "eos": runtime.eos_token_id,
            "unk": int(reloaded.unk_token_id),
            "sentinels": list(sentinel_ids),
            "declared": declared_ids,
        },
    }
    manifest_path = staging_dir / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    staging_dir.rename(output_dir)
    snapshot_path = output_dir / SNAPSHOT_DIRECTORY
    return CanaryUnionTokenizerBuild(
        tokenizer=RegistryBackedTokenizer(reloaded, declared_ids),
        manifest=manifest,
        runtime=runtime,
        snapshot_path=snapshot_path,
    )


def load_verified_canary_union_tokenizer(
    *,
    base_snapshot: Path,
    output_dir: Path,
) -> CanaryUnionTokenizerBuild:
    """Reload a published candidate only after replaying its frozen contract."""

    base_snapshot = Path(base_snapshot).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    snapshot_path = output_dir / SNAPSHOT_DIRECTORY
    if not base_snapshot.is_dir():
        raise CanaryUnionTokenizerError("base_snapshot must be an existing directory")
    if not output_dir.is_dir() or not manifest_path.is_file() or not snapshot_path.is_dir():
        raise CanaryUnionTokenizerError("published tokenizer directory is incomplete")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise CanaryUnionTokenizerError("tokenizer manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise CanaryUnionTokenizerError("tokenizer manifest must be a JSON object")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "candidate"
        or manifest.get("scope") != SCOPE
        or manifest.get("sample_bound") is not True
        or manifest.get("training_admission") is not False
    ):
        raise CanaryUnionTokenizerError("tokenizer manifest publication state is invalid")

    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise CanaryUnionTokenizerError("tokenizer contract must be a JSON object")
    contract_sha256 = _sha256_json(contract)
    if manifest.get("tokenizer_contract_sha256") != contract_sha256:
        raise CanaryUnionTokenizerError("tokenizer contract hash mismatch")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("scope") != SCOPE
        or contract.get("final_pretraining_vocabulary") is not False
        or contract.get("vocabulary_extension_after_checkpoint_forbidden") is not True
        or contract.get("selfies_distribution_version") != SELFIES_DISTRIBUTION_VERSION
    ):
        raise CanaryUnionTokenizerError("tokenizer contract state is invalid")

    observed_snapshot = _observe_snapshot(snapshot_path)
    if manifest.get("snapshot") != observed_snapshot:
        raise CanaryUnionTokenizerError("saved tokenizer tree differs from the manifest")
    snapshot_sha256 = str(observed_snapshot["tree_sha256"])
    if manifest.get("tokenizer_snapshot_sha256") != snapshot_sha256:
        raise CanaryUnionTokenizerError("saved tokenizer tree hash mismatch")

    controls_raw = contract.get("control_special_tokens")
    gports_raw = contract.get("gports_tokens")
    robust_raw = contract.get("robust_selfies_symbols")
    observed_raw = contract.get("observed_selfies_symbols")
    if not isinstance(controls_raw, list) or tuple(controls_raw) != CONTROL_SPECIAL_TOKENS:
        raise CanaryUnionTokenizerError("control special-token contract changed")
    if not isinstance(gports_raw, list) or tuple(gports_raw) != GPORTS_UNION_TOKENS:
        raise CanaryUnionTokenizerError("graph/ports token contract changed")
    if not isinstance(robust_raw, list) or not isinstance(observed_raw, list):
        raise CanaryUnionTokenizerError("SELFIES symbol registries must be lists")
    robust = _sorted_utf8(robust_raw)
    observed = _sorted_utf8(observed_raw)
    if tuple(robust_raw) != robust or tuple(observed_raw) != observed:
        raise CanaryUnionTokenizerError("SELFIES symbol registries are not canonical")
    robust_surfaces = _selfies_token_surfaces(robust)
    observed_surfaces = _selfies_token_surfaces(observed)
    combined_symbols = _sorted_utf8((*robust, *observed))
    combined_surfaces = _selfies_token_surfaces(combined_symbols)
    expected_selfies_registry = [
        {"selfies_symbol": symbol, "token_surface": surface}
        for symbol, surface in zip(combined_symbols, combined_surfaces)
    ]
    if (
        contract.get("robust_selfies_token_surfaces") != list(robust_surfaces)
        or contract.get("observed_selfies_token_surfaces") != list(observed_surfaces)
        or contract.get("selfies_symbol_registry") != expected_selfies_registry
    ):
        raise CanaryUnionTokenizerError("SELFIES symbol-to-token mapping changed")

    macro_rows = contract.get("motif_macro_registry")
    if not isinstance(macro_rows, list):
        raise CanaryUnionTokenizerError("motif macro registry must be a list")
    macro_inputs = []
    required_macro_keys = {
        "rank",
        "identity",
        "identity_sha256",
        "token",
        "occurrence_count",
    }
    for offset, row in enumerate(macro_rows):
        if not isinstance(row, dict) or set(row) != required_macro_keys:
            raise CanaryUnionTokenizerError(
                "normalized macro row {} has invalid fields".format(offset)
            )
        macro_inputs.append(
            {
                "identity": row["identity"],
                "token": row["token"],
                "occurrence_count": row["occurrence_count"],
            }
        )
    macro_registry = _normalize_macro_registry(macro_inputs)
    if list(macro_registry) != macro_rows:
        raise CanaryUnionTokenizerError("motif macro registry is not canonical")
    macros = tuple(str(row["token"]) for row in macro_registry)
    if contract.get("motif_macro_tokens") != list(macros):
        raise CanaryUnionTokenizerError("motif macro token order changed")

    ordinary_tokens = _stable_unique(
        (*GPORTS_UNION_TOKENS, *robust_surfaces, *observed_surfaces, *macros)
    )
    if set(CONTROL_SPECIAL_TOKENS).intersection(ordinary_tokens):
        raise CanaryUnionTokenizerError("control and ordinary token domains overlap")
    declared_tokens = _stable_unique((*CONTROL_SPECIAL_TOKENS, *ordinary_tokens))
    if contract.get("declared_token_order") != list(declared_tokens):
        raise CanaryUnionTokenizerError("declared token order changed")
    contract_declared_ids_raw = contract.get("declared_token_ids")
    if not isinstance(contract_declared_ids_raw, dict) or set(
        contract_declared_ids_raw
    ) != set(declared_tokens) or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in contract_declared_ids_raw.values()
    ):
        raise CanaryUnionTokenizerError("contract declared token IDs are invalid")
    contract_declared_ids = {
        token: int(contract_declared_ids_raw[token]) for token in declared_tokens
    }

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(
        str(base_snapshot),
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )
    base_vocab = {
        str(token): int(token_id) for token, token_id in base.get_vocab().items()
    }
    if (
        contract.get("base_tokenizer_class") != base.__class__.__name__
        or contract.get("base_vocab_size") != len(base)
        or contract.get("base_vocab_sha256") != _sha256_json(base_vocab)
    ):
        raise CanaryUnionTokenizerError("base tokenizer differs from the build contract")
    base_natural_text = {
        surface: list(base.encode(surface, add_special_tokens=False))
        for surface in NATURAL_TEXT_REGRESSION_SURFACES
    }
    if contract.get("natural_text_regression") != base_natural_text:
        raise CanaryUnionTokenizerError("base natural-text regression changed")
    base_sentinel_ids = {
        token: _exact_token_id(base, token) for token in T5_SENTINELS
    }
    if contract.get("base_sentinel_token_ids") != base_sentinel_ids:
        raise CanaryUnionTokenizerError("base T5 sentinel mapping changed")
    if contract.get("registry_token_origin") != _registry_origins(
        base, base_vocab, declared_tokens
    ):
        raise CanaryUnionTokenizerError("declared token origin registry changed")

    reloaded = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        use_fast=False,
        local_files_only=True,
        trust_remote_code=False,
    )
    _validate_token_classification(reloaded, CONTROL_SPECIAL_TOKENS, ordinary_tokens)
    reloaded_vocab = {
        str(token): int(token_id) for token, token_id in reloaded.get_vocab().items()
    }
    if any(reloaded_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise CanaryUnionTokenizerError("offline reload changed a base vocabulary token ID")
    if {
        surface: list(reloaded.encode(surface, add_special_tokens=False))
        for surface in NATURAL_TEXT_REGRESSION_SURFACES
    } != base_natural_text:
        raise CanaryUnionTokenizerError("offline reload changed ordinary natural text")
    reloaded_sentinel_ids = {
        token: _exact_token_id(reloaded, token) for token in T5_SENTINELS
    }
    if reloaded_sentinel_ids != base_sentinel_ids:
        raise CanaryUnionTokenizerError("offline reload changed T5 sentinel IDs")

    token_ids = manifest.get("token_ids")
    if not isinstance(token_ids, dict) or not isinstance(token_ids.get("declared"), dict):
        raise CanaryUnionTokenizerError("manifest token ID table is invalid")
    declared_ids_raw = token_ids["declared"]
    if set(declared_ids_raw) != set(declared_tokens) or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in declared_ids_raw.values()
    ):
        raise CanaryUnionTokenizerError("manifest declared token IDs are invalid")
    declared_ids = {token: int(declared_ids_raw[token]) for token in declared_tokens}
    if declared_ids != contract_declared_ids:
        raise CanaryUnionTokenizerError("manifest and contract token IDs disagree")
    _validate_declared_tokens(reloaded, declared_tokens, declared_ids)

    sentinel_ids = tuple(reloaded_sentinel_ids[token] for token in T5_SENTINELS)
    runtime = ProductionTokenizerRuntime(
        tokenizer_contract_sha256=contract_sha256,
        tokenizer_snapshot_sha256=snapshot_sha256,
        vocab_size=len(reloaded),
        pad_token_id=int(reloaded.pad_token_id),
        eos_token_id=int(reloaded.eos_token_id),
        sentinel_token_ids=sentinel_ids,
    )
    expected_counts = {
        "base_vocab_size": len(base),
        "final_vocab_size": len(reloaded),
        "added_vocab_size": len(reloaded) - len(base),
        "control_special_tokens": len(CONTROL_SPECIAL_TOKENS),
        "gports_tokens": len(GPORTS_UNION_TOKENS),
        "robust_selfies_symbols": len(robust),
        "observed_selfies_symbols": len(observed),
        "unique_selfies_token_surfaces": len(combined_surfaces),
        "motif_macro_tokens": len(macros),
    }
    expected_token_ids = {
        "pad": runtime.pad_token_id,
        "eos": runtime.eos_token_id,
        "unk": int(reloaded.unk_token_id),
        "sentinels": list(sentinel_ids),
        "declared": declared_ids,
    }
    if manifest.get("counts") != expected_counts or token_ids != expected_token_ids:
        raise CanaryUnionTokenizerError("manifest tokenizer dimensions or IDs changed")
    return CanaryUnionTokenizerBuild(
        tokenizer=RegistryBackedTokenizer(reloaded, declared_ids),
        manifest=manifest,
        runtime=runtime,
        snapshot_path=snapshot_path,
    )


__all__ = [
    "CanaryUnionTokenizerBuild",
    "CanaryUnionTokenizerError",
    "CONTROL_SPECIAL_TOKENS",
    "RegistryBackedTokenizer",
    "SCHEMA_VERSION",
    "SCOPE",
    "T5_SENTINELS",
    "build_canary_union_tokenizer",
    "load_verified_canary_union_tokenizer",
]
