#!/usr/bin/env python3
"""Instantiate one anchored tokenizer plan against an offline T5 snapshot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from most_t5_next.p1.production_bridge import ProductionTokenizerRuntime

from most_t5_next.p1.build_anchored_tokenizer_plan_v1 import SCHEMA_VERSION as PLAN_SCHEMA
from most_t5_next.r1.tokenizer.anchored_motif_model_surface_v1 import (
    FALLBACK_MOTIF_PREFIX,
    FALLBACK_MOTIF_SUFFIX,
    FROZEN_GENERATIVE_BOUNDARY_MODE,
    frozen_grammar_contract,
)

SCHEMA_VERSION = "most-t5-next/anchored-candidate-tokenizer/v2"
MANIFEST_NAME = "manifest.json"
SNAPSHOT_DIRECTORY = "tokenizer_snapshot"
T5_SENTINELS = tuple(f"<extra_id_{index}>" for index in range(100))
NATURAL_TEXT_REGRESSION_SURFACES = (
    "This molecule is aromatic.",
    "A. B.",
    "C.O",
    "CCO",
    "The HOMO-LUMO gap is 1.0 eV.",
)


class AnchoredCandidateTokenizerError(RuntimeError):
    """A planned tokenizer cannot be instantiated without contract drift."""


@dataclass(frozen=True)
class AnchoredCandidateTokenizerBuild:
    """Offline-reloaded tokenizer plus its selected semantic-plan identity."""

    tokenizer: Any
    manifest: dict[str, object]
    runtime: ProductionTokenizerRuntime
    snapshot_path: Path


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnchoredCandidateTokenizerError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AnchoredCandidateTokenizerError(f"JSON root must be an object: {path}")
    return value


def _observe_tree(path: Path) -> dict[str, object]:
    rows = []
    for candidate in sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix().encode("utf-8"),
    ):
        rows.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    if not rows:
        raise AnchoredCandidateTokenizerError("saved tokenizer snapshot is empty")
    return {
        "files": rows,
        "tree_sha256": _sha256_bytes(_canonical_json(rows).encode("utf-8")),
    }


def _validate_plan_bundle(
    bundle: Path, plan_name: str
) -> tuple[dict[str, object], dict[str, object], Path]:
    bundle_manifest_path = bundle / "manifest.json"
    manifest = _load_json(bundle_manifest_path)
    if manifest.get("schema_version") != PLAN_SCHEMA or manifest.get("status") != "candidate":
        raise AnchoredCandidateTokenizerError("token plan bundle is not a candidate")
    registries = manifest.get("registries")
    if not isinstance(registries, dict):
        raise AnchoredCandidateTokenizerError("token plan registry contract is absent")
    for name, descriptor in registries.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise AnchoredCandidateTokenizerError("token plan registry contract is malformed")
        path = bundle / name
        if descriptor.get("sha256") != _sha256_file(path):
            raise AnchoredCandidateTokenizerError(f"token plan registry drift: {name}")

    plan_rows = manifest.get("plans")
    if not isinstance(plan_rows, list):
        raise AnchoredCandidateTokenizerError("token plan list is absent")
    matches = [row for row in plan_rows if isinstance(row, dict) and row.get("path") == plan_name]
    if len(matches) != 1:
        raise AnchoredCandidateTokenizerError("requested plan is not uniquely declared")
    plan_path = bundle / plan_name
    if matches[0].get("sha256") != _sha256_file(plan_path):
        raise AnchoredCandidateTokenizerError("requested plan file hash drifted")
    plan = _load_json(plan_path)
    claimed_plan_sha = plan.pop("plan_sha256", None)
    observed_plan_sha = _sha256_bytes(_canonical_json(plan).encode("utf-8"))
    plan["plan_sha256"] = claimed_plan_sha
    if claimed_plan_sha != observed_plan_sha or matches[0].get("plan_sha256") != observed_plan_sha:
        raise AnchoredCandidateTokenizerError("requested plan semantic hash drifted")
    grammar_decision = manifest.get("grammar_decision")
    if (
        not isinstance(grammar_decision, dict)
        or grammar_decision.get("status") != "frozen"
        or grammar_decision.get("boundary_mode") != FROZEN_GENERATIVE_BOUNDARY_MODE
        or grammar_decision.get("contract") != frozen_grammar_contract()
        or plan.get("boundary_mode") != FROZEN_GENERATIVE_BOUNDARY_MODE
        or plan.get("grammar_contract") != frozen_grammar_contract()
    ):
        raise AnchoredCandidateTokenizerError("token plan does not bind the frozen grammar")
    return manifest, plan, plan_path


def _validate_shared_surface_plans(
    bundle: Path,
    manifest: dict[str, object],
    reference_plan: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Prove that macro-policy choices share one tokenizer token surface."""

    rows = manifest.get("plans")
    if not isinstance(rows, list) or not rows:
        raise AnchoredCandidateTokenizerError("token plan list is absent")
    reference_tokens = reference_plan.get("declared_added_tokens")
    compatible = []
    for descriptor in rows:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
            raise AnchoredCandidateTokenizerError("token plan descriptor is malformed")
        path = bundle / descriptor["path"]
        if descriptor.get("sha256") != _sha256_file(path):
            raise AnchoredCandidateTokenizerError("compatible plan file hash drifted")
        plan = _load_json(path)
        claimed = plan.pop("plan_sha256", None)
        observed = _sha256_bytes(_canonical_json(plan).encode("utf-8"))
        plan["plan_sha256"] = claimed
        if (
            claimed != observed
            or descriptor.get("plan_sha256") != observed
            or plan.get("boundary_mode") != FROZEN_GENERATIVE_BOUNDARY_MODE
            or plan.get("grammar_contract") != frozen_grammar_contract()
            or plan.get("declared_added_tokens") != reference_tokens
        ):
            raise AnchoredCandidateTokenizerError(
                "macro-policy plans do not share one frozen token surface"
            )
        compatible.append(
            {
                "macro_policy": plan.get("macro_policy"),
                "plan_path": descriptor["path"],
                "plan_sha256": observed,
            }
        )
    policies = {row["macro_policy"] for row in compatible}
    if policies != {
        "pretrain_train_only",
        "balanced_pretrain_plus_registered_downstream_train",
    }:
        raise AnchoredCandidateTokenizerError("macro-policy plan set is incomplete")
    return tuple(compatible)


def _exact_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0 or token_id == tokenizer.unk_token_id:
        raise AnchoredCandidateTokenizerError(f"token is absent: {token!r}")
    if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
        raise AnchoredCandidateTokenizerError(f"token is not an exact singleton: {token!r}")
    return token_id


def _declared_added_ids(
    tokenizer: Any, planned_rows: Sequence[tuple[str, int]]
) -> dict[str, int]:
    """Verify AddedToken registration without quadratic slow-tokenizer probes."""

    added_vocab = tokenizer.get_added_vocab()
    special = set(tokenizer.all_special_tokens)
    decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    observed = {}
    for token, planned_id in planned_rows:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_object = decoder.get(planned_id)
        if (
            token_id != planned_id
            or added_vocab.get(token) != planned_id
            or token in special
            or token_object is None
            or str(getattr(token_object, "content", token_object)) != token
            or bool(getattr(token_object, "special", False))
        ):
            raise AnchoredCandidateTokenizerError(
                f"ordinary AddedToken contract drifted: {token!r}"
            )
        observed[token] = token_id
    return observed


def _validate_base_vocab_unchanged(
    tokenizer: Any, base_vocab: dict[str, int]
) -> None:
    """Validate in linear time; slow tokenizers rebuild get_vocab on each call."""

    extended_vocab = tokenizer.get_vocab()
    if any(extended_vocab.get(token) != token_id for token, token_id in base_vocab.items()):
        raise AnchoredCandidateTokenizerError("a base token ID changed")


def _selected_semantic_plan(
    manifest: Mapping[str, object], semantic_plan_sha256: str
) -> dict[str, object]:
    if (
        not isinstance(semantic_plan_sha256, str)
        or len(semantic_plan_sha256) != 64
        or any(char not in "0123456789abcdef" for char in semantic_plan_sha256)
    ):
        raise AnchoredCandidateTokenizerError(
            "semantic_plan_sha256 must be a lower-case SHA-256"
        )
    plan = manifest.get("plan")
    if not isinstance(plan, Mapping):
        raise AnchoredCandidateTokenizerError("candidate tokenizer plan is absent")
    compatible = plan.get("compatible_semantic_plans")
    if not isinstance(compatible, list):
        raise AnchoredCandidateTokenizerError("compatible semantic plans are absent")
    matches = [
        row
        for row in compatible
        if isinstance(row, dict) and row.get("plan_sha256") == semantic_plan_sha256
    ]
    if len(matches) != 1:
        raise AnchoredCandidateTokenizerError(
            "selected semantic plan is not uniquely admitted by the snapshot"
        )
    return dict(matches[0])


def load_verified_anchored_candidate_tokenizer(
    *,
    base_snapshot: Path,
    output_dir: Path,
    semantic_plan_sha256: str,
) -> AnchoredCandidateTokenizerBuild:
    """Reload the shared snapshot under one explicit macro-semantic plan.

    The saved token surface is intentionally shared by two macro policies, so
    the snapshot tree hash alone cannot name the molecular semantics.  The
    caller must select one compatible plan SHA; that SHA becomes the tokenizer
    contract identity consumed by records and model checkpoints.
    """

    base_snapshot = Path(base_snapshot).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    snapshot_path = output_dir / SNAPSHOT_DIRECTORY
    if not base_snapshot.is_dir():
        raise AnchoredCandidateTokenizerError("base_snapshot must be an existing directory")
    if not output_dir.is_dir() or not manifest_path.is_file() or not snapshot_path.is_dir():
        raise AnchoredCandidateTokenizerError("published tokenizer directory is incomplete")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "candidate"
        or manifest.get("training_admission") is not False
    ):
        raise AnchoredCandidateTokenizerError("candidate tokenizer publication state is invalid")
    selected_plan = _selected_semantic_plan(manifest, semantic_plan_sha256)

    observed_tree = _observe_tree(snapshot_path)
    if manifest.get("snapshot") != observed_tree:
        raise AnchoredCandidateTokenizerError("saved tokenizer tree differs from the manifest")
    snapshot_sha256 = observed_tree.get("tree_sha256")
    if not isinstance(snapshot_sha256, str):
        raise AnchoredCandidateTokenizerError("saved tokenizer tree identity is absent")

    plan = manifest.get("plan")
    token_ids = manifest.get("token_ids")
    regressions = manifest.get("natural_text_regression")
    contracts = manifest.get("contracts")
    if (
        not isinstance(plan, Mapping)
        or not isinstance(token_ids, Mapping)
        or not isinstance(regressions, Mapping)
        or not isinstance(contracts, Mapping)
    ):
        raise AnchoredCandidateTokenizerError("candidate tokenizer manifest is incomplete")
    if (
        plan.get("boundary_mode") != FROZEN_GENERATIVE_BOUNDARY_MODE
        or contracts.get("one_snapshot_shared_across_macro_policies") is not True
        or contracts.get("all_additions_are_ordinary") is not True
        or contracts.get("offline_reload_verified") is not True
    ):
        raise AnchoredCandidateTokenizerError("candidate tokenizer contracts drifted")

    declared_raw = token_ids.get("declared")
    sentinels_raw = token_ids.get("sentinels")
    if not isinstance(declared_raw, Mapping) or not isinstance(sentinels_raw, Mapping):
        raise AnchoredCandidateTokenizerError("candidate token ID tables are malformed")
    declared_ids: dict[str, int] = {}
    for token, token_id in declared_raw.items():
        if (
            not isinstance(token, str)
            or not token
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
        ):
            raise AnchoredCandidateTokenizerError("candidate declared token IDs are malformed")
        declared_ids[token] = int(token_id)
    if not declared_ids or FALLBACK_MOTIF_SUFFIX not in declared_ids:
        raise AnchoredCandidateTokenizerError("candidate declared token domain is incomplete")

    os.environ.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(
        str(base_snapshot), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    reloaded = AutoTokenizer.from_pretrained(
        str(snapshot_path), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    base_size = plan.get("base_vocab_size")
    final_size = plan.get("final_vocab_size")
    added_size = plan.get("added_vocab_size")
    if (
        isinstance(base_size, bool)
        or not isinstance(base_size, int)
        or isinstance(final_size, bool)
        or not isinstance(final_size, int)
        or isinstance(added_size, bool)
        or not isinstance(added_size, int)
        or len(base) != base_size
        or len(reloaded) != final_size
        or final_size - base_size != added_size
    ):
        raise AnchoredCandidateTokenizerError("candidate tokenizer dimensions drifted")
    base_vocab = dict(base.get_vocab())
    _validate_base_vocab_unchanged(reloaded, base_vocab)

    planned_rows = sorted(declared_ids.items(), key=lambda row: row[1])
    if [token_id for _, token_id in planned_rows] != list(
        range(base_size, final_size)
    ):
        raise AnchoredCandidateTokenizerError("declared token IDs are not the exact added interval")
    _declared_added_ids(reloaded, planned_rows)
    if set(declared_ids).intersection(set(reloaded.all_special_tokens)):
        raise AnchoredCandidateTokenizerError("an anchored molecular token became special")

    observed_regressions = {
        text: reloaded.encode(text, add_special_tokens=False)
        for text in NATURAL_TEXT_REGRESSION_SURFACES
    }
    base_regressions = {
        text: base.encode(text, add_special_tokens=False)
        for text in NATURAL_TEXT_REGRESSION_SURFACES
    }
    if observed_regressions != base_regressions or observed_regressions != dict(regressions):
        raise AnchoredCandidateTokenizerError("natural-text regression changed")
    observed_sentinels = {token: _exact_id(reloaded, token) for token in T5_SENTINELS}
    base_sentinels = {token: _exact_id(base, token) for token in T5_SENTINELS}
    if observed_sentinels != base_sentinels or observed_sentinels != dict(sentinels_raw):
        raise AnchoredCandidateTokenizerError("T5 sentinel mapping changed")
    if (
        token_ids.get("pad") != reloaded.pad_token_id
        or token_ids.get("eos") != reloaded.eos_token_id
        or token_ids.get("unk") != reloaded.unk_token_id
    ):
        raise AnchoredCandidateTokenizerError("base special-token IDs changed")

    runtime = ProductionTokenizerRuntime(
        tokenizer_contract_sha256=semantic_plan_sha256,
        tokenizer_snapshot_sha256=snapshot_sha256,
        vocab_size=len(reloaded),
        pad_token_id=int(reloaded.pad_token_id),
        eos_token_id=int(reloaded.eos_token_id),
        sentinel_token_ids=tuple(observed_sentinels[token] for token in T5_SENTINELS),
    )
    normalized_manifest = dict(manifest)
    normalized_manifest.update(
        {
            "counts": {
                "base_vocab_size": base_size,
                "final_vocab_size": final_size,
                "added_vocab_size": added_size,
            },
            "selected_semantic_plan": selected_plan,
            "tokenizer_contract_sha256": semantic_plan_sha256,
            "tokenizer_snapshot_sha256": snapshot_sha256,
        }
    )
    return AnchoredCandidateTokenizerBuild(
        tokenizer=reloaded,
        manifest=normalized_manifest,
        runtime=runtime,
        snapshot_path=snapshot_path,
    )


def build(args: argparse.Namespace) -> dict[str, object]:
    base = Path(args.base_snapshot).expanduser().resolve()
    bundle = Path(args.plan_bundle).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    staging = output.with_name(output.name + ".staging")
    if not base.is_dir() or not bundle.is_dir():
        raise AnchoredCandidateTokenizerError("base snapshot and plan bundle must exist")
    if output.exists() or staging.exists():
        raise AnchoredCandidateTokenizerError("output and sibling staging must be absent")
    bundle_manifest, plan, plan_path = _validate_plan_bundle(bundle, args.plan_name)
    compatible_surface_plans = _validate_shared_surface_plans(
        bundle, bundle_manifest, plan
    )

    declared = plan.get("declared_added_tokens")
    if not isinstance(declared, list) or not declared:
        raise AnchoredCandidateTokenizerError("plan has no declared token additions")
    planned_rows = []
    surfaces = []
    for offset, row in enumerate(declared):
        if not isinstance(row, dict) or set(row) != {"token_id", "surface_token"}:
            raise AnchoredCandidateTokenizerError("declared token row is malformed")
        token = row["surface_token"]
        token_id = row["token_id"]
        expected_id = plan.get("base_vocab_size") + offset
        if not isinstance(token, str) or not token or token_id != expected_id:
            raise AnchoredCandidateTokenizerError("declared token order/ID is not contiguous")
        planned_rows.append((token, token_id))
        surfaces.append(token)
    if len(surfaces) != len(set(surfaces)):
        raise AnchoredCandidateTokenizerError("declared token surfaces are not unique")
    if (
        surfaces.count(FALLBACK_MOTIF_SUFFIX) != 1
        or FALLBACK_MOTIF_PREFIX in surfaces
        or surfaces[0] != FALLBACK_MOTIF_SUFFIX
    ):
        raise AnchoredCandidateTokenizerError("frozen fallback suffix token contract drifted")

    os.environ.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    from transformers import AddedToken, AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(
        str(base), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    if len(tokenizer) != plan.get("base_vocab_size"):
        raise AnchoredCandidateTokenizerError("base tokenizer size differs from the plan")
    base_vocab = dict(tokenizer.get_vocab())
    natural_before = {
        text: tokenizer.encode(text, add_special_tokens=False)
        for text in NATURAL_TEXT_REGRESSION_SURFACES
    }
    sentinels_before = {token: _exact_id(tokenizer, token) for token in T5_SENTINELS}
    added = tokenizer.add_tokens(
        [
            AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False)
            for token in surfaces
        ],
        special_tokens=False,
    )
    if added != len(surfaces) or len(tokenizer) != plan.get("final_vocab_size"):
        raise AnchoredCandidateTokenizerError("tokenizer did not add the exact planned vocabulary")
    _validate_base_vocab_unchanged(tokenizer, base_vocab)
    observed_ids = _declared_added_ids(tokenizer, planned_rows)
    if any(observed_ids[token] != token_id for token, token_id in planned_rows):
        raise AnchoredCandidateTokenizerError("planned added-token ID differs from tokenizer")
    natural_after = {
        text: tokenizer.encode(text, add_special_tokens=False)
        for text in NATURAL_TEXT_REGRESSION_SURFACES
    }
    if natural_after != natural_before:
        raise AnchoredCandidateTokenizerError("natural-text tokenization changed")
    if {
        token: tokenizer.convert_tokens_to_ids(token) for token in T5_SENTINELS
    } != sentinels_before:
        raise AnchoredCandidateTokenizerError("T5 sentinel IDs changed")

    staging.mkdir(parents=True)
    snapshot_path = staging / SNAPSHOT_DIRECTORY
    tokenizer.save_pretrained(str(snapshot_path))
    reloaded = AutoTokenizer.from_pretrained(
        str(snapshot_path), use_fast=False, local_files_only=True, trust_remote_code=False
    )
    if len(reloaded) != len(tokenizer):
        raise AnchoredCandidateTokenizerError("offline reload changed vocabulary size")
    if _declared_added_ids(reloaded, planned_rows) != observed_ids:
        raise AnchoredCandidateTokenizerError("offline reload changed planned token IDs")
    if {
        text: reloaded.encode(text, add_special_tokens=False)
        for text in NATURAL_TEXT_REGRESSION_SURFACES
    } != natural_before:
        raise AnchoredCandidateTokenizerError("offline reload changed natural text")
    if {
        token: reloaded.convert_tokens_to_ids(token) for token in T5_SENTINELS
    } != sentinels_before:
        raise AnchoredCandidateTokenizerError("offline reload changed sentinels")

    tree = _observe_tree(snapshot_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "scope": "stage3_phrase_and_vocabulary_selection",
        "training_admission": False,
        "base_snapshot": {"path": str(base)},
        "plan_bundle_manifest_sha256": _sha256_file(bundle / "manifest.json"),
        "plan_file": {
            "path": plan_path.name,
            "sha256": _sha256_file(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "plan": {
            "boundary_mode": plan["boundary_mode"],
            "base_vocab_size": plan["base_vocab_size"],
            "added_vocab_size": len(surfaces),
            "final_vocab_size": len(reloaded),
            "surface_source_macro_policy": plan["macro_policy"],
            "compatible_semantic_plans": compatible_surface_plans,
            "macro_identity_mapping_is_external_to_snapshot": True,
        },
        "runtime": {
            "transformers": transformers_version,
            "sentencepiece_distribution": metadata.version("sentencepiece"),
            "use_fast": False,
        },
        "token_ids": {
            "declared": observed_ids,
            "sentinels": sentinels_before,
            "pad": reloaded.pad_token_id,
            "eos": reloaded.eos_token_id,
            "unk": reloaded.unk_token_id,
        },
        "natural_text_regression": natural_before,
        "snapshot": tree,
        "contracts": {
            "frozen_grammar_bound": True,
            "fallback_suffix_is_ordinary": True,
            "one_snapshot_shared_across_macro_policies": True,
            "all_additions_are_ordinary": True,
            "natural_text_unchanged": True,
            "base_ids_unchanged": True,
            "sentinel_ids_unchanged": True,
            "offline_reload_verified": True,
            "model_embeddings_resized": False,
            "training_admission": False,
        },
    }
    (staging / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--plan-bundle", required=True)
    parser.add_argument("--plan-name", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = build(_parser().parse_args(argv))
    except Exception as exc:
        print(f"candidate tokenizer build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
