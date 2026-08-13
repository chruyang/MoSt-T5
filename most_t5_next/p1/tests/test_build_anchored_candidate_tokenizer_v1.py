from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1.build_anchored_candidate_tokenizer_v1 import (
    AnchoredCandidateTokenizerError,
    _selected_semantic_plan,
    _validate_base_vocab_unchanged,
    _validate_shared_surface_plans,
    _validate_plan_bundle,
)
from most_t5_next.p1.build_anchored_tokenizer_plan_v1 import build


def _row(pure: str, train: int, dev: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "pure_motif": pure,
        "pure_motif_sha256": hashlib.sha256(pure.encode()).hexdigest(),
        "train_occurrences": train,
    }
    if dev is not None:
        result["dev_occurrences"] = dev
    return result


class AnchoredCandidateTokenizerV1Test(unittest.TestCase):
    def test_semantic_plan_must_be_explicit_and_uniquely_admitted(self) -> None:
        plan_sha = "a" * 64
        manifest = {
            "plan": {
                "compatible_semantic_plans": [
                    {
                        "macro_policy": "pretrain_train_only",
                        "plan_path": "plan.json",
                        "plan_sha256": plan_sha,
                    }
                ]
            }
        }
        self.assertEqual(
            _selected_semantic_plan(manifest, plan_sha)["macro_policy"],
            "pretrain_train_only",
        )
        with self.assertRaisesRegex(
            AnchoredCandidateTokenizerError, "not uniquely admitted"
        ):
            _selected_semantic_plan(manifest, "b" * 64)
        with self.assertRaisesRegex(AnchoredCandidateTokenizerError, "lower-case"):
            _selected_semantic_plan(manifest, "A" * 64)

    def test_base_vocab_validation_reads_extended_vocab_once(self) -> None:
        class CountingTokenizer:
            def __init__(self) -> None:
                self.calls = 0

            def get_vocab(self) -> dict[str, int]:
                self.calls += 1
                return {"a": 0, "b": 1, "new": 2}

        tokenizer = CountingTokenizer()
        _validate_base_vocab_unchanged(tokenizer, {"a": 0, "b": 1})
        self.assertEqual(tokenizer.calls, 1)

    def test_plan_bundle_hashes_are_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = root / "pretrain.jsonl"
            downstream = root / "downstream.jsonl"
            pretrain.write_text(json.dumps(_row("[C]", 3, 1)) + "\n", encoding="utf-8")
            downstream.write_text(json.dumps(_row("[N]", 2)) + "\n", encoding="utf-8")
            bundle = root / "bundle"
            build(
                argparse.Namespace(
                    pretrain_census=str(pretrain),
                    downstream_train_census=str(downstream),
                    output_dir=str(bundle),
                    base_vocab_size=100,
                    macro_budget=1,
                    max_anchor_id=2,
                )
            )
            manifest, plan, plan_path = _validate_plan_bundle(
                bundle, "plan.pretrain_train_only.fallback_single_suffix.json"
            )
            self.assertEqual(manifest["status"], "candidate")
            self.assertEqual(plan["boundary_mode"], "fallback_single_suffix")
            self.assertEqual(manifest["grammar_decision"]["status"], "frozen")
            self.assertTrue(plan_path.is_file())
            compatible = _validate_shared_surface_plans(bundle, manifest, plan)
            self.assertEqual(len(compatible), 2)
            self.assertEqual(
                {row["macro_policy"] for row in compatible},
                {
                    "pretrain_train_only",
                    "balanced_pretrain_plus_registered_downstream_train",
                },
            )

            with self.assertRaisesRegex(
                AnchoredCandidateTokenizerError, "not uniquely declared"
            ):
                _validate_plan_bundle(
                    bundle, "plan.pretrain_train_only.implicit_sidecar.json"
                )

            registry = bundle / "chemical_registry.jsonl"
            registry.write_text(registry.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AnchoredCandidateTokenizerError, "registry drift"):
                _validate_plan_bundle(
                    bundle, "plan.pretrain_train_only.fallback_single_suffix.json"
                )


if __name__ == "__main__":
    unittest.main()
