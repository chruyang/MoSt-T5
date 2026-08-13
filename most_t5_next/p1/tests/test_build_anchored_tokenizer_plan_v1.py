from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1.build_anchored_tokenizer_plan_v1 import (
    AnchoredTokenizerPlanError,
    build,
)
from most_t5_next.r1.tokenizer.stereo_free_motif_chemical_lexer_v1 import (
    opaque_chemical_token_map,
)


def _census_row(pure: str, train: int, dev: int | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "pure_motif": pure,
        "pure_motif_sha256": hashlib.sha256(pure.encode("utf-8")).hexdigest(),
        "train_occurrences": train,
    }
    if dev is not None:
        row["dev_occurrences"] = dev
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class BuildAnchoredTokenizerPlanV1Test(unittest.TestCase):
    def test_builds_two_macro_policy_plans_with_one_frozen_grammar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = root / "pretrain.jsonl"
            downstream = root / "downstream.jsonl"
            _write_jsonl(
                pretrain,
                [
                    _census_row("[C]", 100, 10),
                    _census_row("[N]", 20, 2),
                    _census_row("[O]", 10, 1),
                ],
            )
            _write_jsonl(
                downstream,
                [
                    _census_row("[C]", 1),
                    _census_row("[P]", 100),
                ],
            )
            output = root / "out"
            manifest = build(
                argparse.Namespace(
                    pretrain_census=str(pretrain),
                    downstream_train_census=str(downstream),
                    output_dir=str(output),
                    base_vocab_size=100,
                    macro_budget=2,
                    max_anchor_id=3,
                )
            )
            self.assertEqual(len(manifest["plans"]), 2)
            self.assertFalse(manifest["contracts"]["tokenizer_snapshot_created"])
            self.assertTrue(manifest["contracts"]["double_boundary_candidate_eliminated"])
            self.assertEqual(manifest["grammar_decision"]["status"], "frozen")
            self.assertEqual(
                manifest["grammar_decision"]["boundary_mode"], "fallback_single_suffix"
            )

            explicit = json.loads(
                (output / "plan.pretrain_train_only.fallback_single_suffix.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(
                (output / "plan.pretrain_train_only.implicit_sidecar.json").exists()
            )
            self.assertEqual(
                explicit["declared_added_tokens"][0]["surface_token"], "<MOST:FALLBACK_END>"
            )
            self.assertEqual(
                explicit["grammar_contract"], manifest["grammar_decision"]["contract"]
            )
            surfaces = [
                row["surface_token"] for row in explicit["declared_added_tokens"]
            ]
            self.assertNotIn(".", surfaces)
            self.assertTrue(any(token.startswith("<MOST:CHEM:") for token in surfaces))
            self.assertTrue(any(token == "<3*>" for token in surfaces))

            balanced_lines = (
                output
                / "macro_registry.balanced_pretrain_plus_registered_downstream_train.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            balanced = [json.loads(line) for line in balanced_lines]
            self.assertEqual(balanced[0]["pure_motif"], "[P]")
            self.assertEqual(len(opaque_chemical_token_map()), len(set(opaque_chemical_token_map())))

    def test_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = root / "pretrain.jsonl"
            downstream = root / "downstream.jsonl"
            _write_jsonl(pretrain, [_census_row("[C]", 1, 0)])
            _write_jsonl(downstream, [_census_row("[C]", 1)])
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(AnchoredTokenizerPlanError, "must be absent"):
                build(
                    argparse.Namespace(
                        pretrain_census=str(pretrain),
                        downstream_train_census=str(downstream),
                        output_dir=str(output),
                        base_vocab_size=100,
                        macro_budget=1,
                        max_anchor_id=1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
