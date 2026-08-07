"""Tests for the PF-1 motif length and macro-budget analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import lmdb

from most_t5_next.p1 import analyze_pf1_motif_length_budget_v1 as subject


TOKENS = {
    "<bom>": 10,
    "<eom>": 11,
    subject.FALLBACK_BEGIN: 12,
    subject.FALLBACK_END: 13,
    "<GPORTS:B41>": 14,
    "<GPORTS:B42>": 15,
    "<GPORTS:B43>": 16,
    "<MOST:M:000000>": 17,
    "<MOST:M:000001>": 18,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document(
    *,
    identities: tuple[str, ...],
    atom_tokens: int,
    edge_count: int,
    tamper_digest: bool = False,
) -> bytes:
    input_ids = [TOKENS["<bom>"]]
    spans = []
    counts = []
    modes = []
    for identity in identities:
        start = len(input_ids)
        if identity == "A":
            input_ids.append(TOKENS["<MOST:M:000000>"])
            modes.append("macro")
        elif identity == "B":
            input_ids.append(TOKENS["<MOST:M:000001>"])
            modes.append("macro")
        else:
            input_ids.extend(
                (
                    TOKENS[subject.FALLBACK_BEGIN],
                    TOKENS["<GPORTS:B43>"],
                    TOKENS[subject.FALLBACK_END],
                )
            )
            modes.append("fallback")
        spans.append([start, len(input_ids)])
        counts.append(len(input_ids) - start)
    graph_tokens = 4 + 2 * edge_count
    input_ids.extend([99] * graph_tokens)
    input_ids.append(TOKENS["<eom>"])
    digests = [_sha(identity) for identity in identities]
    if tamper_digest:
        digests[0] = "0" * 64
    document = {
        "surface_summary": {
            "atom_input_token_count": atom_tokens,
            "cross_motif_connection_count": edge_count,
            "graph_token_count": graph_tokens,
            "motif_identity_modes": modes,
            "motif_identity_token_counts": counts,
            "motif_input_token_count": len(input_ids),
        },
        "motif_training_document": {
            "token_domain": {"input_ids": input_ids},
            "logical_motif_domain": {
                "identity_spans": spans,
                "exact_identity_sha256": digests,
                "cross_motif_bonds": [{} for _ in range(edge_count)],
            },
            "dimensions": {
                "atom_count": atom_tokens - 2,
                "logical_motif_count": len(identities),
            },
        },
    }
    return json.dumps(document, sort_keys=True).encode("utf-8")


def _write_release(root: Path, *, tamper_digest: bool = False) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "counts": {
                    "train_members": 2,
                    "dev_members": 1,
                    "paired_records": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    macro_rows = [
        {
            "rank": 0,
            "identity": "A",
            "identity_sha256": _sha("A"),
            "occurrence_count": 2,
            "token": "<MOST:M:000000>",
        },
        {
            "rank": 1,
            "identity": "B",
            "identity_sha256": _sha("B"),
            "occurrence_count": 2,
            "token": "<MOST:M:000001>",
        },
    ]
    (root / "macro_registry.json").write_text(
        json.dumps({"rows": macro_rows}), encoding="utf-8"
    )
    tokenizer = root / "union_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "manifest.json").write_text(
        json.dumps({"token_ids": {"declared": TOKENS}}), encoding="utf-8"
    )
    train_rows = [
        {"split": "train", "storage_key": "t1", "member_id": "train-a"},
        {"split": "train", "storage_key": "t2", "member_id": "train-b"},
    ]
    dev_rows = [
        {"split": "dev", "storage_key": "d1", "member_id": "dev-c"},
    ]
    (root / "train_membership.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
    )
    (root / "dev_membership.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dev_rows), encoding="utf-8"
    )
    environment = lmdb.open(str(root / "paired_records.lmdb"), map_size=4 << 20)
    try:
        with environment.begin(write=True) as transaction:
            transaction.put(
                b"t1",
                _document(
                    identities=("A", "A"),
                    atom_tokens=8,
                    edge_count=1,
                    tamper_digest=tamper_digest,
                ),
            )
            transaction.put(
                b"t2",
                _document(identities=("B", "B"), atom_tokens=12, edge_count=1),
            )
            transaction.put(
                b"d1",
                _document(identities=("A", "C"), atom_tokens=9, edge_count=0),
            )
    finally:
        environment.close()


class PF1MotifLengthBudgetTest(unittest.TestCase):
    def test_train_only_k_curve_and_graph_bound_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            _write_release(release)
            report = subject.analyze_release(
                release,
                k_values=(0, 1, 2),
                lmdb_module=lmdb,
            )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["observed"]["persisted_macro_k"], 2)
        self.assertEqual(
            report["graph_lower_bound"]["records_exactly_at_bound"], 3
        )
        budgets = {row["k"]: row for row in report["macro_budget"]["rows"]}
        self.assertEqual(budgets[0]["train"]["motif_input_tokens"]["mean"], 14)
        self.assertEqual(budgets[1]["train"]["motif_input_tokens"]["mean"], 12)
        self.assertEqual(budgets[2]["train"]["motif_input_tokens"]["mean"], 10)
        self.assertEqual(budgets[2]["dev"]["motif_input_tokens"]["mean"], 10)
        self.assertFalse(report["macro_budget"]["dev_used_for_ranking"])

    def test_identity_digest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            _write_release(release, tamper_digest=True)
            with self.assertRaisesRegex(
                subject.PF1MotifLengthBudgetError,
                "identity digest",
            ):
                subject.analyze_release(release, lmdb_module=lmdb)


if __name__ == "__main__":
    unittest.main()
