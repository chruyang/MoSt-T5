"""Tests for the graph-surface-only PF-1 v1-to-v2 release derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import lmdb

from most_t5_next.p1 import build_pf1_graph_ports_v2_release_v1 as subject
from most_t5_next.p1.build_pf1_paired_release_v1 import SCHEMA_VERSION
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire
from most_t5_next.r1.adapter.tests.test_production_paired_identity_records_v1 import (
    _build,
    _mol,
    _tokenizer,
)
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v2 as graph_v2


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture():
    mol = _mol("CC(O)F")
    groups = tuple((atom_id,) for atom_id in range(mol.GetNumAtoms()))
    tokenizer = _tokenizer(mol)
    pair = _build(mol, groups, tokenizer=tokenizer)
    payload = paired_wire.encode_paired_training_record(
        pair,
        schedule_index=7,
        sdf_record_index=41,
    )
    return tokenizer, pair, payload


def _token_ids(tokenizer) -> dict[str, int]:
    return {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in graph_v2.GPORTS_V2_UNION_TOKENS
    }


class BuildPF1GraphPortsV2ReleaseTest(unittest.TestCase):
    def test_one_wire_changes_only_the_graph_surface(self) -> None:
        tokenizer, pair, source_payload = _fixture()
        transformed = subject.transform_paired_wire_to_graphports_v2(
            source_payload,
            declared_token_ids=_token_ids(tokenizer),
            replacement_connection_codec_sha256=_digest("graphports-v2"),
        )
        source = paired_wire.decode_paired_training_record(source_payload)
        target = paired_wire.decode_paired_training_record(transformed.payload)

        self.assertEqual(target.atom_record, source.atom_record)
        self.assertEqual(target.receipt, source.receipt)
        self.assertEqual(
            target.motif_record.identity_spans,
            source.motif_record.identity_spans,
        )
        self.assertEqual(
            target.motif_record.exact_identity_sha256,
            source.motif_record.exact_identity_sha256,
        )
        self.assertEqual(
            target.motif_record.full_e3fp_ids,
            source.motif_record.full_e3fp_ids,
        )
        self.assertEqual(
            transformed.source_graph_token_count,
            4 + 5 * transformed.edge_count,
        )
        self.assertEqual(
            transformed.graph_token_count,
            4 + 2 * transformed.edge_count,
        )
        self.assertLess(
            transformed.motif_input_token_count,
            transformed.source_motif_input_token_count,
        )
        target_document = json.loads(transformed.payload)
        self.assertEqual(
            target_document["motif_training_document"]["bindings"][
                "connection_codec_sha256"
            ],
            _digest("graphports-v2"),
        )
        self.assertEqual(
            target_document["motif_training_document"]["logical_motif_domain"][
                "cross_motif_bonds"
            ],
            pair.motif_training_document["logical_motif_domain"][
                "cross_motif_bonds"
            ],
        )

    def test_missing_frozen_graph_token_is_rejected(self) -> None:
        tokenizer, _pair, source_payload = _fixture()
        ids = _token_ids(tokenizer)
        del ids[graph_v2.GPORTS_V2_BOUNDARY_TOKENS[-1]]
        with self.assertRaisesRegex(
            subject.PF1GraphPortsV2ReleaseError,
            "does not cover",
        ):
            subject.transform_paired_wire_to_graphports_v2(
                source_payload,
                declared_token_ids=ids,
                replacement_connection_codec_sha256=_digest("graphports-v2"),
            )

    def test_small_release_is_published_only_after_full_replay(self) -> None:
        tokenizer, pair, payload = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "derived"
            source.mkdir()
            tokenizer_root = source / "union_tokenizer"
            tokenizer_root.mkdir()
            tokenizer_manifest = {
                "tokenizer_contract_sha256": pair.atom_record.union_tokenizer_contract_sha256,
                "tokenizer_snapshot_sha256": pair.atom_record.union_tokenizer_snapshot_sha256,
                "contract": {"declared_token_ids": _token_ids(tokenizer)},
            }
            (tokenizer_root / "manifest.json").write_text(
                json.dumps(tokenizer_manifest), encoding="utf-8"
            )
            (tokenizer_root / "snapshot-marker.txt").write_text(
                "frozen-tokenizer", encoding="utf-8"
            )
            (source / "macro_registry.json").write_text("[]\n", encoding="utf-8")
            (source / "rejects.jsonl").write_text("", encoding="utf-8")

            membership = {
                "schema_version": "fixture",
                "split": "train",
                "split_index": 0,
                "selection_index": 7,
                "sdf_record_index": 41,
                "member_id": pair.atom_record.record_id,
                "storage_key": pair.atom_record.storage_key,
                "motif_input_token_count": len(pair.motif_record.input_ids),
                "wire_bytes": len(payload),
            }
            (source / "train_membership.jsonl").write_text(
                json.dumps(membership) + "\n", encoding="utf-8"
            )
            (source / "dev_membership.jsonl").write_text("", encoding="utf-8")
            source_manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "pass",
                "scope": "fixture-v1",
                "counts": {
                    "scheduled_members": 1,
                    "paired_records": 1,
                    "train_members": 1,
                    "dev_members": 0,
                    "rejected_members": 0,
                },
            }
            (source / "manifest.json").write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )
            environment = lmdb.open(
                str(source / "paired_records.lmdb"),
                subdir=True,
                map_size=16 * 1024 * 1024,
            )
            with environment.begin(write=True) as transaction:
                transaction.put(pair.atom_record.storage_key.encode("ascii"), payload)
            environment.sync(True)
            environment.close()

            manifest = subject.run(
                argparse.Namespace(
                    source_release=str(source),
                    output_dir=str(output),
                    workers=1,
                    max_pending=1,
                    lmdb_map_size_mib=512,
                    commit_every=1,
                )
            )
            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(manifest["counts"]["paired_records"], 1)
            self.assertEqual(
                manifest["replay"]["strict_derived_decode_records"], 1
            )
            self.assertTrue(manifest["paired_invariants"]["same_mask_decision"])
            self.assertTrue((output / "manifest.json").is_file())
            derived_row = json.loads(
                (output / "train_membership.jsonl").read_text(encoding="utf-8")
            )
            self.assertLess(
                derived_row["motif_input_token_count"],
                membership["motif_input_token_count"],
            )


if __name__ == "__main__":
    unittest.main()
