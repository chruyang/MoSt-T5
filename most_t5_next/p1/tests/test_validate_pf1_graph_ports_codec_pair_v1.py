"""Tests for the complete GraphPorts v1/v2 paired corruption gate."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import build_pf1_graph_ports_v2_release_v1 as derive
from most_t5_next.p1 import validate_pf1_graph_ports_codec_pair_v1 as subject
from most_t5_next.r1.adapter import paired_record_wire_v1 as paired_wire
from most_t5_next.r1.adapter.tests.test_production_paired_identity_records_v1 import (
    _bindings,
    _build,
    _mol,
    _tokenizer,
)
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v2 as graph_v2
from most_t5_next.r1.tokenizer import production_graph_ports_codec_v1 as graph_v1


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _wire_pair():
    mol = _mol("CC(O)F")
    groups = tuple((atom_id,) for atom_id in range(mol.GetNumAtoms()))
    tokenizer = _tokenizer(mol)
    source_pair = _build(mol, groups, tokenizer=tokenizer)
    source_payload = paired_wire.encode_paired_training_record(
        source_pair,
        schedule_index=7,
        sdf_record_index=41,
    )
    token_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in graph_v2.GPORTS_V2_UNION_TOKENS
    }
    transformed = derive.transform_paired_wire_to_graphports_v2(
        source_payload,
        declared_token_ids=token_ids,
        replacement_connection_codec_sha256=_digest("graphports-v2"),
    )
    return tokenizer, source_payload, transformed.payload


def _pair():
    _tokenizer_value, source_payload, target_payload = _wire_pair()
    return (
        paired_wire.decode_paired_training_record(source_payload),
        paired_wire.decode_paired_training_record(target_payload),
        _bindings()[1],
    )


class GraphPortsCodecPairGateTest(unittest.TestCase):
    def test_all_frozen_corruption_views_keep_exact_targets(self) -> None:
        source, target, runtime = _pair()
        subject._require_record_pair(source, target)
        for epoch in subject.TRAIN_CORRUPTION_EPOCHS:
            selected, target_tokens, source_tokens, candidate_tokens = (
                subject._require_corruption_pair(
                    source.motif_record,
                    target.motif_record,
                    tokenizer_runtime=runtime,
                    seed=subject.TRAIN_CORRUPTION_SEED,
                    epoch=epoch,
                )
            )
            self.assertGreater(selected, 0)
            self.assertGreater(target_tokens, 0)
            self.assertLess(candidate_tokens, source_tokens)

    def test_identity_token_change_is_rejected(self) -> None:
        source, target, _runtime = _pair()
        motif = target.motif_record
        ids = list(motif.input_ids)
        ids[motif.identity_spans[0].start] += 1
        changed = replace(target, motif_record=replace(motif, input_ids=tuple(ids)))
        with self.assertRaisesRegex(
            subject.PF1GraphPortsCodecPairError,
            "identity token surfaces",
        ):
            subject._require_record_pair(source, changed)

    def test_connection_byte_tamper_is_rejected_even_when_length_is_unchanged(self) -> None:
        tokenizer, source_payload, target_payload = _wire_pair()
        source_document = json.loads(source_payload)
        target_document = json.loads(target_payload)
        motif = target_document["motif_training_document"]
        graph_ids = motif["token_domain"]["input_ids"]
        token_roles = motif["token_domain"]["token_role"]
        connection_index = next(
            index for index, role in enumerate(token_roles) if role == "connection"
        )
        original = graph_ids[connection_index]
        replacement = tokenizer.convert_tokens_to_ids(graph_v1.GPORTS_BYTE_TOKENS[0])
        if replacement == original:
            replacement = tokenizer.convert_tokens_to_ids(
                graph_v1.GPORTS_BYTE_TOKENS[1]
            )
        graph_ids[connection_index] = replacement
        with self.assertRaisesRegex(
            subject.PF1GraphPortsCodecPairError,
            "raw graph token decode failed|persisted cross-motif bonds",
        ):
            subject._require_raw_graph_pair(
                source_document,
                target_document,
                tokenizer=tokenizer,
            )

    def test_missing_or_short_batch_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            subject.PF1GraphPortsCodecPairError,
            "batch counts",
        ):
            tuple(subject._paired_batches(((1,),), ()))
        with self.assertRaisesRegex(
            subject.PF1GraphPortsCodecPairError,
            "batch sizes",
        ):
            tuple(subject._paired_batches(((1, 2),), ((1,),)))

    def test_source_manifest_may_keep_additional_census_counts(self) -> None:
        tokenizer = {"tokenizer_contract_sha256": _digest("tokenizer")}
        source = {
            "counts": {
                "scheduled_members": 3,
                "paired_records": 3,
                "train_members": 2,
                "dev_members": 1,
                "rejects": 0,
                "observed_selfies_symbols": 17,
            },
            "artifacts": {"union_tokenizer": tokenizer},
        }
        target = {
            "counts": {
                "scheduled_members": 3,
                "paired_records": 3,
                "train_members": 2,
                "dev_members": 1,
                "rejected_members": 0,
            },
            "artifacts": {"union_tokenizer": tokenizer},
        }
        subject._require_manifest_pair(source, target)

    def test_manifest_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            (source / "manifest.json").write_text("{}\n", encoding="utf-8")
            (target / "manifest.json").write_text("{}\n", encoding="utf-8")
            target_manifest = {
                "source_release": {"manifest_sha256": "0" * 64},
                "codec": {
                    "source_format_version": graph_v1.FORMAT_VERSION,
                    "target_format_version": graph_v2.FORMAT_VERSION,
                    "target_source_sha256": subject._sha256_file(
                        Path(graph_v2.__file__).resolve()
                    ),
                },
            }
            with self.assertRaisesRegex(
                subject.PF1GraphPortsCodecPairError,
                "provenance",
            ):
                subject._require_release_provenance(
                    source_release=source,
                    target_release=target,
                    target_manifest=target_manifest,
                )


if __name__ == "__main__":
    unittest.main()
