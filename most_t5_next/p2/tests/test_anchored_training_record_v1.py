from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    collate_production_motif_record,
)
from most_t5_next.p2.anchored_training_record_v1 import (
    AnchoredTrainingRecordReader,
    AnchoredTokenizerBinding,
    AnchoredTrainingRecordError,
    _sha256_json,
    bind_anchored_training_record,
    tokenizer_binding_from_candidate_manifest,
)
from most_t5_next.r1.tokenizer.anchored_motif_model_surface_v1 import (
    encode_frozen_phrases,
)


class AnchoredTrainingRecordV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.phrases = [
            {
                "logical_motif_index": 0,
                "pure_motif": "[C]",
                "motif_atom_indices": [0],
                "anchors": [
                    {
                        "anchor_id": 0,
                        "slot_ordinal": 0,
                        "model_atom_index": 0,
                        "source_atom_index": 0,
                    }
                ],
            },
            {
                "logical_motif_index": 1,
                "pure_motif": "[O]",
                "motif_atom_indices": [1],
                "anchors": [
                    {
                        "anchor_id": 0,
                        "slot_ordinal": 0,
                        "model_atom_index": 1,
                        "source_atom_index": 1,
                    }
                ],
            },
        ]
        self.macros = [
            {"pure_motif": "[C]", "surface_token": "<MOST:MACRO:000000>"}
        ]
        encoded = encode_frozen_phrases(self.phrases, self.macros)
        token_rows = tuple((token, 100 + index) for index, token in enumerate(dict.fromkeys(encoded.tokens)))
        self.binding = AnchoredTokenizerBinding(
            tokenizer_contract_sha256="1" * 64,
            tokenizer_snapshot_sha256="2" * 64,
            vocab_size=1000,
            token_id_rows=token_rows,
        )
        surface = {
            "schema_version": "most-t5-next/stereo-free-anchored-motif-surface/v1",
            "member_id": "member-1",
            "model_atom_count": 2,
            "source_atom_count": 2,
            "model_to_source_atom_index": [0, 1],
            "atom_is_attachment": [True, True],
            "phrases": self.phrases,
            "component_motif_ranges": [[0, 2]],
            "cross_motif_bonds": [[0, 0, 0, 0, 1, 0, 1]],
            "geometry_or_e3fp_recomputed": False,
            "graphports_exposed_to_model": False,
        }
        surface["artifact_sha256"] = _sha256_json(surface)
        self.surface_record = {"storage_key": "0001", "surface": surface}
        self.geometry = ProductionMotifRecord(
            record_artifact_sha256="3" * 64,
            record_id="member-1",
            storage_key="0001",
            release_id="old-graphports-release",
            geometry_record_content_sha256="4" * 64,
            tokenizer_contract_sha256="5" * 64,
            tokenizer_snapshot_sha256="6" * 64,
            input_ids=(9, 9),
            token_to_logical_motif=(0, 1),
            token_role=("identity", "identity"),
            identity_spans=(Span(0, 1), Span(1, 2)),
            connection_token_indices=((), ()),
            logical_to_carrier=(0, 1),
            exact_identity_sha256=("7" * 64, "8" * 64),
            source_atom_count=2,
            full_e3fp_ids=((1, 2, 3, 4), (5, 6, 7, 8)),
            atom_valid_mask=(True, True),
            model_to_source_atom_index=(0, 1),
            atom_to_logical_motif=(0, 1),
            atom_is_attachment=(True, True),
        )

    def test_binds_macro_and_fallback_to_old_carrier_compatible_axis(self) -> None:
        record = bind_anchored_training_record(
            self.surface_record,
            self.geometry,
            macro_rows=self.macros,
            tokenizer=self.binding,
            release_id="anchored-release",
        )
        self.assertEqual(record.macro_used, (True, False))
        self.assertEqual(record.logical_to_carrier[0], record.identity_spans[0].stop - 1)
        self.assertEqual(record.logical_to_carrier[1], record.identity_spans[1].stop - 1)
        self.assertEqual(
            tuple(value for value in record.anchor_token_to_atom if value >= 0),
            (0, 1),
        )
        self.assertEqual(record.full_e3fp_ids, self.geometry.full_e3fp_ids)
        self.assertNotEqual(record.input_ids, self.geometry.input_ids)
        factorized = record.as_factorized_record()
        self.assertEqual(factorized.connection_token_to_atom, record.anchor_token_to_atom)
        self.assertEqual(factorized.connection_token_indices, record.anchor_token_indices)

    def test_identity_axis_mismatch_is_rejected(self) -> None:
        broken = dict(self.surface_record)
        broken_surface = dict(self.surface_record["surface"])
        broken_surface["member_id"] = "different"
        broken_surface["artifact_sha256"] = _sha256_json(
            {key: value for key, value in broken_surface.items() if key != "artifact_sha256"}
        )
        broken["surface"] = broken_surface
        with self.assertRaisesRegex(AnchoredTrainingRecordError, "identity axes differ"):
            bind_anchored_training_record(
                broken,
                self.geometry,
                macro_rows=self.macros,
                tokenizer=self.binding,
                release_id="anchored-release",
            )

    def test_corruption_uses_suffix_carrier_until_identity_is_masked(self) -> None:
        record = bind_anchored_training_record(
            self.surface_record,
            self.geometry,
            macro_rows=self.macros,
            tokenizer=self.binding,
            release_id="anchored-release",
        ).as_factorized_record()
        runtime = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=self.binding.tokenizer_contract_sha256,
            tokenizer_snapshot_sha256=self.binding.tokenizer_snapshot_sha256,
            vocab_size=self.binding.vocab_size,
            pad_token_id=0,
            eos_token_id=1,
            sentinel_token_ids=(900, 901, 902),
        )
        example = None
        for seed in range(32):
            candidate = collate_production_motif_record(
                record,
                tokenizer=runtime,
                seed=seed,
                epoch=0,
                mask_probability=0.01,
            )
            if sum(candidate.identity_recovery_mask) == 1:
                example = candidate
                break
        self.assertIsNotNone(example)
        assert example is not None
        for motif_id, selected in enumerate(example.identity_recovery_mask):
            carrier = example.logical_to_carrier[motif_id]
            span = example.identity_input_spans[motif_id]
            if selected:
                self.assertEqual(carrier, span.start)
                self.assertEqual(example.input_token_role[carrier], "identity_sentinel")
            else:
                self.assertEqual(carrier, span.stop - 1)
                self.assertEqual(example.input_token_role[carrier], "identity")
        self.assertEqual(
            tuple(value for row in example.connection_input_indices for value in row),
            tuple(value for row in record.connection_token_indices for value in row),
        )

    def test_candidate_manifest_binding_keeps_surface_and_semantics_separate(self) -> None:
        manifest = {
            "schema_version": "most-t5-next/anchored-candidate-tokenizer/v2",
            "status": "candidate",
            "plan": {
                "boundary_mode": "fallback_single_suffix",
                "final_vocab_size": 1000,
            },
            "plan_file": {"plan_sha256": "1" * 64},
            "snapshot": {"tree_sha256": "2" * 64},
            "token_ids": {"declared": dict(self.binding.token_id_rows)},
            "contracts": {"frozen_grammar_bound": True},
        }
        binding = tokenizer_binding_from_candidate_manifest(manifest)
        self.assertEqual(binding.token_id_rows, self.binding.token_id_rows)

    def test_reader_preserves_split_order_for_serial_and_parallel_decode(self) -> None:
        train_surface = dict(self.surface_record)
        train_surface.update(selection_index=0, split="train")
        dev_surface = dict(self.surface_record)
        dev_document = dict(self.surface_record["surface"])
        dev_document["member_id"] = "member-2"
        dev_document["artifact_sha256"] = _sha256_json(
            {key: value for key, value in dev_document.items() if key != "artifact_sha256"}
        )
        dev_surface.update(
            selection_index=1,
            split="dev",
            storage_key="0002",
            surface=dev_document,
        )
        train_loaded = SimpleNamespace(motif_record=self.geometry)
        dev_loaded = SimpleNamespace(
            motif_record=replace(
                self.geometry,
                record_id="member-2",
                storage_key="0002",
            )
        )

        class GeometryReader:
            train_member_count = 1
            dev_member_count = 1

            def iter_train_epoch(inner_self, *, epoch, batch_size):
                yield (train_loaded,)

            def iter_dev(inner_self, *, batch_size):
                yield (dev_loaded,)

            def iter_strict_parallel_split(
                inner_self, *, split, max_rows, workers, max_pending
            ):
                yield train_loaded if split == "train" else dev_loaded

            def iter_donor_atom_maps(inner_self, **kwargs):
                yield {"storage_key": "unused"}

        manifest = {
            "schema_version": "most-t5-next/anchored-candidate-tokenizer/v2",
            "status": "candidate",
            "plan": {
                "boundary_mode": "fallback_single_suffix",
                "final_vocab_size": 1000,
            },
            "plan_file": {"plan_sha256": "1" * 64},
            "snapshot": {"tree_sha256": "2" * 64},
            "token_ids": {"declared": dict(self.binding.token_id_rows)},
            "contracts": {"frozen_grammar_bound": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            surfaces = root / "surfaces.jsonl"
            surfaces.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in (train_surface, dev_surface)
                ),
                encoding="utf-8",
            )
            macros = root / "macros.jsonl"
            macros.write_text(json.dumps(self.macros[0]) + "\n", encoding="utf-8")
            tokenizer = root / "manifest.json"
            tokenizer.write_text(json.dumps(manifest), encoding="utf-8")
            reader = AnchoredTrainingRecordReader(
                surface_records=surfaces,
                geometry_reader=GeometryReader(),
                macro_registry=macros,
                tokenizer_manifest=tokenizer,
                release_id="anchored-release",
            )
            serial = next(reader.iter_train_epoch(epoch=4, batch_size=1))[0]
            parallel = next(
                reader.iter_strict_parallel_split(
                    split="dev", max_rows=1, workers=2, max_pending=2
                )
            )
        self.assertEqual(serial.selection_index, 0)
        self.assertEqual(serial.motif_record.storage_key, "0001")
        self.assertEqual(parallel.selection_index, 1)
        self.assertEqual(parallel.motif_record.storage_key, "0002")


if __name__ == "__main__":
    unittest.main()
