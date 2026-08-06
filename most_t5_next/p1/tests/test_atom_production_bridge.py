from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from most_t5_next.p1.atom_production_bridge import (
    ATOM_IDENTITY_ROLE,
    IDENTITY_SENTINEL_ROLE,
    AtomProductionBridgeError,
    ProductionAtomSelfiesRecord,
    collate_production_atom_batch,
    collate_production_atom_record,
)
from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
    collate_production_batch,
)
from most_t5_next.p1.experiment_grid import (
    FourGridContractError,
    P1ConditionBatch,
    validate_a1_m1_geometry_atom_parity,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ProductionAtomBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = ProductionAtomSelfiesRecord(
            record_artifact_sha256=digest("atom-record"),
            record_id="pcqm4mv2:atom-fixture",
            storage_key="atom/000001",
            release_id="fixture-release",
            geometry_record_content_sha256=digest("geometry-record"),
            union_tokenizer_contract_sha256=digest("union-contract"),
            union_tokenizer_snapshot_sha256=digest("union-snapshot"),
            selfies="[C][Branch1][O][Ring1]",
            input_ids=(0, 10, 11, 20, 12, 21, 1),
            token_to_atom=(-1, 0, 0, -1, 1, -1, -1),
            token_role=(
                "boundary",
                ATOM_IDENTITY_ROLE,
                ATOM_IDENTITY_ROLE,
                "branch",
                ATOM_IDENTITY_ROLE,
                "ring",
                "boundary",
            ),
            atom_identity_spans=(Span(1, 3), Span(4, 5)),
            atom_to_carrier=(1, 4),
            source_atom_count=4,
            model_to_source_atom_index=(0, 2),
            full_e3fp_ids=((10, 11, 12, 13), (20, 21, 22, 23)),
            atom_valid_mask=(True, True),
        )
        self.tokenizer = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=digest("union-contract"),
            tokenizer_snapshot_sha256=digest("union-snapshot"),
            vocab_size=200,
            pad_token_id=199,
            eos_token_id=99,
            sentinel_token_ids=(100, 101, 102, 103),
        )

    def test_complete_identity_spans_are_corrupted_but_structure_is_preserved(self):
        example = collate_production_atom_record(
            self.record,
            tokenizer=self.tokenizer,
            seed=7,
            epoch=2,
            mask_probability=1.0,
        )
        self.assertEqual(example.input_ids, (0, 100, 20, 101, 21, 1))
        self.assertEqual(example.labels, (100, 10, 11, 101, 12, 102, 99))
        self.assertEqual(example.input_token_to_atom, (-1, 0, -1, 1, -1, -1))
        self.assertEqual(
            example.input_token_role,
            (
                "boundary",
                IDENTITY_SENTINEL_ROLE,
                "branch",
                IDENTITY_SENTINEL_ROLE,
                "ring",
                "boundary",
            ),
        )
        self.assertEqual(example.atom_identity_input_spans, (Span(1, 2), Span(3, 4)))
        self.assertEqual(example.atom_to_carrier, (1, 3))

    def test_a0_and_a1_share_exact_ce_and_only_a1_has_geometry(self):
        kwargs = dict(
            records=(self.record,),
            tokenizer=self.tokenizer,
            seed=7,
            epoch=2,
            mask_probability=1.0,
        )
        a0 = collate_production_atom_batch(condition_id="A0", **kwargs)
        a1 = collate_production_atom_batch(condition_id="A1", **kwargs)
        self.assertEqual(a0.ce_batch, a1.ce_batch)
        self.assertIsNone(a0.geometry)
        self.assertIsNotNone(a1.geometry)
        self.assertEqual(a1.geometry.e3fp_atom_to_token, ((1, 3),))
        active_carriers = a1.geometry.e3fp_atom_to_token[0][
            : a1.geometry.atom_lengths[0]
        ]
        self.assertEqual(len(active_carriers), len(set(active_carriers)))
        self.assertEqual(
            set(a1.geometry_inputs()),
            {"e3fp_ids", "e3fp_atom_mask", "e3fp_atom_to_token"},
        )

    def test_masking_is_stateless_and_epoch_bound(self):
        kwargs = dict(
            record=self.record,
            tokenizer=self.tokenizer,
            seed=37,
            mask_probability=0.5,
        )
        first = collate_production_atom_record(epoch=0, **kwargs)
        replay = collate_production_atom_record(epoch=0, **kwargs)
        next_epoch = collate_production_atom_record(epoch=1, **kwargs)
        self.assertEqual(first, replay)
        self.assertNotEqual(first.mask_decision_sha256, next_epoch.mask_decision_sha256)

    def test_shared_union_tokenizer_binding_is_mandatory(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "snapshot hash differs"):
            collate_production_atom_record(
                self.record,
                tokenizer=replace(
                    self.tokenizer,
                    tokenizer_snapshot_sha256=digest("different-union-snapshot"),
                ),
                seed=7,
                epoch=2,
            )

    def test_motif_cells_and_non_atom_records_are_rejected(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "logical-motif"):
            collate_production_atom_batch(
                (self.record,),
                condition_id="M1",
                tokenizer=self.tokenizer,
                seed=7,
                epoch=2,
            )
        with self.assertRaisesRegex(AtomProductionBridgeError, "independently validated"):
            collate_production_atom_record(
                object(),
                tokenizer=self.tokenizer,
                seed=7,
                epoch=2,
            )

    def test_record_requires_unique_explicit_atom_carriers(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "unique carrier"):
            replace(self.record, atom_to_carrier=(1, 1))
        with self.assertRaisesRegex(AtomProductionBridgeError, "another atom"):
            replace(self.record, token_to_atom=(-1, 1, 0, -1, 1, -1, -1))

    def test_structure_cannot_be_hidden_inside_an_identity_span(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "non-identity structure"):
            replace(
                self.record,
                token_role=(
                    "boundary",
                    ATOM_IDENTITY_ROLE,
                    "branch",
                    "branch",
                    ATOM_IDENTITY_ROLE,
                    "ring",
                    "boundary",
                ),
            )

    def test_geometry_is_rectangular_and_strictly_integer(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "rectangular"):
            replace(
                self.record,
                full_e3fp_ids=((10, 11, 12, 13), (20, 21)),
            )
        with self.assertRaisesRegex(AtomProductionBridgeError, "narrow domain"):
            replace(
                self.record,
                full_e3fp_ids=((10, 11, 12, 13), (20, "21", 22, 23)),
            )

    def test_record_arrays_are_deeply_immutable_and_boolean_mask_is_exact(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "immutable tuples"):
            replace(self.record, input_ids=list(self.record.input_ids))
        with self.assertRaisesRegex(AtomProductionBridgeError, "Boolean"):
            replace(self.record, atom_valid_mask=(1, 1))

    def test_post_projection_source_mapping_is_explicit_and_ordered(self):
        with self.assertRaisesRegex(AtomProductionBridgeError, "strictly increasing"):
            replace(self.record, model_to_source_atom_index=(2, 0))
        with self.assertRaisesRegex(AtomProductionBridgeError, "source range"):
            replace(self.record, model_to_source_atom_index=(0, 4))

    def test_a1_m1_geometry_parity_checks_rows_not_carriers(self):
        a1 = collate_production_atom_batch(
            (self.record,),
            condition_id="A1",
            tokenizer=self.tokenizer,
            seed=7,
            epoch=2,
            mask_probability=1.0,
        )
        motif_record = ProductionMotifRecord(
            record_artifact_sha256=digest("motif-record"),
            record_id=self.record.record_id,
            storage_key=self.record.storage_key,
            release_id=self.record.release_id,
            geometry_record_content_sha256=self.record.geometry_record_content_sha256,
            tokenizer_contract_sha256=self.record.union_tokenizer_contract_sha256,
            tokenizer_snapshot_sha256=self.record.union_tokenizer_snapshot_sha256,
            input_ids=(0, 30, 31, 40, 1),
            token_to_logical_motif=(-1, 0, 0, 0, -1),
            token_role=("boundary", "identity", "identity", "connection", "boundary"),
            identity_spans=(Span(1, 3),),
            connection_token_indices=((3,),),
            logical_to_carrier=(1,),
            exact_identity_sha256=(digest("logical-motif"),),
            source_atom_count=self.record.source_atom_count,
            full_e3fp_ids=self.record.full_e3fp_ids,
            atom_valid_mask=self.record.atom_valid_mask,
            model_to_source_atom_index=self.record.model_to_source_atom_index,
            atom_to_logical_motif=(0, 0),
        )
        m1 = collate_production_batch(
            (motif_record,),
            condition_id="M1",
            tokenizer=self.tokenizer,
            seed=7,
            epoch=2,
            mask_probability=1.0,
        )
        self.assertEqual(a1.geometry.e3fp_atom_to_token, ((1, 3),))
        self.assertEqual(m1.geometry.e3fp_atom_to_token, ((1, 1),))
        validate_a1_m1_geometry_atom_parity(a1, m1)

        different_source_subset = P1ConditionBatch(
            "M1",
            m1.ce_batch,
            replace(m1.geometry, model_to_source_atom_index=((0, 3),)),
        )
        with self.assertRaisesRegex(FourGridContractError, "source atom mapping"):
            validate_a1_m1_geometry_atom_parity(a1, different_source_subset)

        different_e3fp_rows = P1ConditionBatch(
            "M1",
            m1.ce_batch,
            replace(
                m1.geometry,
                e3fp_ids=(((10, 11, 12, 13), (20, 21, 22, 24)),),
            ),
        )
        with self.assertRaisesRegex(FourGridContractError, "E3FP atom rows"):
            validate_a1_m1_geometry_atom_parity(a1, different_e3fp_rows)


try:
    import torch

    from most_t5_next.p1.shared_geometry_fusion import SharedE3FPCarrierFusion
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional for the pure-Python bridge")
class ProductionAtomSharedFusionTests(unittest.TestCase):
    def test_a1_sidecar_runs_through_the_same_scatter_fusion(self):
        record = ProductionAtomSelfiesRecord(
            record_artifact_sha256=digest("torch-atom-record"),
            record_id="pcqm4mv2:torch-atom-fixture",
            storage_key="atom/torch",
            release_id="fixture-release",
            geometry_record_content_sha256=digest("torch-geometry"),
            union_tokenizer_contract_sha256=digest("union-contract"),
            union_tokenizer_snapshot_sha256=digest("union-snapshot"),
            selfies="[C][O]",
            input_ids=(0, 10, 11, 1),
            token_to_atom=(-1, 0, 1, -1),
            token_role=("boundary", ATOM_IDENTITY_ROLE, ATOM_IDENTITY_ROLE, "boundary"),
            atom_identity_spans=(Span(1, 2), Span(2, 3)),
            atom_to_carrier=(1, 2),
            source_atom_count=2,
            model_to_source_atom_index=(0, 1),
            full_e3fp_ids=((2, 4, -1, -1), (6, 8, -1, -1)),
            atom_valid_mask=(True, True),
        )
        tokenizer = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=digest("union-contract"),
            tokenizer_snapshot_sha256=digest("union-snapshot"),
            vocab_size=200,
            pad_token_id=199,
            eos_token_id=99,
            sentinel_token_ids=(100, 101, 102),
        )
        batch = collate_production_atom_batch(
            (record,),
            condition_id="A1",
            tokenizer=tokenizer,
            seed=1,
            epoch=0,
            mask_probability=1.0,
        )
        fusion = SharedE3FPCarrierFusion(num_e3fp_embeddings=32, hidden_size=2)
        with torch.no_grad():
            values = torch.arange(32, dtype=torch.float32)
            for embedding in fusion.level_embeddings:
                embedding.weight.zero_()
                embedding.weight[1:, 0] = values
                embedding.weight[1:, 1] = -values
        embeddings = torch.zeros((1, len(batch.ce_batch.input_ids[0]), 2))
        attention_mask = torch.tensor(batch.ce_batch.attention_mask, dtype=torch.long)
        output = fusion(embeddings, batch.geometry, attention_mask=attention_mask)
        self.assertTrue(torch.equal(output[0, 1], torch.tensor([6.0, -6.0])))
        self.assertTrue(torch.equal(output[0, 2], torch.tensor([14.0, -14.0])))
        self.assertTrue(torch.equal(output[0, 0], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
