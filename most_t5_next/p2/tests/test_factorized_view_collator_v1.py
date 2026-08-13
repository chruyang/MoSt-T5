from __future__ import annotations

import hashlib
import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from most_t5_next.p1.bound_record import Span
    from most_t5_next.p1.production_bridge import (
        ProductionMotifRecord,
        ProductionTokenizerRuntime,
    )
    from most_t5_next.p2.factorized_view_collator_v1 import (
        collate_factorized_motif_view,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class FactorizedViewCollatorTest(unittest.TestCase):
    def setUp(self):
        contract = _digest("tokenizer-contract")
        snapshot = _digest("tokenizer-snapshot")
        self.tokenizer = ProductionTokenizerRuntime(
            tokenizer_contract_sha256=contract,
            tokenizer_snapshot_sha256=snapshot,
            vocab_size=128,
            pad_token_id=0,
            eos_token_id=1,
            sentinel_token_ids=tuple(range(127, 117, -1)),
        )
        self.record = ProductionMotifRecord(
            record_artifact_sha256=_digest("record"),
            record_id="molecule-1",
            storage_key="fixture/1",
            release_id="fixture",
            geometry_record_content_sha256=_digest("geometry"),
            tokenizer_contract_sha256=contract,
            tokenizer_snapshot_sha256=snapshot,
            input_ids=(10, 11, 12, 13, 14, 15),
            token_to_logical_motif=(0, 0, 0, 1, 1, 1),
            token_role=(
                "identity",
                "identity",
                "connection",
                "identity",
                "identity",
                "connection",
            ),
            identity_spans=(Span(0, 2), Span(3, 5)),
            connection_token_indices=((2,), (5,)),
            logical_to_carrier=(0, 3),
            exact_identity_sha256=(_digest("left"), _digest("right")),
            source_atom_count=3,
            full_e3fp_ids=((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)),
            atom_valid_mask=(True, True, True),
            model_to_source_atom_index=(0, 1, 2),
            atom_to_logical_motif=(0, 0, 1),
            atom_is_attachment=(False, True, False),
        )

    def test_grammar_hides_selected_motif_state_without_adding_state_loss(self):
        batch = collate_factorized_motif_view(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="grammar",
            seed=7,
            epoch=0,
            identity_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertIsNotNone(batch.labels)
        self.assertIsNone(batch.state_target_ids)
        self.assertTrue(bool((batch.e3fp_input_ids[0] == 17).all()))
        self.assertEqual(tuple(batch.identity_span_bounds[0, :, 1] - batch.identity_span_bounds[0, :, 0]), (1, 1))

    def test_state_keeps_identity_visible_and_masks_at_most_one_atom_per_motif(self):
        batch = collate_factorized_motif_view(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="state",
            seed=11,
            epoch=2,
            state_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertIsNone(batch.labels)
        self.assertIsNone(batch.label_to_motif)
        self.assertTrue(torch.equal(batch.input_ids[0], torch.tensor(self.record.input_ids)))
        self.assertTrue(bool(batch.state_target_mask.any()))
        self.assertTrue(
            bool((batch.state_target_mask & ~batch.state_corruption_mask).any()) is False
        )
        selected_atoms = batch.state_corruption_mask[0].any(dim=-1)
        self.assertEqual(int(selected_atoms[:2].sum()), 1)
        self.assertFalse(bool(selected_atoms[2]))
        self.assertTrue(
            bool(
                (
                    batch.e3fp_input_ids[batch.state_corruption_mask]
                    == 17
                ).all()
            )
        )
        again = collate_factorized_motif_view(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="state",
            seed=11,
            epoch=2,
            state_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertTrue(torch.equal(batch.e3fp_input_ids, again.e3fp_input_ids))

    def test_formal_collator_rejects_nonidentifiable_block_masking(self):
        with self.assertRaisesRegex(ValueError, "at most one atom row"):
            collate_factorized_motif_view(
                (self.record,),
                tokenizer=self.tokenizer,
                objective_mode="state",
                seed=1,
                epoch=0,
                state_masking_strategy="motif_block",
                num_e3fp_embeddings=16,
            )

    def test_cross_view_keeps_aligned_state_visible(self):
        batch = collate_factorized_motif_view(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="cross_view",
            seed=13,
            epoch=0,
            identity_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertIsNotNone(batch.labels)
        self.assertEqual(
            tuple(batch.label_to_motif[0].tolist()),
            (-1, 0, 0, -1, 1, 1, -1, -1),
        )
        self.assertTrue(
            torch.equal(
                batch.e3fp_input_ids[0],
                torch.tensor(self.record.full_e3fp_ids),
            )
        )

    def test_same_collator_accepts_an_aligned_2d_state_provider(self):
        class Provider:
            state_kind = "morgan_r3_4096"

            def get(self, record_id):
                self.record_id = record_id
                return ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))

        provider = Provider()
        batch = collate_factorized_motif_view(
            (self.record,),
            tokenizer=self.tokenizer,
            objective_mode="cross_view",
            seed=13,
            epoch=0,
            atom_state_provider=provider,
            num_e3fp_embeddings=16,
        )
        self.assertEqual(batch.state_kind, "morgan_r3_4096")
        self.assertEqual(provider.record_id, self.record.record_id)
        self.assertTrue(
            torch.equal(
                batch.e3fp_input_ids[0],
                torch.tensor(((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))),
            )
        )

    def test_state_provider_cannot_silently_truncate_float_ids(self):
        class Provider:
            state_kind = "bad_float_state"

            def get(self, _record_id):
                return ((1.5, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))

        with self.assertRaisesRegex(ValueError, "discrete integer IDs"):
            collate_factorized_motif_view(
                (self.record,),
                tokenizer=self.tokenizer,
                objective_mode="cross_view",
                seed=13,
                epoch=0,
                atom_state_provider=Provider(),
                num_e3fp_embeddings=16,
            )


if __name__ == "__main__":
    unittest.main()
