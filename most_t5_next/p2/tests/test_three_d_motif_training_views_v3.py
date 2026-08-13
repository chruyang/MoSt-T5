from __future__ import annotations

import unittest

from most_t5_next.p2.tests.test_factorized_view_collator_v3 import (
    FactorizedViewCollatorV3Test,
    _AddressProvider,
)
from most_t5_next.p2.three_d_motif_training_views_v3 import (
    collate_3d_motif_training_view_v3,
)


class ThreeDMotifTrainingViewsV3Test(FactorizedViewCollatorV3Test):
    def test_m_only_disables_geometry_without_changing_the_ce_contract(self) -> None:
        batch = collate_3d_motif_training_view_v3(
            (self.record,),
            view_id="m_only",
            tokenizer=self.tokenizer,
            seed=5,
            epoch=0,
            atom_address_provider=_AddressProvider(),
            identity_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertEqual(batch.objective_mode, "grammar")
        self.assertEqual(batch.model_inputs()["state_memory_mode"], "zero")

    def test_m_plus_g_keeps_aligned_geometry(self) -> None:
        batch = collate_3d_motif_training_view_v3(
            (self.record,),
            view_id="m_plus_g",
            tokenizer=self.tokenizer,
            seed=5,
            epoch=0,
            atom_address_provider=_AddressProvider(),
            identity_mask_probability=1.0,
            num_e3fp_embeddings=16,
        )
        self.assertEqual(batch.objective_mode, "cross_view")
        self.assertEqual(batch.model_inputs()["state_memory_mode"], "aligned")

    def test_g_only_masks_every_motif_identity_but_keeps_graph_endpoints(self) -> None:
        batch = collate_3d_motif_training_view_v3(
            (self.record,),
            view_id="g_only",
            tokenizer=self.tokenizer,
            seed=5,
            epoch=0,
            atom_address_provider=_AddressProvider(),
            num_e3fp_embeddings=16,
        )
        self.assertEqual(tuple(batch.input_ids.shape), (1, 4))
        self.assertEqual(int((batch.endpoint_token_to_atom >= 0).sum()), 2)
        for start, stop in batch.identity_span_bounds[0].tolist():
            self.assertEqual(stop - start, 1)


if __name__ == "__main__":
    unittest.main()
