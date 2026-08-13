from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class FactorizedModelInitV9Test(unittest.TestCase):
    def test_contract_restores_only_level_embedding(self):
        from most_t5_next.p2.factorized_model_init_v9 import (
            factorized_initialization_contract_v9,
        )

        contract = factorized_initialization_contract_v9(
            semantic_plan_sha256="a" * 64,
            adapter_seed=17,
            num_e3fp_embeddings=16,
            state_level2_weight=0.25,
            state_embedding_dim=4,
            atom_memory_dim=6,
            max_identity_span_length=8,
            max_atoms_per_motif=4,
            geometry_fraction=0.5,
        )
        self.assertTrue(contract["level_embedding"])
        self.assertEqual(contract["level_embedding_count"], 4)
        self.assertFalse(contract["attachment_role_is_learned_atom_input"])
        self.assertFalse(contract["presence_feature"])


if __name__ == "__main__":
    unittest.main()
