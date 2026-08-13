from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class FactorizedModelInitV7Test(unittest.TestCase):
    def test_contract_binds_the_single_linear_projection(self):
        from most_t5_next.p2.factorized_model_init_v7 import (
            factorized_initialization_contract_v7,
        )

        contract = factorized_initialization_contract_v7(
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
        self.assertEqual(contract["projection_parameter_count"], 72)
        self.assertEqual(contract["projection"], "one_bias_free_linear_2d_to_d")
        self.assertFalse(contract["level_embedding"])
        self.assertFalse(contract["attachment_role_is_learned_atom_input"])
        self.assertFalse(contract["atom_mlp"])


if __name__ == "__main__":
    unittest.main()
