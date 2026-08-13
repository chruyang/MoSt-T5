from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class FactorizedModelInitV5Test(unittest.TestCase):
    def _tiny_t5(self):
        from types import SimpleNamespace

        class TinyT5(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(vocab_size=20, d_model=8)
                self.shared = nn.Embedding(20, 8)
                self.encoder = nn.Identity()

            def get_input_embeddings(self):
                return self.shared

        return TinyT5()

    def _kwargs(self):
        return dict(
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

    def test_contract_records_the_minimal_reference_semantics(self) -> None:
        from most_t5_next.p2.factorized_model_init_v5 import (
            factorized_initialization_contract_v5,
        )

        contract = factorized_initialization_contract_v5(**self._kwargs())
        self.assertEqual(contract["shell_reduction"], "arithmetic_mean_fixed_denominator_4")
        self.assertEqual(contract["missing_shell_contribution"], "zero")
        self.assertFalse(contract["level_embedding"])
        self.assertFalse(contract["attachment_role_is_learned_atom_input"])
        self.assertFalse(contract["atom_mlp"])

    def test_loader_is_deterministic_and_restores_rng(self) -> None:
        from most_t5_next.p2.factorized_model_init_v5 import (
            load_deterministic_factorized_model_v5,
        )
        from types import SimpleNamespace

        def loader(**_kwargs):
            return SimpleNamespace(model=self._tiny_t5())

        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        first = load_deterministic_factorized_model_v5(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self._kwargs(),
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = load_deterministic_factorized_model_v5(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self._kwargs(),
        )
        for key, left in first.adapter.state_dict().items():
            right = second.adapter.state_dict()[key]
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right), key)


if __name__ == "__main__":
    unittest.main()
