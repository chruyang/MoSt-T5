from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - local lightweight environment
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class FactorizedModelInitV4Test(unittest.TestCase):
    def _symbols(self):
        from most_t5_next.p2.factorized_model_init_v4 import (
            FactorizedModelInitV4Error,
            factorized_initialization_contract_v4,
            load_deterministic_factorized_model_v4,
        )

        return (
            FactorizedModelInitV4Error,
            factorized_initialization_contract_v4,
            load_deterministic_factorized_model_v4,
        )

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

    def kwargs(self):
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
            shell_fusion_mode="l0_shell_attention_l123",
        )

    def test_contract_names_l0_and_higher_shell_roles(self) -> None:
        _, factorized_initialization_contract_v4, _ = self._symbols()
        contract = factorized_initialization_contract_v4(**self.kwargs())
        self.assertEqual(contract["atom_state_input"], "e3fp_four_explicit_levels")
        self.assertIn("not_3d_evidence", contract["l0_interpretation"])
        self.assertTrue(contract["one_adapter_rng_stream"])

    def test_loader_is_deterministic_and_restores_external_rng(self) -> None:
        _, _, load_deterministic_factorized_model_v4 = self._symbols()
        calls = []

        def loader(**kwargs):
            calls.append(kwargs)
            from types import SimpleNamespace

            return SimpleNamespace(model=self._tiny_t5())

        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        first = load_deterministic_factorized_model_v4(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self.kwargs(),
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = load_deterministic_factorized_model_v4(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self.kwargs(),
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["semantic_plan_sha256"], "a" * 64)
        for key, left in first.adapter.state_dict().items():
            right = second.adapter.state_dict()[key]
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right), key)

    def test_shell_mode_is_checkpoint_bound(self) -> None:
        FactorizedModelInitV4Error, factorized_initialization_contract_v4, _ = (
            self._symbols()
        )
        values = self.kwargs()
        values["shell_fusion_mode"] = "unknown"
        with self.assertRaisesRegex(FactorizedModelInitV4Error, "shell_fusion_mode"):
            factorized_initialization_contract_v4(**values)


if __name__ == "__main__":
    unittest.main()
