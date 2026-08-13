from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is optional in the local CPU fixture")
class FactorizedModelInitV6Test(unittest.TestCase):
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
            shell_reducer_mode="adaptive_l0_high",
        )

    def test_contract_records_one_nested_adaptive_parameter(self) -> None:
        from most_t5_next.p2.factorized_model_init_v6 import (
            factorized_initialization_contract_v6,
        )

        contract = factorized_initialization_contract_v6(**self._kwargs())
        self.assertEqual(contract["initial_l0_weight"], 0.25)
        self.assertEqual(contract["learned_shell_parameters"], 1)
        self.assertEqual(contract["shell_reducer_mode"], "adaptive_l0_high")
        fixed = dict(self._kwargs(), shell_reducer_mode="fixed_four_mean")
        self.assertEqual(
            factorized_initialization_contract_v6(**fixed)["learned_shell_parameters"],
            0,
        )

    def test_loader_is_deterministic_and_restores_rng(self) -> None:
        from most_t5_next.p2.factorized_model_init_v6 import (
            load_deterministic_factorized_model_v6,
        )
        from types import SimpleNamespace

        def loader(**_kwargs):
            return SimpleNamespace(model=self._tiny_t5())

        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        first = load_deterministic_factorized_model_v6(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **self._kwargs(),
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = load_deterministic_factorized_model_v6(
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

    def test_v5_and_v6_share_bitwise_common_initial_state(self) -> None:
        from most_t5_next.p2.factorized_model_init_v5 import (
            load_deterministic_factorized_model_v5,
        )
        from most_t5_next.p2.factorized_model_init_v6 import (
            load_deterministic_factorized_model_v6,
        )
        from types import SimpleNamespace

        def loader(**_kwargs):
            return SimpleNamespace(model=self._tiny_t5())

        common = dict(self._kwargs())
        common.pop("shell_reducer_mode")
        fixed = load_deterministic_factorized_model_v5(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            **common,
        )
        adaptive = load_deterministic_factorized_model_v6(
            base_model_snapshot="base",
            base_tokenizer_snapshot="tokenizer",
            anchored_tokenizer_dir="anchored",
            union_init_dir="init",
            union_geometry_fusion_seed=3,
            union_loader=loader,
            shell_reducer_mode="adaptive_l0_high",
            **common,
        )
        fixed_state = fixed.adapter.state_dict()
        adaptive_state = adaptive.adapter.state_dict()
        common_keys = sorted(set(fixed_state).intersection(adaptive_state))
        self.assertTrue(common_keys)
        for key in common_keys:
            left = fixed_state[key]
            right = adaptive_state[key]
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right), key)


if __name__ == "__main__":
    unittest.main()
