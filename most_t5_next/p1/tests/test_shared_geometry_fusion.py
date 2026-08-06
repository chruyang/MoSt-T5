from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from most_t5_next.p1.experiment_grid import GeometryBatchSidecar
    from most_t5_next.p1.shared_geometry_fusion import (
        GeometryFusionError,
        GeometryTensorSidecar,
        SharedE3FPCarrierFusion,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for geometry fusion tests")
class SharedGeometryFusionTest(unittest.TestCase):
    def setUp(self):
        self.module = SharedE3FPCarrierFusion(
            num_e3fp_embeddings=8,
            hidden_size=2,
        )
        with torch.no_grad():
            for level_index, embedding in enumerate(self.module.level_embeddings):
                embedding.weight.zero_()
                values = torch.arange(1, 9, dtype=torch.float32) * (level_index + 1)
                embedding.weight[1:, 0] = values
                embedding.weight[1:, 1] = -values
        self.input_embeddings = torch.zeros((1, 5, 2), dtype=torch.float32)
        self.attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)

    @staticmethod
    def geometry(ids, mask, carriers):
        return GeometryTensorSidecar(
            e3fp_ids=torch.tensor(ids, dtype=torch.long),
            e3fp_atom_mask=torch.tensor(mask, dtype=torch.bool),
            e3fp_atom_to_token=torch.tensor(carriers, dtype=torch.long),
        )

    def test_a1_size_one_carriers_equal_their_atom_states(self):
        geometry = self.geometry(
            [[[0, 2, -1, -1], [4, -1, -1, -1]]],
            [[True, True]],
            [[1, 3]],
        )
        output = self.module(
            self.input_embeddings,
            geometry,
            attention_mask=self.attention_mask,
        )

        self.assertTrue(torch.equal(output[0, 1], torch.tensor([7.0, -7.0])))
        self.assertTrue(torch.equal(output[0, 3], torch.tensor([5.0, -5.0])))
        self.assertTrue(torch.equal(output[0, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(output[0, 2], torch.zeros(2)))
        self.assertTrue(torch.equal(output[0, 4], torch.zeros(2)))

    def test_m1_is_the_invariant_mean_of_atoms_on_one_motif_carrier(self):
        geometry = self.geometry(
            [[[0, 2, -1, -1], [4, -1, -1, -1]]],
            [[True, True]],
            [[2, 2]],
        )
        output = self.module(
            self.input_embeddings,
            geometry,
            attention_mask=self.attention_mask,
        )

        self.assertTrue(torch.equal(output[0, 2], torch.tensor([6.0, -6.0])))

    def test_shell_position_is_not_exchangeable_and_levels_are_summed(self):
        ordered = self.geometry(
            [[[1, 2, -1, -1]]],
            [[True]],
            [[1]],
        )
        exchanged = self.geometry(
            [[[2, 1, -1, -1]]],
            [[True]],
            [[1]],
        )
        ordered_output = self.module(
            self.input_embeddings,
            ordered,
            attention_mask=self.attention_mask,
        )
        exchanged_output = self.module(
            self.input_embeddings,
            exchanged,
            attention_mask=self.attention_mask,
        )

        # level-0 id=1 contributes 2; level-1 id=2 contributes 6.
        self.assertTrue(torch.equal(ordered_output[0, 1], torch.tensor([8.0, -8.0])))
        # Exchanging IDs contributes 3 + 4, proving that shell identity remains.
        self.assertTrue(torch.equal(exchanged_output[0, 1], torch.tensor([7.0, -7.0])))

    def test_atom_permutation_does_not_change_carrier_states(self):
        first = self.geometry(
            [
                [
                    [0, 2, -1, -1],
                    [4, -1, -1, -1],
                    [3, 5, 7, -1],
                    [-1, -1, -1, -1],
                ]
            ],
            [[True, True, True, False]],
            [[1, 2, 1, -1]],
        )
        permuted = self.geometry(
            [
                [
                    [3, 5, 7, -1],
                    [0, 2, -1, -1],
                    [4, -1, -1, -1],
                    [-1, -1, -1, -1],
                ]
            ],
            [[True, True, True, False]],
            [[1, 1, 2, -1]],
        )

        first_output = self.module(
            self.input_embeddings,
            first,
            attention_mask=self.attention_mask,
        )
        permuted_output = self.module(
            self.input_embeddings,
            permuted,
            attention_mask=self.attention_mask,
        )
        self.assertTrue(torch.allclose(first_output, permuted_output, atol=1e-7))

    def test_padding_and_minus_one_levels_cannot_leak(self):
        geometry = self.geometry(
            [[[2, -1, -1, -1], [-1, -1, -1, -1]]],
            [[True, False]],
            [[1, -1]],
        )
        output = self.module(
            self.input_embeddings,
            geometry,
            attention_mask=self.attention_mask,
        )
        expected = torch.zeros_like(output)
        expected[0, 1] = torch.tensor([3.0, -3.0])
        self.assertTrue(torch.equal(output, expected))

    def test_gradients_reach_inputs_and_the_shared_embedding_table(self):
        inputs = torch.randn((1, 5, 2), requires_grad=True)
        geometry = self.geometry(
            [[[1, 2, -1, -1], [3, 4, -1, -1]]],
            [[True, True]],
            [[1, 1]],
        )
        loss = self.module(
            inputs,
            geometry,
            attention_mask=self.attention_mask,
        ).square().sum()
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.module.level_embeddings[0].weight.grad)
        self.assertGreater(
            float(self.module.level_embeddings[0].weight.grad[[2, 4]].abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(self.module.level_embeddings[1].weight.grad[[3, 5]].abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(self.module.level_embeddings[0].weight.grad[0].abs().sum()), 0.0
        )
        self.assertEqual(
            float(self.module.level_embeddings[2].weight.grad.abs().sum()), 0.0
        )

    def test_a1_and_m1_calls_reuse_the_identical_parameter_object(self):
        parameters = tuple(self.module.parameters())
        a1 = self.geometry(
            [[[1, -1, -1, -1], [2, -1, -1, -1]]],
            [[True, True]],
            [[1, 2]],
        )
        m1 = self.geometry(
            [[[1, -1, -1, -1], [2, -1, -1, -1]]],
            [[True, True]],
            [[1, 1]],
        )
        self.module(self.input_embeddings, a1, attention_mask=self.attention_mask)
        self.module(self.input_embeddings, m1, attention_mask=self.attention_mask)

        self.assertEqual(tuple(self.module.parameters()), parameters)
        self.assertEqual(len(parameters), 4)
        self.assertTrue(
            all(
                self.module.level_embeddings[index].weight is parameters[index]
                for index in range(4)
            )
        )
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 4 * 9 * 2)

    def test_python_contract_is_accepted_without_changing_the_chain(self):
        contract = GeometryBatchSidecar(
            record_ids=("member:0",),
            e3fp_ids=(((0, 2, -1, -1), (4, -1, -1, -1)),),
            e3fp_atom_mask=((True, True),),
            e3fp_atom_to_token=((2, 2),),
            model_to_source_atom_index=((0, 1),),
            atom_lengths=(2,),
            e3fp_level_count=4,
            token_width=5,
        )
        output = self.module(
            self.input_embeddings,
            contract,
            attention_mask=self.attention_mask,
        )
        self.assertTrue(torch.equal(output[0, 2], torch.tensor([6.0, -6.0])))

    def test_invalid_padding_and_carriers_fail_closed(self):
        leaks_padding = self.geometry(
            [[[1, -1, -1, -1], [2, -1, -1, -1]]],
            [[True, True]],
            [[1, 4]],
        )
        with self.assertRaisesRegex(GeometryFusionError, "T5 padding"):
            self.module(
                self.input_embeddings,
                leaks_padding,
                attention_mask=self.attention_mask,
            )

        invalid_padded_atom = self.geometry(
            [[[1, -1, -1, -1], [2, -1, -1, -1]]],
            [[True, False]],
            [[1, -1]],
        )
        with self.assertRaisesRegex(GeometryFusionError, "padded atoms"):
            self.module(
                self.input_embeddings,
                invalid_padded_atom,
                attention_mask=self.attention_mask,
            )

        wrong_shape = GeometryTensorSidecar(
            e3fp_ids=torch.tensor([[[1, -1, -1, -1]]], dtype=torch.long),
            e3fp_atom_mask=torch.tensor([[True, False]], dtype=torch.bool),
            e3fp_atom_to_token=torch.tensor([[1]], dtype=torch.long),
        )
        with self.assertRaisesRegex(GeometryFusionError, "mask"):
            self.module(
                self.input_embeddings,
                wrong_shape,
                attention_mask=self.attention_mask,
            )

    def test_non_four_level_input_is_rejected(self):
        geometry = self.geometry(
            [[[1, 2, -1]]],
            [[True]],
            [[1]],
        )
        with self.assertRaisesRegex(GeometryFusionError, "exactly four"):
            self.module(
                self.input_embeddings,
                geometry,
                attention_mask=self.attention_mask,
            )


if __name__ == "__main__":
    unittest.main()
