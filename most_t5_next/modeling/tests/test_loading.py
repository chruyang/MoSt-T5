from __future__ import annotations

import tempfile
import unittest

import torch
from transformers import T5Config, T5ForConditionalGeneration

from most_t5_next.modeling.loading import load_model_from_config, load_pretrained_model


class ModelLoadingTest(unittest.TestCase):
    def test_loading_applies_configured_dropout(self) -> None:
        config = T5Config(
            vocab_size=64,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dropout_rate=0.1,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            T5ForConditionalGeneration(config).save_pretrained(directory)
            before = torch.random.get_rng_state().clone()
            model = load_pretrained_model(
                directory,
                adapter_seed=17,
                expected_vocab_size=64,
                fp_bits=16,
                atom_embedding_dim=32,
                dropout_rate=0.25,
            )
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(model.config.dropout_rate, 0.25)
        self.assertTrue(
            all(
                module.p == 0.25
                for module in model.modules()
                if isinstance(module, torch.nn.Dropout)
            )
        )

    def test_loading_rejects_invalid_dropout(self) -> None:
        config = T5Config(
            vocab_size=16,
            d_model=8,
            d_ff=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            T5ForConditionalGeneration(config).save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "dropout_rate"):
                load_pretrained_model(
                    directory,
                    expected_vocab_size=16,
                    fp_bits=8,
                    atom_embedding_dim=8,
                    dropout_rate=1.0,
                )

    def test_public_config_values_reach_the_model(self) -> None:
        backbone_config = T5Config(
            vocab_size=16,
            d_model=8,
            d_ff=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
        )
        public_config = {
            "seed": 7,
            "model": {
                "vocabulary_size": 16,
                "e3fp_bits": 8,
                "e3fp_embedding_dim": 8,
                "geometry_fraction": 0.25,
                "dropout_rate": 0.2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            T5ForConditionalGeneration(backbone_config).save_pretrained(directory)
            model = load_model_from_config(directory, public_config)
        self.assertEqual(model.config.dropout_rate, 0.2)
        self.assertEqual(model.geometry.geometry_fraction, 0.25)
        self.assertEqual(model.geometry.e3fp.fp_bits, 8)

    def test_loading_rejects_a_different_tokenizer_size(self) -> None:
        config = T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            T5ForConditionalGeneration(config).save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                load_pretrained_model(
                    directory,
                    adapter_seed=1,
                    expected_vocab_size=33,
                    fp_bits=8,
                    atom_embedding_dim=16,
                )


if __name__ == "__main__":
    unittest.main()
