"""CPU-only tests for the one shared P1 union-vocabulary initializer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
except ModuleNotFoundError:  # pragma: no cover - dependency-free hosts
    torch = None
    T5Config = None
    T5ForConditionalGeneration = None

from most_t5_next.p1 import build_union_init_checkpoint_v1 as subject


class _MockTokenizer:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


def _verified_tokenizer(base_size: int, union_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=_MockTokenizer(union_size),
        runtime=SimpleNamespace(vocab_size=union_size),
        manifest={
            "tokenizer_contract_sha256": "contract-v1",
            "tokenizer_snapshot_sha256": "snapshot-v1",
            "counts": {
                "base_vocab_size": base_size,
                "final_vocab_size": union_size,
            },
        },
    )


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class UnionInitCheckpointTest(unittest.TestCase):
    BASE_TOKENIZER_SIZE = 10
    BASE_MODEL_VOCAB_SIZE = 12
    UNION_SIZE = 15
    HIDDEN_SIZE = 8
    SEED = 1701
    GEOMETRY_FUSION_SEED = 2903
    NUM_E3FP_EMBEDDINGS = 7

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base_tokenizer = self.root / "base-tokenizer"
        self.union_tokenizer = self.root / "union-tokenizer"
        self.base_tokenizer.mkdir()
        self.union_tokenizer.mkdir()
        self.verified = _verified_tokenizer(
            self.BASE_TOKENIZER_SIZE,
            self.UNION_SIZE,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_base(self, *, tied: bool, name: str) -> tuple[Path, object, object]:
        path = self.root / name
        config = T5Config(
            vocab_size=self.BASE_MODEL_VOCAB_SIZE,
            d_model=self.HIDDEN_SIZE,
            d_kv=4,
            d_ff=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dropout_rate=0.0,
            initializer_factor=0.5,
            tie_word_embeddings=tied,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
        model = T5ForConditionalGeneration(config)
        with torch.no_grad():
            # Make the model-only rows conspicuous so the reset cannot pass by
            # accidentally retaining their checkpoint values.
            model.get_input_embeddings().weight[
                self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE
            ].fill_(7.0)
            if not tied:
                model.get_output_embeddings().weight[
                    self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE
                ].fill_(-7.0)
        old_input = model.get_input_embeddings().weight.detach().clone()
        old_output = model.get_output_embeddings().weight.detach().clone()
        model.save_pretrained(str(path), safe_serialization=True)
        return path, old_input, old_output

    def _build(self, *, base: Path, output: Path, seed: int | None = None):
        with patch.object(subject, "_load_verified_tokenizer", return_value=self.verified):
            return subject.build_union_init_checkpoint(
                base_model_snapshot=base,
                base_tokenizer_snapshot=self.base_tokenizer,
                union_tokenizer_dir=self.union_tokenizer,
                output_dir=output,
                seed=self.SEED if seed is None else seed,
                geometry_fusion_seed=self.GEOMETRY_FUSION_SEED,
                num_e3fp_embeddings=self.NUM_E3FP_EMBEDDINGS,
            )

    def _load(self, *, base: Path, output: Path):
        with patch.object(subject, "_load_verified_tokenizer", return_value=self.verified):
            return subject.load_verified_union_init_checkpoint(
                base_model_snapshot=base,
                base_tokenizer_snapshot=self.base_tokenizer,
                union_tokenizer_dir=self.union_tokenizer,
                output_dir=output,
                geometry_fusion_seed=self.GEOMETRY_FUSION_SEED,
                num_e3fp_embeddings=self.NUM_E3FP_EMBEDDINGS,
            )

    def _load_wrapper(self, *, base: Path, output: Path, condition_id: str):
        with patch.object(subject, "_load_verified_tokenizer", return_value=self.verified):
            return subject.load_verified_four_grid_wrapper(
                condition_id=condition_id,
                base_model_snapshot=base,
                base_tokenizer_snapshot=self.base_tokenizer,
                union_tokenizer_dir=self.union_tokenizer,
                output_dir=output,
                geometry_fusion_seed=self.GEOMETRY_FUSION_SEED,
                num_e3fp_embeddings=self.NUM_E3FP_EMBEDDINGS,
            )

    def test_untied_reset_starts_at_base_tokenizer_not_model_vocab(self) -> None:
        base, old_input, old_output = self._make_base(tied=False, name="untied-base")
        output = self.root / "untied-init"
        rng_before = torch.random.get_rng_state().clone()
        built = self._build(base=base, output=output)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

        model = built.model
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        self.assertEqual(tuple(input_weight.shape), (self.UNION_SIZE, self.HIDDEN_SIZE))
        self.assertEqual(model.config.vocab_size, self.UNION_SIZE)
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertIsNot(input_weight, output_weight)
        self.assertNotEqual(input_weight.data_ptr(), output_weight.data_ptr())
        self.assertTrue(
            torch.equal(input_weight[: self.BASE_TOKENIZER_SIZE], old_input[: self.BASE_TOKENIZER_SIZE])
        )
        self.assertTrue(
            torch.equal(output_weight[: self.BASE_TOKENIZER_SIZE], old_output[: self.BASE_TOKENIZER_SIZE])
        )
        self.assertFalse(
            torch.equal(
                input_weight[self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE],
                old_input[self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE],
            )
        )
        self.assertFalse(
            torch.equal(
                output_weight[self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE],
                old_output[self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE],
            )
        )
        self.assertTrue(torch.isfinite(input_weight[self.BASE_TOKENIZER_SIZE :]).all())
        self.assertTrue(torch.isfinite(output_weight[self.BASE_TOKENIZER_SIZE :]).all())
        self.assertFalse(
            torch.equal(
                input_weight[self.BASE_TOKENIZER_SIZE :],
                output_weight[self.BASE_TOKENIZER_SIZE :],
            )
        )
        self.assertEqual(
            built.manifest["initialization"]["initialized_id_range_half_open"],
            [self.BASE_TOKENIZER_SIZE, self.UNION_SIZE],
        )
        self.assertEqual(
            built.manifest["initialization"]["reclaimed_checkpoint_id_range_half_open"],
            [self.BASE_TOKENIZER_SIZE, self.BASE_MODEL_VOCAB_SIZE],
        )
        self.assertEqual(
            built.manifest["initialization"]["newly_allocated_id_range_half_open"],
            [self.BASE_MODEL_VOCAB_SIZE, self.UNION_SIZE],
        )
        self.assertIsNone(built.manifest["resize"]["pad_to_multiple_of"])
        self.assertFalse(built.manifest["resize"]["explicit_tie_weights_call"])
        self.assertTrue(
            built.manifest["resize"]["hf_resize_api_internally_calls_tie_weights"]
        )
        self.assertTrue(
            built.manifest["resize"]["post_resize_tie_state_verified_against_config"]
        )
        self.assertEqual(
            built.manifest["four_grid_wrapper"]["geometry_fusion_seed"],
            self.GEOMETRY_FUSION_SEED,
        )
        self.assertEqual(
            built.manifest["four_grid_wrapper"]["num_e3fp_embeddings"],
            self.NUM_E3FP_EMBEDDINGS,
        )
        self.assertTrue((output / subject.MANIFEST_NAME).is_file())
        self.assertTrue((output / subject.CHECKPOINT_DIRECTORY).is_dir())
        self.assertFalse((self.root / "untied-init.staging").exists())

    def test_tied_checkpoint_stays_tied_and_initializes_shared_rows_once(self) -> None:
        base, old_input, _ = self._make_base(tied=True, name="tied-base")
        built = self._build(base=base, output=self.root / "tied-init")
        input_weight = built.model.get_input_embeddings().weight
        output_weight = built.model.get_output_embeddings().weight
        self.assertTrue(built.model.config.tie_word_embeddings)
        self.assertEqual(input_weight.data_ptr(), output_weight.data_ptr())
        self.assertTrue(
            torch.equal(input_weight[: self.BASE_TOKENIZER_SIZE], old_input[: self.BASE_TOKENIZER_SIZE])
        )
        self.assertFalse(
            torch.equal(
                input_weight[self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE],
                old_input[self.BASE_TOKENIZER_SIZE : self.BASE_MODEL_VOCAB_SIZE],
            )
        )
        self.assertIsNone(built.manifest["initialization"]["output_stream_seed"])
        self.assertTrue(built.manifest["initialization"]["input_initialized_once_when_tied"])

    def test_fixed_private_seed_is_reproducible_but_independent_by_matrix(self) -> None:
        base, _, _ = self._make_base(tied=False, name="deterministic-base")
        first = self._build(base=base, output=self.root / "init-one")
        second = self._build(base=base, output=self.root / "init-two")
        for key, value in first.model.state_dict().items():
            self.assertTrue(torch.equal(value, second.model.state_dict()[key]), key)

        third = self._build(base=base, output=self.root / "init-three", seed=self.SEED + 9)
        self.assertTrue(
            torch.equal(
                first.model.get_input_embeddings().weight[: self.BASE_TOKENIZER_SIZE],
                third.model.get_input_embeddings().weight[: self.BASE_TOKENIZER_SIZE],
            )
        )
        self.assertFalse(
            torch.equal(
                first.model.get_input_embeddings().weight[self.BASE_TOKENIZER_SIZE :],
                third.model.get_input_embeddings().weight[self.BASE_TOKENIZER_SIZE :],
            )
        )

    def test_verified_loader_rejects_geometry_wrapper_contract_drift(self) -> None:
        base, _, _ = self._make_base(tied=False, name="contract-base")
        output = self.root / "contract-init"
        self._build(base=base, output=output)
        for geometry_seed, embedding_count in (
            (self.GEOMETRY_FUSION_SEED + 1, self.NUM_E3FP_EMBEDDINGS),
            (self.GEOMETRY_FUSION_SEED, self.NUM_E3FP_EMBEDDINGS + 1),
        ):
            with self.subTest(
                geometry_fusion_seed=geometry_seed,
                num_e3fp_embeddings=embedding_count,
            ):
                with patch.object(
                    subject,
                    "_load_verified_tokenizer",
                    return_value=self.verified,
                ):
                    with self.assertRaisesRegex(
                        subject.UnionInitCheckpointError,
                        "manifest differs",
                    ):
                        subject.load_verified_union_init_checkpoint(
                            base_model_snapshot=base,
                            base_tokenizer_snapshot=self.base_tokenizer,
                            union_tokenizer_dir=self.union_tokenizer,
                            output_dir=output,
                            geometry_fusion_seed=geometry_seed,
                            num_e3fp_embeddings=embedding_count,
                        )

    def test_verified_loader_yields_four_independent_equal_raw_backbones(self) -> None:
        base, _, _ = self._make_base(tied=False, name="four-grid-base")
        output = self.root / "four-grid-init"
        self._build(base=base, output=output)
        loaded = [self._load(base=base, output=output) for _ in range(4)]
        reference = loaded[0].model.state_dict()
        for item in loaded[1:]:
            for key, value in reference.items():
                self.assertTrue(torch.equal(value, item.model.state_dict()[key]), key)
        pointers = {
            item.model.get_input_embeddings().weight.data_ptr() for item in loaded
        }
        self.assertEqual(len(pointers), 4)

        wrappers = []
        external_rng_states = []
        for external_seed, condition in zip((11, 22, 33, 44), ("A0", "A1", "M0", "M1")):
            torch.random.default_generator.manual_seed(external_seed)
            before = torch.random.get_rng_state().clone()
            wrapper = self._load_wrapper(
                base=base,
                output=output,
                condition_id=condition,
            )
            after = torch.random.get_rng_state().clone()
            self.assertTrue(torch.equal(before, after))
            external_rng_states.append(before)
            wrappers.append(wrapper)
        self.assertEqual([wrapper.condition_id for wrapper in wrappers], ["A0", "A1", "M0", "M1"])
        for wrapper in wrappers:
            self.assertEqual(wrapper.config.vocab_size, self.UNION_SIZE)
            self.assertEqual(
                wrapper.geometry_fusion.num_e3fp_embeddings,
                self.NUM_E3FP_EMBEDDINGS,
            )
        self.assertTrue(
            any(
                not torch.equal(external_rng_states[0], state)
                for state in external_rng_states[1:]
            )
        )

        complete_reference = wrappers[0].state_dict()
        for wrapper in wrappers[1:]:
            candidate = wrapper.state_dict()
            self.assertEqual(set(candidate), set(complete_reference))
            for key, value in complete_reference.items():
                self.assertTrue(torch.equal(value, candidate[key]), key)
        # Every cell owns independent storage even though every tensor value is
        # identical.  Aliases inside one T5 (shared/encoder/decoder) remain
        # intentional, so independence is checked per state key across cells.
        for key in complete_reference:
            self.assertEqual(
                len({wrapper.state_dict()[key].data_ptr() for wrapper in wrappers}),
                4,
                key,
            )


if __name__ == "__main__":
    unittest.main()
