from __future__ import annotations

from dataclasses import replace
import unittest

from most_t5_next.p1 import GeometryBatchSidecar, P1ConditionBatch, PaddedCEBatch
from most_t5_next.p1.run_t5_ce_smoke import (
    REPORT_SCHEMA,
    _build_builtin_batch,
    _functional_config_payload,
)
from most_t5_next.p1.training_adapter import (
    FOUR_GRID_MODEL_INPUT_KEYS,
    GEOMETRY_MODEL_INPUT_KEYS,
    MODEL_INPUT_KEYS,
    TrainingAdapterError,
    select_four_grid_forward_inputs,
    select_t5_forward_inputs,
    to_four_grid_batch_encoding,
    to_t5_batch_encoding,
)


class FakeTensor:
    def __init__(self, data, *, dtype, device):
        def freeze(value):
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value

        def shape(value):
            dimensions = []
            current = value
            while isinstance(current, tuple):
                dimensions.append(len(current))
                current = current[0] if current else ()
            return tuple(dimensions)

        self.data = freeze(data)
        self.dtype = dtype
        self.device = device
        self.shape = shape(self.data)


class FakeTorch:
    long = "torch.int64"
    bool = "torch.bool"

    @staticmethod
    def as_tensor(data, *, dtype, device):
        return FakeTensor(data, dtype=dtype, device=device)


class FakeBatchEncoding(dict):
    pass


class FakeConfig:
    def __init__(self, source):
        self.source = source

    def to_dict(self):
        return {
            "_name_or_path": self.source,
            "_commit_hash": "runtime-only",
            "torch_dtype": None,
            "d_model": 768,
        }


class FakeParameter:
    dtype = "torch.float32"


class FakeModel:
    def __init__(self, source):
        self.config = FakeConfig(source)

    def parameters(self):
        return iter((FakeParameter(),))


class TrainingAdapterTest(unittest.TestCase):
    def setUp(self):
        self.batch = PaddedCEBatch(
            record_ids=("member:0", "member:1"),
            input_ids=((4, 5, 1), (6, 1, 0)),
            attention_mask=((True, True, True), (True, True, False)),
            labels=((7, 8, 1), (9, 1, -100)),
            input_lengths=(3, 2),
            target_lengths=(3, 2),
        )

    def test_long_tensor_shapes_label_padding_and_allowlist(self):
        encoded = to_t5_batch_encoding(
            self.batch,
            device="fake:0",
            torch_module=FakeTorch,
            batch_encoding_cls=FakeBatchEncoding,
        )
        self.assertEqual(tuple(encoded), MODEL_INPUT_KEYS)
        for tensor in encoded.values():
            self.assertEqual(tensor.dtype, FakeTorch.long)
            self.assertEqual(tensor.shape, (2, 3))
            self.assertEqual(tensor.device, "fake:0")
        self.assertEqual(encoded["labels"].data[1][-1], -100)

        encoded["audit_sha256"] = "not-a-model-input"
        encoded["record_ids"] = self.batch.record_ids
        forward = select_t5_forward_inputs(encoded)
        self.assertEqual(tuple(forward), MODEL_INPUT_KEYS)
        self.assertNotIn("audit_sha256", forward)
        self.assertNotIn("record_ids", forward)

    def test_rejects_non_minus_100_label_padding(self):
        invalid = replace(self.batch, labels=((7, 8, 1), (9, 1, 0)))
        with self.assertRaisesRegex(TrainingAdapterError, "label padding"):
            to_t5_batch_encoding(
                invalid,
                torch_module=FakeTorch,
                batch_encoding_cls=FakeBatchEncoding,
            )

    def test_builtin_batch_is_deterministic_and_adapter_compatible(self):
        first = _build_builtin_batch(vocab_size=32, pad_token_id=0, eos_token_id=1)
        second = _build_builtin_batch(vocab_size=32, pad_token_id=0, eos_token_id=1)
        self.assertEqual(first, second)
        encoded = to_t5_batch_encoding(
            first,
            torch_module=FakeTorch,
            batch_encoding_cls=FakeBatchEncoding,
        )
        self.assertEqual(encoded["labels"].data[1][-1], -100)

    def test_functional_config_ignores_load_address_and_binds_effective_dtype(self):
        first = _functional_config_payload(FakeModel("/snapshot/original"))
        second = _functional_config_payload(FakeModel("/checkpoint/reloaded"))
        self.assertEqual(first, second)
        self.assertEqual(first["torch_dtype"], "float32")
        self.assertNotIn("_name_or_path", first)
        self.assertNotIn("_commit_hash", first)
        self.assertTrue(REPORT_SCHEMA.endswith("/v2"))

    def test_geometry_condition_adds_only_wrapper_tensor_fields(self):
        geometry = GeometryBatchSidecar(
            record_ids=self.batch.record_ids,
            e3fp_ids=(
                ((3, -1, -1, -1), (4, 5, -1, -1)),
                ((6, -1, -1, -1), (-1, -1, -1, -1)),
            ),
            e3fp_atom_mask=((True, True), (True, False)),
            e3fp_atom_to_token=((0, 1), (0, -1)),
            model_to_source_atom_index=((0, 1), (0, -1)),
            atom_lengths=(2, 1),
            e3fp_level_count=4,
            token_width=3,
        )
        batch = P1ConditionBatch("M1", self.batch, geometry)
        encoded = to_four_grid_batch_encoding(
            batch,
            device="fake:0",
            torch_module=FakeTorch,
            batch_encoding_cls=FakeBatchEncoding,
        )
        self.assertEqual(tuple(encoded), FOUR_GRID_MODEL_INPUT_KEYS)
        self.assertEqual(encoded["e3fp_ids"].shape, (2, 2, 4))
        self.assertEqual(encoded["e3fp_ids"].dtype, FakeTorch.long)
        self.assertEqual(encoded["e3fp_atom_mask"].dtype, FakeTorch.bool)
        self.assertEqual(encoded["e3fp_atom_to_token"].dtype, FakeTorch.long)
        self.assertNotIn("model_to_source_atom_index", encoded)
        self.assertNotIn("record_ids", encoded)
        forward = select_four_grid_forward_inputs(encoded)
        self.assertEqual(tuple(forward), FOUR_GRID_MODEL_INPUT_KEYS)

    def test_ce_only_condition_and_partial_geometry_allowlist(self):
        encoded = to_four_grid_batch_encoding(
            P1ConditionBatch("M0", self.batch),
            torch_module=FakeTorch,
            batch_encoding_cls=FakeBatchEncoding,
        )
        self.assertEqual(tuple(encoded), MODEL_INPUT_KEYS)
        self.assertEqual(tuple(select_four_grid_forward_inputs(encoded)), MODEL_INPUT_KEYS)
        encoded[GEOMETRY_MODEL_INPUT_KEYS[0]] = object()
        with self.assertRaisesRegex(TrainingAdapterError, "all-or-none"):
            select_four_grid_forward_inputs(encoded)


if __name__ == "__main__":
    unittest.main()
