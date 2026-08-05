from __future__ import annotations

from dataclasses import replace
import unittest

from most_t5_next.p1 import PaddedCEBatch
from most_t5_next.p1.run_t5_ce_smoke import (
    REPORT_SCHEMA,
    _build_builtin_batch,
    _functional_config_payload,
)
from most_t5_next.p1.training_adapter import (
    MODEL_INPUT_KEYS,
    TrainingAdapterError,
    select_t5_forward_inputs,
    to_t5_batch_encoding,
)


class FakeTensor:
    def __init__(self, data, *, dtype, device):
        self.data = tuple(tuple(row) for row in data)
        self.dtype = dtype
        self.device = device
        self.shape = (len(self.data), len(self.data[0]))


class FakeTorch:
    long = "torch.int64"

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


if __name__ == "__main__":
    unittest.main()
