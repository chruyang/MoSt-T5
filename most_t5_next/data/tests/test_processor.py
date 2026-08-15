from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from most_t5_next.data.processor import (
    InputPreparationError,
    MolecularInput,
    MoStT5Collator,
    MoStT5Processor,
)


class FakeTokenizer:
    eos_token_id = 1

    def __call__(self, text, *, add_special_tokens, truncation, max_length):
        del add_special_tokens, truncation
        values = [10 + index for index, _ in enumerate(text.split())] + [1]
        return {"input_ids": values[:max_length]}


def molecule() -> MolecularInput:
    return MolecularInput(
        input_ids=(20, 21, 22, 23),
        e3fp_ids=((3, 4, -1, -1), (5, 6, -1, -1)),
        atom_to_fragment=(0, 0),
        fragment_to_carrier=(20 - 20,),
        identity_span_bounds=((0, 2),),
        endpoint_to_atom=(1,),
        endpoint_to_token=(2,),
        endpoint_to_fragment=(0,),
        endpoint_is_explicit=(True,),
        token_is_connector_endpoint=(False, False, True, False),
        atom_is_attachment=(False, True),
    )


class ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = MoStT5Processor(
            FakeTokenizer(), max_input_length=8, max_target_length=4
        )
        self.collator = MoStT5Collator(pad_token_id=0)

    def test_text_only_inference_has_no_geometry_or_labels(self):
        batch = self.collator([self.processor.text("alpha beta")])
        self.assertEqual(set(batch), {"input_ids", "attention_mask"})
        self.assertEqual(batch["input_ids"].tolist(), [[10, 11, 1]])

    def test_missing_geometry_is_encoded_as_all_minus_one(self):
        example = self.processor.molecule(molecule(), use_geometry=False)
        batch = self.collator([example])
        self.assertTrue(batch["atom_mask"].all())
        self.assertTrue(batch["e3fp_ids"].eq(-1).all())

    def test_joint_input_shifts_all_token_addresses(self):
        example = self.processor.joint("alpha beta", molecule())
        self.assertEqual(example.molecule.fragment_to_carrier, (3,))
        self.assertEqual(example.molecule.identity_span_bounds, ((3, 5),))
        self.assertEqual(example.molecule.endpoint_to_token, (5,))
        self.assertEqual(len(example.input_ids), 7)

    def test_mixed_batch_keeps_text_row_geometry_empty(self):
        text = self.processor.text("alpha", target="answer")
        molecular = self.processor.molecule(molecule(), target="answer")
        batch = self.collator([text, molecular])
        self.assertFalse(batch["atom_mask"][0].any())
        self.assertTrue(batch["e3fp_ids"][0].eq(-1).all())
        self.assertTrue(batch["atom_mask"][1].all())
        self.assertIn("labels", batch)

    def test_structural_tokens_are_never_truncated(self):
        processor = MoStT5Processor(FakeTokenizer(), max_input_length=3)
        with self.assertRaisesRegex(InputPreparationError, "truncation is unsafe"):
            processor.molecule(molecule())

    def test_text_truncation_retains_eos(self):
        example = self.processor.text("a b c d e f g h i")
        self.assertEqual(len(example.input_ids), 8)
        self.assertEqual(example.input_ids[-1], 1)

    def test_cache_adapter_derives_explicit_connector_mask(self):
        record = SimpleNamespace(
            input_ids=np.asarray([2, 3, 4, 5]),
            e3fp=np.asarray([[1, -1, -1, -1], [2, -1, -1, -1]]),
            atom_to_fragment=np.asarray([0, 0]),
            fragment_carriers=np.asarray([0]),
            fragment_spans=np.asarray([[0, 2]]),
            endpoints=np.asarray([[7, 0, 0, 1, 2, 1]]),
            atom_is_attachment=np.asarray([0, 1]),
        )
        converted = MolecularInput.from_cache_record(record)
        self.assertEqual(
            converted.token_is_connector_endpoint, (False, False, True, False)
        )


if __name__ == "__main__":
    unittest.main()
