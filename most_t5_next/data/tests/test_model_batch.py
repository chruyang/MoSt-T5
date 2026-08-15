from __future__ import annotations

import unittest

import torch

from most_t5_next.data.model_batch import disable_geometry, model_batch
from most_t5_next.interfaces import REQUIRED_GEOMETRY_INPUT_NAMES


def _batch() -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.ones((2, 3), dtype=torch.long),
    }
    for name in REQUIRED_GEOMETRY_INPUT_NAMES:
        cache_name = {
            "atom_mask": "e3fp_atom_mask",
            "token_is_connector_endpoint": "connector_endpoint_mask",
        }.get(name, name)
        batch[cache_name] = torch.zeros((2, 1), dtype=torch.long)
    return batch


class UnifiedModelBatchTest(unittest.TestCase):
    def test_unlabeled_batch_supports_generation(self):
        batch = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        }
        self.assertEqual(set(model_batch(batch)), set(batch))

    def test_text_only_batch_uses_the_same_public_function(self) -> None:
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.ones((1, 2), dtype=torch.long),
            "metadata": ("not forwarded",),
        }
        self.assertEqual(set(model_batch(batch)), {"input_ids", "attention_mask", "labels"})

    def test_aliases_and_complete_molecular_schema_are_normalized(self) -> None:
        result = model_batch(_batch())
        self.assertTrue(REQUIRED_GEOMETRY_INPUT_NAMES.issubset(result))
        self.assertNotIn("e3fp_atom_mask", result)
        self.assertNotIn("connector_endpoint_mask", result)

    def test_partial_molecular_schema_fails_closed(self) -> None:
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "labels": torch.ones((1, 2), dtype=torch.long),
            "e3fp_ids": torch.full((1, 1, 4), -1, dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, "partial molecular batch"):
            model_batch(batch)

    def test_disabling_geometry_does_not_mutate_the_source_batch(self) -> None:
        batch = model_batch(_batch())
        original = batch["e3fp_ids"].clone()
        disabled = disable_geometry(batch)
        self.assertTrue(disabled["e3fp_ids"].eq(-1).all())
        torch.testing.assert_close(batch["e3fp_ids"], original)


if __name__ == "__main__":
    unittest.main()
