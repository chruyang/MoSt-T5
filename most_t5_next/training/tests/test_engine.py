from __future__ import annotations

import unittest

import torch

from most_t5_next.interfaces import GEOMETRY_INPUT_NAMES
from most_t5_next.training.engine import forward_task


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


def _batch(*, molecular: bool) -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.ones((1, 2), dtype=torch.long),
    }
    if molecular:
        for name in GEOMETRY_INPUT_NAMES:
            if name not in {"fragment_geometry_mask", "endpoint_geometry_mask"}:
                cache_name = {
                    "atom_mask": "e3fp_atom_mask",
                    "token_is_connector_endpoint": "connector_endpoint_mask",
                }.get(name, name)
                batch[cache_name] = torch.zeros((1, 1), dtype=torch.long)
        batch["records"] = ("metadata must not reach the model",)
    return batch


class ForwardTaskTest(unittest.TestCase):
    def test_task_names_do_not_control_the_public_model_route(self) -> None:
        model = _RecordingModel()
        for task in ("M", "MG", "SYN", "TXT", "CAP", "T2M"):
            result = forward_task(
                model, task, _batch(molecular=task in {"M", "MG", "SYN", "CAP"})
            )
            self.assertNotIn("geometry_mode", result)

    def test_m_uses_the_same_schema_with_an_all_minus_one_e3fp_payload(self) -> None:
        molecular = _batch(molecular=True)
        m_result = forward_task(_RecordingModel(), "M", molecular)
        mg_result = forward_task(_RecordingModel(), "MG", molecular)
        self.assertTrue(m_result["e3fp_ids"].eq(-1).all())
        self.assertTrue(mg_result["e3fp_ids"].eq(0).all())

    def test_cache_aliases_are_mapped_and_metadata_is_removed(self) -> None:
        result = forward_task(_RecordingModel(), "M", _batch(molecular=True))
        self.assertIn("atom_mask", result)
        self.assertIn("token_is_connector_endpoint", result)
        self.assertNotIn("e3fp_atom_mask", result)
        self.assertNotIn("connector_endpoint_mask", result)
        self.assertNotIn("records", result)


if __name__ == "__main__":
    unittest.main()
