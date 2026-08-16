from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.training.runner import run_two_phase_pretraining


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "pretrain.yaml"


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 32)

    def forward(self, input_ids, attention_mask, *, labels=None, **_):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        logits = self.projection(hidden)
        loss = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


class Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, task, update):
        self.calls.append((task.name, update))
        return tuple(
            {
                "input_ids": torch.tensor([[2, 3, 4], [5, 6, 7]]),
                "attention_mask": torch.ones((2, 3), dtype=torch.bool),
                "labels": torch.tensor([[3, 4, 5], [6, 7, 8]]),
            }
            for _ in range(2)
        )


class RunnerTest(unittest.TestCase):
    def test_two_phases_use_balanced_update_tasks_and_model_only_boundary(self):
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        for phase_name in ("phase_one", "phase_two"):
            config["curriculum"][phase_name]["total_updates"] = 4
            config["optimization"][phase_name]["base_learning_rate"] = 1.0e-3
            config["optimization"][phase_name]["warmup_updates"] = 1
        config["optimization"]["warmup_start_factor"] = 0.5
        config["optimization"]["final_learning_rate"] = 1.0e-5
        config["optimization"]["precision"] = "fp32"
        config["batching"]["micro_batch_size"] = 2
        config["batching"]["gradient_accumulation_steps"] = 2
        config["batching"]["effective_batch_size"] = 4
        config["monitoring"]["checkpoint_every_updates"] = 2
        phase_one_provider = Provider()
        phase_two_provider = Provider()
        with tempfile.TemporaryDirectory() as temporary:
            report = run_two_phase_pretraining(
                model=TinyModel(),
                phase_one_batch_provider=phase_one_provider,
                phase_two_batch_provider_factory=lambda: phase_two_provider,
                config=config,
                output_dir=temporary,
                device="cpu",
            )
            self.assertEqual(
                report["phase_one"]["task_updates"], {"M": 2, "MG": 2}
            )
            self.assertEqual(
                report["phase_two"]["task_updates"],
                {"CAP": 1, "SYN": 1, "T2M": 1, "TXT": 1},
            )
            self.assertEqual(
                report["phase_two"]["optimizer_update_batching"],
                {
                    "logical_batch_size": 4,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "sample_before_microbatch_split": True,
                },
            )
            boundary = torch.load(Path(temporary) / "phase-one-model-boundary.pt")
            self.assertFalse(boundary["optimizer_state_included"])
            self.assertFalse(boundary["scheduler_state_included"])
            self.assertTrue((Path(temporary) / "training-manifest.json").is_file())
            self.assertEqual(
                [name for name, _ in phase_one_provider.calls],
                ["M", "MG", "M", "MG"],
            )
            self.assertEqual(
                [name for name, _ in phase_two_provider.calls],
                ["SYN", "TXT", "CAP", "T2M"],
            )


if __name__ == "__main__":
    unittest.main()
