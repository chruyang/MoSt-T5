from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.training.optimization import OptimizationConfig
from most_t5_next.training.runner import (
    TrainingError,
    read_checkpoint_metadata,
    run_training_phase,
    run_two_phase_pretraining,
)
from most_t5_next.training.runtime import TrainingRuntimeConfig, seed_everything


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


class PartitionedProvider:
    def __init__(self, partitions):
        self.partitions = partitions

    def partition_for_task(self, task):
        return self.partitions[task]

    def __call__(self, task, update):
        micro_batch_size, accumulation_steps = self.partition_for_task(task.name)
        return tuple(
            {
                "input_ids": torch.full((micro_batch_size, 3), 2, dtype=torch.long),
                "attention_mask": torch.ones((micro_batch_size, 3), dtype=torch.bool),
                "labels": torch.full((micro_batch_size, 3), 3, dtype=torch.long),
            }
            for _ in range(accumulation_steps)
        )


class NoSyncTinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = TinyModel()
        self.no_sync_calls = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


class DropoutTinyModel(TinyModel):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask, *, labels=None, **_):
        hidden = self.dropout(self.embedding(input_ids)) * attention_mask.unsqueeze(-1)
        logits = self.projection(hidden)
        loss = F.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


class RandomProvider:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at

    def __call__(self, task, update):
        if self.fail_at is not None and update >= self.fail_at:
            raise RuntimeError("intentional interruption")
        offset = (
            random.randrange(7)
            + int(np.random.randint(0, 7))
            + int(torch.randint(0, 7, ()).item())
        ) % 7
        return (
            {
                "input_ids": torch.tensor(
                    [[2 + offset, 3 + offset], [4 + offset, 5 + offset]]
                ),
                "attention_mask": torch.ones((2, 2), dtype=torch.bool),
                "labels": torch.tensor(
                    [[3 + offset, 4 + offset], [5 + offset, 6 + offset]]
                ),
            },
        )


class RunnerTest(unittest.TestCase):
    def test_checkpoint_resume_is_bitwise_equivalent_and_restores_all_rngs(self):
        optimization = OptimizationConfig(
            total_updates=6,
            warmup_updates=1,
            base_learning_rate=1.0e-3,
            warmup_start_factor=0.5,
            final_learning_rate=1.0e-5,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0e-6,
        )
        runtime = TrainingRuntimeConfig(
            seed=42,
            precision="fp32",
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=0,
            log_every_updates=1,
            checkpoint_every_updates=2,
        )
        protocol = {"fixture": "resume-v1"}
        with tempfile.TemporaryDirectory() as full_dir, tempfile.TemporaryDirectory() as resumed_dir:
            seed_everything(123)
            full_model = DropoutTinyModel()
            run_training_phase(
                model=full_model,
                phase=1,
                batch_provider=RandomProvider(),
                optimization=optimization,
                runtime=runtime,
                output_dir=full_dir,
                device="cpu",
                checkpoint_protocol=protocol,
            )

            seed_everything(123)
            interrupted_model = DropoutTinyModel()
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                run_training_phase(
                    model=interrupted_model,
                    phase=1,
                    batch_provider=RandomProvider(fail_at=2),
                    optimization=optimization,
                    runtime=runtime,
                    output_dir=resumed_dir,
                    device="cpu",
                    checkpoint_protocol=protocol,
                )
            checkpoint = Path(resumed_dir) / "phase-1-step-00000002.pt"
            self.assertEqual(
                read_checkpoint_metadata(checkpoint)["next_update"], 2
            )

            seed_everything(999)
            resumed_model = DropoutTinyModel()
            run_training_phase(
                model=resumed_model,
                phase=1,
                batch_provider=RandomProvider(),
                optimization=optimization,
                runtime=runtime,
                output_dir=resumed_dir,
                device="cpu",
                resume_checkpoint=checkpoint,
                checkpoint_protocol=protocol,
            )

            full = torch.load(
                Path(full_dir) / "phase-1-step-00000006.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed = torch.load(
                Path(resumed_dir) / "phase-1-step-00000006.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, value in full["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][name]), name)
            self.assertEqual(full["schedule"], resumed["schedule"])
            self.assertEqual(
                full["schedule_completed_updates"],
                resumed["schedule_completed_updates"],
            )
            self.assertEqual(full["optimizer"].keys(), resumed["optimizer"].keys())
            self.assertEqual(
                full["rank_rng_states"][0]["python_rng_state"],
                resumed["rank_rng_states"][0]["python_rng_state"],
            )
            np.testing.assert_equal(
                full["rank_rng_states"][0]["numpy_rng_state"],
                resumed["rank_rng_states"][0]["numpy_rng_state"],
            )
            for parameter_id, state in full["optimizer"]["state"].items():
                for key, value in state.items():
                    other = resumed["optimizer"]["state"][parameter_id][key]
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, other), key)
                    else:
                        self.assertEqual(value, other)
            full_progress = dict(full["rank_progress_states"][0])
            resumed_progress = dict(resumed["rank_progress_states"][0])
            full_progress.pop("wall_seconds")
            resumed_progress.pop("wall_seconds")
            self.assertEqual(full_progress, resumed_progress)

            with self.assertRaisesRegex(TrainingError, "protocol differs"):
                run_training_phase(
                    model=DropoutTinyModel(),
                    phase=1,
                    batch_provider=RandomProvider(),
                    optimization=optimization,
                    runtime=runtime,
                    output_dir=resumed_dir,
                    device="cpu",
                    resume_checkpoint=checkpoint,
                    checkpoint_protocol={"fixture": "different"},
                )

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
                    "rank_local_logical_batch_size": 4,
                    "global_logical_batch_size": 4,
                    "micro_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "sample_before_microbatch_split": True,
                    "gradient_syncs_per_optimizer_update": 1,
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
            resumed_phase_two_provider = Provider()
            resumed = run_two_phase_pretraining(
                model=TinyModel(),
                phase_one_batch_provider=None,
                phase_two_batch_provider_factory=lambda: resumed_phase_two_provider,
                config=config,
                output_dir=temporary,
                device="cpu",
                resume_checkpoint=(
                    Path(temporary) / "phase-2-step-00000004.pt"
                ),
            )
            self.assertEqual(resumed_phase_two_provider.calls, [])
            self.assertEqual(
                resumed["phase_two"]["task_updates"],
                {"CAP": 1, "SYN": 1, "T2M": 1, "TXT": 1},
            )
            self.assertTrue(resumed["phase_two"]["resumed_from"].endswith(
                "phase-2-step-00000004.pt"
            ))

    def test_phase_reports_task_specific_physical_partitions(self):
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        for phase_name in ("phase_one", "phase_two"):
            config["curriculum"][phase_name]["total_updates"] = 4
            config["optimization"][phase_name]["base_learning_rate"] = 1.0e-3
            config["optimization"][phase_name]["warmup_updates"] = 1
        config["optimization"]["warmup_start_factor"] = 0.5
        config["optimization"]["final_learning_rate"] = 1.0e-5
        config["optimization"]["precision"] = "fp32"
        config["batching"]["micro_batch_size"] = 4
        config["batching"]["gradient_accumulation_steps"] = 1
        config["batching"]["effective_batch_size"] = 4
        config["monitoring"]["checkpoint_every_updates"] = None
        phase_one = PartitionedProvider({"M": (4, 1), "MG": (4, 1)})
        phase_two = PartitionedProvider(
            {"SYN": (4, 1), "TXT": (2, 2), "CAP": (2, 2), "T2M": (2, 2)}
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = run_two_phase_pretraining(
                model=TinyModel(),
                phase_one_batch_provider=phase_one,
                phase_two_batch_provider_factory=lambda: phase_two,
                config=config,
                output_dir=temporary,
                device="cpu",
            )
        self.assertEqual(
            report["phase_two"]["optimizer_update_batching"],
            {
                "rank_local_logical_batch_size": 4,
                "global_logical_batch_size": 4,
                "sample_before_microbatch_split": True,
                "gradient_syncs_per_optimizer_update": 1,
                "task_partitions": {
                    "CAP": {"micro_batch_size": 2, "gradient_accumulation_steps": 2},
                    "SYN": {"micro_batch_size": 4, "gradient_accumulation_steps": 1},
                    "T2M": {"micro_batch_size": 2, "gradient_accumulation_steps": 2},
                    "TXT": {"micro_batch_size": 2, "gradient_accumulation_steps": 2},
                },
            },
        )

    def test_only_nonfinal_microbatches_use_no_sync(self):
        config = deepcopy(load_pretraining_config(CONFIG_PATH))
        for phase_name in ("phase_one", "phase_two"):
            config["curriculum"][phase_name]["total_updates"] = 4
            config["optimization"][phase_name]["base_learning_rate"] = 1.0e-3
            config["optimization"][phase_name]["warmup_updates"] = 1
        config["optimization"]["warmup_start_factor"] = 0.5
        config["optimization"]["final_learning_rate"] = 1.0e-5
        config["optimization"]["precision"] = "fp32"
        config["batching"]["micro_batch_size"] = 4
        config["batching"]["gradient_accumulation_steps"] = 1
        config["batching"]["effective_batch_size"] = 4
        config["monitoring"]["checkpoint_every_updates"] = None
        phase_one = PartitionedProvider({"M": (4, 1), "MG": (4, 1)})
        phase_two = PartitionedProvider(
            {"SYN": (4, 1), "TXT": (2, 2), "CAP": (2, 2), "T2M": (2, 2)}
        )
        model = NoSyncTinyModel()
        with tempfile.TemporaryDirectory() as temporary:
            run_two_phase_pretraining(
                model=model,
                phase_one_batch_provider=phase_one,
                phase_two_batch_provider_factory=lambda: phase_two,
                config=config,
                output_dir=temporary,
                device="cpu",
            )
        self.assertEqual(model.no_sync_calls, 3)


if __name__ == "__main__":
    unittest.main()
