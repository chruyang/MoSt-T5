"""CPU mock tests for the sequential real-GPU four-grid canary runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency-free hosts
    torch = None

from most_t5_next.p1.experiment_grid import (
    GeometryBatchSidecar,
    P1ConditionBatch,
)
from most_t5_next.p1.runtime_bridge import PaddedCEBatch
from most_t5_next.p1 import run_four_grid_gpu_canary_v1 as subject


def _ce_batch(*, motif: bool) -> PaddedCEBatch:
    if motif:
        return PaddedCEBatch(
            record_ids=("member-0", "member-1"),
            input_ids=((12, 13, 1), (14, 1, 0)),
            attention_mask=((True, True, True), (True, True, False)),
            labels=((20, 21, 1), (22, 1, -100)),
            input_lengths=(3, 2),
            target_lengths=(3, 2),
        )
    return PaddedCEBatch(
        record_ids=("member-0", "member-1"),
        input_ids=((2, 3, 4, 1), (5, 6, 1, 0)),
        attention_mask=((True, True, True, True), (True, True, True, False)),
        labels=((30, 31, 1), (32, 1, -100)),
        input_lengths=(4, 3),
        target_lengths=(3, 2),
    )


def _geometry(*, token_width: int, motif: bool) -> GeometryBatchSidecar:
    carriers = ((0, 0), (0, -1)) if motif else ((0, 1), (0, -1))
    return GeometryBatchSidecar(
        record_ids=("member-0", "member-1"),
        e3fp_ids=(((1, 2, 3, 4), (5, 6, 7, 8)), ((9, 10, 11, 12), (-1, -1, -1, -1))),
        e3fp_atom_mask=((True, True), (True, False)),
        e3fp_atom_to_token=carriers,
        model_to_source_atom_index=((0, 2), (1, -1)),
        atom_lengths=(2, 1),
        e3fp_level_count=4,
        token_width=token_width,
    )


def _grid_batches() -> dict[str, P1ConditionBatch]:
    atom = _ce_batch(motif=False)
    motif = _ce_batch(motif=True)
    return {
        "A0": P1ConditionBatch("A0", atom),
        "A1": P1ConditionBatch("A1", atom, _geometry(token_width=4, motif=False)),
        "M0": P1ConditionBatch("M0", motif),
        "M1": P1ConditionBatch("M1", motif, _geometry(token_width=3, motif=True)),
    }


class FourGridCollationContractTest(unittest.TestCase):
    def test_frozen_collation_uses_one_declared_epoch_and_closes_parity(self) -> None:
        batches = _grid_batches()
        calls: list[tuple[str, int, int, float]] = []

        def atom_collator(_records, **kwargs):
            calls.append(
                (
                    kwargs["condition_id"],
                    kwargs["seed"],
                    kwargs["epoch"],
                    kwargs["mask_probability"],
                )
            )
            return batches[kwargs["condition_id"]]

        def motif_collator(_records, **kwargs):
            calls.append(
                (
                    kwargs["condition_id"],
                    kwargs["seed"],
                    kwargs["epoch"],
                    kwargs["mask_probability"],
                )
            )
            return batches[kwargs["condition_id"]]

        records = (
            SimpleNamespace(atom_record=object(), motif_record=object()),
            SimpleNamespace(atom_record=object(), motif_record=object()),
        )
        with patch.object(subject, "collate_production_atom_batch", atom_collator), patch.object(
            subject, "collate_production_batch", motif_collator
        ):
            observed = subject.build_frozen_grid_batches(
                records, tokenizer_runtime=object()
            )

        self.assertEqual(tuple(observed), subject.CONDITION_ORDER)
        self.assertEqual(
            calls,
            [
                ("A0", 0, 0, 0.15),
                ("A1", 0, 0, 0.15),
                ("M0", 0, 0, 0.15),
                ("M1", 0, 0, 0.15),
            ],
        )
        self.assertEqual(observed["A0"].ce_batch, observed["A1"].ce_batch)
        self.assertEqual(observed["M0"].ce_batch, observed["M1"].ce_batch)


@unittest.skipIf(torch is None, "PyTorch is required")
class FourGridExecutionMockTest(unittest.TestCase):
    def test_four_cells_load_run_and_release_sequentially_without_step(self) -> None:
        batches = _grid_batches()
        loaded: list[str] = []
        wrappers: list[_FakeWrapper] = []

        def loader(**kwargs):
            condition = kwargs["condition_id"]
            loaded.append(condition)
            wrapper = _FakeWrapper(condition)
            wrappers.append(wrapper)
            return wrapper

        def encode(batch, *, device):
            values = {
                "input_ids": torch.tensor(batch.ce_batch.input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(
                    batch.ce_batch.attention_mask, dtype=torch.long, device=device
                ),
                "labels": torch.tensor(batch.ce_batch.labels, dtype=torch.long, device=device),
            }
            if batch.geometry is not None:
                values.update(
                    {
                        "e3fp_ids": torch.tensor(
                            batch.geometry.e3fp_ids, dtype=torch.long, device=device
                        ),
                        "e3fp_atom_mask": torch.tensor(
                            batch.geometry.e3fp_atom_mask,
                            dtype=torch.bool,
                            device=device,
                        ),
                        "e3fp_atom_to_token": torch.tensor(
                            batch.geometry.e3fp_atom_to_token,
                            dtype=torch.long,
                            device=device,
                        ),
                    }
                )
            return values

        with patch.object(subject, "to_four_grid_batch_encoding", side_effect=encode):
            results = subject.execute_four_grid(
                batches,
                base_model_snapshot=Path("base-model"),
                base_tokenizer_snapshot=Path("base-tokenizer"),
                union_tokenizer_dir=Path("union-tokenizer"),
                union_init_dir=Path("union-init"),
                geometry_fusion_seed=19,
                num_e3fp_embeddings=4096,
                expected_vocab_size=64,
                device=torch.device("cpu"),
                use_bf16=False,
                torch_module=torch,
                wrapper_loader=loader,
            )

        self.assertEqual(loaded, list(subject.CONDITION_ORDER))
        self.assertEqual(
            [row["condition"] for row in results], list(subject.CONDITION_ORDER)
        )
        self.assertEqual(
            [row["geometry_enabled"] for row in results],
            [False, True, False, True],
        )
        for row in results:
            self.assertTrue(row["loss_finite"])
            self.assertTrue(row["gradients_finite"])
            self.assertGreater(row["gradient_norm"], 0.0)
            self.assertEqual(row["optimizer_steps"], 0)
            self.assertEqual(row["peak_gpu_memory_bytes"], 0)
        for wrapper in wrappers:
            self.assertEqual(float(wrapper.scale.detach()), 0.25)
            self.assertIsNone(wrapper.scale.grad)
        self.assertEqual(
            [wrapper.saw_geometry for wrapper in wrappers],
            [False, True, False, True],
        )


class _FakeWrapper(torch.nn.Module if torch is not None else object):
    def __init__(self, condition_id: str) -> None:
        super().__init__()
        self.condition_id = condition_id
        self.config = SimpleNamespace(vocab_size=64)
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.saw_geometry = False

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        labels,
        e3fp_ids=None,
        e3fp_atom_mask=None,
        e3fp_atom_to_token=None,
        **_kwargs,
    ):
        self.saw_geometry = e3fp_ids is not None
        signal = (input_ids.float() * attention_mask.float()).sum()
        signal = signal + labels.masked_fill(labels < 0, 0).float().sum()
        if e3fp_ids is not None:
            signal = signal + e3fp_ids.clamp_min(0).float().sum() * 0.01
            self.assert_geometry_triplet = (
                e3fp_atom_mask is not None and e3fp_atom_to_token is not None
            )
        loss = (self.scale * signal / 100.0 - 1.0).square()
        return SimpleNamespace(loss=loss)


if __name__ == "__main__":
    unittest.main()
