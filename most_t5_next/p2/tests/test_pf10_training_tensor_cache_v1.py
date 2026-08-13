from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import (
    ProductionMotifRecord,
    ProductionTokenizerRuntime,
)
from most_t5_next.p2.pf10_training_tensor_cache_v1 import (
    CachedCanonicalAtomAddressProvider,
    CachedMorganAtomStateProvider,
    PF10TrainingTensorCache,
    PF10TrainingTensorCacheError,
    V3EpochViewBatchSampler,
    build_v3_cache_dataloader,
    build_pf10_training_tensor_cache,
)
from most_t5_next.p2.benchmark_pf10_training_tensor_cache_v1 import (
    benchmark_cache_loader,
)
from most_t5_next.p2.three_d_motif_training_views_v3 import (
    collate_3d_motif_training_view_v3,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _runtime() -> ProductionTokenizerRuntime:
    return ProductionTokenizerRuntime(
        tokenizer_contract_sha256=_digest("contract"),
        tokenizer_snapshot_sha256=_digest("snapshot"),
        vocab_size=128,
        pad_token_id=0,
        eos_token_id=1,
        sentinel_token_ids=tuple(range(127, 117, -1)),
    )


def _record(record_id: str, offset: int = 0) -> ProductionMotifRecord:
    runtime = _runtime()
    return ProductionMotifRecord(
        record_artifact_sha256=_digest("record-" + record_id),
        record_id=record_id,
        storage_key="fixture/" + record_id,
        release_id="fixture-release",
        geometry_record_content_sha256=_digest("geometry-" + record_id),
        tokenizer_contract_sha256=runtime.tokenizer_contract_sha256,
        tokenizer_snapshot_sha256=runtime.tokenizer_snapshot_sha256,
        input_ids=(10, 11, 12, 13, 14, 15),
        token_to_logical_motif=(0, 0, 0, 1, 1, 1),
        token_role=(
            "identity",
            "identity",
            "connection",
            "identity",
            "identity",
            "connection",
        ),
        identity_spans=(Span(0, 2), Span(3, 5)),
        connection_token_indices=((2,), (5,)),
        logical_to_carrier=(0, 3),
        exact_identity_sha256=(_digest("left"), _digest("right")),
        source_atom_count=3,
        full_e3fp_ids=(
            (1 + offset, 2 + offset, 3 + offset, 4 + offset),
            (5 + offset, 6 + offset, 7 + offset, 8 + offset),
            (9 + offset, 10 + offset, 11 + offset, 12 + offset),
        ),
        atom_valid_mask=(True, True, True),
        model_to_source_atom_index=(0, 1, 2),
        atom_to_logical_motif=(0, 0, 1),
        atom_is_attachment=(False, True, True),
        connection_token_to_atom=(-1, -1, 1, -1, -1, 2),
    )


def _donor_row(record: ProductionMotifRecord, split: str) -> dict[str, object]:
    return {
        "member_id": record.record_id,
        "storage_key": record.storage_key,
        "split": split,
        "overlay_planning_sidecar": {
            "canonical_local_atom_to_model_atom": [
                [[1, 0], [2, 1]],
                [[1, 2]],
            ]
        },
    }


class _Reader:
    train_records: tuple[ProductionMotifRecord, ...] = ()
    dev_records: tuple[ProductionMotifRecord, ...] = ()
    parallel_calls: list[tuple[str, int | None, int, int]] = []

    def __init__(self, _root: Path) -> None:
        self.train_member_count = len(self.train_records)
        self.dev_member_count = len(self.dev_records)

    @staticmethod
    def _batches(records, batch_size):
        for start in range(0, len(records), batch_size):
            yield tuple(
                SimpleNamespace(motif_record=record)
                for record in records[start : start + batch_size]
            )

    def iter_train_epoch(self, *, epoch: int, batch_size: int):
        if epoch != 0:
            raise AssertionError("cache compilation must use epoch zero")
        return self._batches(self.train_records, batch_size)

    def iter_dev(self, *, batch_size: int):
        return self._batches(self.dev_records, batch_size)

    def iter_strict_parallel_split(
        self, *, split: str, max_rows=None, workers: int, max_pending: int
    ):
        self.parallel_calls.append((split, max_rows, workers, max_pending))
        records = self.train_records if split == "train" else self.dev_records
        if max_rows is not None:
            records = records[:max_rows]
        for record in records:
            yield SimpleNamespace(motif_record=record)

    def iter_donor_atom_maps(self, *, split: str, max_rows: int | None = None):
        records = self.train_records if split == "train" else self.dev_records
        if max_rows is not None:
            records = records[:max_rows]
        for record in records:
            yield _donor_row(record, split)


class _Morgan:
    states: dict[str, tuple[tuple[int, ...], ...]] = {}

    def __init__(self, _root: Path) -> None:
        pass

    def get(self, record_id: str):
        return self.states[record_id]

    def close(self) -> None:
        pass


class _AddressProvider:
    def get(self, _record: ProductionMotifRecord):
        return (0, 1, 0)


class _StateProvider:
    state_kind = "most-t5-p2/coordinate-blind-morgan-atom-state/r3-fp4096-v1"

    def __init__(self, states) -> None:
        self.states = states

    def get(self, record_id: str):
        return self.states[record_id]


class PF10TrainingTensorCacheV1Test(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paired = self.root / "paired"
        self.morgan = self.root / "morgan"
        self.output = self.root / "cache"
        self.caches: list[PF10TrainingTensorCache] = []
        self.paired.mkdir()
        self.morgan.mkdir()
        (self.paired / "manifest.json").write_text("{}\n", encoding="utf-8")
        (self.paired / "donor_atom_maps.jsonl").write_text("{}\n", encoding="utf-8")
        (self.morgan / "manifest.json").write_text("{}\n", encoding="utf-8")
        self.train = (_record("train-a"), _record("train-b", 20))
        self.dev = (_record("dev-a", 40),)
        _Reader.train_records = self.train
        _Reader.dev_records = self.dev
        _Reader.parallel_calls = []
        _Morgan.states = {
            record.record_id: tuple(
                tuple(value + 100 for value in atom)
                for atom in record.full_e3fp_ids
            )
            for record in self.train + self.dev
        }

    def tearDown(self) -> None:
        for cache in self.caches:
            cache.close()
        self.temporary.cleanup()

    def _build(self):
        return build_pf10_training_tensor_cache(
            paired_release=self.paired,
            morgan_overlay=self.morgan,
            output_dir=self.output,
            reader_factory=_Reader,
            morgan_provider_factory=_Morgan,
            source_extensions={"surface": "anchored-fixture"},
        )

    def test_round_trip_recreates_every_production_field(self) -> None:
        manifest = self._build()
        self.assertEqual(manifest["counts"]["records"], 3)
        self.assertEqual(
            manifest["source"]["derived_representation"],
            {"surface": "anchored-fixture"},
        )
        cache = PF10TrainingTensorCache(self.output)
        self.caches.append(cache)
        self.assertEqual(cache.split_indices("train"), (0, 1))
        self.assertEqual(cache.split_indices("dev"), (2,))
        for expected, index in zip(self.train + self.dev, range(3)):
            actual = cache[index]
            for field in fields(ProductionMotifRecord):
                self.assertEqual(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    field.name,
                )
            self.assertEqual(cache.atom_local_positions(actual), (0, 1, 0))

    def test_cached_and_authoritative_dynamic_v3_batches_are_identical(self) -> None:
        self._build()
        cache = PF10TrainingTensorCache(self.output)
        self.caches.append(cache)
        cached_records = tuple(cache[index] for index in cache.split_indices("train"))
        runtime = _runtime()
        authoritative = collate_3d_motif_training_view_v3(
            self.train,
            view_id="m_plus_g",
            tokenizer=runtime,
            seed=20260807,
            epoch=3,
            atom_address_provider=_AddressProvider(),
            atom_state_provider=_StateProvider(_Morgan.states),
            num_e3fp_embeddings=4096,
        ).model_inputs()
        cached = collate_3d_motif_training_view_v3(
            cached_records,
            view_id="m_plus_g",
            tokenizer=runtime,
            seed=20260807,
            epoch=3,
            atom_address_provider=CachedCanonicalAtomAddressProvider(cache),
            atom_state_provider=CachedMorganAtomStateProvider(cache, cached_records),
            num_e3fp_embeddings=4096,
        ).model_inputs()
        self.assertEqual(set(authoritative), set(cached))
        for key in authoritative:
            if isinstance(authoritative[key], torch.Tensor):
                self.assertTrue(torch.equal(authoritative[key], cached[key]), key)
            else:
                self.assertEqual(authoritative[key], cached[key], key)

    def test_parallel_build_decodes_only_the_selected_split_prefixes(self) -> None:
        manifest = build_pf10_training_tensor_cache(
            paired_release=self.paired,
            morgan_overlay=self.morgan,
            output_dir=self.output,
            max_train_records=1,
            max_dev_records=1,
            decode_workers=2,
            decode_max_pending=4,
            reader_factory=_Reader,
            morgan_provider_factory=_Morgan,
        )
        self.assertEqual(manifest["counts"]["records"], 2)
        self.assertEqual(
            _Reader.parallel_calls,
            [("train", 1, 2, 4), ("dev", 1, 2, 4)],
        )
        self.assertTrue(manifest["build"]["strict_ordered_selected_decode"])
        self.assertFalse(
            manifest["build"]["decoded_records_retained_in_python_cache"]
        )

    def test_sampler_preserves_short_tail_and_advances_epoch(self) -> None:
        sampler = V3EpochViewBatchSampler(
            range(10),
            cell="F3D",
            micro_batch_size=4,
            gradient_accumulation_steps=2,
            total_updates=2,
        )
        batches = tuple(sampler)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2, 4])
        self.assertEqual(
            [[row.record_index for row in batch] for batch in batches],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [0, 1, 2, 3]],
        )
        self.assertEqual([batch[0].epoch for batch in batches], [0, 0, 0, 1])
        self.assertEqual(
            [batch[0].view_id for batch in batches],
            ["m_plus_g", "m_plus_g", "g_only", "g_only"],
        )

    def test_epoch_shuffle_is_deterministic_and_never_drops_the_tail(self) -> None:
        def flattened():
            sampler = V3EpochViewBatchSampler(
                range(10),
                cell="F3D",
                micro_batch_size=4,
                gradient_accumulation_steps=1,
                total_updates=6,
                shuffle_seed=20260807,
            )
            return tuple(
                (row.epoch, row.record_index)
                for batch in sampler
                for row in batch
            )

        left = flattened()
        right = flattened()
        self.assertEqual(left, right)
        epoch_zero = [index for epoch, index in left if epoch == 0]
        epoch_one = [index for epoch, index in left if epoch == 1]
        self.assertEqual(sorted(epoch_zero), list(range(10)))
        self.assertEqual(sorted(epoch_one), list(range(10)))
        self.assertNotEqual(epoch_zero, epoch_one)

    def test_fixed_view_keeps_dynamic_epoch_masks_without_changing_objective(self) -> None:
        sampler = V3EpochViewBatchSampler(
            range(3),
            cell="F3D",
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            total_updates=3,
            fixed_view_id="m_plus_g",
        )
        batches = tuple(sampler)
        self.assertEqual([row.view_id for batch in batches for row in batch], [
            "m_plus_g",
            "m_plus_g",
            "m_plus_g",
            "m_plus_g",
            "m_plus_g",
        ])
        self.assertEqual([batch[0].epoch for batch in batches], [0, 0, 1])

    def test_loader_rejects_a_truncated_array(self) -> None:
        self._build()
        path = self.output / "input_ids.bin"
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
        with self.assertRaisesRegex(PF10TrainingTensorCacheError, "truncated"):
            PF10TrainingTensorCache(self.output)

    def test_spawn_workers_collate_dynamic_batches_from_shared_mmaps(self) -> None:
        self._build()
        loader = build_v3_cache_dataloader(
            cache_root=self.output,
            tokenizer=_runtime(),
            cell="F3D",
            seed=20260807,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            total_updates=1,
            num_workers=2,
            prefetch_factor=2,
        )
        iterator = iter(loader)
        try:
            batch = next(iterator)
            self.assertEqual(batch.view_id, "m_plus_g")
            self.assertEqual(batch.record_ids, ("train-a", "train-b"))
            self.assertEqual(
                batch.exact_identity_sha256,
                tuple(record.exact_identity_sha256 for record in self.train),
            )
            self.assertEqual(tuple(batch.labels.shape)[0], 2)
            self.assertIn("endpoint_token_to_atom", batch.inputs)
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            loader.dataset.close()

    def test_benchmark_reports_single_process_dynamic_throughput(self) -> None:
        self._build()
        with mock.patch(
            "most_t5_next.p2.benchmark_pf10_training_tensor_cache_v1."
            "load_verified_canary_union_tokenizer",
            return_value=SimpleNamespace(runtime=_runtime()),
        ):
            report = benchmark_cache_loader(
                cache_root=self.output,
                base_tokenizer_snapshot=self.root / "base",
                union_tokenizer_dir=self.root / "union",
                workers=(0,),
                batches=2,
                micro_batch_size=1,
                prefetch_factor=2,
                cell="F3D",
            )
        row = report["results"][0]
        self.assertEqual(row["workers"], 0)
        self.assertEqual(row["members"], 2)
        self.assertGreater(row["members_per_second_including_startup"], 0.0)

    def test_worker_count_does_not_change_order_masks_or_padding(self) -> None:
        self._build()

        def collect(workers: int):
            loader = build_v3_cache_dataloader(
                cache_root=self.output,
                tokenizer=_runtime(),
                cell="B0",
                seed=20260807,
                micro_batch_size=2,
                gradient_accumulation_steps=1,
                total_updates=2,
                num_workers=workers,
                prefetch_factor=2,
            )
            iterator = iter(loader)
            rows = []
            try:
                for batch in iterator:
                    rows.append(
                        (
                            batch.epoch,
                            batch.record_ids,
                            {
                                key: value.clone()
                                for key, value in batch.inputs.items()
                                if isinstance(value, torch.Tensor)
                            },
                        )
                    )
            finally:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
                loader.dataset.close()
            return rows

        serial = collect(0)
        parallel = collect(2)
        self.assertEqual(len(serial), 2)
        self.assertEqual([row[:2] for row in serial], [row[:2] for row in parallel])
        self.assertEqual([row[0] for row in serial], [0, 1])
        for (_, _, left), (_, _, right) in zip(serial, parallel):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]), key)


if __name__ == "__main__":
    unittest.main()
