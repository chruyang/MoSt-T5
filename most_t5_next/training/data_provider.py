"""Persistent multi-worker provider for the six formal pretraining tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from most_t5_next.p2.build_phase2_paired_text_cache_v1 import Phase2PairedTextCache
from most_t5_next.p2.fragsmiles_paired_pretraining_collator_v1 import (
    Phase2PairedSample,
    collate_phase2_cap_samples,
    collate_phase2_t2m_samples,
)
from most_t5_next.p2.fragsmiles_pretraining_collator_v1 import (
    collate_molecular_denoising_samples,
)
from most_t5_next.p2.fragsmiles_training_tensor_cache_v1 import (
    CachedFragSmilesSample,
    FragSmilesTrainingTensorCache,
)
from most_t5_next.p2.pf10_text_denoising_cache_v1 import (
    PF10PackedTextTrainingCorpus,
    PF10TextDenoisingCollator,
)

from .curriculum import CurriculumSchedule, TaskSpec
from .runtime import TrainingRuntimeConfig


class DataProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class CurriculumIndex:
    task: str
    source_index: int
    epoch: int
    update: int
    microbatch: int


@dataclass(frozen=True)
class TaggedSample:
    index: CurriculumIndex
    payload: Any


class CurriculumDataset(Dataset):
    def __init__(
        self,
        *,
        pcqm_cache: str | Path,
        pubchem_cache: str | Path,
        paired_text_cache: str | Path,
        pubmed_cache: str | Path,
    ) -> None:
        self.roots = {
            "pcqm": Path(pcqm_cache),
            "pubchem": Path(pubchem_cache),
            "paired": Path(paired_text_cache),
            "pubmed": Path(pubmed_cache),
        }
        self._open()

    def _open(self) -> None:
        self.pcqm = FragSmilesTrainingTensorCache(
            self.roots["pcqm"], verify_hashes=False
        )
        self.pubchem = FragSmilesTrainingTensorCache(
            self.roots["pubchem"], verify_hashes=False
        )
        self.paired = Phase2PairedTextCache(
            self.roots["paired"], molecular_cache_root=self.roots["pubchem"]
        )
        self.pubmed = PF10PackedTextTrainingCorpus(self.roots["pubmed"])

    def __getstate__(self) -> dict[str, Any]:
        return {"roots": self.roots}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.roots = dict(state["roots"])
        self._open()

    def population_sizes(self) -> dict[str, int]:
        return {
            "M": len(self.pcqm) + len(self.pubchem),
            "MG": len(self.pcqm),
            "SYN": len(self.pcqm) + len(self.pubchem),
            "TXT": len(self.pubmed),
            "CAP": len(self.pubchem),
            "T2M": len(self.pubchem),
        }

    def _union_record(self, index: int) -> Any:
        if index < len(self.pcqm):
            return self.pcqm[index]
        return self.pubchem[index - len(self.pcqm)]

    def __getitem__(self, index: CurriculumIndex) -> TaggedSample:
        if not isinstance(index, CurriculumIndex):
            raise IndexError("curriculum dataset requires CurriculumIndex")
        if index.task in {"M", "SYN"}:
            payload: Any = CachedFragSmilesSample(
                self._union_record(index.source_index), index.epoch
            )
        elif index.task == "MG":
            payload = CachedFragSmilesSample(
                self.pcqm[index.source_index], index.epoch
            )
        elif index.task in {"CAP", "T2M"}:
            record = self.pubchem[index.source_index]
            cid, text_ids = self.paired[index.source_index]
            if cid != record.ordinal:
                raise DataProviderError("paired PubChem row identity differs")
            payload = Phase2PairedSample(record, text_ids)
        elif index.task == "TXT":
            payload = replace(self.pubmed[index.source_index], epoch=index.epoch)
        else:
            raise IndexError(index.task)
        return TaggedSample(index, payload)

    def close(self) -> None:
        self.pcqm.close()
        self.pubchem.close()
        self.paired.close()
        self.pubmed.close()


class CurriculumBatchSampler(Sampler):
    """Emit deterministic task-homogeneous microbatches for one phase."""

    def __init__(
        self,
        *,
        phase: int,
        total_updates: int,
        populations: Mapping[str, Sequence[int]],
        micro_batch_size: int,
        accumulation_steps: int,
        seed: int,
        start_update: int = 0,
    ) -> None:
        self.curriculum = CurriculumSchedule(phase, total_updates)
        self.populations = {}
        for task, indices in populations.items():
            if isinstance(indices, range):
                array = np.arange(
                    indices.start, indices.stop, indices.step, dtype=np.int64
                )
            else:
                array = np.asarray(indices, dtype=np.int64)
            if array.ndim != 1:
                raise DataProviderError("population indices must be one-dimensional")
            self.populations[task] = array
        self.micro_batch_size = int(micro_batch_size)
        self.accumulation_steps = int(accumulation_steps)
        self.seed = int(seed)
        self.start_update = int(start_update)
        if not 0 <= self.start_update <= total_updates:
            raise DataProviderError("start_update is outside the phase")
        required = set(self.curriculum.tasks)
        if any(task not in self.populations or not len(self.populations[task]) for task in required):
            raise DataProviderError("a curriculum task has no admitted population")

    def __len__(self) -> int:
        return (
            self.curriculum.total_updates - self.start_update
        ) * self.accumulation_steps

    def __iter__(self) -> Iterator[list[CurriculumIndex]]:
        task_microbatches = {task: 0 for task in self.curriculum.tasks}
        for previous in range(self.start_update):
            task_microbatches[self.curriculum.task_at(previous).name] += (
                self.accumulation_steps
            )
        order_cache: dict[tuple[str, int], np.ndarray] = {}
        for update in range(self.start_update, self.curriculum.total_updates):
            task = self.curriculum.task_at(update).name
            population = self.populations[task]
            batches_per_epoch = int(math.ceil(len(population) / self.micro_batch_size))
            for microbatch in range(self.accumulation_steps):
                draw = task_microbatches[task]
                task_microbatches[task] += 1
                epoch, batch_in_epoch = divmod(draw, batches_per_epoch)
                key = (task, epoch)
                order = order_cache.get(key)
                if order is None:
                    digest = hashlib.sha256(
                        f"{self.seed}\0{task}\0{epoch}".encode("utf-8")
                    ).digest()
                    generator = np.random.Generator(
                        np.random.PCG64(int.from_bytes(digest[:8], "little"))
                    )
                    order = generator.permutation(len(population))
                    order_cache[key] = order
                start = batch_in_epoch * self.micro_batch_size
                stop = min(start + self.micro_batch_size, len(population))
                chosen = population[order[start:stop]]
                yield [
                    CurriculumIndex(task, int(value), epoch, update, microbatch)
                    for value in chosen
                ]


@dataclass(frozen=True)
class CurriculumCollator:
    pad_token_id: int
    sentinel_token_ids: tuple[int, ...]
    eos_token_id: int
    seed: int

    def __call__(self, rows: Sequence[TaggedSample]) -> dict[str, Any]:
        if not rows:
            raise DataProviderError("cannot collate an empty microbatch")
        signatures = {
            (row.index.task, row.index.epoch, row.index.update, row.index.microbatch)
            for row in rows
        }
        if len(signatures) != 1:
            raise DataProviderError("one microbatch mixed curriculum coordinates")
        task, _, update, microbatch = next(iter(signatures))
        payloads = tuple(row.payload for row in rows)
        if task in {"M", "MG", "SYN"}:
            view = {"M": "P2-M", "MG": "P2-MG", "SYN": "P1-SYN"}[task]
            batch = collate_molecular_denoising_samples(
                payloads,
                view=view,
                pad_token_id=self.pad_token_id,
                sentinel_token_ids=self.sentinel_token_ids,
                eos_token_id=self.eos_token_id,
                global_seed=self.seed,
            )
        elif task == "CAP":
            batch = collate_phase2_cap_samples(
                payloads,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        elif task == "T2M":
            batch = collate_phase2_t2m_samples(
                payloads,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
        elif task == "TXT":
            batch = PF10TextDenoisingCollator(
                sentinel_token_ids=self.sentinel_token_ids,
                eos_token_id=self.eos_token_id,
                global_seed=self.seed,
            )(payloads)
        else:
            raise DataProviderError(f"unknown task: {task}")
        batch["curriculum_task"] = task
        batch["curriculum_update"] = update
        batch["curriculum_microbatch"] = microbatch
        return batch


class CurriculumDataLoaderProvider:
    """Return prefetched microbatches to the phase runner."""

    def __init__(
        self,
        *,
        phase: int,
        total_updates: int,
        pcqm_cache: str | Path,
        pubchem_cache: str | Path,
        paired_text_cache: str | Path,
        pubmed_cache: str | Path,
        pad_token_id: int,
        sentinel_token_ids: Sequence[int],
        eos_token_id: int,
        runtime: TrainingRuntimeConfig,
        populations: Mapping[str, Sequence[int]] | None = None,
        start_update: int = 0,
    ) -> None:
        self.runtime = runtime
        self.dataset = CurriculumDataset(
            pcqm_cache=pcqm_cache,
            pubchem_cache=pubchem_cache,
            paired_text_cache=paired_text_cache,
            pubmed_cache=pubmed_cache,
        )
        sizes = self.dataset.population_sizes()
        resolved = (
            {task: range(size) for task, size in sizes.items()}
            if populations is None
            else dict(populations)
        )
        sampler = CurriculumBatchSampler(
            phase=phase,
            total_updates=total_updates,
            populations=resolved,
            micro_batch_size=runtime.micro_batch_size,
            accumulation_steps=runtime.gradient_accumulation_steps,
            seed=runtime.seed,
            start_update=start_update,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_sampler": sampler,
            "collate_fn": CurriculumCollator(
                int(pad_token_id),
                tuple(int(value) for value in sentinel_token_ids),
                int(eos_token_id),
                runtime.seed,
            ),
            "num_workers": runtime.num_workers,
            "pin_memory": runtime.pin_memory,
        }
        if runtime.num_workers:
            loader_kwargs.update(
                {
                    "prefetch_factor": runtime.prefetch_factor,
                    "persistent_workers": runtime.persistent_workers,
                    "multiprocessing_context": "spawn",
                }
            )
        self.loader = DataLoader(**loader_kwargs)
        self.iterator = iter(self.loader)

    def __call__(self, task: TaskSpec, update: int) -> tuple[Mapping[str, Any], ...]:
        batches = tuple(
            next(self.iterator)
            for _ in range(self.runtime.gradient_accumulation_steps)
        )
        for microbatch, batch in enumerate(batches):
            if (
                batch.get("curriculum_task") != task.name
                or int(batch.get("curriculum_update", -1)) != update
                or int(batch.get("curriculum_microbatch", -1)) != microbatch
            ):
                raise DataProviderError("prefetched batch differs from runner schedule")
        return batches

    def close(self) -> None:
        shutdown = getattr(self.iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        self.dataset.close()


__all__ = [
    "CurriculumBatchSampler",
    "CurriculumCollator",
    "CurriculumDataLoaderProvider",
    "CurriculumDataset",
    "CurriculumIndex",
    "DataProviderError",
]
