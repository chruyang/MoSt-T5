"""Read and corrupt a packed PF-10 text cache with the reference T5 objective.

The immutable cache stores raw 568-token blocks only.  Random span corruption
is generated online from ``(seed, epoch, split, block_index)`` so workers and
batch ordering cannot change a sample.  For the frozen 15 percent / mean-span-3
contract, every raw block becomes exactly 512 encoder tokens and 114 labels.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .fragsmiles_pretraining_collator_v1 import standard_t5_noise_mask
from .semantic_span_corruption_v1 import (
    SemanticUnit,
    apply_t5_semantic_span_corruption,
)


SCHEMA_VERSION = "most-t5-p2/pf10-text-denoising-cache/v1"
SPLITS = ("train", "dev")


class PF10TextDenoisingCacheError(RuntimeError):
    pass


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PF10TextDenoisingCacheError(f"{name} must be an integer")
    return value


def _sample_seed(global_seed: int, epoch: int, split: str, block_index: int) -> int:
    payload = f"{global_seed}\0{epoch}\0{split}\0{block_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class PackedTextSample:
    split: str
    block_index: int
    input_ids: np.ndarray
    epoch: int = 0


class PF10PackedTextCache(Dataset):
    """Read-only mmap view over one split of a frozen PF-10 text cache."""

    def __init__(self, root: str | Path, *, split: str) -> None:
        self.root = Path(root)
        if split not in SPLITS:
            raise PF10TextDenoisingCacheError("split must be train or dev")
        self.split = split
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        except Exception as exc:
            raise PF10TextDenoisingCacheError("text cache manifest is unreadable") from exc
        if (
            self.manifest.get("schema_version") != SCHEMA_VERSION
            or self.manifest.get("status") != "pass"
            or self.manifest.get("training_admission") is not True
        ):
            raise PF10TextDenoisingCacheError("text cache is not admitted")
        packing = self.manifest.get("packing")
        if not isinstance(packing, Mapping) or not isinstance(packing.get(split), Mapping):
            raise PF10TextDenoisingCacheError("text packing manifest is incomplete")
        self.raw_length = _plain_int(packing.get("raw_block_length"), "raw_block_length")
        self.block_count = _plain_int(packing[split].get("blocks"), f"{split}.blocks")
        if self.raw_length <= 0 or self.block_count < 0:
            raise PF10TextDenoisingCacheError("text cache dimensions are invalid")
        path = self.root / f"{split}_input_ids.bin"
        expected_bytes = self.block_count * self.raw_length * np.dtype("<i4").itemsize
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise PF10TextDenoisingCacheError("text cache binary size disagrees with manifest")
        self._ids = np.memmap(
            path,
            mode="r",
            dtype="<i4",
            shape=(self.block_count, self.raw_length),
        )

    def __getstate__(self) -> dict[str, object]:
        return {"root": self.root, "split": self.split}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__init__(Path(state["root"]), split=str(state["split"]))

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, block_index: int) -> PackedTextSample:
        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise IndexError("block index must be an integer")
        if block_index < 0:
            block_index += self.block_count
        if not 0 <= block_index < self.block_count:
            raise IndexError(block_index)
        return PackedTextSample(
            split=self.split,
            block_index=block_index,
            input_ids=np.asarray(self._ids[block_index], dtype=np.int64),
        )

    def close(self) -> None:
        mmap = getattr(self._ids, "_mmap", None)
        if mmap is not None:
            mmap.close()


class PF10PackedTextTrainingCorpus(Dataset):
    """Expose both legacy physical shards as one pretraining population."""

    def __init__(self, root: str | Path) -> None:
        self.shards = tuple(PF10PackedTextCache(root, split=split) for split in SPLITS)
        total = 0
        boundaries: list[int] = []
        for shard in self.shards:
            total += len(shard)
            boundaries.append(total)
        self._boundaries = tuple(boundaries)

    def __len__(self) -> int:
        return self._boundaries[-1]

    def __getitem__(self, index: int) -> PackedTextSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise IndexError("index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._boundaries, index)
        shard_start = 0 if shard_index == 0 else self._boundaries[shard_index - 1]
        return self.shards[shard_index][index - shard_start]

    def close(self) -> None:
        for shard in self.shards:
            shard.close()


def _noise_units(mask: Sequence[bool]) -> tuple[SemanticUnit, ...]:
    units: list[SemanticUnit] = []
    start = -1
    for position, selected in enumerate(mask):
        if selected and start < 0:
            start = position
        if start >= 0 and (not selected or position == len(mask) - 1):
            stop = position if not selected else position + 1
            units.append(SemanticUnit(start, stop, len(units), semantic_type="text"))
            start = -1
    return tuple(units)


@dataclass
class PF10TextDenoisingCollator:
    sentinel_token_ids: Sequence[int]
    eos_token_id: int
    global_seed: int
    epoch: int = 0
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    encoder_length: int = 512
    target_length: int = 114

    def __call__(self, samples: Sequence[PackedTextSample]) -> dict[str, torch.Tensor]:
        if not samples:
            raise PF10TextDenoisingCacheError("cannot collate an empty text batch")
        if len({sample.epoch for sample in samples}) != 1:
            raise PF10TextDenoisingCacheError("one text batch must contain one epoch")
        inputs: list[tuple[int, ...]] = []
        labels: list[tuple[int, ...]] = []
        indices: list[int] = []
        for sample in samples:
            source = tuple(int(value) for value in sample.input_ids.tolist())
            seed = _sample_seed(
                self.global_seed, sample.epoch, sample.split, sample.block_index
            )
            mask = standard_t5_noise_mask(
                len(source),
                noise_density=self.noise_density,
                mean_noise_span_length=self.mean_noise_span_length,
                seed=seed,
            )
            units = _noise_units(mask)
            corruption = apply_t5_semantic_span_corruption(
                source,
                units,
                sentinel_token_ids=self.sentinel_token_ids,
                eos_token_id=self.eos_token_id,
            )
            encoder = (*corruption.input_ids, int(self.eos_token_id))
            if len(encoder) != self.encoder_length or len(corruption.labels) != self.target_length:
                raise PF10TextDenoisingCacheError(
                    "online T5 corruption does not satisfy the frozen 568->512/114 contract"
                )
            inputs.append(encoder)
            labels.append(corruption.labels)
            indices.append(sample.block_index)
        input_tensor = torch.tensor(inputs, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "attention_mask": torch.ones_like(input_tensor, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "block_index": torch.tensor(indices, dtype=torch.long),
        }


__all__ = [
    "PF10PackedTextCache",
    "PF10PackedTextTrainingCorpus",
    "PF10TextDenoisingCacheError",
    "PF10TextDenoisingCollator",
    "PackedTextSample",
]
