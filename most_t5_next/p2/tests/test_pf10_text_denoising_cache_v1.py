from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from most_t5_next.p2.pf10_text_denoising_cache_v1 import (
    PF10PackedTextCache,
    PF10PackedTextTrainingCorpus,
    PF10TextDenoisingCacheError,
    PF10TextDenoisingCollator,
)


class PF10TextDenoisingCacheV1Tests(unittest.TestCase):
    def _cache(self, root: Path) -> Path:
        cache = root / "cache"
        cache.mkdir()
        train = np.arange(2 * 568, dtype="<i4").reshape(2, 568) + 10
        dev = np.arange(568, dtype="<i4").reshape(1, 568) + 20
        train.tofile(cache / "train_input_ids.bin")
        dev.tofile(cache / "dev_input_ids.bin")
        manifest = {
            "schema_version": "most-t5-p2/pf10-text-denoising-cache/v1",
            "status": "pass",
            "training_admission": True,
            "packing": {
                "raw_block_length": 568,
                "train": {"blocks": 2},
                "dev": {"blocks": 1},
            },
        }
        (cache / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        return cache

    def test_reference_lengths_and_determinism(self) -> None:
        (Path.cwd() / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as raw:
            dataset = PF10PackedTextCache(self._cache(Path(raw)), split="train")
            collator = PF10TextDenoisingCollator(
                sentinel_token_ids=tuple(range(50000, 49900, -1)),
                eos_token_id=1,
                global_seed=17,
            )
            first = collator([dataset[0], dataset[1]])
            second = collator([dataset[0], dataset[1]])
            self.assertEqual(tuple(first["input_ids"].shape), (2, 512))
            self.assertEqual(tuple(first["labels"].shape), (2, 114))
            self.assertTrue(first["input_ids"].equal(second["input_ids"]))
            self.assertEqual(first["input_ids"][0, -1].item(), 1)
            self.assertEqual(first["labels"][0, -1].item(), 1)

    def test_binary_size_is_fail_closed(self) -> None:
        (Path.cwd() / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as raw:
            cache = self._cache(Path(raw))
            with (cache / "train_input_ids.bin").open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(PF10TextDenoisingCacheError, "size"):
                PF10PackedTextCache(cache, split="train")

    def test_training_corpus_includes_both_physical_shards(self) -> None:
        (Path.cwd() / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as raw:
            dataset = PF10PackedTextTrainingCorpus(self._cache(Path(raw)))
            self.assertEqual(len(dataset), 3)
            self.assertEqual(
                [dataset[index].split for index in range(3)],
                ["train", "train", "dev"],
            )
            self.assertEqual(dataset[-1].split, "dev")
            dataset.close()


if __name__ == "__main__":
    unittest.main()
