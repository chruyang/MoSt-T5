"""Exercise the public processor against one admitted fragSMILES cache row."""

from __future__ import annotations

import argparse
from pathlib import Path

from most_t5_next.data import (
    MolecularInput,
    MoStT5Collator,
    MoStT5Processor,
    model_batch,
)
from most_t5_next.p2.fragsmiles_training_tensor_cache_v1 import (
    FragSmilesTrainingTensorCache,
)


class _Tokenizer:
    eos_token_id = 1

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [7, self.eos_token_id]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    cache = FragSmilesTrainingTensorCache(args.cache, verify_hashes=False)
    try:
        molecule = MolecularInput.from_cache_record(cache[0])
        example = MoStT5Processor(_Tokenizer()).molecule(
            molecule, use_geometry=False
        )
        batch = model_batch(MoStT5Collator(pad_token_id=0)([example]))
        print(
            {
                "tokens": tuple(batch["input_ids"].shape),
                "atoms": tuple(batch["e3fp_ids"].shape),
                "fragments": tuple(batch["fragment_mask"].shape),
                "endpoints": tuple(batch["endpoint_mask"].shape),
                "all_minus_one": bool(batch["e3fp_ids"].eq(-1).all()),
                "labels": "labels" in batch,
            }
        )
    finally:
        cache.close()


if __name__ == "__main__":
    main()
