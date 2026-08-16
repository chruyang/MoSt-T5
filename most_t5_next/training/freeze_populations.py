"""Freeze task populations for a finite two-phase pretraining budget."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import numpy as np

from most_t5_next.p2.build_phase2_paired_text_cache_v1 import Phase2PairedTextCache
from most_t5_next.p2.fragsmiles_pretraining_collator_v1 import (
    FragSmilesPretrainingCollatorError,
    corrupt_cached_fragsmiles_record,
)
from most_t5_next.p2.fragsmiles_training_tensor_cache_v1 import (
    CachedFragSmilesSample,
    FragSmilesTrainingTensorCache,
)
from most_t5_next.p2.pf10_text_denoising_cache_v1 import PF10PackedTextTrainingCorpus


SCHEMA_VERSION = "most-t5/pretraining-populations/v1"
_CACHE: FragSmilesTrainingTensorCache | None = None
_SENTINELS: tuple[int, ...] = ()
_EOS = -1
_VIEW = ""
_SEED = -1


class PopulationError(RuntimeError):
    pass


def required_max_epoch(
    population: int, *, task_updates: int, micro_batch_size: int, accumulation_steps: int
) -> int:
    if min(population, task_updates, micro_batch_size, accumulation_steps) <= 0:
        raise PopulationError("population epoch settings must be positive")
    examples = task_updates * micro_batch_size * accumulation_steps
    return (examples - 1) // population


def _init_worker(
    cache_root: str,
    sentinels: tuple[int, ...],
    eos: int,
    view: str,
    seed: int,
) -> None:
    global _CACHE, _SENTINELS, _EOS, _VIEW, _SEED
    _CACHE = FragSmilesTrainingTensorCache(cache_root, verify_hashes=False)
    _SENTINELS = sentinels
    _EOS = eos
    _VIEW = view
    _SEED = seed


def _check_row(item: tuple[int, tuple[int, ...]]) -> tuple[int, int, list[dict[str, object]]]:
    index, epochs = item
    if _CACHE is None:
        raise PopulationError("population worker has no cache")
    record = _CACHE[index]
    failures: list[dict[str, object]] = []
    for epoch in epochs:
        try:
            corrupt_cached_fragsmiles_record(
                CachedFragSmilesSample(record, epoch),
                view=_VIEW,
                sentinel_token_ids=_SENTINELS,
                eos_token_id=_EOS,
                global_seed=_SEED,
                encoder_cap=512,
                target_cap=114,
            )
        except FragSmilesPretrainingCollatorError as exc:
            failures.append({"epoch": epoch, "reason": str(exc)})
    return index, int(record.ordinal), failures


def _scan_component(
    indices: Sequence[int],
    *,
    cache_root: Path,
    epochs: tuple[int, ...],
    sentinels: tuple[int, ...],
    eos: int,
    view: str,
    seed: int,
    workers: int,
    chunksize: int,
) -> tuple[list[int], list[dict[str, object]]]:
    kept: list[int] = []
    excluded: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_init_worker,
        initargs=(str(cache_root), sentinels, eos, view, seed),
    ) as pool:
        for index, ordinal, failures in pool.imap(
            _check_row, ((int(index), epochs) for index in indices), chunksize=chunksize
        ):
            if failures:
                excluded.append(
                    {"cache_index": index, "record_ordinal": ordinal, "failures": failures}
                )
            else:
                kept.append(index)
    return kept, excluded


def molecular_population(
    components: Sequence[tuple[str, Path, int, int]],
    *,
    task: str,
    view: str,
    task_updates: int,
    micro_batch_size: int,
    accumulation_steps: int,
    sentinels: tuple[int, ...],
    eos: int,
    seed: int,
    workers: int,
    chunksize: int,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    survivors = {
        name: np.arange(size, dtype=np.int64) for name, _root, _shift, size in components
    }
    exclusions: list[dict[str, object]] = []
    tested_through = -1
    while True:
        population = sum(len(values) for values in survivors.values())
        required = required_max_epoch(
            population,
            task_updates=task_updates,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
        )
        if required <= tested_through:
            break
        epochs = tuple(range(tested_through + 1, required + 1))
        for name, root, shift, _size in components:
            kept, failed = _scan_component(
                survivors[name],
                cache_root=root,
                epochs=epochs,
                sentinels=sentinels,
                eos=eos,
                view=view,
                seed=seed,
                workers=workers,
                chunksize=chunksize,
            )
            survivors[name] = np.asarray(kept, dtype=np.int64)
            for row in failed:
                row.update(
                    {
                        "task": task,
                        "cache": name,
                        "global_index": int(row["cache_index"]) + shift,
                        "action": "exclude_task_view_keep_source_record",
                    }
                )
            exclusions.extend(failed)
        tested_through = required
    arrays = [survivors[name] + shift for name, _root, shift, _size in components]
    return np.concatenate(arrays).astype("<i4"), exclusions, tested_through


def freeze_populations(
    *,
    pcqm_cache: Path,
    pubchem_cache: Path,
    paired_text_cache: Path,
    pubmed_cache: Path,
    output_dir: Path,
    sentinels: tuple[int, ...],
    eos: int,
    seed: int,
    phase_one_updates: int,
    phase_two_updates: int,
    micro_batch_size: int,
    accumulation_steps: int,
    workers: int,
    chunksize: int = 64,
) -> dict[str, object]:
    staging = output_dir.with_name(output_dir.name + ".staging")
    if output_dir.exists() or staging.exists():
        raise PopulationError("population output already exists")
    pcqm = FragSmilesTrainingTensorCache(pcqm_cache, verify_hashes=False)
    pubchem = FragSmilesTrainingTensorCache(pubchem_cache, verify_hashes=False)
    paired = Phase2PairedTextCache(paired_text_cache, molecular_cache_root=pubchem_cache)
    pubmed = PF10PackedTextTrainingCorpus(pubmed_cache)
    try:
        pcqm_size, pubchem_size, txt_size = len(pcqm), len(pubchem), len(pubmed)
        union = (
            ("pcqm", pcqm_cache, 0, pcqm_size),
            ("pubchem", pubchem_cache, pcqm_size, pubchem_size),
        )
        task_updates_one = phase_one_updates // 2
        task_updates_two = phase_two_updates // 4
        m, m_excluded, m_epoch = molecular_population(
            union,
            task="M",
            view="P2-M",
            task_updates=task_updates_one,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
            sentinels=sentinels,
            eos=eos,
            seed=seed,
            workers=workers,
            chunksize=chunksize,
        )
        mg, mg_excluded, mg_epoch = molecular_population(
            (("pcqm", pcqm_cache, 0, pcqm_size),),
            task="MG",
            view="P2-MG",
            task_updates=task_updates_one,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
            sentinels=sentinels,
            eos=eos,
            seed=seed,
            workers=workers,
            chunksize=chunksize,
        )
        syn, syn_excluded, syn_epoch = molecular_population(
            union,
            task="SYN",
            view="P1-SYN",
            task_updates=task_updates_two,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
            sentinels=sentinels,
            eos=eos,
            seed=seed,
            workers=workers,
            chunksize=chunksize,
        )
        supervised: list[int] = []
        supervised_excluded: list[dict[str, object]] = []
        for index in range(pubchem_size):
            record = pubchem[index]
            cid, _text = paired[index]
            if cid != record.ordinal:
                raise PopulationError("paired-text identity differs from PubChem cache")
            if len(record.input_ids) <= 512:
                supervised.append(index)
            else:
                supervised_excluded.append(
                    {
                        "task": "CAP,T2M",
                        "cache": "pubchem",
                        "cache_index": index,
                        "record_ordinal": int(record.ordinal),
                        "action": "exclude_structural_task_view_keep_source_record",
                        "molecule_length": len(record.input_ids),
                    }
                )
    finally:
        pcqm.close()
        pubchem.close()
        paired.close()
        pubmed.close()
    arrays = {
        "M": m,
        "MG": mg,
        "SYN": syn,
        "CAP": np.asarray(supervised, dtype="<i4"),
        "T2M": np.asarray(supervised, dtype="<i4"),
        "TXT": np.arange(txt_size, dtype="<i4"),
    }
    staging.mkdir(parents=True)
    descriptors: dict[str, object] = {}
    for task, values in arrays.items():
        filename = f"{task}.cache_index.bin"
        values.tofile(staging / filename)
        descriptors[task] = {"file": filename, "dtype": "<i4", "shape": [len(values)]}
    exclusions = [*m_excluded, *mg_excluded, *syn_excluded, *supervised_excluded]
    with (staging / "length_action_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in exclusions:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "seed": seed,
        "phase_updates": {"phase_one": phase_one_updates, "phase_two": phase_two_updates},
        "batching": {
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": micro_batch_size * accumulation_steps,
            "sample_before_microbatch_split": True,
        },
        "arrays": descriptors,
        "counts": {task: len(values) for task, values in arrays.items()},
        "molecular_max_epoch_scanned": {"M": m_epoch, "MG": mg_epoch, "SYN": syn_epoch},
        "excluded_task_views": len(exclusions),
        "contracts": {
            "source_records_retained": True,
            "text_overlength_action": "right_truncate_keep_eos",
            "structural_overlength_action": "exclude_task_view",
            "pretraining_validation_split": False,
            "length_action_ledger": "length_action_ledger.jsonl",
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output_dir)
    return manifest


__all__ = [
    "PopulationError",
    "SCHEMA_VERSION",
    "freeze_populations",
    "molecular_population",
    "required_max_epoch",
]
