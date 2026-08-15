"""Materialize the Phase-II enriched-text axis for CAP/T2M pretraining.

The molecular tensor cache intentionally contains no strings.  This companion
cache follows the exact same frozen membership order and stores complete T5
enriched-description IDs as one flat int32 array plus int64 offsets.  It never pads,
truncates, corrupts, or chooses a task view; those remain online operations.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np


SCHEMA_VERSION = "most-t5-p2/phase2-paired-text-cache/v2-enriched-description"
MEMBER_PREFIX = "pubchem_cid:"
TEXT_FIELD = "enriched_description"


class Phase2PairedTextCacheError(RuntimeError):
    pass


def _member_cids(path: Path) -> Iterator[int]:
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            member_id = row.get("member_id") if isinstance(row, dict) else None
            if not isinstance(member_id, str) or not member_id.startswith(MEMBER_PREFIX):
                raise Phase2PairedTextCacheError(f"invalid member at line {line_number}")
            suffix = member_id[len(MEMBER_PREFIX) :]
            if not suffix.isdigit():
                raise Phase2PairedTextCacheError("member CID is not decimal")
            cid = int(suffix)
            if cid in seen or not 0 <= cid < 2**31:
                raise Phase2PairedTextCacheError("member CID repeats or exceeds int32")
            seen.add(cid)
            yield cid


def _default_tokenizer(tokenizer_root: Path):
    try:
        from transformers import T5Tokenizer
    except ImportError as exc:
        raise Phase2PairedTextCacheError("transformers is required") from exc
    tokenizer = T5Tokenizer.from_pretrained(
        tokenizer_root / "tokenizer_snapshot", local_files_only=True, legacy=True
    )

    def encode_batch(texts: Sequence[str]) -> list[list[int]]:
        rows = tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        return [[int(value) for value in row] for row in rows]

    return encode_batch, int(tokenizer.eos_token_id), int(tokenizer.pad_token_id), len(tokenizer)


def _percentiles(lengths: np.ndarray) -> dict[str, int | float]:
    return {
        "minimum": int(lengths.min()),
        "maximum": int(lengths.max()),
        "mean": float(lengths.mean()),
        **{
            name: int(np.percentile(lengths, value, method="nearest"))
            for name, value in (("p50", 50), ("p95", 95), ("p99", 99))
        },
        "over_512": int(np.count_nonzero(lengths > 512)),
    }


def run(
    args: argparse.Namespace,
    *,
    tokenizer_factory: Callable[[Path], tuple[Callable[[Sequence[str]], list[list[int]]], int, int, int]] | None = None,
) -> dict[str, object]:
    import lmdb

    if args.output_dir.exists():
        raise Phase2PairedTextCacheError("paired-text output already exists")
    staging = args.output_dir.with_name(args.output_dir.name + ".staging")
    if staging.exists():
        raise Phase2PairedTextCacheError("paired-text staging already exists")
    if args.batch_size <= 0:
        raise Phase2PairedTextCacheError("batch size must be positive")
    factory = tokenizer_factory or _default_tokenizer
    encode_batch, eos_id, pad_id, vocab_size = factory(args.tokenizer_root)
    staging.mkdir(parents=True)
    environment = lmdb.open(
        str(args.source_lmdb), subdir=False, readonly=True, lock=False,
        readahead=False, meminit=False, max_readers=8,
    )
    offsets = [0]
    cids: list[int] = []
    lengths: list[int] = []
    token_count = 0
    token_path = staging / "text_input_ids.bin"
    try:
        with environment.begin(write=False) as transaction, token_path.open("wb") as output:
            pending_cids: list[int] = []
            pending_texts: list[str] = []

            def flush() -> None:
                nonlocal token_count
                if not pending_cids:
                    return
                encoded = encode_batch(pending_texts)
                if len(encoded) != len(pending_cids):
                    raise Phase2PairedTextCacheError("tokenizer batch size changed")
                for cid, ids in zip(pending_cids, encoded):
                    if not ids or ids[-1] != eos_id:
                        raise Phase2PairedTextCacheError(f"{TEXT_FIELD} lacks EOS: {cid}")
                    np.asarray(ids, dtype="<i4").tofile(output)
                    cids.append(cid)
                    lengths.append(len(ids))
                    token_count += len(ids)
                    offsets.append(token_count)
                pending_cids.clear()
                pending_texts.clear()

            for cid in _member_cids(args.membership):
                raw = transaction.get(str(cid).encode("ascii"))
                if raw is None:
                    raise Phase2PairedTextCacheError(f"Phase-II CID is absent: {cid}")
                payload = pickle.loads(raw)
                if not isinstance(payload, dict) or str(payload.get("cid")) != str(cid):
                    raise Phase2PairedTextCacheError(f"payload CID differs: {cid}")
                text = payload.get(TEXT_FIELD)
                if not isinstance(text, str) or not text.strip():
                    raise Phase2PairedTextCacheError(f"{TEXT_FIELD} is absent: {cid}")
                pending_cids.append(cid)
                pending_texts.append(text.strip())
                if len(pending_cids) >= args.batch_size:
                    flush()
                if args.max_records is not None and len(cids) + len(pending_cids) >= args.max_records:
                    break
            flush()
    finally:
        environment.close()
    if not cids:
        raise Phase2PairedTextCacheError("paired-text cache would be empty")
    if args.max_records is None and len(cids) != args.expected_records:
        raise Phase2PairedTextCacheError("paired-text record count differs")
    np.asarray(offsets, dtype="<i8").tofile(staging / "text_offsets.bin")
    np.asarray(cids, dtype="<i4").tofile(staging / "record_cid.bin")
    length_array = np.asarray(lengths, dtype=np.int64)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "training_admission": True,
        "source": {
            "lmdb": str(args.source_lmdb.resolve()),
            "membership": str(args.membership.resolve()),
            "text_field": TEXT_FIELD,
        },
        "counts": {"records": len(cids), "text_tokens": token_count},
        "lengths": _percentiles(length_array),
        "tokenizer": {
            "root": str(args.tokenizer_root.resolve()),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        },
        "arrays": {
            "text_input_ids": {"file": token_path.name, "dtype": "<i4", "shape": [token_count], "bytes": token_path.stat().st_size},
            "text_offsets": {"file": "text_offsets.bin", "dtype": "<i8", "shape": [len(offsets)], "bytes": (staging / "text_offsets.bin").stat().st_size},
            "record_cid": {"file": "record_cid.bin", "dtype": "<i4", "shape": [len(cids)], "bytes": (staging / "record_cid.bin").stat().st_size},
        },
        "online_boundary": {
            "complete_text_retained": True,
            "truncation_cached": False,
            "padding_cached": False,
            "task_view_cached": False,
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    staging.rename(args.output_dir)
    return manifest


class Phase2PairedTextCache:
    def __init__(self, root: Path, *, molecular_cache_root: Path | None = None) -> None:
        self.root = Path(root)
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        except Exception as exc:
            raise Phase2PairedTextCacheError("paired-text manifest is unreadable") from exc
        if self.manifest.get("schema_version") != SCHEMA_VERSION or self.manifest.get("training_admission") is not True:
            raise Phase2PairedTextCacheError("paired-text cache is not admitted")
        arrays = self.manifest["arrays"]
        self._arrays: dict[str, np.ndarray] = {}
        for name, spec in arrays.items():
            path = self.root / spec["file"]
            shape = tuple(int(value) for value in spec["shape"])
            dtype = np.dtype(spec["dtype"])
            if not path.is_file() or path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
                raise Phase2PairedTextCacheError(f"paired-text array differs: {name}")
            self._arrays[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape)
        if molecular_cache_root is not None:
            molecular = json.loads((Path(molecular_cache_root) / "manifest.json").read_text("utf-8"))
            ordinal_spec = molecular["arrays"]["record_ordinal"]
            ordinals = np.memmap(
                Path(molecular_cache_root) / ordinal_spec["file"], mode="r",
                dtype=ordinal_spec["dtype"], shape=tuple(ordinal_spec["shape"]),
            )
            if len(ordinals) != len(self) or not np.array_equal(ordinals, self._arrays["record_cid"]):
                raise Phase2PairedTextCacheError("molecular and paired-text row axes differ")

    def __getstate__(self) -> dict[str, object]:
        return {"root": self.root}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__init__(Path(state["root"]))

    def __len__(self) -> int:
        return int(self.manifest["counts"]["records"])

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        offsets = self._arrays["text_offsets"]
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return int(self._arrays["record_cid"][index]), self._arrays["text_input_ids"][start:stop]

    def close(self) -> None:
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lmdb", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=296614)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-records", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(build_parser().parse_args(argv))
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "lengths": manifest["lengths"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Phase2PairedTextCache", "Phase2PairedTextCacheError", "run"]
