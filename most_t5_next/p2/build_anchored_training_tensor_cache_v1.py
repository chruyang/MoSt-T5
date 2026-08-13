"""Compile the frozen anchored motif surface into the shared mmap training ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from most_t5_next.p1.build_pf1_paired_release_v1 import PF1PairedReleaseReader

from .anchored_training_record_v1 import AnchoredTrainingRecordReader
from .pf10_training_tensor_cache_v1 import build_pf10_training_tensor_cache


SCHEMA_VERSION = "most-t5-p2/anchored-training-tensor-cache-build/v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_anchored_training_tensor_cache(
    *,
    paired_release: Path,
    surface_records: Path,
    macro_registry: Path,
    tokenizer_manifest: Path,
    donor_atom_maps: Path,
    morgan_overlay: Path,
    output_dir: Path,
    release_id: str,
    max_train_records: int | None = None,
    max_dev_records: int | None = None,
    decode_workers: int = 0,
    decode_max_pending: int = 128,
) -> dict[str, object]:
    paths = {
        "surface_records": Path(surface_records).expanduser().resolve(),
        "macro_registry": Path(macro_registry).expanduser().resolve(),
        "tokenizer_manifest": Path(tokenizer_manifest).expanduser().resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("one anchored training-cache input is absent")
    geometry_reader = PF1PairedReleaseReader(Path(paired_release))
    reader = AnchoredTrainingRecordReader(
        surface_records=paths["surface_records"],
        geometry_reader=geometry_reader,
        macro_registry=paths["macro_registry"],
        tokenizer_manifest=paths["tokenizer_manifest"],
        release_id=release_id,
        donor_atom_maps=Path(donor_atom_maps),
    )

    def reader_factory(_paired_release: Path) -> AnchoredTrainingRecordReader:
        return reader

    return build_pf10_training_tensor_cache(
        paired_release=Path(paired_release),
        morgan_overlay=Path(morgan_overlay),
        output_dir=Path(output_dir),
        max_train_records=max_train_records,
        max_dev_records=max_dev_records,
        decode_workers=decode_workers,
        decode_max_pending=decode_max_pending,
        reader_factory=reader_factory,
        donor_atom_maps_path=Path(donor_atom_maps),
        source_extensions={
            "schema_version": SCHEMA_VERSION,
            "surface_records_sha256": _sha256_file(paths["surface_records"]),
            "macro_registry_sha256": _sha256_file(paths["macro_registry"]),
            "tokenizer_manifest_sha256": _sha256_file(paths["tokenizer_manifest"]),
            "anchored_release_id": release_id,
            "graphports_token_axis_exposed": False,
            "epoch_corruption_cached": False,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-release", type=Path, required=True)
    parser.add_argument("--surface-records", type=Path, required=True)
    parser.add_argument("--macro-registry", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--donor-atom-maps", type=Path, required=True)
    parser.add_argument("--morgan-overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-dev-records", type=int)
    parser.add_argument("--decode-workers", type=int, default=0)
    parser.add_argument("--decode-max-pending", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = build_anchored_training_tensor_cache(
        paired_release=args.paired_release,
        surface_records=args.surface_records,
        macro_registry=args.macro_registry,
        tokenizer_manifest=args.tokenizer_manifest,
        donor_atom_maps=args.donor_atom_maps,
        morgan_overlay=args.morgan_overlay,
        output_dir=args.output_dir,
        release_id=args.release_id,
        max_train_records=args.max_train_records,
        max_dev_records=args.max_dev_records,
        decode_workers=args.decode_workers,
        decode_max_pending=args.decode_max_pending,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA_VERSION", "build_anchored_training_tensor_cache"]
