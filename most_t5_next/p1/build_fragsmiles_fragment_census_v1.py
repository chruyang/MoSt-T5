"""Build a training-surface fragment census for canonical fragSMILES identities.

The authoritative chemistry path remains the compact stereo codec plus the
whole-molecule lossless fallback.  Only compact records contribute fragment
macro observations.  A record routed to the whole-molecule fallback therefore
contributes zero macro observations rather than an identity from a codec path
that will not be used by the model.

This is an offline CPU builder.  It keeps random corruption, padding, and all
training-time operations out of the artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
import json
from itertools import islice
import multiprocessing
from pathlib import Path
import signal
import time
from typing import Iterator

from rdkit import Chem, rdBase

from most_t5_next.p1.audit_fragsmiles_compact_stereo_domains_v1 import _iter_source
from most_t5_next.p1.fragsmiles_lossless_fallback_v1 import (
    encode_lossless_fallback,
    encode_main_or_fallback,
)
from most_t5_next.p1.build_phase2_anchored_pure_motif_census_v1 import (
    ACKNOWLEDGEMENT as TRUSTED_PICKLE_ACKNOWLEDGEMENT,
    _payload_smiles,
)


SCHEMA_VERSION = "most-t5-next/fragsmiles-fragment-census/v1"


class FragSmilesFragmentCensusError(RuntimeError):
    """The source or emitted census violates its frozen contract."""


class TrainingProjectionDomainError(FragSmilesFragmentCensusError):
    """The molecule cannot enter the heavy-atom compact training surface."""


def _iter_legacy_lmdb(path: Path) -> Iterator[tuple[int, str | None, str | None]]:
    """Read the hash-locked Phase-II source in ascending numeric-CID order.

    LMDB cursor order is byte-lexicographic (``1, 10, 100, ...``), whereas the
    authoritative Phase-II extraction orders payload keys numerically.  Shard
    ranges must therefore be defined over the numeric order, not over storage
    traversal order.  The only tolerated non-payload key is the legacy
    ``__len__`` metadata key, when present and consistent.
    """

    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise RuntimeError("legacy-lmdb input requires python-lmdb") from exc
    environment = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=16,
    )
    try:
        with environment.begin(write=False, buffers=True) as transaction:
            payload_keys: list[tuple[int, bytes]] = []
            declared_length: int | None = None
            for raw_key, raw_value in transaction.cursor():
                key_bytes = bytes(raw_key)
                if key_bytes == b"__len__":
                    try:
                        declared_length = int(bytes(raw_value).decode("ascii"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise FragSmilesFragmentCensusError(
                            "legacy LMDB __len__ metadata is invalid"
                        ) from exc
                    continue
                try:
                    key_text = key_bytes.decode("ascii")
                    numeric_key = int(key_text)
                except (UnicodeDecodeError, ValueError) as exc:
                    raise FragSmilesFragmentCensusError(
                        "legacy LMDB contains an undeclared metadata key"
                    ) from exc
                if numeric_key <= 0 or str(numeric_key) != key_text:
                    raise FragSmilesFragmentCensusError(
                        "legacy LMDB contains a noncanonical payload key"
                    )
                payload_keys.append((numeric_key, key_bytes))
            payload_keys.sort(key=lambda row: row[0])
            if len(payload_keys) != len({row[0] for row in payload_keys}):
                raise FragSmilesFragmentCensusError(
                    "legacy LMDB contains duplicate numeric payload keys"
                )
            if declared_length is not None and declared_length != len(payload_keys):
                raise FragSmilesFragmentCensusError(
                    "legacy LMDB __len__ differs from payload count"
                )
            for source_index, (_numeric_key, key_bytes) in enumerate(payload_keys):
                raw_value = transaction.get(key_bytes)
                if raw_value is None:
                    raise FragSmilesFragmentCensusError(
                        "legacy LMDB payload disappeared during read transaction"
                    )
                try:
                    key = key_bytes.decode("ascii")
                    yield source_index, _payload_smiles(key, bytes(raw_value)), None
                except Exception as exc:
                    yield source_index, None, f"legacy LMDB payload: {type(exc).__name__}: {exc}"
    finally:
        environment.close()


def _iter_census_source(
    path: Path, input_format: str, field: str
) -> Iterator[tuple[int, str | None, str | None]]:
    if input_format == "legacy-lmdb":
        yield from _iter_legacy_lmdb(path)
    else:
        yield from _iter_source(path, input_format, field)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fragment_identity(fragment_smiles: str) -> str:
    """Return a traversal-invariant, stereo-free fragment identity."""

    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise FragSmilesFragmentCensusError("fragment identity cannot be parsed")
    first = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    reparsed = Chem.MolFromSmiles(first)
    if reparsed is None:
        raise FragSmilesFragmentCensusError("canonical fragment cannot be reparsed")
    second = Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=False)
    if first != second:
        raise FragSmilesFragmentCensusError("fragment identity is not a fixed point")
    return second


def _project_training_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Apply the same defining-stereo hydrogen policy as PCQM production."""

    parameters = Chem.RemoveHsParameters()
    if not hasattr(parameters, "removeDefiningBondStereo"):
        raise FragSmilesFragmentCensusError(
            "RDKit lacks RemoveHsParameters.removeDefiningBondStereo"
        )
    parameters.removeDefiningBondStereo = True
    projected = Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)
    Chem.SanitizeMol(projected)
    Chem.AssignStereochemistry(projected, cleanIt=True, force=True)
    if projected.GetNumAtoms() == 0 or any(
        atom.GetAtomicNum() <= 1 for atom in projected.GetAtoms()
    ):
        raise TrainingProjectionDomainError(
            "training projection is not heavy-atom-only"
        )
    return projected


def _load_membership(path: Path | None) -> tuple[bytearray | None, int]:
    if path is None:
        return None, 0
    indices: list[int] = []
    prefix = "ogb_pcqm4mv2_train_row_index:"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            member_id = row.get("member_id")
            if not isinstance(member_id, str) or not member_id.startswith(prefix):
                raise FragSmilesFragmentCensusError(
                    f"invalid PCQM membership row {line_number}"
                )
            suffix = member_id[len(prefix) :]
            if not suffix.isdigit():
                raise FragSmilesFragmentCensusError(
                    f"invalid PCQM row index at membership row {line_number}"
                )
            indices.append(int(suffix))
    if not indices or len(indices) != len(set(indices)):
        raise FragSmilesFragmentCensusError("membership is empty or duplicated")
    mask = bytearray(max(indices) + 1)
    for index in indices:
        mask[index] = 1
    return mask, len(indices)


def _selected_source(
    *,
    input_path: Path,
    input_format: str,
    smiles_field: str,
    membership_mask: bytearray | None,
    start_record: int,
    max_records: int | None,
) -> Iterator[tuple[int, int, str | None, str | None]]:
    selected = 0
    emitted = 0
    for source_index, smiles, error in _iter_census_source(
        input_path, input_format, smiles_field
    ):
        if membership_mask is not None:
            if source_index >= len(membership_mask) or not membership_mask[source_index]:
                continue
        if selected < start_record:
            selected += 1
            continue
        yield selected, source_index, smiles, error
        selected += 1
        emitted += 1
        if max_records is not None and emitted >= max_records:
            return


def _census_one(
    task: tuple[int, int, str | None, str | None, str, int | None]
) -> dict[str, object]:
    selection_index, source_index, smiles, source_error, root_text, timeout = task
    if source_error is not None:
        return {
            "selection_index": selection_index,
            "source_index": source_index,
            "status": "reject",
            "error_type": "source_parse_failure",
            "error": source_error,
        }
    old_handler = None
    if timeout is not None and hasattr(signal, "SIGALRM"):
        def _timeout(_signum, _frame):
            raise TimeoutError(f"fragSMILES census exceeded {timeout} seconds")

        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(timeout)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FragSmilesFragmentCensusError("RDKit MolFromSmiles returned None")
        projection_mode = "heavy_atom_projection"
        try:
            projected = _project_training_hydrogens(mol)
        except TrainingProjectionDomainError:
            # Isotopic/charged explicit hydrogens are chemically meaningful and
            # must not be stripped.  The byte-token fallback preserves them and
            # deliberately leaves their E3FP row unset, while retaining E3FP
            # addressing for every heavy atom.
            encode_lossless_fallback(mol)
            routed = None
            mode = "whole_molecule_fallback"
            projection_mode = "original_molecule_lossless_fallback"
            fallback_reason_type = TrainingProjectionDomainError.__name__
        else:
            routed = encode_main_or_fallback(
                projected, chemicalgof_root=Path(root_text)
            )
            mode = routed.mode
            fallback_reason_type = routed.fallback_reason_type
        if mode == "compact":
            assert routed is not None
            assert routed.compact_surface is not None
            raw_identities = tuple(
                row.fragment_smiles
                for row in routed.compact_surface.connectivity_record.fragments
            )
            canonical_identities = tuple(
                canonical_fragment_identity(identity) for identity in raw_identities
            )
            eligible = tuple(
                raw == canonical
                for raw, canonical in zip(raw_identities, canonical_identities)
            )
            identities = raw_identities
            if not identities:
                raise FragSmilesFragmentCensusError("compact record has no fragments")
        else:
            identities = ()
            eligible = ()
        return {
            "selection_index": selection_index,
            "source_index": source_index,
            "status": "pass",
            "mode": mode,
            "projection_mode": projection_mode,
            "fragment_identities": identities,
            "fragment_macro_eligible": eligible,
            "fallback_reason_type": fallback_reason_type,
        }
    except Exception as exc:
        return {
            "selection_index": selection_index,
            "source_index": source_index,
            "status": "reject",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "smiles": smiles,
        }
    finally:
        if old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _ordered_results(
    tasks: Iterator[tuple[int, int, str | None, str | None, str, int | None]],
    *,
    workers: int,
    max_pending: int,
    start_index: int = 0,
) -> Iterator[dict[str, object]]:
    if workers == 1:
        for task in tasks:
            yield _census_one(task)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = {}
        ready: dict[int, dict[str, object]] = {}
        exhausted = False
        next_output = start_index
        while pending or ready or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    task = next(tasks)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(_census_one, task)
                pending[future] = task[0]
            if pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    result = future.result()
                    if result["selection_index"] != index:
                        raise FragSmilesFragmentCensusError("worker result index drift")
                    ready[index] = result
            while next_output in ready:
                yield ready.pop(next_output)
                next_output += 1
        if ready:
            raise FragSmilesFragmentCensusError("ordered worker output has a gap")


def run_census(
    *,
    input_path: Path,
    input_format: str,
    smiles_field: str,
    chemicalgof_root: Path,
    output_dir: Path,
    membership_path: Path | None = None,
    workers: int = 1,
    max_pending: int = 4,
    expected_records: int | None = None,
    start_record: int = 0,
    max_records: int | None = None,
    progress_every: int = 10000,
    record_timeout_seconds: int | None = 30,
    expected_source_sha256: str | None = None,
    trusted_pickle_acknowledgement: str | None = None,
) -> dict[str, object]:
    if workers <= 0 or max_pending < workers:
        raise ValueError("workers must be positive and max_pending >= workers")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if start_record < 0:
        raise ValueError("start_record must be non-negative")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    if input_format == "legacy-lmdb":
        if trusted_pickle_acknowledgement != TRUSTED_PICKLE_ACKNOWLEDGEMENT:
            raise FragSmilesFragmentCensusError(
                "trusted pickle acknowledgement is absent"
            )
        if (
            not isinstance(expected_source_sha256, str)
            or len(expected_source_sha256) != 64
            or _sha256_file(input_path) != expected_source_sha256
        ):
            raise FragSmilesFragmentCensusError("legacy LMDB SHA-256 differs")
        if membership_path is not None:
            raise FragSmilesFragmentCensusError(
                "PCQM membership cannot be applied to Phase-II LMDB"
            )
    output_dir.mkdir(parents=True)
    membership_mask, membership_count = _load_membership(membership_path)
    source = _selected_source(
        input_path=input_path,
        input_format=input_format,
        smiles_field=smiles_field,
        membership_mask=membership_mask,
        start_record=start_record,
        max_records=max_records,
    )
    root_text = str(chemicalgof_root.resolve())
    tasks = (
        (selection, source_index, smiles, error, root_text, record_timeout_seconds)
        for selection, source_index, smiles, error in source
    )
    occurrence_counts: Counter[str] = Counter()
    molecule_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    projection_modes: Counter[str] = Counter()
    ineligible_fragment_surfaces = 0
    processed = rejected = 0
    start = time.time()
    cache_path = output_dir / "molecule_fragments.jsonl.gz"
    rejects_path = output_dir / "rejects.jsonl"
    with gzip.open(cache_path, "wt", encoding="utf-8", newline="\n") as cache, rejects_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rejects:
        for result in _ordered_results(
            tasks,
            workers=workers,
            max_pending=max_pending,
            start_index=start_record,
        ):
            if result["selection_index"] != start_record + processed:
                raise FragSmilesFragmentCensusError("parent output order drift")
            processed += 1
            if result["status"] != "pass":
                rejected += 1
                rejects.write(json.dumps(result, sort_keys=True) + "\n")
                continue
            identities = tuple(result["fragment_identities"])
            eligible = tuple(result["fragment_macro_eligible"])
            mode = str(result["mode"])
            if len(identities) != len(eligible):
                raise FragSmilesFragmentCensusError("fragment eligibility length drifted")
            mode_counts[mode] += 1
            projection_mode = str(result["projection_mode"])
            projection_modes[projection_mode] += 1
            eligible_identities = tuple(
                identity for identity, admitted in zip(identities, eligible) if admitted
            )
            ineligible_fragment_surfaces += len(identities) - len(eligible_identities)
            occurrence_counts.update(eligible_identities)
            molecule_counts.update(set(eligible_identities))
            reason = result.get("fallback_reason_type")
            if reason is not None:
                fallback_reasons[str(reason)] += 1
            cache.write(
                json.dumps(
                    {
                        "selection_index": result["selection_index"],
                        "source_index": result["source_index"],
                        "mode": mode,
                        "projection_mode": projection_mode,
                        "fragment_identities": identities,
                        "fragment_macro_eligible": eligible,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            if progress_every and processed % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "global_processed_stop": start_record + processed,
                            "compact": mode_counts["compact"],
                            "fallback": mode_counts["whole_molecule_fallback"],
                            "reject": rejected,
                            "wall_seconds": round(time.time() - start, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    required = expected_records
    if required is None and membership_path is not None and max_records is None:
        required = membership_count
    if required is not None and processed != required:
        raise FragSmilesFragmentCensusError(
            f"processed {processed} records, expected {required}"
        )
    registry_path = output_dir / "fragment_census.jsonl"
    ranking = sorted(
        occurrence_counts,
        key=lambda identity: (-occurrence_counts[identity], identity.encode("utf-8")),
    )
    with registry_path.open("w", encoding="utf-8", newline="\n") as handle:
        for rank, identity in enumerate(ranking):
            handle.write(
                json.dumps(
                    {
                        "rank": rank,
                        "fragment_identity": identity,
                        "fragment_identity_sha256": hashlib.sha256(
                            identity.encode("utf-8")
                        ).hexdigest(),
                        "occurrences": occurrence_counts[identity],
                        "molecule_occurrences": molecule_counts[identity],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    status = "pass" if rejected == 0 else "failed"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "training_admission": False,
        "source": {
            "path": str(input_path.resolve()),
            "format": input_format,
            "smiles_field": smiles_field,
            "bytes": input_path.stat().st_size,
            "sha256": (
                expected_source_sha256 if input_format == "legacy-lmdb" else None
            ),
        },
        "membership": (
            {
                "path": str(membership_path.resolve()),
                "bytes": membership_path.stat().st_size,
                "sha256": _sha256_file(membership_path),
                "members": membership_count,
            }
            if membership_path is not None
            else None
        ),
        "selected_record_range": {
            "start_inclusive": start_record,
            "stop_exclusive": start_record + processed,
        },
        "counts": {
            "processed_records": processed,
            "admitted_records": processed - rejected,
            "rejected_records": rejected,
            "modes": dict(sorted(mode_counts.items())),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "projection_modes": dict(sorted(projection_modes.items())),
            "fragment_occurrences": sum(occurrence_counts.values()),
            "unique_fragment_identities": len(occurrence_counts),
            "noncanonical_fragment_surfaces_forced_to_semantic_fallback": ineligible_fragment_surfaces,
        },
        "runtime": {
            "workers": workers,
            "max_pending": max_pending,
            "rdkit_version": rdBase.rdkitVersion,
            "wall_seconds": time.time() - start,
        },
        "contracts": {
            "compact_only_contributes_macro_counts": True,
            "whole_molecule_fallback_contributes_zero_macro_counts": True,
            "identity_is_rdkit_stereo_free_canonical_fixed_point": True,
            "compact_candidates_use_remove_defining_bond_stereo_projection": True,
            "compact_fragments_all_in_finite_chemical_lexer_domain": True,
            "explicit_isotopic_or_charged_hydrogens_use_lossless_original_surface": True,
            "molecule_cache_is_selection_ordered": True,
            "legacy_lmdb_payload_order_is_numeric_cid_ascending": (
                input_format == "legacy-lmdb"
            ),
            "validation_or_test_used_for_vocabulary": False,
        },
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in (registry_path, cache_path, rejects_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--format",
        required=True,
        choices=("jsonl", "parquet", "sdf", "sdf-tar", "legacy-lmdb"),
    )
    parser.add_argument("--smiles-field", default="smiles")
    parser.add_argument("--chemicalgof-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--membership")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pending", type=int, default=4)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--start-record", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--record-timeout-seconds", type=int, default=30)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--trusted-pickle-acknowledgement")
    args = parser.parse_args(argv)
    manifest = run_census(
        input_path=Path(args.input),
        input_format=args.format,
        smiles_field=args.smiles_field,
        chemicalgof_root=Path(args.chemicalgof_root),
        output_dir=Path(args.output_dir),
        membership_path=Path(args.membership) if args.membership else None,
        workers=args.workers,
        max_pending=args.max_pending,
        expected_records=args.expected_records,
        start_record=args.start_record,
        max_records=args.max_records,
        progress_every=args.progress_every,
        record_timeout_seconds=args.record_timeout_seconds,
        expected_source_sha256=args.expected_source_sha256,
        trusted_pickle_acknowledgement=args.trusted_pickle_acknowledgement,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FragSmilesFragmentCensusError",
    "SCHEMA_VERSION",
    "canonical_fragment_identity",
    "run_census",
]
