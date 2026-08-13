"""Bind frozen anchored motif text to existing validated atom geometry rows.

The anchored surface owns the model-facing token axis.  A historical
``ProductionMotifRecord`` is used only as the already validated source of the
same molecule's atom axis, E3FP rows and provenance; none of its GraphPorts
tokens enter the returned record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

from most_t5_next.p1.bound_record import Span
from most_t5_next.p1.production_bridge import ProductionMotifRecord
from most_t5_next.r1.tokenizer.anchored_motif_model_surface_v1 import (
    FALLBACK_MOTIF_SUFFIX,
    encode_frozen_phrases,
)


SCHEMA_VERSION = "most-t5-next/anchored-training-record/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnchoredTrainingRecordError(ValueError):
    """The frozen text surface and validated atom geometry do not align."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnchoredTokenizerBinding:
    tokenizer_contract_sha256: str
    tokenizer_snapshot_sha256: str
    vocab_size: int
    token_id_rows: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not SHA256_RE.fullmatch(self.tokenizer_contract_sha256) or not SHA256_RE.fullmatch(
            self.tokenizer_snapshot_sha256
        ):
            raise AnchoredTrainingRecordError("tokenizer hashes must be lower-case SHA-256")
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise AnchoredTrainingRecordError("vocab_size must be positive")
        tokens = [row[0] for row in self.token_id_rows]
        ids = [row[1] for row in self.token_id_rows]
        if (
            not self.token_id_rows
            or len(tokens) != len(set(tokens))
            or len(ids) != len(set(ids))
            or any(not isinstance(token, str) or not token for token in tokens)
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.vocab_size for value in ids)
        ):
            raise AnchoredTrainingRecordError("token ID registry is malformed")

    def token_to_id(self) -> dict[str, int]:
        return dict(self.token_id_rows)


@dataclass(frozen=True)
class AnchoredTrainingRecord:
    record_artifact_sha256: str
    record_id: str
    storage_key: str
    release_id: str
    geometry_record_content_sha256: str
    tokenizer_contract_sha256: str
    tokenizer_snapshot_sha256: str
    input_ids: tuple[int, ...]
    token_to_logical_motif: tuple[int, ...]
    token_role: tuple[str, ...]
    identity_spans: tuple[Span, ...]
    anchor_token_indices: tuple[tuple[int, ...], ...]
    logical_to_carrier: tuple[int, ...]
    exact_identity_sha256: tuple[str, ...]
    source_atom_count: int
    full_e3fp_ids: tuple[tuple[int, ...], ...]
    atom_valid_mask: tuple[bool, ...]
    model_to_source_atom_index: tuple[int, ...]
    atom_to_logical_motif: tuple[int, ...]
    atom_is_attachment: tuple[bool, ...]
    anchor_token_to_atom: tuple[int, ...]
    macro_used: tuple[bool, ...]

    def as_factorized_record(self) -> ProductionMotifRecord:
        """Expose the established collator shape without exposing GraphPorts text."""

        return ProductionMotifRecord(
            record_artifact_sha256=self.record_artifact_sha256,
            record_id=self.record_id,
            storage_key=self.storage_key,
            release_id=self.release_id,
            geometry_record_content_sha256=self.geometry_record_content_sha256,
            tokenizer_contract_sha256=self.tokenizer_contract_sha256,
            tokenizer_snapshot_sha256=self.tokenizer_snapshot_sha256,
            input_ids=self.input_ids,
            token_to_logical_motif=self.token_to_logical_motif,
            token_role=self.token_role,
            identity_spans=self.identity_spans,
            connection_token_indices=self.anchor_token_indices,
            logical_to_carrier=self.logical_to_carrier,
            exact_identity_sha256=self.exact_identity_sha256,
            source_atom_count=self.source_atom_count,
            full_e3fp_ids=self.full_e3fp_ids,
            atom_valid_mask=self.atom_valid_mask,
            model_to_source_atom_index=self.model_to_source_atom_index,
            atom_to_logical_motif=self.atom_to_logical_motif,
            atom_is_attachment=self.atom_is_attachment,
            connection_token_to_atom=self.anchor_token_to_atom,
        )


@dataclass(frozen=True)
class LoadedAnchoredTrainingRecord:
    """One training-facing row; the historical atom record stays out of the ABI."""

    selection_index: int
    split: str
    motif_record: ProductionMotifRecord


class AnchoredTrainingRecordReader:
    """Join one frozen anchored surface release to a validated geometry reader.

    The surface documents are small enough for the current PF1/PF10 gates to
    index by storage key once.  Geometry decoding can still use the base
    reader's bounded spawn pool; binding remains deterministic and ordered in
    the parent process.  No epoch mask or padded tensor is cached here.
    """

    def __init__(
        self,
        *,
        surface_records: Path,
        geometry_reader: Any,
        macro_registry: Path,
        tokenizer_manifest: Path,
        release_id: str,
        donor_atom_maps: Path | None = None,
    ) -> None:
        self.geometry_reader = geometry_reader
        self.release_id = release_id
        if not isinstance(release_id, str) or not release_id:
            raise AnchoredTrainingRecordError("anchored release_id cannot be empty")
        try:
            manifest = json.loads(Path(tokenizer_manifest).read_text("utf-8"))
            self.tokenizer = tokenizer_binding_from_candidate_manifest(manifest)
            with Path(macro_registry).open(encoding="utf-8") as handle:
                self.macro_rows = tuple(
                    json.loads(line) for line in handle if line.strip()
                )
        except (OSError, json.JSONDecodeError) as exc:
            raise AnchoredTrainingRecordError(
                "anchored tokenizer or macro registry is unreadable"
            ) from exc
        if not self.macro_rows:
            raise AnchoredTrainingRecordError("macro registry cannot be empty")

        surfaces: dict[str, Mapping[str, object]] = {}
        split_keys: dict[str, list[str]] = {"train": [], "dev": []}
        try:
            with Path(surface_records).open(encoding="utf-8") as handle:
                for expected_index, line in enumerate(handle):
                    row = json.loads(line)
                    key = row.get("storage_key") if isinstance(row, Mapping) else None
                    split = row.get("split") if isinstance(row, Mapping) else None
                    if (
                        not isinstance(key, str)
                        or not key
                        or split not in split_keys
                        or row.get("selection_index") != expected_index
                        or key in surfaces
                    ):
                        raise AnchoredTrainingRecordError(
                            "anchored surface order, split, or storage key is invalid"
                        )
                    surfaces[key] = row
                    split_keys[str(split)].append(key)
        except (OSError, json.JSONDecodeError) as exc:
            raise AnchoredTrainingRecordError(
                "anchored surface records are unreadable"
            ) from exc
        self._surfaces = surfaces
        self._split_keys = {key: tuple(value) for key, value in split_keys.items()}
        self.train_member_count = len(self._split_keys["train"])
        self.dev_member_count = len(self._split_keys["dev"])
        if (
            self.train_member_count != getattr(geometry_reader, "train_member_count", None)
            or self.dev_member_count != getattr(geometry_reader, "dev_member_count", None)
        ):
            raise AnchoredTrainingRecordError(
                "anchored and geometry split counts differ"
            )
        self._donor_rows: dict[str, tuple[Mapping[str, object], ...]] | None = None
        if donor_atom_maps is not None:
            from most_t5_next.r1.adapter.graphports_donor_atom_map_sidecar_v1 import (
                iter_release_rows,
            )

            grouped: dict[str, list[Mapping[str, object]]] = {"train": [], "dev": []}
            for expected_index, row in enumerate(iter_release_rows(Path(donor_atom_maps))):
                split = row.get("split")
                key = row.get("storage_key")
                surface = self._surfaces.get(str(key))
                if (
                    row.get("selection_index") != expected_index
                    or split not in grouped
                    or surface is None
                    or surface.get("selection_index") != expected_index
                    or surface.get("split") != split
                ):
                    raise AnchoredTrainingRecordError(
                        "donor atom maps differ from the anchored surface order"
                    )
                grouped[str(split)].append(row)
            if any(
                tuple(str(row["storage_key"]) for row in grouped[split])
                != self._split_keys[split]
                for split in ("train", "dev")
            ):
                raise AnchoredTrainingRecordError(
                    "donor atom maps do not cover the anchored split keys"
                )
            self._donor_rows = {
                split: tuple(rows) for split, rows in grouped.items()
            }

    def _bind(self, loaded: Any, *, split: str) -> LoadedAnchoredTrainingRecord:
        geometry = getattr(loaded, "motif_record", None)
        if not isinstance(geometry, ProductionMotifRecord):
            raise AnchoredTrainingRecordError(
                "geometry reader row lacks a production motif record"
            )
        surface = self._surfaces.get(geometry.storage_key)
        if surface is None or surface.get("split") != split:
            raise AnchoredTrainingRecordError(
                "anchored surface is absent or belongs to another split"
            )
        selection_index = surface.get("selection_index")
        if isinstance(selection_index, bool) or not isinstance(selection_index, int):
            raise AnchoredTrainingRecordError("selection_index is malformed")
        record = bind_anchored_training_record(
            surface,
            geometry,
            macro_rows=self.macro_rows,
            tokenizer=self.tokenizer,
            release_id=self.release_id,
        ).as_factorized_record()
        return LoadedAnchoredTrainingRecord(
            selection_index=selection_index,
            split=split,
            motif_record=record,
        )

    def _iter_split(self, split: str, *, batch_size: int) -> Iterator[tuple[Any, ...]]:
        source = (
            self.geometry_reader.iter_train_epoch(epoch=0, batch_size=batch_size)
            if split == "train"
            else self.geometry_reader.iter_dev(batch_size=batch_size)
        )
        seen: list[str] = []
        for batch in source:
            bound = tuple(self._bind(row, split=split) for row in batch)
            seen.extend(row.motif_record.storage_key for row in bound)
            yield bound
        if tuple(seen) != self._split_keys[split]:
            raise AnchoredTrainingRecordError(
                "anchored and geometry split orders differ"
            )

    def iter_train_epoch(self, *, epoch: int, batch_size: int) -> Iterator[tuple[Any, ...]]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise AnchoredTrainingRecordError("epoch must be nonnegative")
        yield from self._iter_split("train", batch_size=batch_size)

    def iter_dev(self, *, batch_size: int) -> Iterator[tuple[Any, ...]]:
        yield from self._iter_split("dev", batch_size=batch_size)

    def iter_strict_parallel_split(
        self,
        *,
        split: str,
        max_rows: int | None,
        workers: int,
        max_pending: int,
    ) -> Iterator[LoadedAnchoredTrainingRecord]:
        source = self.geometry_reader.iter_strict_parallel_split(
            split=split,
            max_rows=max_rows,
            workers=workers,
            max_pending=max_pending,
        )
        expected = self._split_keys[split]
        if max_rows is not None:
            expected = expected[:max_rows]
        seen: list[str] = []
        for loaded in source:
            row = self._bind(loaded, split=split)
            seen.append(row.motif_record.storage_key)
            yield row
        if tuple(seen) != expected:
            raise AnchoredTrainingRecordError(
                "parallel geometry decode changed anchored surface order"
            )

    def iter_donor_atom_maps(self, **kwargs: object) -> Iterator[Mapping[str, object]]:
        if self._donor_rows is None:
            yield from self.geometry_reader.iter_donor_atom_maps(**kwargs)
            return
        split = kwargs.get("split")
        max_rows = kwargs.get("max_rows")
        if split not in self._donor_rows:
            raise AnchoredTrainingRecordError("donor split must be train or dev")
        rows = self._donor_rows[str(split)]
        if max_rows is not None:
            if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
                raise AnchoredTrainingRecordError("donor max_rows must be positive")
            rows = rows[:max_rows]
        yield from rows


def _verify_surface_document(surface: Mapping[str, object]) -> None:
    artifact = surface.get("artifact_sha256")
    projection = dict(surface)
    projection.pop("artifact_sha256", None)
    if not isinstance(artifact, str) or artifact != _sha256_json(projection):
        raise AnchoredTrainingRecordError("anchored surface artifact digest mismatch")
    if surface.get("graphports_exposed_to_model") is not False:
        raise AnchoredTrainingRecordError("anchored surface must not expose GraphPorts")


def _record_projection(record: AnchoredTrainingRecord) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record.record_id,
        "storage_key": record.storage_key,
        "release_id": record.release_id,
        "geometry_record_content_sha256": record.geometry_record_content_sha256,
        "tokenizer_contract_sha256": record.tokenizer_contract_sha256,
        "tokenizer_snapshot_sha256": record.tokenizer_snapshot_sha256,
        "input_ids": list(record.input_ids),
        "token_to_logical_motif": list(record.token_to_logical_motif),
        "token_role": list(record.token_role),
        "identity_spans": [[row.start, row.stop] for row in record.identity_spans],
        "anchor_token_indices": [list(row) for row in record.anchor_token_indices],
        "logical_to_carrier": list(record.logical_to_carrier),
        "exact_identity_sha256": list(record.exact_identity_sha256),
        "source_atom_count": record.source_atom_count,
        "full_e3fp_ids": [list(row) for row in record.full_e3fp_ids],
        "atom_valid_mask": list(record.atom_valid_mask),
        "model_to_source_atom_index": list(record.model_to_source_atom_index),
        "atom_to_logical_motif": list(record.atom_to_logical_motif),
        "atom_is_attachment": list(record.atom_is_attachment),
        "anchor_token_to_atom": list(record.anchor_token_to_atom),
        "macro_used": list(record.macro_used),
    }


def bind_anchored_training_record(
    surface_record: Mapping[str, object],
    geometry_record: ProductionMotifRecord,
    *,
    macro_rows: Sequence[Mapping[str, object]],
    tokenizer: AnchoredTokenizerBinding,
    release_id: str,
) -> AnchoredTrainingRecord:
    """Create one immutable anchored record without recomputing E3FP or chemistry."""

    surface = surface_record.get("surface")
    if not isinstance(surface, Mapping):
        raise AnchoredTrainingRecordError("surface record has no anchored document")
    _verify_surface_document(surface)
    phrases = surface.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise AnchoredTrainingRecordError("anchored document has no motif phrases")
    if (
        surface.get("member_id") != geometry_record.record_id
        or surface_record.get("storage_key") != geometry_record.storage_key
        or surface.get("source_atom_count") != geometry_record.source_atom_count
        or tuple(surface.get("model_to_source_atom_index", ()))
        != geometry_record.model_to_source_atom_index
        or tuple(surface.get("atom_is_attachment", ()))
        != geometry_record.atom_is_attachment
    ):
        raise AnchoredTrainingRecordError("anchored and geometry record identity axes differ")

    encoded = encode_frozen_phrases(phrases, macro_rows)
    token_to_id = tokenizer.token_to_id()
    try:
        input_ids = tuple(token_to_id[token] for token in encoded.tokens)
    except KeyError as exc:
        raise AnchoredTrainingRecordError("anchored token is absent from frozen tokenizer") from exc

    token_owners = [-1] * len(encoded.tokens)
    roles = [""] * len(encoded.tokens)
    anchor_map = [-1] * len(encoded.tokens)
    atom_to_motif = [-1] * len(geometry_record.full_e3fp_ids)
    anchor_indices = []
    identity_hashes = []
    for motif_id, (phrase, phrase_span, identity_span, positions) in enumerate(
        zip(phrases, encoded.phrase_spans, encoded.identity_spans, encoded.anchor_token_positions)
    ):
        start, stop = phrase_span
        identity_start, identity_stop = identity_span
        for position in range(start, stop):
            token_owners[position] = motif_id
            roles[position] = "identity" if identity_start <= position < identity_stop else "connection"
        anchors = phrase.get("anchors")
        atoms = phrase.get("motif_atom_indices")
        pure = phrase.get("pure_motif")
        if not isinstance(anchors, list) or not isinstance(atoms, list) or not isinstance(pure, str):
            raise AnchoredTrainingRecordError("motif phrase sidecar is malformed")
        if len(anchors) != len(positions):
            raise AnchoredTrainingRecordError("anchor token/atom rows disagree")
        for position, anchor in zip(positions, anchors):
            atom_index = anchor.get("model_atom_index") if isinstance(anchor, Mapping) else None
            if isinstance(atom_index, bool) or not isinstance(atom_index, int):
                raise AnchoredTrainingRecordError("anchor atom address is malformed")
            anchor_map[position] = atom_index
        for atom_index in atoms:
            if (
                isinstance(atom_index, bool)
                or not isinstance(atom_index, int)
                or not 0 <= atom_index < len(atom_to_motif)
                or atom_to_motif[atom_index] != -1
            ):
                raise AnchoredTrainingRecordError("motif atom partition is malformed")
            atom_to_motif[atom_index] = motif_id
        anchor_indices.append(tuple(positions))
        identity_hashes.append(hashlib.sha256(pure.encode("utf-8")).hexdigest())
    if any(owner < 0 for owner in atom_to_motif) or any(not role for role in roles):
        raise AnchoredTrainingRecordError("anchored token/atom ownership is incomplete")
    if tuple(atom_to_motif) != geometry_record.atom_to_logical_motif:
        raise AnchoredTrainingRecordError("anchored motif partition differs from geometry source")
    for used_macro, span in zip(encoded.macro_used, encoded.identity_spans):
        if used_macro != (span[1] - span[0] == 1):
            raise AnchoredTrainingRecordError("macro/fallback identity span semantics drifted")
        if not used_macro and encoded.tokens[span[1] - 1] != FALLBACK_MOTIF_SUFFIX:
            raise AnchoredTrainingRecordError("fallback carrier is not the frozen suffix")

    placeholder = AnchoredTrainingRecord(
        record_artifact_sha256="",
        record_id=geometry_record.record_id,
        storage_key=geometry_record.storage_key,
        release_id=release_id,
        geometry_record_content_sha256=geometry_record.geometry_record_content_sha256,
        tokenizer_contract_sha256=tokenizer.tokenizer_contract_sha256,
        tokenizer_snapshot_sha256=tokenizer.tokenizer_snapshot_sha256,
        input_ids=input_ids,
        token_to_logical_motif=tuple(token_owners),
        token_role=tuple(roles),
        identity_spans=tuple(Span(*row) for row in encoded.identity_spans),
        anchor_token_indices=tuple(anchor_indices),
        logical_to_carrier=tuple(stop - 1 for _start, stop in encoded.phrase_spans),
        exact_identity_sha256=tuple(identity_hashes),
        source_atom_count=geometry_record.source_atom_count,
        full_e3fp_ids=geometry_record.full_e3fp_ids,
        atom_valid_mask=geometry_record.atom_valid_mask,
        model_to_source_atom_index=geometry_record.model_to_source_atom_index,
        atom_to_logical_motif=tuple(atom_to_motif),
        atom_is_attachment=geometry_record.atom_is_attachment,
        anchor_token_to_atom=tuple(anchor_map),
        macro_used=encoded.macro_used,
    )
    return AnchoredTrainingRecord(
        **{
            **placeholder.__dict__,
            "record_artifact_sha256": _sha256_json(_record_projection(placeholder)),
        }
    )


def tokenizer_binding_from_candidate_manifest(
    manifest: Mapping[str, object],
) -> AnchoredTokenizerBinding:
    """Load the frozen token-ID surface; macro semantics stay in a registry."""

    plan = manifest.get("plan")
    plan_file = manifest.get("plan_file")
    snapshot = manifest.get("snapshot")
    token_ids = manifest.get("token_ids")
    contracts = manifest.get("contracts")
    if not all(isinstance(value, Mapping) for value in (plan, plan_file, snapshot, token_ids, contracts)):
        raise AnchoredTrainingRecordError("candidate tokenizer manifest is malformed")
    assert isinstance(plan, Mapping)
    assert isinstance(plan_file, Mapping)
    assert isinstance(snapshot, Mapping)
    assert isinstance(token_ids, Mapping)
    assert isinstance(contracts, Mapping)
    declared = token_ids.get("declared")
    if (
        manifest.get("schema_version") != "most-t5-next/anchored-candidate-tokenizer/v2"
        or manifest.get("status") != "candidate"
        or plan.get("boundary_mode") != "fallback_single_suffix"
        or contracts.get("frozen_grammar_bound") is not True
        or not isinstance(declared, Mapping)
    ):
        raise AnchoredTrainingRecordError("candidate tokenizer does not bind the frozen grammar")
    rows = tuple(
        sorted(
            ((str(token), int(token_id)) for token, token_id in declared.items()),
            key=lambda row: row[1],
        )
    )
    return AnchoredTokenizerBinding(
        tokenizer_contract_sha256=str(plan_file.get("plan_sha256")),
        tokenizer_snapshot_sha256=str(snapshot.get("tree_sha256")),
        vocab_size=int(plan.get("final_vocab_size")),
        token_id_rows=rows,
    )


__all__ = [
    "AnchoredTrainingRecordReader",
    "AnchoredTokenizerBinding",
    "AnchoredTrainingRecord",
    "AnchoredTrainingRecordError",
    "LoadedAnchoredTrainingRecord",
    "SCHEMA_VERSION",
    "bind_anchored_training_record",
    "tokenizer_binding_from_candidate_manifest",
]
