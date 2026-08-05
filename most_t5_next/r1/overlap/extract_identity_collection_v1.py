#!/usr/bin/env python3
"""Extract one hash-only identity collection from one explicitly mapped source.

Supported sources are the surviving project legacy LMDB (with an explicit
trusted-pickle SHA acknowledgement) and JSON array, JSONL, CSV, or Parquet
files.  Every source field and collection role is supplied by a frozen config;
the extractor does not infer CID, SMILES, split, task, or metadata keys.
"""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import sqlite3
import sys
import unicodedata
from pathlib import Path


CONTRACT_SCHEMA = "most-t5-r1/identity-collection-extraction-contract/v1"
CONFIG_SCHEMA = "most-t5-r1/identity-collection-extraction-config/v1"
REPORT_SCHEMA = "most-t5-r1/identity-collection-extraction-report/v1"
COLLECTION_SCHEMA = "most-t5-r1/identity-collection-manifest/v1"
MOLECULE_ROW_SCHEMA = "most-t5-r1/molecule-identity-row/v1"
TEXT_ROW_SCHEMA = "most-t5-r1/text-pair-identity-row/v1"

SHA256_HEX = frozenset("0123456789abcdef")
FORMATS = frozenset(("legacy_lmdb_pickle", "json_array", "jsonl", "csv", "parquet"))
MEMBER_SOURCES = frozenset(("lmdb_key", "field", "row_index"))
TEXT_NORMALIZATIONS = frozenset(("identity_utf8_v1", "unicode_nfkc_whitespace_v1"))
ROLES = frozenset(
    (
        "p1_structure_train",
        "p2_permitted_train_membership",
        "p2_alignment_train",
        "p2_geometry_replay_train",
        "downstream_train",
        "downstream_validation",
        "downstream_test",
    )
)

CONFIG_FIELDS = frozenset(("schema_version", "extraction_id", "contract_sha256", "collection", "source", "identity", "mapping"))
COLLECTION_FIELDS = frozenset(
    ("collection_id", "dataset_id", "release_id", "phase", "split", "role", "task_family", "source_identity_namespace")
)
SOURCE_FIELDS = frozenset(("path", "format", "expected_bytes", "expected_sha256", "format_options"))
FORMAT_OPTION_FIELDS = frozenset(("json_top_level", "csv_dialect", "parquet_batch_size", "lmdb"))
LMDB_OPTION_FIELDS = frozenset(
    ("subdir", "metadata_keys_permitted", "metadata_keys_required", "trusted_pickle_source_sha256")
)
IDENTITY_FIELDS = frozenset(
    (
        "normalization_contract_path",
        "normalization_contract_sha256",
        "normalizer_path",
        "normalizer_sha256",
        "required_rdkit_version",
    )
)
MAPPING_FIELDS = frozenset(("member_id", "smiles_field", "record_filter", "text_identity"))
MEMBER_FIELDS = frozenset(("source", "field", "prefix", "crosscheck_field"))
FILTER_FIELDS = frozenset(("field", "allowed_values"))
TEXT_IDENTITY_FIELDS = frozenset(("status", "normalization", "unit"))
TEXT_UNIT_FIELDS = frozenset(("unit_name", "semantic_role", "serialization", "components"))
COMPONENT_FIELDS = frozenset(("name", "field"))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and not (set(value) - SHA256_HEX)


def require_sha256(value, label, nullable=False):
    if value is None and nullable:
        return
    if not is_sha256(value):
        raise ValueError("{} must be a lowercase SHA-256{}".format(label, " or null" if nullable else ""))


def require_string(value, label, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value) or any(character in value for character in "\x00\r\n\t"):
        raise ValueError("{} must be a {}control-free string".format(label, "possibly empty " if allow_empty else "non-empty "))
    return value


def require_exact_fields(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(label))
    actual = frozenset(value)
    if actual != expected:
        raise ValueError("{} fields differ; missing={}, extra={}".format(label, sorted(expected - actual), sorted(actual - expected)))


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: {}".format(key))
        result[key] = value
    return result


def reject_nonfinite_json_constant(value):
    raise ValueError("non-finite JSON constant is forbidden: {}".format(value))


def load_json(path, label):
    with open(str(path), "r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicate_pairs, parse_constant=reject_nonfinite_json_constant)
    if not isinstance(value, dict):
        raise ValueError("{} must contain an object".format(label))
    return value


def regular_nonsymlink(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("{} is not a regular non-symlink file: {}".format(label, path))
    return path


def resolve_file(raw, parent, label):
    require_string(raw, label)
    path = Path(raw)
    if not path.is_absolute():
        path = parent / path
    regular_nonsymlink(path, label)
    return path.resolve()


def scalar_string(value, label):
    if isinstance(value, bool) or value is None or not isinstance(value, (str, int)):
        raise ValueError("{} must be a string or integer scalar".format(label))
    result = str(value)
    return require_string(result, label)


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct module specification for {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_collection(value):
    require_exact_fields(value, COLLECTION_FIELDS, "collection")
    for key in COLLECTION_FIELDS:
        require_string(value[key], "collection.{}".format(key))
    if value["role"] not in ROLES:
        raise ValueError("collection.role is unknown")
    expected = {
        "p1_structure_train": ("p1", "train"),
        "p2_permitted_train_membership": ("p2", "train"),
        "p2_alignment_train": ("p2", "train"),
        "p2_geometry_replay_train": ("p2", "train"),
        "downstream_train": ("downstream", "train"),
        "downstream_validation": ("downstream", "validation"),
        "downstream_test": ("downstream", "test"),
    }[value["role"]]
    if (value["phase"], value["split"]) != expected:
        raise ValueError("collection role is inconsistent with phase/split")
    if value["role"] in ("p1_structure_train", "p2_permitted_train_membership") and value["task_family"] != "none":
        raise ValueError("membership-only collection task_family must be none")
    if value["role"].startswith("downstream_") and value["task_family"] == "none":
        raise ValueError("downstream collection must explicitly name its task family")


def validate_source(value):
    require_exact_fields(value, SOURCE_FIELDS, "source")
    require_string(value["path"], "source.path")
    if value["format"] not in FORMATS:
        raise ValueError("source.format is unsupported")
    if not isinstance(value["expected_bytes"], int) or isinstance(value["expected_bytes"], bool) or value["expected_bytes"] < 0:
        raise ValueError("source.expected_bytes must be non-negative")
    require_sha256(value["expected_sha256"], "source.expected_sha256")
    options = value["format_options"]
    require_exact_fields(options, FORMAT_OPTION_FIELDS, "source.format_options")
    source_format = value["format"]
    if source_format == "json_array":
        if options["json_top_level"] != "array" or any(options[key] is not None for key in ("csv_dialect", "parquet_batch_size", "lmdb")):
            raise ValueError("json_array requires only json_top_level=array")
    elif source_format == "csv":
        if options["csv_dialect"] != "excel" or any(options[key] is not None for key in ("json_top_level", "parquet_batch_size", "lmdb")):
            raise ValueError("csv requires only csv_dialect=excel")
    elif source_format == "parquet":
        batch_size = options["parquet_batch_size"]
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("parquet_batch_size must be positive")
        if any(options[key] is not None for key in ("json_top_level", "csv_dialect", "lmdb")):
            raise ValueError("parquet options contain an irrelevant value")
    elif source_format == "legacy_lmdb_pickle":
        if any(options[key] is not None for key in ("json_top_level", "csv_dialect", "parquet_batch_size")):
            raise ValueError("legacy LMDB options contain an irrelevant value")
        lmdb_options = options["lmdb"]
        require_exact_fields(lmdb_options, LMDB_OPTION_FIELDS, "source.format_options.lmdb")
        if lmdb_options["subdir"] is not False:
            raise ValueError("v1 accepts only single-file LMDB with subdir=false")
        for key in ("metadata_keys_permitted", "metadata_keys_required"):
            items = lmdb_options[key]
            if not isinstance(items, list) or len(items) != len(set(items)):
                raise ValueError("lmdb.{} must be a duplicate-free string array".format(key))
            for item in items:
                require_string(item, "lmdb metadata key")
        if not set(lmdb_options["metadata_keys_required"]).issubset(set(lmdb_options["metadata_keys_permitted"])):
            raise ValueError("required LMDB metadata keys must be permitted")
        require_sha256(lmdb_options["trusted_pickle_source_sha256"], "trusted pickle source SHA-256")
        if lmdb_options["trusted_pickle_source_sha256"] != value["expected_sha256"]:
            raise ValueError("trusted pickle acknowledgement must equal the locked source SHA-256")
    elif source_format == "jsonl":
        if any(options[key] is not None for key in FORMAT_OPTION_FIELDS):
            raise ValueError("jsonl format options must all be null")


def validate_identity(value):
    require_exact_fields(value, IDENTITY_FIELDS, "identity")
    for key in ("normalization_contract_path", "normalizer_path", "required_rdkit_version"):
        require_string(value[key], "identity.{}".format(key))
    require_sha256(value["normalization_contract_sha256"], "identity normalization contract SHA-256")
    require_sha256(value["normalizer_sha256"], "identity normalizer SHA-256")


def validate_mapping(value, source_format, role):
    require_exact_fields(value, MAPPING_FIELDS, "mapping")
    member = value["member_id"]
    require_exact_fields(member, MEMBER_FIELDS, "mapping.member_id")
    if member["source"] not in MEMBER_SOURCES:
        raise ValueError("member ID source is unsupported")
    require_string(member["prefix"], "member ID prefix", allow_empty=True)
    for key in ("field", "crosscheck_field"):
        if member[key] is not None:
            require_string(member[key], "member ID {}".format(key))
    if member["source"] == "field" and member["field"] is None:
        raise ValueError("field member-ID source requires a field")
    if member["source"] != "field" and member["field"] is not None:
        raise ValueError("non-field member-ID source forbids member_id.field")
    if source_format == "legacy_lmdb_pickle" and member["source"] != "lmdb_key":
        raise ValueError("legacy LMDB requires member_id.source=lmdb_key")
    if source_format != "legacy_lmdb_pickle" and member["source"] == "lmdb_key":
        raise ValueError("lmdb_key member source is only valid for legacy LMDB")
    require_string(value["smiles_field"], "mapping.smiles_field")
    record_filter = value["record_filter"]
    if record_filter is not None:
        require_exact_fields(record_filter, FILTER_FIELDS, "mapping.record_filter")
        require_string(record_filter["field"], "record filter field")
        allowed = record_filter["allowed_values"]
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("record filter allowed_values must be non-empty")
        for item in allowed:
            if isinstance(item, (dict, list, float)):
                raise ValueError("record filter values must be exact JSON string/integer/boolean/null scalars")
    text = value["text_identity"]
    require_exact_fields(text, TEXT_IDENTITY_FIELDS, "mapping.text_identity")
    if text["status"] not in ("available", "unavailable"):
        raise ValueError("text identity status must be available or unavailable")
    if text["status"] == "unavailable":
        if text["normalization"] is not None or text["unit"] is not None:
            raise ValueError("unavailable text identity forbids normalization/unit")
    else:
        if text["normalization"] not in TEXT_NORMALIZATIONS:
            raise ValueError("text normalization is unsupported")
        unit = text["unit"]
        require_exact_fields(unit, TEXT_UNIT_FIELDS, "text identity unit")
        for key in ("unit_name", "semantic_role"):
            require_string(unit[key], "text unit {}".format(key))
        if unit["serialization"] != "canonical_component_object_utf8_v1":
            raise ValueError("text unit serialization is unsupported")
        components = unit["components"]
        if not isinstance(components, list) or not components:
            raise ValueError("text unit components must be non-empty")
        component_names = []
        for component in components:
            require_exact_fields(component, COMPONENT_FIELDS, "text component")
            require_string(component["name"], "text component name")
            require_string(component["field"], "text component field")
            component_names.append(component["name"])
        if len(component_names) != len(set(component_names)):
            raise ValueError("text component names must be unique")
    if role in ("p1_structure_train", "p2_permitted_train_membership") and text["status"] != "unavailable":
        raise ValueError("membership-only collection cannot carry text rows")
    if role == "p2_alignment_train" and text["status"] != "available":
        raise ValueError("P2 alignment collection must explicitly map a text unit")


def validate_config(config, contract_sha256):
    require_exact_fields(config, CONFIG_FIELDS, "extraction config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("extraction config schema mismatch")
    require_string(config["extraction_id"], "extraction ID")
    require_sha256(config["contract_sha256"], "config contract SHA-256")
    if config["contract_sha256"] != contract_sha256:
        raise ValueError("config does not bind the supplied extraction contract")
    validate_collection(config["collection"])
    validate_source(config["source"])
    validate_identity(config["identity"])
    validate_mapping(config["mapping"], config["source"]["format"], config["collection"]["role"])


def iter_source_records(source_path, source, observed_metadata):
    source_format = source["format"]
    if source_format == "legacy_lmdb_pickle":
        try:
            import lmdb
        except ImportError as exc:
            raise RuntimeError("python-lmdb is required for legacy_lmdb_pickle") from exc
        options = source["format_options"]["lmdb"]
        permitted = set(options["metadata_keys_permitted"])
        environment = lmdb.open(
            str(source_path),
            subdir=options["subdir"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=8,
        )
        try:
            with environment.begin() as transaction:
                for row_index, (raw_key, raw_value) in enumerate(transaction.cursor()):
                    try:
                        key = raw_key.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError("LMDB key is not UTF-8") from exc
                    if key in permitted:
                        observed_metadata.add(key)
                        continue
                    if key.startswith("__"):
                        raise ValueError("undeclared LMDB metadata-like key: {}".format(key))
                    record = pickle.loads(raw_value)
                    if not isinstance(record, dict):
                        raise ValueError("LMDB member value is not a dictionary")
                    yield row_index, key, record
        finally:
            environment.close()
        required = set(options["metadata_keys_required"])
        if not required.issubset(observed_metadata):
            raise ValueError("required LMDB metadata keys were not observed: {}".format(sorted(required - observed_metadata)))
        return
    if source_format == "json_array":
        with open(str(source_path), "r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=reject_duplicate_pairs, parse_constant=reject_nonfinite_json_constant)
        if not isinstance(payload, list):
            raise ValueError("json_array source does not contain a top-level array")
        for row_index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError("JSON array record is not an object")
            yield row_index, None, record
        return
    if source_format == "jsonl":
        with open(str(source_path), "r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError("JSONL source contains a blank row")
                record = json.loads(line, object_pairs_hook=reject_duplicate_pairs, parse_constant=reject_nonfinite_json_constant)
                if not isinstance(record, dict):
                    raise ValueError("JSONL record is not an object")
                yield row_index, None, record
        return
    if source_format == "csv":
        with open(str(source_path), "r", encoding="utf-8", newline="") as handle:
            for row_index, record in enumerate(csv.DictReader(handle, dialect="excel")):
                yield row_index, None, dict(record)
        return
    if source_format == "parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for parquet extraction") from exc
        batch_size = source["format_options"]["parquet_batch_size"]
        row_index = 0
        parquet_file = parquet.ParquetFile(str(source_path))
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for record in batch.to_pylist():
                if not isinstance(record, dict):
                    raise ValueError("Parquet row is not an object")
                yield row_index, None, record
                row_index += 1
        return
    raise RuntimeError("unreachable source format")


def mapped_member_id(mapping, row_index, lmdb_key, record):
    spec = mapping["member_id"]
    if spec["source"] == "lmdb_key":
        base = scalar_string(lmdb_key, "LMDB member key")
    elif spec["source"] == "row_index":
        base = str(int(row_index))
    else:
        if spec["field"] not in record:
            raise ValueError("member ID field is absent: {}".format(spec["field"]))
        base = scalar_string(record[spec["field"]], "member ID field")
    if spec["crosscheck_field"] is not None:
        field = spec["crosscheck_field"]
        if field not in record:
            raise ValueError("member ID crosscheck field is absent: {}".format(field))
        if scalar_string(record[field], "member ID crosscheck field") != base:
            raise ValueError("member ID crosscheck differs from the selected source ID")
    return spec["prefix"] + base


def record_selected(record, filter_spec):
    if filter_spec is None:
        return True
    field = filter_spec["field"]
    if field not in record:
        raise ValueError("record filter field is absent: {}".format(field))
    return record[field] in filter_spec["allowed_values"]


def normalize_text(value, policy):
    if policy == "identity_utf8_v1":
        return value
    if policy == "unicode_nfkc_whitespace_v1":
        return " ".join(unicodedata.normalize("NFKC", value).split())
    raise RuntimeError("unreachable text normalization")


def text_specs(text_config):
    if text_config["status"] == "unavailable":
        return {"status": "unavailable", "exact_spec_sha256": None, "normalized_spec_sha256": None}
    unit = text_config["unit"]
    semantic = {
        "domain": "most-t5-r1/text-unit-exact-spec/v1",
        "semantic_role": unit["semantic_role"],
        "serialization": unit["serialization"],
        "component_names": [item["name"] for item in unit["components"]],
    }
    normalized = dict(semantic)
    normalized["domain"] = "most-t5-r1/text-unit-normalized-spec/v1"
    normalized["normalization"] = text_config["normalization"]
    return {
        "status": "available",
        "exact_spec_sha256": sha256_bytes(canonical_json_bytes(semantic)),
        "normalized_spec_sha256": sha256_bytes(canonical_json_bytes(normalized)),
    }


def pair_digest(domain, molecule_sha256, text_sha256):
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": domain,
                "molecule_identity_sha256": molecule_sha256,
                "text_normalized_sha256": text_sha256,
            }
        )
    )


def build_text_row(collection, member_id, connectivity_sha, stereo_sha, record, text_config):
    if text_config["status"] == "unavailable":
        return None
    unit = text_config["unit"]
    exact = {}
    normalized = {}
    for component in unit["components"]:
        field = component["field"]
        if field not in record or not isinstance(record[field], str):
            raise ValueError("text component field is absent or not a string: {}".format(field))
        exact[component["name"]] = record[field]
        normalized[component["name"]] = normalize_text(record[field], text_config["normalization"])
    exact_sha = sha256_bytes(canonical_json_bytes(exact))
    normalized_sha = sha256_bytes(canonical_json_bytes(normalized))
    pair_id = "textpair:" + sha256_bytes(
        canonical_json_bytes(
            {
                "collection_id": collection["collection_id"],
                "domain": "most-t5-r1/text-pair-source-address/v1",
                "member_id": member_id,
                "unit_name": unit["unit_name"],
            }
        )
    )
    return {
        "schema_version": TEXT_ROW_SCHEMA,
        "collection_id": collection["collection_id"],
        "pair_id": pair_id,
        "member_id": member_id,
        "task_family": collection["task_family"],
        "text_exact_sha256": exact_sha,
        "text_normalized_sha256": normalized_sha,
        "connectivity_text_pair_sha256": pair_digest(
            "most-t5-r1/connectivity-text-pair/v1", connectivity_sha, normalized_sha
        ),
        "stereo_text_pair_sha256": pair_digest(
            "most-t5-r1/stereo-text-pair/v1", stereo_sha, normalized_sha
        ),
    }


def create_database():
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE molecules (
            member_id TEXT PRIMARY KEY,
            row_json BLOB NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE text_pairs (
            pair_id TEXT PRIMARY KEY,
            row_json BLOB NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


def write_rows(connection, table, key_column, output_path):
    digest = hashlib.sha256()
    key_digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    previous = None
    with open(str(output_path), "xb") as handle:
        query = "SELECT {}, row_json FROM {} ORDER BY {} COLLATE BINARY".format(key_column, table, key_column)
        for key, raw in connection.execute(query):
            encoded = key.encode("utf-8")
            if previous is not None and encoded <= previous:
                raise RuntimeError("SQLite output order is not strict UTF-8 byte order")
            previous = encoded
            line = bytes(raw) + b"\n"
            handle.write(line)
            digest.update(line)
            key_digest.update(encoded + b"\n")
            byte_count += len(line)
            row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": output_path.name,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "row_count": row_count,
        "key_lf_sha256": key_digest.hexdigest(),
    }


def write_json_new(path, value):
    with open(str(path), "xb") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def report_payload_sha256(report):
    payload = dict(report)
    payload.pop("report_canonical_payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def run_extraction(contract_path, config_path, output_dir):
    contract_path = regular_nonsymlink(contract_path, "extraction contract").resolve()
    config_path = regular_nonsymlink(config_path, "extraction config").resolve()
    contract_bytes, contract_sha = sha256_file(contract_path)
    config_bytes, config_sha = sha256_file(config_path)
    extractor_path = Path(__file__).resolve()
    extractor_observed = sha256_file(extractor_path)
    contract = load_json(contract_path, "extraction contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("extraction contract schema mismatch")
    if set(contract.get("supported_source_formats", [])) != set(FORMATS):
        raise ValueError("extraction contract formats differ from the extractor")
    config = load_json(config_path, "extraction config")
    validate_config(config, contract_sha)
    source_path = resolve_file(config["source"]["path"], config_path.parent, "source.path")
    identity_contract_path = resolve_file(
        config["identity"]["normalization_contract_path"], config_path.parent, "identity.normalization_contract_path"
    )
    normalizer_path = resolve_file(config["identity"]["normalizer_path"], config_path.parent, "identity.normalizer_path")
    identity_contract_observed = sha256_file(identity_contract_path)
    normalizer_observed = sha256_file(normalizer_path)
    if identity_contract_observed[1] != config["identity"]["normalization_contract_sha256"]:
        raise ValueError("identity normalization contract SHA-256 mismatch")
    if normalizer_observed[1] != config["identity"]["normalizer_sha256"]:
        raise ValueError("identity normalizer SHA-256 mismatch")
    source_pre = sha256_file(source_path)
    if source_pre != (config["source"]["expected_bytes"], config["source"]["expected_sha256"]):
        raise ValueError("source bytes/SHA-256 differ from extraction config")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for molecular identity extraction") from exc
    rdkit_version = Chem.rdBase.rdkitVersion
    if rdkit_version != config["identity"]["required_rdkit_version"]:
        raise RuntimeError("RDKit version differs from the extraction config")
    normalizer = import_module(normalizer_path, "r1_identity_extractor_normalizer_" + config_sha[:12])
    if not callable(getattr(normalizer, "canonical_forms", None)):
        raise RuntimeError("identity normalizer does not expose canonical_forms(Chem, mol)")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    connection = create_database()
    observed_metadata = set()
    source_records = 0
    filtered_records = 0
    selected_records = 0
    try:
        for row_index, lmdb_key, record in iter_source_records(source_path, config["source"], observed_metadata):
            source_records += 1
            if not record_selected(record, config["mapping"]["record_filter"]):
                filtered_records += 1
                continue
            member_id = mapped_member_id(config["mapping"], row_index, lmdb_key, record)
            smiles_field = config["mapping"]["smiles_field"]
            if smiles_field not in record or not isinstance(record[smiles_field], str) or not record[smiles_field]:
                raise ValueError("SMILES field is absent or not a non-empty string")
            molecule = Chem.MolFromSmiles(record[smiles_field])
            if molecule is None:
                raise ValueError("RDKit failed to parse selected SMILES at source row {}".format(row_index))
            forms = normalizer.canonical_forms(Chem, molecule)
            if not isinstance(forms, dict) or not isinstance(forms.get("connectivity"), str) or not isinstance(forms.get("strict"), str):
                raise RuntimeError("identity normalizer returned an invalid canonical-forms object")
            connectivity_sha = sha256_bytes(forms["connectivity"].encode("utf-8"))
            stereo_sha = sha256_bytes(forms["strict"].encode("utf-8"))
            molecule_row = {
                "schema_version": MOLECULE_ROW_SCHEMA,
                "collection_id": config["collection"]["collection_id"],
                "member_id": member_id,
                "connectivity_identity_sha256": connectivity_sha,
                "stereo_identity_sha256": stereo_sha,
                "conformer_identity_sha256": None,
            }
            try:
                connection.execute(
                    "INSERT INTO molecules VALUES (?,?)", (member_id, canonical_json_bytes(molecule_row))
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate selected member_id: {}".format(member_id)) from exc
            text_row = build_text_row(
                config["collection"], member_id, connectivity_sha, stereo_sha, record, config["mapping"]["text_identity"]
            )
            if text_row is not None:
                try:
                    connection.execute(
                        "INSERT INTO text_pairs VALUES (?,?)",
                        (text_row["pair_id"], canonical_json_bytes(text_row)),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate selected text pair ID") from exc
            selected_records += 1
        if selected_records <= 0:
            raise ValueError("explicit mapping/filter selected zero records")
        connection.commit()
        source_post = sha256_file(source_path)
        if source_post != source_pre:
            raise RuntimeError("source bytes changed during extraction")
        if sha256_file(identity_contract_path) != identity_contract_observed or sha256_file(normalizer_path) != normalizer_observed:
            raise RuntimeError("identity contract or normalizer bytes changed during extraction")
        if sha256_file(contract_path) != (contract_bytes, contract_sha) or sha256_file(config_path) != (config_bytes, config_sha):
            raise RuntimeError("extraction contract or config bytes changed during extraction")
        if sha256_file(extractor_path) != extractor_observed:
            raise RuntimeError("extractor bytes changed during extraction")
        molecule_artifact = write_rows(connection, "molecules", "member_id", output_dir / "molecule_identity_rows.jsonl")
        text_artifact = None
        if config["mapping"]["text_identity"]["status"] == "available":
            text_artifact = write_rows(connection, "text_pairs", "pair_id", output_dir / "text_pair_identity_rows.jsonl")
            if text_artifact["row_count"] != selected_records:
                raise RuntimeError("one-text-unit-per-selected-record invariant failed")
        source_lock = {
            "schema_version": "most-t5-r1/identity-extraction-source-lock/v1",
            "extraction_id": config["extraction_id"],
            "source": {
                "path": str(source_path),
                "format": config["source"]["format"],
                "bytes": source_pre[0],
                "sha256": source_pre[1],
            },
            "config_sha256": config_sha,
            "identity_normalization_contract_sha256": identity_contract_observed[1],
            "identity_normalizer_sha256": normalizer_observed[1],
            "rdkit_version": rdkit_version,
            "lmdb_metadata_keys_permitted": (
                config["source"]["format_options"]["lmdb"]["metadata_keys_permitted"]
                if config["source"]["format"] == "legacy_lmdb_pickle"
                else []
            ),
            "lmdb_metadata_keys_required": (
                config["source"]["format_options"]["lmdb"]["metadata_keys_required"]
                if config["source"]["format"] == "legacy_lmdb_pickle"
                else []
            ),
            "lmdb_metadata_keys_observed_and_excluded": sorted(observed_metadata),
        }
        source_lock_path = output_dir / "source_lock.json"
        write_json_new(source_lock_path, source_lock)
        source_lock_sha = sha256_file(source_lock_path)[1]
        resolved_config_path = output_dir / "resolved_config.json"
        write_json_new(resolved_config_path, config)
        resolved_config_observation = sha256_file(resolved_config_path)
        collection = config["collection"]
        collection_manifest = {
            "schema_version": COLLECTION_SCHEMA,
            "collection_id": collection["collection_id"],
            "dataset_id": collection["dataset_id"],
            "release_id": collection["release_id"],
            "phase": collection["phase"],
            "split": collection["split"],
            "role": collection["role"],
            "task_family": collection["task_family"],
            "identity_specs": {
                "connectivity_identity_spec_sha256": identity_contract_observed[1],
                "stereo_identity_spec_sha256": identity_contract_observed[1],
                "conformer_identity": {"status": "unavailable", "spec_sha256": None},
                "text_identity": text_specs(config["mapping"]["text_identity"]),
            },
            "molecule_rows": molecule_artifact,
            "text_pair_rows": text_artifact,
            "provenance": {
                "source_identity_namespace": collection["source_identity_namespace"],
                "source_release_manifest_sha256": source_lock_sha,
                "extractor_sha256": extractor_observed[1],
                "excluded_source_metadata_keys": sorted(observed_metadata),
            },
        }
        collection_manifest_path = output_dir / "collection_manifest.json"
        write_json_new(collection_manifest_path, collection_manifest)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "pass",
            "p1_training_admission": False,
            "p2_training_admission": False,
            "extraction_id": config["extraction_id"],
            "generated_at_utc": utc_now(),
            "counts": {
                "source_data_records_seen": source_records,
                "records_filtered_by_explicit_policy": filtered_records,
                "selected_molecule_members": selected_records,
                "emitted_text_pairs": text_artifact["row_count"] if text_artifact is not None else 0,
                "observed_excluded_lmdb_metadata_keys": len(observed_metadata),
            },
            "artifacts": {
                "source_lock": {"path": source_lock_path.name, "sha256": source_lock_sha},
                "resolved_config": {
                    "path": resolved_config_path.name,
                    "bytes": resolved_config_observation[0],
                    "sha256": resolved_config_observation[1]
                },
                "molecule_rows": molecule_artifact,
                "text_pair_rows": text_artifact,
                "collection_manifest": {
                    "path": collection_manifest_path.name,
                    "bytes": sha256_file(collection_manifest_path)[0],
                    "sha256": sha256_file(collection_manifest_path)[1],
                },
            },
            "provenance": {
                "contract_path": str(contract_path),
                "contract_bytes": contract_bytes,
                "contract_sha256": contract_sha,
                "config_path": str(config_path),
                "config_bytes": config_bytes,
                "config_sha256": config_sha,
                "extractor_path": str(extractor_path),
                "extractor_sha256": extractor_observed[1],
                "source_path": str(source_path),
                "source_bytes": source_pre[0],
                "source_sha256": source_pre[1],
                "identity_contract_sha256": identity_contract_observed[1],
                "normalizer_sha256": normalizer_observed[1],
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "rdkit_version": rdkit_version,
            },
            "policy_boundary": "No P1/P2 overlap, 301655-vs-301658 P2 membership, downstream scope, tokenizer, or training-admission decision is made by this extractor.",
        }
        report["report_canonical_payload_sha256"] = report_payload_sha256(report)
        write_json_new(output_dir / "extraction_report.json", report)
        return report
    finally:
        connection.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_extraction(Path(args.contract), Path(args.config), Path(args.output_dir))
    print(json.dumps({"status": report["status"], "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
