"""Fail-closed integrity checks for the PCQM4Mv2 R1 companion inputs.

This module deliberately imports only the Python standard library.  Its
``verify_pcqm_inputs`` function is the mandatory boundary before either the
bounded builder or a validator may parse the SDF/CSV or deserialize
``split_dict.pt``.  In particular, a legacy PyTorch unpickling fallback is
available only through :func:`load_verified_split_dict` after the archive,
CSV, split file, companion ZIP, and locked manifests have all been verified.

The code is shared for *integrity only*.  It contains no RDKit, E3FP, motif,
or record-building logic, so using it in a later reference audit does not
compromise that audit's semantic independence.
"""

from __future__ import print_function

import hashlib
import json
import pickle
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


SOURCE_CONTRACT_SCHEMA = "most-t5-r1/pcqm4mv2-source-contract/v3"
_HEX64 = frozenset("0123456789abcdef")
_REQUIRED_ARTIFACTS = (
    "train_3d_sdf_archive",
    "companion_archive",
    "companion_data_csv_gz",
    "companion_split_dict_pt",
)
_REQUIRED_MANIFESTS = (
    "train_sdf_source_manifest",
    "train_sdf_member_hash_report",
    "companion_source_manifest",
    "companion_content_validation",
)


def sha256_file(path):
    """Return a streaming SHA-256; never materialize a source artifact."""
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(handle):
    digest = hashlib.sha256()
    observed_bytes = 0
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        observed_bytes += len(block)
        digest.update(block)
    return observed_bytes, digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _regular_nonsymlink_file(path, label):
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise RuntimeError("{} must be an absolute path".format(label))
    if target.is_symlink():
        raise RuntimeError("{} must not be a symlink: {}".format(label, target))
    if not target.is_file():
        raise FileNotFoundError("{} is not a regular file: {}".format(label, target))
    return target.resolve(strict=True)


def _load_json_object(path, label):
    target = _regular_nonsymlink_file(path, label)
    try:
        with open(str(target), "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise RuntimeError("cannot parse {}: {}".format(label, type(exc).__name__)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("{} must contain a JSON object".format(label))
    return target, value


def _require_dict(value, label):
    if not isinstance(value, dict):
        raise RuntimeError("{} must be an object".format(label))
    return value


def _require_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError("{} must be a non-negative integer".format(label))
    return value


def _verify_locked_file(actual_path, spec, label):
    """Verify canonical path, byte count, and content hash for one file."""
    spec = _require_dict(spec, "{} contract entry".format(label))
    declared_path = spec.get("path")
    expected_bytes = _require_int(spec.get("bytes"), "{} bytes".format(label))
    expected_sha256 = spec.get("sha256")
    if not _is_sha256(expected_sha256):
        raise RuntimeError("{} SHA-256 is invalid".format(label))
    if not isinstance(declared_path, str):
        raise RuntimeError("{} path is invalid".format(label))

    actual = _regular_nonsymlink_file(actual_path, label)
    declared = _regular_nonsymlink_file(declared_path, "{} locked path".format(label))
    if not actual.samefile(declared):
        raise RuntimeError("{} is not the exact locked artifact".format(label))
    observed_bytes = int(actual.stat().st_size)
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            "{} byte count differs from the source contract ({} != {})".format(
                label, observed_bytes, expected_bytes
            )
        )
    observed_sha256 = sha256_file(actual)
    if observed_sha256 != expected_sha256:
        raise RuntimeError("{} SHA-256 differs from the source contract".format(label))
    return {
        "path": str(actual),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
    }


def _source_artifact_specs(contract):
    source = _require_dict(contract.get("source"), "source")
    companion = _require_dict(contract.get("official_companion"), "official_companion")
    archive = {
        "path": source.get("remote_archive_path"),
        "bytes": source.get("remote_archive_bytes"),
        "sha256": source.get("remote_archive_sha256"),
    }
    companion_archive = _require_dict(companion.get("official_archive"), "official_companion.official_archive")
    data_csv = _require_dict(companion.get("data_csv"), "official_companion.data_csv")
    split_dict = _require_dict(companion.get("split_dict"), "official_companion.split_dict")
    return {
        "train_3d_sdf_archive": archive,
        "companion_archive": companion_archive,
        "companion_data_csv_gz": data_csv,
        "companion_split_dict_pt": split_dict,
    }


def _manifest_specs(contract):
    source = _require_dict(contract.get("source"), "source")
    companion = _require_dict(contract.get("official_companion"), "official_companion")
    return {
        "train_sdf_source_manifest": source.get("remote_source_manifest"),
        "train_sdf_member_hash_report": source.get("remote_sdf_member_hash_report"),
        "companion_source_manifest": companion.get("remote_source_manifest"),
        "companion_content_validation": companion.get("remote_content_validation"),
    }


def _verify_contract_shape(contract):
    if contract.get("schema_version") != SOURCE_CONTRACT_SCHEMA:
        raise RuntimeError("source contract schema is not the expected R1 PCQM v3")
    source = _require_dict(contract.get("source"), "source")
    companion = _require_dict(contract.get("official_companion"), "official_companion")
    if source.get("official_train_sdf_records") != 3_378_606:
        raise RuntimeError("source contract train record count is not 3378606")
    member = _require_dict(source.get("train_sdf_member"), "source.train_sdf_member")
    if not isinstance(member.get("tar_member_name"), str) or not member["tar_member_name"].lower().endswith(".sdf"):
        raise RuntimeError("source contract train_sdf_member name is invalid")
    if member.get("member_type") != "regular_file":
        raise RuntimeError("source contract train_sdf_member type is invalid")
    _require_int(member.get("uncompressed_bytes"), "source.train_sdf_member uncompressed_bytes")
    if not _is_sha256(member.get("sha256")):
        raise RuntimeError("source contract train_sdf_member SHA-256 is invalid")
    if not isinstance(companion.get("archive_members"), dict):
        raise RuntimeError("source contract lacks locked companion archive member metadata")
    for role, spec in _source_artifact_specs(contract).items():
        _require_dict(spec, "{} contract entry".format(role))
        if not isinstance(spec.get("path"), str):
            raise RuntimeError("{} contract entry has no path".format(role))
        _require_int(spec.get("bytes"), "{} bytes".format(role))
        if not _is_sha256(spec.get("sha256")):
            raise RuntimeError("{} contract entry has invalid SHA-256".format(role))
    for role, spec in _manifest_specs(contract).items():
        _require_dict(spec, "{} contract entry".format(role))
        if not isinstance(spec.get("path"), str):
            raise RuntimeError("{} contract entry has no path".format(role))
        _require_int(spec.get("bytes"), "{} bytes".format(role))
        if not _is_sha256(spec.get("sha256")):
            raise RuntimeError("{} contract entry has invalid SHA-256".format(role))


def _verify_archive_members(companion_archive_path, companion, artifact_observations):
    members = _require_dict(companion.get("archive_members"), "official_companion.archive_members")
    expected = {
        "companion_data_csv_gz": "data_csv",
        "companion_split_dict_pt": "split_dict",
    }
    if set(members) != set(expected.values()):
        raise RuntimeError("source contract companion archive member set is not exact")
    results = {}
    with zipfile.ZipFile(str(companion_archive_path), "r") as archive:
        corrupted = archive.testzip()
        if corrupted is not None:
            raise RuntimeError("locked companion ZIP has a corrupt member: {}".format(corrupted))
        infos_by_name = {}
        for info in archive.infolist():
            infos_by_name.setdefault(info.filename, []).append(info)
        for artifact_role, member_key in expected.items():
            spec = _require_dict(members[member_key], "archive_members.{}".format(member_key))
            member_name = spec.get("zip_member")
            if not isinstance(member_name, str) or not member_name:
                raise RuntimeError("archive member {} has no zip_member".format(member_key))
            infos = infos_by_name.get(member_name, [])
            if len(infos) != 1:
                raise RuntimeError("archive member {} is absent or duplicated".format(member_name))
            info = infos[0]
            expected_crc32 = spec.get("crc32")
            if not isinstance(expected_crc32, str) or len(expected_crc32) != 8:
                raise RuntimeError("archive member {} has invalid crc32".format(member_key))
            if "{:08x}".format(int(info.CRC)) != expected_crc32:
                raise RuntimeError("archive member {} CRC32 differs from contract".format(member_key))
            if int(info.compress_size) != _require_int(spec.get("compressed_bytes"), "archive member compressed_bytes"):
                raise RuntimeError("archive member {} compressed size differs from contract".format(member_key))
            if int(info.file_size) != _require_int(spec.get("uncompressed_bytes"), "archive member uncompressed_bytes"):
                raise RuntimeError("archive member {} uncompressed size differs from contract".format(member_key))
            with archive.open(info, "r") as handle:
                observed_bytes, observed_sha256 = _sha256_stream(handle)
            target = artifact_observations[artifact_role]
            if observed_bytes != target["bytes"] or observed_sha256 != target["sha256"]:
                raise RuntimeError(
                    "archive member {} does not equal its separately locked extracted artifact".format(member_name)
                )
            results[member_key] = {
                "zip_member": member_name,
                "crc32": expected_crc32,
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": observed_bytes,
                "sha256": observed_sha256,
            }
    return results


def _verify_train_sdf_member(archive_path, source):
    """Bind the selected stream member to the hash-locked tar.gz archive."""
    spec = _require_dict(source.get("train_sdf_member"), "source.train_sdf_member")
    expected_name = spec["tar_member_name"]
    expected_bytes = _require_int(spec["uncompressed_bytes"], "source.train_sdf_member uncompressed_bytes")
    found = None
    with tarfile.open(str(archive_path), mode="r|gz") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".sdf"):
                found = member
                break
    if found is None:
        raise RuntimeError("locked train archive contains no regular SDF member")
    if found.name != expected_name or int(found.size) != expected_bytes:
        raise RuntimeError("locked train SDF member name or uncompressed byte count differs from contract")
    return {
        "tar_member_name": found.name,
        "member_type": "regular_file",
        "uncompressed_bytes": int(found.size),
        "sha256": spec["sha256"],
    }


def _verify_train_sdf_member_hash_report(observation, train_sdf_member):
    """Bind the one-time streamed member digest to the verified outer archive."""
    try:
        with open(observation["path"], "rb") as handle:
            observed = handle.read()
    except Exception as exc:
        raise RuntimeError("cannot read verified train SDF member hash report") from exc
    expected = "{}  -\n".format(train_sdf_member["sha256"]).encode("ascii")
    if observed != expected:
        raise RuntimeError("verified train SDF member hash report differs from the locked member SHA-256")


def _read_verified_json(observation, label):
    try:
        with open(observation["path"], "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise RuntimeError("cannot parse verified {}: {}".format(label, type(exc).__name__)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("verified {} must contain a JSON object".format(label))
    return value


def _verify_manifest_semantics(contract, artifact_observations, manifest_observations):
    """Check the fixed semantic facts after each supporting manifest is hashed."""
    source = contract["source"]
    companion = contract["official_companion"]
    source_manifest = _read_verified_json(manifest_observations["train_sdf_source_manifest"], "train SDF manifest")
    if source_manifest.get("source_id") != "ogb-pcqm4mv2-train-3d-v1":
        raise RuntimeError("verified train SDF manifest source_id differs from the frozen source")
    if source_manifest.get("official_train_3d_sdf_count") != source["official_train_sdf_records"]:
        raise RuntimeError("verified train SDF manifest record count differs from source contract")
    if source_manifest.get("expected_md5") != source.get("official_md5"):
        raise RuntimeError("verified train SDF manifest MD5 claim differs from source contract")

    companion_manifest = _read_verified_json(manifest_observations["companion_source_manifest"], "companion source manifest")
    archive_claim = companion_manifest.get("files", {}).get("archive", {})
    selected_claims = companion_manifest.get("files", {}).get("selected_members", {})
    if archive_claim.get("sha256") != artifact_observations["companion_archive"]["sha256"]:
        raise RuntimeError("verified companion source manifest archive hash disagrees with locked archive")
    if selected_claims.get("raw/data.csv.gz", {}).get("sha256") != artifact_observations["companion_data_csv_gz"]["sha256"]:
        raise RuntimeError("verified companion source manifest CSV hash disagrees with locked CSV")
    if selected_claims.get("split/split_dict.pt", {}).get("sha256") != artifact_observations["companion_split_dict_pt"]["sha256"]:
        raise RuntimeError("verified companion source manifest split hash disagrees with locked split")

    content = _read_verified_json(manifest_observations["companion_content_validation"], "companion content validation")
    expected = companion.get("validated_invariants")
    if not isinstance(expected, dict):
        raise RuntimeError("source contract lacks validated_invariants")
    comparisons = {
        "csv_rows": content.get("csv_rows"),
        "csv_idx_is_zero_based_contiguous": content.get("csv_idx_is_zero_based_contiguous"),
        "train_split_records": content.get("split_counts", {}).get("train"),
        "train_split_is_contiguous_prefix": content.get("train_is_contiguous_prefix"),
        "train_split_min": (content.get("split_minmax", {}).get("train") or [None, None])[0],
        "train_split_max": (content.get("split_minmax", {}).get("train") or [None, None])[1],
    }
    for key, observed in comparisons.items():
        if expected.get(key) != observed:
            raise RuntimeError("verified companion content invariant {} differs from contract".format(key))
    overlap_counts = content.get("split_overlap_counts")
    if not isinstance(overlap_counts, dict) or any(value != 0 for value in overlap_counts.values()):
        raise RuntimeError("verified companion split overlap summary is not zero")


@dataclass(frozen=True)
class VerifiedPCQMInputs:
    """Opaque-ish result returned only after the full source chain has passed."""

    source_contract_path: str
    source_contract_sha256: str
    source_record_count: int
    artifact_observations: tuple
    manifest_observations: tuple
    archive_member_observations: tuple
    train_sdf_member: tuple

    def _lookup(self, values, key):
        for candidate_key, candidate_value in values:
            if candidate_key == key:
                return candidate_value
        raise KeyError(key)

    def artifact(self, role):
        return self._lookup(self.artifact_observations, role)

    def manifest(self, role):
        return self._lookup(self.manifest_observations, role)

    def report(self):
        return {
            "verification_schema_version": "most-t5-r1/pcqm-verified-input-lock/v1",
            "source_contract_path": self.source_contract_path,
            "source_contract_sha256": self.source_contract_sha256,
            "source_record_count": self.source_record_count,
            "artifacts": {key: value for key, value in self.artifact_observations},
            "manifests": {key: value for key, value in self.manifest_observations},
            "companion_archive_members": {key: value for key, value in self.archive_member_observations},
            "train_sdf_member": {key: value for key, value in self.train_sdf_member},
        }


def verify_pcqm_inputs(source_contract_path, archive_path, data_csv_path, split_dict_path):
    """Verify all trusted PCQM inputs before parsing or deserialization.

    The caller-supplied archive/CSV/split paths must resolve to exactly the
    canonical remote files declared by the v2 contract.  Hashing a path only
    after it is parsed is explicitly forbidden by the R1 source policy.
    """
    contract_path, contract = _load_json_object(source_contract_path, "source contract")
    _verify_contract_shape(contract)
    specs = _source_artifact_specs(contract)
    actual_paths = {
        "train_3d_sdf_archive": archive_path,
        "companion_archive": specs["companion_archive"]["path"],
        "companion_data_csv_gz": data_csv_path,
        "companion_split_dict_pt": split_dict_path,
    }
    artifacts = {
        role: _verify_locked_file(actual_paths[role], specs[role], role)
        for role in _REQUIRED_ARTIFACTS
    }
    manifests = {
        role: _verify_locked_file(spec["path"], spec, role)
        for role, spec in _manifest_specs(contract).items()
    }
    archive_members = _verify_archive_members(
        artifacts["companion_archive"]["path"], contract["official_companion"], artifacts
    )
    train_sdf_member = _verify_train_sdf_member(artifacts["train_3d_sdf_archive"]["path"], contract["source"])
    _verify_train_sdf_member_hash_report(manifests["train_sdf_member_hash_report"], train_sdf_member)
    _verify_manifest_semantics(contract, artifacts, manifests)
    return VerifiedPCQMInputs(
        source_contract_path=str(contract_path),
        source_contract_sha256=sha256_file(contract_path),
        source_record_count=int(contract["source"]["official_train_sdf_records"]),
        artifact_observations=tuple((key, artifacts[key]) for key in _REQUIRED_ARTIFACTS),
        manifest_observations=tuple((key, manifests[key]) for key in _REQUIRED_MANIFESTS),
        archive_member_observations=tuple((key, archive_members[key]) for key in sorted(archive_members)),
        train_sdf_member=tuple(sorted(train_sdf_member.items())),
    )


def load_verified_split_dict(verified_inputs, allow_unsafe_legacy_torch_load):
    """Deserialize only the split file bound by a :class:`VerifiedPCQMInputs`.

    ``weights_only=True`` remains preferred.  The known OGB NumPy payload can
    require the legacy fallback; that fallback is permitted only after the
    official archive, extracted member, and split file were all SHA-locked in
    the current process.
    """
    if not isinstance(verified_inputs, VerifiedPCQMInputs):
        raise TypeError("split_dict.pt may be loaded only from a VerifiedPCQMInputs lock")
    split = verified_inputs.artifact("companion_split_dict_pt")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read split_dict.pt") from exc
    try:
        value = torch.load(split["path"], map_location="cpu", weights_only=True)
        return value, {
            "method": "torch.load(weights_only=True)",
            "unsafe_fallback_authorized_by": None,
            "verified_companion_split_dict_pt_sha256": split["sha256"],
            "source_contract_sha256": verified_inputs.source_contract_sha256,
        }
    except (TypeError, pickle.UnpicklingError, RuntimeError) as exc:
        if not allow_unsafe_legacy_torch_load:
            raise RuntimeError(
                "safe torch.load(weights_only=True) could not read the verified official split_dict.pt ({}); "
                "re-run only with an explicit --allow-unsafe-legacy-torch-load acknowledgement".format(
                    type(exc).__name__
                )
            ) from exc
        value = torch.load(split["path"], map_location="cpu", weights_only=False)
        return value, {
            "method": "torch.load(legacy_unsafe_acknowledged)",
            "unsafe_fallback_authorized_by": "verified_companion_split_dict_pt_sha256",
            "verified_companion_split_dict_pt_sha256": split["sha256"],
            "source_contract_sha256": verified_inputs.source_contract_sha256,
        }
