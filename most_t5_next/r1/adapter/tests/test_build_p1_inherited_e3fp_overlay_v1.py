import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from most_t5_next.r1.adapter import build_p1_inherited_e3fp_overlay_v1 as subject
from most_t5_next.r1.adapter import sidecar_v2_codec


class _Conformer:
    def __init__(self, coordinates):
        self.coordinates = np.asarray(coordinates, dtype=np.float64)

    def GetPositions(self):
        return self.coordinates


class _Mol:
    def __init__(self, coordinates):
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.properties = {}

    def GetNumAtoms(self):
        return int(self.coordinates.shape[0])

    def GetConformer(self, index):
        if index != 0:
            raise IndexError(index)
        return _Conformer(self.coordinates)

    def SetProp(self, key, value):
        self.properties[key] = value


class _Shell:
    def __init__(self, center, radius, identifier, atoms, *, duplicate=None):
        self.center_atom = center
        self.radius = radius
        self.identifier = identifier
        self.substruct = frozenset(atoms)
        self.is_duplicate = duplicate is not None
        self.duplicate = duplicate


class _Fingerprinter:
    bits = 4096
    level = 3
    radius_multiplier = 1.5
    stereo = True
    include_disconnected = True
    rdkit_invariants = True
    exclude_floating = False
    remove_duplicate_substructs = True
    fp_type = type("Fingerprint", (), {})

    def __init__(self, shells):
        self.all_shells = list(shells)


class _Fingerprint:
    def __init__(self, indices):
        self.indices = tuple(indices)


def _unsigned(identifier):
    return int(identifier) & 0xFFFFFFFF


class _ReadTransaction:
    def __init__(self, payloads):
        self.payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def stat(self):
        return {"entries": len(self.payloads)}

    def get(self, key):
        return self.payloads.get(key)


class _ReadEnvironment:
    def __init__(self, payloads):
        self.payloads = payloads
        self.closed = False

    def begin(self, *, write):
        if write:
            raise AssertionError("read-only fixture must not open a write transaction")
        return _ReadTransaction(self.payloads)

    def close(self):
        self.closed = True


class _ReadOnlyLMDB:
    def __init__(self, payloads):
        self.payloads = payloads
        self.open_calls = []
        self.environment = None

    def open(self, path, **kwargs):
        self.open_calls.append((path, kwargs))
        if kwargs.get("readonly") is not True or kwargs.get("lock") is not False:
            raise AssertionError("overlay loader must open LMDB read-only without a lock")
        self.environment = _ReadEnvironment(self.payloads)
        return self.environment


class InheritedE3FPOverlayTests(unittest.TestCase):
    def _fixture(self, *, schedule_index=0):
        ordinal = subject.frozen_schedule()[schedule_index]
        coordinates = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        raw = np.asarray([[6, -1, -1, -1], [8, 5, -1, -1]], dtype=np.int32)
        mapping = np.asarray([0, 1], dtype=np.int32)
        record = {
            "record_schema_version": subject.EXPECTED_BASE_RECORD_SCHEMA,
            "member": {
                "member_id": f"ogb_pcqm4mv2_train_row_index:{ordinal}",
                "sdf_record_index": ordinal,
                "storage_key": f"{ordinal:09d}",
                "source_address_sha256": "1" * 64,
            },
            "atom_universe": {
                "source_atom_count": 2,
                "model_atom_count": 2,
                "model_to_source_atom_index": mapping,
                "geometry_mol_identity_sha256": "2" * 64,
            },
            "geometry": {
                "coordinates": coordinates,
                "coordinates_sha256": subject._array_sha256(coordinates),
                "e3fp": raw,
                "e3fp_sha256": subject._array_sha256(raw),
            },
        }
        membership = {
            "disposition": "admit",
            "sdf_record_index": ordinal,
            "record_storage_key": f"{ordinal:09d}",
            "member_id": record["member"]["member_id"],
            "record_content_sha256": sidecar_v2_codec.logical_record_sha256(np, record),
            "sidecar_id": "fixture-release",
        }
        binding = {"record": record, "membership": membership, "shard_index": schedule_index}
        geometry_mol = _Mol(coordinates)
        accepted = _Shell(0, 0.0, 6, {0, 1})
        level_zero = _Shell(1, 0.0, 8, {1})
        duplicate = _Shell(1, 1.5, 5, {0, 1}, duplicate=accepted)
        fingerprinter = _Fingerprinter([accepted, level_zero, duplicate])

        def generate(mol, fprint_params):
            return [_Fingerprint({6, 8})], fingerprinter

        e3fp_api = {
            "fprints_from_mol_verbose": generate,
            "signed_to_unsigned_int": _unsigned,
        }
        return binding, geometry_mol, e3fp_api

    def _build(self, binding, geometry_mol, e3fp_api, *, schedule_index=0):
        with mock.patch.object(
            subject.projection,
            "tag_source_atoms",
            return_value=(object(), 2, [0, 1]),
        ), mock.patch.object(
            subject.projection,
            "project_hydrogens",
            return_value=(geometry_mol, [0, 1]),
        ):
            return subject.build_overlay_record(
                object(),
                np,
                schedule_index=schedule_index,
                binding=binding,
                source_mol=object(),
                e3fp_api=e3fp_api,
                source_full_release_manifest_sha256="a" * 64,
                source_logical_release_root_sha256="b" * 64,
                inheritance_implementation_sha256="c" * 64,
            )

    @staticmethod
    def _remove_loader_fixture(root):
        for path in (
            root / subject.MANIFEST_NAME,
            root / subject.FINAL_MEMBERSHIP_NAME,
            root / subject.FINAL_SCHEDULE_NAME,
            root / subject.FINAL_LMDB_NAME / "data.mdb",
        ):
            if path.is_file():
                path.unlink()
        lmdb_path = root / subject.FINAL_LMDB_NAME
        if lmdb_path.is_dir():
            lmdb_path.rmdir()
        if root.is_dir():
            root.rmdir()

    def _loader_fixture(self):
        root = Path(tempfile.mkdtemp(prefix="most-t5-overlay-reader-"))
        self.addCleanup(self._remove_loader_fixture, root)
        binding, geometry_mol, e3fp_api = self._fixture()
        template, _ = self._build(binding, geometry_mol, e3fp_api)
        payloads = {}
        memberships = []
        for schedule_index, ordinal in enumerate(subject.frozen_schedule()):
            record = copy.deepcopy(template)
            record["selection"]["schedule_index"] = schedule_index
            record["member"]["member_id"] = f"ogb_pcqm4mv2_train_row_index:{ordinal}"
            record["member"]["sdf_record_index"] = ordinal
            record["member"]["record_storage_key"] = f"{ordinal:09d}"
            subject.validate_overlay_record(np, record, subject.schedule_sha256())
            payload = sidecar_v2_codec.encode_record(np, record)
            membership = subject._membership_row(np, record, payload)
            subject._validate_membership(membership, record)
            payloads[membership["record_storage_key"].encode("ascii")] = payload
            memberships.append(membership)

        membership_path = root / subject.FINAL_MEMBERSHIP_NAME
        schedule_path = root / subject.FINAL_SCHEDULE_NAME
        lmdb_path = root / subject.FINAL_LMDB_NAME
        subject._write_jsonl(membership_path, memberships)
        subject._write_json(schedule_path, subject.build_schedule_document())
        lmdb_path.mkdir()
        (lmdb_path / "data.mdb").write_bytes(b"read-only-lmdb-fixture")
        manifest = {
            "schema_version": subject.MANIFEST_SCHEMA,
            "status": "pass",
            "sample_scope_only": True,
            "training_admission": False,
            "selection": {
                "schedule_schema_version": subject.SCHEDULE_SCHEMA,
                "schedule_sha256": subject.schedule_sha256(),
                "sample_count": subject.SAMPLE_COUNT,
                "source_record_count": subject.SOURCE_RECORD_COUNT,
                "selection_rule": subject.SCHEDULE_RULE,
                "no_next_admitted_replacement": True,
            },
            "counts": {
                "scheduled_records": subject.SAMPLE_COUNT,
                "overlay_records": subject.SAMPLE_COUNT,
                "raw_parity_count": subject.SAMPLE_COUNT,
                "failed_records": 0,
            },
            "artifacts": {
                "membership": subject.release_reader._artifact(
                    membership_path, subject.SAMPLE_COUNT
                ),
                "schedule": subject.release_reader._artifact(schedule_path),
                "overlay_lmdb": subject._lmdb_artifact(lmdb_path),
            },
        }
        subject._write_json(root / subject.MANIFEST_NAME, manifest)
        return root, payloads, manifest

    def test_schedule_is_exactly_128_cross_shard_ordinals(self):
        ordinals = subject.frozen_schedule()
        self.assertEqual(len(ordinals), 128)
        self.assertEqual(ordinals[0], 0)
        self.assertEqual(ordinals[-1], 3_352_210)
        self.assertEqual(len(set(ordinals)), 128)
        self.assertTrue(all(left < right for left, right in zip(ordinals, ordinals[1:])))
        manifest = {
            "shards": [
                {
                    "shard_index": index,
                    "range_start": index * 25_000,
                    "range_end": min((index + 1) * 25_000, subject.SOURCE_RECORD_COUNT),
                }
                for index in range(136)
            ]
        }
        shard_indices = {
            subject.release_reader._shard_for_ordinal(manifest, ordinal)["shard_index"]
            for ordinal in ordinals
        }
        self.assertEqual(len(shard_indices), 128)
        document = subject.build_schedule_document()
        self.assertEqual(document["ordinals"], list(ordinals))
        self.assertTrue(document["sample_scope_only"])
        self.assertFalse(document["training_admission"])

    def test_overlay_record_requires_raw_parity_and_codec_round_trip(self):
        binding, geometry_mol, e3fp_api = self._fixture()
        record, resolved = self._build(binding, geometry_mol, e3fp_api)
        self.assertEqual(record["e3fp"]["inherited"].tolist(), [[6, -1, -1, -1], [8, 6, -1, -1]])
        self.assertTrue(bool(record["e3fp"]["duplicate_mask"][1, 1]))
        self.assertTrue(record["e3fp"]["raw_matches_frozen"])
        self.assertEqual(resolved["bits"], 4096)
        payload = sidecar_v2_codec.encode_record(np, record)
        decoded, logical_hash = sidecar_v2_codec.decode_record(np, payload)
        subject.validate_overlay_record(np, decoded, subject.schedule_sha256())
        self.assertTrue(np.array_equal(decoded["e3fp"]["inherited"], record["e3fp"]["inherited"]))
        self.assertTrue(np.array_equal(decoded["e3fp"]["duplicate_mask"], record["e3fp"]["duplicate_mask"]))
        self.assertEqual(logical_hash, sidecar_v2_codec.logical_record_sha256(np, record))

    def test_raw_mismatch_fails_before_record_encoding(self):
        binding, geometry_mol, e3fp_api = self._fixture()
        binding["record"]["geometry"]["e3fp"][1, 1] = 99
        binding["record"]["geometry"]["e3fp_sha256"] = subject._array_sha256(
            binding["record"]["geometry"]["e3fp"]
        )
        binding["membership"]["record_content_sha256"] = sidecar_v2_codec.logical_record_sha256(
            np, binding["record"]
        )
        with mock.patch.object(sidecar_v2_codec, "encode_record") as encode:
            with self.assertRaisesRegex(subject.InheritedE3FPOverlayError, "raw E3FP"):
                self._build(binding, geometry_mol, e3fp_api)
            encode.assert_not_called()

    def test_mapping_mismatch_fails_before_e3fp_or_write(self):
        binding, geometry_mol, e3fp_api = self._fixture()
        with mock.patch.object(
            subject.projection,
            "tag_source_atoms",
            return_value=(object(), 2, [0, 1]),
        ), mock.patch.object(
            subject.projection,
            "project_hydrogens",
            return_value=(geometry_mol, [1, 0]),
        ), mock.patch.object(
            subject.inheritance,
            "generate_e3fp_projection_pair",
            side_effect=AssertionError("E3FP must not run"),
        ) as generate, mock.patch.object(sidecar_v2_codec, "encode_record") as encode:
            with self.assertRaisesRegex(subject.InheritedE3FPOverlayError, "atom mapping"):
                subject.build_overlay_record(
                    object(),
                    np,
                    schedule_index=0,
                    binding=binding,
                    source_mol=object(),
                    e3fp_api=e3fp_api,
                    source_full_release_manifest_sha256="a" * 64,
                    source_logical_release_root_sha256="b" * 64,
                    inheritance_implementation_sha256="c" * 64,
                )
            generate.assert_not_called()
            encode.assert_not_called()

    def test_output_uses_base_storage_key_and_binds_base_content(self):
        binding, geometry_mol, e3fp_api = self._fixture()
        record, _ = self._build(binding, geometry_mol, e3fp_api)
        self.assertEqual(
            record["member"]["record_storage_key"],
            binding["membership"]["record_storage_key"],
        )
        self.assertEqual(
            record["member"]["base_record_content_sha256"],
            binding["membership"]["record_content_sha256"],
        )
        payload = sidecar_v2_codec.encode_record(np, record)
        membership = subject._membership_row(np, record, payload)
        subject._validate_membership(membership, record)
        self.assertEqual(membership["record_storage_key"], binding["membership"]["record_storage_key"])

    def test_swapped_base_key_or_content_binding_is_rejected(self):
        binding, _, _ = self._fixture()
        wrong_key = copy.deepcopy(binding)
        wrong_key["membership"]["record_storage_key"] = "000000001"
        with self.assertRaisesRegex(subject.InheritedE3FPOverlayError, "storage key"):
            subject.validate_base_binding(np, wrong_key, 0)

        wrong_content = copy.deepcopy(binding)
        wrong_content["membership"]["record_content_sha256"] = "f" * 64
        with self.assertRaisesRegex(subject.InheritedE3FPOverlayError, "logical content"):
            subject.validate_base_binding(np, wrong_content, 0)

    def test_overlay_release_validator_does_not_require_topology_lock(self):
        binding, _, _ = self._fixture()
        binding["record"]["topology"] = {"linearizer_spec_sha256": "different"}
        subject.validate_overlay_release_record(
            binding["record"], binding["membership"], subject.frozen_schedule()[0]
        )

    def test_readonly_loader_returns_all_ordinal_bindings_and_pass_manifest(self):
        root, payloads, expected_manifest = self._loader_fixture()
        lmdb_module = _ReadOnlyLMDB(payloads)
        bound, manifest = subject.load_overlay_readonly(
            root, np=np, lmdb_module=lmdb_module
        )
        self.assertEqual(tuple(bound), subject.frozen_schedule())
        self.assertEqual(manifest, expected_manifest)
        first_ordinal = subject.frozen_schedule()[0]
        self.assertEqual(
            bound[first_ordinal]["record"]["member"]["sdf_record_index"], first_ordinal
        )
        self.assertEqual(
            bound[first_ordinal]["membership"]["schedule_index"], 0
        )
        self.assertEqual(len(lmdb_module.open_calls), 1)
        self.assertTrue(lmdb_module.environment.closed)

    def test_readonly_loader_rejects_lmdb_membership_wire_mismatch(self):
        root, payloads, _ = self._loader_fixture()
        first_key = f"{subject.frozen_schedule()[0]:09d}".encode("ascii")
        payloads[first_key] = b"x" + payloads[first_key][1:]
        with self.assertRaisesRegex(
            subject.InheritedE3FPOverlayError, "wire bytes differ"
        ):
            subject.load_overlay_readonly(
                root, np=np, lmdb_module=_ReadOnlyLMDB(payloads)
            )

    def test_readonly_loader_rejects_nonpass_manifest(self):
        root, payloads, manifest = self._loader_fixture()
        manifest["status"] = "failed"
        (root / subject.MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            subject.InheritedE3FPOverlayError, "pass sample-scope"
        ):
            subject.load_overlay_readonly(
                root, np=np, lmdb_module=_ReadOnlyLMDB(payloads)
            )

    def test_readonly_loader_rejects_rewritten_schedule_even_if_rebound(self):
        root, payloads, manifest = self._loader_fixture()
        schedule_path = root / subject.FINAL_SCHEDULE_NAME
        schedule = subject.build_schedule_document()
        schedule["ordinals"][1] += 1
        schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
        manifest["artifacts"]["schedule"] = subject.release_reader._artifact(schedule_path)
        (root / subject.MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            subject.InheritedE3FPOverlayError, "not the frozen 128-record schedule"
        ):
            subject.load_overlay_readonly(
                root, np=np, lmdb_module=_ReadOnlyLMDB(payloads)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
