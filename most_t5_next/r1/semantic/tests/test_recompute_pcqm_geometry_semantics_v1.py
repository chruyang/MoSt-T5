import copy
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from most_t5_next.r1.semantic import recompute_pcqm_geometry_semantics_v1 as gate


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "p0_pcqm_independent_semantic_recompute_contract_v1.json"
IDENTITY_CONTRACT_PATH = ROOT / "contracts" / "pcqm4mv2_identity_normalization_contract.json"
PAYLOAD_CONTRACT_PATH = ROOT / "contracts" / "p1_pcqm_geometry_payload_format_contract.json"
PRODUCTION_BUILDER_PATH = ROOT / "adapter" / "build_pcqm_p1_geometry_sidecar.py"
ORDINAL_517_IPC_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "ordinal_000000517_worker_ipc_diagnostic_v1.json"
)


class _Shell:
    def __init__(self, center, radius, identifier):
        self.center_atom = center
        self.radius = radius
        self.identifier = identifier


class _Fingerprinter:
    radius_multiplier = 1.718

    def __init__(self, shells):
        self.all_shells = shells


class SemanticGateUnitTests(unittest.TestCase):
    @staticmethod
    def _embedded(smiles, seed=4321):
        source = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDGv2()
        params.randomSeed = seed
        if int(AllChem.EmbedMolecule(source, params)) != 0:
            raise AssertionError("fixture embedding failed")
        return source

    @staticmethod
    def _source_lock(ordinal):
        archive_sha = "a" * 64
        locked = {
            "tar_member_name": "fixture.sdf",
            "member_type": "regular_file",
            "uncompressed_bytes": 1,
            "sha256": "b" * 64,
        }
        address = gate.source_address_sha256(archive_sha, locked, ordinal)
        return archive_sha, locked, address

    @staticmethod
    def _running_script_binding():
        path = Path(gate.__file__).resolve()
        size, digest = gate.sha256_file(path)
        return {"path": str(path), "bytes": size, "sha256": digest}

    @staticmethod
    def _synthetic_release_rehash(phase):
        observations = [
            {
                "kind": "synthetic_artifact",
                "relative_path": "shard-000000/membership.jsonl",
                "bytes": 1,
                "sha256": "a" * 64,
            }
        ]
        return {
            "phase": phase,
            "file_count": 1,
            "total_bytes": 1,
            "aggregate_sha256": gate.sha256_json(observations),
            "observations": observations,
        }

    def _write_valid_staged_bundle(self, output, semantic_status="pass",
                                   header_updates=None, ledger_relative=None):
        script_binding = self._running_script_binding()
        release_id = "synthetic-release"
        header = {
            "document_kind": "semantic_recompute_result_header",
            "schema_version": gate.LEDGER_SCHEMA,
            "release_id": release_id,
            "release_manifest_sha256": gate.FROZEN_RELEASE_MANIFEST_SHA256,
            "structural_audit_report_sha256": gate.FROZEN_STRUCTURAL_AUDIT_SHA256,
            "semantic_plan_sha256": gate.FROZEN_SEMANTIC_PLAN_SHA256,
            "semantic_gate_contract_sha256": gate.SEMANTIC_CONTRACT_SHA256,
            "semantic_gate_script_sha256": script_binding["sha256"],
            "selected_admitted_count": 1,
            "selected_reject_count": 1,
            "raw_smiles_or_molecule_output": False,
        }
        if header_updates:
            header.update(header_updates)
        ledger_path = output / gate.RESULT_LEDGER_FILE_NAME
        gate.write_jsonl_new(ledger_path, [header])
        ledger_bytes, ledger_sha = gate.sha256_file(ledger_path)
        result_binding = {
            "relative_path": ledger_relative or gate.RESULT_LEDGER_FILE_NAME,
            "bytes": ledger_bytes,
            "sha256": ledger_sha,
        }

        def external(file_name, digest, byte_count=1):
            return {
                "path": str((output.parent / file_name).resolve()),
                "bytes": byte_count,
                "sha256": digest,
            }

        archive_binding = external("train.sdf.tar.gz", "b" * 64, 2)
        csv_binding = external("data.csv.gz", "c" * 64, 3)
        before = self._synthetic_release_rehash("before_source_replay")
        after = self._synthetic_release_rehash(
            "after_source_replay_before_completion"
        )
        report = {
            "schema_version": gate.REPORT_SCHEMA,
            "created_utc": "2026-08-05T00:00:00+00:00",
            "semantic_recompute_status": semantic_status,
            "completion_status": "pending_authoritative_receipt",
            "authoritative_overall_gate_status": None,
            "release_id": release_id,
            "bindings": {
                "release_manifest": external(
                    "full_release_manifest.json",
                    gate.FROZEN_RELEASE_MANIFEST_SHA256,
                ),
                "structural_audit_report": external(
                    "independent_audit_report.json",
                    gate.FROZEN_STRUCTURAL_AUDIT_SHA256,
                ),
                "semantic_plan": external(
                    "semantic_review_plan.jsonl", gate.FROZEN_SEMANTIC_PLAN_SHA256
                ),
                "semantic_gate_contract": external(
                    "p0_pcqm_independent_semantic_recompute_contract_v1.json",
                    gate.SEMANTIC_CONTRACT_SHA256, gate.SEMANTIC_CONTRACT_BYTES,
                ),
                "identity_contract": external(
                    "pcqm4mv2_identity_normalization_contract.json",
                    gate.IDENTITY_CONTRACT_SHA256, gate.IDENTITY_CONTRACT_BYTES,
                ),
                "payload_contract": external(
                    "p1_pcqm_geometry_payload_format_contract.json",
                    gate.PAYLOAD_CONTRACT_SHA256, gate.PAYLOAD_CONTRACT_BYTES,
                ),
                "semantic_gate_script": script_binding,
                "source_archive": archive_binding,
                "official_data_csv": csv_binding,
                "result_ledger": result_binding,
            },
            "full_release_artifact_rehash": {
                "before_source_replay": before,
                "after_source_replay_before_completion": after,
            },
            "counts": {"selected_admitted": 1, "selected_reject": 1},
        }
        report[gate.REPORT_HASH_FIELD] = gate.report_payload_sha256(report)
        report_path = output / gate.STAGED_REPORT_FILE_NAME
        gate.write_json_new(report_path, report)
        final_checks = {
            "source_archive_final_rehash": archive_binding,
            "official_data_csv_final_rehash": csv_binding,
            "full_release_artifacts_final_rehash": after,
        }
        return {
            "release_id": release_id,
            "script_binding": script_binding,
            "report": report,
            "report_path": report_path,
            "ledger_path": ledger_path,
            "final_checks": final_checks,
        }

    @staticmethod
    def _completion_observation(bundle, phase):
        report_bytes, report_sha = gate.sha256_file(bundle["report_path"])
        ledger_bytes, ledger_sha = gate.sha256_file(bundle["ledger_path"])
        del report_bytes, ledger_bytes
        return {
            "phase": phase,
            "script_sha256": bundle["script_binding"]["sha256"],
            "critical_input_aggregate_sha256": "d" * 64,
            "e3fp_closure": {"phase": phase, "closure_sha256": "e" * 64},
            "ledger_sha256": ledger_sha,
            "staged_report_sha256": report_sha,
        }

    def test_contract_is_frozen_and_valid(self):
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        gate.validate_contract(contract)
        self.assertFalse(contract["pass_boundary"]["p1_training_admission"])
        self.assertEqual(
            contract["selection"]["expected_full_release_reject_count"], 13029
        )

    def test_strict_json_rejects_duplicate_keys_and_noncanonical_jsonl(self):
        with self.assertRaises(gate.GateError):
            gate.strict_json_bytes(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaises(gate.GateError):
            gate.strict_json_bytes(b'{"b": 2, "a": 1}', "noncanonical", True)
        self.assertEqual(
            gate.strict_json_bytes(b'{"a":1,"b":2}', "canonical", True),
            {"a": 1, "b": 2},
        )

    def test_independent_payload_decoder_round_trip_fixture(self):
        array = np.ascontiguousarray(np.asarray([3, -1, 9], dtype=np.int32))
        descriptor = gate._array_descriptor(array)
        record = {"array": array, "label": "fixture"}
        projected = {
            "array": {
                "__array_block__": 0,
                "dtype": descriptor["dtype"],
                "shape": descriptor["shape"],
                "order": "C",
                "sha256": descriptor["sha256"],
            },
            "label": "fixture",
        }
        header = {
            "payload_schema_version": gate.PAYLOAD_SCHEMA,
            "record": projected,
            "array_blocks": [
                {
                    "index": 0,
                    "dtype": descriptor["dtype"],
                    "shape": descriptor["shape"],
                    "order": "C",
                    "offset": 0,
                    "nbytes": int(array.nbytes),
                    "sha256": descriptor["sha256"],
                }
            ],
            "logical_record_sha256": gate.sha256_json(gate.logical_projection(np, record)),
        }
        header_raw = gate.canonical_json_bytes(header)
        payload = gate.MAGIC + struct.pack(">I", len(header_raw)) + header_raw + array.tobytes()
        decoded, logical_hash = gate.decode_payload(np, payload)
        self.assertTrue(np.array_equal(decoded["array"], array))
        self.assertEqual(logical_hash, header["logical_record_sha256"])
        damaged = bytearray(payload)
        damaged[-1] ^= 1
        with self.assertRaises(gate.GateError):
            gate.decode_payload(np, damaged)

    def test_rdkit_strict_and_connectivity_classification(self):
        same_a = Chem.MolFromSmiles("F[C@H](Cl)Br")
        same_b = Chem.MolFromSmiles("F[C@H](Cl)Br")
        opposite = Chem.MolFromSmiles("F[C@@H](Cl)Br")
        _, a = gate.canonical_forms(Chem, same_a)
        _, b = gate.canonical_forms(Chem, same_b)
        _, c = gate.canonical_forms(Chem, opposite)
        self.assertEqual(gate.classify_identity(a, b), "strict_isomeric_match")
        self.assertEqual(
            gate.classify_identity(a, c), "PCQM_STEREO_2D3D_DIVERGENCE"
        )
        self.assertNotEqual(a["strict_sha256"], c["strict_sha256"])
        self.assertEqual(a["connectivity_sha256"], c["connectivity_sha256"])

    def test_projected_atom_provenance_and_identity_are_deterministic(self):
        source = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDGv2()
        params.randomSeed = 12345
        self.assertEqual(int(AllChem.EmbedMolecule(source, params)), 0)
        first_hash = gate.molecule_identity_sha256(np, source)
        copied_hash = gate.molecule_identity_sha256(np, Chem.Mol(source))
        self.assertEqual(first_hash, copied_hash)
        geometry, mapping = gate.project_geometry_mol(Chem, np, source)
        self.assertEqual(geometry.GetNumAtoms(), 3)
        self.assertTrue(np.array_equal(mapping, np.asarray([0, 1, 2], dtype=np.int32)))
        changed = Chem.Mol(source)
        point = changed.GetConformer(0).GetAtomPosition(0)
        changed.GetConformer(0).SetAtomPosition(0, (point.x + 0.25, point.y, point.z))
        self.assertNotEqual(first_hash, gate.molecule_identity_sha256(np, changed))

    def test_worker_ipc_roundtrip_is_coordinate_only_float32_quantization(self):
        source = self._embedded("CCOC(=O)N[C@H](C)Cl", 20260805)
        worker, summary = gate.worker_ipc_roundtrip_summary(Chem, np, source)
        self.assertTrue(summary["atom_component_equal"])
        self.assertTrue(summary["bond_component_equal"])
        self.assertTrue(summary["conformer_count_equal"])
        self.assertTrue(summary["coordinates_equal_float32_roundtrip"])
        self.assertGreater(summary["coordinate_changed_value_count"], 0)
        self.assertLess(summary["coordinate_max_abs_delta"], 5e-7)
        self.assertNotEqual(
            summary["raw_identity_sha256"], summary["worker_identity_sha256"]
        )
        raw_geometry, raw_mapping = gate.project_geometry_mol(Chem, np, source)
        worker_geometry, worker_mapping = gate.project_geometry_mol(Chem, np, worker)
        self.assertTrue(np.array_equal(raw_mapping, worker_mapping))
        self.assertTrue(
            np.array_equal(
                np.asarray(raw_geometry.GetConformer(0).GetPositions(), dtype=np.float32),
                np.asarray(worker_geometry.GetConformer(0).GetPositions(), dtype=np.float32),
            )
        )
        _, raw_forms = gate.canonical_forms(Chem, raw_geometry)
        _, worker_forms = gate.canonical_forms(Chem, worker_geometry)
        self.assertEqual(raw_forms, worker_forms)

    def test_ordinal_517_nonraw_worker_ipc_fixture_matches_production(self):
        observation = json.loads(ORDINAL_517_IPC_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(observation["sdf_record_index"], 517)
        self.assertTrue(observation["continuous_raw_equals_single_record_raw"])
        self.assertTrue(observation["continuous_worker_equals_single_record_worker"])
        self.assertTrue(observation["atom_component_equal_after_worker_ipc"])
        self.assertTrue(observation["bond_component_equal_after_worker_ipc"])
        self.assertTrue(observation["coordinates_equal_float32_roundtrip"])
        self.assertEqual(
            observation["worker_source_mol_identity_sha256"],
            observation["production_reject_ledger_source_mol_identity_sha256"],
        )
        self.assertEqual(
            observation["worker_geometry_mol_identity_sha256"],
            observation["production_reject_ledger_geometry_mol_identity_sha256"],
        )
        self.assertNotEqual(
            observation["raw_source_mol_identity_sha256"],
            observation["worker_source_mol_identity_sha256"],
        )
        self.assertFalse(observation["raw_molecule_or_coordinates_serialized"])
        self.assertEqual(
            gate.sha256_file(PRODUCTION_BUILDER_PATH)[1],
            observation["production_builder_sha256"],
        )

    def test_isotopic_hydrogen_is_reproduced_as_terminal_projection_reject(self):
        source = Chem.AddHs(Chem.MolFromSmiles("[2H]C"))
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDGv2()
        params.randomSeed = 2468
        self.assertEqual(int(AllChem.EmbedMolecule(source, params)), 0)
        _, _, residual_h, non_e3fp = gate.project_hydrogen_candidate(Chem, np, source)
        self.assertEqual(residual_h, 1)
        self.assertGreaterEqual(non_e3fp, residual_h)
        with self.assertRaises(gate.GateError):
            gate.project_geometry_mol(Chem, np, source)

    def test_three_reject_classes_follow_their_first_terminal_stage(self):
        cases = [
            (
                "F[C@H](Cl)Br",
                "F[C@@H](Cl)Br",
                "PCQM_STEREO_2D3D_DIVERGENCE",
                "identity",
                "strict_mismatch_connectivity_match",
            ),
            (
                "F[C@H](Cl)Br",
                "CC",
                "PCQM_SDF_CSV_CONNECTIVITY_MISMATCH",
                "identity",
                "connectivity_mismatch",
            ),
        ]
        for index, (source_smiles, official_smiles, reason, stage, diagnostic) in enumerate(cases):
            ordinal = 10 + index
            source = self._embedded(source_smiles, 900 + index)
            geometry, _ = gate.project_geometry_mol(Chem, np, source)
            archive_sha, locked, address = self._source_lock(ordinal)
            expected = {
                "disposition": "reject",
                "shard_index": 0,
                "membership": {"source_address_sha256": address},
                "reject": {
                    "reason_code": reason,
                    "stage": stage,
                    "diagnostic_code": diagnostic,
                    "source_mol_identity_sha256": gate.molecule_identity_sha256(np, source),
                    "geometry_mol_identity_sha256": gate.molecule_identity_sha256(np, geometry),
                },
            }
            result = gate.compare_target(
                Chem, np, None, ordinal, source, official_smiles, expected,
                archive_sha, locked, Chem.rdBase.rdkitVersion,
            )
            self.assertEqual(result["status"], "pass", result["mismatch_codes"])
            self.assertEqual(result["recomputed_terminal_classification"], reason)
            self.assertFalse(result["e3fp_recomputed"])

        ordinal = 12
        source = self._embedded("[2H]C", 902)
        archive_sha, locked, address = self._source_lock(ordinal)
        expected = {
            "disposition": "reject",
            "shard_index": 0,
            "membership": {"source_address_sha256": address},
            "reject": {
                "reason_code": "HYDROGEN_PROJECTION_RESIDUAL_H",
                "stage": "hydrogen_projection",
                "diagnostic_code": "preflight_hydrogen_projection_residual_h",
                "source_mol_identity_sha256": gate.molecule_identity_sha256(np, source),
                "geometry_mol_identity_sha256": None,
            },
        }
        result = gate.compare_target(
            Chem, np, None, ordinal, source, "C", expected,
            archive_sha, locked, Chem.rdBase.rdkitVersion,
        )
        self.assertEqual(result["status"], "pass", result["mismatch_codes"])
        self.assertEqual(
            result["recomputed_terminal_classification"], "HYDROGEN_PROJECTION_RESIDUAL_H"
        )
        self.assertGreater(result["recomputed_residual_hydrogen_count"], 0)

    def test_admitted_branch_exactly_compares_identity_mapping_coordinates_and_e3fp(self):
        ordinal = 21
        source = self._embedded("CCO", 123)
        geometry, mapping = gate.project_geometry_mol(Chem, np, source)
        _, forms = gate.canonical_forms(Chem, geometry)
        coordinates = np.ascontiguousarray(
            np.asarray(geometry.GetConformer(0).GetPositions(), dtype=np.float32)
        )
        e3fp = np.ascontiguousarray(
            np.arange(geometry.GetNumAtoms() * 4, dtype=np.int32).reshape(geometry.GetNumAtoms(), 4)
        )
        resolved = {
            "bits": 4096,
            "level": 3,
            "radius_multiplier": 1.718,
            "stereo": True,
            "include_disconnected": True,
            "rdkit_invariants": True,
            "exclude_floating": False,
            "remove_duplicate_substructs": True,
            "fingerprint_type": "Fingerprint",
            "all_iters": True,
        }
        archive_sha, locked, address = self._source_lock(ordinal)
        logical_hash = "c" * 64
        expected = {
            "disposition": "admit",
            "shard_index": 0,
            "membership": {"source_address_sha256": address},
            "logical_hash": logical_hash,
            "record": {
                "member": {
                    "source_address_sha256": address,
                    "source_mol_identity_sha256": gate.molecule_identity_sha256(np, source),
                },
                "identity": {
                    "rdkit_version": Chem.rdBase.rdkitVersion,
                    "sdf_strict_smiles_sha256": forms["strict_sha256"],
                    "official_strict_smiles_sha256": forms["strict_sha256"],
                    "canonical_connectivity_sha256": forms["connectivity_sha256"],
                },
                "atom_universe": {
                    "geometry_mol_identity_sha256": gate.molecule_identity_sha256(np, geometry),
                    "model_to_source_atom_index": mapping,
                },
                "geometry": {
                    "coordinates": coordinates,
                    "coordinates_sha256": gate.sha256_bytes(coordinates.tobytes(order="C")),
                    "e3fp": e3fp,
                    "e3fp_sha256": gate.sha256_bytes(e3fp.tobytes(order="C")),
                    "e3fp_params_sha256": gate.sha256_json(resolved),
                },
            },
        }
        with mock.patch.object(gate, "generate_e3fp", return_value=(e3fp, resolved)):
            result = gate.compare_target(
                Chem, np, {}, ordinal, source, "CCO", expected,
                archive_sha, locked, Chem.rdBase.rdkitVersion,
            )
        self.assertEqual(result["status"], "pass", result["mismatch_codes"])
        self.assertEqual(result["record_content_sha256"], logical_hash)
        self.assertTrue(result["e3fp_recomputed"])

    def test_explicit_shell_matrix_uses_center_and_radius_not_list_position(self):
        shells = [
            _Shell(1, 0.0, 8),
            _Shell(0, 1.718, 5),
            _Shell(0, 0.0, 3),
            _Shell(1, 1.718, 7),
        ]
        matrix = gate.build_explicit_shell_matrix(
            np, _Fingerprinter(shells), lambda value: value & 0xFFFFFFFF, 2
        )
        self.assertEqual(matrix.tolist(), [[3, 5, -1, -1], [8, 7, -1, -1]])
        duplicate = shells + [_Shell(0, 1.718, 99)]
        with self.assertRaises(gate.GateError):
            gate.build_explicit_shell_matrix(
                np, _Fingerprinter(duplicate), lambda value: value & 0xFFFFFFFF, 2
            )

    def test_plan_selection_is_exact_and_not_post_hoc(self):
        release_hash = "1" * 64
        plan_hash = "2" * 64
        plan = [
            {
                "document_kind": "semantic_review_plan_header",
                "schema_version": gate.SEMANTIC_PLAN_SCHEMA,
                "release_id": "release",
                "release_manifest_sha256": release_hash,
                "semantic_recompute_executed_by_this_gate": False,
                "all_rejects_included": True,
                "admitted_sample_count": 1,
                "reject_review_count": 1,
            },
            {
                "document_kind": "admitted_payload_sample",
                "shard_index": 0,
                "sdf_record_index": 2,
                "record_storage_key": "000000002",
                "required_review": "wire_hash_decode_and_logical_structure",
            },
            {
                "document_kind": "reject_semantic_review",
                "shard_index": 0,
                "sdf_record_index": 3,
                "reason_code": "PCQM_STEREO_2D3D_DIVERGENCE",
                "stage": "identity",
                "selection_reason": "all_rejects_without_exception",
                "required_review": "independent_source_and_feature_semantic_recompute",
            },
        ]
        report = {
            "semantic_review_plan": {
                "relative_path": "semantic_review_plan.jsonl",
                "sha256": plan_hash,
            },
            "release_manifest_sha256": release_hash,
        }
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        contract = copy.deepcopy(contract)
        contract["selection"]["expected_full_release_admitted_sample_count"] = 1
        contract["selection"]["expected_full_release_reject_count"] = 1
        contract["reject_recompute"]["expected_reason_counts"] = {
            "PCQM_STEREO_2D3D_DIVERGENCE": 1
        }
        _, admitted, rejected = gate.validate_plan(
            plan, plan_hash, release_hash, report, contract
        )
        self.assertEqual(set(admitted), {2})
        self.assertEqual(set(rejected), {3})
        duplicate = plan + [dict(plan[1])]
        with self.assertRaises(gate.GateError):
            gate.validate_plan(duplicate, plan_hash, release_hash, report, contract)

    def test_selected_molblock_parser_keeps_explicit_hydrogens(self):
        source = Chem.AddHs(Chem.MolFromSmiles("CO"))
        params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDGv2()
        params.randomSeed = 777
        self.assertEqual(int(AllChem.EmbedMolecule(source, params)), 0)
        block = Chem.MolToMolBlock(source).encode("utf-8")
        parsed = gate._parse_selected_mol(Chem, block)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.GetNumAtoms(), source.GetNumAtoms())
        self.assertEqual(parsed.GetNumConformers(), 1)

    def test_executable_contract_pins_fail_before_mutated_bytes_are_parsed(self):
        semantic_snapshot, semantic_contract = gate.load_pinned_json_snapshot(
            CONTRACT_PATH, "semantic contract fixture", gate.SEMANTIC_CONTRACT_SHA256
        )
        self.assertEqual(semantic_contract["schema_version"], gate.CONTRACT_SCHEMA)
        self.assertEqual(semantic_snapshot["sha256"], gate.SEMANTIC_CONTRACT_SHA256)
        identity_snapshot, _ = gate.load_pinned_json_snapshot(
            IDENTITY_CONTRACT_PATH, "identity contract fixture",
            gate.IDENTITY_CONTRACT_SHA256, gate.IDENTITY_CONTRACT_BYTES,
        )
        payload_snapshot, _ = gate.load_pinned_json_snapshot(
            PAYLOAD_CONTRACT_PATH, "payload contract fixture",
            gate.PAYLOAD_CONTRACT_SHA256, gate.PAYLOAD_CONTRACT_BYTES,
        )
        self.assertEqual(identity_snapshot["bytes"], 4193)
        self.assertEqual(payload_snapshot["bytes"], 2228)

        descriptor, mutated_name = tempfile.mkstemp(prefix="r5-mutated-contract-", suffix=".json")
        os.close(descriptor)
        mutated_path = Path(mutated_name)
        try:
            mutated_path.write_bytes(CONTRACT_PATH.read_bytes() + b" ")
            with self.assertRaises(gate.GateError):
                gate.load_pinned_json_snapshot(
                    mutated_path, "mutated semantic contract", gate.SEMANTIC_CONTRACT_SHA256
                )
        finally:
            mutated_path.unlink()

    def test_replaced_plan_is_rejected_by_fixed_sha_before_jsonl_use(self):
        scratch = Path(tempfile.mkdtemp(prefix="r5-plan-"))
        plan_path = scratch / "semantic_review_plan.jsonl"
        try:
            original = gate.canonical_json_bytes({"document_kind": "header"}) + b"\n"
            plan_path.write_bytes(original)
            expected = gate.sha256_bytes(original)
            snapshot = gate.require_frozen_evidence_snapshot(
                plan_path, "synthetic plan", "semantic_review_plan.jsonl", expected
            )
            self.assertEqual(snapshot["sha256"], expected)
            plan_path.write_bytes(
                gate.canonical_json_bytes({"document_kind": "replacement"}) + b"\n"
            )
            with self.assertRaises(gate.GateError):
                gate.require_frozen_evidence_snapshot(
                    plan_path, "replaced semantic plan", "semantic_review_plan.jsonl",
                    expected,
                )
        finally:
            plan_path.unlink()
            os.rmdir(scratch)

    def test_full_release_rehash_catches_unsampled_payload_index_tamper(self):
        scratch = Path(tempfile.mkdtemp(prefix="r5-release-"))
        shard = scratch / "shard-000000"
        records = shard / "geometry_records.lmdb"
        records.mkdir(parents=True)
        artifact_paths = {
            "geometry_records_lmdb_data": records / "data.mdb",
            "membership": shard / "membership.jsonl",
            "reject_ledger": shard / "reject_ledger.jsonl",
            "payload_index": shard / "payload_index.jsonl",
            "motif_census": shard / "motif_census.jsonl",
        }
        artifact_paths["geometry_records_lmdb_data"].write_bytes(b"lmdb")
        artifact_paths["membership"].write_bytes(b"membership\n")
        artifact_paths["reject_ledger"].write_bytes(b"")
        artifact_paths["payload_index"].write_bytes(b"unsampled-index\n")
        artifact_paths["motif_census"].write_bytes(b"motif\n")
        artifacts = {}
        for role, relative in gate.SHARD_ARTIFACT_PATHS.items():
            path = shard / Path(relative)
            size, digest = gate.sha256_file(path)
            artifacts[role] = {
                "relative_path": relative, "bytes": size, "sha256": digest
            }
        shard_manifest = {
            "schema_version": gate.SHARD_MANIFEST_SCHEMA,
            "release_status": "complete",
            "release_id": "synthetic-release",
            "shard_index": 0,
            "range_start": 0,
            "range_end": 1,
            "artifacts": artifacts,
        }
        shard_manifest_path = shard / "shard_manifest.json"
        shard_manifest_path.write_bytes(gate.canonical_json_bytes(shard_manifest) + b"\n")
        global_census = scratch / "motif_census.jsonl"
        global_census.write_bytes(b"global-motif\n")
        global_size, global_sha = gate.sha256_file(global_census)
        manifest = {
            "release_id": "synthetic-release",
            "shards": [
                {
                    "shard_index": 0,
                    "range_start": 0,
                    "range_end": 1,
                    "shard_manifest_sha256": gate.sha256_file(shard_manifest_path)[1],
                }
            ],
            "global_motif_census": {
                "relative_path": "motif_census.jsonl",
                "bytes": global_size,
                "sha256": global_sha,
            },
        }
        try:
            first = gate.verify_full_release_artifacts(
                scratch, manifest, "before_source_replay"
            )
            self.assertEqual(first["file_count"], 7)
            artifact_paths["payload_index"].write_bytes(b"tampered-unsampled-index\n")
            with self.assertRaises(gate.GateError):
                gate.verify_full_release_artifacts(
                    scratch, manifest, "after_source_replay_before_completion"
                )
        finally:
            artifact_paths["geometry_records_lmdb_data"].unlink()
            artifact_paths["membership"].unlink()
            artifact_paths["reject_ledger"].unlink()
            artifact_paths["payload_index"].unlink()
            artifact_paths["motif_census"].unlink()
            shard_manifest_path.unlink()
            global_census.unlink()
            os.rmdir(records)
            os.rmdir(shard)
            os.rmdir(scratch)

    def test_e3fp_closure_change_is_detected_after_initial_attestation(self):
        scratch = Path(tempfile.mkdtemp(prefix="r5-e3fp-"))
        package = scratch / "e3fp"
        fingerprint = package / "fingerprint"
        fingerprint.mkdir(parents=True)
        init_path = package / "__init__.py"
        pipeline_path = package / "pipeline.py"
        fprinter_path = fingerprint / "fprinter.py"
        init_path.write_bytes(b"VERSION = 1\n")
        pipeline_path.write_bytes(b"def run(): return 1\n")
        fprinter_path.write_bytes(b"def fp(): return 1\n")
        expected = []
        for relative in ("__init__.py", "fingerprint/fprinter.py", "pipeline.py"):
            path = package / Path(relative)
            size, digest = gate.sha256_file(path)
            expected.append({"relative_path": relative, "bytes": size, "sha256": digest})
        closure = {
            "files": expected,
            "exclusion_policy": {"directories": [], "file_suffixes": [".pyc"]},
            "closure_sha256": gate.sha256_bytes(
                gate.canonical_json_bytes(expected) + b"\n"
            ),
        }
        imported = {
            "e3fp": init_path,
            "e3fp.pipeline": pipeline_path,
            "e3fp.fingerprint.fprinter": fprinter_path,
        }
        try:
            gate.scan_e3fp_closure(package, closure, "initial", imported)
            pipeline_path.write_bytes(b"def run(): return 2\n")
            with self.assertRaises(gate.GateError):
                gate.scan_e3fp_closure(package, closure, "final", imported)
        finally:
            init_path.unlink()
            pipeline_path.unlink()
            fprinter_path.unlink()
            os.rmdir(fingerprint)
            os.rmdir(package)
            os.rmdir(scratch)

    def test_final_check_failure_never_publishes_completed_marker(self):
        output = Path(tempfile.mkdtemp(prefix="r5-finalize-"))
        bundle = self._write_valid_staged_bundle(output)
        calls = []

        def verifier(phase):
            calls.append(phase)
            if phase == "before_completed_marker":
                raise gate.GateError("synthetic final check failure")
            return self._completion_observation(bundle, phase)

        try:
            with self.assertRaises(gate.GateError):
                gate.publish_completion(
                    output, bundle["release_id"], bundle["script_binding"],
                    bundle["final_checks"], verifier,
                )
            self.assertEqual(
                calls, ["before_completion_receipt", "before_completed_marker"]
            )
            self.assertTrue((output / "completion_receipt.json").is_file())
            self.assertFalse((output / "COMPLETED").exists())
        finally:
            (output / "completion_receipt.json").unlink()
            bundle["report_path"].unlink()
            bundle["ledger_path"].unlink()
            os.rmdir(output)

    def test_consumer_requires_and_validates_receipt_plus_completed(self):
        output = Path(tempfile.mkdtemp(prefix="r5-completed-"))
        bundle = self._write_valid_staged_bundle(output)
        try:
            gate.publish_completion(
                output, bundle["release_id"], bundle["script_binding"],
                bundle["final_checks"],
                lambda phase: self._completion_observation(bundle, phase),
            )
            validated = gate.validate_completed_output(
                output, bundle["script_binding"]["sha256"]
            )
            self.assertEqual(validated["overall_gate_status"], "pass")
            (output / "COMPLETED").unlink()
            with self.assertRaises(gate.GateError):
                gate.validate_completed_output(
                    output, bundle["script_binding"]["sha256"]
                )
        finally:
            if (output / "COMPLETED").is_file():
                (output / "COMPLETED").unlink()
            (output / "completion_receipt.json").unlink()
            bundle["report_path"].unlink()
            bundle["ledger_path"].unlink()
            os.rmdir(output)

    def test_arbitrary_staged_report_bytes_cannot_be_published(self):
        output = Path(tempfile.mkdtemp(prefix="r51-arbitrary-report-"))
        report_path = output / gate.STAGED_REPORT_FILE_NAME
        report_path.write_bytes(b"arbitrary staged report bytes\n")
        script_binding = self._running_script_binding()
        try:
            with self.assertRaises(gate.GateError):
                gate.publish_completion(
                    output, "synthetic-release", script_binding, {},
                    lambda phase: {"phase": phase},
                )
            self.assertFalse((output / gate.COMPLETION_RECEIPT_FILE_NAME).exists())
            self.assertFalse((output / gate.COMPLETED_MARKER_FILE_NAME).exists())
        finally:
            report_path.unlink()
            os.rmdir(output)

    def test_fail_report_cannot_be_forged_into_consumer_pass(self):
        output = Path(tempfile.mkdtemp(prefix="r51-fail-report-"))
        bundle = self._write_valid_staged_bundle(output, semantic_status="fail")
        receipt_path = output / gate.COMPLETION_RECEIPT_FILE_NAME
        marker_path = output / gate.COMPLETED_MARKER_FILE_NAME
        try:
            receipt_path, completed = gate.publish_completion(
                output, bundle["release_id"], bundle["script_binding"],
                bundle["final_checks"],
                lambda phase: self._completion_observation(bundle, phase),
            )
            self.assertIsNone(completed)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["overall_gate_status"], "fail")
            receipt["overall_gate_status"] = "pass"
            receipt[gate.RECEIPT_HASH_FIELD] = gate.receipt_payload_sha256(receipt)
            receipt_path.write_bytes(gate.canonical_json_bytes(receipt) + b"\n")
            receipt_bytes, receipt_sha = gate.sha256_file(receipt_path)
            marker = {
                "schema_version": gate.COMPLETED_MARKER_SCHEMA,
                "created_utc": "2026-08-05T00:00:00+00:00",
                "release_id": bundle["release_id"],
                "overall_gate_status": "pass",
                "completion_receipt": {
                    "relative_path": gate.COMPLETION_RECEIPT_FILE_NAME,
                    "bytes": receipt_bytes,
                    "sha256": receipt_sha,
                    gate.RECEIPT_HASH_FIELD: receipt[gate.RECEIPT_HASH_FIELD],
                },
                "staged_report": receipt["staged_report"],
                "result_ledger": receipt["result_ledger"],
                "semantic_gate_script": receipt["semantic_gate_script"],
                "frozen_bindings": receipt["frozen_bindings"],
                "final_pre_marker_observation": self._completion_observation(
                    bundle, "before_completed_marker"
                ),
            }
            marker["completed_marker_canonical_payload_sha256"] = gate.sha256_json(marker)
            marker_path.write_bytes(gate.canonical_json_bytes(marker) + b"\n")
            with self.assertRaises(gate.GateError):
                gate.validate_completed_output(
                    output, bundle["script_binding"]["sha256"]
                )
        finally:
            marker_path.unlink()
            receipt_path.unlink()
            bundle["report_path"].unlink()
            bundle["ledger_path"].unlink()
            os.rmdir(output)

    def test_misbound_ledger_header_blocks_receipt(self):
        output = Path(tempfile.mkdtemp(prefix="r51-ledger-misbind-"))
        bundle = self._write_valid_staged_bundle(
            output, header_updates={"semantic_plan_sha256": "0" * 64}
        )
        try:
            with self.assertRaises(gate.GateError):
                gate.publish_completion(
                    output, bundle["release_id"], bundle["script_binding"],
                    bundle["final_checks"],
                    lambda phase: self._completion_observation(bundle, phase),
                )
            self.assertFalse((output / gate.COMPLETION_RECEIPT_FILE_NAME).exists())
        finally:
            bundle["report_path"].unlink()
            bundle["ledger_path"].unlink()
            os.rmdir(output)

    def test_ledger_path_traversal_or_substitution_is_rejected(self):
        output = Path(tempfile.mkdtemp(prefix="r51-ledger-path-"))
        bundle = self._write_valid_staged_bundle(
            output, ledger_relative="../semantic_recompute_results.jsonl"
        )
        try:
            with self.assertRaises(gate.GateError):
                gate.publish_completion(
                    output, bundle["release_id"], bundle["script_binding"],
                    bundle["final_checks"],
                    lambda phase: self._completion_observation(bundle, phase),
                )
            self.assertFalse((output / gate.COMPLETION_RECEIPT_FILE_NAME).exists())
        finally:
            bundle["report_path"].unlink()
            bundle["ledger_path"].unlink()
            os.rmdir(output)

    def test_mispassed_overall_pass_boolean_cannot_create_receipt(self):
        output = Path(tempfile.mkdtemp(prefix="r51-bool-mispass-"))
        bundle = self._write_valid_staged_bundle(output)
        try:
            with self.assertRaises(gate.GateError):
                gate.publish_completion(
                    output, True, bundle["script_binding"], bundle["final_checks"],
                    lambda phase: self._completion_observation(bundle, phase),
                )
            self.assertFalse((output / gate.COMPLETION_RECEIPT_FILE_NAME).exists())
            self.assertFalse((output / gate.COMPLETED_MARKER_FILE_NAME).exists())
        finally:
            bundle["report_path"].unlink()
            bundle["ledger_path"].unlink()
            os.rmdir(output)

    def test_staged_report_cannot_claim_pass(self):
        gate.validate_staged_report_claims(
            {
                "completion_status": "pending_authoritative_receipt",
                "authoritative_overall_gate_status": None,
                "semantic_recompute_status": "pass",
            }
        )
        with self.assertRaises(gate.GateError):
            gate.validate_staged_report_claims(
                {
                    "completion_status": "pending_authoritative_receipt",
                    "authoritative_overall_gate_status": None,
                    "gate_status": "pass",
                }
            )


if __name__ == "__main__":
    unittest.main()
