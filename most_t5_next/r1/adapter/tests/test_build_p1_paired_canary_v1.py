from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from most_t5_next.r1.adapter import build_p1_paired_canary_v1 as subject


class _Conformer:
    def __init__(self, coordinates):
        self._coordinates = coordinates

    def GetPositions(self):
        return self._coordinates


class _ProjectedMol:
    def __init__(self, coordinates):
        self._coordinates = coordinates

    def GetNumAtoms(self):
        return len(self._coordinates)

    def GetConformer(self, index):
        assert index == 0
        return _Conformer(self._coordinates)


class _FakeTransaction:
    def __init__(self, store, write):
        self._store = store
        self._write = write

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def put(self, key, value, overwrite):
        assert self._write is True
        if not overwrite and key in self._store:
            return False
        self._store[key] = bytes(value)
        return True

    def get(self, key):
        return self._store.get(key)

    def stat(self):
        return {"entries": len(self._store)}

    def cursor(self):
        return iter(sorted(self._store.items()))


class _FakeEnvironment:
    def __init__(self, module, path):
        self._module = module
        self._store = module.stores.setdefault(path, {})

    def begin(self, write=False):
        return _FakeTransaction(self._store, write)

    def sync(self, force):
        self._module.sync_arguments.append(force)

    def close(self):
        return None


class _FakeLMDB:
    def __init__(self):
        self.stores = {}
        self.sync_arguments = []

    def open(self, path, **kwargs):
        return _FakeEnvironment(self, path)


class MacroRegistryTest(unittest.TestCase):
    def test_count_cutoff_and_rank_are_frequency_then_utf8(self):
        registry, summary = subject.build_macro_registry(
            {"z": 1, "beta": 3, "alpha": 3, "gamma": 2}
        )
        self.assertEqual(
            [(row["identity"], row["token"], row["occurrence_count"]) for row in registry],
            [
                ("alpha", "<MOST:M:000000>", 3),
                ("beta", "<MOST:M:000001>", 3),
                ("gamma", "<MOST:M:000002>", 2),
            ],
        )
        self.assertEqual(summary["minimum_macro_occurrences"], 2)
        self.assertEqual(summary["macro_occurrences"], 8)
        self.assertEqual(summary["fallback_occurrences"], 1)

    def test_sample_must_exercise_both_surface_modes(self):
        with self.assertRaisesRegex(subject.PairedCanaryBuildError, "both macro and fallback"):
            subject.build_macro_registry({"only_macro": 2})
        with self.assertRaisesRegex(subject.PairedCanaryBuildError, "both macro and fallback"):
            subject.build_macro_registry({"only_fallback": 1})


class TopologyAdapterTest(unittest.TestCase):
    def test_validated_augmentation_maps_model_endpoints_and_rdkit_bond_name(self):
        document = {
            "logical_motif_domain": {
                "cross_motif_bonds": [
                    {
                        "edge_id": 0,
                        "left": {"model_atom_index": 4},
                        "right": {"model_atom_index": 9},
                        "source_bond_type": "Double",
                    }
                ]
            }
        }
        self.assertEqual(
            subject.cross_edges_from_augmentation(document),
            (subject.graph_codec.CrossEdgeInput(4, 9, "DOUBLE"),),
        )

    def test_non_dense_edge_is_a_declared_topology_reject(self):
        document = {
            "logical_motif_domain": {
                "cross_motif_bonds": [
                    {
                        "edge_id": 7,
                        "left": {"model_atom_index": 0},
                        "right": {"model_atom_index": 1},
                        "source_bond_type": "SINGLE",
                    }
                ]
            }
        }
        with self.assertRaises(subject.RecordRejected) as caught:
            subject.cross_edges_from_augmentation(document)
        self.assertEqual(caught.exception.stage, "TOPOLOGY_PARITY")


class PrepareMemberTest(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.asarray(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32
        )
        self.mapping = np.asarray([0, 2], dtype=np.int32)
        self.projected = _ProjectedMol(self.coordinates.copy())
        self.base_record = {
            "atom_universe": {
                "source_atom_count": 3,
                "model_atom_count": 2,
                "model_to_source_atom_index": self.mapping.copy(),
            },
            "geometry": {"coordinates": self.coordinates.copy()},
            "topology": {
                "motif_atom_indices": [[0], [1]],
                "motif_lexeme_sha256": ["a" * 64, "b" * 64],
            },
        }
        self.base_membership = {
            "member_id": "pcqm:17",
            "record_storage_key": "000000017",
            "record_content_sha256": "c" * 64,
        }
        self.overlay_record = {
            "e3fp": {"inherited": np.asarray([[1, -1], [2, 3]], dtype=np.int32)}
        }
        self.overlay_membership = {"overlay_record_content_sha256": "d" * 64}
        self.augmentation = {
            "logical_motif_domain": {
                "motif_atom_indices": [[0], [1]],
                "cross_motif_bonds": [
                    {
                        "edge_id": 0,
                        "left": {"model_atom_index": 0},
                        "right": {"model_atom_index": 1},
                        "source_bond_type": "SINGLE",
                    }
                ],
            }
        }

    def _patches(self):
        return (
            mock.patch.object(
                subject,
                "_join_overlay_to_base",
                return_value=(
                    self.base_record,
                    self.base_membership,
                    self.overlay_record,
                    self.overlay_membership,
                ),
            ),
            mock.patch.object(
                subject.projection,
                "tag_source_atoms",
                return_value=("tagged", 3, None),
            ),
            mock.patch.object(
                subject.projection,
                "project_hydrogens",
                return_value=(self.projected, tuple(self.mapping.tolist())),
            ),
            mock.patch.object(
                subject.mol_linearizer, "linearize_mol", return_value="linearized"
            ),
            mock.patch.object(
                subject.topology,
                "build_topology_augmentation",
                return_value=self.augmentation,
            ),
            mock.patch.object(
                subject.paired,
                "discover_production_paired_identity_surfaces",
                return_value="prepared-surfaces",
            ),
        )

    def test_projection_linearizer_and_discovery_each_run_once(self):
        patches = self._patches()
        with patches[0] as join, patches[1] as tag, patches[2] as project, patches[3] as linearize, patches[4] as augment, patches[5] as discover:
            result = subject.prepare_member(
                object(),
                object(),
                np,
                schedule_index=4,
                ordinal=17,
                source_mol="source",
                base_binding={},
                overlay_binding={},
                linearizer_sha256="e" * 64,
            )
        for patched in (join, tag, project, linearize, augment, discover):
            self.assertEqual(patched.call_count, 1)
        self.assertEqual(result.model_to_source_atom_index, (0, 2))
        self.assertEqual(result.inherited_e3fp, ((1, -1), (2, 3)))
        self.assertEqual(result.motif_count, 2)
        self.assertEqual(result.edge_count, 1)
        discover.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            self.projected,
            ((0,), (1,)),
            (subject.graph_codec.CrossEdgeInput(0, 1, "SINGLE"),),
        )

    def test_float32_coordinate_mismatch_stops_before_linearizer(self):
        self.base_record["geometry"]["coordinates"][0, 0] = 99.0
        patches = self._patches()
        with patches[0], patches[1], patches[2], patches[3] as linearize, patches[4], patches[5]:
            with self.assertRaises(subject.RecordRejected) as caught:
                subject.prepare_member(
                    object(),
                    object(),
                    np,
                    schedule_index=4,
                    ordinal=17,
                    source_mol="source",
                    base_binding={},
                    overlay_binding={},
                    linearizer_sha256="e" * 64,
                )
        self.assertEqual(caught.exception.stage, "PROJECTION_PARITY")
        linearize.assert_not_called()

    def test_projection_closed_reject_preserves_stage_and_reason(self):
        with mock.patch.object(
            subject, "_join_overlay_to_base", return_value=(
                self.base_record,
                self.base_membership,
                self.overlay_record,
                self.overlay_membership,
            )
        ), mock.patch.object(
            subject.projection,
            "tag_source_atoms",
            side_effect=subject.projection.RecordRejected(
                "SOURCE_ATOM_TAG_MISSING", "source_atom_tags"
            ),
        ):
            with self.assertRaises(subject.RecordRejected) as caught:
                subject.prepare_member(
                    object(), object(), np,
                    schedule_index=0,
                    ordinal=0,
                    source_mol="source",
                    base_binding={},
                    overlay_binding={},
                    linearizer_sha256="f" * 64,
                )
        self.assertEqual(caught.exception.stage, "PROJECTION_SOURCE_ATOM_TAGS")
        self.assertEqual(caught.exception.reason, "SOURCE_ATOM_TAG_MISSING")


class LmdbReplayTest(unittest.TestCase):
    def test_all_128_canonical_payloads_are_written_and_decoded(self):
        lmdb_module = _FakeLMDB()
        payloads = tuple(
            ("{:09d}".format(index), "payload-{}".format(index).encode("ascii"))
            for index in range(subject.SAMPLE_COUNT)
        )
        expected = tuple("loaded-{}".format(index) for index in range(subject.SAMPLE_COUNT))
        by_payload = {payload: row for (_key, payload), row in zip(payloads, expected)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            subject.paired_wire,
            "decode_paired_training_record",
            side_effect=lambda payload: by_payload[payload],
        ) as decode:
            replayed = subject._write_lmdb_and_replay(
                staging_root=Path(directory),
                payloads=payloads,
                expected_loaded=expected,
                lmdb_module=lmdb_module,
            )
        self.assertEqual(replayed, expected)
        self.assertEqual(decode.call_count, subject.SAMPLE_COUNT)
        self.assertEqual(lmdb_module.sync_arguments, [True])

    def test_lmdb_publication_refuses_less_than_128(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(subject.PairedCanaryBuildError, "exactly 128"):
                subject._write_lmdb_and_replay(
                    staging_root=Path(directory),
                    payloads=(("000000000", b"one"),),
                    expected_loaded=("one",),
                    lmdb_module=_FakeLMDB(),
                )


class StaticMaskCapacityTest(unittest.TestCase):
    @staticmethod
    def _row(atom_spans, motif_spans, record_id="member"):
        return SimpleNamespace(
            atom_record=SimpleNamespace(
                record_id=record_id,
                atom_identity_spans=tuple(
                    SimpleNamespace(start=start, stop=stop)
                    for start, stop in atom_spans
                ),
            ),
            motif_record=SimpleNamespace(
                identity_spans=tuple(
                    SimpleNamespace(start=start, stop=stop)
                    for start, stop in motif_spans
                )
            ),
        )

    def test_all_mask_formula_and_sentinel_capacity_cover_all_128(self):
        row = self._row(((1, 2), (2, 3)), ((1, 4), (4, 5)))
        report = subject.static_all_mask_capacity(
            (row,) * subject.SAMPLE_COUNT,
            sentinel_id_count=100,
        )
        self.assertTrue(report["all_masks_all_epochs_proven"])
        self.assertEqual(
            report["atom"]["all_mask_target_upper_bound"]["max"], 6
        )
        self.assertEqual(
            report["motif"]["all_mask_target_upper_bound"]["max"], 8
        )
        self.assertEqual(report["motif"]["required_sentinels"]["max"], 3)

    def test_static_gate_rejects_target_or_sentinel_overflow(self):
        long_row = self._row(((1, 512),), ((1, 2),), record_id="long")
        with self.assertRaisesRegex(subject.PairedCanaryBuildError, "exceeds 512"):
            subject.static_all_mask_capacity(
                (long_row,) * subject.SAMPLE_COUNT,
                sentinel_id_count=100,
            )
        many_units = tuple((index, index + 1) for index in range(3))
        sentinel_row = self._row(many_units, ((1, 2),), record_id="sentinels")
        with self.assertRaisesRegex(subject.PairedCanaryBuildError, "needs 4 sentinels"):
            subject.static_all_mask_capacity(
                (sentinel_row,) * subject.SAMPLE_COUNT,
                sentinel_id_count=3,
            )


class FourGridDryRunTest(unittest.TestCase):
    @staticmethod
    def _batch(condition_id, input_lengths=(10, 11), target_lengths=(4, 5)):
        return SimpleNamespace(
            condition_id=condition_id,
            ce_batch=SimpleNamespace(
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            ),
        )

    @staticmethod
    def _records(count=2):
        rows = []
        for index in range(count):
            atom_span = SimpleNamespace(start=1, stop=2)
            motif_span = SimpleNamespace(start=1, stop=3)
            rows.append(
                SimpleNamespace(
                    atom_record=SimpleNamespace(
                        record_id="a-{}".format(index),
                        atom_identity_spans=(atom_span,),
                    ),
                    motif_record=SimpleNamespace(
                        record_id="m-{}".format(index),
                        identity_spans=(motif_span,),
                        atom_to_logical_motif=(0, 0),
                    ),
                )
            )
        return tuple(rows)

    @staticmethod
    def _example(input_length, target_length, selected_field):
        values = {
            "input_ids": tuple(range(input_length)),
            "labels": tuple(range(target_length)),
            "selected_atom_ids_in_input_order": (),
            "selected_logical_motif_ids_in_input_order": (),
        }
        values[selected_field] = (0,)
        return SimpleNamespace(**values)

    def test_all_four_cells_share_ce_by_family_and_cross_representation_geometry(self):
        records = self._records()
        atom_batches = {
            "A0": self._batch("A0"),
            "A1": self._batch("A1"),
        }
        # Equal-value CE namespaces prove that only geometry differs.
        atom_batches["A1"].ce_batch = atom_batches["A0"].ce_batch
        motif_batches = {
            "M0": self._batch("M0", (12, 13), (6, 7)),
            "M1": self._batch("M1", (12, 13), (6, 7)),
        }
        motif_batches["M1"].ce_batch = motif_batches["M0"].ce_batch
        atom_examples = iter(
            (
                self._example(10, 4, "selected_atom_ids_in_input_order"),
                self._example(11, 5, "selected_atom_ids_in_input_order"),
            )
        )
        motif_examples = iter(
            (
                self._example(12, 6, "selected_logical_motif_ids_in_input_order"),
                self._example(13, 7, "selected_logical_motif_ids_in_input_order"),
            )
        )
        fake_tensor = SimpleNamespace(shape=(2, 13), dtype="torch.int64")
        with mock.patch.object(
            subject,
            "collate_production_atom_batch",
            side_effect=lambda _rows, **kwargs: atom_batches[kwargs["condition_id"]],
        ) as atom_collate, mock.patch.object(
            subject,
            "collate_production_batch",
            side_effect=lambda _rows, **kwargs: motif_batches[kwargs["condition_id"]],
        ) as motif_collate, mock.patch.object(
            subject, "validate_a1_m1_geometry_atom_parity"
        ) as parity, mock.patch.object(
            subject,
            "to_four_grid_batch_encoding",
            return_value={"input_ids": fake_tensor},
        ) as tensor_adapter, mock.patch.object(
            subject,
            "collate_production_atom_record",
            side_effect=lambda *_args, **_kwargs: next(atom_examples),
        ), mock.patch.object(
            subject,
            "collate_production_motif_record",
            side_effect=lambda *_args, **_kwargs: next(motif_examples),
        ):
            report = subject.run_cpu_four_grid_dry_collate(
                records, tokenizer_runtime="runtime"
            )
        self.assertEqual(atom_collate.call_count, 2)
        self.assertEqual(motif_collate.call_count, 2)
        self.assertTrue(
            all(call.kwargs["mask_probability"] == 0.15 for call in atom_collate.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["mask_probability"] == 0.15 for call in motif_collate.call_args_list)
        )
        parity.assert_called_once_with(atom_batches["A1"], motif_batches["M1"])
        self.assertEqual(tensor_adapter.call_count, 4)
        self.assertTrue(report["a0_a1_ce_equal"])
        self.assertEqual(report["lengths"]["M_collated_label"]["max"], 7)
        self.assertEqual(report["atom_selection"]["selected_unit_count"], 2)
        self.assertEqual(report["motif_selection"]["selected_atom_count"], 4)
        self.assertEqual(report["tensor_interfaces"]["M1"]["keys"], ["input_ids"])
        self.assertEqual(
            report["tensor_interfaces"]["M1"]["dtypes"],
            {"input_ids": "torch.int64"},
        )

    def test_over_512_collated_target_is_not_publishable(self):
        records = self._records(count=1)
        a = self._batch("A0", input_lengths=(10,), target_lengths=(4,))
        m = self._batch("M0", input_lengths=(10,), target_lengths=(513,))
        fake_tensor = SimpleNamespace(shape=(1, 10), dtype="torch.int64")
        with mock.patch.object(
            subject,
            "collate_production_atom_batch",
            side_effect=lambda _rows, **kwargs: SimpleNamespace(
                condition_id=kwargs["condition_id"], ce_batch=a.ce_batch
            ),
        ), mock.patch.object(
            subject,
            "collate_production_batch",
            side_effect=lambda _rows, **kwargs: SimpleNamespace(
                condition_id=kwargs["condition_id"], ce_batch=m.ce_batch
            ),
        ), mock.patch.object(
            subject, "validate_a1_m1_geometry_atom_parity"
        ), mock.patch.object(
            subject,
            "to_four_grid_batch_encoding",
            return_value={"input_ids": fake_tensor},
        ), mock.patch.object(
            subject,
            "collate_production_atom_record",
            return_value=self._example(10, 4, "selected_atom_ids_in_input_order"),
        ), mock.patch.object(
            subject,
            "collate_production_motif_record",
            return_value=self._example(
                10, 513, "selected_logical_motif_ids_in_input_order"
            ),
        ):
            with self.assertRaisesRegex(subject.PairedCanaryBuildError, "exceeds 512"):
                subject.run_cpu_four_grid_dry_collate(
                    records, tokenizer_runtime="runtime"
                )


class StaticSingleScanContractTest(unittest.TestCase):
    def test_builder_contains_one_sdf_stream_invocation_and_no_e3fp_generator(self):
        source = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("release_reader.stream_selected_sdf("), 1)
        self.assertIn(
            "record_validator=release_reader.validate_bound_record", source
        )
        self.assertNotIn("generate_e3fp_projection_pair", source)
        self.assertNotIn("build_overlay_record(", source)


if __name__ == "__main__":
    unittest.main()
