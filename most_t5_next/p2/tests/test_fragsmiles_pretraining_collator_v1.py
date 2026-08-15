from __future__ import annotations

import unittest

import numpy as np

from most_t5_next.p2.fragsmiles_pretraining_collator_v1 import (
    collate_molecular_denoising_samples,
    corrupt_cached_fragsmiles_record,
    standard_t5_noise_mask,
)
from most_t5_next.p2.fragsmiles_training_tensor_cache_v1 import (
    CachedFragSmilesRecord,
    CachedFragSmilesSample,
    ROLE_TO_ID,
)


def _record(ordinal: int = 7) -> CachedFragSmilesRecord:
    return CachedFragSmilesRecord(
        cache_index=0,
        ordinal=ordinal,
        source_segment=0,
        mode="compact",
        component_count=1,
        molecule_carrier=-1,
        input_ids=np.asarray([10, 11, 12, 20, 13, 14, 1]),
        token_roles=np.asarray(
            [
                ROLE_TO_ID["molecule_boundary"],
                ROLE_TO_ID["fragment_phrase"],
                ROLE_TO_ID["fragment_phrase"],
                ROLE_TO_ID["connector_endpoint"],
                ROLE_TO_ID["fragment_phrase"],
                ROLE_TO_ID["fragment_phrase"],
                ROLE_TO_ID["control"],
            ]
        ),
        token_to_fragment=np.asarray([-1, 0, 0, 0, 1, 1, -1]),
        fragment_spans=np.asarray([[1, 3], [4, 6]]),
        fragment_carriers=np.asarray([2, 5]),
        fragment_components=np.asarray([0, 0]),
        fragment_representations=np.asarray([1, 1]),
        atom_to_fragment=np.asarray([0, 1, 1]),
        atom_local_index=np.asarray([0, 0, 1]),
        atom_components=np.asarray([0, 0, 0]),
        atom_carriers=np.asarray([2, 5, 5]),
        atom_is_attachment=np.asarray([True, True, False]),
        e3fp=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        endpoints=np.asarray(
            [
                [0, 0, 0, 0, 3, 1],
                [0, 1, 1, 1, 5, 0],
            ]
        ),
    )


def _recover(corrupted, sentinels):
    source = list(corrupted.record.input_ids)
    labels = list(corrupted.labels)
    sentinel_set = set(sentinels)
    payload = {}
    index = 0
    while index < len(labels) and labels[index] in sentinel_set:
        marker = labels[index]
        index += 1
        values = []
        while index < len(labels) and labels[index] not in sentinel_set and labels[index] != 1:
            values.append(labels[index])
            index += 1
        payload[marker] = values
    recovered = []
    for token in source:
        recovered.extend(payload[token] if token in payload else [token])
    return recovered


class FragSmilesPretrainingCollatorV1Tests(unittest.TestCase):
    def test_p2_m_and_mg_share_identity_corruption(self) -> None:
        record = _record()
        sample = CachedFragSmilesSample(record=record, epoch=2)
        kwargs = {
            "sentinel_token_ids": (90, 91, 92, 93),
            "eos_token_id": 1,
            "global_seed": 77,
        }
        m = corrupt_cached_fragsmiles_record(sample, view="P2-M", **kwargs)
        mg = corrupt_cached_fragsmiles_record(sample, view="P2-MG", **kwargs)
        self.assertEqual(m.record.input_ids.tolist(), mg.record.input_ids.tolist())
        self.assertEqual(m.labels, mg.labels)
        self.assertFalse(any(m.fragment_geometry_mask))
        self.assertTrue(any(mg.fragment_geometry_mask))

    def test_reference_mask_has_exact_noise_count(self) -> None:
        mask = standard_t5_noise_mask(
            20, noise_density=0.15, mean_noise_span_length=3.0, seed=4
        )
        self.assertEqual(len(mask), 20)
        self.assertEqual(sum(mask), 3)

    def test_complete_fragment_corruption_remaps_every_address(self) -> None:
        original = _record()
        row = corrupt_cached_fragsmiles_record(
            CachedFragSmilesSample(original, epoch=2),
            view="P2-MG",
            sentinel_token_ids=(100, 99, 98, 97, 96),
            eos_token_id=1,
            global_seed=13,
        )
        self.assertEqual(_recover(row, (100, 99, 98, 97, 96)), original.input_ids.tolist())
        self.assertTrue(row.selected_fragment_ids)
        self.assertEqual(
            len(set(row.record.fragment_carriers.tolist())),
            len(row.record.fragment_carriers),
        )
        for fragment_id in row.selected_fragment_ids:
            self.assertFalse(row.fragment_geometry_mask[fragment_id])
            start, stop = row.record.fragment_spans[fragment_id]
            self.assertEqual(int(stop - start), 1)
            self.assertEqual(int(row.record.fragment_carriers[fragment_id]), int(start))
        for endpoint_index, endpoint in enumerate(row.record.endpoints):
            owner = int(endpoint[2])
            if owner in row.selected_fragment_ids:
                self.assertFalse(row.endpoint_geometry_mask[endpoint_index])
        self.assertTrue(
            np.all(row.record.atom_carriers >= 0)
            and np.all(row.record.atom_carriers < len(row.record.input_ids))
        )

    def test_selected_motif_masks_its_explicit_endpoint_as_one_choice(self) -> None:
        original = _record()
        row = corrupt_cached_fragsmiles_record(
            CachedFragSmilesSample(original, epoch=0),
            view="P2-MG",
            sentinel_token_ids=(100, 99, 98, 97, 96),
            eos_token_id=1,
            global_seed=2,
        )
        self.assertEqual(row.selected_fragment_ids, (0,))
        self.assertNotIn(20, row.record.input_ids.tolist())
        self.assertIn(20, row.labels)
        self.assertEqual(row.selected_motif_spans, ((1, 3), (3, 4)))
        self.assertFalse(row.endpoint_geometry_mask[0])
        self.assertTrue(row.endpoint_geometry_mask[1])
        self.assertNotIn((3, 4), row.selected_syntax_spans)

    def test_opposite_explicit_endpoint_is_retained(self) -> None:
        original = _record()
        row = corrupt_cached_fragsmiles_record(
            CachedFragSmilesSample(original, epoch=0),
            view="P2-MG",
            sentinel_token_ids=(100, 99, 98, 97, 96),
            eos_token_id=1,
            global_seed=0,
        )
        self.assertEqual(row.selected_fragment_ids, (1,))
        self.assertEqual(row.selected_motif_spans, ((4, 6),))
        self.assertIn(20, row.record.input_ids.tolist())
        self.assertTrue(row.endpoint_geometry_mask[0])
        self.assertFalse(row.endpoint_geometry_mask[1])

    def test_coordinate_blind_view_retains_structure_but_disables_geometry(self) -> None:
        row = corrupt_cached_fragsmiles_record(
            CachedFragSmilesSample(_record(), epoch=0),
            view="P2-M",
            sentinel_token_ids=(100, 99, 98, 97, 96),
            eos_token_id=1,
            global_seed=1,
        )
        self.assertFalse(any(row.fragment_geometry_mask))
        self.assertFalse(any(row.endpoint_geometry_mask))

    def test_batch_padding_is_dynamic_and_masks_are_owned(self) -> None:
        samples = (
            CachedFragSmilesSample(_record(7), epoch=1),
            CachedFragSmilesSample(_record(8), epoch=1),
        )
        batch = collate_molecular_denoising_samples(
            samples,
            view="P1-SYN",
            pad_token_id=0,
            sentinel_token_ids=(100, 99, 98, 97, 96),
            eos_token_id=1,
            global_seed=9,
        )
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["labels"].shape[0], 2)
        self.assertTrue(
            bool((batch["endpoint_geometry_mask"] <= batch["endpoint_mask"]).all())
        )
        owners = batch["endpoint_to_fragment"].clamp_min(0)
        owner_visible = batch["fragment_geometry_mask"].gather(1, owners)
        self.assertFalse(
            bool((batch["endpoint_geometry_mask"] & ~owner_visible).any())
        )

if __name__ == "__main__":
    unittest.main()
