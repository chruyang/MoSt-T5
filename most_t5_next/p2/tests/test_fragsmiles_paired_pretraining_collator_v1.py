from __future__ import annotations

import unittest

import numpy as np

from most_t5_next.p2.fragsmiles_paired_pretraining_collator_v1 import (
    Phase2PairedSample,
    collate_phase2_cap_samples,
    collate_phase2_t2m_samples,
    factorized_model_inputs_from_batch,
)
from most_t5_next.p2.fragsmiles_training_tensor_cache_v1 import CachedFragSmilesRecord


def _record(ordinal: int, tokens: int) -> CachedFragSmilesRecord:
    return CachedFragSmilesRecord(
        cache_index=ordinal,
        ordinal=ordinal,
        source_segment=0,
        mode="compact",
        component_count=1,
        molecule_carrier=-1,
        input_ids=np.arange(tokens, dtype=np.int64) + 5,
        token_roles=np.asarray([1] * (tokens - 1) + [0]),
        token_to_fragment=np.asarray([0] * (tokens - 1) + [-1]),
        fragment_spans=np.asarray([[0, tokens - 1]]),
        fragment_carriers=np.asarray([0]),
        fragment_components=np.asarray([0]),
        fragment_representations=np.asarray([0]),
        atom_to_fragment=np.asarray([0]),
        atom_local_index=np.asarray([0]),
        atom_components=np.asarray([0]),
        atom_carriers=np.asarray([0]),
        atom_is_attachment=np.asarray([True]),
        e3fp=np.asarray([[1, 2, 3, 4]]),
        endpoints=np.asarray([[0, 0, 0, 0, 0, 1]]),
    )


class FragSmilesPairedPretrainingCollatorV1Tests(unittest.TestCase):
    def test_paired_text_is_right_truncated_and_retains_eos(self) -> None:
        long_text = np.asarray(list(range(20, 30)) + [1])
        sample = [Phase2PairedSample(_record(1, 5), long_text)]
        cap = collate_phase2_cap_samples(
            sample, pad_token_id=0, target_cap=5, eos_token_id=1
        )
        self.assertEqual(cap["labels"].tolist(), [[20, 21, 22, 23, 1]])
        t2m = collate_phase2_t2m_samples(
            sample, pad_token_id=0, encoder_cap=5, eos_token_id=1
        )
        self.assertEqual(t2m["input_ids"].tolist(), [[20, 21, 22, 23, 1]])

    def test_cap_retains_aligned_geometry_and_maps_wrapper_inputs(self) -> None:
        samples = [Phase2PairedSample(_record(1, 5), np.asarray([30, 31, 1]))]
        batch = collate_phase2_cap_samples(samples, pad_token_id=0)
        self.assertTrue(batch["fragment_geometry_mask"].all())
        self.assertTrue(batch["endpoint_geometry_mask"].all())
        mapped = factorized_model_inputs_from_batch(batch)
        self.assertEqual(set(mapped), {
            "input_ids", "attention_mask", "e3fp_input_ids", "atom_mask",
            "atom_to_fragment", "fragment_mask", "fragment_to_carrier",
            "identity_span_bounds", "endpoint_mask", "endpoint_to_atom",
            "endpoint_to_token", "endpoint_to_fragment", "endpoint_is_explicit",
            "token_is_connector_endpoint", "atom_is_attachment",
            "fragment_geometry_mask", "endpoint_geometry_mask", "labels",
        })

    def test_t2m_is_plain_t5_and_pads_labels_with_ignore_index(self) -> None:
        samples = [
            Phase2PairedSample(_record(1, 5), np.asarray([20, 1])),
            Phase2PairedSample(_record(2, 4), np.asarray([21, 22, 1])),
        ]
        batch = collate_phase2_t2m_samples(samples, pad_token_id=0)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(batch["labels"][1, -1].item(), -100)
        self.assertEqual(batch["view"], "P2-T2M")


if __name__ == "__main__":
    unittest.main()
