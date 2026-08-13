from __future__ import print_function

import json
import tempfile
import unittest
from pathlib import Path

from most_t5_next.r1.overlap import validate_downstream_3dmolt5_hf_source_policy_v1 as gate


POLICY = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "downstream_3dmolt5_hf_source_policy_20260812_v1.json"
)


class DownstreamSourcePolicyTests(unittest.TestCase):
    def mutated_policy(self, mutate):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "policy.json"
        value = json.loads(POLICY.read_text(encoding="utf-8"))
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_frozen_policy_passes(self):
        value = gate.load_and_validate(POLICY)
        self.assertEqual(len(value["huggingface_datasets"]), 11)
        self.assertEqual(value["moleculenet_exception"]["tasks"], ["BACE", "BBBP", "HIV", "ClinTox"])

    def test_legacy_substitution_is_rejected(self):
        path = self.mutated_policy(
            lambda value: value["huggingface_datasets"][0].update(
                {"repository_id": "legacy-local/pubchemqc"}
            )
        )
        with self.assertRaisesRegex(ValueError, "dataset map mismatch"):
            gate.load_and_validate(path)

    def test_moving_head_instead_of_full_revision_is_rejected(self):
        path = self.mutated_policy(
            lambda value: value["huggingface_datasets"][0].update({"revision": "main"})
        )
        with self.assertRaisesRegex(ValueError, "dataset map mismatch"):
            gate.load_and_validate(path)

    def test_silently_adding_retrieval_source_is_rejected(self):
        path = self.mutated_policy(
            lambda value: value["deferred_without_3dmolt5_hf_source"].clear()
        )
        with self.assertRaisesRegex(ValueError, "retrieval must remain explicitly deferred"):
            gate.load_and_validate(path)


if __name__ == "__main__":
    unittest.main()
