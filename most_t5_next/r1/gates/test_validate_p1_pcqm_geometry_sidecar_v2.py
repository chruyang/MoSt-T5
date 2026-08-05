"""Static regression tests for the v2 builder-linked replay gate.

They deliberately do not import RDKit, E3FP, LMDB, or a molecular dataset.
The remote integration test remains the fresh bounded v2 sidecar replay.
"""

from __future__ import print_function

import ast
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "gates" / "validate_p1_pcqm_geometry_sidecar.py"
LOCK_PATH = ROOT / "contracts" / "p1_pcqm_geometry_replay_gate_lock.json"


def import_validator():
    spec = importlib.util.spec_from_file_location("r1_pcqm_replay_v2_test", str(VALIDATOR_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = import_validator()


class ReplayV2StaticTest(unittest.TestCase):
    def test_v2_schema_and_conclusion_boundary_are_explicit(self):
        self.assertEqual(validator.SIDE_CAR_SCHEMA, "most-t5-r1/p1-pcqm-geometry-smoke/v2")
        self.assertEqual(validator.RECORD_SCHEMA, "most-t5-r1/p1-pcqm-geometry-pretokenizer-record/v2")
        self.assertEqual(validator.REPLAY_LOCK_SCHEMA, "most-t5-r1/p1-pcqm-geometry-replay-gate-lock/v2")
        source = VALIDATOR_PATH.read_text(encoding="utf-8")
        self.assertIn("builder_linked_deterministic_replay", source)
        self.assertIn('"independent_semantic_validation": False', source)

    def test_replay_never_imports_or_calls_pickle(self):
        tree = ast.parse(VALIDATOR_PATH.read_text(encoding="utf-8"), filename=str(VALIDATOR_PATH))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn("pickle", [alias.name for alias in node.names])
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "pickle")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                self.assertNotEqual(node.value.id, "pickle")

    def test_current_replay_lock_binds_this_exact_file(self):
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            validator.validate_replay_gate_lock(LOCK_PATH, lock, VALIDATOR_PATH),
            validator.builder_sha256(LOCK_PATH),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
