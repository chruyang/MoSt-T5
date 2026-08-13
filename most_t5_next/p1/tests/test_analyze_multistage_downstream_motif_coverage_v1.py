import argparse
import json
from pathlib import Path
import tempfile
import unittest

from most_t5_next.p1 import analyze_multistage_downstream_motif_coverage_v1 as subject


class MultistageDownstreamCoverageTest(unittest.TestCase):
    @staticmethod
    def _registry(path: Path, rows):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for rank, (pure, count) in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "pure_motif": pure,
                            "pure_motif_id": rank,
                            "occurrences": count,
                            "rank": rank,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    def test_joint_base_and_all_chebi_are_distinct_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1 = root / "p1.jsonl"
            phase2 = root / "p2.jsonl"
            chebi = root / "chebi.jsonl"
            source = root / "source.jsonl"
            config = root / "config.json"
            output = root / "report.json"
            self._registry(phase1, [("A", 10), ("B", 1)])
            self._registry(phase2, [("B", 10), ("A", 1)])
            chebi.write_text(
                json.dumps({"pure_motif": "C", "train_occurrences": 1}) + "\n",
                encoding="utf-8",
            )
            source.write_text(
                json.dumps({"record_id": "x", "smiles": "CC"}) + "\n",
                encoding="utf-8",
            )
            config.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "fixture",
                                "scientific_role": "input_only",
                                "molecular_output_task": False,
                                "sources": [
                                    {
                                        "path": str(source),
                                        "format": "jsonl",
                                        "smiles_field": "smiles",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                dataset_config=str(config),
                phase1_registry=str(phase1),
                phase2_registry=str(phase2),
                chebi_train_census=str(chebi),
                output_report=str(output),
                base_budget=1,
                ranking_policy="phase1_phase2_equal_stage_mass",
                workers=1,
                chunksize=1,
            )
            original = subject._project_smiles
            try:
                subject._project_smiles = lambda task: {
                    "error": None,
                    "pure_motifs": ("C",),
                }
                original_executor = subject.concurrent.futures.ProcessPoolExecutor

                class _Executor:
                    def __init__(self, **_kwargs):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        return False

                    def map(self, fn, tasks, chunksize):
                        return map(fn, tasks)

                subject.concurrent.futures.ProcessPoolExecutor = _Executor
                report = subject.run(args)
            finally:
                subject._project_smiles = original
                subject.concurrent.futures.ProcessPoolExecutor = original_executor
            row = report["datasets"][0]
            self.assertEqual(report["selection"]["base_macro_count"], 1)
            self.assertEqual(report["selection"]["all_chebi_macro_count"], 2)
            self.assertEqual(
                row["policies"]["joint_pretraining_base"][
                    "macro_occurrence_coverage"
                ],
                0.0,
            )
            self.assertEqual(
                row["policies"]["joint_pretraining_base_plus_all_chebi20_train"][
                    "macro_occurrence_coverage"
                ],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
