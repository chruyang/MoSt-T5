"""Hermetic contracts for the streaming PF-1 paired release and reader."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from most_t5_next.p1 import build_pf1_paired_release_v1 as subject
from most_t5_next.p1 import freeze_pf1_connectivity_sample_v1 as selection


def frozen_row(index: int, ordinal: int, group: str, split: str) -> dict[str, object]:
    return {
        "schema_version": selection.MEMBERSHIP_SCHEMA,
        "selection_index": index,
        "group_order_index": index,
        "member_id": "ogb_pcqm4mv2_train_row_index:{}".format(ordinal),
        "sdf_record_index": ordinal,
        "connectivity_identity_sha256": group,
        "split": split,
    }


def output_row(
    *, split: str, split_index: int, selection_index: int, ordinal: int
) -> dict[str, object]:
    return {
        "schema_version": subject.OUTPUT_MEMBERSHIP_SCHEMA,
        "split": split,
        "split_index": split_index,
        "selection_index": selection_index,
        "group_order_index": selection_index,
        "connectivity_identity_sha256": "group-{}".format(split),
        "member_id": "member-{}".format(ordinal),
        "sdf_record_index": ordinal,
        "storage_key": "{:09d}".format(ordinal),
        "wire_bytes": 4,
        "atom_input_token_count": 3,
        "motif_input_token_count": 3,
        "atom_count": 2,
        "motif_count": 1,
        "edge_count": 0,
        "macro_identity_occurrences": 1,
        "fallback_identity_occurrences": 0,
        "effective_geometry_content_sha256": "a" * 64,
    }


class FakeTransaction:
    def __init__(self, payloads: dict[bytes, bytes]) -> None:
        self.payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def get(self, key: bytes):
        return self.payloads.get(key)


class FakeEnvironment:
    def __init__(self, payloads: dict[bytes, bytes]) -> None:
        self.payloads = payloads
        self.closed = False

    def begin(self, write: bool = False):
        assert write is False
        return FakeTransaction(self.payloads)

    def close(self) -> None:
        self.closed = True


class FakeLMDB:
    def __init__(self, payloads: dict[bytes, bytes]) -> None:
        self.payloads = payloads
        self.environments: list[FakeEnvironment] = []

    def open(self, *_args, **_kwargs):
        environment = FakeEnvironment(self.payloads)
        self.environments.append(environment)
        return environment


class PF1PairedReleaseTest(unittest.TestCase):
    def test_formal_cli_accepts_28_workers_and_84_pending(self) -> None:
        args = subject.build_parser().parse_args(
            [
                "--frozen-membership",
                "membership.jsonl",
                "--release-root",
                "production",
                "--source-archive",
                "source.tar.gz",
                "--e3fp-source",
                "e3fp",
                "--base-tokenizer",
                "tokenizer",
                "--output-dir",
                "output",
                "--workers",
                "28",
                "--max-pending",
                "84",
            ]
        )
        self.assertEqual((args.workers, args.max_pending), (28, 84))
        self.assertEqual(args.lmdb_map_size_gib, 4)

    def test_selfies_syntax_registry_covers_cohort_and_reports_split_provenance(self) -> None:
        coverage = subject.split_selfies_coverage(
            train_observed={"[C]", "[train-extra]"},
            dev_observed={"[C]", "[train-extra]", "[dev-extra]"},
            robust_alphabet={"[C]", "[O]"},
        )
        self.assertEqual(
            coverage["cohort_observed"],
            ("[C]", "[dev-extra]", "[train-extra]"),
        )
        self.assertEqual(coverage["train_nonrobust"], ("[train-extra]",))
        self.assertEqual(
            coverage["dev_nonrobust"], ("[dev-extra]", "[train-extra]")
        )
        self.assertEqual(coverage["dev_only_nonrobust"], ("[dev-extra]",))

    def test_frozen_membership_preserves_order_and_group_disjoint_split(self) -> None:
        rows = [
            frozen_row(0, 7, "g-train", "train"),
            frozen_row(1, 9, "g-train", "train"),
            frozen_row(2, 11, "g-dev", "dev"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "membership.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loaded = subject.load_frozen_membership(path, expected_members=3)
        self.assertEqual([row.selection_index for row in loaded], [0, 1, 2])
        self.assertEqual([row.sdf_record_index for row in loaded], [7, 9, 11])
        self.assertEqual([row.split for row in loaded], ["train", "train", "dev"])

    def test_frozen_membership_rejects_one_group_across_train_dev(self) -> None:
        rows = [
            frozen_row(0, 7, "same", "train"),
            frozen_row(1, 9, "same", "dev"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "membership.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(subject.PF1PairedReleaseError, "crosses"):
                subject.load_frozen_membership(path)

    def test_prepared_spool_is_selection_index_addressable(self) -> None:
        first = subject.FrozenMember(0, 0, "member-0", 0, "group-a", "train")
        second = subject.FrozenMember(1, 1, "member-1", 1, "group-b", "dev")

        def prepared(row: subject.FrozenMember) -> subject.PreparedMember:
            return subject.PreparedMember(
                frozen=row,
                storage_key="{:09d}".format(row.sdf_record_index),
                source_atom_count=2,
                model_to_source_atom_index=(0, 1),
                inherited_e3fp=((1, 2, 3, 4), (5, 6, 7, 8)),
                base_record_content_sha256="a" * 64,
                effective_geometry_content_sha256="b" * 64,
                prepared_surfaces="test-only",  # type: ignore[arg-type]
                atom_count=2,
                motif_count=1,
                edge_count=0,
                inheritance_summary={"slots_populated": 8},
            )

        with tempfile.TemporaryDirectory() as temporary:
            spool = subject._PreparedSpool(Path(temporary) / "spool.sqlite3", create=True)
            spool.put(prepared(second))
            spool.put(prepared(first))
            spool.commit()
            self.assertEqual(spool.count(), 2)
            self.assertEqual(spool.get(0).frozen, first)
            self.assertEqual(spool.get(1).frozen, second)
            spool.close()

    def test_reader_replays_frozen_split_order_and_dev_tail(self) -> None:
        train_rows = [
            output_row(split="train", split_index=index, selection_index=index, ordinal=10 + index)
            for index in range(4)
        ]
        dev_rows = [
            output_row(split="dev", split_index=index, selection_index=4 + index, ordinal=20 + index)
            for index in range(3)
        ]
        all_rows = train_rows + dev_rows
        payloads = {
            str(row["storage_key"]).encode("ascii"): str(row["selection_index"]).encode("ascii")
            for row in all_rows
        }
        record_by_index = {
            int(row["selection_index"]): SimpleNamespace(
                schedule_index=row["selection_index"],
                sdf_record_index=row["sdf_record_index"],
                atom_record=SimpleNamespace(record_id=row["member_id"]),
            )
            for row in all_rows
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            root.mkdir()
            (root / subject.LMDB_DIRECTORY).mkdir()
            (root / subject.MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "schema_version": subject.SCHEMA_VERSION,
                        "status": "pass",
                        "counts": {
                            "train_members": 4,
                            "dev_members": 3,
                            "paired_records": 7,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / subject.TRAIN_MEMBERSHIP_NAME).write_text(
                "".join(json.dumps(row) + "\n" for row in train_rows),
                encoding="utf-8",
            )
            (root / subject.DEV_MEMBERSHIP_NAME).write_text(
                "".join(json.dumps(row) + "\n" for row in dev_rows),
                encoding="utf-8",
            )
            lmdb = FakeLMDB(payloads)
            reader = subject.PF1PairedReleaseReader(
                root,
                lmdb_module=lmdb,
                decoder=lambda payload: record_by_index[int(payload.decode("ascii"))],
            )
            train = list(reader.iter_train_epoch(epoch=3, batch_size=2))
            dev = list(reader.iter_dev(batch_size=2))

        self.assertEqual(
            [[row.schedule_index for row in batch] for batch in train],
            [[0, 1], [2, 3]],
        )
        self.assertEqual(
            [[row.schedule_index for row in batch] for batch in dev],
            [[4, 5], [6]],
        )
        self.assertTrue(all(environment.closed for environment in lmdb.environments))


if __name__ == "__main__":
    unittest.main()
