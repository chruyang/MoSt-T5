from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from most_t5_next.r1.adapter import graphports_donor_atom_map_sidecar_v1 as subject


def _encoding(index: int) -> SimpleNamespace:
    if index % 2:
        maps = (((1, 1), (2, 0)), ((1, 2),))
    else:
        maps = (((1, 0), (2, 1)), ((1, 2),))
    return SimpleNamespace(
        format_version="most-t5-r1/graph-ports/v1",
        motifs=tuple(
            SimpleNamespace(motif_id=motif_id, source_atom_map=source_map)
            for motif_id, source_map in enumerate(maps)
        ),
    )


class GraphPortsDonorAtomMapSidecarTests(unittest.TestCase):
    def test_renumbered_model_axis_is_persisted_in_canonical_local_order(self) -> None:
        sidecar = subject.build_graphports_donor_atom_map_sidecar(_encoding(1))

        self.assertEqual(
            sidecar["canonical_local_atom_to_model_atom"],
            [[[1, 1], [2, 0]], [[1, 2]]],
        )

    def test_1024_row_fixture_closes_streaming_writer_reader_and_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "donor_atom_maps.jsonl"
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                encoded_bytes = 0
                for index in range(1024):
                    row = subject.build_release_row(
                        selection_index=index,
                        member_id=f"member-{index}",
                        sdf_record_index=10_000 + index,
                        split="train" if index < 900 else "dev",
                        storage_key=f"{index:09d}",
                        graph_encoding=_encoding(index),
                    )
                    encoded_bytes += subject.write_release_row(handle, row)

            count = 0
            final_row = None
            for final_row in subject.iter_release_rows(path):
                count += 1
            benchmark = subject.benchmark_release_prefix(path)

        self.assertEqual(count, 1024)
        self.assertEqual(final_row["selection_index"], 1023)  # type: ignore[index]
        self.assertGreater(encoded_bytes, 0)
        self.assertEqual(benchmark["requested_max_rows"], 1024)
        self.assertEqual(benchmark["rows_replayed"], 1024)
        self.assertEqual(benchmark["motifs_replayed"], 2048)
        self.assertEqual(benchmark["atom_mappings_replayed"], 3072)
        self.assertTrue(benchmark["bounded_prefix_complete"])

    def test_writer_rejects_maps_that_do_not_partition_model_atom_axis(self) -> None:
        encoding = SimpleNamespace(
            format_version="most-t5-r1/graph-ports/v1",
            motifs=(
                SimpleNamespace(motif_id=0, source_atom_map=((1, 0),)),
                SimpleNamespace(motif_id=1, source_atom_map=((1, 2),)),
            ),
        )
        row = subject.build_release_row(
            selection_index=0,
            member_id="member-0",
            sdf_record_index=0,
            split="dev",
            storage_key="000000000",
            graph_encoding=encoding,
        )
        with self.assertRaisesRegex(
            subject.GraphPortsDonorAtomMapError, "complete row axis"
        ):
            subject.validate_release_row(row)


if __name__ == "__main__":
    unittest.main()
