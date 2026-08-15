from dataclasses import dataclass
import unittest

from most_t5_next.data.curriculum_data import (
    CurriculumDataError,
    CurriculumDataRouter,
    MolecularCacheUnion,
)


@dataclass(frozen=True)
class _Record:
    ordinal: int


class CurriculumDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pcqm = [_Record(10), _Record(11)]
        self.pubchem = [_Record(20), _Record(21), _Record(22)]
        self.paired = [(20, [1]), (21, [2]), (22, [3])]
        self.text = [[100], [101], [102], [103]]

    def test_union_is_zero_copy_and_source_aware(self) -> None:
        union = MolecularCacheUnion(self.pcqm, self.pubchem)
        self.assertEqual(len(union), 5)
        self.assertEqual((union[0].source, union[0].source_index), ("pcqm", 0))
        self.assertEqual((union[2].source, union[2].source_index), ("pubchem", 0))
        self.assertIs(union[4].record, self.pubchem[2])

    def test_tasks_route_to_the_frozen_populations(self) -> None:
        router = CurriculumDataRouter(
            pcqm=self.pcqm,
            pubchem=self.pubchem,
            pubchem_text=self.paired,
            text=self.text,
        )
        self.assertEqual(router.population_size("M"), 5)
        self.assertEqual(router.population_size("MG"), 2)
        self.assertEqual(router.population_size("SYN"), 5)
        self.assertEqual(router.population_size("CAP"), 3)
        self.assertEqual(router.population_size("T2M"), 3)
        self.assertEqual(router.population_size("TXT"), 4)
        self.assertEqual(router.get("M", 3).record.ordinal, 21)
        self.assertEqual(router.get("MG", 1).record.ordinal, 11)
        self.assertEqual(router.get("SYN", 2).record.ordinal, 20)
        self.assertEqual(router.get("CAP", 1).text_input_ids, [2])
        self.assertEqual(router.get("T2M", 2).record.ordinal, 22)
        self.assertEqual(router.get("TXT", 0), [100])

    def test_paired_identity_mismatch_fails(self) -> None:
        router = CurriculumDataRouter(
            pcqm=self.pcqm,
            pubchem=self.pubchem,
            pubchem_text=[(99, [1]), *self.paired[1:]],
            text=self.text,
        )
        with self.assertRaisesRegex(CurriculumDataError, "identities"):
            router.get("CAP", 0)


if __name__ == "__main__":
    unittest.main()
