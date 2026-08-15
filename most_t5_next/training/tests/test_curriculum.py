import unittest

from most_t5_next.training.curriculum import CurriculumSchedule, TASKS


class CurriculumScheduleTest(unittest.TestCase):
    def test_phase_one_alternates_m_and_mg(self) -> None:
        schedule = CurriculumSchedule(1, 8)
        self.assertEqual(
            [schedule.task_at(i).name for i in range(8)],
            ["M", "MG", "M", "MG", "M", "MG", "M", "MG"],
        )
        self.assertEqual(schedule.updates_per_task(), {"M": 4, "MG": 4})

    def test_phase_two_uses_one_equal_four_task_cycle(self) -> None:
        schedule = CurriculumSchedule(2, 8)
        self.assertEqual(
            [schedule.task_at(i).name for i in range(8)],
            ["SYN", "TXT", "CAP", "T2M", "SYN", "TXT", "CAP", "T2M"],
        )
        self.assertEqual(
            schedule.updates_per_task(),
            {"SYN": 2, "TXT": 2, "CAP": 2, "T2M": 2},
        )

    def test_task_is_constant_inside_one_optimizer_update(self) -> None:
        accumulation_steps = 4
        schedule = CurriculumSchedule(1, 4)
        microbatch_tasks = [
            schedule.task_at(update).name
            for update in range(len(schedule))
            for _ in range(accumulation_steps)
        ]
        self.assertEqual(
            microbatch_tasks,
            ["M"] * 4 + ["MG"] * 4 + ["M"] * 4 + ["MG"] * 4,
        )

    def test_information_routes_are_frozen(self) -> None:
        self.assertFalse(hasattr(TASKS["M"], "geometry_mode"))
        self.assertFalse(hasattr(TASKS["TXT"], "input_modality"))
        self.assertEqual(TASKS["TXT"].source, "pubmed")


if __name__ == "__main__":
    unittest.main()
