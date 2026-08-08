from most_t5_next.p1.run_g1_state_sensitivity_v1 import same_size_derangement


def _record(atom_count, member_id):
    return {
        "member_id": member_id,
        "e3fp_ids": [[1, 2, 3, 4] for _ in range(atom_count)],
    }


def test_same_size_derangement_is_complete_except_singletons():
    records = [
        _record(3, "a"),
        _record(4, "b"),
        _record(3, "c"),
        _record(3, "d"),
    ]
    mapping, excluded = same_size_derangement(records)
    assert excluded == (1,)
    assert set(mapping) == {0, 2, 3}
    assert all(index != donor for index, donor in mapping.items())
    assert all(
        len(records[index]["e3fp_ids"]) == len(records[donor]["e3fp_ids"])
        for index, donor in mapping.items()
    )
