from most_t5_next.p1.audit_g1_state_predictability_v1 import (
    CompactRecord,
    evaluate_context_predictor,
)


def _record(level3_shift=0):
    return CompactRecord(
        e3fp_ids=((1, 10, 20, 30 + level3_shift), (2, 11, 21, 31 + level3_shift)),
        atom_is_attachment=(False, True),
        atom_to_motif=(0, 0),
    )


def test_prefix_context_recovers_reused_shell_relation():
    report = evaluate_context_predictor(
        [_record(), _record()],
        [_record()],
        level=3,
        context_kind="atom_prefix",
        num_classes=64,
    )
    assert report["dev_seen_context_fraction"] == 1.0
    assert report["conditional_accuracy"] == 1.0
    assert report["nll_improvement_over_unigram"] > 0.0


def test_unseen_context_backs_off_to_train_unigram():
    report = evaluate_context_predictor(
        [_record()],
        [_record(level3_shift=3)],
        level=3,
        context_kind="atom_other_shells",
        num_classes=64,
    )
    assert report["dev_seen_context_fraction"] == 1.0
    assert report["conditional_accuracy"] == 0.0
    assert report["conditional_nll"] > report["laplace_unigram_nll"]


def test_motif_multiset_context_is_permutation_invariant():
    original = _record()
    permuted = CompactRecord(
        e3fp_ids=tuple(reversed(original.e3fp_ids)),
        atom_is_attachment=tuple(reversed(original.atom_is_attachment)),
        atom_to_motif=(0, 0),
    )
    report = evaluate_context_predictor(
        [original, original],
        [permuted],
        level=2,
        context_kind="motif_prefix_multiset",
        num_classes=64,
    )
    assert report["dev_seen_context_fraction"] == 1.0
    assert report["conditional_accuracy"] == 1.0
