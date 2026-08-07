import unittest

import numpy as np

from most_t5_next.r1.semantic import e3fp_duplicate_inheritance_v1 as subject


class _Shell:
    def __init__(
        self,
        center,
        radius,
        identifier,
        substruct,
        *,
        is_duplicate=False,
        duplicate=None,
    ):
        self.center_atom = center
        self.radius = radius
        self.identifier = identifier
        self.substruct = frozenset(substruct)
        self.is_duplicate = is_duplicate
        self.duplicate = duplicate


class _Fingerprinter:
    bits = 4096
    level = 3
    radius_multiplier = 1.5
    stereo = True
    include_disconnected = True
    rdkit_invariants = True
    exclude_floating = False
    remove_duplicate_substructs = True
    fp_type = type("Fingerprint", (), {})

    def __init__(self, shells):
        self.all_shells = list(shells)


class _Fingerprint:
    def __init__(self, indices):
        self.indices = tuple(indices)


class _Mol:
    def __init__(self, atom_count):
        self.atom_count = atom_count
        self.properties = {}

    def GetNumAtoms(self):
        return self.atom_count

    def SetProp(self, key, value):
        self.properties[key] = value


def _unsigned(identifier):
    return int(identifier) & 0xFFFFFFFF


class E3FPDuplicateInheritanceTests(unittest.TestCase):
    def test_slots_use_center_and_radius_not_list_position(self):
        shells = [
            _Shell(1, 1.5, 17, {1, 2}),
            _Shell(0, 0.0, 3, {0}),
            _Shell(1, 0.0, 8, {1}),
            _Shell(0, 1.5, 5, {0, 2}),
        ]
        raw, inherited, duplicate_mask, summary = subject.build_shell_projection_pair(
            np, _Fingerprinter(shells), _unsigned, 2, {3, 5, 8, 17}
        )
        self.assertEqual(raw.tolist(), [[3, 5, -1, -1], [8, 17, -1, -1]])
        self.assertTrue(np.array_equal(raw, inherited))
        self.assertEqual(raw.dtype, np.int32)
        self.assertEqual(inherited.dtype, np.int32)
        self.assertEqual(duplicate_mask.dtype, np.bool_)
        self.assertTrue(raw.flags.c_contiguous)
        self.assertTrue(inherited.flags.c_contiguous)
        self.assertTrue(duplicate_mask.flags.c_contiguous)
        self.assertTrue(np.array_equal(raw == -1, inherited == -1))
        self.assertTrue(np.array_equal(raw[:, 0], inherited[:, 0]))
        self.assertFalse(bool(duplicate_mask.any()))
        self.assertEqual(summary["slots_populated"], 4)
        self.assertEqual(summary["duplicate_slots"], 0)

    def test_explicit_pointer_wins_when_raw_fold_collides_with_final_bit(self):
        # An unrelated accepted identifier folds to raw bit 5.  A
        # folded-membership heuristic would therefore keep the duplicate's raw
        # identifier 5.  Explicit inheritance follows its pointer to bit 6.
        accepted = _Shell(0, 0.0, 6, {0, 2})
        unrelated_collision = _Shell(1, 0.0, 4101, {1})  # 4101 % 4096 == 5
        level_zero = _Shell(2, 0.0, 7, {2})
        duplicate = _Shell(
            2,
            1.5,
            5,
            {0, 2},
            is_duplicate=True,
            duplicate=accepted,
        )
        identifiers_before = [shell.identifier for shell in (accepted, unrelated_collision, level_zero, duplicate)]
        raw, inherited, duplicate_mask, summary = subject.build_shell_projection_pair(
            np,
            _Fingerprinter([duplicate, unrelated_collision, accepted, level_zero]),
            _unsigned,
            3,
            {5, 6, 7},
        )
        self.assertEqual(raw[2, 1], 5)
        self.assertEqual(inherited[2, 1], 6)
        self.assertTrue(bool(duplicate_mask[2, 1]))
        self.assertEqual(summary["changed_identifier_slots"], 1)
        self.assertEqual(summary["changed_token_slots"], 1)
        self.assertEqual(
            [shell.identifier for shell in (accepted, unrelated_collision, level_zero, duplicate)],
            identifiers_before,
            "projection must not mutate vendored E3FP shells",
        )

    def test_identifier_collision_can_hide_a_duplicate_token_change(self):
        accepted = _Shell(0, 0.0, 4101, {0, 1})
        level_zero = _Shell(1, 0.0, 9, {1})
        duplicate = _Shell(
            1,
            1.5,
            5,
            {0, 1},
            is_duplicate=True,
            duplicate=accepted,
        )
        raw, inherited, duplicate_mask, summary = subject.build_shell_projection_pair(
            np, _Fingerprinter([accepted, level_zero, duplicate]), _unsigned, 2, {5, 9}
        )
        self.assertEqual(raw[1, 1], inherited[1, 1])
        self.assertTrue(bool(duplicate_mask[1, 1]))
        self.assertEqual(summary["changed_identifier_slots"], 1)
        self.assertEqual(summary["changed_token_slots"], 0)

    def test_duplicate_pointer_invariants_are_closed(self):
        level_zero_0 = _Shell(0, 0.0, 3, {0})
        level_zero_1 = _Shell(1, 0.0, 8, {1})
        cases = []

        missing = _Shell(1, 1.5, 11, {0, 1}, is_duplicate=True, duplicate=None)
        cases.append((missing, "E3FP_DUPLICATE_POINTER_MISSING"))

        prior_duplicate = _Shell(0, 1.5, 12, {0, 1}, is_duplicate=True)
        chained = _Shell(
            1, 1.5, 13, {0, 1}, is_duplicate=True, duplicate=prior_duplicate
        )
        cases.append((chained, "E3FP_DUPLICATE_POINTER_NOT_ACCEPTED"))

        no_identifier = _Shell(0, 1.5, None, {0, 1})
        missing_identifier = _Shell(
            1, 1.5, 14, {0, 1}, is_duplicate=True, duplicate=no_identifier
        )
        cases.append((missing_identifier, "E3FP_DUPLICATE_IDENTIFIER_MISSING"))

        other_substructure = _Shell(0, 1.5, 15, {0})
        mismatch = _Shell(
            1, 1.5, 16, {0, 1}, is_duplicate=True, duplicate=other_substructure
        )
        cases.append((mismatch, "E3FP_DUPLICATE_SUBSTRUCTURE_MISMATCH"))

        for duplicate, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(subject.E3FPInheritanceError) as caught:
                    subject.build_shell_projection_pair(
                        np,
                        _Fingerprinter([level_zero_0, level_zero_1, duplicate]),
                        _unsigned,
                        2,
                        {3, 8, 11, 12, 13, 14, 15, 16},
                    )
                self.assertEqual(caught.exception.code, expected_code)

    def test_inherited_bit_must_be_in_final_fingerprint(self):
        accepted = _Shell(0, 0.0, 6, {0, 1})
        level_zero = _Shell(1, 0.0, 8, {1})
        duplicate = _Shell(
            1, 1.5, 5, {0, 1}, is_duplicate=True, duplicate=accepted
        )
        with self.assertRaises(subject.E3FPInheritanceError) as caught:
            subject.build_shell_projection_pair(
                np, _Fingerprinter([accepted, level_zero, duplicate]), _unsigned, 2, {5, 8}
            )
        self.assertEqual(
            caught.exception.code, "E3FP_INHERITED_BIT_NOT_IN_FINAL_FINGERPRINT"
        )

    def test_generate_calls_e3fp_once_and_returns_both_projections(self):
        accepted = _Shell(0, 0.0, 6, {0, 1})
        level_zero = _Shell(1, 0.0, 8, {1})
        duplicate = _Shell(
            1, 1.5, 5, {0, 1}, is_duplicate=True, duplicate=accepted
        )
        fingerprinter = _Fingerprinter([accepted, level_zero, duplicate])
        calls = []

        def generate(mol, fprint_params):
            calls.append((mol, dict(fprint_params)))
            return [_Fingerprint({6, 8})], fingerprinter

        mol = _Mol(2)
        raw, inherited, duplicate_mask, summary, resolved = (
            subject.generate_e3fp_projection_pair(
                np,
                {
                    "fprints_from_mol_verbose": generate,
                    "signed_to_unsigned_int": _unsigned,
                },
                mol,
                17,
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], subject.E3FP_INVOCATION)
        self.assertEqual(mol.properties["_Name"], "r1_pcqm_e3fp_inheritance_000000017")
        self.assertEqual(raw[1, 1], 5)
        self.assertEqual(inherited[1, 1], 6)
        self.assertTrue(bool(duplicate_mask[1, 1]))
        self.assertEqual(summary["semantics_id"], subject.SEMANTICS_ID)
        self.assertEqual(resolved["bits"], 4096)
        self.assertTrue(resolved["remove_duplicate_substructs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
