import math

from most_t5_next.p1 import build_g3a_multiconformer_relation_dataset_v1 as builder
from most_t5_next.p1.run_g1_multiconformer_sensitivity_v1 import (
    _distance_matrix_rms,
    _motif_edges,
)


class _Point:
    def __init__(self, x, y, z=0.0):
        self.values = (x, y, z)

    def __getitem__(self, index):
        return self.values[index]


class _Conformer:
    def __init__(self, points):
        self.points = points

    def GetAtomPosition(self, index):
        return self.points[index]


class _Bond:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def GetBeginAtomIdx(self):
        return self.left

    def GetEndAtomIdx(self):
        return self.right


class _Mol:
    def __init__(self, points, bonds=()):
        self.points = points
        self.bonds = [_Bond(*bond) for bond in bonds]

    def GetNumAtoms(self):
        return len(self.points)

    def GetConformer(self, index):
        assert index == 0
        return _Conformer(self.points)

    def GetBonds(self):
        return self.bonds


def test_distance_matrix_rms_is_rigid_invariant_and_has_angstrom_scale():
    left = _Mol([_Point(0, 0), _Point(1, 0), _Point(0, 1)])
    rigid = _Mol([_Point(5, -2), _Point(5, -1), _Point(4, -2)])
    stretched = _Mol([_Point(0, 0), _Point(2, 0), _Point(0, 1)])
    assert _distance_matrix_rms(left, rigid) < 1e-12
    expected = math.sqrt(((1.0 - 2.0) ** 2 + 0.0 + (math.sqrt(2) - math.sqrt(5)) ** 2) / 3)
    assert abs(_distance_matrix_rms(left, stretched) - expected) < 1e-12


def test_motif_edges_are_unique_undirected_cross_group_edges():
    mol = _Mol([], bonds=((0, 1), (1, 2), (2, 3), (3, 0)))
    assert _motif_edges(mol, (0, 0, 1, 2)) == ((0, 1), (0, 2), (1, 2))


def test_molecule_split_is_deterministic_and_identity_disjoint():
    rows = [{"member_id": "m{}".format(index)} for index in range(10)]
    first = builder._split_members(rows, train_fraction=0.8, seed=7)
    second = builder._split_members(rows, train_fraction=0.8, seed=7)
    assert first == second
    assert sum(value == "train" for value in first.values()) == 8
    assert sum(value == "dev" for value in first.values()) == 2

