from most_t5_next.p1.run_g1_multiconformer_sensitivity_v1 import _attachment_roles


class _Bond:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def GetBeginAtomIdx(self):
        return self.left

    def GetEndAtomIdx(self):
        return self.right


class _Mol:
    def GetNumAtoms(self):
        return 4

    def GetBonds(self):
        return [_Bond(0, 1), _Bond(1, 2), _Bond(2, 3)]


def test_attachment_roles_are_cross_motif_bond_endpoints():
    assert _attachment_roles(_Mol(), ((0, 1), (2, 3))) == (False, True, True, False)
