import torch

from most_t5_next.p1.run_g3a_continuous_geometry_relation_v1 import (
    ContinuousDistanceMotifEncoder,
    ContinuousMolecule,
    _pack_graphs,
)


def _molecule(positions):
    return ContinuousMolecule(
        member_id="m",
        split="dev",
        atomic_numbers=torch.tensor([6, 7, 8]),
        conformer_positions=(torch.tensor(positions, dtype=torch.float32),),
        atom_to_motif=torch.tensor([0, 0, 1]),
        motif_count=2,
        directed_motif_edges=torch.tensor([[0, 1], [1, 0]]),
        pairs=(),
    )


def test_continuous_encoder_is_rigid_invariant():
    torch.manual_seed(3)
    model = ContinuousDistanceMotifEncoder(hidden_dim=16, output_dim=8, radial_basis=16).eval()
    original = _molecule([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    rigid = _molecule([[5, -2, 1], [5, -1, 1], [3, -2, 1]])
    first = model(_pack_graphs([(original, 0)], device=torch.device("cpu")))
    second = model(_pack_graphs([(rigid, 0)], device=torch.device("cpu")))
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_pack_graphs_offsets_atom_and_motif_domains():
    first = _molecule([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    second = _molecule([[0, 0, 0], [0, 1, 0], [2, 0, 0]])
    batch = _pack_graphs([(first, 0), (second, 0)], device=torch.device("cpu"))
    assert batch["atomic_numbers"].shape[0] == 6
    assert batch["num_motifs"] == 4
    assert batch["motif_graph_index"].tolist() == [0, 0, 1, 1]
    assert batch["atom_to_motif"].tolist() == [0, 0, 1, 2, 2, 3]
    assert batch["atom_edges"].shape == (2, 12)

