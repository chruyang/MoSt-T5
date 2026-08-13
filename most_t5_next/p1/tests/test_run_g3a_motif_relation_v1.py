import torch

from most_t5_next.p1.run_g3a_motif_relation_v1 import (
    FrozenMoleculeFeatures,
    MotifTopologyRelationEncoder,
    _metrics,
    _ordering_accuracy,
    _pack_graphs,
    _predict_pairs,
)


def test_metrics_identify_perfect_relation_and_ordering():
    rows = [
        ("a", 0.1, 0.2),
        ("a", 0.4, 0.8),
        ("a", 0.7, 1.4),
        ("b", 0.2, 0.4),
        ("b", 0.5, 1.0),
        ("b", 0.8, 1.6),
    ]
    metrics = _metrics(rows)
    assert abs(metrics["pearson"] - 1.0) < 1e-12
    assert abs(metrics["spearman"] - 1.0) < 1e-12
    assert _ordering_accuracy(rows) == 1.0


def test_topology_encoder_is_invariant_to_node_renumbering():
    torch.manual_seed(4)
    model = MotifTopologyRelationEncoder(hidden_dim=8, output_dim=4).eval()
    nodes = torch.randn(3, 8)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    graph_index = torch.zeros(3, dtype=torch.long)
    original = model(nodes, edges, graph_index, 1)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(3)
    permuted_nodes = nodes[permutation]
    permuted_edges = inverse[edges]
    permuted = model(permuted_nodes, permuted_edges, graph_index, 1)
    assert torch.allclose(original, permuted, atol=1e-6, rtol=1e-6)


def test_pack_graphs_offsets_edges_without_cross_graph_links():
    first_nodes = torch.zeros(2, 4)
    second_nodes = torch.zeros(3, 4)
    first_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    second_edges = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    nodes, edges, graph_index = _pack_graphs(
        ((first_nodes, first_edges), (second_nodes, second_edges)),
        device=torch.device("cpu"),
    )
    assert nodes.shape == (5, 4)
    assert edges.tolist() == [[0, 1, 2, 4], [1, 0, 4, 2]]
    assert graph_index.tolist() == [0, 0, 1, 1, 1]


def test_wrong_conformer_diagnostic_uses_a_third_state():
    class _MeanModel(torch.nn.Module):
        def forward(self, nodes, edges, graph_index, num_graphs):
            result = nodes.new_zeros((num_graphs, nodes.shape[1]))
            counts = nodes.new_zeros((num_graphs, 1))
            result.index_add_(0, graph_index, nodes)
            counts.index_add_(0, graph_index, nodes.new_ones((nodes.shape[0], 1)))
            return result / counts

    molecule = FrozenMoleculeFeatures(
        member_id="m",
        split="dev",
        conformer_nodes=(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 2.0]]),
            torch.tensor([[3.0, 0.0]]),
        ),
        directed_edges=torch.empty((2, 0), dtype=torch.long),
        pairs=((0, 1, 1.0),),
    )
    aligned = _predict_pairs(
        _MeanModel(), [molecule], [(0, 0, 1, 1.0)], device=torch.device("cpu"), batch_size=1
    )
    wrong = _predict_pairs(
        _MeanModel(),
        [molecule],
        [(0, 0, 1, 1.0)],
        device=torch.device("cpu"),
        batch_size=1,
        mismatched_geometry=True,
    )
    assert aligned[0][2] == 1.0
    assert wrong[0][2] == 2.0
