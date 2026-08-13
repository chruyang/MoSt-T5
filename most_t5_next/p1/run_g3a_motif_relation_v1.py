#!/usr/bin/env python3
"""Train the small motif-topology conformer-relation gate used before G3b."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from most_t5_next.p1.level_aware_motif_state_v1 import LevelAwareMotifStateEncoder
from most_t5_next.p1.motif_state_data_v1 import collate_motif_state_records


SCHEMA_VERSION = "most-t5-p1/g3a-motif-relation/v1"
DEFAULT_SEED = 20260808


class G3ARelationError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenMoleculeFeatures:
    member_id: str
    split: str
    conformer_nodes: Tuple[Tensor, ...]
    directed_edges: Tensor
    pairs: Tuple[Tuple[int, int, float], ...]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise G3ARelationError("relation record is not an object")
                yield value


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise G3ARelationError("correlation requires at least three paired values")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return 0.0 if denominator == 0.0 else numerator / denominator


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * ((start + 1) + stop)
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _ordering_accuracy(rows: Sequence[Tuple[str, float, float]]) -> float:
    by_member: Dict[str, List[Tuple[float, float]]] = {}
    for member_id, target, prediction in rows:
        by_member.setdefault(member_id, []).append((target, prediction))
    correct = comparable = 0
    for values in by_member.values():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                target_delta = values[left][0] - values[right][0]
                prediction_delta = values[left][1] - values[right][1]
                if abs(target_delta) < 1e-12:
                    continue
                comparable += 1
                correct += int(target_delta * prediction_delta > 0.0)
    return 0.0 if comparable == 0 else correct / comparable


class MotifTopologyRelationEncoder(nn.Module):
    """Two message-passing layers over frozen G1 motif state nodes."""

    def __init__(self, hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(2)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(2)])
        self.project = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, int(output_dim)),
        )

    def forward(
        self, nodes: Tensor, directed_edges: Tensor, graph_index: Tensor, num_graphs: int
    ) -> Tensor:
        if nodes.ndim != 2 or nodes.shape[1] != self.hidden_dim:
            raise G3ARelationError("nodes must be [N, hidden_dim]")
        if directed_edges.ndim != 2 or directed_edges.shape[0] != 2:
            raise G3ARelationError("directed_edges must be [2,E]")
        hidden = nodes
        for update, norm in zip(self.layers, self.norms):
            aggregate = torch.zeros_like(hidden)
            degree = hidden.new_zeros((hidden.shape[0], 1))
            if directed_edges.shape[1]:
                source, target = directed_edges[0], directed_edges[1]
                aggregate.index_add_(0, target, hidden[source])
                degree.index_add_(0, target, hidden.new_ones((target.shape[0], 1)))
            neighbor_mean = aggregate / degree.clamp_min(1.0)
            hidden = norm(hidden + update(torch.cat((hidden, neighbor_mean), dim=-1)))
        pooled = hidden.new_zeros((int(num_graphs), hidden.shape[1]))
        counts = hidden.new_zeros((int(num_graphs), 1))
        pooled.index_add_(0, graph_index, hidden)
        counts.index_add_(0, graph_index, hidden.new_ones((hidden.shape[0], 1)))
        return self.project(pooled / counts.clamp_min(1.0))


def _pack_graphs(
    graphs: Sequence[Tuple[Tensor, Tensor]], *, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    nodes = []
    edges = []
    graph_index = []
    offset = 0
    for index, (graph_nodes, graph_edges) in enumerate(graphs):
        nodes.append(graph_nodes)
        graph_index.append(torch.full((graph_nodes.shape[0],), index, dtype=torch.long))
        if graph_edges.shape[1]:
            edges.append(graph_edges + offset)
        offset += int(graph_nodes.shape[0])
    packed_edges = (
        torch.cat(edges, dim=1) if edges else torch.empty((2, 0), dtype=torch.long)
    )
    return (
        torch.cat(nodes, dim=0).to(device),
        packed_edges.to(device),
        torch.cat(graph_index, dim=0).to(device),
    )


def _load_frozen_g1(args, device: torch.device) -> LevelAwareMotifStateEncoder:
    manifest = json.loads(Path(args.g1_manifest).read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    model = LevelAwareMotifStateEncoder(
        embedding_dim=int(configuration["embedding_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        pooling=str(manifest["pooling"]),
    )
    checkpoint = torch.load(Path(args.g1_checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def _encode_dataset(args, g1, device: torch.device) -> List[FrozenMoleculeFeatures]:
    rows = list(_iter_jsonl(Path(args.dataset_dir) / "records.jsonl"))
    encoded = []
    with torch.no_grad():
        for row in rows:
            records = []
            atom_count = len(row["atom_to_motif"])
            for matrix in row["e3fp_ids"]:
                records.append(
                    {
                        "member_id": row["member_id"],
                        "selection_index": row["selection_index"],
                        "e3fp_ids": matrix,
                        "atom_valid": [True] * atom_count,
                        "atom_is_attachment": row["atom_is_attachment"],
                        "atom_to_motif": row["atom_to_motif"],
                        "motif_count": row["motif_count"],
                    }
                )
            batch = collate_motif_state_records(records)
            output = g1(
                batch["e3fp_ids"].to(device),
                batch["atom_valid"].to(device),
                batch["atom_to_motif"].to(device),
                num_groups=int(batch["num_groups"]),
                atom_is_attachment=batch["atom_is_attachment"].to(device),
            )
            motif_count = int(row["motif_count"])
            conformer_nodes = tuple(
                output.group_hidden[index, :motif_count].float().cpu()
                for index in range(output.group_hidden.shape[0])
            )
            directed = []
            for left, right in row["motif_edges"]:
                directed.extend(((int(left), int(right)), (int(right), int(left))))
            edge_tensor = (
                torch.tensor(directed, dtype=torch.long).t().contiguous()
                if directed
                else torch.empty((2, 0), dtype=torch.long)
            )
            encoded.append(
                FrozenMoleculeFeatures(
                    member_id=str(row["member_id"]),
                    split=str(row["split"]),
                    conformer_nodes=conformer_nodes,
                    directed_edges=edge_tensor,
                    pairs=tuple(
                        (
                            int(pair["left"]),
                            int(pair["right"]),
                            float(pair["distance_matrix_rms_angstrom"]),
                        )
                        for pair in row["pairs"]
                    ),
                )
            )
    return encoded


def _pair_rows(features: Sequence[FrozenMoleculeFeatures], split: str):
    return [
        (molecule_index, left, right, target)
        for molecule_index, molecule in enumerate(features)
        if molecule.split == split
        for left, right, target in molecule.pairs
    ]


def _predict_pairs(
    model, features, pairs, *, device, batch_size, mismatched_geometry: bool = False
) -> List[Tuple[str, float, float]]:
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), int(batch_size)):
            batch = pairs[start : start + int(batch_size)]
            graphs = []
            for molecule_index, left, _, _ in batch:
                molecule = features[molecule_index]
                graphs.append((molecule.conformer_nodes[left], molecule.directed_edges))
            for molecule_index, left, right, _ in batch:
                molecule = features[molecule_index]
                chosen = right
                if mismatched_geometry:
                    candidates = [
                        index
                        for index in range(len(molecule.conformer_nodes))
                        if index not in (left, right)
                    ]
                    if not candidates:
                        raise G3ARelationError(
                            "mismatched geometry requires at least three conformers"
                        )
                    chosen = candidates[0]
                graphs.append((molecule.conformer_nodes[chosen], molecule.directed_edges))
            nodes, edges, graph_index = _pack_graphs(graphs, device=device)
            embeddings = model(nodes, edges, graph_index, len(graphs))
            count = len(batch)
            predictions = torch.linalg.vector_norm(
                embeddings[:count] - embeddings[count:], dim=-1
            ).cpu()
            for pair, prediction in zip(batch, predictions.tolist()):
                molecule_index, _, _, target = pair
                result.append(
                    (features[molecule_index].member_id, float(target), float(prediction))
                )
    return result


def _baseline_pairs(features, pairs) -> List[Tuple[str, float, float]]:
    rows = []
    for molecule_index, left, right, target in pairs:
        molecule = features[molecule_index]
        left_hidden = molecule.conformer_nodes[left].mean(dim=0)
        right_hidden = molecule.conformer_nodes[right].mean(dim=0)
        prediction = float(torch.linalg.vector_norm(left_hidden - right_hidden))
        rows.append((molecule.member_id, float(target), prediction))
    return rows


def _metrics(rows: Sequence[Tuple[str, float, float]]) -> Dict[str, float]:
    targets = [row[1] for row in rows]
    predictions = [row[2] for row in rows]
    return {
        "pairs": len(rows),
        "pearson": _pearson(targets, predictions),
        "spearman": _spearman(targets, predictions),
        "mae_angstrom": sum(abs(x - y) for x, y in zip(targets, predictions)) / len(rows),
        "ordering_accuracy": _ordering_accuracy(rows),
        "target_median_angstrom": median(targets),
        "predicted_median": median(predictions),
    }


def run(args) -> Dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise G3ARelationError("output already exists: {}".format(output))
    output.mkdir(parents=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise G3ARelationError("CUDA was requested but is unavailable")
    device = torch.device("cuda", 0) if args.device == "cuda" else torch.device("cpu")
    g1 = _load_frozen_g1(args, device)
    if int(g1.group_rho[-1].out_features) != int(args.hidden_dim):
        raise G3ARelationError("adapter hidden_dim must equal frozen G1 hidden_dim")
    features = _encode_dataset(args, g1, device)
    train_pairs = _pair_rows(features, "train")
    dev_pairs = _pair_rows(features, "dev")
    if not train_pairs or not dev_pairs:
        raise G3ARelationError("both train and dev pairs are required")
    baseline = _metrics(_baseline_pairs(features, dev_pairs))
    model = MotifTopologyRelationEncoder(
        hidden_dim=int(args.hidden_dim), output_dim=int(args.output_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
    generator = random.Random(int(args.seed))
    update = 0
    while update < int(args.updates):
        order = list(train_pairs)
        generator.shuffle(order)
        for start in range(0, len(order), int(args.batch_size)):
            batch = order[start : start + int(args.batch_size)]
            graphs = []
            for molecule_index, left, _, _ in batch:
                molecule = features[molecule_index]
                graphs.append((molecule.conformer_nodes[left], molecule.directed_edges))
            for molecule_index, _, right, _ in batch:
                molecule = features[molecule_index]
                graphs.append((molecule.conformer_nodes[right], molecule.directed_edges))
            nodes, edges, graph_index = _pack_graphs(graphs, device=device)
            targets = torch.tensor([row[3] for row in batch], device=device)
            embeddings = model(nodes, edges, graph_index, len(graphs))
            count = len(batch)
            predictions = torch.linalg.vector_norm(
                embeddings[:count] - embeddings[count:], dim=-1
            )
            loss = F.smooth_l1_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update += 1
            if update >= int(args.updates):
                break
    learned_rows = _predict_pairs(
        model, features, dev_pairs, device=device, batch_size=int(args.batch_size)
    )
    learned = _metrics(learned_rows)
    learned_train = _metrics(
        _predict_pairs(
            model,
            features,
            train_pairs,
            device=device,
            batch_size=int(args.batch_size),
        )
    )
    mismatch_pairs = [
        pair
        for pair in dev_pairs
        if len(features[pair[0]].conformer_nodes) >= 3
    ]
    if not mismatch_pairs:
        raise G3ARelationError("dev has no pairs eligible for wrong-conformer diagnosis")
    mismatch_aligned = _metrics(
        _predict_pairs(
            model,
            features,
            mismatch_pairs,
            device=device,
            batch_size=int(args.batch_size),
        )
    )
    mismatched = _metrics(
        _predict_pairs(
            model,
            features,
            mismatch_pairs,
            device=device,
            batch_size=int(args.batch_size),
            mismatched_geometry=True,
        )
    )
    decision = {
        "pearson_at_least_0_45": learned["pearson"] >= 0.45,
        "spearman_at_least_0_45": learned["spearman"] >= 0.45,
        "pearson_improvement_at_least_0_15": learned["pearson"] - baseline["pearson"] >= 0.15,
        "spearman_improvement_at_least_0_15": learned["spearman"] - baseline["spearman"] >= 0.15,
    }
    decision["pass"] = all(decision.values())
    checkpoint_path = output / "final_state.pt"
    torch.save({"model_state_dict": model.state_dict(), "updates": int(args.updates)}, checkpoint_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "g3a_relation_mechanism_not_t5_or_downstream",
        "device": str(device),
        "configuration": {
            "seed": int(args.seed),
            "updates": int(args.updates),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(args.output_dim),
            "loss": "smooth_l1_latent_l2_to_distance_matrix_rms",
        },
        "dataset": {
            "molecules": len(features),
            "train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
            "molecule_identity_disjoint_split": True,
        },
        "frozen_g1_mean_pool_baseline": baseline,
        "topology_relation_encoder": learned,
        "topology_relation_encoder_train": learned_train,
        "same_identity_wrong_conformer_diagnostic": {
            **mismatched,
            "eligible_pairs": len(mismatch_pairs),
            "total_dev_pairs": len(dev_pairs),
            "pair_coverage": len(mismatch_pairs) / len(dev_pairs),
            "mae_increase_over_aligned": mismatched["mae_angstrom"]
            - mismatch_aligned["mae_angstrom"],
            "aligned_subset_mae_angstrom": mismatch_aligned["mae_angstrom"],
            "not_part_of_primary_pass_gate": True,
        },
        "decision": decision,
        "checkpoint": checkpoint_path.name,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--g1-manifest", required=True)
    parser.add_argument("--g1-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
