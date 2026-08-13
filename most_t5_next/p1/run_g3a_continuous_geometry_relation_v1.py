#!/usr/bin/env python3
"""G3a continuous-distance baseline before any geometry-to-T5 bridge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from most_t5_next.p1.run_g3a_motif_relation_v1 import (
    DEFAULT_SEED,
    G3ARelationError,
    MotifTopologyRelationEncoder,
    _metrics,
)


SCHEMA_VERSION = "most-t5-p1/g3a-continuous-geometry-relation/v1"


@dataclass(frozen=True)
class ContinuousMolecule:
    member_id: str
    split: str
    atomic_numbers: Tensor
    conformer_positions: Tuple[Tensor, ...]
    atom_to_motif: Tensor
    motif_count: int
    directed_motif_edges: Tensor
    pairs: Tuple[Tuple[int, int, float], ...]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise G3ARelationError("continuous relation record is not an object")
                yield value


def _load_dataset(dataset_dir: Path) -> List[ContinuousMolecule]:
    molecules = []
    for row in _iter_jsonl(dataset_dir / "records.jsonl"):
        directed = []
        for left, right in row["motif_edges"]:
            directed.extend(((int(left), int(right)), (int(right), int(left))))
        motif_edges = (
            torch.tensor(directed, dtype=torch.long).t().contiguous()
            if directed
            else torch.empty((2, 0), dtype=torch.long)
        )
        atomic_numbers = torch.tensor(row["atomic_numbers"], dtype=torch.long)
        positions = tuple(
            torch.tensor(value, dtype=torch.float32)
            for value in row["conformer_positions"]
        )
        if any(position.shape != (atomic_numbers.shape[0], 3) for position in positions):
            raise G3ARelationError("position and atom domains disagree")
        molecules.append(
            ContinuousMolecule(
                member_id=str(row["member_id"]),
                split=str(row["split"]),
                atomic_numbers=atomic_numbers,
                conformer_positions=positions,
                atom_to_motif=torch.tensor(row["atom_to_motif"], dtype=torch.long),
                motif_count=int(row["motif_count"]),
                directed_motif_edges=motif_edges,
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
    return molecules


def _pair_rows(molecules: Sequence[ContinuousMolecule], split: str):
    return [
        (molecule_index, left, right, target)
        for molecule_index, molecule in enumerate(molecules)
        if molecule.split == split
        for left, right, target in molecule.pairs
    ]


def _complete_directed_edges(atom_count: int) -> Tensor:
    if atom_count < 2:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        [(left, right) for left in range(atom_count) for right in range(atom_count) if left != right],
        dtype=torch.long,
    ).t().contiguous()


def _pack_graphs(graphs, *, device: torch.device):
    atomic_numbers = []
    positions = []
    atom_edges = []
    atom_to_motif = []
    motif_edges = []
    motif_graph_index = []
    atom_offset = motif_offset = 0
    for graph_index, (molecule, conformer_index) in enumerate(graphs):
        atom_count = int(molecule.atomic_numbers.shape[0])
        atomic_numbers.append(molecule.atomic_numbers)
        positions.append(molecule.conformer_positions[int(conformer_index)])
        local_atom_edges = _complete_directed_edges(atom_count)
        if local_atom_edges.shape[1]:
            atom_edges.append(local_atom_edges + atom_offset)
        atom_to_motif.append(molecule.atom_to_motif + motif_offset)
        if molecule.directed_motif_edges.shape[1]:
            motif_edges.append(molecule.directed_motif_edges + motif_offset)
        motif_graph_index.append(
            torch.full((molecule.motif_count,), graph_index, dtype=torch.long)
        )
        atom_offset += atom_count
        motif_offset += molecule.motif_count
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    return {
        "atomic_numbers": torch.cat(atomic_numbers).to(device),
        "positions": torch.cat(positions).to(device),
        "atom_edges": (torch.cat(atom_edges, dim=1) if atom_edges else empty_edges).to(device),
        "atom_to_motif": torch.cat(atom_to_motif).to(device),
        "num_motifs": motif_offset,
        "motif_edges": (torch.cat(motif_edges, dim=1) if motif_edges else empty_edges).to(device),
        "motif_graph_index": torch.cat(motif_graph_index).to(device),
        "num_graphs": len(graphs),
    }


class ContinuousDistanceMotifEncoder(nn.Module):
    """Small SchNet-style invariant atom encoder followed by motif topology."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        output_dim: int = 64,
        radial_basis: int = 64,
        max_distance: float = 16.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.atom_embedding = nn.Embedding(119, self.hidden_dim, padding_idx=0)
        centers = torch.linspace(0.0, float(max_distance), int(radial_basis))
        self.register_buffer("rbf_centers", centers)
        spacing = float(centers[1] - centers[0])
        self.rbf_gamma = 1.0 / (spacing * spacing)
        self.filters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(radial_basis), self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.sources = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(3)]
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in range(3)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(3)])
        self.motif_head = MotifTopologyRelationEncoder(
            hidden_dim=self.hidden_dim, output_dim=int(output_dim)
        )

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        atomic_numbers = batch["atomic_numbers"]
        positions = batch["positions"]
        edges = batch["atom_edges"]
        hidden = self.atom_embedding(atomic_numbers)
        if edges.shape[1]:
            source, target = edges[0], edges[1]
            distances = torch.linalg.vector_norm(positions[source] - positions[target], dim=-1)
            rbf = torch.exp(
                -self.rbf_gamma * (distances.unsqueeze(-1) - self.rbf_centers) ** 2
            )
        for filter_network, source_network, update, norm in zip(
            self.filters, self.sources, self.updates, self.norms
        ):
            aggregate = torch.zeros_like(hidden)
            if edges.shape[1]:
                messages = source_network(hidden[source]) * filter_network(rbf)
                aggregate.index_add_(0, target, messages)
            hidden = norm(hidden + update(aggregate))
        motif_hidden = hidden.new_zeros((int(batch["num_motifs"]), self.hidden_dim))
        motif_counts = hidden.new_zeros((int(batch["num_motifs"]), 1))
        motif_hidden.index_add_(0, batch["atom_to_motif"], hidden)
        motif_counts.index_add_(
            0, batch["atom_to_motif"], hidden.new_ones((hidden.shape[0], 1))
        )
        motif_hidden = motif_hidden / motif_counts.clamp_min(1.0)
        return self.motif_head(
            motif_hidden,
            batch["motif_edges"],
            batch["motif_graph_index"],
            int(batch["num_graphs"]),
        )


def _predict(model, molecules, pairs, *, device, batch_size, mismatched=False):
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), int(batch_size)):
            selected = pairs[start : start + int(batch_size)]
            graphs = [(molecules[index], left) for index, left, _, _ in selected]
            for index, left, right, _ in selected:
                chosen = right
                if mismatched:
                    alternatives = [
                        value
                        for value in range(len(molecules[index].conformer_positions))
                        if value not in (left, right)
                    ]
                    if not alternatives:
                        raise G3ARelationError("wrong-conformer pair is ineligible")
                    chosen = alternatives[0]
                graphs.append((molecules[index], chosen))
            embeddings = model(_pack_graphs(graphs, device=device))
            count = len(selected)
            predictions = torch.linalg.vector_norm(
                embeddings[:count] - embeddings[count:], dim=-1
            ).cpu().tolist()
            for pair, prediction in zip(selected, predictions):
                rows.append((molecules[pair[0]].member_id, float(pair[3]), float(prediction)))
    return rows


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
    molecules = _load_dataset(Path(args.dataset_dir).expanduser().resolve())
    train_pairs = _pair_rows(molecules, "train")
    dev_pairs = _pair_rows(molecules, "dev")
    model = ContinuousDistanceMotifEncoder(
        hidden_dim=int(args.hidden_dim),
        output_dim=int(args.output_dim),
        radial_basis=int(args.radial_basis),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
    generator = random.Random(int(args.seed))
    update_index = 0
    while update_index < int(args.updates):
        order = list(train_pairs)
        generator.shuffle(order)
        for start in range(0, len(order), int(args.batch_size)):
            selected = order[start : start + int(args.batch_size)]
            graphs = [(molecules[index], left) for index, left, _, _ in selected]
            graphs += [(molecules[index], right) for index, _, right, _ in selected]
            embeddings = model(_pack_graphs(graphs, device=device))
            count = len(selected)
            predictions = torch.linalg.vector_norm(
                embeddings[:count] - embeddings[count:], dim=-1
            )
            targets = torch.tensor([pair[3] for pair in selected], device=device)
            loss = F.smooth_l1_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_index += 1
            if update_index >= int(args.updates):
                break
    train_metrics = _metrics(
        _predict(model, molecules, train_pairs, device=device, batch_size=args.batch_size)
    )
    dev_metrics = _metrics(
        _predict(model, molecules, dev_pairs, device=device, batch_size=args.batch_size)
    )
    mismatch_pairs = [
        pair
        for pair in dev_pairs
        if len(molecules[pair[0]].conformer_positions) >= 3
    ]
    aligned_subset = _metrics(
        _predict(model, molecules, mismatch_pairs, device=device, batch_size=args.batch_size)
    )
    wrong = _metrics(
        _predict(
            model,
            molecules,
            mismatch_pairs,
            device=device,
            batch_size=args.batch_size,
            mismatched=True,
        )
    )
    decision = {
        "pearson_at_least_0_45": dev_metrics["pearson"] >= 0.45,
        "spearman_at_least_0_45": dev_metrics["spearman"] >= 0.45,
        "wrong_conformer_mae_higher": wrong["mae_angstrom"] > aligned_subset["mae_angstrom"],
    }
    decision["pass"] = all(decision.values())
    checkpoint = output / "final_state.pt"
    torch.save({"model_state_dict": model.state_dict(), "updates": int(args.updates)}, checkpoint)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "scope": "g3a_continuous_geometry_mechanism_not_t5_or_downstream",
        "device": str(device),
        "configuration": {
            "seed": int(args.seed),
            "updates": int(args.updates),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "hidden_dim": int(args.hidden_dim),
            "output_dim": int(args.output_dim),
            "radial_basis": int(args.radial_basis),
            "atom_graph": "complete_directed_distance_rbf",
            "motif_interface": "atom_mean_incidence_then_two_layer_motif_message_passing",
        },
        "dataset": {
            "molecules": len(molecules),
            "train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
            "identity_disjoint": True,
        },
        "train": train_metrics,
        "dev": dev_metrics,
        "same_identity_wrong_conformer": {
            **wrong,
            "aligned_subset_mae_angstrom": aligned_subset["mae_angstrom"],
            "mae_increase_over_aligned": wrong["mae_angstrom"]
            - aligned_subset["mae_angstrom"],
            "pair_coverage": len(mismatch_pairs) / len(dev_pairs),
        },
        "decision": decision,
        "checkpoint": checkpoint.name,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--radial-basis", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()

