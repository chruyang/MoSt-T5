"""Launch task-homogeneous four-rank MoSt-T5 pretraining with torchrun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from most_t5_next.configuration import load_pretraining_config
from most_t5_next.modeling.loading import load_model_from_config
from most_t5_next.training.data_provider import CurriculumDataLoaderProvider
from most_t5_next.training.distributed import (
    rank_task_assignment,
    task_batch_partitions,
)
from most_t5_next.training.runner import (
    read_checkpoint_metadata,
    run_two_phase_pretraining,
)
from most_t5_next.training.runtime import runtime_from_config


POPULATION_SCHEMA = "most-t5/pretraining-populations/v2"
IDENTITY_FILES = (
    "manifest.json",
    "validation_receipt.json",
    "training_manifest.json",
    "config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(root: Path) -> dict[str, str]:
    """Fingerprint compact manifests without hashing multi-gigabyte tensors."""

    identity = {
        name: _sha256(root / name)
        for name in IDENTITY_FILES
        if (root / name).is_file()
    }
    if not identity:
        raise ValueError(f"artifact has no identity manifest: {root}")
    return identity


def _checkpoint_protocol(args: argparse.Namespace, config: Mapping[str, object]) -> dict:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "resolved_config_sha256": hashlib.sha256(canonical).hexdigest(),
        "base_model": _artifact_identity(args.checkpoint),
        "tokenizer": _artifact_identity(args.tokenizer_root),
        "pcqm_cache": _artifact_identity(args.pcqm_cache),
        "pubchem_cache": _artifact_identity(args.pubchem_cache),
        "paired_text_cache": _artifact_identity(args.paired_text_cache),
        "pubmed_cache": _artifact_identity(args.pubmed_cache),
        "populations": _artifact_identity(args.population_root),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _set_launch_status(
    path: Path, status: str, *, error: BaseException | None = None
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    attempt = payload["attempts"][-1]
    attempt["status"] = status
    if error is not None:
        attempt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    _write_json_atomic(path, payload)


def _tokenizer_contract(root: Path) -> tuple[int, tuple[int, ...], int]:
    tokenizer = AutoTokenizer.from_pretrained(
        root / "tokenizer_snapshot", use_fast=False, local_files_only=True
    )
    pad = int(tokenizer.pad_token_id)
    eos = int(tokenizer.eos_token_id)
    sentinels = tuple(
        int(tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>"))
        for index in range(100)
    )
    if pad < 0 or eos < 0 or min(sentinels) < 0 or len(set(sentinels)) != 100:
        raise ValueError("tokenizer PAD/EOS/sentinel contract failed")
    return pad, sentinels, eos


def _load_populations(root: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != POPULATION_SCHEMA
        or manifest.get("status") != "pass"
        or manifest.get("training_admission") is not True
    ):
        raise ValueError("pretraining population manifest is not admitted")
    populations: dict[str, np.ndarray] = {}
    for task in ("M", "MG", "SYN", "CAP", "T2M", "TXT"):
        descriptor = manifest["arrays"][task]
        path = root / descriptor["file"]
        values = np.memmap(path, mode="r", dtype=descriptor["dtype"])
        if values.shape != tuple(descriptor["shape"]) or not len(values):
            raise ValueError(f"population array is invalid: {task}")
        populations[task] = values
    return populations


def _phase_populations(
    populations: Mapping[str, Sequence[int]], tasks: Sequence[str]
) -> dict[str, Sequence[int]]:
    return {task: populations[task] for task in tasks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--pcqm-cache", type=Path, required=True)
    parser.add_argument("--pubchem-cache", type=Path, required=True)
    parser.add_argument("--paired-text-cache", type=Path, required=True)
    parser.add_argument("--pubmed-cache", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume the matching phase from a checkpoint inside --output-dir",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pretraining_config(
        args.config, overrides=args.overrides, require_launch_values=True
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size = int(config["distributed"]["world_size"])
    if world_size != expected_world_size:
        raise RuntimeError(
            f"formal pretraining requires torchrun with {expected_world_size} ranks"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("formal distributed pretraining requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    resume_metadata = (
        read_checkpoint_metadata(args.resume_checkpoint)
        if args.resume_checkpoint is not None
        else None
    )
    output_state = (
        {
            "exists": args.output_dir.exists(),
            "checkpoint_inside_output": args.resume_checkpoint.resolve().parent
            == args.output_dir.resolve(),
        }
        if rank == 0 and args.resume_checkpoint is not None
        else {"exists": args.output_dir.exists()} if rank == 0 else None
    )
    state = [output_state]
    dist.broadcast_object_list(state, src=0)
    if args.resume_checkpoint is None and state[0]["exists"]:
        raise FileExistsError(f"output already exists: {args.output_dir}")
    if args.resume_checkpoint is not None:
        if not state[0]["exists"]:
            raise FileNotFoundError("resume output directory does not exist")
        if not state[0]["checkpoint_inside_output"]:
            raise ValueError("resume checkpoint must be a direct child of --output-dir")
    runtime = runtime_from_config(config)
    checkpoint_protocol = _checkpoint_protocol(args, config)
    if resume_metadata is not None and resume_metadata["protocol"] != checkpoint_protocol:
        raise RuntimeError("resume checkpoint data/config identity differs from this launch")
    pad, sentinels, eos = _tokenizer_contract(args.tokenizer_root)
    populations = _load_populations(args.population_root)
    provider_kwargs = {
        "pcqm_cache": args.pcqm_cache,
        "pubchem_cache": args.pubchem_cache,
        "paired_text_cache": args.paired_text_cache,
        "pubmed_cache": args.pubmed_cache,
        "pad_token_id": pad,
        "sentinel_token_ids": sentinels,
        "eos_token_id": eos,
        "runtime": runtime,
    }
    resume_phase = resume_metadata["phase"] if resume_metadata is not None else None
    resume_update = (
        int(resume_metadata["next_update"])
        if resume_metadata is not None
        else 0
    )
    phase_one = None
    if resume_phase != 2:
        phase_one_assignment = rank_task_assignment(
            config, phase=1, rank=rank, world_size=world_size
        )
        phase_one_partitions = task_batch_partitions(config, phase=1)
        phase_one = CurriculumDataLoaderProvider(
            phase=1,
            total_updates=int(config["curriculum"]["phase_one"]["total_updates"]),
            populations=_phase_populations(populations, (phase_one_assignment.task,)),
            start_update=resume_update if resume_phase == 1 else 0,
            task_partitions={
                phase_one_assignment.task: phase_one_partitions[
                    phase_one_assignment.task
                ]
            },
            fixed_task=phase_one_assignment.task,
            task_replica_index=phase_one_assignment.task_replica_index,
            task_replicas=phase_one_assignment.task_replicas,
            **provider_kwargs,
        )
    phase_one_closed = phase_one is None
    phase_two_holder: list[CurriculumDataLoaderProvider] = []

    def phase_two_factory() -> CurriculumDataLoaderProvider:
        nonlocal phase_one_closed
        if phase_one is not None:
            phase_one.close()
        phase_one_closed = True
        assignment = rank_task_assignment(
            config, phase=2, rank=rank, world_size=world_size
        )
        partitions = task_batch_partitions(config, phase=2)
        provider = CurriculumDataLoaderProvider(
            phase=2,
            total_updates=int(config["curriculum"]["phase_two"]["total_updates"]),
            populations=_phase_populations(populations, (assignment.task,)),
            start_update=resume_update if resume_phase == 2 else 0,
            task_partitions={assignment.task: partitions[assignment.task]},
            fixed_task=assignment.task,
            task_replica_index=assignment.task_replica_index,
            task_replicas=assignment.task_replicas,
            **provider_kwargs,
        )
        phase_two_holder.append(provider)
        return provider
    model = load_model_from_config(args.checkpoint, config).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=bool(
            config["distributed"]["find_unused_parameters"]
        ),
    )
    tensorboard = Path(config["monitoring"]["tensorboard_root"]) / args.output_dir.name
    writer = SummaryWriter(tensorboard) if rank == 0 else None
    launch_manifest_path = args.output_dir / "launch-manifest.json"
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        previous_attempts = []
        if args.resume_checkpoint is not None and launch_manifest_path.is_file():
            previous_attempts = json.loads(
                launch_manifest_path.read_text(encoding="utf-8")
            ).get("attempts", [])
        attempt = {
            "status": "running",
            "git_commit": _git_commit(),
            "resume_checkpoint": str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None
            else None,
            "resume_metadata": resume_metadata,
        }
        launch_manifest = {
            "schema_version": "most-t5/pretraining-launch/v1",
            "status": "running",
            "world_size": world_size,
            "checkpoint_protocol": checkpoint_protocol,
            "resolved_config": config,
            "attempts": [*previous_attempts, attempt],
        }
        _write_json_atomic(launch_manifest_path, launch_manifest)
    dist.barrier()
    try:
        report = run_two_phase_pretraining(
            model=model,
            phase_one_batch_provider=phase_one,
            phase_two_batch_provider_factory=phase_two_factory,
            config=config,
            output_dir=args.output_dir,
            device=device,
            resume_checkpoint=args.resume_checkpoint,
            checkpoint_protocol=checkpoint_protocol,
            writer=writer,
        )
    except BaseException as error:
        if rank == 0 and launch_manifest_path.is_file():
            _set_launch_status(launch_manifest_path, "interrupted", error=error)
        raise
    finally:
        if writer is not None:
            writer.close()
        if not phase_one_closed and phase_one is not None:
            phase_one.close()
        for provider in phase_two_holder:
            provider.close()
        dist.destroy_process_group()
    if rank == 0:
        _set_launch_status(launch_manifest_path, "pass")
        print(json.dumps({"status": report["status"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
