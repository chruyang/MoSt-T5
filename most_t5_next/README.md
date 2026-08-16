# MoSt-T5

MoSt-T5 is a T5 encoder-decoder with motif-level 3D state injection. Molecular
sequences use fragSMILES carriers and endpoints as the only geometry-bearing
token positions; ordinary text follows the unchanged T5 embedding path.
Training starts from the frozen 53,368-row union checkpoint; loading the raw
T5-v1.1 checkpoint directly is invalid because registered motif IDs would lie
outside its embedding and language-model head.

## Pretraining curriculum

Phase I learns molecular structure. `M` denoises the PCQM + PubChem molecular
union without geometry, while `MG` denoises PCQM with geometry. Four DDP ranks
form each update as `[M, M, MG, MG]`.

Phase II grounds that representation in language. `SYN` preserves molecular
denoising with geometry, `TXT` preserves T5 text denoising, `CAP` generates a
description from a geometry-enabled molecule, and `T2M` generates the molecular
sequence from text. Its rank layout is `[SYN, TXT, CAP, T2M]`. Every rank owns
96 logical records, so one synchronized update contains 384 records. Phase II
starts from the Phase I model weights with a fresh optimizer and learning-rate
schedule.

The formal schedule is frozen at 100,000 Phase-I optimizer updates followed by
200,000 Phase-II optimizer updates. Phase I warms from `1e-3` to `2e-3` and
Phase II from `5e-4` to `1e-3`, each over 10,000 updates, then each phase uses
cosine decay to `1e-5`. Two ranks serve each Phase-I task and one rank serves
each Phase-II task, so every task receives the same 19.2 million record
presentations. Shorter overrides are execution smokes only. Full-state atomic
checkpoints are written every 10,000 updates and at phase end.

Physical microbatching is task-specific while the logical batch remains 96 on
every rank: Phase I uses `M=48 x 2`, `MG=96 x 1`; Phase II uses
`SYN/TXT=48 x 2`, `CAP/T2M=32 x 3`. Only the last backward pass synchronizes,
so each rank still contributes once to each optimizer update.

## Active code

- `data/`: zero-copy cache union and task-to-dataset routing;
- `data/motif_corruption.py`: heavy-atom sampling of compound motif/endpoint units;
- `modeling/`: E3FP shell embeddings, carrier/endpoint adapter, and T5 wrapper;
- `training/`: balanced task schedule, optimizer, shared forward path, and phase handoff;
- `configs/`: public training configuration with phase-local launch values;
- `scripts/`: focused model and data-path smoke tests.

`configuration.py` validates the public YAML. Architecture, corruption, batch,
data-loader, phase-budget, learning-rate, and checkpoint defaults are usable as
written. Resume restores model, optimizer, scheduler, progress counters, and
Python/NumPy/Torch/CUDA random-number generators; it also verifies the frozen
configuration and data identities before continuing.

See `PUBLIC_RELEASE.md` for the boundary between the active implementation and
the experiment history retained in this workspace.
