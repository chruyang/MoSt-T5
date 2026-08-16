# Public release boundary

The public MoSt-T5 implementation is the small active path below:

- `configs/`: model, corruption, curriculum, and optimization settings;
- `data/`: PCQM, PubChem, paired-text, and PubMed routing plus compound motif corruption;
- `modeling/`: E3FP embeddings, fragSMILES geometry injection, and T5 wrapper;
- `training/`: task schedule, optimizer, model calls, and phase handoff.
- `scripts/`: focused readiness and information-flow checks.

The `p1/`, `p2/`, and `r1/` directories contain experiment-specific builders,
audits, and historical baselines. They remain in the research workspace for
reproducibility but are not part of the first public package. A release should
be assembled from the active path instead of publishing the workspace tree.

The formal phase budgets in `configs/pretrain.yaml` are frozen at 100,000 and
200,000 optimizer updates. Phase-I and Phase-II peak learning rates are `2e-3`
and `1e-3`; both use 10,000 linear warmup updates from factor `0.5` followed by
cosine decay to `1e-5`. Full-state checkpoints are written every 10,000 updates.
Formal pretraining has no validation split or evaluation loop.

The public input boundary is `MoStT5Processor` plus `MoStT5Collator`. It
supports text-only, molecule-only, joint text-plus-molecule and mixed batches.
Geometry availability is data-driven: absent or disabled E3FP is represented
by all `-1` per row rather than inferred from a pretraining task name.

Common scientific controls must remain open in the same practical spirit as
3D-MolT5: dropout, seed, precision, optimizer/scheduler and learning-rate
settings, warmup, weight decay, gradient clipping, batch/accumulation,
DataLoader, logging/checkpoint and standard generation settings. Frozen MoSt-T5
values are defaults in reproducible recipe files, not unconditional rewrites in
model-loading code. Every validated setting must affect the instantiated run,
be written to its manifest and round-trip through checkpoint resume. Artifact-
dependent dimensions remain strictly compatibility-checked, and mathematically
derived tensor shapes remain derived rather than duplicated as free knobs.

Resume is phase-aware. Within a phase it restores the model, optimizer,
scheduler, completed-update boundary, cumulative counters, and Python, NumPy,
Torch CPU, and rank-local CUDA RNG states. A Phase-II resume requires the
persisted Phase-I model-only boundary and admitted Phase-I manifest, but uses
the Phase-II checkpoint's optimizer and scheduler. The checkpoint protocol
rejects a different resolved configuration, distributed runtime, population,
or source-cache identity.

The release sampler selects a complete rank-local optimizer batch before
physical microbatch splitting. Each of four task-homogeneous DDP ranks owns 96
logical records, for a global effective batch of 384. The physical Phase-I
partitions are `M=48 x 2` and `MG=96 x 1`. Phase II uses `SYN/TXT=48 x 2`
and `CAP/T2M=32 x 3`, with one gradient
reduction per optimizer update. Alternative physical partitions must retain
the same selected records and per-sample corruption epochs. Whole-molecule fallback rows use
zero fragments and unowned atoms and remain lexical-only even in mixed padded
batches.  These contracts are documented in `docs/training_contracts.md` and
must be covered by the release test suite.
