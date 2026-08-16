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

Before release, the unresolved values marked `null` in `configs/pretrain.yaml`
must be replaced by the final two-phase update, phase-local learning-rate and
warmup protocol, and checkpoint cadence. Formal pretraining has no validation
split or evaluation loop. Launch validation refuses to run while any required
value remains unresolved.

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

The release sampler selects a complete optimizer-update batch before physical
microbatch splitting.  The current logical batch is 96, with `32 x 3` as the
safe single-GPU baseline; other partitions must retain the same selected
records and per-sample corruption epochs.  Whole-molecule fallback rows use
zero fragments and unowned atoms and remain lexical-only even in mixed padded
batches.  These contracts are documented in `docs/training_contracts.md` and
must be covered by the release test suite.
