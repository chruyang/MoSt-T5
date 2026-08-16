# Training data contracts

This note records two data-boundary rules that must remain identical in local,
remote, and released MoSt-T5 versions.

## Whole-molecule fallback rows

A compact fragSMILES row has one or more active fragments. Every active atom
must own an active fragment, and every active fragment must own at least one
atom. A `whole_molecule_fallback` row instead has:

- zero active fragments;
- active atoms with `atom_to_fragment = -1`;
- no active carrier or endpoint address.

These rules are evaluated per sample. A padded batch may therefore contain a
fragmented row and a fallback row even though the shared fragment tensor has a
nonzero width. The fallback atom states cannot be injected because the row has
no carrier or endpoint destination, so its encoder path is exactly lexical.
Assigning a fallback atom to a fragment remains an error.

This per-row distinction fixes the failure first reproduced on westd at Phase-I
update 0, microbatch 0, row 43: source index 3,535,658, ordinal 19,789,253,
mode `whole_molecule_fallback`, seven atoms and zero fragments. The record was
valid; the former batch-level ownership check was not.

## Optimizer-update sampling

The logical record batch is selected before it is divided into physical
microbatches. Each distributed rank owns 96 records per optimizer update. For
a task/rank update, the sampler:

1. advances that task's deterministic shuffled stream by 96 records;
2. crosses an epoch boundary when necessary without dropping a tail or
   producing a short optimizer update;
3. retains the epoch on every sample for deterministic online corruption; and
4. divides the selected records into the configured physical microbatches.

Consequently, `96 x 1`, `48 x 2`, and `32 x 3` select the same records in the
same order when seed, task schedule, populations, and logical batch size are
equal. This is a sample-stream guarantee, not a bitwise numerical-equivalence
claim: BF16 reduction order, dynamic padding, dropout, and kernel selection can
still change floating-point results across physical partitions.

Population freezing uses the same exposure equation. Here `task_rank_updates`
is the phase optimizer-update budget multiplied by the number of ranks assigned
to that task (two for Phase-I `M` and `MG`, one for every Phase-II task):

```text
max_epoch = floor((task_rank_updates * logical_batch_size - 1) / population_size)
```

The training manifest records the logical batch, physical partition, and
`sample_before_microbatch_split=true`. The population manifest records the
same contract.

## Task-homogeneous distributed update

Formal pretraining uses four synchronized ranks and one shared model,
optimizer, scheduler, and global update clock. Phase I assigns ranks
`[M, M, MG, MG]`; the two replicas of a task draw disjoint logical batches.
Phase II assigns `[SYN, TXT, CAP, T2M]`. Every rank contributes one logical
batch of 96 records, so the global effective batch is 384 and the frozen task
weights are respectively `0.5/0.5` and `0.25/0.25/0.25/0.25`.

The physical partitions are:

```text
Phase I:  M=96x1, MG=96x1
Phase II: SYN=96x1, TXT=48x2, CAP=48x2, T2M=48x2
```

For accumulated tasks, all non-final forward/backward passes run under DDP
`no_sync`; the final pass performs the only gradient reduction of the optimizer
update. Losses are weighted by each microbatch's number of non-ignored target
tokens, making `48 x 2` equivalent to one rank-local logical token-normalized
batch rather than an unweighted mean of two variable-length losses. DDP then
gives each rank equal weight. This preserves the explicit task balance instead
of allowing a task with longer targets to dominate the global update. All ranks
then clip the synchronized gradient and execute one optimizer and scheduler
step.

Every phase update contains every phase task through its rank layout, so a
formal distributed phase budget need only be positive; it is not divided among
an alternating task cycle. Population schema `v2` records rank multiplicities
and rejects older `v1` populations at launch, forcing the exposure scan and
length-action ledger to be regenerated for this layout.

## Frozen optimizer-update budget

Formal pretraining runs 100,000 Phase-I optimizer updates and 200,000 Phase-II
optimizer updates.  This 1:2 ratio follows directly from rank multiplicity:

```text
Phase I per task:  2 ranks x 96 records x 100,000 updates = 19,200,000
Phase II per task: 1 rank  x 96 records x 200,000 updates = 19,200,000
```

All six tasks therefore receive the same record-presentation budget.  These
values are formal-recipe defaults. Reduced values are admitted only through
the explicit `--execution-smoke-updates-per-phase` launcher path. That path
requires `smoke` in the output name, records `formal_protocol=false`, and uses
a distinct resolved-config identity, so its checkpoints cannot enter or resume
the formal run.

Text-only ranks do not execute the geometry adapter, while molecular ranks do.
The DDP wrapper therefore uses `find_unused_parameters=true`; this is a
distributed execution requirement and does not change the public geometry-off
semantics.

## Checkpoint and resume boundary

Each phase writes an atomic full-state checkpoint every 10,000 completed
optimizer updates and at phase end. The checkpoint, metadata sidecar, and
`latest-checkpoint.json` are flushed and synced before atomic replacement. The
pointer is updated only after the corresponding checkpoint and metadata have
been made durable, so an interruption during a new save leaves the previous
checkpoint and pointer intact.

A resumable checkpoint contains the model, AdamWScale state, scheduler state,
completed update, per-rank task and loss counters, and Python, NumPy, Torch CPU,
and rank-local CUDA random-number generator states. It also contains the
resolved runtime and a compact identity contract for the configuration,
tokenizer, model initialization, task populations, and four source caches.
Resume refuses a different phase, world size, batching/runtime configuration,
optimization schedule, or artifact identity.

The data sampler has no hidden mutable cursor: `start_update` deterministically
reconstructs the same task-replica stream, epoch labels, logical records, and
online-corruption seeds. Consequently, recovery starts at the first update not
included in the checkpoint. After loading the tensor payload, the runner also
requires its `next_update` to equal the already-prefetched sampler's
`start_update`; metadata/payload disagreement therefore aborts instead of
silently shifting the data stream. The release test suite compares an
uninterrupted CPU run with an intentionally interrupted and resumed run,
including exact model, optimizer, scheduler, progress, and RNG equality.

At the Phase-I/Phase-II boundary the model weights are deliberately retained
while optimizer and scheduler state are deliberately restarted. This is distinct
from an accidental Phase-II interruption: resuming a Phase-II checkpoint restores
the Phase-II optimizer and scheduler rather than restarting them.

## Version boundary

The first westd commits carrying these contracts are:

- `47e2449d071cf369cea78a84ed6fd37db68d8149` -- row-wise fallback geometry
  validation;
- `ab5be2e517cb1448ded24564078162ab04d4daf7` -- logical-batch-first sampling.

Any training checkout must contain both changes or a descendant of the release
branch, have no modified tracked files, and persist its full commit in the
launch manifest. Untracked operational receipts may coexist with the checkout
but are not part of the recorded training protocol.
