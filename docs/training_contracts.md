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
microbatches. The current single-GPU baseline uses 96 records per optimizer
update (`32 x 3`). For a task update, the sampler:

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

Population freezing uses the same exposure equation:

```text
max_epoch = floor((task_updates * logical_batch_size - 1) / population_size)
```

The training manifest records the logical batch, physical partition, and
`sample_before_microbatch_split=true`. The population manifest records the
same contract.

## Version boundary

The first westd commits carrying these contracts are:

- `47e2449d071cf369cea78a84ed6fd37db68d8149` -- row-wise fallback geometry
  validation;
- `ab5be2e517cb1448ded24564078162ab04d4daf7` -- logical-batch-first sampling.

Any training checkout must contain both changes or a descendant, use a clean
worktree, and persist its full commit in the launch manifest. The remote branch
contains westd-only commits beyond the current GitHub branch; the complete
series must be pushed or rebased deliberately before formal training rather
than copied as an unversioned directory.
