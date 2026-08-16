# Formal pretraining checkpoint and resume contract

## Frozen cadence

Both formal phases save at every 10,000 completed optimizer updates and at the
phase end. With budgets of 100,000 and 200,000 updates, every regular save lies
on an optimizer boundary. Checkpoint files use
`phase-{phase}-step-{completed:08d}.pt`; `latest-checkpoint.json` is an atomic
pointer to the last successfully renamed file. The tensor payload, metadata
sidecar, and pointer are flushed with `fsync` before their atomic replacements
become authoritative.

## State carried across an accidental restart

Within a phase, resume restores:

- model parameters and buffers;
- all AdamWScale optimizer tensors and parameter-group values;
- learning-rate scheduler state and completed-update count;
- the next global optimizer update;
- per-rank task, record, target-token, loss, and physical-partition counters;
- Python, NumPy, Torch CPU, and rank-local CUDA RNG states; and
- the resolved optimization/runtime contract and compact source-artifact
  fingerprints.

The curriculum sampler reconstructs its cursor from `next_update`, task replica,
population, and seed. It therefore resumes the exact logical batch stream before
physical microbatch splitting. Worker-side online corruption is derived from
sample identity and epoch rather than an untracked worker cursor.

## Phase boundary

The intentional Phase-I to Phase-II transition remains model-only: Phase II
starts from the Phase-I model but creates a new optimizer and scheduler. An
accidental interruption inside Phase II is different and restores the Phase-II
optimizer and scheduler from its checkpoint. Phase-II resume additionally
requires the persisted Phase-I boundary file and admitted Phase-I manifest.

## Operational resume

The formal launcher accepts `--resume-checkpoint`. The checkpoint must be a
direct child of the same existing `--output-dir`; all configuration and data
arguments must be identical to the original launch. Before constructing the
DataLoader, the launcher reads the checkpoint phase and `next_update`, then
creates only the required phase provider at that exact update.

The runner subsequently loads the tensor payload and compares its authoritative
`next_update` with the provider sampler's `start_update`. If a stale or damaged
metadata sidecar would have selected a different cursor, resume fails before
the next forward pass.

The resume gate rejects protocol, runtime, world-size, population-manifest, and
source-cache identity drift. This prevents a syntactically successful restart
from silently changing the experiment.

## Verification gate

The CPU test intentionally interrupts after a stable checkpoint, reinitializes
the process state, resumes, and compares the final model, optimizer, scheduler,
progress ledger, and RNG states against an uninterrupted reference. A four-rank
GPU smoke remains required after the instance is switched from no-GPU mode.
