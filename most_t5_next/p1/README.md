# P1 paired molecular training boundary

## Current production status (2026-08-07)

This directory remains isolated from the historical training tree, but it is
no longer limited to synthetic fixtures.  The production AtomSELFIES and
GraphPorts codecs now feed one paired A/M wire record, a frozen union
tokenizer, the shared A0/A1/M0/M1 wrapper, and the PF-1 reader/training loop.
The frozen 33,600-member PF-1 run3 release completed strict codec generation,
full LMDB replay, and a four-condition full-collator gate with zero rejects.
It is admitted only to the sample-bound PF-1 failure screen; it is not the
final full-pretraining vocabulary or evidence of model superiority.

The first completed 1,000-update four-grid screen found that the current raw
E3FP level-sum/direct-addition path loses aligned-vs-shuffled geometry
sensitivity by step 500 and does not improve A1 over A0 or M1 over M0.  That
specific CE-only fusion is therefore not admitted to PF-10 unchanged.  The
result is a fusion/objective failure-screen outcome, not evidence against
E3FP or motif-level 3D in general; see protocol document 47 for the metrics.

The authoritative current protocol and evidence are recorded in
`docs/remote_project_analysis/47_PF1_frozen_subset_training_protocol_and_resource_gate_20260807.md`.
The sections below retain the contract's development history and model
boundaries that remain relevant.

## GPU execution modes

`run_pf1_four_grid_v1.py` keeps the original sequential A0/A1/M0/M1 mode and
also accepts `--condition-id A0|A1|M0|M1`.  The latter is the supported
one-process-per-GPU boundary: four processes may read the same published LMDB,
union tokenizer and union-init checkpoint while writing four distinct output
directories.  `merge_pf1_condition_manifests_v1.py` accepts the four passed
condition manifests, verifies their shared data/model/optimization contract,
and publishes one ordered four-grid manifest.  Merely exposing more GPUs to
the sequential mode does not create parallelism.

Within each process, training uses a depth-two ordered producer queue.  The
producer decodes and collates the next update while the GPU executes the
current update.  A checkpoint records only the cursor belonging to a
successfully completed optimizer update, never the producer's prefetched
cursor.  Sync and prefetched paths have bitwise-equal batch order, dropout RNG,
parameter trajectory and step-500 resume behavior in the frozen tests.

## Historical contract skeleton

The first implementation supported deterministic synthetic fixtures and
proved the indexing skeleton needed before a tokenizer-bound release:

- macro and forced-fallback surfaces decode to one logical identity digest;
- attachment slot positions remain in identity while molecule-local edge IDs
  remain in a separate, exactly paired connection schema;
- token (L), logical motif (M), and atom (A) domains are explicit;
- every logical motif has one carrier at the first token of its identity span;
- connection, atom partition, E3FP shape, token table and digest mismatches fail
  closed.

At that stage it did **not** implement a chemistry-aware fallback grammar,
production token IDs, C1-R, C3, or P1 training admission.  The chemistry-aware
GraphPorts fallback, strict graph round-trip and PF-1 release binding have
since closed the relevant PF-1 gates; C1-R and C3 remain outside this screen.

The production-side boundary is now separate from that synthetic fixture.
`experiment_grid.py` freezes the A0/A1/M0/M1 two-factor configuration and one
shared model-facing contract: standard T5 receives only `input_ids`,
`attention_mask`, and `labels`; geometry-enabled cells receive an additional
explicit atom-E3FP-to-carrier sidecar through the shared wrapper.  A1 uses one
atom per atom/SELFIES carrier, while M1 maps all atoms of a logical motif to
the same carrier so the wrapper can apply one invariant scatter mean.

`production_bridge.py` consumes the existing validated
`p1-logical-motif-training-record/vnext1` document without re-tokenizing or
inferring any mapping.  Its runtime special IDs and vocabulary bounds are
hash-bound to the tokenizer contract/snapshot named by each record.  It
supports M0 and M1, preserves arbitrary explicit
connection-token index sets during whole-motif corruption, and guarantees that
M0/M1 share the exact same CE batch while only M1 exposes E3FP.  It does not
reinterpret the historical synthetic fallback as chemistry-aware.  The
production geometry/topology -> frozen union tokenizer -> vNext record gate is
now closed by the paired release builder.  A logical-motif record is rejected
rather than silently reinterpreted as an atom baseline.

`atom_production_bridge.py` closes the corresponding **runtime contract** for
A0/A1.  Its immutable
`ProductionAtomSelfiesRecord` binds the raw SELFIES audit string and token IDs
to the same frozen union-tokenizer contract/snapshot used by the comparison,
declares one complete identity span and one unique carrier token per atom, and
stores E3FP as an explicit `[atom, level]` array.  Whole atom-identity spans use
the same stateless `(seed, epoch, record_id, objective, identity-unit-id)`
selection rule as motif CE.  SELFIES branch, ring, boundary and other declared
structure roles remain visible and byte-for-token unchanged.  A0 and A1 emit
the exact same padded CE batch; only A1 exposes the geometry sidecar, with one
active atom per carrier.

This atom carrier design is the narrow baseline motivated by 3D-MolT5's
SELFIES/E3FP alignment, but the implementation is deliberately more explicit:
it never infers atom alignment from token position, and A1 uses the same
`SharedE3FPCarrierFusion` module and parameters as M1.  The production
topology -> SELFIES -> frozen union-tokenizer producer and its
text/ID/span/carrier mapping are now exercised by the run3 release and full
collator gate.  That admission remains limited to PF-1 rather than the final
full pretraining corpus.

The common geometry row axis is not inferred from a generic "heavy atom"
heuristic.  The frozen PCQM geometry policy
`project_explicit_hydrogens_before_e3fp_v1` tags source atoms before
`RemoveHs`, rejects residual hydrogen/non-E3FP atoms, and stores the surviving
ordered `model_to_source_atom_index`.  Both production bridges now copy that
mapping beside the same post-projection E3FP rows.  The A0/A1 producer must
prove that SELFIES atom-carrier order follows those model rows; absence or
reordering of the mapping fails closed.  `validate_a1_m1_geometry_atom_parity`
then compares record order, active rows, source mapping and exact E3FP values
while deliberately allowing the A1 and M1 carrier-token mappings to differ.

For full-corpus use, the strict JSON gate runs once when a Dataset row becomes
an immutable `ProductionMotifRecord`.  The hot collator path reuses that row
and computes only the stateless `(seed, epoch, member_id, objective)` mask;
sampled audit replay can still compare it with the mask embedded in a vNext
document.  Contract serialization and recursive validation are therefore not
repeated for every epoch or mini-batch.

The synthetic CE-first collator adds one deliberately narrow training-side
skeleton: it samples in the logical-motif domain using a stateless
`(seed, epoch, record_id, objective)` key, replaces every selected identity
span as a whole with one sentinel, and emits ordinary Python `input_ids` and
T5-style decoder `labels`.  Connection spans and E3FP remain explicitly
visible.  State prediction and C3 are rejected by construction.

`runtime_bridge.py` makes the synthetic audit layer executable: it materializes
an uncorrupted JSON audit view bound to the exact epoch mask, and right-pads
any structurally compatible synthetic or production CE row into only
`input_ids`, `attention_mask`, and `labels` for the standard T5 CE forward
allowlist.  It remains pure Python and does not itself admit production
training.

`training_adapter.py` is the optional training boundary.  It lazily converts a
`PaddedCEBatch` to a Transformers `BatchEncoding` of three `torch.long`
tensors.  `record_ids`, contract documents and artifact hashes remain outside
the model call.  `run_t5_ce_smoke.py` is a thin, offline-only executable probe:
given an explicit local T5 snapshot, it runs forward/backward, one AdamW step,
then saves and reloads the model.  Report schema v2 separates checkpoint
serialization from backend numerics: functional config hashes and all state
tensors must match exactly, the CPU functional probe must be within its
reported tolerance, and reloaded runtime logits must remain finite.  CUDA
same-instance and cross-reload differences are retained as diagnostics rather
than being mistaken for weight corruption.  Its built-in two-record batch
tests CE plumbing only; it is not chemical evidence or P1 admission.

`shared_geometry_fusion.py` is the single geometry path used by both A1 and
M1: each of the four ordered E3FP shell levels has its own embedding table;
the four level states are summed into one atom state, followed by an invariant
scatter mean over atoms that share one explicit carrier token.  The resulting
carrier state is added to the ordinary T5 input embeddings.  A1 and M1 differ
only in their `e3fp_atom_to_token` mapping and can share the exact
same module parameters.  Tokens without mapped atoms receive zero geometry.
The module has no condition-specific branch, teacher, regression loss, gate,
or concatenation; A0 and M0 bypass it and retain the ordinary T5 path.

`four_grid_t5_wrapper.py` is the corresponding Hugging Face Trainer boundary.
Every A0/A1/M0/M1 instance contains the same base-T5 and
`SharedE3FPCarrierFusion` state-dict schema.  A0/M0 reject geometry side inputs
and return the untouched standard `input_ids`/`labels` T5 CE forward.  A1/M1
require all three E3FP side tensors, add the common fusion result to the base
input embeddings, and pass `inputs_embeds` plus the same `labels` to that T5.
The output object is returned unchanged; there is no wrapper loss term.
`condition_id` is fixed when the instance is created and any batch-carried
condition tag is checked against it.  For distributed A0/M0 training, their
deliberately present but unused fusion table means Trainer/DDP must allow
unused parameters; it must not be connected through a zero-valued surrogate
loss merely to satisfy DDP.
