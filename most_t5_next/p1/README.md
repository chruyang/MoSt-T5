# P1 hybrid codec / BoundRecord candidate

This directory is isolated from the historical training tree.  The current
implementation supports only deterministic synthetic fixtures.  It proves the
indexing skeleton needed before a tokenizer-bound release:

- macro and forced-fallback surfaces decode to one logical identity digest;
- attachment slot positions remain in identity while molecule-local edge IDs
  remain in a separate, exactly paired connection schema;
- token (L), logical motif (M), and atom (A) domains are explicit;
- every logical motif has one carrier at the first token of its identity span;
- connection, atom partition, E3FP shape, token table and digest mismatches fail
  closed.

It does **not** implement a chemistry-aware fallback grammar, production token
IDs, C1-R, C3, or P1 training admission.  Full chemical graph round-trip and
release binding remain later gates.

The production-side boundary is now separate from that synthetic fixture.
`experiment_grid.py` freezes the A0/A1/M0/M1 two-factor configuration and one
shared model-facing contract: standard T5 receives only `input_ids`,
`attention_mask`, and `labels`; geometry-enabled cells receive an additional
explicit atom-E3FP-to-carrier sidecar through a future wrapper.  A1 uses one
atom per atom/SELFIES carrier, while M1 maps all atoms of a logical motif to
the same carrier so the wrapper can apply one invariant scatter mean.

`production_bridge.py` consumes the existing validated
`p1-logical-motif-training-record/vnext1` document without re-tokenizing or
inferring any mapping.  Its runtime special IDs and vocabulary bounds are
hash-bound to the tokenizer contract/snapshot named by each record.  It
supports M0 and M1, preserves arbitrary explicit
connection-token index sets during whole-motif corruption, and guarantees that
M0/M1 share the exact same CE batch while only M1 exposes E3FP.  It does not
pretend that the synthetic fallback codec is chemistry-aware.  The remaining
producer gate is production geometry/topology -> frozen hybrid tokenizer ->
that vNext record.  A logical-motif record is rejected rather than silently
reinterpreted as an atom baseline.

`atom_production_bridge.py` closes the corresponding **runtime contract** for
A0/A1 without fabricating the missing chemistry producer.  Its immutable
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
`SharedE3FPCarrierFusion` module and parameters as M1.  The actual
topology -> SELFIES -> frozen union-tokenizer producer is still a required
pre-training gate.  Until it proves the text/ID/span/carrier mapping, these
strict dataclasses and CPU fixtures are interface evidence, not data admission.

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
