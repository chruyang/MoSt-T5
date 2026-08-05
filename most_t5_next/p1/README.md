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

The synthetic CE-first collator adds one deliberately narrow training-side
skeleton: it samples in the logical-motif domain using a stateless
`(seed, epoch, record_id, objective)` key, replaces every selected identity
span as a whole with one sentinel, and emits ordinary Python `input_ids` and
T5-style decoder `labels`.  Connection spans and E3FP remain explicitly
visible.  State prediction and C3 are rejected by construction.

`runtime_bridge.py` now makes the layer boundary executable: it materializes
an uncorrupted JSON audit view bound to the exact epoch mask, and right-pads
only `input_ids`, `attention_mask`, and `labels` for the standard T5 CE forward
allowlist.  It remains pure Python and does not admit production training.

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
