# Current architecture, cache, and downstream freeze (2026-08-11)

> Status: **user-approved working freeze, pending implementation and the active
> fragSMILES vocabulary censuses**.  This document records design choices; it
> does not promote choices that have not yet been tested into empirical
> conclusions.

## 1. E3FP input contract

The E3FP atom-state comparison is now reduced to three controlled candidates:

- **reference:** one shared `4097 x 768` table, used by L0--L3, with external
  `-1` shifted to zero padding and a fixed four-slot mean;
- **project candidate:** four separately parameterized `4097 x 768` tables,
  one for each level, with the same padding and fixed four-slot mean.
- **hierarchy candidate:** one `4097 x 768` table for L0 and a second shared
  table for L1--L3, with the same padding and fixed four-slot mean.

The last point applies to ID-domain handling, missing-level representation,
initialization, embedding combination, normalization, dtype, padding, and the
placement of E3FP information on the encoder input path.  These details must be
checked against the pinned official source rather than reconstructed from the
paper alone.  A difference is acceptable when it is required by the
atom-to-fragment architecture, but the implementation must classify it as one
of:

1. required by the motif/sidecar interface;
2. supported by a controlled experiment;
3. an unresolved deviation that remains at the reference default.

The pinned 3D-MolT5 implementation uses the first formulation: it does **not**
use four independent tables.  The four-table formulation is therefore a
project candidate, not inherited precedent.  Its four weights are initialized
by copying the same reference table, making the two candidates numerically
identical at update zero; subsequent difference isolates level-specific
parameterization.  The shared table has 3,146,496 parameters, the four-table
candidate 12,585,984, an increase of 9,439,488.  No level embedding, role
embedding, atom MLP, learned shell weight or E3FP mask row is included in this
comparison.

The reusable implementation is
`most_t5_next/p2/e3fp_atom_embedding_v1.py`.  It now implements all three
parameter-tying arms.  Its tests verify literal 3D-MolT5 lookup/mean
equivalence, exact shared-to-two/four-table initialization, the three
`4097 x 768` parameter inventories, explicit L0 versus higher-shell routing,
zero padding, padded-atom zero output and closed ID bounds.  The V10
adapter/model/init tests additionally prove that the complete atom memory and
all non-E3FP adapter parameters are identical at update zero.  These modules
do not yet choose a winner; every added table must justify its cost in the
frozen 10% comparison.

### 1.1 Admitted hierarchy candidate: L0 identity/context and L1--L3 state

The earlier PF1 and QM9 screens make an L0/high-shell split worth a controlled
test, but do not yet freeze it as the final encoder.  L0 is generated from the
central atom invariants and may be called **atom identity/local 2D context**.
L1--L3 recursively consume the lower identifiers together with bonded
neighbourhood and spatial ordering, so they are **identity-conditioned,
spatially enriched environment states**, not pure geometry.  A paper or
manifest must retain this distinction: an L0 gain is not evidence of 3D use,
and removing the explicit L0 branch does not remove identity from higher
shells.

The minimal hierarchy candidate uses two `4097 x 768` tables rather than a new
MLP:

```text
atom identity stream = E_identity(fp[L0])
atom state stream    = E_state(fp[L1]) + E_state(fp[L2]) + E_state(fp[L3])
combined atom state  = (identity stream + state stream) / 4
```

Missing shells contribute the same fixed zero row used by the reference.  Both
tables are initialized by copying the same pinned 3D-MolT5 table.  Consequently
the two-table candidate is numerically identical to the shared-table
fixed-four mean at update zero; training changes only the parameter-tying
assumption.  It contains 6,292,992 E3FP parameters, midway between the shared
reference (3,146,496) and four-table candidate (12,585,984).  Identity and
state streams remain separately observable through motif pooling and endpoint
addressing, but are fused by the fixed reference-equivalent arithmetic rather
than an unmotivated role embedding, level embedding, gate or atom MLP.

This gives the project a coherent hierarchy:

```text
atom identity/context (L0) + atom environment state (L1--L3)
                         -> motif-owned aggregation
                         -> motif identity phrase + motif state carrier
                         -> optional attachment-local endpoint state
                         -> molecular T5
```

It is an architectural hypothesis, not a semantic disentanglement theorem.
The already frozen rule that molecular atom/glyph **tokens** receive no direct
E3FP remains unchanged; the atom streams live in the geometry sidecar/module,
not as extra text tokens.

The minimum decision experiment compares exactly three parameter-tying arms:

1. one shared table and fixed four-slot mean (3D-MolT5 reference);
2. the two-table L0/state hierarchy above;
3. four level-specific tables as the maximum-capacity project candidate.

All three start from the same numerical function and use the same members,
tokenizer, carrier-only injection during this isolated decision, optimizer,
training-token budget and downstream probe.  Report L0-only, L1--L3-only,
both and zero ablations after training.  Identity denoising alone cannot select
the winner: the decision must include the paired B2D control, a registered
3D-sensitive probe such as QM9/PubChemQC, and aligned/zero/matched-state
diagnostics.  The two-table formulation is admitted only if its F3D gain is not
reproduced by the same B2D parameterization and if disrupting the corresponding
state worsens a 3D-sensitive endpoint.  Otherwise retain the simpler shared
reference.  Run this decision before, rather than factorially together with,
the endpoint-site and staged-versus-joint comparisons.

The executable decision entrypoints are
`most_t5_next/p2/run_e3fp_parameter_tying_screen_v1.py` and
`most_t5_next/p2/launch_e3fp_parameter_tying_screen_v1.py`.  The frozen matrix
is the three arms crossed with B2D/F3D.  Every cell uses 10,000 updates,
effective batch 128 (`microbatch=32`, accumulation=4), BF16, atom memory width
768, eight persistent cache workers and prefetch factor five.  Geometry is
written to motif carriers only during this isolated comparison.  The launcher
can assign one candidate pair to each of three GPUs; B2D then F3D remain
sequential on each GPU.  Its final closure refuses unequal update-zero
evaluations within B2D or F3D.  The existing anchored PF-10 cache can exercise
this compatibility gate, but the final decision must be repeated or confirmed
through the production fragSMILES cache after its dynamic-collator integration;
an anchored-surface result is not silently relabelled as a fragSMILES result.

## 2. Unified fragSMILES and geometry-sidecar interface

Endpoint injection remains an admitted candidate rather than a discarded
idea.  The implementation should now converge on one typed interface shared by
ordinary macro fragments, semantic-lexer fallback fragments, and the rare
whole-molecule lossless fallback.

The interface must expose at least:

- serialized fragment occurrence -> source atom/E3FP rows;
- connector endpoint -> local attachment atom -> source atom/E3FP row;
- fragment carrier positions in the T5 sequence;
- connector/endpoint token positions;
- component, padding, and fallback mode metadata;
- a reversible binding to the authoritative molecular surface.

The initial model can consume fragment-carrier geometry alone.  Endpoint
injection should be wired through the same sidecar and retained as a controlled
ablation or later enabled branch, not implemented as a second incompatible
record format.  Multi-endpoint fragments must retain separate connector-local
addresses.

**Frozen carrier rule (2026-08-13).**  Chemical atom/glyph tokens inside a
locally lexed fragment are identity/grammar tokens only and do not receive a
direct E3FP residual.  Atom-aligned E3FP rows remain explicit in the sidecar,
are encoded on the atom axis, and are aggregated exactly once into the owning
logical fragment carrier: the macro token for a registered fragment or
`<MOST:FS:FRAG_END>` for a locally lexed fragment.  This keeps registered and
fallback fragment occurrences on the same 3D-motif interface and prevents the
same atom state from being injected once at an atom token and again at the
fragment carrier.  Direct atom-token E3FP injection is outside the frozen
mainline rather than an implicit implementation choice.

**Three-example token trace (2026-08-13).**  The frozen rule was exercised on
the remote RDKit 2024.03.5 CPU runtime with deterministic illustrative 3D
conformers and the historical four-level E3FP producer.  The trace covers (i)
a registered single benzene motif, (ii) an unregistered
`C1=CC[SiH2]C=C1` fragment using the Smirk-glyph local fallback, and (iii)
aspirin as a six-fragment connected molecule.  Every pretokenized molecular
surface item mapped to exactly one HF T5 vocabulary row and the model input
ended with the ordinary T5 `</s>` row.  No atom/glyph row received E3FP; the
six atom rows of the local fallback were owned by its single
`<MOST:FS:FRAG_END>` carrier.  Aspirin exposed both implicit endpoints (whose
address currently aliases the owning fragment carrier) and explicit
`<MOST:FS:CONN> digit <MOST:FS:CONN_END>` endpoints, but endpoint injection
remained disabled as planned.  The machine-readable evidence is
`tmp/fragsmiles_t5_three_examples_v1.json`.

This trace uses the current 53,368-row candidate tokenizer snapshot only to
demonstrate exact HF IDs and input layout.  Its manifest records Transformers
4.48.3 while the frozen training runtime is 4.45.2, so it remains a
`candidate_runtime_mismatch` artifact with `training_admission=false`; the
same registry must be republished under the final runtime before training.

### 2.1 Implemented V2 training ABI (2026-08-13)

The first typed projection is now implemented in
`most_t5_next/p2/fragsmiles_geometry_sidecar_v1.py`.  It does not replace or
modify the authoritative fragSMILES, compact-stereo, macro/fallback or
whole-molecule fallback codecs.  It binds their already verified output to one
model-facing record containing:

- fragment identity phrase spans, one carrier per phrase, component ownership,
  and an explicit `macro` / `fragment_lexer` representation mode;
- three separately named atom axes: `source_sdf_atom_index`,
  `projected_atom_index`, and `e3fp_row`; compact records require the producer
  to supply an explicit bijection instead of inferring one axis from another;
- two typed endpoint rows per fragment edge, including local atom, all three
  atom addresses, E3FP row and the current candidate endpoint carrier;
- token roles and token-to-fragment ownership for identity phrases, explicit
  endpoints, component controls and fixed-arity stereo records;
- a plain provenance table recording the source dataset/file, representation
  version, tokenizer version and E3FP producer/version used for the record.

This project does not require recursive content hashes as an admission gate.
The files are controlled research artifacts rather than externally mutable
inputs.  A human-readable correspondence table plus count/shape/address checks
is sufficient once the named files and versions have been confirmed.  Hashes
may appear as incidental transport metadata, but training must not be blocked
on redundant byte-level attestations.

The endpoint rule follows the pinned official fragSMILES grammar.  A connector
`<n>` is omitted when that endpoint fragment has exactly one possible linker.
Therefore an edge does not necessarily serialize two connector records.  V1
freezes the following rule without adding any token:

```text
explicit endpoint:  endpoint carrier = its existing <MOST:FS:CONN_END>
implicit endpoint:  endpoint carrier = its owning fragment carrier
```

Both cases retain the exact attachment atom and E3FP row.  This makes endpoint
injection executable as a later ablation while allowing the initial model to
consume only fragment carriers through the same record.

The whole-molecule fallback follows the already frozen no-`<MOST:FB:MOL>`
decision as `<bom> canonical-SMILES-glyphs <eom>`.  It has zero logical
fragments and zero connectors; `<eom>` is the optional molecule-summary
carrier.  Every lexical atom remains in the atom table, including an explicit
hydrogen that has no geometry row, using `has_e3fp_row=false`.  Component
ownership remains atom metadata and no fallback atom receives motif fields.
Record sidecars are deliberately unpadded; padding and padding masks remain a
collator/cache concern.

Nine focused tests now cover mixed macro/local fallback, explicit and implicit
endpoints, a four-edge branching fragment, disconnected compact and fallback
surfaces, atom renumbering with non-identical source/projected axes, mandatory
axis coverage, lexical hydrogen without a fabricated E3FP row, and fail-closed
endpoint tampering.  The adjacent macro/fallback, lossless fallback and
compact-stereo suites also remain green (30 tests total).

### 2.2 Clarifications after method-level review (2026-08-13)

**Macro versus local fallback.**  They are not two simultaneous decodings of
one occurrence.  Routing is deterministic: a registered fragment is emitted as
one macro token; an unregistered fragment is expanded by the local chemical
lexer.  “Shared logical fragment” only means that both paths expose the same
downstream fields (identity, atom ownership and geometry addresses).  No motif
is duplicated and no consistency loss between two parallel views is required.

**Source-index issue: resolved in V2.**  The compact builder now rejects calls
that do not provide the complete three-axis correspondence and rejects missing,
duplicate or extra E3FP rows.  Atom renumbering tests use deliberately different
source and projected indices, so the old accidental `source == e3fp` shortcut
cannot silently return.

**Whole-molecule fallback revision: implemented in V2.**  Treating the fallback
as a degenerate motif was an interface compromise, not a chemical claim.  The
V2 projection is `<bom> canonical-SMILES-glyphs <eom>` with zero logical
fragments.  Atom spans, components and E3FP addresses remain in the sidecar,
but the record is excluded from motif-level pooling, motif masking and motif
losses.  `<eom>` may serve only as an optional molecule-summary position; it is
not a motif carrier.  `<MOST:FS:FRAG_END>` remains only on local fragment
fallback phrases.

**Hydrogen alignment.**  The reference 3D-MolT5 tokenizer computes E3FP over the
actual atoms present in its input SDF `Mol`; it does not synthesize a learned
geometry row for a hydrogen that is absent from that atom axis.  MoSt-T5 should
follow the same rule.  A hydrogen explicitly present as an atom in the admitted
3D input participates in E3FP and receives a row.  Hydrogen notation that is
only an atom attribute in the molecular string does not create a separate
geometry row.  The final fallback projection must preserve the lexical atom
occurrence either way and distinguish `has_e3fp_row`, rather than silently
dropping it from the atom-address table.

This rule was checked against the released downstream data rather than inferred
from comments alone.  The official `3d_tokenize.py` reads the SDF `Mol` and does
not call `Chem.AddHs`; its `get_num_atoms_wH` name means all atoms already
present in that `Mol`.  A separate `verify.py` comment says E3FP was generated
with `AddHs`, but that claim conflicts with both the production code and the
published tensors.  In 100 released train rows from each of
`e3fp-chebi-molgen`, `e3fp-mol-instructions-qm9`, and
`e3fp-pubchemqc-prop`, all 300 parseable rows had exactly one active E3FP row
per atom in the released SMILES, and zero rows matched the larger non-trivial
`Chem.AddHs` atom count.  The corresponding active/AddHs totals were
2885/5843, 900/1900, and 2152/4152.  Therefore MoSt-T5 must not add implicit
hydrogen geometry rows merely to follow that stale verification comment.

**Endpoint policy.**  The authoritative official fragSMILES surface continues
to omit connector records for unique-linker endpoints.  The model-facing
candidate should nevertheless expose every edge endpoint explicitly through
the sidecar.  Two implementation choices remain:

1. add redundant endpoint tokens to the model surface, while stripping and
   validating them before official fragSMILES decode; or
2. keep official tokens unchanged and materialize a dense endpoint tensor/table
   with two rows per edge.

The second choice is preferred: it makes endpoint use symmetric without adding
tokens, preserves official fragSMILES compatibility, and allows carrier-only
and endpoint-aware models to consume the same cached data.  If “explicit” is
defined as “every endpoint has a concrete training record”, this choice already
satisfies it.  If explicit textual endpoint tokens are required, their sequence
cost must be measured before changing the frozen tokenizer.

### 2.3 Open question: where endpoint geometry enters the T5 sequence

The endpoint-to-attachment-to-E3FP **address contract is frozen**, but its
neural injection site is not.  The available evidence is adjacent rather than
a direct precedent for the current `<MOST:FS:CONN_END>` candidate:

- 3D-MolT5 supports aligning each atom's E3FP state with the corresponding
  atom/SELFIES sequence position, but has no fragment endpoint token;
- fragSMILES and t-SMILES support explicit or inferable attachment locations
  in a fragment string, but do not inject 3D features into those locations;
- HierVAE/HierG2G use a separate attachment layer between atoms and motifs;
- FragGen lets geometric features of frontier/attachment atoms control
  attachment and placement, but does so in a graph/geometric network rather
  than at a language boundary token.

The earlier anchored design should be described precisely: each ordered anchor
occurrence was already a sequence token and the sidecar bound that token to one
attachment atom/E3FP row.  It did not add a second endpoint token, and it was
not a sidecar-only representation.  By contrast, official fragSMILES may omit
the connector occurrence at a unique-linker endpoint.  Making every endpoint a
row in the new sidecar therefore freezes an address, not a T5 representation:
the row carries 3D information only after a neural module reads it and writes
or attends that information somewhere.

Accordingly, assigning an explicit endpoint's E3FP-derived state directly to
its existing `<MOST:FS:CONN_END>` token, and assigning an implicit endpoint to
the owning fragment carrier, is a compact **project-specific candidate**.  It
must not be described as a method inherited from those references.  In
particular, `<MOST:FS:CONN_END>` is a shared syntactic terminator rather than a
chemical entity, and the implicit fallback mixes endpoint state with the whole
fragment summary.

Before formal architecture freeze, compare under the same initialization,
members, token budget and optimizer schedule:

1. fragment-carrier geometry only;
2. fragment carrier plus direct endpoint injection at `<MOST:FS:CONN_END>`
   (implicit endpoints fall back to the fragment carrier);
3. fragment carrier plus a separate endpoint memory derived from attachment
   atoms, queried by the connector span or fragment context;
4. explicit endpoint sequence tokens plus the same sidecar addresses;
5. the selected endpoint formulation with same-identity matched E3FP donor
   shuffling.

Report aligned-versus-shuffled sensitivity, a 3D-sensitive task such as the
registered QM9 probe, explicit/implicit endpoint strata, endpoint-only versus
carrier-only ablations, sequence throughput, peak memory and parameter cost.
The direct boundary-token formulation is admitted only if it is not materially
worse than the separate-memory formulation; otherwise the endpoint sidecar
remains frozen while the model consumes it through separate endpoint memory.
The explicit-token candidate must additionally report corpus-level token-count
increase and examples-per-second before the 10% run; endpoint injection is not
frozen until this comparison is complete.

### 2.4 Evaluation strata for whole-molecule fallback

Removing whole-molecule fallback from the motif domain does not prevent its 3D
representation from being measured.  The evaluation axes are orthogonal and
must be reported separately:

1. **compact 2D motif:** compact fragSMILES records with geometry disabled;
2. **compact 3D motif:** the same records with aligned geometry, compared with
   B2D, zero and matched-shuffle controls;
3. **fallback whole-molecule 2D:** `<bom> SMILES <eom>` fallback records with
   geometry disabled;
4. **fallback whole-molecule 3D:** the same fallback records with atom-level
   E3FP enabled and molecule-level pooling/evaluation, but no motif metric.

An all-record downstream score may also be reported, stratified by compact and
fallback mode.  A fallback record contributes to molecule-level 2D/3D metrics,
never to motif count, motif pooling, motif masking, or “3D motif improvement.”
Because fallback prevalence may be small, its mechanism result should use a
fixed fallback evaluation panel and report its sample count rather than infer
capability from the aggregate score alone.

## 3. Training-stage decision

The Phase-I/Phase-II question already has a frozen 10% comparison and should
not be reopened during vocabulary construction:

- **A:** Phase I followed by Phase II;
- **B:** joint training of all registered objectives from initialization.

The two arms must use the same initialization, frozen tokenizer, training
members, total training tokens, precision, effective batch, optimizer family,
and evaluation schedule.  This is the decision experiment for staged training;
small diagnostic S/G schedules are not substitutes for it.

### 3.1 PubChem/PubChemQC multitask sampling

For the registered PubChem and PubChemQC downstream instruction tasks,
MoSt-T5 will reproduce the sampling policy reported by 3D-MoLM and adopted by
3D-MolT5.  A task with training population `n_t` receives unnormalized weight

```text
w_t = n_t ** 0.25
p_t = w_t / sum_j(w_j)
```

This is **fourth-root-of-dataset-size weighting**, not linear proportional
sampling.  It preserves a size-dependent preference for larger tasks while
preventing PubChemQC or another large task from overwhelming the smaller
PubChem computed-property, descriptive-property and captioning objectives.
The population used in this formula is the frozen number of admitted training
examples for each task after its source-specific validity processing; neither
validation/test rows nor repeated round-robin views increase `n_t`.

The rule applies to task selection in the corresponding downstream multitask
or Generalist training mixture.  It does not alter Phase-I/Phase-II corpus
membership, vocabulary frequency, ChEBI decontamination, effective batch size,
or the equal-token-budget staged-vs-joint experiment.  If a batch is assembled
from several task streams, the implementation must record whether the sampled
task probabilities are realized per example, per microbatch or by a
deterministic batch composition; in every case its long-run task counts must
match the frozen fourth-root probabilities.  Small streams may cycle through
their training members, as in the reference round-robin treatment, but a cycle
does not redefine the underlying task population.

Before formal use, the launcher/manifest must publish for every task:

- admitted train count `n_t`;
- raw fourth-root weight and normalized probability;
- realized example and optimizer-update counts;
- number of completed/repeated passes through the task;
- the deterministic sampler seed and resume cursor.

An exact-resume test must show that interruption does not change task order or
the realized mixture.  Any departure from this reference policy requires a
separate controlled comparison rather than an undocumented convenience
setting.

The reusable sampler is now implemented in
`most_t5_next/p2/fourth_root_task_sampler_v1.py`.  It selects one task per
fixed-size microbatch; therefore the task-draw probability is also the expected
example proportion.  Each task reader consumes its own frozen-order cursor and
cycles without dropping a short tail or redefining `n_t`.  The serialized state
contains RNG state, global draw count, per-task selections/examples, cursor and
completed passes.  Five tests cover exact fourth-root normalization, 50,000-draw
realization, boundary wrap, JSON round-trip resume and corrupt/changed contract
rejection.  DataLoader/cache construction remains outside the sampler.

### 3.2 Open question: batch-level balancing during pretraining

Whether Phase I/Phase II should adopt the **batch-level balancing** used by
3D-MolT5 remains an unresolved training-policy question.  This entry records a
reference implementation to investigate; it does **not** freeze that policy for
MoSt-T5 and must not be described as a method already adopted by this project.

3D-MolT5 reports a pretraining batch size of 768 and states that each batch
evenly includes examples from all pretraining tasks.  Its official source
constructs eight task streams (text denoising, molecule denoising, joint
1D--3D denoising, 3D-to-1D, wrapped/in-context text, molecule-to-text,
text-to-molecule and PubMed text).  A stream that is exhausted is restarted,
so a small source such as the PubChem molecule--text pairs is reused in a
round-robin manner.  In the single-worker path, one item from every stream is
assembled into a tuple and the collator flattens these tuples, giving strict
equal example counts.  In the multi-worker path, workers are assigned to task
streams, so the operational guarantee is better understood as balanced supply
over the worker/output stream rather than necessarily an exact eight-way
composition in every physical microbatch.

This is an established equal-mixing/oversampling practice, not a novel
3D-MolT5 algorithm.  It balances **record counts**, but does not by itself
equalize input tokens, target tokens, loss scales, gradient influence, data
quality or the number of times each unique molecule is reused.  Strict equal
mixing may also overfit small tasks and underexpose large tasks, a limitation
already discussed in the original T5 multitask analysis.  It must therefore be
treated as a credible reference baseline, not copied solely because it appears
in 3D-MolT5.

Before freezing the MoSt-T5 pretraining mixture, compare or analytically rule
out at least the following candidates under the same admitted data,
initialization and total training-token budget:

1. equal record quotas per registered task, following the high-level
   3D-MolT5 policy;
2. temperature- or power-scaled task sampling, including a fourth-root
   diagnostic where appropriate;
3. balancing by effective target-token exposure rather than raw record count;
4. a hybrid policy that caps repeated passes through small paired datasets
   while retaining nonzero coverage of every registered task.

The decision evidence must report, per task, nominal probabilities or quotas,
realized records, input and target tokens, completed/repeated dataset passes,
loss scale and gradient statistics.  The decision should also distinguish
strict composition inside every optimizer update from approximate long-run
balance produced by independent workers.  Exact sampler/cursor resume remains
required for whichever policy is selected.

This open pretraining question is separate from Section 3.1.  The downstream
PubChem/PubChemQC Generalist mixture retains its frozen fourth-root policy; no
conclusion about Phase-I/Phase-II batching is implied by that downstream
choice.

## 4. Model-size decision

The final parameter target is intentionally deferred until the following are
known from actual artifacts:

- final fragment macro count;
- total tokenizer size after fixed grammar/fallback tokens;
- whether the LM head is tied;
- the exact four-table E3FP contract;
- the retained carrier and optional endpoint modules.

The final report must enumerate the T5 backbone, input embeddings, LM head,
four E3FP tables, fragment reducer, endpoint branch, and task heads separately.
No target parameter total should be inferred from the historical anchored or
GraphPorts vocabulary.

## 5. Training cache and worker policy

The training path should directly follow established implementation practice
unless the MoSt-T5 architecture requires a deviation:

- 3D-MolT5 performs deterministic E3FP/SELFIES preparation before training,
  uses Hugging Face dataset mapping with multiple processes, and trains through
  a PyTorch DataLoader;
- CAMT5 trains from prepared datasets/streams with multiple DataLoader workers,
  pinned memory, and explicit prefetching;
- FineMolTex writes deterministic motif/graph objects to processed artifacts
  and uses the PyG DataLoader for training.

Pinned local references:

- `reference_repos/3D-MolT5_official_src_82dbe088/3d_molt5/utils/model_utils.py`;
- `reference_repos/CAMT5_official_src_5875a0a/train/task/continual_pretrain.py`;
- `reference_repos/FineMolTex_official_src_c976faa/scripts/FineMolTex/datasets/PubChemSTM.py`.

MoSt-T5 will therefore retain two layers:

```text
authoritative release (LMDB/JSON/manifest)
    -> deterministic training tensor cache (flat arrays + offsets/shards)
    -> multi-worker DataLoader and bounded prefetch
    -> online random corruption, dynamic padding, pinned-memory transfer
    -> GPU
```

The cache may contain token IDs, token roles, fragment spans, atom-to-fragment
mapping, fragment carriers, connector endpoints, E3FP rows, split membership,
and variable-length offsets.  It must not freeze epoch-dependent corruption,
padding, dropout, or final task views.

Reference parameters and code details are the default starting point.  If
worker count, `pin_memory`, `prefetch_factor`, `persistent_workers`,
`drop_last`, preprocessing process count, or batch assembly differs from a
reference implementation, the change must be attributed to:

1. a demonstrated architectural/data-format requirement;
2. a measured throughput or memory result;
3. a known platform limitation.

When the effect of a detail is uncertain, preserve the closest reference
setting first.  Do not silently omit the parameter and later interpret the
result as an architectural property.  Throughput changes alone do not require
retraining a scientifically equivalent completed condition, but every formal
run must persist its resolved loader/cache settings.

## 6. Downstream tasks and prepared data

The registered downstream portfolio remains unchanged for now.  Dataset-cache
construction, source manifests, split membership, and duplicate policy must be
part of preparation rather than left implicit in individual training scripts.

### Unified downstream source decision (2026-08-12)

All selected downstream tasks that are reproduced from 3D-MolT5 must use the
official `QizhiPei/*` Hugging Face dataset released by 3D-MolT5, pinned to a
full repository commit. A local LMDB, Arrow cache, tensor cache, or transferred
archive is permitted only as a hash-bound transport/cache of that exact
repository revision; it is not an independent source. Similarly named legacy
data, reconstructed splits, or third-party mirrors may not silently replace
the official release.

The frozen source set covers PubChemQC computed properties, PubChem computed
properties, PubChem descriptive properties, PubChem 3D captioning, QM9,
Mol-Instructions reagent/forward/retro/all reaction tasks, ChEBI-20, and the
optional USPTO-50K task. Exact repository IDs and revisions are machine-bound
in
`most_t5_next/r1/overlap/configs/downstream_3dmolt5_hf_source_policy_20260812_v1.json`.

MoleculeNet is the only source-family exception and continues to use the
separately frozen KPGT/MoleculeNet preparation. The provenance caveat remains:
KPGT does not publish HIV, so KPGT must not be cited as HIV's source; the
separately frozen authoritative HIV contract remains in force. Zero-shot
retrieval has no official 3D-MolT5 Hugging Face dataset in the published source
table, so it remains deferred until a separate versioned source/protocol
decision is made. It is not silently sourced from another project.

This source decision does not broaden the Phase-II leakage filter below. The
current filter still protects only the official ChEBI-20 validation and test
sets.

Where 3D-MolT5 publishes a prepared version of the same downstream task, that
release is required as the reproduction source. Its existing records and split
should be consumed directly; the project does not need to redo deduplication or
redefine the task merely to create another local copy.
The inherited preparation and split must still be recorded in the local
manifest.  A new deduplication analysis is required only when MoSt-T5 changes
the source population, combines releases, or constructs a new split.

This policy applies to the current portfolio including PubChem/PubChemQC
property and text tasks, QM9, ChEBI-20, the selected MoleculeNet tasks,
3D-molecule captioning, zero-shot retrieval, and the optional USPTO-50K
retrosynthesis task.  Vocabulary construction may use only registered training
splits under the separately frozen vocabulary policy; validation and test
records remain evaluation-only.

### Phase-II decontamination decision (2026-08-12)

The frozen Phase-II release uses one deliberately narrow rule.  Only the
official ChEBI-20 **test** molecule set is protected; validation does not
participate.  If a Phase-II record has the same canonical **non-stereo
connectivity identity** as a test molecule, the complete Phase-II record is
excluded.  Thus represented R/S or E/Z siblings of the same connectivity are
removed together.  There are no task-specific or objective-specific
exceptions: caption, matching, generation, and structural views of a matched
record are all removed together.

No other downstream dataset participates in this Phase-II exclusion policy at
this point.  Stereo and text hashes remain report-only and do not change the
membership decision.  Phase-I grammar learning is not re-derived under
this rule.  The authoritative Phase-II LMDB is preserved; the operation publishes a
derived permitted-membership list, a complete exclusion ledger, and a policy
receipt.  The single-purpose executable is
`most_t5_next.r1.overlap.derive_phase2_chebi20_clean_membership_v1`; it rejects
missing ChEBI test input and rejects validation or any extra/substituted
protected dataset.

The fresh full derivation from the 301,655-row Phase-II identity collection is
complete.  The 3,300 ChEBI-20 test rows expose 3,237 unique non-stereo
connectivity identities.  Of these, 2,587 connectivities occur in Phase-II and
exclude **5,041** Phase-II records, leaving a frozen permitted membership of
**296,614** records.  The authoritative artifact is
`/root/autodl-tmp/most-t5-r1/derived/p2-clean-membership-chebi20-test-connectivity-final-v1`.
Its manifest SHA-256 is
`bfb46e2cb55710719b7e8e3138aa6e0262c93ca6725c28ab8e85db61fe4eed44`.
The earlier validation+test connectivity derivation that retained 291,889 and
the test-only stereo diagnostic that retained 299,296 are both superseded and
must not be used for training.

### Phase-II E3FP evidence boundary (2026-08-12)

The geometry membership itself is established independently of the ChEBI
filter.  The raw Phase-II partition contains 301,658 CIDs.  Exactly three
single-atom records cannot produce pairwise-distance E3FP geometry and are
recorded as `E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS`; the geometry-ready source is
therefore 301,655 records.  Its exact LMDB bytes are hash locked, and two fresh
processes decoded every payload while preserving the complete source hash.
All 301,655 stored records expose coordinates, atom rows, atom-to-motif mapping
and E3FP fields; subsequent full corpus motif/fragSMILES scans consumed all
301,655 with zero record reject.

This proves **published E3FP availability and consumability**, not a new
301,655-record coordinate-to-E3FP recomputation.  The recent full scans record
`stored_e3fp_recomputed=false`.  Thus the current formal claim is that the
locked, upstream-generated E3FP sidecar is valid for the admitted membership
and atom-address pipeline.  It does not claim that folded E3FP is collision
free, uniquely identifies a conformer, or that every row has been independently
recomputed from its coordinates in the current runtime.  A bounded or full
source-conformer recomputation audit is a stronger provenance check that may be
run separately; it is not silently substituted for the evidence above.

The E3FP reuse and atom-address policy is now frozen in
`85_QM9_e3fp_correspondence_and_downstream_atom_mapping_gate_20260812.md`.
QM9 has passed full atom-axis validation and an exact public-conformer
recomputation audit. After formal pretraining starts, idle CPU resources will
run the complete atom-bijection gate over every downstream split. This does not
block GPU pretraining, but it blocks admission of an individual downstream
dataset until every geometry-consuming model atom has exactly one published
E3FP row and the complete fragment/connector address chain passes. For releases
without the exact source conformer, published E3FP is reused and mapped; a
newly generated RDKit conformer must not be called a reproduction of the
published state.

## 7. Immediate implementation order

The 21,100-row macro registry, 53,368-row tokenizer and ChEBI-test-only
Phase-II connectivity membership are now frozen.  The V2 typed sidecar now
closes the three atom-axis and zero-motif whole-fallback contracts.  The
remaining pretraining-critical work is:

1. Integrate the V2 sidecar into the production/cache builder and full-corpus
   validate macro, local fallback, whole-molecule fallback and multi-endpoint
   records; the builder must supply the explicit source/projected/E3FP table.
2. Resolve the endpoint injection-site comparison in Section 2.3; do not treat
   `<MOST:FS:CONN_END>` injection as a reference-backed default.
3. Integrate the controlled shared-table/two-table-L0-state/four-table
   768-wide E3FP module into the final fragSMILES model.  Resolve this
   parameter-tying comparison first, then carry only its winner into endpoint
   and staged-versus-joint experiments; do not reintroduce historical
   role/MLP/level-embedding variables.
4. Build the deterministic tensor-cache conversion and reproduce reference
   DataLoader settings before tuning worker/prefetch values.
5. Integrate the implemented PubChem/PubChemQC fourth-root sampler with the
   registered downstream readers and persist its report in training manifests.
6. Resolve the open Phase-I/Phase-II batch-mixture question in Section 3.2;
   treat 3D-MolT5 equal task balancing as a reference candidate rather than an
   inherited default.
7. Materialize the frozen 10% staged-vs-joint comparison with the same cache
   and evaluation contract.
8. Freeze the final parameter inventory/model-init contract and run a complete
   forward/backward, capacity, memory and throughput smoke before full-scale
   training.

## 8. Phase-I three-part corpus boundary and the interim PF-10 decision (2026-08-13)

The final Phase-I corpus is logically assembled from three populations, even
though it is backed by only two payload stores:

1. the historical final-v4 main membership: **3,360,067** records;
2. **5,510** historically downstream-filtered records restored for Phase-I;
3. the separately reprocessed stereo-recovery supplement: **12,978** records.

The resulting Phase-I membership is **3,378,555** records.  The 5,510 restored
records already exist in the immutable production-v2 shards and must not be
copied or appended as a second payload segment.  Only the 12,978 stereo rows
are read from the supplement LMDB.  The current source population contains 51
unresolved chemistry/projection rejects, which remain outside the corpus.

This distinction is a mandatory input contract for the eventual fragSMILES
training cache.  Its builder must expose three logical source labels (main,
restored-membership, stereo supplement), use a globally unique record key such
as `(source_kind, ordinal)`, and prove that the restored 5,510 are selected
from production-v2 exactly once.  This work is deliberately deferred to one
dedicated full-data pass so it is not mixed with the atom-encoder experiment.

The immediate 10% experiment is instead an architecture screen over the
already published **336,006-member PF-10** release (302,406 train and 33,600
dev).  A newly materialized anchored training surface/cache may be used only
for this controlled screen.  It does not establish the final fragSMILES
tokenizer, fallback, Phase-I source-union, or full pretraining data contract.
No result from the historical 33,600-member PF1 cache may be relabelled as a
10% result merely by increasing the number of optimizer updates.

The frozen screen crosses three E3FP table-sharing candidates with B2D/F3D:

- one shared table with a fixed four-slot mean (3D-MolT5 reference);
- two tables, with L0 separated from the shared L1-L3 state table;
- four independent level-specific tables.

All six cells use a 768-wide atom memory, carrier-only geometry injection,
identical update-zero outputs within each state kind, an effective batch of
128 (`32 x 4`), 10,000 optimizer updates, and evaluation at updates 0, 2,500,
5,000, 7,500 and 10,000.  Endpoint placement, final fragSMILES fallback
semantics, staged-versus-joint training and the full Phase-I three-part cache
remain outside this factorial comparison.
