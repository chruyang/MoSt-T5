# fragSMILES adoption and preflight plan (2026-08-11)

> Status: **candidate mainline, not training-admitted**.  The project will
> evaluate official fragSMILES as the 2D molecular language instead of
> continuing to expand a project-specific GraphPorts/anchor string grammar.
> T5, E3FP, atom-to-fragment geometry and the open-vocabulary fallback remain.

## 1. Decision and motivation

Motif construction is not intended to be the main research contribution of
MoSt-T5.  The preferred direction is therefore:

```text
official fragSMILES reduction + canonical traversal
                         |
                         +-- fragment/connector atom sidecar --> E3FP rows
                         |
                         +-- registered fragment macros
                              or lossless chemical-lexer fallback --> T5
```

fragSMILES already supplies the pieces for which the project had been building
custom syntax: canonical fragment SMILES, explicit connector atom indices,
branches and a canonical traversal of the reduced fragment graph.  The paper
reports shorter sequences than SELFIES on ZINC-250K and the representation has
also been used in a reaction Transformer.  These are adjacent implementation
precedents, not proof that it is optimal for MoSt-T5.

Primary sources:

- paper: https://www.nature.com/articles/s42004-025-01423-3
- official implementation: https://github.com/f48r1/chemicalgof
- archived release: https://zenodo.org/records/14616839
- reaction application: https://pubs.rsc.org/en/content/articlepdf/2025/cc/d5cc02641e

## 2. What changes and what does not

| Layer | Decision |
|---|---|
| source SDF, frozen split and E3FP rows | retain |
| motif partition | replace by official fragSMILES exocyclic-single-bond reduction |
| old ordered anchors | replace by connector-local atom indices in the surface |
| atom-to-motif and anchor-to-atom metadata | migrate to fragment and connector sidecar |
| custom GraphPorts/anchored model language | retire if the preflight gates pass |
| T5 backbone and E3FP geometry branch | retain |
| macro vocabulary | rebuild from canonical fragSMILES fragment identities |
| unseen fragment handling | retain a lossless chemical lexer; never raw `<unk>` |
| corruption, padding and training cache | rebuild after the representation freezes |

Thus the raw geometry release does not need to be recomputed.  The derived 2D
surface, vocabulary, paired records, tensor cache and tokenizer/model rows do.

### 2.1 Anchor migration

The old anchor object mixed two roles: molecule reconstruction and the address
of the attachment atom whose E3FP was consumed by the model.  Under fragSMILES:

- reconstruction uses the connector syntax and graph traversal;
- a sidecar binds every serialized fragment occurrence to its source atom rows;
- every connector direction binds its local atom index to the corresponding
  source attachment atom;
- motifs with multiple attachment atoms remain distinguishable because each
  connector carries a fragment-local atom index when the fragment has multiple
  possible linker atoms.

The anchor information is therefore not discarded; it is normalized into the
representation's native connector plus a geometry sidecar.

## 3. Why raw official strings cannot be adopted blindly

The local official source computes `mapsFrag2Mol` and `mapsMol2Frag` internally,
but public `encode()` returns only a string.  MoSt-T5 needs a thin adapter that
exports these maps and binds them back to fragment occurrences in the canonical
serialized sequence.  This is an integration layer, not a new molecular
language.

The official tests also document unresolved graph and chirality cases.  More
importantly, the default representation does not provide a complete strict
isomeric round trip for all E/Z/directional-bond cases.  This matters even if
E3FP carries conformational state: constitutional/tetrahedral/double-bond
identity policy must be explicit rather than lost accidentally.

### 3.1 Frozen identity/state decision

Discrete R/S and E/Z are retained as **molecular identity**.  E3FP remains a
separate atom-aligned **3D state** channel.  The project will no longer test a
mainline in which E3FP is expected to reconstruct missing discrete stereo.

The reasons are structural rather than benchmark-dependent:

- stereoisomers share a connectivity graph but are distinct molecular objects;
- folded E3FP IDs are many-to-one features and are not an invertible stereo
  codec;
- conformer availability and conformer choice must not determine whether the
  text representation can reconstruct molecular identity;
- molecule generation requires an explicit output for stereo, not merely a
  latent state from which stereo might sometimes be inferred.

This is also the most consistent interpretation of adjacent models.  3D-MolT5
adds E3FP to a SELFIES identity sequence rather than deleting identity from that
sequence.  fragSMILES explicitly represents R/S, while its published
ZINC/ChEMBL preprocessing removed geometric double-bond stereo; that removal is
a dataset policy and not evidence that E/Z is semantically dispensable.

References:

- 3D-MolT5: https://proceedings.iclr.cc/paper_files/paper/2025/file/3d4dc72d715bd6415d356293079adf3d-Paper-Conference.pdf
- fragSMILES stereo notation and preprocessing: https://pmc.ncbi.nlm.nih.gov/articles/PMC11779804/

### 3.2 Candidate local serialization

The connectivity fragment remains a fixed local phrase.  Stereo is appended as
fixed-arity local records, so it neither changes macro identity nor forces a
second fragment vocabulary:

```text
fragment-identity
<ST:A> local-atom R|S
<ST:B> bond-endpoints ordered-supports E|Z
```

- `<ST:A>` identifies a tetrahedral center by canonical fragment-local atom;
- `<ST:B>` owns an internal double bond and stores the ordered support atoms
  used to interpret E/Z;
- a support outside the owning fragment is addressed through the canonical
  connector occurrence;
- an implicit stereo-defining hydrogen uses a dedicated `H` support value and
  never receives an E3FP row;
- record type fixes arity, so no closing stereo boundary is required;
- macro and chemical-lexer fallback fragments use the same stereo records.

Internally the codec must retain ordered support references, not only the E/Z
letter.  Reordering a substituent can otherwise invert the interpretation of an
unchanged E/Z enum.  The public model surface may expose the chemically readable
R/S/E/Z label after this structural binding is fixed.

The remaining comparison is therefore **placement**, not **retention**:

1. local stereo records immediately after the owning fragment phrase;
2. a molecule-level stereo block using the same canonical fragment-local
   addresses;
3. an auxiliary stereo decoder head.

The first is the default candidate because it preserves local phrase alignment,
uses ordinary T5 cross-entropy, works identically for macro and fallback motifs,
and keeps exact generation in one sequence.  The other two are ablations.

### 3.3 Compact stereo codec preflight

The first executable compact codec is
`most_t5_next/p1/fragsmiles_compact_stereo_codec_v1.py`.  It keeps the owning
fragment implicit by placing records immediately after that fragment token and
uses a shared byte-address domain:

```text
<ST:A:R|S:local-parity> <ST:N:local-atom>
<ST:B:E|Z:CIS|TRANS>   <ST:N:local-double-bond> <ST:N:support-selector>
```

The support selector packs two indices into the deterministically ordered
neighbour candidates.  It therefore preserves the structural CIS/TRANS
relation required by RDKit while still exposing the readable E/Z identity.
Full molecule atom indices and verbose support-atom tuples do not enter the T5
surface.  A tetrahedral source tag that has no R/S CIP identity and disappears
from canonical isomeric SMILES is not promoted to a molecular identity token.

Directed R/S, E/Z, explicit stereo-defining hydrogen, disconnected-component,
atom-renumbering and symmetric-cage tests pass.  The codec first normalizes the
source through canonical isomeric SMILES.  Its usual path restores the emitted
local parities directly; only a rare ambiguous symmetric case enumerates local
parity assignments.  Repeated decode/re-encode then enters a deterministic
surface cycle, and the lexicographically minimum *cycle member* is selected.
Pre-cycle transient representations are deliberately excluded, because they
cannot be reproduced when encoding the decoded molecule again.

The final strict pass over 10,000 parseable QM9 molecules is 10,000/10,000,
with zero codec rejects, 8,528 tetrahedral records and 409 E/Z records.  The
complete compact sidecar adds 1.8283 tokens per molecule on average and at most
12 tokens.  Runtime was 220.66 seconds in the local single-process preflight.
Every admitted row passes molecular stereo identity and exact
decode/re-encode surface fixed-point gates; the result was not obtained by
dropping stereo or relaxing to connectivity-only equality.

This closes the QM9 compact-stereo gate, but the codec remains a preflight
artifact rather than training-admitted until the same support-domain and
maximum-stereocentre census is completed for PCQM, ChEBI and USPTO.

### 3.4 Cross-domain CPU support gate

The streaming audit entry point is
`most_t5_next/p1/audit_fragsmiles_compact_stereo_domains_v1.py`.  It reads
JSONL, Parquet, SDF, or a compressed SDF tar without materializing the corpus;
uses a bounded spawn process pool; and records every fallback by source index.
Statistics are collected in worker-completion order because they are
commutative.  Training order is not generated by this audit.  Each molecule
has a 30-second offline codec budget so that a rare automorphism search cannot
stall the full corpus; a timeout is an explicit lossless-fallback case.

Production-runtime results available so far are:

| Domain | Compact pass | Fallback | Mean stereo tokens on pass |
|---|---:|---:|---:|
| PubChem shared unique set | 14,935 / 14,936 (99.993%) | 1 | 8.881 |
| ChEBI-20 train | 26,312 / 26,407 (99.640%) | 95 | 10.493 |
| USPTO-50k reactant mixtures | 49,626 / 49,631 (99.990%) | 5 | 0.712 |
| USPTO-50k products | 49,590 / 49,594 (99.992%) | 4 | 0.752 |
| PCQM4Mv2 full SDF | 3,376,387 / 3,378,606 (99.934%) | 2,219 | 2.132 |

The PubChem caption, description, and property sources have the same 14,936
unique-SMILES SHA-256 and therefore require only one codec replay.  Its sole
fallback is a hypervalent iodine whose neutral double-bond form is rebuilt as
an equivalent charge-separated form; strict molecular representation was not
relaxed to admit it.

ChEBI fallback is dominated by chemistry outside the normal heavy-atom motif
path: standalone proton/isotopic-hydrogen components, very large symmetric or
highly stereogenic molecules, and two hypervalent-iodine representations.
Sixteen rows hit the 30-second bound and one very large nucleic-acid-like row
exceeded the local byte-address domain.  USPTO fallback is nine rows total,
mostly very large protected or highly stereogenic molecules that hit the same
time bound.  These results support an ordinary macro path plus an open,
lossless chemical-lexer fallback; they do not support enlarging the normal
motif grammar to absorb every exceptional structure.

The full PCQM replay completed over all **3,378,606** source records in
4,238.65 seconds with 28 workers.  The compact path admitted **3,376,387**
records and assigned **2,219 (0.06568%)** to fallback.  The fallback ledger is
not an undifferentiated runtime failure: 1,271 rows failed strict stereo
round-trip, 592 had no fragment-local parity assignment that restored the
source stereo identity, 340 contained a defined double bond without a usable
E/Z CIP label, five failed the compact surface fixed-point check, three had
fragment-local atom-offset drift, four hit an upstream `IndexError`, and four
hit the 30-second bound.  Consequently the corpus-scale result strengthens the
two-path design: the compact grammar is the ordinary path, while these explicit
rows require a lossless chemical-lexer path with the same atom/E3FP addressing
contract.  It does not justify relaxing strict identity or adding exceptional
states to every ordinary fragment.

The first executable universal path is now
`most_t5_next/p1/fragsmiles_lossless_fallback_v1.py`.  Its current v1 wire
format reserves one opaque molecule-fallback prefix plus 256 opaque UTF-8 byte
tokens, stores atom-token spans, and maps every retained heavy atom to its
projected E3FP row.  An
explicit hydrogen that has no E3FP row is represented by `None`, never by a
fabricated state ID.  RDKit directional-ring SMILES can alternate between two
equivalent slash surfaces; the fallback therefore propagates atom-row mappings
through parse boundaries and selects the minimum member of the closed
serialization cycle, matching the compact codec's fixed-point discipline.
All **2,219/2,219** PCQM compact rejects replay through this path with strict
decode/re-encode equality; the largest fallback surface has 102 tokens and 93
explicit-hydrogen atom occurrences have no E3FP row.  This byte path is only
the universal whole-molecule escape hatch.  Ordinary fragments missing from a
future macro registry still use the semantic chemical lexer so that rare
motifs retain compositional chemical units rather than becoming byte strings.

The frozen model-facing simplification for the next revision removes the
redundant `<MOST:FB:MOL>` token.  This is now a design decision rather than an
open candidate.  Every molecular surface, whether compact or fallback, is
bounded by the shared modality delimiters `<bom>` and `<eom>`.  These delimiters
separate molecular language from surrounding natural language; they do not
identify a codec mode.  The fallback mode remains release/sidecar metadata and
is never exposed as a learned molecule-class token.  A connected whole-molecule fallback is exposed
through the ordinary logical-motif ABI as one degenerate motif phrase:
canonical-isomeric-SMILES byte tokens followed by the already existing
`<MOST:FS:FRAG_END>` carrier.  All atoms map to logical motif zero and retain
their byte spans and E3FP rows.  A disconnected reagent/mixture is represented
as one such phrase per connected component separated by the existing component
control, rather than pooling unrelated components into one geometry object.
The sidecar still records `fallback_mode`; that metadata is not a learned token.
The dedicated byte namespace, the phrase suffix, and the outer `<bom>/<eom>`
modality boundary make the removed prefix unnecessary for parsing.  This
unifies batching and carrier geometry without
claiming that the degenerate phrase is a chemically selected fragSMILES motif.

That local path is now executable in
`most_t5_next/p1/fragsmiles_macro_fallback_surface_v1.py`.  Because official
fragSMILES emits a fragment SMILES and its connector `<n>` as separate tokens,
the adapter does not alter the official partition or invent another anchor
grammar.  A registry hit emits one fragment macro; a miss emits the existing
155-token semantic chemical alphabet plus one fragment suffix.  The two added
units are the bracket-only aromatic `si` and `te` spellings actually emitted by
the pinned RDKit producer; they do not broaden the unbracketed organic subset.
Connector
indices use a finite digit stream, branches/components use six fixed opaque
controls, and the 266 compact-stereo tokens remain immediately after their
owning macro or fallback phrase.  Encode/decode canonical replay and the
fragment-to-E3FP carrier map are tested for mixed macro/fallback molecules.

The current executable-v1 non-macro tokenizer addition budget is **684**
rows: 155 semantic chemical tokens, 257 universal molecule-fallback tokens,
six fragSMILES controls and 266 compact-stereo tokens.  This is an exact grammar
budget, not the final tokenizer size; the macro count must still be rebuilt
from the new canonical fragment identities.  Under the prefixless model-facing
revision above, the fallback contribution becomes 256 and the non-macro budget
becomes **683**.

### 3.4 Reuse an established SMILES vocabulary before retaining a new lexer

The preferred tokenizer direction is now to **reuse an established molecular
SMILES lexical vocabulary**, rather than treating the locally defined
155-symbol semantic alphabet as the default final design.  The closest checked
implementation is the atom-wise MegaMolBART tokenizer vendored by FineMolTex:
it uses the established SMILES regular-expression boundary for atoms, bracket
atoms, bonds, branches and ring closures, and its tokenizer can be constructed
from a frozen vocabulary file.  CAMT5 likewise extends a T5 tokenizer from
frozen additional-token assets.  We therefore have implementation precedent
for borrowing the molecular lexical units while retaining one unified T5
embedding table.

"Reuse" here means reusing the documented token strings and lexical boundary
rules, then assigning deterministic IDs inside the frozen unified T5
tokenizer.  It does **not** mean copying an upstream integer-ID layout or
replacing the shared T5 tokenizer with a second independently indexed
tokenizer.  The checked FineMolTex `bart_vocab.txt` is also not suitable for
blind byte-for-byte adoption: its 523 rows include task-property labels and
only 215 corpus-observed whole bracket-atom strings.  Because its regular
expression treats `[... ]` as one token, an unseen valid bracket atom can still
become `?` under that finite snapshot.

Before tokenizer freeze, compare the referenced MegaMolBART/FineMolTex
vocabulary (and, if available, another atomically complete published SMILES
vocabulary) against the current decomposed lexer on the complete Phase-I,
Phase-II and train-only generative downstream molecular targets.  Admit the
borrowed vocabulary as the default only if it proves all of the following:

1. zero molecular `<unk>`/`?` tokens on every admitted training surface;
2. exact token-to-string reconstruction and strict molecule replay;
3. unchanged SentencePiece segmentation for natural-language text outside
   explicit `<bom> ... <eom>` molecular spans;
4. deterministic coverage of isotopes, charges, atom classes, two-digit ring
   closures and every retained reaction symbol;
5. the ordinary stereo-free motif path cannot consume `@`, `/` or `\\` as
   latent identity, while any raw-reaction surface that intentionally retains
   them has an explicit registered path.

If a referenced finite vocabulary misses only valid long-tail bracket atoms,
the minimal repair is a documented compositional bracket-atom fallback inside
the same molecular span, not a return to an opaque whole-molecule class token.
The 155-symbol local lexer remains a comparison and emergency lossless floor
until this full-corpus gate passes; it is no longer presumed to be the final
scientific vocabulary.

#### Frozen result: Smirk 0.3.0 plus two pinned RDKit glyphs

The comparison gate is now complete and supersedes the provisional paragraph
above.  The selected lexical floor is Smirk 0.3.0's 158 non-UNK model glyphs,
in the exact order of its published `vocab_smiles.json`, plus the lower-case
aromatic `si` and `te` spellings emitted by pinned RDKit 2024.03.5.  The final
molecular lexical floor therefore has **160 glyphs**.  The upstream source is
fixed at `BattModels/smirk` commit
`5b8210612cdecb57e1cbc1aaa8cf38a081c1453e`.

Evidence:

- official Smirk 0.3.0 alone round-tripped 104,145 of 104,165 unique
  Phase-I/Phase-II/ChEBI-train fragment identities; all 20 misses were exactly
  aromatic `si` or `te`;
- on those 104,145 upstream-supported identities, the local pinned front end
  produced exactly the same glyph sequence as official Smirk 0.3.0 in every
  case (zero token-boundary drift);
- the pinned 160-glyph implementation round-tripped all **104,165/104,165**
  unique fragment identities with zero unknown glyphs;
- it also round-tripped all 25,496 unique compact fragment identities in the
  322,893-row cached downstream panel;
- it round-tripped all 115,958 unique raw SMILES in the available PubChem,
  DrugBank retrieval and USPTO-50K source caches, including reaction stereo,
  with zero unknown glyphs.

The model uses a dual front end and one embedding table.  Natural language
continues through the unmodified base-T5 SentencePiece tokenizer.  Molecular
text inside `<bom> ... <eom>` is segmented by the pinned Smirk grammar, and its
glyphs map one-to-one to collision-free ordinary T5 extension rows
`<MOST:SMI:000>` through `<MOST:SMI:159>`.  Bare `C`, `N` and digit strings are
not registered globally as `AddedToken` objects, so English text segmentation
cannot be stolen by molecular glyphs.

Macro misses and strict whole-molecule fallback now share these same 160 rows.
The UTF-8 byte namespace (256 rows) and the molecule-fallback class token are
removed from the active design.  Routing mode remains release metadata.  The
ordinary compact motif policy is still stereo-free: `/`, `\\` and `@...`
glyphs are legal in whole-molecule/reaction SMILES but are rejected from normal
motif identity, whose stereo state remains in the explicit sidecar.

With six fragSMILES controls and 266 compact-stereo rows, the executable
fragSMILES-specific non-macro vocabulary budget falls from 684 to **432**
rows.  A tokenizer built from bare T5-v1.1 also adds the two shared modality
boundaries `<bom>` and `<eom>`; these are reported separately rather than
silently charged to either the atom or motif representation.  For the current
18,427-macro candidate this gives a projected total vocabulary of **50,961**.
The former
155-symbol lexer and byte functions remain only in historical experiment files
until their remaining inactive imports are mechanically retired; neither is
part of the frozen fragSMILES tokenizer construction.

The audit used Python 3.8.20, RDKit 2024.03.5, 28 workers and 112 bounded
pending tasks.  The official chemicalgof source requires Python 3.9/3.10 APIs;
the isolated audit copy applies only syntax/API-equivalent compatibility
changes (deferred annotations, stdlib `pairwise`, and dictionary unpacking),
with original and patched hashes recorded.  The full PCQM replay and universal
lossless fallback replay are complete.  The active admission gate is now the
Phase-I/Phase-II/ChEBI-train macro census under the new canonical fragment
identity.

## 4. CPU prototype and preliminary evidence

The independent prototype is
`most_t5_next/p1/audit_fragsmiles_adoption_v1.py`.  It:

1. loads the pinned local official `chemicalgof` source;
2. reproduces the official reduction and canonical surface byte-for-byte;
3. exports serialized fragment occurrence -> projected source atom rows;
4. exports every connector's two local and source attachment atom indices;
5. decodes with the official parser and checks strict and non-isomeric identity;
6. repeats encoding after reversing the input atom numbering;
7. records source hash and runtime versions and always marks the result
   `training_admission=false`.

Directed tests cover a multi-fragment chain, two attachment atoms on one ring,
formal charges and tetrahedral chirality.  They pass 4/4 locally.

An initial audit of 1,000 parseable heavy-atom QM9 records under local RDKit
2025.09.1 and the pinned official source produced:

| Gate | Result |
|---|---:|
| official encode completed | 979/1,000 |
| connectivity round trip | 979/1,000 |
| strict isomeric round trip | 956/1,000 |
| sidecar atom partition | 979/1,000 |
| exact surface after reverse atom numbering | 979/1,000 |
| mean/max fragSMILES structural lexemes | 4.390 / 16 |

The 21 exceptions and 23 additional strict-identity mismatches mean the
unmodified official implementation is **not** production-admissible.  Example
mismatches preserve connectivity but lose directional C=N stereochemistry.
The lexeme count is not yet a final T5 token count: fragment macros and chemical
fallback pieces still need to be applied.

The 44 strict failures were subsequently classified rather than pooled:

- 11 directional-double-bond cases retained a stereo-defining explicit H and
  raised an official `IndexError`;
- 23 cases encoded but lost E/Z identity, predominantly C=N chemistry;
- 8 complex tetrahedral or pseudo-asymmetric ring systems were rejected by the
  official strict chirality decoder;
- 2 fused-ring cases failed inside the official chirality reconstruction.

This supports a connectivity language, but not an unmodified isomeric language.
The prototype now has an explicit `connectivity_only` policy: stereochemistry is
removed before hydrogen projection, while atom centers and double-bond stereo
are extracted into a structured audit sidecar.  Each heavy stereo atom is bound
to `(serialized fragment occurrence, fragment-local atom)`; a stereo-defining
implicit hydrogen is represented as an absent support rather than a fabricated
E3FP row.  The addressed E/Z signature is invariant under a directed reverse
atom-renumbering fixture.  That sidecar is not yet a training target or an
admitted decoder restoration mechanism.

A second pass over 10,000 parseable QM9 records found 16 failures caused solely
by disconnected components.  Encoding every connected component independently
with official fragSMILES and joining components with one reserved `<COMP>` token
closed those cases without changing the internal fragment grammar.  The final
connectivity-only result was:

| Gate | Result |
|---|---:|
| encode/decode connectivity | 10,000/10,000 |
| fragment atom sidecar partition | 10,000/10,000 |
| exact surface after reverse atom numbering | 10,000/10,000 |
| failures | 0 |
| mean/max structural lexemes, including `<COMP>` | 6.492 / 17 |

This is stronger evidence for the partition/connector migration, but it does
not resolve the scientific choice between identity-complete stereo tokens and
stereo-as-state supervision.

Reproduction command:

```powershell
python -m most_t5_next.p1.audit_fragsmiles_adoption_v1 `
  --input-sdf dataset/qm9-standard-v1/gdb9.sdf `
  --chemicalgof-root reference_repos/chemicalgof-master `
  --max-records 1000 `
  --output-report tmp/fragsmiles_adoption_qm9_1000_v1.json
```

For the connectivity-only 10k gate, add
`--stereo-policy connectivity_only --max-records 10000`.

## 5. Admission gates

Before replacing the current representation, all of the following are required:

1. bind an exact official source version and production RDKit runtime;
2. classify and close every encode failure on PCQM, ChEBI, QM9 and USPTO
   support domains without silent molecule substitution;
3. freeze and test the stereo identity/state policy on R/S, E/Z, isotopes,
   charges, radicals, salts and disconnected components;
4. prove fragment and connector sidecars partition and address the persisted
   E3FP atom axis under random atom renumberings;
5. measure final macro-plus-fallback T5 token length, not only fragSMILES
   structural lexemes;
6. rebuild vocabulary counts from Phase I, Phase II and downstream training
   domains under the new canonical fragment identity;
7. obtain zero-UNK lossless fallback and whole-molecule reconstruction;
8. run an equal-token 10% paired comparison before full pretraining.

Until these gates pass, documents 67 and 82 describe the current fallback
implementation and this document describes its candidate replacement.

## 6. Research contribution after adoption

Adopting fragSMILES narrows and strengthens the novelty claim.  The contribution
is not a new fragment string syntax.  It becomes:

> a T5 molecular language using an established canonical fragment grammar,
> an open-vocabulary macro/fallback tokenizer, and an atom-aligned E3FP sidecar
> that constructs and injects motif-level 3D state while retaining exact
> fragment/connector correspondence.

This remains distinct from fragSMILES alone, CAMT5's representation learning,
3D-MolT5's one-atom-token/one-E3FP-row fusion, and FineMolTex's graph-based
motif objectives.

## 7. Canonical fragment census and final registry build

`most_t5_next/p1/build_fragsmiles_fragment_census_v1.py` freezes the actual
training-surface population rather than reusing the traversal-sensitive
anchored-motif registry.  Every source molecule first receives the same
heavy-atom hydrogen projection used by production geometry.  A strict compact
surface contributes its fragment identities only when the serialized fragment
is already an RDKit stereo-free canonical fixed point.  A strict codec failure
uses the whole-molecule fallback and contributes zero macro observations.
This prevents an unsupported molecule from influencing a macro vocabulary for
a surface it will never use.

The builder emits both a frequency census and a selection-ordered compressed
per-molecule fragment cache.  The latter is necessary to measure whole-molecule
macro coverage and semantic-fallback counts exactly.  It does not cache epoch
corruption or padded tensors.

`most_t5_next/p1/build_fragsmiles_macro_registry_v1.py` then applies the frozen
policy:

1. rank the Phase-I and Phase-II train union by exact
   `count_P1/total_P1 + count_P2/total_P2` stage mass;
2. take the first 18,000 general fragment macros;
3. append every ChEBI-20 train fragment identity absent from that base;
4. use no validation/test record and add no token after pretraining begins;
5. report macro, semantic-fragment-fallback and whole-molecule-fallback
   coverage separately.

The non-macro tokenizer budget remains exactly 684 rows.  The final macro count
and union vocabulary size are intentionally not copied from the historical
20,325-row anchored registry; they will be written only after all three new
censuses pass.

### 7.1 Completed census and frozen macro candidate (2026-08-12)

All three current-surface censuses now pass.  Phase I contains 3,360,067
records, 24,066,857 admitted fragment occurrences and 86,074 unique fragment
identities; 3,359,588 records use the compact surface and 479 use the lossless
whole-molecule fallback.  Phase II contains 301,655 records, 4,608,752
admitted fragment occurrences and 22,571 identities; 301,169 are compact and
486 are whole-molecule fallback.  ChEBI-20 train contains 26,407 records,
486,287 admitted occurrences and 2,729 identities; 26,308 are compact and 99
are whole-molecule fallback.  No census has a hard reject.

The exact equal-stage-mass K=18,000 base plus all ChEBI-20-train identities
absent from that base yields **18,427 fragment macros**: 18,000 general rows
and 427 ChEBI-train extension rows.  Together with the frozen 684 non-macro
rows and the 32,100-row T5 base vocabulary, the projected union vocabulary is
**51,211**.  The macro registry SHA-256 is
`ac56caeab3a7815f2013338f79d8adfba4e6ee76a39c1e7980c3be992eb6a417`.

An exhaustive lexer replay over the union of all three fragment censuses
covered **104,165/104,165 unique identities** after the bounded `si`/`te`
extension.  Before that extension, exactly 20 identities (83 occurrences) were
outside the lexer and every failure was one of those two RDKit aromatic
spellings.  Compact records are now also required to pass the finite lexer
before entering the model surface; a fragment with leaked `/` or `\\` stereo
is routed at record level to the already strict whole-molecule fallback.  This
closed one real PubChem case without accepting stereo inside a pure motif.

Coverage under this candidate is:

| train population | macro occurrence coverage | fully macro among compact | whole-molecule fallback |
|---|---:|---:|---:|
| Phase I | 99.4235% | 95.8732% | 479 |
| Phase II | 99.7995% | 96.9535% | 486 |
| ChEBI-20 train | 99.9856% | 99.7757% | 99 |

Phase I is strongly long-tailed: 53,313 of its 86,074 identities occur once,
11,736 occur twice, and only 1,175 occur more than 100 times.  Nevertheless,
the top 1,000/8,000/18,000 identities by Phase-I frequency alone cover
98.6826%/99.4464%/99.6432% of its 24,066,857 occurrences; 2,009 identities are
enough for 99% occurrence coverage.  The lower 99.4235% Phase-I coverage in the
joint registry is therefore an intentional cost of equal-stage ranking, which
reserves capacity for Phase-II chemistry rather than a failure of the 18k
budget.  Phase II is also long-tailed (9,898 singleton identities of 22,571),
but its own top 18k cover 99.9008% of occurrences.

Of the 427 ChEBI extension rows, 380 occur once in ChEBI train, 424 occur at
most twice, and 110 are absent from both pretraining-stage molecule corpora.
They remain ordinary macro rows despite that sparse support; no separate rare-
macro treatment is part of the research plan.  This artifact freezes the macro
identity list, but remains `training_admission=false` until the union tokenizer
snapshot and the normal training schedule are published.

### 7.2 Downstream coverage and USPTO-50K train-only extension (2026-08-12)

A single 30-worker census was run over 322,894 unique source SMILES pooled from
ChEBI-20, the three PubChem task releases, QM9, USPTO-50K reaction components,
BACE, BBBP, ClinTox, HIV and MoleculeSTM DrugBank retrieval.  It completed in
844.6 seconds.  The initial run contained 322,663 compact records, 230 lossless
whole-molecule fallback records and one strict reject: a charged Fe/S HIV
complex for which official chemicalgof raised `TypeError` while iterating an
internal `None`.  RDKit's SMILES parse/reserialize boundary also changes its
coordination-graph matching behavior, so it remains a reported reject rather
than being admitted through the byte fallback.  A second PubChem record whose
fragment retained `/\\` directional stereo is routed to whole-molecule
fallback by the new finite-lexer gate.  This is a record-level support-domain
route, not a relaxed chemical comparison; the updated full-corpus replay is
the final count gate.

The retained slash is not evidence that the source-level stereo projection was
skipped.  The real source is a large conjugated fused-ring system.  Stereo is
removed before fragSMILES reduction, but the upstream fragment pipeline then
canonicalizes a dummy-bearing fragment, replaces its dummies by hydrogen and
serializes it again.  At that later graph boundary RDKit can reassign a valid
double-bond support relation and emit directional `/` or `\\` notation for the
isolated fragment.  In other words, the notation is regenerated after the
earlier removal step because fragmentation changes the local support/ranking
context.  Blind string deletion would be chemically unsafe.  The present
whole-molecule fallback is therefore the correct fail-closed behavior.  A
future compact fix may structurally clear stereo only after the final dummy-to-
hydrogen fragment transformation and serialize that *identity* surface with
`isomericSmiles=False`, while retaining all source stereo in the sidecar; it
must pass atom-address and strict sidecar reconstruction tests before replacing
the fallback.

The lexer-gated replay completed in 842.6 seconds with 322,893 admitted rows
and the same single Fe/S reject.  It proved the new PubChem route, but it also
showed why worker wall-clock timeout is not a chemical class: under equivalent
30-worker load, 68 rows reached the 30-second timeout versus 73 in the earlier
run.  These rows are conservatively lossless whole-molecule fallback in both
runs, so model correctness is unaffected, but their exact compact/fallback
count is load-dependent.  The final report therefore binds the chemistry
reject and lexer-domain result, while treating timeout rows as an engineering
throughput census rather than evidence about fragSMILES support.  Importantly,
the USPTO frequency-at-least-two registry is byte-identical across the rerun
(`f87bd1e0...2774774`), so this variance does not alter the vocabulary
candidate.

The production Python 3.8 runtime uses a semantic-equivalent chemicalgof
compatibility mirror: `from __future__ import annotations` is prepended to the
upstream modules that evaluate PEP-604 annotations at import time, and one
Python-3.9 dictionary union is rewritten as the equivalent `{**left,
**right}` expression.  No fragmentation, traversal or decoding algorithm is
changed.  This boundary must be published as a tiny patch against the pinned
upstream source rather than as an untracked alternative implementation.

Under the 18,427-row Phase-I/II-plus-ChEBI registry, the principal held-out
coverage is:

| population | macro occurrence coverage | fully macro overall |
|---|---:|---:|
| ChEBI-20 validation / test | 99.9052% / 99.9165% | 97.879% / 98.394% |
| PubChem validation / test | 99.7538% / 99.7868% | 96.888% / 96.781% |
| QM9 validation / test | 89.9315% / 89.9315% | 62.585% / 62.585% |
| USPTO product validation / test | 99.4462% / 99.5040% | 94.132% / 94.743% |
| USPTO reactant validation / test | 99.4526% / 99.5148% | 95.732% / 96.215% |

The corresponding train-split values are ChEBI 99.9856%/99.402%, PubChem
99.7818%/96.762%, QM9 89.5890%/61.117%, USPTO product
99.4681%/94.360%, and USPTO reactant 99.4765%/95.645% (macro occurrence /
fully macro overall).  The first pooled-source pass also produced lexical
stress diagnostics for BACE, BBBP, ClinTox and HIV.  Those values are **not**
the formal MoleculeNet result and are superseded by the already frozen
downstream protocol: BACE/BBBP/ClinTox use the official KPGT
`scaffold-0/1/2` memberships, while HIV uses the project-frozen
DeepChem-compatible deterministic Murcko 8:1:1 derived split.  KPGT does not
publish HIV.  The pooled-source rows remain useful only for detecting
unsupported chemistry.  MoleculeSTM DrugBank retrieval remains a whole-source
diagnostic because it does not participate in macro selection.

All 56,217 frozen MoleculeNet membership rows mapped to the pooled census by
exact source SMILES (zero canonical-match ambiguity).  Under the 18,427-row
base registry, formal split coverage is:

| task / formal split | macro occurrence coverage | fully macro overall |
|---|---:|---:|
| KPGT BACE train, three replicas | 98.091%--98.378% | 71.818%--75.785% |
| KPGT BACE validation/test, three replicas | 97.727%--99.233% | 68.874%--88.742% |
| KPGT BBBP train, three replicas | 99.213%--99.260% | 90.926%--91.600% |
| KPGT BBBP validation/test, three replicas | 99.109%--99.482% | 90.148%--95.122% |
| KPGT ClinTox train, three replicas | 99.416%--99.498% | 93.486%--94.416% |
| KPGT ClinTox validation/test, three replicas | 99.226%--99.676% | 90.476%--95.918% |
| HIV derived train | 97.929% | 79.873% |
| HIV derived validation/test | 94.430%--94.936% | 57.671%--58.352% |

The lower HIV whole-record macro rate reflects its broader organometallic and
unusual chemistry; it does not justify fitting HIV validation/test tokens.
HIV is an input-only classification task and remains handled by the semantic
lexer or whole-molecule fallback.  The USPTO extension changes these
MoleculeNet numbers only modestly, so no MoleculeNet-specific macro rows are
added.

A corresponding-train support audit reaches the same conclusion.  Each KPGT
BACE held-out split has only 17--47 missing macro occurrences in validation
and 19--36 in test; BBBP has 10--16 and 10--13; ClinTox has 6--14 and 8--13.
Some are seen in the corresponding scaffold-replica train set and some are
true held-out-only motifs, but the absolute counts are too small to justify a
shared-vocabulary expansion.  HIV is qualitatively different: validation has
1,826 missing occurrences, 80.2% unseen in train, and test has 1,741, 88.9%
unseen in train.  This is an open-set stress case for the deterministic lexer,
not evidence that HIV held-out chemistry should be fitted into the pretraining
macro registry.

The three PubChem releases use exactly the same molecular split, so their
chemical coverage is identical.  The released QM9 instruction validation and
test parquet files are byte-identical, and 1,895 of their 1,919 unique SMILES
also occur in train.  QM9 remains useful for vocabulary support measurement,
but these files are not independent generalization evidence.

USPTO additions are selected from train products and reactants only.  There
are **1,375** train identities absent from the current registry.  The default
task-priority policy now admits all 1,375, giving **19,802 macros**, a projected
union vocabulary of **52,586**, and 2,112,000 additional untied 768-wide
vocabulary parameters relative to the 18,427-macro candidate.  The candidate
registry SHA-256 is
`6ed41a48a9bcf9e77790e125ec46304d7a01d31e62445a77e4eb8bc6f8db4fca`.
This supersedes the previous frequency-at-least-two default; the 1,045-row
registry remains a useful size ablation, not the default tokenizer.

This decision matches the official fragSMILES reaction implementation more
closely: its vocabulary is constructed from every chemical word observed in
the training set and does not apply a minimum-frequency cutoff.  That upstream
implementation does **not** solve rare-token undertraining: it supplies a
generic `<unk>` lookup rather than an open compositional vocabulary.  The
project now treats the quality of already registered rare rows as a reported
vocabulary limitation, not as a separate model objective.

Of the 1,375 USPTO additions, 330 occur once in USPTO train and 671 have zero
direct occurrence in Phase I or Phase II.  All 1,375 also have an exact finite
chemical-lexer phrase, but a registered row continues to use the ordinary macro
surface and ordinary sequence objective.  No macro/lexer dual-view objective,
compositional row initialization, persistent embedding regularizer, frequency-
dependent branch or dynamic macro composition belongs to the mainline.  The
deterministic lexer remains mandatory only for identities that are not in the
registry.  Upstream fragSMILES traversal/string augmentation can remain ordinary
data augmentation, but is not a rare-token remedy or admission requirement.

At reaction level, fully macro-tokenized coverage changes as follows:

| USPTO extensions | train | validation | test |
|---:|---:|---:|---:|
| 0 | 93.4488% | 93.1614% | 93.8486% |
| 512 | 97.9979% | 96.6607% | 97.0641% |
| 1,024 | 99.3001% | 97.5405% | 97.6633% |
| frequency >= 2 (1,045) | 99.3551% | 97.5605% | 97.7432% |
| all 1,375 | 99.9900% | 97.7205% | 97.8430% |

The 330 singletons buy only 0.1600 and 0.0998 percentage points of full reaction
coverage on validation and test and add 506,880 untied vocabulary parameters.
They are admitted because USPTO-50K is now a default, motif-dependent downstream
task, not because those marginal held-out percentages establish a generally
optimal vocabulary.  The complete registry is not conditional on a dedicated
rare-token training mechanism.  All 1,375 train additions still leave about
2.2% of held-out reactions with at least one unseen macro, so the deterministic
lexer path cannot be removed.

The downstream inclusion rule is task-direction aware.  ChEBI-20 and
USPTO-50K generate molecular strings, so their train-side target motifs may
justify vocabulary rows.  PubChem caption/description/property prediction,
QM9 and MoleculeNet consume molecules but do not generate them; their motif
coverage is an efficiency diagnostic, not a reason to fit additional tokens to
their validation or test molecules.  Retrieval likewise does not justify
task-specific output tokens.  This distinction prevents every downstream
benchmark from continually enlarging the shared pretraining vocabulary.

Mol-Instructions reaction tasks are the next train-only vocabulary census.
They are generative tasks in the official 3D-MolT5 suite, so every missing
motif on a molecular **target** side is eligible for the same all-train-token
policy.  Input-only motifs do not by themselves justify a macro row.  Reagent,
forward-reaction and retrosynthesis are separate official task releases and must
retain their official train rows independently: they are **not** sample-
deduplicated against USPTO-50K or against one another.  Their train-target motif
candidate sets and coverage reports are likewise computed separately.  The
3D-MolT5 repository actually fine-tunes the three reaction families through the
combined `e3fp-mol-instructions-react-all` dataset and evaluates the three task
releases separately; this combined sampling still does not imply deleting a row
that overlaps another source.  If one shared tokenizer is ultimately frozen, an
identical canonical motif necessarily occupies one vocabulary row in the set
union; that storage-level identity merge is not dataset deduplication and does
not remove any training example.  Official validation/test records never
participate.  Phase II in this project is the PubChem molecular-text stage and
is not a substitute for this Mol-Instructions reaction census.

The complete train-target census now closes this item.  Official SELFIES targets
were decoded under SELFIES 2.2.0 and the three task releases were processed
independently with 32 workers; no train row was removed for overlap and no
validation/test row was observed.  The comparison baseline is the 19,802-row
registry containing the 18,427 Phase-I/II+ChEBI rows and all 1,375 USPTO-50K
train additions:

| Mol-Instructions train target | rows | compact / whole fallback | hard reject | unique motifs | new identities vs 19,802 | new occurrence share | compact targets fully covered |
|---|---:|---:|---:|---:|---:|---:|---:|
| reagent | 124,384 | 116,362 / 8,022 | 0 | 633 | 111 | 0.2818% | 97.3196% |
| forward reaction product | 124,384 | 124,367 / 17 | 0 | 3,404 | 1,041 | 0.1511% | 98.4353% |
| retrosynthesis reactants | 128,684 | 128,667 / 17 | 0 | 3,422 | 1,042 | 0.1356% | 98.3757% |

The forward and retro candidate lists strongly overlap (880 new identities),
whereas reagent chemistry is different: its leading new surfaces include
`[LiH]`, `[PdH4]`, high-valence metals and metal oxides.  Across all three
reports there are 1,298 distinct new identities after the unavoidable shared-
registry identity union.  Mol-Instructions does not intrinsically require a
motif tokenizer: an atom-level molecular language can also solve these tasks.
It **does**, however, depend on the chosen molecular output language.  In this
project all three tasks generate fragSMILES molecular targets; leaving a motif
outside the registry is lossless through the open fallback, but removes that
motif's independent learned macro row.  Because reagent prediction, forward
reaction and retrosynthesis remain default downstream tasks and the vocabulary
is frozen before pretraining, every motif observed on their official training-
target sides is admitted before pretraining.  Validation/test motifs remain
excluded from selection.

The resulting set-union appends all 1,298 identities and yields a **21,100**
row macro candidate.  This is tokenizer-storage union only: the three datasets
remain separate and no training row is deduplicated.  No atom-count, glyph-
count or frequency filter is applied.  The 1,226 candidates with at most ten
combined occurrences and 311 combined singletons remain optimization
diagnostics, not vocabulary-admission criteria.  Fallback remains necessary
for truly unseen validation/test chemistry but is not used to delete known
training-target motifs.

Exact-source overlap inspection also found no within-task split overlap for
ChEBI-20 or PubChem.  USPTO does reuse common reagents across splits, which is
expected and is reported separately from product overlap.  MoleculeNet
validation/test members never contribute macro candidates.  Formal coverage
is keyed by each KPGT scaffold replica and the frozen HIV split; the old
pooled-source diagnostic must not be relabeled as train/validation/test
evidence.

### 7.3 Content audit of the 19,802-row candidate

The actual all-USPTO candidate was replayed row by row rather than judged only
by coverage.  All **19,802** ranks, identity hashes and macro surfaces are
unique; all **19,802** identities are connected and exactly accepted by the
finite chemical lexer; and no multi-component macro is present.  The role
counts are 18,000 Phase-I/II base rows, 427 ChEBI-train rows, and 1,375
USPTO-train rows.  The median/p90/p95/p99/max macro sizes are
14/24/29/43/222 atoms and 22/40/50/75/384 lexer tokens.

The audit also identifies a long tail of large identities: 621 base rows are
above 32 atoms, 325 above 64 lexer tokens, 37 above 64 atoms, and 17 above 128
lexer tokens.  These may include cyclic peptides, cages, or cases where the
fragmenter made no useful local cut.  This is useful evidence about phrase
length and batching cost, but it is not sufficient evidence that an identity
is chemically invalid or should lose its learned macro row.  The candidate
rule `32 atoms AND 64 glyphs` is therefore withdrawn as a vocabulary filter.
All 19,802 base/ChEBI/USPTO identities are retained; atom and glyph
distributions remain report-only diagnostics for sequence length, curriculum
and future size ablations.

The same audit found 52 metal/high-valence identities whose text is not a
parse-reserialize fixed point in the current local RDKit runtime (for example,
`O=[CrH2]=O` becomes `[O]=[CrH2]=[O]`).  None collides after normalization and
none comes from the USPTO extension.  This is a runtime/canonical-surface gate,
not evidence of traversal duplicates.  The final registry must be replayed
under the single pinned production RDKit build, normalized or rejected there,
and then hashed; it must not mix spellings produced by different runtimes.

Phase II contains 8,781 identities outside the candidate, but every one is a
singleton and together they account for only 8,781 of 4,608,752 Phase-II motif
occurrences (0.1905%).  They should remain lexer phrases rather than adding
8,781 nearly untrained macro rows.

### 7.4 Downstream comparison suite

The primary comparison suite should reuse every officially released 3D-MolT5
downstream family: PubChemQC computed-property prediction; PubChem computed and
descriptive property prediction; PubChem 3D molecule captioning; QM9 property
prediction; Mol-Instructions reagent, forward-reaction and retrosynthesis
tasks; ChEBI-20 text-guided molecule generation; and USPTO-50K
retrosynthesis.  Reusing the published raw examples, splits and task metrics
reduces preparation work and makes comparison defensible, while each molecule
still has to be translated once into the frozen fragSMILES-plus-geometry
sidecar representation.  MoleculeNet and zero-shot retrieval remain secondary
transfer/mechanism panels rather than replacements for this matched suite.

Using all official tasks does not imply fitting the vocabulary to every task.
Only generative molecular target strings may contribute train-only macro rows;
property, captioning-input and retrieval molecules use the common lexer path.
Exact reaction/source overlap must be reported when Mol-Instructions and
USPTO-50K are both present.

## 8. Current downstream architecture freeze

The E3FP, unified sidecar, staged-training comparison, model-size timing,
reference-aligned tensor-cache/DataLoader policy, and downstream prepared-data
decisions are recorded in
`84_current_architecture_cache_and_downstream_freeze_20260811.md`.  Those are
working implementation constraints rather than evidence that the pending
choices have already won an ablation.

## 9. HF tokenizer single-digit and opaque-macro compatibility gate

The final tokenizer will reuse the base T5 rows for decimal digits ``0``--``9``
and force every molecular numeric field to emit one digit per token, following
the explicit check in the pinned 3D-MolT5 tokenizer.  This policy applies to
connector payloads, SMILES ring/isotope/count/class fields and compact stereo
local indices.  It does **not** apply to natural-language text or to the
characters inside an already recognized macro token name.

Macro identity resolution remains structural:

```text
parsed canonical fragment identity -> frozen registry -> token ID
```

The opaque HF token name is only a serialization/debugging handle.  It is not
used to recognize the chemical fragment and may be renamed without changing
the identity-to-ID contract.  Nevertheless, a tokenizer snapshot still has to
prove that forcing the ten digit rows atomic does not split an opaque name such
as ``<MOST:FM:001237>``.  The final tokenizer freeze therefore requires all of
the following checks over **every** registered macro, not a sample:

1. before the digit operation, ``encode(token_name, add_special_tokens=False)``
   is exactly the declared one ID;
2. after the 3D-MolT5-style
   ``add_tokens(["0",...,"9"], special_tokens=True)`` operation, tokenizer
   length is unchanged and every digit encodes to exactly its existing base-T5
   ID;
3. every macro token name still encodes to exactly its declared one ID after
   digit registration;
4. after ``save_pretrained`` and offline reload, the digit IDs, all macro IDs,
   base vocabulary IDs and sentinel IDs are unchanged;
5. ``ID -> token_name -> ID`` and ``fragment_identity -> ID ->
   fragment_identity`` are bijective;
6. molecule encoding writes registry IDs directly and never performs a global
   regular-expression digit rewrite over a serialized token stream;
7. deliberately adversarial names containing digits, prefixes of other names
   and names adjacent to connector digits remain atomic;
8. the complete training surface contains no ``<unk>`` and a decoded molecular
   sequence reproduces the canonical fragment/connector surface.

Failure of any item blocks tokenizer publication.  The project does not rely
on an undocumented assumption about AddedToken longest-match priority.

The complete 21,100-row candidate has now passed this implementation gate
under Hugging Face Transformers 4.48.3 and SentencePiece 0.2.0: all 21,100
macro names remained atomic before and after digit registration and after an
offline save/reload; base and sentinel IDs remained unchanged; registering
the ten decimal digits added zero rows.  The measured tokenizer size is
**53,368** = 32,100 base-T5 rows + 21,100 macro rows + 168 non-macro molecular
rows.  The same complete registry also passed RDKit 2024.03.5 replay at
21,100/21,100 canonical fixed points with zero collisions.  This remains a
runtime-candidate until checked under the frozen training stack.  That final
promotion has now also passed under Transformers 4.45.2, SentencePiece 0.2.0
and RDKit 2024.03.5: every one of the 21,100 macro IDs, all base-T5 and
sentinel IDs, and the reused decimal IDs were identical, and the tokenizer
snapshot tree hash remained
`fe69e26a5c659049588476ea8241ce162cbaf4f6e55cff68d26285de5d209d76`.
The runtime-promotion manifest is therefore `training_admission=true`; the
53,368-row tokenizer and 21,100-row macro registry are frozen for the next
record/cache gates.

## 10. Remaining gates before formal pretraining

### 10.1 Blocking gates

1. **Freeze the actual macro registry.** Retain all 19,802 existing identities
   without an atom/glyph locality cap and append the set-union of all 1,298
   Mol-Instructions training-target identities, yielding 21,100 macro rows.
   Replay the 52 runtime-sensitive metal/high-valence identities under the
   single pinned production RDKit build, then freeze rank, identity, token ID
   and hashes once before optimization.
2. **Publish the final tokenizer snapshot — complete.** The structural
   identity-to-ID registry, Smirk-derived open fallback without UTF-8 byte
   rows, connector boundaries and base-T5 single digits are frozen at 53,368
   rows.  Full offline reload has passed under Transformers 4.45.2; embedding
   and tied-LM-head parameter consequences must be carried into model-init.
3. **Freeze and materialize one fragSMILES/geometry record contract.** Macro,
   fragment fallback and whole-molecule fallback must expose the same
   atom-to-fragment, carrier, connector endpoint, source atom and E3FP-row
   mapping.  Full encode/decode, atom-renumber and multi-endpoint tests must
   pass before building the training corpus.
4. **Freeze the E3FP input contract.** Decide the PF10-surviving atom reducer
   and level-embedding choice, four 768-wide tables, missing-shell behavior,
   4097 versus 4098 rows, initialization, normalization and carrier/endpoint
   injection.  Every departure from pinned 3D-MolT5 must be attributed to the
   motif interface or an experiment.
5. **Build the deterministic training cache and hot path.** Convert the
   authoritative release once to flat IDs/offsets and geometry arrays; retain
   online corruption and dynamic padding; benchmark the reference-aligned
   multi-worker/pinned-memory/prefetch settings.  The formal runner must support
   exact checkpoint/resume and report resolved loader parameters.
6. **Run the frozen 10% decision experiment.** Compare Phase-I then Phase-II
   against joint-from-initialization with identical tokenizer, initialization,
   members, total training tokens, optimizer/precision and evaluation cadence.
   This freezes the full-corpus schedule; historical S/G mechanism screens do
   not replace it.
7. **Execute a final model/data smoke and resource estimate.** Run all admitted
   objectives through forward/backward, validate sequence/sentinel limits and
   no truncation, measure peak 4090 memory and throughput, enumerate every
   parameter block and checkpoint/storage requirement, and freeze the formal
   launch manifest.

### 10.2 Required but parallelizable after pretraining starts

- Complete the atom/E3FP/fragment/connector bijection gate for each downstream
  dataset before that individual dataset is admitted to fine-tuning.
- Run the formal-corpus L3-present/absent coverage and distribution audit using
  idle CPU resources.
- Materialize downstream caches from the published 3D-MolT5 splits and record
  provenance/duplicate policy.  These tasks do not need to delay the
  pretraining GPU launch once the seven blocking gates above have passed.
