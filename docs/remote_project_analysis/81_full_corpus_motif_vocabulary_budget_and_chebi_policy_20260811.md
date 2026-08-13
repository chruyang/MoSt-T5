# Full-corpus motif vocabulary budget and ChEBI policy (2026-08-11)

> Status: Phase-I/Phase-II/ChEBI CPU evidence is complete under the historical
> traversal-sensitive motif identity.  The 18,000-row general registry and its
> 20,325-row all-ChEBI-train variant are pre-canonicalization candidates only.
> Document 82 now blocks final admission until graph-canonical motif identities
> are rebuilt and replayed over the same corpora.

## 1. Paper-first literature boundary

Parameters in this decision are taken from paper text or appendices first.
Official code is used only to verify mechanics where the paper is silent; a CLI
default is not treated as a reported model parameter.

| Work | Formal paper evidence | Relevance to MoSt-T5 |
|---|---|---|
| CAMT5 (Findings EMNLP 2025) | Appendix reports **24,735 motif tokens** for ChEBI-20 and PCDes. The model treats motifs as individual T5 tokens. | Closest precedent for a generative motif-language T5 and direct support for a tens-of-thousands candidate. |
| FineMolTex (KDD 2025 / arXiv 2409.14106) | Paper reports **30,080 unique motifs**. Its selective masking set is a separate 2,457-motif subset after frequency filtering; those two quantities must not be conflated. | Supports a large identity vocabulary, but not the claim that every identity should be equally masked or supervised. |
| HierVAE (ICML 2020) | Paper reports vocabularies below 500; its table lists 436/478/462/307/307 for the evaluated datasets. | Shows that the field does not have a universal 10k rule; graph generation and T5 token generation have different trade-offs. |
| fragSMILES (Communications Chemistry 2025) | Paper compares fragment vocabularies of 5,869 and 13,035 and discusses the sequence-length/type-count trade-off. | Supports selecting a Pareto knee, not maximizing vocabulary size. |
| Group SELFIES (Digital Discovery 2023) | Demonstrates a small generic group set and compositional fallback rather than prescribing a huge fixed dictionary. | Supports retaining an open, lossless compositional path. |
| t-SMILES (Nature Communications 2024) | Encodes fragments compositionally as SMILES with tree/link syntax rather than requiring one opaque token per fragment. | Supports the chemical-lexer long tail and argues against treating macro coverage as representability. |

Primary sources:

- CAMT5: https://aclanthology.org/2025.findings-emnlp.1221.pdf
- FineMolTex: https://arxiv.org/pdf/2409.14106
- HierVAE: https://people.csail.mit.edu/tommi/papers/JBJ_ICML2020a.pdf
- fragSMILES: https://www.nature.com/articles/s42004-025-01423-3
- Group SELFIES: https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00012e
- t-SMILES: https://www.nature.com/articles/s41467-024-49388-6

The literature therefore supports **testing** a 10k-scale vocabulary for our
generative setting, but does not make that scale a domain-wide rule.

## 2. Exact full-corpus experiment

The analysis used all 3,360,067 final-v4 permitted PCQM members and the
published 136-shard production release.  It read the already-audited per-record
motif-digest sequence directly from read-only LMDB headers.  It did not rescan
SDF, recompute E3FP, or copy the 33 GB release.  Thirty-two CPU workers produced
a compact motif-ID/offset cache and completed in 102.13 seconds.

The full corpus contains:

- 24,152,754 pure-motif occurrences;
- 214,353 observed stereo-free pure-motif identities;
- lossless representability of 100% by macro plus chemical-lexer fallback.

### 2.1 Missing Phase-II evidence and corrected corpus boundary

The previous phrase "full corpus" meant the full **Phase-I PCQM support**, not
all pretraining data.  Phase II additionally uses 301,655 PubChem train
molecules paired with text.  Their molecular side is part of pretraining and
must influence the vocabulary before any tokenizer is frozen.  Text content
does not influence motif selection.

The stored Phase-II LMDB also contains a legacy `motif_seq`, but that sequence
was produced under an older anchor/stereochemistry convention.  Combining its
114,736 legacy motif types with current Phase-I counts would mix two identity
domains.  The corrected procedure is:

1. verify the registered Phase-II train LMDB before its trusted pickle boundary;
2. read only `(CID, smiles)` from each of its 301,655 payloads;
3. run the same current frozen molecule linearizer and stereo-free anchored
   projection used for Phase I and registered downstream data;
4. discard molecule-local anchor labels before counting pure motifs;
5. publish a compact motif-ID/offset cache and current-surface registry;
6. jointly replay candidate registries over the complete Phase-I and Phase-II
   molecule populations.

This scan does not recompute E3FP, process descriptions, or modify the source.
Its implementation is
`most_t5_next/p1/build_phase2_anchored_pure_motif_census_v1.py`.

The completed current-surface scan admitted all **301,655/301,655** Phase-II
train molecules with zero rejects in 100.06 seconds using 16 CPU workers.  It
found **4,626,550** motif occurrences and **55,291** unique pure motifs.  Of
these, 8,308 also occur in Phase I and **46,983 are Phase-II-only**.  The source
LMDB SHA-256 is
`465d89f4aafb36043a5964441feffceb3e3e6493fe2ffee9d53190ec7587d5e5`,
and the published current-surface registry SHA-256 is
`c37871fa420ad22e67d0a49c872ea1813b57d1c76cc9e8b1deb35b55604a0d85`.
No stored legacy motif sequence, text field, or E3FP payload participated in
the count.

`Fully macro-tokenized` below is stricter: every motif in the molecule must be
one registered whole-motif macro.

| K | occurrence coverage | fully macro molecules | mean fallback motifs/molecule | selected minimum frequency | identity length P50/P95 | untied T5 vocab params |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 95.5854% | 68.7951% | 0.3173 | 740 | 25 / 39 | 0.79M |
| 2,048 | 97.2459% | 80.3234% | 0.1980 | 104 | 22 / 37 | 3.15M |
| 4,096 | 97.7893% | 84.1661% | 0.1589 | 42 | 22 / 37 | 6.29M |
| 8,192 | 98.2501% | 87.4465% | 0.1258 | 18 | 22 / 37 | 12.58M |
| 12,000 | 98.4797% | 89.0870% | 0.1093 | 12 | 22 / 37 | 18.43M |
| **16,000** | **98.6425%** | **90.2523%** | **0.0976** | **8** | **22 / 37** | **24.58M** |
| 24,735 | 98.8729% | 91.9052% | 0.0810 | 5 | 20 / 36 | 37.99M |
| 30,080 | 98.9718% | 92.6134% | 0.0739 | 4 | 19 / 36 | 46.20M |
| 32,768 | 99.0163% | 92.9328% | 0.0707 | 4 | 19 / 36 | 50.33M |

All untied costs include both the 768-dimensional input row and independent LM
head row, hence 1,536 parameters per macro.

## 3. Joint Phase-I + Phase-II base registry

The general vocabulary is selected before training from both pretraining train
populations.  Three rankings were replayed without changing K:

- Phase-I-only frequency: historical control;
- raw pooled frequency: ranks by `count_P1 + count_P2` and therefore inherits
  the much larger Phase-I occurrence mass;
- equal-stage mass: ranks by
  `count_P1/24,152,754 + count_P2/4,626,550`.

At K=16,000, equal-stage mass is the only policy that substantially improves
Phase-II coverage without using labels or an assumed Phase-I/Phase-II training
schedule:

| ranking at K=16,000 | Phase-I full / occurrence | Phase-II full / occurrence | ChEBI-train occurrence |
|---|---:|---:|---:|
| Phase-I only | 90.25% / 98.64% | 36.43% / 95.47% | 97.19% |
| raw pooled | 89.58% / 98.55% | 67.96% / 97.86% | 99.05% |
| **equal-stage mass** | **87.47% / 98.25%** | **77.81% / 98.53%** | **99.37%** |

The equal-stage-mass Pareto sweep is:

| K | Phase-I full / occurrence | Phase-II full / occurrence | ChEBI-train occurrence | selected min / median exposure | untied macro-row params |
|---:|---:|---:|---:|---:|---:|
| 8,000 | 84.63% / 97.85% | 69.08% / 97.94% | 99.07% | 9 / 52 | 12.29M |
| 12,000 | 86.18% / 98.07% | 74.49% / 98.31% | 99.26% | 6 / 32 | 18.43M |
| 16,000 | 87.47% / 98.25% | 77.81% / 98.53% | 99.37% | 4 / 23 | 24.58M |
| **18,000** | **87.59% / 98.27%** | **80.15% / 98.69%** | **99.44%** | **4 / 20** | **27.65M** |
| 20,000 | 88.42% / 98.39% | 80.59% / 98.72% | 99.46% | 4 / 18 | 30.72M |
| 24,735 | 88.74% / 98.43% | 84.50% / 98.98% | 99.56% | 3 / 14 | 37.99M |

K=18,000 is the provisional general base.  Relative to K=16,000 it costs
3.072M untied parameters and gains +2.34 Phase-II whole-molecule coverage
points, while Phase-I changes by only +0.12 points.  K=20,000 adds another
3.072M parameters for only +0.44 Phase-II and +0.83 Phase-I points.  The old
Phase-I-only argument that K=16,000 has a minimum frequency of eight is not a
valid joint-corpus boundary: equal-stage ranking intentionally promotes
Phase-II-common motifs and has minimum combined exposure four at both 16k and
18k.

This selects candidates, not model quality.  K=16,000 remains the smaller
ablation and K=18,000 must still be compared under the same training-token
budget before final tokenizer admission.

> Sections 4--7 retain the earlier Phase-I-only analysis as provenance.  Their
> PCQM-top-16k totals are superseded for the mainline by the joint-corpus
> decision in Sections 3 and 8.

## 4. ChEBI-20 train-aware comparison

Only the 26,407-record ChEBI-20 **train** split participated.  Validation and
test were excluded.  It contains 492,227 motif occurrences and 5,774 unique
pure motifs; 3,721 identities are absent from PCQM pretraining.

At the same fixed K=16,000:

| Registry source | PCQM fully macro | PCQM occurrence | ChEBI fully macro | ChEBI occurrence |
|---|---:|---:|---:|---:|
| PCQM pretraining frequency only | 90.2523% | 98.6425% | 58.5034% | 97.1887% |
| equal corpus-mass PCQM + ChEBI-train | 89.0085% | 98.4687% | 100.0000% | 100.0000% |

The task-aware registry contains 12,279 PCQM-seen identities and all 3,721
ChEBI-only identities.  Relative to the general registry it trades 1.24 points
of PCQM whole-molecule coverage for 41.50 points of ChEBI whole-molecule
coverage without increasing K or model parameters.

This is a **task-aware specialist**, not a zero-shot vocabulary.  Every
ChEBI-only row must receive real Phase-I training exposure before evaluation;
pre-registering cold output rows is not sufficient.  No token may be added
after Phase I begins.

## 5. ChEBI extension and frequency sufficiency

The proposed `PCQM-top-16k union all ChEBI-20-train motifs` registry contains
20,382 macros, not 21,774: 1,392 of ChEBI's 5,774 types are already in the
PCQM top 16,000.  It adds 4,382 rows and 6,730,752 parameters to the untied
T5 vocabulary, for 31,306,752 macro-row parameters in total.

The added rows are extremely long-tailed.  Of 4,382 additions, 2,863 occur
once in ChEBI train, 3,550 occur at most twice, and only 198 occur at least
eight times.  Among the 3,721 types absent from PCQM altogether, 2,470 are
singletons.  Therefore the full union guarantees lexical coverage but does
not, by itself, provide enough positive evidence to estimate every independent
randomly initialized input and untied output row.

| Registry | macros | ChEBI fully macro | ChEBI occurrence | ChEBI <=1 fallback | PCQM fully macro |
|---|---:|---:|---:|---:|---:|
| PCQM top 16k | 16,000 | 58.5034% | 97.1887% | 93.7592% | 90.2523% |
| + ChEBI count >= 8 | 16,198 | 75.6087% | 98.6145% | 98.7238% | 90.2565% |
| + ChEBI count >= 5 | 16,385 | 79.3237% | 98.8349% | 99.0684% | 90.2617% |
| + ChEBI count >= 2 | 17,519 | 89.3589% | 99.4184% | 99.8182% | 90.2789% |
| + all ChEBI train | 20,382 | 100.0000% | 100.0000% | 100.0000% | 90.3147% |

The count-at-least-two registry is the efficient independent-row baseline.
The full 20,382 registry remains a serious specialist candidate only if Phase
I initializes both input and untied output rows from their lossless lexer
expansions and trains macro/fallback equivalence; random cold rows plus uniform
sampling are not an adequate scientific implementation.

## 6. Cross-task vocabulary replay

All registered task populations below are evaluation-only for this analysis;
only ChEBI-20 train changes a candidate registry.  The table reports strict
whole-molecule macro coverage (`full`) and motif occurrence coverage (`occ`).

| Dataset population | molecules | top16k full / occ | +count>=2 full / occ | +all ChEBI full / occ |
|---|---:|---:|---:|---:|
| PubChem computed/descriptive/caption shared set | 14,936 | 48.37 / 95.75 | 72.92 / 98.08 | 81.94 / 98.71 |
| USPTO-50K products | 49,594 | 71.33 / 97.17 | 74.54 / 97.54 | 75.37 / 97.63 |
| USPTO-50K reactant mixtures | 49,631 | 65.90 / 96.90 | 73.30 / 97.73 | 74.32 / 97.83 |
| MoleculeSTM DrugBank retrieval | 2,370 | 48.52 / 94.39 | 68.44 / 97.18 | 74.14 / 97.69 |
| QM9 | 128,836 | 42.25 / 82.98 | 43.32 / 83.51 | 46.52 / 85.73 |
| BACE | 1,513 | 28.22 / 93.73 | 55.45 / 96.90 | 55.65 / 96.92 |
| BBBP | 1,975 | 45.01 / 93.49 | 60.46 / 95.71 | 64.25 / 96.14 |
| ClinTox | 1,459 | 26.80 / 91.58 | 65.18 / 96.79 | 72.45 / 97.51 |
| HIV | 41,126 accepted / 1 rejected | 45.45 / 93.61 | 51.13 / 94.60 | 52.42 / 94.81 |

The ChEBI singleton tail is thus not purely task-local: it materially improves
whole-molecule coverage on PubChem, DrugBank, BBBP and ClinTox.  It is much
less important for USPTO, BACE and HIV, and it does not solve QM9's distinct
long tail.  The chemical lexer remains mandatory under every policy.

These figures are tokenizer coverage, not downstream accuracy.  A full-task
population replay also does not prove that every validation/test macro occurs
in that task's train split; split-conditioned exposure is a later fine-tuning
data contract.

## 7. ChEBI split replay and trained fallback policy

The task-conditioned vocabulary was selected from PCQM plus ChEBI-20 **train**
only.  A separate replay over all 3,301 validation and 3,300 test molecules had
zero representation rejects and produced the following curve.  `N` means the
top-N ChEBI-train additions outside the fixed PCQM top 16,000, ranked by train
frequency; validation and test never influence selection.

| added N | total macros | val occurrence | val fully macro | test occurrence | test fully macro |
|---:|---:|---:|---:|---:|---:|
| 0 | 16,000 | 97.080% | 59.285% | 97.177% | 58.606% |
| 32 | 16,032 | 98.170% | 69.131% | 98.127% | 68.606% |
| 64 | 16,064 | 98.295% | 70.797% | 98.244% | 70.424% |
| 128 | 16,128 | 98.481% | 73.644% | 98.397% | 72.667% |
| 256 | 16,256 | 98.645% | 76.250% | 98.594% | 75.515% |
| 512 | 16,512 | 98.864% | 79.855% | 98.785% | 78.697% |
| **1,024** | **17,024** | **99.074%** | **83.429%** | **98.955%** | **81.576%** |
| **2,048** | **18,048** | **99.234%** | **86.216%** | **99.124%** | **84.515%** |
| 4,096 | 20,096 | 99.423% | 89.609% | 99.377% | 88.939% |
| all 4,382 | 20,382 | 99.446% | 89.942% | 99.403% | 89.394% |

At 17,024 macros, at most one fallback motif already covers 99.576% of
validation and 99.485% of test molecules.  Expanding from 17,024 to 20,382
adds 3,358 low-frequency rows for only +0.372/+0.448 occurrence points and
+6.513/+7.818 fully-macro points on validation/test.  More importantly,
validation contains 380 motif types (388 occurrences) and test contains 419
types (436 occurrences) that never occur in ChEBI train.  Consequently no
train-only finite registry can eliminate fallback, and a full train registry
would leave the model with almost no natural fallback exposure during
fine-tuning.

The fallback is therefore promoted from an emergency branch to a **trained
second lexical surface**:

1. a registered motif has one canonical macro-token surface;
2. an unregistered motif is represented losslessly by chemical-lexer tokens
   followed by the single `<MOST:FALLBACK_END>` suffix;
3. the full lexer span is one indivisible logical motif for corruption and
   masking, and the suffix is its motif carrier;
4. anchor order, endpoint ownership, atom-to-motif mapping and atom E3FP rows
   remain attached to the same logical motif, so fallback does not discard or
   duplicate geometry;
5. Phase I includes an auxiliary stochastic decomposition view in which some
   registered macros are deliberately expanded to their lexer surface before
   denoising.  This is an additive or paired view: it may not remove the only
   canonical exposure of a low-frequency macro.  The auxiliary view trains
   both encoder and decoder composition; the normal text-to-molecule task
   keeps one deterministic canonical target;
6. decomposition sampling is frequency-aware, but preserves a minimum amount
   of canonical macro exposure.  Very low-frequency motifs are better left
   unregistered and always composed than represented by cold independent
   rows.  Exact sampling rates remain a small ablation rather than a frozen
   constant;
7. macro input and untied output rows may be initialized from pooled lexer-row
   representations, with a trainable residual, but this remains an ablation and
   is not assumed beneficial without evidence.

This design is literature-adjacent rather than copied verbatim.  Group SELFIES
places fragment/group symbols on top of a robust atomic language; SMILES Pair
Encoding keeps frequent substructures while retaining atom-level
decomposability; t-SMILES encodes fragments compositionally rather than
requiring dictionary IDs; and BPE-Dropout provides the training precedent for
exposing alternate fine-grained segmentations instead of reserving them only
for inference.  None of those papers proves the exact anchored single-suffix
grammar, so macro decomposition must be tested as its own mechanism.

The evaluation contract must stratify text-to-molecule results by zero, one,
and at-least-two fallback motifs, and separately report motifs that were seen
as macros, seen through auxiliary fallback decomposition, and absent from
ChEBI train.  Exact match, molecular validity and fingerprint similarities
must all be reported; aggregate scores alone can hide a long-tail failure.

## 8. Frozen candidate policy before the 10% experiment

The CPU evidence now closes the missing Phase-II question.  The vocabulary
policy for the next experiment is:

1. build the **general base** with equal-stage-mass ranking over Phase-I and
   Phase-II train only;
2. use **K=18,000** as the main candidate and K=16,000 as the smaller budget
   control;
3. add **all 2,325 ChEBI-train identities** outside the selected base, giving
   20,325 macro rows in the default task-aware tokenizer;
4. retain 18,256 (+256) and 18,512 (+512) only as compact ablations;
5. never use ChEBI validation/test, downstream labels, or Phase-II text to
   select or rank motif rows;
6. never append tokens after Phase I begins; the same tokenizer is used by
   both staged and from-scratch-joint training arms;
7. keep the lossless chemical lexer and train its decomposition surface under
   every finite macro budget.

The ChEBI split replay for the equal-stage K=18,000 base is:

| ChEBI-train additions | total macros | train full / occurrence | validation full / occurrence | test full / occurrence |
|---:|---:|---:|---:|---:|
| 0 | 18,000 | 89.73% / 99.44% | 90.49% / 99.47% | 89.88% / 99.43% |
| 256 | 18,256 | 91.88% / 99.56% | 91.31% / 99.52% | 90.33% / 99.46% |
| 512 | 18,512 | 93.21% / 99.63% | 91.79% / 99.55% | 90.82% / 99.49% |
| **all 2,325** | **20,325** | **100.00% / 100.00%** | **93.52% / 99.64%** | **92.91% / 99.61%** |

The all-train policy is intentionally ChEBI-prioritized.  Of the 2,325 added
rows, 1,963 are ChEBI-train singletons, every row has train count below eight,
and 317 are absent from both general pretraining stages.  The full extension
therefore costs 3.5712M untied parameters beyond K=18,000 for +2.42 validation
and +3.03 test whole-molecule coverage points over +256.  Coverage alone would
favour a smaller extension, but the project policy prioritizes making every
ChEBI-train motif a first-class output symbol before pretraining starts.

The 20,325 candidate has no post-hoc token addition: its complete token
set is frozen before Phase I and its ChEBI-aware rows must receive explicit
macro/decomposition exposure during pretraining.  The 317 rows absent from both
general pretraining corpora cannot be treated as trained merely because they
are registered.  Lexer-derived input/output initialization, stochastic
macro/fallback equivalence views, and ChEBI-train-only auxiliary exposure are
therefore admission requirements; random cold rows are not acceptable.

The final decision is deliberately limited.  CPU coverage supports registry
construction and budget, but does not prove downstream generation quality.
Before full pretraining, the planned 10% equal-token comparison must keep
initialization, data, vocabulary and total training tokens fixed while
comparing Phase-I-then-Phase-II training against joint multitask training.  A
smaller K=16k tokenizer comparison is needed only if the 18k parameter cost or
optimization is measurably harmful; no further full-corpus census is required.

Implementations:

- Phase-II current-surface census:
  `most_t5_next/p1/build_phase2_anchored_pure_motif_census_v1.py`;
- joint ranking and exposure audit:
  `most_t5_next/p1/analyze_multistage_anchored_vocab_v1.py`;
- split-isolated ChEBI replay:
  `most_t5_next/p1/analyze_multistage_chebi_split_coverage_v1.py`.
- frozen candidate registry builder:
  `most_t5_next/p1/build_multistage_anchored_macro_registry_v1.py`.

The resulting 20,325-row candidate registry is
`tmp/multistage_anchored_macro_registry_k18_all_chebi_v1_b0693bb/`
and its canonical JSONL SHA-256 is
`cf21aa58ac6e80d327e18b4896361867e5b0a83bef7c297be2bb58c6ad20f88b`.

### 8.1 Other registered downstream populations

The current 18,000 base and 20,325 all-ChEBI tokenizer were replayed over every
locally registered downstream population.  Other datasets never select rows in
this analysis.

| population | accepted | 18k full / occurrence | 20,325 full / occurrence | 20,325 <=1 fallback |
|---|---:|---:|---:|---:|
| PubChem computed/descriptive/caption | 14,936 | 82.45% / 98.82% | 89.27% / 99.28% | 99.67% |
| USPTO-50K products | 49,594 | 71.70% / 97.25% | 72.07% / 97.29% | 98.96% |
| USPTO-50K reactant mixtures | 49,631 | 70.45% / 97.47% | 71.26% / 97.55% | 98.30% |
| MoleculeSTM DrugBank retrieval | 2,370 | 73.38% / 97.82% | 76.96% / 98.11% | 98.99% |
| QM9 | 128,845 | 42.31% / 84.55% | 43.65% / 84.96% | 99.70% |
| MoleculeNet BACE | 1,513 | 52.15% / 96.68% | 52.48% / 96.70% | 98.94% |
| MoleculeNet BBBP | 2,039 | 61.55% / 95.85% | 64.05% / 96.12% | 99.61% |
| MoleculeNet ClinTox | 1,478 | 73.14% / 97.61% | 80.72% / 98.30% | 99.46% |
| MoleculeNet HIV | 41,126 accepted / 1 rejected | 51.69% / 94.67% | 52.24% / 94.80% | 97.43% |

This does not justify unioning every downstream train motif into the main
vocabulary.  PubChem is already part of Phase II.  QM9, MoleculeNet and
retrieval consume molecules primarily on the input side; their remaining long
tail is losslessly handled by the chemical lexer, and nearly all molecules
have at most one fallback motif.  ChEBI is different because text-to-molecule
generation must emit motif identities.  USPTO-50K is also molecular-output
generation, but it is optional in the current plan and the ChEBI extension
barely changes its coverage.  USPTO may receive its own train-only extension
only if retrosynthesis is promoted to a required task before tokenizer freeze;
it does not silently expand the current mainline.

The full replay is implemented by
`most_t5_next/p1/analyze_multistage_downstream_motif_coverage_v1.py` and recorded
in `tmp/multistage_downstream_motif_coverage_k18_all_chebi_v1_b0693bb.json`.

### 8.2 Historical planned comparison (superseded by the completed replay)

The PCQM-only frequency grid is closed as a Phase-I diagnostic, but the final
registry comparison is not.  It now proceeds in two nested decisions.

First, construct the general pretraining base from **Phase I + Phase II train**
and compare at identical K:

1. `phase1-only-frequency`: historical control, not a final candidate;
2. `phase1-phase2-raw-pooled-frequency`: reflects raw occurrence mass and is
   expected to be dominated by the much larger Phase-I corpus;
3. `phase1-phase2-equal-stage-mass`: ranks by
   `count_P1/total_P1 + count_P2/total_P2` and is the schedule-neutral default
   candidate while the staged-versus-joint mixer remains under comparison.

Both pooled policies must be reported.  Equal-stage mass is not accepted merely
by definition: it must retain adequate coverage on each full corpus and avoid
promoting unsupported rare rows.  The selected registry is fixed before the
10% staged-versus-joint experiment so that schedule and vocabulary are not
confounded.

Second, starting from the selected Phase-I+Phase-II base, compare task-aware
ChEBI-20 **train-only** additions:

1. no ChEBI-specific additions (general registry);
2. top 1,024 ChEBI-train motifs absent from the selected base;
3. top 2,048 ChEBI-train motifs absent from the selected base;
4. all ChEBI-train additions only as a specialist upper bound.

The resulting totals are no longer assumed to be exactly 17,024 or 18,048:
some ChEBI motifs may already enter through Phase II.  The decisive fallback
ablation remains necessary at one fixed final registry.  Every analysis must
report Phase-I, Phase-II and ChEBI coverage separately; validation/test never
select tokens, and no token may be appended after Phase I starts.

The joint analysis implementation is
`most_t5_next/p1/analyze_multistage_anchored_vocab_v1.py`.  This historical
plan has now been executed; its former 1,024/2,048-addition candidates are
superseded by the measured 18,000 base and 20,325 all-ChEBI candidate above.

## 9. Evidence artifacts

- `tmp/anchored_full_vocab_finalv4_v1_report.json`
- `tmp/anchored_full_vocab_finalv4_v1_registry.jsonl`
- `tmp/anchored_chebi20_task_aware_vocab_v1.json`
- `tmp/phase2_anchored_pure_motif_census_v1_b0693bb_manifest.json`
- `tmp/multistage_anchored_vocab_analysis_v1_b0693bb.json`
- `tmp/multistage_anchored_vocab_analysis_k16_fine_v1_b0693bb.json`
- `tmp/multistage_anchored_vocab_pareto_v1_b0693bb.json`
- `tmp/multistage_chebi_split_coverage_v1_b0693bb.json`
- `tmp/multistage_anchored_vocab_k18_fine_v1_b0693bb.json`
- `tmp/multistage_chebi_split_coverage_k18_v1_b0693bb.json`
- `tmp/multistage_anchored_vocab_k18_all_chebi_v1_b0693bb.json`
- `tmp/multistage_chebi_split_k18_all_v1_b0693bb.json`
- `tmp/multistage_downstream_motif_coverage_k18_all_chebi_v1_b0693bb.json`
- `tmp/multistage_anchored_macro_registry_k18_all_chebi_v1_b0693bb/manifest.json`
- `tmp/anchored_task_aware_full_cache_v1.json`
- `tmp/anchored_chebi20_thresholds_v1.json`
- `tmp/anchored_task_aware_full_cache_thresholds_v1.json`
- `tmp/downstream_motif_coverage_v4.json`
- builder: `most_t5_next/p1/analyze_full_anchored_vocab_v1.py`
- ChEBI analysis: `most_t5_next/p1/analyze_chebi20_task_aware_vocab_v1.py`
- reverse replay: `most_t5_next/p1/analyze_task_aware_registry_on_full_cache_v1.py`
- downstream replay: `most_t5_next/p1/analyze_downstream_motif_coverage_v1.py`
