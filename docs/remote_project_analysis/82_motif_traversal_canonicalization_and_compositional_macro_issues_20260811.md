# Motif traversal canonicalization and compositional-macro open issues (2026-08-11)

> Status: traversal aliasing is a tokenizer P0.  A graph-canonical prototype
> and molecule-level round-trip gate are implemented locally; the full
> Phase-I/II/ChEBI registries have not yet been rebuilt under the new identity.

## 1. Two distinct open problems

### 1.1 Traversal aliases in the motif vocabulary

The current census historically used the SMILES-like spelling left after
deleting molecule-local `<n*>` anchors.  That spelling is not a graph identity.
For example, `<0*>C=N` and `N=C<0*>` describe the same port-labelled motif but
previously produced `[C=N]` and `[N=C]`-style traversal variants.  Ring start
positions create the same problem on a larger scale.

This is not the macro/fallback dual-surface issue.  It creates multiple macro
rows for one chemical object, splits frequency evidence, wastes input/output
parameters and makes vocabulary size depend on traversal accidents.

A graph-only diagnostic over the existing 512-row registry found 350 candidate
port-graph keys and 104 multi-row alias classes (266 rows in those classes;
162 candidate redundant rows).  Because the old registry does not retain every
occurrence-level port mapping in the row itself, 162 is an upper-bound
diagnostic rather than an authorized deletion count.

### 1.2 Rare ChEBI macros and compositional initialization/exposure

The all-ChEBI-train candidate adds many rows with one or two positive training
occurrences.  Registration guarantees lexical coverage but does not make an
independent random input row and untied output row well estimated.

The open candidate called **Compositional Macro** is therefore:

- retain one canonical whole-motif token at inference;
- derive its lossless chemical-lexer expansion before training;
- initialize macro input/output rows from the expansion rather than an
  unrelated random vector;
- expose both macro and expansion surfaces during pretraining/fine-tuning so
  they remain functionally related.

This has adjacent precedents but not an exact molecule-T5 precedent.  SPE/BPE
construct frequent chemical substrings over an atom-level base; Group SELFIES
uses groups with attachment points plus compositional primitives; subword
initialization work such as IndoBERTweet initializes new domain tokens from
existing subwords.  None directly proves that tying or distilling a whole
motif macro to this project's anchored lexer is optimal.  It remains a
controlled 10% ablation, not part of the traversal fix.

## 2. Frozen canonicalization rule

The identity object is a **stereo-free port-labelled molecular graph**, not a
bare string:

1. parse every historical `<n*>` as a degree-one dummy port;
2. preserve real-atom element, isotope, charge, radical, hydrogen,
   aromaticity and internal bond order;
3. require every cross-motif port bond to be `SINGLE`;
4. clear atom/bond stereochemistry only, consistent with the architecture's
   identity/state separation;
5. hide occurrence-local anchor numbers while selecting the canonical graph
   traversal;
6. read dummy ports in canonical atom-output order and apply that permutation
   to anchor IDs, attachment atoms and connection sidecars;
7. remove dummy symbols to obtain the canonical pure-motif carrier.

The anchor numbers are never sorted independently.  They follow the canonical
port slots, so motifs with multiple anchors remain reconstructible.

This follows the relevant design principle in fragSMILES: fragment identity
should be canonical and independent of molecular-neighborhood traversal, while
connectors are represented separately.  It is also compatible with t-SMILES
and Group SELFIES treating attachment information as explicit structure rather
than part of an opaque traversal spelling.

Primary references:

- fragSMILES: https://www.nature.com/articles/s42004-025-01423-3
- t-SMILES: https://www.nature.com/articles/s41467-024-49388-6
- Group SELFIES: https://arxiv.org/abs/2211.13322
- SMILES Pair Encoding: https://pubs.acs.org/doi/10.1021/acs.jcim.0c01127
- IndoBERTweet vocabulary adaptation: https://aclanthology.org/2021.emnlp-main.833/

## 3. Acceptance gates

The change is admitted only if all gates pass:

1. **Alias collapse:** known reversed-chain, ring-start and branch-order
   variants produce one canonical pure motif.
2. **Port bijection:** old occurrence slots map one-to-one to canonical slots;
   no anchor or attachment atom is dropped, duplicated or independently
   reordered.
3. **Phrase fixed point:** canonical pure motif plus ordered anchors restores
   one canonical legacy fragment, and decoding rejects noncanonical aliases.
4. **Whole-molecule round trip:** encode every motif, decode every phrase,
   reconnect paired anchors and recover the same projected source molecule.
   Equality is strict canonical stereo-free molecular identity because stereo
   is intentionally carried by E3FP/audit state, not motif text.
5. **Graph preservation:** atom count, element/isotope/charge/radical/H,
   aromaticity, internal bond order, disconnected components and every
   cross-motif single bond remain unchanged.
6. **Renumbering invariance:** random input atom renumberings preserve the
   canonical motif multiset and reconstructed molecule identity.
7. **Full-corpus replay:** Phase I, Phase II and ChEBI train must be re-counted
   under the new schema before choosing 18k/20,325 or any replacement K.

The initial local prototype passes gates 1--4 on directed fixtures including a
two-anchor `C=N` motif, an aromatic traversal pair, a disconnected component
and reversed atom order.  A second local pass over the first 1,000 QM9 SDF
records found six upstream RDKit SDF parse failures; all 994 parseable projected
molecules passed encode/decode/reconnect identity, and the first 100 also
passed reverse atom-renumbering with an unchanged canonical motif multiset.
This is implementation evidence, not yet the full-corpus admission required by
gates 5--7.

## 4. Consequences for the vocabulary plan

- Existing 18,000 and 20,325 counts are **pre-canonicalization candidates**.
- Do not delete 162 rows from a registry by string grouping; regenerate counts
  from canonical occurrence surfaces and re-rank by the frozen corpus policy.
- The chemical lexer remains mandatory for open-world motifs.
- Only after canonicalization is frozen should the Compositional Macro
  initialization/exposure ablation be run.  Otherwise it would train two
  mechanisms to repair an avoidable tokenizer alias.

## 5. Implementation boundary

The prototype is in
`most_t5_next/r1/tokenizer/stereo_free_anchored_motif_surface_v1.py` but emits
surface schema `.../v2`.  It adds graph canonicalization, canonical slot
permutation, molecule reconstruction and a strict stereo-free round-trip
validator.  The full-corpus artifacts and tokenizer must be rebuilt; v1 and v2
surfaces must not be mixed in one registry or checkpoint.
