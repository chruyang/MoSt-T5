# QM9 E3FP correspondence and downstream atom-mapping gate (2026-08-12)

> Status: **QM9 correspondence PASS; full downstream validation deferred until
> formal pretraining has started.**  Deferral is a scheduling decision only:
> no downstream dataset is admitted to model training until its own complete
> atom-mapping gate has passed.

## 1. Question and evidence boundary

The 3D-MolT5 downstream artifacts publish `molecule_fp`, but most releases do
not publish the exact conformer from which every tensor was produced. It is
therefore scientifically invalid to generate a new RDKit conformer and treat
its E3FP as a numerical reproduction of the published state.

QM9 is the appropriate implementation-verification dataset because both sides
are available: the published 3D-MolT5 `(SMILES, SELFIES, molecule_fp)` records
and the public QM9 SDF conformers. QM9 validates the producer and atom-axis
mapping. For a downstream release without the exact conformer, the published
E3FP tensor is reused and its atom correspondence is validated, but numerical
regeneration is not claimed.

## 2. Locked QM9 inputs and runtime

- QM9 SDF SHA-256:
  `98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e`
- 3D-MolT5 QM9 train parquet SHA-256:
  `95731ca5b9ea05fb5b22b41ee9654e01d12ce47aa54b3f162c1470782ff7a574`
- official `3d_tokenize.py` SHA-256:
  `12f11c280560033fc2a443e35487c08c4ea2bdb8af8c8b0f7903eaa8b920992c`
- vendored E3FP `fprinter.py` SHA-256:
  `bfc442ab7384cd91940f1ffedbcbbff2396d26e57eb692a5b6fb5c97867c8381`
- Python 3.8.20, RDKit 2024.03.5, SELFIES distribution 2.1.1,
  mmh3 4.1.0 and PyArrow 17.0.0.
- E3FP: 4096 real IDs, level 3, `rdkit_invariants=True`,
  `all_iters=True`, `exclude_floating=False`.

No RDKit conformer was generated. Numeric recomputation used the coordinates
already present in the public QM9 SDF.

## 3. Complete QM9 atom-axis result

The parquet contains 128,860 unique `(SMILES, SELFIES, molecule_fp)` surfaces.

- 27 surfaces contain no usable E3FP row: every row is
  `[-1,-1,-1,-1]`;
- all 128,833 nonempty surfaces have exactly as many retained E3FP rows as
  stereo-free heavy atoms;
- all 128,833 have identical ordered atom signatures after parsing the source
  SMILES and decoding the published SELFIES;
- there were zero SELFIES decode or molecule-parse failures.

For this locked QM9 artifact, retaining rows whose L0 value is nonnegative
while preserving their order produces the correct heavy-atom axis. This is an
observed property of this producer artifact, not a generic rule for unrelated
SELFIES data.

## 4. Numerical E3FP reproduction result

The audit reproduced the official path: kekulized canonical SMILES,
`_smilesAtomOutputOrder`, molecule renumbering, SELFIES decoder attribution,
coordinate centering, vendored E3FP generation, official merged-shell repair,
folding modulo 4096, and placement on the SELFIES symbol axis.

- deterministic hash sample: 256 SDF molecules;
- all SDF candidates belonging to ambiguous published surfaces: 78 molecules;
- combined distinct recomputations: 334;
- exact matches to a published `molecule_fp`: 334/334;
- mismatches: 0.

There are 39 exact `(SMILES, SELFIES)` keys for which the parquet contains two
different E3FP tensors. The public SDF contains 78 corresponding conformers.
Recomputation recovered both published tensor sets for every key: no published
state remained unexplained and no regenerated state was absent from the
parquet. These are real state/conformer variants, not atom-order errors or
nondeterministic fingerprints.

This freezes an important join rule: **SMILES alone is not an E3FP state
identifier.** A downstream record must bind the full published state, for
example by exact SELFIES plus serialized E3FP/state hash.

## 5. Full downstream validation after formal pretraining starts

Once formal pretraining is running, otherwise idle CPU resources will validate
every registered downstream dataset in full. This work may overlap GPU
pretraining, but each downstream dataset remains training-inadmissible until
its report is complete and passes.

The gate is applied separately to every split of every task, including current
PubChem/PubChemQC property and text releases, QM9, ChEBI-20, 3D-molecule
captioning, selected MoleculeNet tasks when geometry is consumed, zero-shot
retrieval, and any admitted USPTO-50K or Mol-Instructions molecular component
that consumes E3FP.

For every record, the following chain must be explicit and bijective:

```text
published E3FP row
  -> published SELFIES atom occurrence
  -> canonical/source heavy atom
  -> model atom ID
  -> fragSMILES fragment-local atom ID
  -> fragment carrier and, when applicable, connector endpoint
```

The acceptance contract is:

1. every retained E3FP row maps to exactly one source heavy atom;
2. every model heavy atom expected to consume geometry maps to exactly one
   retained E3FP row;
3. there are no duplicate, missing, crossed or out-of-range assignments;
4. the mapping preserves atomic number, isotope, formal charge, aromaticity,
   total hydrogen count and ordered bond environment; defined stereo/CIP is
   checked at the authoritative source boundary even when the model's motif
   identity surface is intentionally stereo-free;
5. atom renumbering into fragment-local/model order is an explicit permutation,
   never an identity-order assumption;
6. every connector endpoint resolves to the same attachment atom whose E3FP
   row it consumes, including multi-endpoint fragments;
7. explicit hydrogen symbols with no published E3FP row map to `None` and
   never borrow a heavy-atom row;
8. macro, chemical-lexer fallback and whole-molecule lossless fallback records
   obey the same atom-address contract;
9. reserialization and cache reload preserve the mapping and tensor bytes;
10. duplicate identities with different E3FP states remain distinct and are
    bound by a state hash, not collapsed by SMILES.

Atom-count equality alone is insufficient for this gate.

## 6. Dataset-specific numeric policy

- **Exact source conformer available:** regenerate under the locked producer
  when useful and require exact per-atom/per-level equality, as done for QM9.
- **Exact source conformer unavailable:** reuse the published tensor, validate
  the complete atom bijection and provenance, and do not claim numeric
  regeneration.
- **No trusted published E3FP and no authoritative 3D geometry:** run the task
  as 2D-only or omit its geometry condition; do not silently synthesize an
  unregistered state.

## 7. Required output and failure policy

Each dataset/split report must bind source revision and hashes, runtime,
producer contract, record count, E3FP-present/absent counts, atom and endpoint
coverage, fallback modes, state-duplicate counts and every rejection category.
It must report both directions of the bijection and include deterministic
boundary examples.

Any unmapped or multiply mapped atom is a fail-closed dataset error. The
pipeline may not fix it by truncation, padding, row-order guessing, replacement
sampling or newly generated conformers. A chemistry-supported correction must
be implemented at the dataset-adapter boundary and the complete affected split
must be replayed before training admission.
