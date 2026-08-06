# R1 identity overlap utilities

`derive_clean_pretrain_membership_v1.py` derives a membership-only view of an
existing pretraining release. It does not rewrite the release or copy molecule
payloads.

The rule is intentionally small:

- hard exclusion key: `connectivity_identity_sha256`;
- protected sources: downstream `validation` and `test` collection manifests;
- stereo identity: overlap counts are reported but never filter members;
- text identity: manifest availability is reported but text rows are not read
  and never filter members.

The command consumes the existing `identity-collection-manifest/v1` and
`molecule-identity-row/v1` schemas. The caller supplies manifest paths only.
The program validates their schemas, scientific roles, referenced molecule-row
closure, and connectivity-identity specification. Observed artifact digests
remain in the result as provenance/interface identifiers; callers do not
repeat them as scientific-admission arguments.

```bash
python -m most_t5_next.r1.overlap.derive_clean_pretrain_membership_v1 \
  --pretrain-manifest /path/to/pretrain/collection_manifest.json \
  --protected-manifest /path/to/task-a-validation/collection_manifest.json \
  --protected-manifest /path/to/task-a-test/collection_manifest.json \
  --output-dir /path/to/clean-membership-v1
```

Outputs are deterministic for the same source bytes, regardless of protected
manifest argument order:

- `permitted_member_ids.jsonl`: source member IDs retained for training;
- `excluded_member_ledger.jsonl`: excluded source member ID, connectivity hash,
  and every matched protected task/split/collection;
- `clean_membership_manifest.json`: source release bindings, policy, counts,
  report-only stereo/text facts, and output hashes.

The original pretraining payload remains the only payload store. A training
dataset joins its original records to `permitted_member_ids.jsonl`; this
offline derivation is not part of the per-batch model path.

## QM9 molecule-group-disjoint split

`build_qm9_identity_split.py` derives the clean HOMO/LUMO/gap view from the
official 3D-MolT5 Hugging Face revision
`QizhiPei/e3fp-mol-instructions-qm9@bfe55090be9ebf1c9cbbe6687a5796711ac0edd8`.
Production admission is scientific rather than byte-layout based: exactly one
Parquet is accepted for each of `train` and `validation`, released `test` is
forbidden because it duplicates validation, every required field and row is
parsed, and the fixed semantic census, molecule identities, PCG64 seed 42, and
110000/10000/rest group split must all match the protocol under RDKit
2024.03.5. Exact row counts per partition are reported outcomes of the frozen
group membership, not independent admission targets. Input file names,
byte sizes, and SHA-256 values are recorded only as provenance observations;
they never decide admission.

Molecule grouping uses canonical non-isomeric connectivity after the shared
identity contract's minimal explicit-H projection. Distinct stereochemical
states remain distinct identities and instruction states, while every state
of one connectivity is assigned to the same split.

```bash
python -m most_t5_next.r1.overlap.build_qm9_identity_split \
  --train /path/to/revision/train.parquet \
  --validation /path/to/revision/validation.parquet \
  --output-dir /path/to/new/qm9-connectivity-clean-split-v2
```

## Official KPGT scaffold membership

`build_kpgt_scaffold_manifests.py` freezes the released BACE, BBBP, and
ClinTox `scaffold-0/1/2.npy` memberships after checking index coverage,
partition isolation, binary-label evaluability, canonical identities, and
both chiral and achiral Bemis-Murcko isolation. It refuses candidate layouts:
the CLI requires the exact provenance assertion
`verified_official_kpgt_figshare` and the downloaded official archive path
before entering the release's NumPy-pickle boundary. The tool hashes that
regular, non-symlink ZIP/TAR archive itself, rejects traversal,
links, duplicate or ambiguous members, and requires every one of the 12 input
files to be byte-identical to its unique archive member before reading CSV or
pickle content. Nothing is extracted to disk. `--official-archive-sha256` is
optional integrity metadata and never the scientific-admission criterion.

```bash
python -m most_t5_next.r1.overlap.build_kpgt_scaffold_manifests \
  --dataset-root /path/to/extracted/KPGT/datasets \
  --output-dir /path/to/new/kpgt-membership-v1 \
  --official-archive-path /path/to/Figshare-file-35391163 \
  --source-provenance verified_official_kpgt_figshare
```

The new output directory contains nine task/replica member JSONL files, 27
partition-specific `identity-collection-manifest/v1` collections, the union of
validation/test connectivity identities across all three replicas, and source
and summary manifests. Validation/test collection manifests can be supplied
directly to `derive_clean_pretrain_membership_v1.py`; training memberships are
not added to the protected union.

## HIV authoritative-source derived split

`build_hiv_murcko_split.py` admits the official DeepChem MoleculeNet `HIV.csv`
by its official URL/revision, exact fields and label semantics, and frozen
population accounting under the same RDKit 2024.03.5 canonicalization used by
the present PCQM/KPGT data path. SHA-256, MD5/ETag, and byte size remain provenance
observations: a missing or different digest is recorded but does not block the
semantic protocol. All 41,127 source members are parseable under the frozen
version and deterministically follow the non-chiral Bemis-Murcko group ordering
and greedy 8:1:1 allocation semantics of DeepChem 2.8.0 `ScaffoldSplitter`.
RDKit 2025.09.1 rejects seven organometallic rows and 1,116 rows in the released
QM9 artifact, so that newer-version result is retained only as a version-drift
diagnostic and is not mixed into the official membership. The result is
deliberately named
`HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v2`: it is a new published
project membership, not an exact released 3D-MolT5, KPGT, or DeepChem split.
Version 2 computes both molecule identities and the non-chiral Murcko scaffold
from the same post-projection molecule returned by
`shared_identity_normalization_v1.py`.

```bash
python -m most_t5_next.r1.overlap.build_hiv_murcko_split \
  --source-csv /path/to/official/HIV.csv \
  --source-url https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv \
  --source-revision deepchem-HIV.csv-etag-9ad10c88f82f1dac7eb5c52b668c30a7 \
  --output-dir /path/to/new/hiv-membership-v2
```

`--source-sha256` may optionally record a caller-supplied observation; it is
never a hard gate. The builder rejects wrong source population accounting,
malformed fields, label disagreement, incomplete eligible coverage, scaffold
leakage, and splits for which ROC-AUC is undefined. It emits source/split
manifests, all 41,127 member rows, an empty invalid ledger, and
validation/test connectivity identities for the protected union.

## QM9 and HIV identity-collection adapter

`build_qm9_hiv_identity_collections_v1.py` projects the already-frozen QM9
group split and HIV member split into the same
`identity-collection-manifest/v1` interface used by KPGT and PCQM. It performs
no chemistry, no resplitting, and no task filtering. Both upstream releases
must declare the same executable identity-normalization contract used by PCQM.
QM9 connectivity-group membership is preserved while one collection member is
emitted per distinct stereochemical identity within the group; repeated
instruction rows for the same stereochemical identity are not multiplied.
HIV rows retain their frozen source-member identity. Only validation and test
collections are emitted.

```bash
python -m most_t5_next.r1.overlap.build_qm9_hiv_identity_collections_v1 \
  --qm9-split-manifest /path/to/qm9/split_manifest.jsonl \
  --qm9-summary /path/to/qm9/split_summary.json \
  --hiv-member-manifest /path/to/hiv/member_manifest.jsonl \
  --hiv-split-manifest /path/to/hiv/split_manifest.json \
  --output-dir /path/to/new/qm9-hiv-identity-collections
```

The adapter records source observations in its summary, while scientific
membership remains defined by the two upstream split manifests and their
semantic invariants.

## Reported and connectivity-clean downstream views

`derive_downstream_connectivity_clean_view_v1.py` preserves an imported
reported train/validation/test protocol and derives a second, explicitly named
view whose molecular connectivity groups are disjoint.  A connectivity is
assigned once using the fixed priority `test > validation > train`.  Every
occurrence and every molecule-text pair in the owning split is retained; lower
priority occurrences are written to a disposition ledger with their reason.
The reported inputs are never rewritten, and the test molecule and text-pair
projections must remain unchanged.

```bash
python -m most_t5_next.r1.overlap.derive_downstream_connectivity_clean_view_v1 \
  --train-manifest /path/to/reported/train/collection_manifest.json \
  --validation-manifest /path/to/reported/validation/collection_manifest.json \
  --test-manifest /path/to/reported/test/collection_manifest.json \
  --output-dir /path/to/connectivity-clean-view
```

This is a molecule-identity split guarantee.  It does not imply that distinct
molecules cannot share identical or normalized text; text-overlap statistics
remain a separate reported diagnostic.  Paper tables should retain the
published/reported split result and label the connectivity-clean result as a
second protocol rather than replacing the former silently.

## Controlled Motif Editing evaluation membership

`build_controlled_editing_memberships_v1.py` freezes the 200 published
MoleculeSTM editing molecules as a sealed compatibility test and samples a
separate 400-molecule development membership from the complete ZINC250K source
at the same repository revision.  The ZINC population is canonicalized under
RDKit 2024.03.5 using the shared explicit-hydrogen projection, deduplicated by
canonical non-isomeric connectivity, and filtered against test connectivity
before the deterministic seed-42 selection.  The test set is used only for
this leakage exclusion; it is not used for model or hyperparameter selection.

```bash
python -m most_t5_next.r1.overlap.build_controlled_editing_memberships_v1 \
  --sealed-test-smiles /path/to/single_multi_property_SMILES.txt \
  --zinc-csv /path/to/250k_rndm_zinc_drugs_clean_3.csv \
  --source-revision ff2de71fa6bb0533d5e740db6d88a0442a0d38e8 \
  --output-dir /path/to/controlled-editing-membership-v2
```

Only validation/test molecule membership and the twelve published prompt IDs
are frozen here.  No supervised train set, molecule-prompt Cartesian product,
or text-pair identity is invented.  The `n=400` binomial half-width recorded in
the summary is a membership-size precision heuristic for a future
one-result-per-molecule Bernoulli metric, not a power guarantee for multi-prompt
or stratified analyses.
