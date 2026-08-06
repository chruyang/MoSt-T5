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
`molecule-identity-row/v1` schemas. Manifest SHA-256 values are supplied
explicitly so the derivation is bound to the intended releases.

```bash
python -m most_t5_next.r1.overlap.derive_clean_pretrain_membership_v1 \
  --pretrain-manifest /path/to/pretrain/collection_manifest.json \
  --pretrain-manifest-sha256 <sha256> \
  --protected-manifest /path/to/task-a-validation/collection_manifest.json <sha256> \
  --protected-manifest /path/to/task-a-test/collection_manifest.json <sha256> \
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

## Official KPGT scaffold membership

`build_kpgt_scaffold_manifests.py` freezes the released BACE, BBBP, and
ClinTox `scaffold-0/1/2.npy` memberships after checking index coverage,
partition isolation, binary-label evaluability, canonical identities, and
both chiral and achiral Bemis-Murcko isolation. It refuses candidate layouts:
the CLI requires the exact provenance assertion
`verified_official_kpgt_figshare` and the downloaded official archive's
path and SHA-256 before entering the release's NumPy-pickle boundary. The tool
hashes that regular, non-symlink ZIP/TAR archive itself, rejects traversal,
links, duplicate or ambiguous members, and requires every one of the 12 input
files to be byte-identical to its unique archive member before reading CSV or
pickle content. Nothing is extracted to disk.

```bash
python -m most_t5_next.r1.overlap.build_kpgt_scaffold_manifests \
  --dataset-root /path/to/extracted/KPGT/datasets \
  --output-dir /path/to/new/kpgt-membership-v1 \
  --official-archive-path /path/to/Figshare-file-35391163 \
  --official-archive-sha256 <sha256-of-Figshare-file-35391163> \
  --source-provenance verified_official_kpgt_figshare
```

The new output directory contains nine task/replica member JSONL files, 27
partition-specific `identity-collection-manifest/v1` collections, the union of
validation/test connectivity identities across all three replicas, and source
and summary manifests. Validation/test collection manifests can be supplied
directly to `derive_clean_pretrain_membership_v1.py`; training memberships are
not added to the protected union.

## HIV authoritative-source derived split

`build_hiv_murcko_split.py` binds the official DeepChem MoleculeNet `HIV.csv`
object by revision, byte size, SHA-256, MD5/ETag, header, and 41,127 members.
It deterministically reproduces the non-chiral Bemis-Murcko group ordering and
greedy 8:1:1 allocation semantics of DeepChem 2.8.0 `ScaffoldSplitter`. The
result is deliberately named
`HIV-MoleculeNet/DeepChem-Murcko-8:1:1-derived-v1`: it is a new published
project membership, not an exact released 3D-MolT5, KPGT, or DeepChem split.

```bash
python -m most_t5_next.r1.overlap.build_hiv_murcko_split \
  --source-csv /path/to/official/HIV.csv \
  --source-sha256 9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22 \
  --source-revision deepchem-HIV.csv-etag-9ad10c88f82f1dac7eb5c52b668c30a7 \
  --output-dir /path/to/new/hiv-membership-v1
```

The builder rejects invalid source bytes, malformed or invalid molecules,
label disagreement, incomplete coverage, scaffold leakage, and splits for
which ROC-AUC is undefined. It emits the source/split manifests, exact member
rows, and validation/test connectivity identities for the protected union.
