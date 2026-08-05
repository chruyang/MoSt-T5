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
