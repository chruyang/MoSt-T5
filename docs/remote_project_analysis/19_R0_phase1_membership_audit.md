# R0.2 — Phase-1 membership audit and data-flow admission record

**Status:** completed remotely, read-only source audit; this is an admission
record, not permission to launch Phase 1.

**Remote report:**
`/root/autodl-fs/most-t5-p0/reports/r0-legacy-p1-membership-v2-20260730T171305Z/`

No molecular data or model asset was downloaded or copied to the local
workspace during this audit.  The local copy of the harness is only a small
sidecar script; all scans and generated evidence live on the remote shared
filesystem.

## Result

The 3D-MoIT PubChemQC split is internally consistent with the intended
Phase-1 **molecule** membership:

| Set / stage | Unique molecule IDs | Result |
| --- | ---: | --- |
| Official `pretrain` split | 3,119,717 | matches the declared target |
| Train + validation + test + pretrain union | 3,899,647 | matches the declared full release |
| `pretrain_whitelist.json` | 3,119,717 | exactly the Phase-1 pretrain set |
| Raw PubChemQC LMDB | 3,899,647 | complete against the official union |
| Recovered E3FP LMDB | 3,899,644 | three IDs absent after raw ingestion |
| Recovered final P1 LMDB | 3,899,644 | the same three IDs absent |

The four split memberships are disjoint.  The P1 whitelist is disjoint from
the downstream blacklist, while the `pretrain_blacklist` represents the full
official union rather than a P1 exclusion list.  Therefore it must **not** be
used as a training whitelist.

## The three-record delta

The only official P1 pretrain IDs absent from both the recovered E3FP and
final LMDBs are:

| PubChemQC ID | In raw source | In P1 pretrain split | In E3FP/final | Evidence-based classification |
| ---: | :---: | :---: | :---: | --- |
| 601 | yes | yes | no | H₂; current legacy E3FP replay is unsupported |
| 17,042 | yes | yes | no | H₂; current legacy E3FP replay is unsupported |
| 44,985 | yes | yes | no | H₂; current legacy E3FP replay is unsupported |

The membership audit proves the first available disappearance point.  A
separate read-only replay then establishes a stronger mechanism-level fact:
each is a two-explicit-hydrogen H₂ record for which the legacy E3FP procedure
raises `ValueError` and cannot generate a fingerprint.  This supports the
release label `E3FP_UNSUPPORTED_H2`; it still does not prove the exact historic
process that first omitted them.

## Correct P1 data-flow contract

```text
official P1 pretrain IDs (3,119,717)
  ∩ raw PubChemQC LMDB (3,899,647)
  ∩ reproducibly generated/validated E3FP stage
  ∩ validated motif-final stage
  -> frozen P1 membership manifest
```

For the current recovered artifacts this intersection is **3,119,714**, not
3,119,717.  A new mainline must choose one explicitly documented policy:

1. Define and validate a dedicated H₂/no-pair geometry representation, then
   recreate the three targets without confusing them with padding; or
2. Exclude exactly these three `E3FP_UNSUPPORTED_H2` IDs and revise the frozen
   P1 manifest and all reported counts to 3,119,714.

Neither option authorizes selecting all 3,899,644 final-LMDB records: that
would leak the downstream train/validation/test membership into P1.

## Invalid artifact path identified

`/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchemqc/pubchemqc_e3fp.lmdb`
has a 1-TB apparent size but only an 8-KB allocated footprint and yields no
usable keys in the audit.  It is a sparse placeholder, not the E3FP source
for P1.  The validated recovered source is:

`/root/autodl-fs/migrations/most-t5-20260730T211141+0800/recovery_candidates/legacy_lmdb/pubchemqc_e3fp_shrunk.lmdb`

The recovered E3FP and final key sets are identical; their shared E3FP arrays
and shared metadata agree record by record for all 3,899,644 surviving keys.
Future harnesses must reject a required stage whose LMDB scan is not `ok`.
`r0_legacy_p1_membership_audit.py` was strengthened accordingly: it aborts
before rendering exceptions for an invalid raw/E3FP/final stage, and it bounds
exception-context output while retaining the total and a stable set hash.

## Remaining R0 gates

- P2 still needs a reproducible 301,658-member source list and an explanation
  for its current three-record delta.
- The final deterministic tokenizer contract still needs a successful
  cross-`PYTHONHASHSEED` process gate and frozen permitted vocabulary scope.
- Existing P0 geometry blockers remain: 353 atom mappings reference E3FP
  padding and 1,508 motif groups lack geometry.
- The current `train1.py` launcher does not apply the P1 whitelist and its
  configured final LMDB is absent; neither may be used for a new run.

## Reproducibility evidence

- Frozen P1 instruction JSON SHA-256:
  `8a3573a74de28d5915b1f1a42ee0748221342c672359e2992dab7e71dd9efa6f`
- Audited three-ID exceptional-set SHA-256:
  `47f6dceb9cebe8de7aacd95edc60fcb85655880894f7b8f5e17c75f5dd39cb65`
- Harness path:
  `/root/autodl-fs/most-t5-p0/harness/p0_validation/r0_legacy_p1_membership_audit.py`
