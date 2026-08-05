# R0.2 — Phase-2 membership audit and singleton-geometry policy

**Status:** completed remotely, metadata-only.  This record establishes the
P2 split and the reason for its three-record geometric delta; it does not
authorize a training launch.

**Remote evidence:**
`/root/autodl-fs/most-t5-p0/reports/r0-p2-membership-20260731/r0_p2_membership_audit.json`

No dataset bytes were downloaded or copied to the local workspace.  The
evidence JSON is a 6-KB remote sidecar report generated from read-only scans.

## Which P2 source is authoritative?

The correct Phase-2 pretraining membership is the CID-keyed LMDB:

`/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb`

It contains **301,658** records.  It is not interchangeable with the
316,658-record all-data LMDB.  The complete 3D-MoIT partition is:

| Split | Molecules | Relation |
| --- | ---: | --- |
| P2 pretrain | 301,658 | authoritative P2 membership |
| Downstream train | 12,000 | disjoint holdout |
| Downstream validation | 1,000 | disjoint holdout |
| Downstream test | 2,000 | disjoint holdout |
| All PubChem records | 316,658 | exact union of the four splits |

Both P2 pretrain text sources independently contain exactly the same 301,658
unique CIDs:

- `2d_computed_properties.json`: 1,199,066 task rows;
- `2d_descriptive_properties.json`: 1,508,290 task rows.

This establishes that P2 text alignment and P2 3D membership agree before
E3FP preprocessing.

## Why does the processed P2 set contain 301,655 rather than 301,658?

The processed Phase-2 final and motif-ready LMDBs each contain exactly
**301,655** payload CIDs.  Their key set equals the intersection of the
301,658 P2 pretrain CIDs with the 316,655-entry 3D-MolT5 E3FP source.  The
three omitted CIDs are:

| CID | Molecular case | Reproduced E3FP outcome |
| ---: | --- | --- |
| 135,476,785 | one atom, finite `[1, 1, 3]` coordinates | all-padding E3FP tensor |
| 135,476,786 | one atom, finite `[1, 1, 3]` coordinates | all-padding E3FP tensor |
| 135,476,787 | one atom, finite `[1, 1, 3]` coordinates | all-padding E3FP tensor |

RDKit parsing and motif linearization succeed for all three.  The failure is
specific to the current E3FP construction: it calls pairwise-distance logic
(`scipy.spatial.distance.pdist/squareform`), which has no atom pair for a
single-atom coordinate set and raises before a valid geometry representation
is produced.  Thus this is a **singleton-geometry representation boundary**,
not a P2 split error or evidence of arbitrary data loss.

## Data-flow contract

```text
3D-MoIT P2 pretrain CID set (301,658)
  + matching text task rows (same 301,658 CIDs)
  -> E3FP-valid geometry set (301,655)
  -> atom-to-motif mapping / motif-ready P2 set (same 301,655 CIDs)
```

The historical intermediate `phase2_pubchem.lmdb` is absent, so the exact
historic filtering line cannot be proven.  What is proven is the full
set-level propagation: all processed artifacts omit the same three singleton
CIDs, and no foreign CIDs were added.

## Release policy

The default geometry-training policy is:

1. Freeze and publish an explicit **301,655-member P2 geometry manifest**.
2. Store the three CIDs in a reject ledger with reason
   `E3FP_SINGLE_ATOM_NO_DISTANCE_PAIRS`; do not silently report 301,658 as the
   processed geometry count.
3. If all text examples are retained, route these three only through 2D/text
   losses, with their 3D target and 3D MSE disabled and counted separately.
4. Reintroduce them into the 3D branch only after defining a singleton-specific
   geometric sentinel that is distinguishable from padding, then validate it
   with no-confusion tests and an ablation.

This policy is preferable to inventing a standard E3FP target, because a
single atom has no pairwise geometry under the current representation.

## Relevant surviving assets

- Raw P2 pretrain: `/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb`
- Raw all-data split source: `/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/3d-pubchem-all.lmdb`
- 3D-MolT5 full E3FP source:
  `/root/autodl-tmp/3D-MolT5/3d_tokenization/3d-pubchem-all-e3fp.lmdb`
- P2 mapping output:
  `/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_final.lmdb`
- P2 motif-ready output:
  `/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_ready.lmdb`

