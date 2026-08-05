# R1 molecule-native motif linearizer

`mol_linearizer.py` is an isolated replacement candidate for the historical
SMILES-driven motif preparation step.  It does not modify
`model/CAMT5/representation.py` and is not yet wired into a training launcher.

Its public entry point is `linearize_mol(existing_rdkit_mol)`.  The caller must
keep the returned `LinearizationResult` together with the source-record ID.
The result deliberately makes three provenance boundaries explicit:

1. `motif_atom_groups` are aligned with `fragment_sequence` and contain only
   original input atom indices.  Dummy atoms never enter this mapping.
2. `metadata.cross_motif_bonds` records every paired dummy-anchor connection;
   its motif IDs refer to `canonical_motif_atom_groups`.
3. `metadata.component_fragment_ranges` represents disconnected inputs without
   inventing a molecule-wide edge.  `fragment_string` supplies `[.]` separators
   only as a legacy-compatible rendering convenience.

The motif policy is intentionally the legacy policy in deterministic form:
rings and non-single bonds seed groups; overlapping seeds merge; uncovered
atoms are singleton motifs.  Canonical group order, anchor IDs, component
order, DFS root choice, and neighbor traversal order are all stated in
`metadata.ordering_policy`.

This module is an R1 release candidate, not a claim of compatibility with the
old nondeterministic tokenizer IDs.  A new P1 vocabulary/checkpoint must be
released from this exact policy and its manifest; an old checkpoint must not be
silently reused.

`p1_topology_augmentation_v1.py` is the bounded bridge for tokenizer and PF
canary records. It reruns only this linearizer, proves that ordered motif
groups and lexeme digests equal production-v2, remaps canonical motif IDs to
logical sequence IDs, and records each extracted anchor occurrence as an
explicit `(logical motif, slot, model atom, source atom)` endpoint. It neither
recomputes E3FP nor modifies the production-v2 release.

## Fixed 32 + 256 topology canary

`run_p1_topology_canary_v1.py` applies that bridge to a frozen, disjoint
32-record smoke set and 256-record canary set. Its data path mirrors the
production builder at module granularity:

```text
official SDF record
  -> RDKit binary worker round-trip
  -> frozen explicit-H projection
  -> mol_linearizer
  -> equality with production-v2 motif groups and lexeme digests
  -> topology_augmentation.jsonl + manifest.json
```

It opens production LMDB shards read-only, emits no coordinates or E3FP, and
writes to a new directory outside the production release. Selection is an
input artifact rather than a runner option chosen after inspection. The JSON
shape is:

```json
{
  "schema_version": "most-t5-r1/p1-topology-canary-selection/v1",
  "selection_id": "p1-topology-canary-<version>",
  "release": {
    "release_id": "<production-v2 release id>",
    "full_release_manifest_sha256": "<sha256>",
    "logical_release_root_sha256": "<sha256>"
  },
  "groups": {
    "smoke": [
      {"sdf_record_index": 0, "selection_tags": ["overfit"]}
    ],
    "canary": [
      {"sdf_record_index": 1, "selection_tags": ["coverage"]}
    ]
  }
}
```

The real arrays must contain exactly 32 and 256 unique admitted ordinals;
`selection_tags` are optional but, when present, are sorted and unique. A
remote invocation is:

```bash
python -B -m most_t5_next.r1.adapter.run_p1_topology_canary_v1 \
  --selection /path/to/frozen_topology_canary_selection.json \
  --release-root /path/to/pcqm-geometry-production-v2 \
  --source-archive /path/to/pcqm4m-v2-train.sdf.tar.gz \
  --output-dir /path/to/new/topology-canary-output
```

The runner streams the tar member only through the largest selected ordinal;
it does not extract the 9.7 GB SDF or rescan the unused suffix. The manifest
records the production-v2 member lock plus the observed prefix byte count and
SHA-256 without claiming a second full-source audit. Keeping the explicit
schedule inside the first completed shard (ordinals below 25,000) makes this
topology check fast and keeps all 288 selected LMDB reads local to one shard.
Freeze those ordinals from admitted `membership.jsonl` rows before looking at
augmentation output (for example, by one declared SHA-256 rank and seed), and
keep that selection JSON as an input artifact rather than regenerating it on
each run.
Progress is printed every 250,000 records for schedules that intentionally
reach farther into the corpus. This stage is CPU/I/O work; an idle GPU is
expected.

### Building the frozen selection

`build_p1_topology_canary_selection_v1.py` creates the selection before any
SDF or topology-augmentation replay. It reads only admitted records from
production `shard-000000` and retains three existing statistics:
`model_atom_count`, `motif_count`, and motif-group sizes. Smoke members are the
first 32 under `SHA256(seed|smoke|member_id)`. The disjoint 256-member canary
uses frozen quotas for one-motif, high-motif-count, low/high-atom-count,
singleton-heavy, and large-motif records; SHA-256 resolves each quota and a
global SHA-256 rank fills any unused slots. The output records the thresholds,
quotas, observed eligibility, seed, shard manifest, and release hashes.
The atom-count boundaries are fixed at `<=12` and `>=18`; `18` is the observed
shard0 maximum from the frozen feature census, not a dynamically recomputed
percentile.

```bash
python -B -m most_t5_next.r1.adapter.build_p1_topology_canary_selection_v1 \
  --release-root /path/to/pcqm-geometry-production-v2 \
  --selection-id p1-topology-canary-shard0-20260806-v1 \
  --seed most-t5-p1-topology-canary-20260806-v1 \
  --output /path/to/new/frozen_topology_canary_selection.json
```

This scans one 25,000-row LMDB shard, not the complete release. Rejected
membership rows are skipped without opening a payload. The builder stops if
the shard has fewer than 288 admitted records or the resulting canary fails to
represent any declared boundary tag.
