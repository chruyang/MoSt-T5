# R1 deterministic tokenizer freeze and digest binding

This directory is a new sidecar path.  It does not modify or import the locked
production runner, production codec, auditor, historical `MotifTokenizer`,
Dataset, model, or launcher.

## What is implemented

- `build_motif_tokenizer_binding_release_v1.py` validates an explicit base
  snapshot lock, P1/P2 scope locks, phase census files and a selection-policy
  decision; projects exact lexemes to pure motif tokens; saves/reloads a local
  tokenizer; runs at least three independent `PYTHONHASHSEED` probes; and emits
  a digest-to-token-ID table.
- `validate_motif_tokenizer_binding_release_v1.py` does not import the builder.
  It independently rehashes inputs and artifacts, reprojects every lexeme,
  recomputes aggregation/ranking/OOV binding, reloads the saved tokenizer
  offline, and optionally validates provenance-locked record digest sequences.
- `../contracts/motif_tokenizer_binding_release_contract_v1.json` freezes the
  engineering rules and explicitly keeps `p1_training_admission=false`.

Neither path reads SDF, imports RDKit, invokes the linearizer, nor recomputes
E3FP.  A tokenizer release remains separate from the P1 admission decision.

## Scientific status of the v1 pure projection (2026-08-07)

The v1 builder remains useful only as a deterministic engineering fixture. It
must not be used to freeze the final motif vocabulary. Its "pure motif" is
created by deleting every ``<N*>`` anchor from a fragment string; deleting an
anchor inside a branch leaves an empty branch such as ``C()`` or ``C()()``.

The complete paper-scope-v2 clean-P1 census established:

| Projection | Unique values invalid to RDKit | Occurrence-weighted invalid |
|---|---:|---:|
| exact lexeme after anchor deletion | 278,695 / 441,452 (63.1314%) | 6,001,259 / 24,153,133 (24.8467%) |
| aggregated deleted-anchor pure core | 108,506 / 214,378 (50.6143%) | same invalid occurrences as above |
| exact lexeme with each anchor replaced by legal ``[*]`` port | 0 / 441,452 | 0 / 24,153,133 |
| slot template with ``<*>`` represented as legal ``[*]`` port | 0 / 229,359 | 0 / 229,359 |

These counts are a full-census diagnosis of the projection used at that point,
not the final vocabulary input.  The later paper-scope-final-v4 membership
excludes 5,510 rather than 5,386 PCQM members, so its exact frequencies and
coverage must be rematerialized before selecting a vocabulary.  The v2 result
still establishes the structural failure mode: every invalid surface contains
an empty branch left by anchor deletion, whereas preserving the same anchors as
legal dummy-atom ports made every tested exact and slot surface parseable.

Thus "no projection failure" in the historical census meant only that the
regular expression returned a non-empty string; it did not establish a valid
chemical graph. The replacement design is a canonical core graph plus ordered
motif-local attachment ports and a reversible rare-identity fallback. Macro
and fallback forms must round-trip to the same graph+port identity before any
new tokenizer release is admitted.

## Required caller-owned decisions

The builder has no default for any item that changes research semantics.  A
`most-t5-r1/motif-token-selection-policy/v1` JSON object must explicitly set:

- `decision_status`: `approved_for_candidate` or
  `approved_for_frozen_release`;
- `discovery_scope`: `p1_only` or
  `p1_p2_permitted_train_union`;
- `min_selection_score` and `max_motif_tokens` (`null` means no cap);
- exact non-negative integer `p1_weight` and `p2_weight`;
- `base_unk` versus `dedicated_motif_unk` and its exact token;
- maximum reserved anchor ID and reserved special-token count;
- base model identifier plus immutable revision;
- the exact, UTF-8-sorted base-vocabulary overlap allow-list.

For a frozen release, every used scope lock must be `complete` and bind both a
downstream identity-exclusion proof and a permitted-membership census derivation
audit.  Candidate/global census inputs cannot be relabelled as frozen.

## Historical deleted-anchor observations (superseded for vocabulary design)

These values were observed on 2026-08-05 from the completed production-v2
global census. They describe only the now-rejected deleted-anchor string
projection; they are retained for provenance and must not determine a cutoff
or vocabulary release:

| Candidate rule | Pure motif tokens | P1 occurrence coverage |
|---|---:|---:|
| all / minimum count 1 | 214,554 | 100.0000% |
| minimum count 2 | 74,576 | 99.4211% |
| minimum count 5 | 27,258 | 98.9246% |
| minimum count 10 | 14,317 | 98.5807% |
| top 20,000 | 20,000 | 98.7626% |
| top 50,000 | 50,000 | 99.2178% |

Additional observations:

- 441,769 exact lexemes project to 214,554 pure motif tokens;
- 54,847 pure tokens aggregate more than one exact lexeme; the maximum observed
  multiplicity is 1,171 exact lexemes to one pure token;
- maximum P1 anchor ID is 15 and no projection failure was observed;
- none of the 214,554 P1 pure motifs exactly overlaps the 32,100-token frozen
  base T5 vocabulary.

P2 is not represented by these statistics.  They therefore cannot justify a
P1+P2-union vocabulary or predict P2 OOV coverage.

## P2 `phase2_pubchem_ready.lmdb` schema boundary

Read-only inspection on the 4090 host established the following physical
facts.  The ready LMDB is a single-file (`subdir=False`) legacy pickle store;
`txn.stat()['entries'] == 301655`, its payload keys are PubChem CID strings and
it has no `__len__` metadata key.  By contrast, the final LMDB has 301,656
entries: the same 301,655 payload CIDs plus `__len__`.  Extractors must enumerate
and explicitly exclude metadata keys instead of treating LMDB `entries` as a
molecule count.

A deterministic boundary sample consisting of the first and last 256
lexicographic ready keys had one closed field set in all 512 payloads:

```text
atom_to_motif_map, atoms, cid, coordinates, description, e3fp,
enriched_description, motif_seq, raw_smiles, smiles
```

All 512 keys equalled `str(record['cid'])`.  `motif_seq` was a string with the
layout `<bom>[fragment][fragment]...[.]...[fragment]<eom>`; no inter-token
whitespace was present in this sample.  Every sequence was losslessly parsed by
a square-bracket-depth state machine, every non-separator fragment count equalled
`len(atom_to_motif_map)`, and no recovered fragment contained NUL/TAB/CR/LF.
The sample schedule/key-set hash is
`12a0040afd1be3a62c61fffc2457ab8992f7cd864b9d1db9784d033edc8b9917`.

Therefore ready already contains a reversible **legacy exact fragment
sequence** inside `motif_seq`; it does not contain a separate pure-motif-token
field or per-fragment digest.  A P2 census can be generated without rerunning
the linearizer:

1. open ready read-only with `subdir=False` and exclude a closed metadata-key
   list (currently empty for ready; `__len__` is present in final);
2. deserialize each legacy pickle in an isolated, hash-locked extractor;
3. validate CID/key equality and the closed payload field set;
4. parse the `<bom>/<eom>` envelope with a bracket-depth state machine -- never
   use whitespace split or a non-greedy bracket regex because fragment SMILES
   contain nested square brackets;
5. preserve `[.]` as explicit component ranges, but exclude it from motif
   census counts;
6. for each recovered exact fragment, compute its UTF-8 SHA-256 and apply the
   frozen anchor-deletion/outer-bracket projection;
7. require motif count to equal `len(atom_to_motif_map)`, emit a sorted
   digest/fragment/count census plus membership/reject/derivation receipts, and
   repeat the extraction in a second process for byte equality.

Step 6 is retained only to reproduce the historical P1/P2 compatibility
diagnostic. It cannot supply the final vocabulary after the full-census
parseability result above; a P2 slot-aware graph+port projection must be added
before P1/P2 vocabulary union is reconsidered.

This conclusion is currently a bounded schema proof, not a completed full P2
census.  The full extractor must still prove these invariants for all 301,655
payloads before a P1+P2-union tokenizer can be frozen.

There is also a semantic compatibility blocker.  In the 512-record sample,
300 records retained `@` stereochemical markers in one or more recovered
fragments.  Anchor IDs reached 16 and behaved as legacy local attachment labels:
499/512 records violated the P1 production-v2 rule that each anchor ID occurs
exactly twice per connected component (maximum observed multiplicity 80).
Consequently P2 component boundaries must come from stored `[.]`, and P2 anchor
labels must not be interpreted using P1's global bond-ID recovery rule.  A
frozen P1+P2 union therefore requires a hash-bound projection-domain
compatibility audit (or a user-approved P2 relinearization policy); merely
concatenating the two census files is not sufficient scientific justification.

## Implemented P2 census and compatibility gates

The bounded observations above are now represented by three candidate-only
tools and three immutable contracts/specifications:

- `extract_p2_phase2_ready_motif_census_v1.py` verifies the single-file LMDB
  SHA-256 and byte count before the first `pickle.loads`, requires the literal
  acknowledgement `I_ACKNOWLEDGE_TRUSTED_PICKLE_CAN_EXECUTE_CODE`, enumerates a
  closed metadata-key set, validates the closed payload field set and CID/key
  identity, parses nested brackets, and writes membership/reject/projection,
  exact/pure census, anchor summary and a derivation receipt;
- `verify_p2_phase2_ready_motif_census_rerun_v1.py` starts the same hash-bound
  extractor in a fresh process with another `PYTHONHASHSEED` and compares the
  six deterministic artifacts byte-for-byte;
- `audit_p1_p2_motif_projection_compatibility_v1.py` independently projects
  both exact censuses and reports exact/pure overlap, stereo retention,
  exact-to-pure collapse, anchor statistics and producer/spec provenance;
- `../contracts/p2_phase2_ready_motif_census_contract_v1.json` freezes the
  extraction/rejection/artifact rules;
- `../contracts/p2_phase2_ready_legacy_motif_sequence_spec_v1.json` freezes the
  retrospective stored-sequence interpretation while explicitly marking the
  original producer unknown;
- `../contracts/p1_p2_motif_projection_compatibility_contract_v1.json` forbids
  the audit from choosing union scope, cutoff, OOV policy, tokenizer or
  training admission.

`atom_to_motif_map` is treated here only as the legacy outer motif-to-atom
group sequence needed for a cardinality check.  Its name does not justify
reinterpreting it as the later Dataset tensor direction.

The extractor source lock has the closed schema below.  The hashes must refer
to real, retained evidence; placeholder hashes are not acceptable for a real
run.  `legacy_linearization_spec_sha256` should currently bind the retrospective
spec above.  If the actual producer is later recovered, use
`motif_sequence_producer_status=hash_locked` and its real file SHA instead of
silently attributing the ready LMDB to the current local writer.

```json
{
  "schema_version": "most-t5-r1/p2-phase2-ready-source-lock/v1",
  "source_role": "phase2_pubchem_ready_lmdb",
  "source_format": "lmdb_single_file_pickle_values",
  "source_sha256": "<sha256 of the complete single LMDB file>",
  "source_bytes": 0,
  "expected_payload_entry_count": 301655,
  "expected_metadata_keys": [],
  "expected_payload_fields": [
    "atom_to_motif_map", "atoms", "cid", "coordinates", "description",
    "e3fp", "enriched_description", "motif_seq", "raw_smiles", "smiles"
  ],
  "identity_namespace": "pubchem_cid",
  "membership_status": "candidate_geometry_ready",
  "source_copy_manifest_sha256": "<retained transfer/copy manifest sha256>",
  "pickle_trust_basis_sha256": "<reviewed trust decision sha256>",
  "motif_sequence_producer_status": "unknown_legacy_producer",
  "motif_sequence_producer_sha256": null,
  "legacy_linearization_spec_sha256": "<retrospective spec sha256>"
}
```

The `source_bytes` value above is a schema illustration and must be replaced by
the observed positive byte count.

### Remote CPU/IO CLI (do not start implicitly)

Run on the 4090 host only after the source lock, copy manifest and pickle trust
basis have been reviewed.  The work is CPU/fast-disk I/O only and does not
benefit from a GPU.  Put both extraction outputs on `/root/autodl-tmp`; each
must be a new path.  Adjust `REPO` and `P1_RELEASE` to the actual remote paths.

```bash
export REPO=/root/autodl-tmp/my_code
export READY=/root/autodl-tmp/3D-MoIT/3d-mol-dataset/pubchem/pretrain/phase2_pubchem_ready.lmdb
export P2_LOCK=/root/autodl-tmp/most-t5-r1/locks/phase2_pubchem_ready_source_lock_v1.json
export P2_A=/root/autodl-tmp/most-t5-r1/tokenizer/p2_census_candidate_run_a
export P2_B=/root/autodl-tmp/most-t5-r1/tokenizer/p2_census_candidate_run_b
export P1_RELEASE=/root/autodl-fs/most-t5-r1/p1/<complete-production-v2-release>
```

```bash
PYTHONHASHSEED=104729 python "$REPO/most_t5_next/r1/tokenizer/extract_p2_phase2_ready_motif_census_v1.py" \
  --source-lmdb "$READY" \
  --source-lock "$P2_LOCK" \
  --contract "$REPO/most_t5_next/r1/contracts/p2_phase2_ready_motif_census_contract_v1.json" \
  --output-dir "$P2_A" \
  --legacy-pickle-acknowledgement I_ACKNOWLEDGE_TRUSTED_PICKLE_CAN_EXECUTE_CODE
```

```bash
python "$REPO/most_t5_next/r1/tokenizer/verify_p2_phase2_ready_motif_census_rerun_v1.py" \
  --baseline-release "$P2_A" \
  --rerun-output "$P2_B" \
  --extractor "$REPO/most_t5_next/r1/tokenizer/extract_p2_phase2_ready_motif_census_v1.py" \
  --source-lmdb "$READY" \
  --source-lock "$P2_LOCK" \
  --contract "$REPO/most_t5_next/r1/contracts/p2_phase2_ready_motif_census_contract_v1.json" \
  --report /root/autodl-tmp/most-t5-r1/tokenizer/p2_census_rerun_report_v1.json \
  --pythonhashseed 130363 \
  --legacy-pickle-acknowledgement I_ACKNOWLEDGE_TRUSTED_PICKLE_CAN_EXECUTE_CODE
```

```bash
python "$REPO/most_t5_next/r1/tokenizer/audit_p1_p2_motif_projection_compatibility_v1.py" \
  --p1-release "$P1_RELEASE" \
  --p1-contract "$REPO/most_t5_next/r1/contracts/p1_pcqm_geometry_production_release_contract.json" \
  --p2-release "$P2_A" \
  --p2-contract "$REPO/most_t5_next/r1/contracts/p2_phase2_ready_motif_census_contract_v1.json" \
  --p2-source-lock "$P2_LOCK" \
  --contract "$REPO/most_t5_next/r1/contracts/p1_p2_motif_projection_compatibility_contract_v1.json" \
  --output /root/autodl-tmp/most-t5-r1/tokenizer/p1_p2_projection_compatibility_report_v1.json
```

The source file is hashed before and after each extraction and scanned once by
LMDB, so the baseline plus independent rerun intentionally performs several
full 1.8 GB reads and 603,310 trusted-pickle deserializations.  The main risks
are arbitrary-code execution from a wrongly trusted pickle source, incomplete
source provenance, and I/O time; no GPU allocation should be enabled for this
gate.  A failed attempt is preserved as a partial directory and is never
overwritten or bulk-deleted.

The currently available base snapshot is an explicit CLI input rather than a
hard-coded path.  Its observed canonical tree hash is
`71c3fab438d892230c5aa9eaff5c8054518cafc14382a414850d387723b82f02`
(six regular files, no symlink).  A caller must still provide a matching
`most-t5-r1/base-model-snapshot-lock/v1` and explicitly approve its candidate or
frozen status.

## Minimal real 128-record binding smoke

The smallest defensible real smoke requires all of the following:

1. The completed production-v2 manifest and its hash-matching global census.
2. A byte-complete local base snapshot lock with explicit identifier, immutable
   revision, expected tokenizer class and same-revision model/tokenizer claim.
3. A P1 candidate scope lock.  A frozen smoke additionally requires downstream
   exclusion and permitted-membership census derivation proofs.  P1+P2 union
   additionally requires the corresponding complete P2 lock and census.
4. A user/experiment-approved candidate selection policy.  The smoke must not
   silently choose cutoff, cap, union scope, OOV, or base checkpoint.
5. Exactly 128 admitted records selected by a frozen schedule, decoded from the
   immutable LMDB and checked against membership plus payload-index hashes.  The
   extracted JSONL contains only member ID, record content hash, ordered motif
   digest sequence and the three independently checked motif cardinalities.
6. A `most-t5-r1/tokenizer-binding-sample-extraction-receipt/v1` binding the
   production root/manifest, sample schedule, sample JSONL, safe decoder,
   payload-index verification report, and bounded component-reference audit.
7. The sample set covers at least anchor-present, disconnected-component,
   selected-token, frozen-OOV, largest observed anchor, and high-motif-count
   cases.  Synthetic negative fixtures separately cover missing digest,
   singleton/repeated anchor, anchor overflow, non-contiguous components,
   digest/fragment mismatch and token-map drift.
8. Builder pass, independent validator pass with `--require-sample-count 128`,
   and byte-identical mapping hashes across at least three distinct
   `PYTHONHASHSEED` processes.

The bounded component-reference audit may inspect only the 128 scheduled source
records to compare anchor-derived `[.]` placement with the frozen linearizer.
It is not a full SDF scan and must not recompute E3FP.  If this audit is not
available, digest-to-token binding can be tested, but component-separator
equivalence remains an explicit blocker for the later Dataset/GPU gate.

## Test command

```text
python -m unittest -v most_t5_next.r1.tokenizer.tests.test_motif_tokenizer_binding_release_v1
python -m unittest -v most_t5_next.r1.tokenizer.tests.test_p2_motif_census_and_compatibility_v1
```

The hermetic suite uses a tiny local tokenizer fixture with 100 T5-style
sentinels.  It performs no network access and marks its output as
`candidate_tokenizer_built_non_release`.
