# R1 CPU worker handoff

Status: the versioned production v2 data release has completed one real
PCQM4Mv2 full run and passed independent v3 audits on both the CPU source and
the retained-region copy.  This is a production-release sub-gate, not P1
training admission; overlap, tokenizer binding, semantic recomputation policy,
and the 4090 model-interface gate remain open.

## Purpose

Run the CPU-heavy PCQM4Mv2 geometry release work on a CPU-only AutoDL instance,
then publish only verified immutable artifacts to `autodl-fs`.  Windows keeps
small source, contracts, environment evidence, and reports only; raw PCQM4Mv2
and generated LMDB shards never transit through the local machine.

## Local source material already retained

- `../r1/`: adapter, gates, contracts, tests, and R1 design notes.
- `../../tokenization/3d_tokenization/e3fp/`: complete vendored E3FP source
  closure, including `defaults.cfg`.
- `../../docs/remote_project_analysis/`: R0/R1 audit decisions and evidence
  interpretation.

The two executable source trees above are currently below 1 MiB in total.
Generated `__pycache__` files are not part of a future bundle.

## Canonical shared inputs

- `/root/autodl-fs/most-t5-p0/sources/pcqm4mv2/ogb-pcqm4mv2-train-3d-v1/archive/pcqm4m-v2-train.sdf.tar.gz`
- `/root/autodl-fs/most-t5-r1/sources/ogb_pcqm4mv2_companion/ogb-pcqm4mv2-v1/`
- `/root/autodl-fs/migrations/most-t5-20260730T211141+0800/environment/3dmolt5-conda-explicit.txt`
- `/root/autodl-fs/migrations/most-t5-20260730T211141+0800/environment/3dmolt5-environment.yml`

The source contract remains authoritative. A byte-identical working copy in
`autodl-tmp` requires a hash-bound staging receipt; a symlink or an unrecorded
path substitution is not acceptable.

## Storage layout

Shared, persistent storage:

```text
/root/autodl-fs/most-t5-r1/
  sources/
  cpu-bundles/<bundle-id>/
  runs/<run-id>/completed-shards/
  reports/<run-id>/
  releases/<release-id>/
```

CPU-instance fast working storage:

```text
/root/autodl-tmp/most-t5-r1/<run-id>/
  bundle/
  inputs/
  work/shards/
  state/
  logs/
```

Copy the compressed canonical inputs from `autodl-fs` to `autodl-tmp` once,
verify their hashes, and stream/decompress the SDF once through a producer into
multiple CPU workers. Full extraction is optional and must be justified by the
10k I/O profile before the source contract is revised.

## Production capabilities exercised in the 2026-08-05 full run

1. A separately versioned full-release schema was used; the bounded smoke
   harness was not relaxed in place.
2. One streaming producer, 64 RDKit/motif/E3FP workers, deterministic
   ordinal-ordered collection, and immutable 25k-record shards produced 136
   shards covering all 3,378,606 records.
3. Partial attempts and completed shard boundaries are distinct; completed
   outputs are never overwritten.
4. Membership, reject ledger, identity hashes, ordered motif digest bindings,
   global motif census, payload index, and LMDB payloads were emitted in the
   same pass.
5. The run used a hash-bound staging receipt, runtime attestation, global
   logical release root, and immutable full manifest.
6. The standalone v3 auditor fully checks structural closure and independently
   decodes a deterministic sample; every reject is scheduled for later
   semantic recomputation. It intentionally does not claim a full independent
   RDKit/E3FP recomputation.
7. P1 admission remains split from the data release and is explicitly false.

## Required validation sequence

1. Capture the CPU runtime lock and verify all source/bundle hashes.
2. Run fresh 128-record correctness, failure injection, and golden hashes.
3. Run one deterministic stratified 10k benchmark at 1, 8, 16, and the target
   worker count. The old prefix-1000 run is not an additional mandatory stage.
4. Record per-stage throughput, aggregate records/s, p50/p95/p99 latency, RSS,
   I/O, bytes/record, reject classes, and scaling efficiency.
5. Run the full pass exactly once, followed by full structural checks and the
   independent stratified semantic audit.
6. Freeze the tokenizer from the completed motif census without rescanning the
   SDF for each `PYTHONHASHSEED`.
7. Copy only complete verified shards and reports to `autodl-fs`; recompute the
   destination hashes and global release root.
8. On the 4090 instance, copy the admitted release to its `autodl-tmp` and run
   Dataset -> Collator -> CE/MSE -> forward/backward -> save/reload gates.

## CPU sizing and observed fit

- Proven configuration: 96 visible vCPU, 180 GiB RAM, 64 workers, and a 50 GiB
  fast disk with about 51.6 GB initially available.
- The completed release occupied 34,918,330,652 apparent bytes (about 32.52
  GiB), leaving about 16 GB available on the fast disk.  For an identical
  rerun, use at least 55--60 GB free; 80 GB is the safer operational target
  when retaining additional benchmarks or failed attempts.
- A 48--64 vCPU / 128 GiB machine remains a reasonable cost-saving candidate,
  but its full-pass time must be inferred from a real 10k gate on that exact
  host rather than assumed from core count.
- Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` per worker.
- Scale 8 -> 16 -> 32 workers first. More cores are justified only if the 10k
  benchmark shows useful scaling.

## Observed compatibility baseline

The previous working environment reported:

- CPython 3.8.20
- NumPy 1.24.4
- SciPy 1.10.1
- RDKit 2024.03.5
- LMDB 1.7.5
- mmh3 4.1.0
- smart-open 7.3.0.post1
- sdaxen-python-utilities 0.1.5
- vendored E3FP 1.2.5 source
- Torch 2.1.0+cu118 only for the legacy split artifact/model environment

The CPU production environment was independently attested with CPython 3.8.20,
NumPy 1.24.4, SciPy 1.10.1, RDKit 2024.03.5, python-lmdb 1.7.5, liblmdb
0.9.33, and the content-addressed vendored E3FP source. Torch was not used by
the CPU data pass. Any new instance must still produce its own runtime
attestation and pass the 128-record comparison; this paragraph is evidence,
not permission to reuse an unverified environment.

## Observed time budget

- 10k gate at 64 workers: 9.9 s including input revalidation, about 1,010
  records/s.
- Full 3,378,606-record production pass: 46 min 27 s.
- CPU-source independent audit: 238.6 s.
- final incremental cross-region sync: 38 s.
- retained-region independent audit on ordinary `autodl-fs`: 442.5 s.
- The complete CPU setup/gating/production/audit effort finished in the same
  working session, comfortably below one day.

For an already staged and tested repeat, reserve 2--4 hours for the full pass,
both audits, sync variance, and one non-destructive recovery attempt.  A
one-day allocation is therefore conservative for this data-release step on the
proven host.  This estimate does not include tokenizer/overlap decisions or the
4090 P1 functional gate.
