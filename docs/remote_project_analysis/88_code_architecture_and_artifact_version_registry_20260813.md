# Code, architecture, and artifact version registry (2026-08-13)

> Status: operational source-of-truth policy.  This document separates the
> active 10% architecture screen from historical experiments and from the
> still-pending final fragSMILES training surface.

## 1. Why Git is now mandatory

The local worktree accumulated several independently meaningful generations
of the model (GraphPorts, factorized V2/V3, anchored V4, reducer V5--V9, and
the current V10 parameter-tying screen).  A directory name such as `b0693bb`
is no longer sufficient to identify the code that generated an artifact.

From this checkpoint onward:

1. source code and architecture contracts are versioned by an immutable Git
   commit;
2. a remote training directory must be a clean checkout of that commit;
3. each training manifest records the full 40-character commit and the
   architecture contract ID;
4. datasets, tensor caches, tokenizer snapshots, union-init weights,
   checkpoints, and logs remain outside Git and are linked by their own
   manifests and ordinary path/count metadata;
5. historical model modules remain in Git for reproducibility but are not
   interchangeable with the active entry point.

This is version control, not a new scientific validation layer.  It prevents
mixing implementations without repeatedly re-hashing stable research assets.

## 2. Active architecture for the next GPU decision

Architecture contract:

`most-t5/anchored-v10-e3fp-parameter-tying/20260813`

The active experiment is the six-cell 10% screen implemented by:

- `p2/run_e3fp_parameter_tying_screen_v1.py`;
- `p2/launch_e3fp_parameter_tying_screen_v1.py`;
- `p2/factorized_model_init_v10.py`;
- `p2/factorized_motif_t5_v10.py`;
- `p2/motif_geometry_adapter_v10.py`;
- `p2/e3fp_atom_embedding_v1.py`.

It compares three update-zero-equivalent E3FP parameter-sharing schemes:

| Semantic ID | Tables | Role |
|---|---:|---|
| `reference_shared_fixed4` | 1 | literal 3D-MolT5-style shared lookup and fixed four-slot mean |
| `l0_state_fixed4` | 2 | L0 identity/context table plus shared L1--L3 state table |
| `level_specific_fixed4` | 4 | independent L0, L1, L2, and L3 tables |

Each is crossed with B2D and F3D.  The frozen screen uses carrier-only
injection, width 768, microbatch 32, accumulation 4, and 10,000 updates.
Endpoint selection and a final 3D-sensitive downstream judgment remain later
decisions; identity CE alone does not select the final encoder.

## 3. Data and initialization bound to the active screen

The current runnable screen consumes the anchored interim PF-10 tensor cache:

- records: 336,006;
- train: 302,406;
- dev: 33,600;
- cache schema: `most-t5-p2/anchored-training-tensor-cache-build/v1`.

It also consumes the frozen 33,041-token anchored tokenizer and its matching
union-init checkpoint.  These artifacts are intentionally not committed to
Git.  The launcher binds them by their passed manifests and records that this
cache is an interim architecture-screen surface, not the final Phase-I
three-population cache and not the final fragSMILES training surface.

## 4. Target architecture after the screen

The user-approved target remains:

```text
stereo-free fragSMILES motif phrase
    + ordered connector/anchor structure
    + atom-to-motif and attachment endpoint geometry sidecar
    + E3FP atom states
    -> motif carrier / optional endpoint injection
    -> T5 encoder-decoder
```

The final fragSMILES cache and tokenizer are not silently substituted for the
anchored PF-10 screen.  Their modules are tracked as implementation work, but
formal pretraining begins only after their data contracts are materialized.

## 5. Historical architecture map

| Generation | Meaning | Current status |
|---|---|---|
| GraphPorts/PF1/PF2 | lossless graph language and early direct E3FP fusion | historical evidence; not the model language |
| factorized V2/V3 | staged state/grammar and Motif-FPT5 mechanism probes | historical diagnostics |
| anchored V4 | restored pure motif phrase, ordered anchors, carrier/endpoint route | architectural ancestor |
| V5--V9 | atom reducer and level-fusion alternatives | tested and not promoted |
| anchored V10 | reference/two-table/four-table E3FP parameter-tying screen | active GPU decision |
| final fragSMILES model | compact motif language plus geometry sidecar | target, pending final cache/tokenizer |

Names such as “V4” or “V10” alone must not be used in run directories or
papers.  Use the semantic architecture ID and Git commit together.

## 6. Remote deployment rule

Remote machines must no longer receive an unlabelled copy of `most_t5_next`.
The deployment sequence is:

1. commit the intended source scope locally without including user-staged
   legacy files or datasets;
2. push the research branch;
3. clone/fetch into a new remote directory named with the short commit;
4. checkout the exact commit in detached mode or on the same research branch;
5. run the focused CPU tests;
6. pass the full commit through `--code-commit` when launching the matrix.

The runner rejects a mismatched commit or tracked worktree modifications
before loading a CUDA model.  Old manually copied directories remain historical
snapshots and must not be used for new runs.

