# R1 PCQM E3FP bounded preflight

`gates/pcqm_e3fp_preflight.py` is a release diagnostic only. It does not
extract the archive, create an LMDB, write molecule records, or use a full
corpus mode. It streams at most 1,000 records (128 by default), projects
explicit hydrogen using only `removeDefiningBondStereo=True`, and calculates
E3FP from that same conformer-bearing RDKit Mol.

Before `RemoveHs`, every source atom receives `_r1_source_atom_index`. The
gate rejects a record if a surviving tag is missing, noninteger, duplicated,
out of source range, or no longer source-ordered. This is deliberate: a later
adapter must preserve the geometry-mol-to-source mapping rather than infer it
from compacted RDKit indices.

The locked historical P1 E3FP invocation is `bits=4096`, `level=3`,
`rdkit_invariants=True`, `all_iters=True`, and `exclude_floating=False`.
Unlike the old bulk script, the `[N, 4]` matrix is populated using each
shell's declared `center_atom` and `radius`, never by list ordering.

Run this only on the remote machine. First verify the harness itself with an
in-memory deterministic molecule:

```bash
/root/miniconda3/envs/3dmolt5/bin/python -B \
  /root/autodl-fs/most-t5-r1/harness/gates/pcqm_e3fp_preflight.py \
  --self-test \
  --e3fp-source /root/autodl-tmp/MoSt-T5/tokenization/3d_tokenization \
  --output /root/autodl-fs/most-t5-r1/reports/r1-pcqm-source-20260731/e3fp_preflight_selftest.json
```

Then run the default 128-record archive preflight:

```bash
/root/miniconda3/envs/3dmolt5/bin/python -B \
  /root/autodl-fs/most-t5-r1/harness/gates/pcqm_e3fp_preflight.py \
  --archive /root/autodl-fs/most-t5-p0/sources/pcqm4mv2/ogb-pcqm4mv2-train-3d-v1/archive/pcqm4m-v2-train.sdf.tar.gz \
  --e3fp-source /root/autodl-tmp/MoSt-T5/tokenization/3d_tokenization \
  --max-records 128 \
  --output /root/autodl-fs/most-t5-r1/reports/r1-pcqm-source-20260731/e3fp_preflight_128.json
```

The output paths must be new. An all-pass report permits only the design of a
full remote membership/reject-ledger adapter; it is not P1 admission.
