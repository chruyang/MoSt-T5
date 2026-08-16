# MoSt-T5

MoSt-T5 is a T5 encoder-decoder for molecular structure and scientific text.
It keeps the ordinary T5 interface and adds geometry only when a batch carries
valid E3FP states.  The same model therefore accepts text, fragSMILES, or a
joint text-molecule sequence without task-name-specific routing.

## Pretraining curriculum

The formal recipe has two optimizer phases:

| phase | tasks | source | update ratio |
|---|---|---|---|
| I | `M`, `MG` | `M`: PCQM + PubChem; `MG`: PCQM | 1:1 |
| II | `SYN`, `TXT`, `CAP`, `T2M` | PCQM + PubChem, PubMed, and paired PubChem text | 1:1:1:1 |

Phase II starts from the Phase-I model weights with a fresh optimizer and
learning-rate scheduler.  Task selection occurs at optimizer-update level;
all microbatches accumulated into one update belong to the same task.
The sampler first selects the complete logical batch of 96 records and then
applies the physical partition.  Thus `96 x 1`, `48 x 2`, and `32 x 3` share
one deterministic record stream; `32 x 3` is the current safe baseline.

`CAP` predicts `enriched_description`; `T2M` uses it as input.  The original
`description` field remains the downstream reference used by 3D-MolT5-style
captioning evaluation.

## Input and geometry contract

`most_t5_next.data.MoStT5Processor` and `MoStT5Collator` form the public input
boundary.  They support:

- text-only input;
- molecule-only input;
- text followed by a molecule;
- mixed rows in one batch.

Geometry is represented by the structural tensors in the batch.  An E3FP row
filled with `-1` means that geometry is unavailable for that row; carrier and
endpoint injection then contribute exactly zero.  This is also how `M`, `TXT`,
and `T2M` disable the 3D channel.  `MG`, `SYN`, and `CAP` use geometry when it
is available.  Explicit geometry ablations remain possible through
`geometry_mode`.

Molecular corruption treats one motif as its fragment span plus every explicit
endpoint span owned by that fragment.  Selecting the motif masks that compound
unit, keeps the opposite endpoint visible, and disables the affected carrier
and endpoint geometry.  Endpoints are not sampled again as independent syntax
units.

A `whole_molecule_fallback` row has zero fragments and unowned atoms
(`atom_to_fragment = -1`).  Mixed batches validate this rule per row.  Because
the fallback has no carrier or endpoint destination, its geometry contribution
is exactly zero; inventing a fragment or silently discarding the row is not
permitted.  See [training data contracts](docs/training_contracts.md).

## Length policy

- PubMed `TXT` uses the derived T5 span-corruption contract `568 -> 512/114`.
- `T2M` right-truncates only its text input.
- `CAP` right-truncates only its text target and restores EOS.
- fragSMILES is never token-truncated.  An unsafe structural task view is
  excluded while the source record remains in the cache.

Finite-budget task populations and the exclusion ledger are generated after
the two update budgets are fixed:

```bash
python -m scripts.freeze_pretraining_populations \
  --config most_t5_next/configs/pretrain.yaml \
  --tokenizer-root /path/to/tokenizer \
  --pcqm-cache /path/to/pcqm-cache \
  --pubchem-cache /path/to/pubchem-cache \
  --paired-text-cache /path/to/enriched-description-cache \
  --pubmed-cache /path/to/pubmed-cache \
  --output-dir /path/to/populations \
  --set curriculum.phase_one.total_updates=... \
  --set curriculum.phase_two.total_updates=...
```

Population membership depends on the finite update budgets and effective
batch size, but not on optimizer or scheduler hyperparameters.

## Training

Ordinary research parameters are exposed in
`most_t5_next/configs/pretrain.yaml` and may be overridden with strict dotted
keys.  Unknown keys fail rather than being ignored.

```bash
python -m scripts.pretrain \
  --config most_t5_next/configs/pretrain.yaml \
  --checkpoint /path/to/initialized-union-checkpoint \
  --tokenizer-root /path/to/tokenizer \
  --pcqm-cache /path/to/pcqm-cache \
  --pubchem-cache /path/to/pubchem-cache \
  --paired-text-cache /path/to/enriched-description-cache \
  --pubmed-cache /path/to/pubmed-cache \
  --population-root /path/to/populations \
  --output-dir /root/autodl-tmp/most-t5-pretraining \
  --set curriculum.phase_one.total_updates=... \
  --set curriculum.phase_two.total_updates=... \
  --set optimization.phase_one.base_learning_rate=... \
  --set optimization.phase_two.base_learning_rate=... \
  --set optimization.warmup_start_factor=... \
  --set optimization.final_learning_rate=... \
  --set monitoring.checkpoint_every_updates=...
```

Formal pretraining uses BF16 computation with FP32 parameters and optimizer
state, AdamWScale, 10,000 warmup updates per phase, dynamic padding, online
corruption, and no pretraining validation/evaluation split.  TensorBoard events
are written beneath `/root/tf-logs` by default.

## Repository layout

- `most_t5_next/modeling`: T5 wrapper and geometry adapter;
- `most_t5_next/data`: unified public processor and length policy;
- `most_t5_next/training`: curriculum, optimizer, data provider, population
  freezer, and runner;
- `most_t5_next/p1`, `most_t5_next/p2`, `most_t5_next/r1`: reproducibility
  builders and experiment-specific audits;
- `scripts`: public command-line entry points and smoke checks.

The unresolved update counts and learning rates are intentional: they are the
remaining protocol values to be selected on the final training hardware before
the population ledger and formal run are frozen.
