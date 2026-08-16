#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/autodl-tmp/envs/most-t5-blackwell-v1/bin/python}
CODE=${CODE:-/root/autodl-tmp/MoSt-T5-pretraining-ready-20260816}
READY=${READY:-/root/autodl-tmp/pf10-six-view-ready-v1}
CONFIG=${CONFIG:-${CODE}/most_t5_next/configs/pretrain.yaml}
TOKENIZER=${TOKENIZER:-${READY}/fragsmiles-tokenizer-runtime-candidate-v2}
PCQM_CACHE=${PCQM_CACHE:-${READY}/phase1-fragsmiles-training-cache-v1-r5}
PUBCHEM_CACHE=${PUBCHEM_CACHE:-${READY}/phase2-fragsmiles-training-cache-v5}
PAIRED_TEXT_CACHE=${PAIRED_TEXT_CACHE:-/root/autodl-tmp/phase2-paired-text-enriched-v4}
PUBMED_CACHE=${PUBMED_CACHE:-/root/autodl-tmp/medrag-pubmed-p2-txt-formal-v1}
POPULATION_ROOT=${POPULATION_ROOT:-/root/autodl-tmp/most-t5-formal-populations-v3}
WORKERS=${WORKERS:-24}
CHUNKSIZE=${CHUNKSIZE:-64}

for path in "${PYTHON}" "${CONFIG}" "${TOKENIZER}" "${PCQM_CACHE}" \
  "${PUBCHEM_CACHE}" "${PAIRED_TEXT_CACHE}" "${PUBMED_CACHE}"; do
  [[ -e "${path}" ]] || { echo "Missing required asset: ${path}" >&2; exit 3; }
done
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "WORKERS must be positive" >&2; exit 3; }
[[ ! -e "${POPULATION_ROOT}" ]] || {
  echo "Population output already exists: ${POPULATION_ROOT}" >&2
  exit 3
}
[[ ! -e "${POPULATION_ROOT}.staging" ]] || {
  echo "Population staging output already exists: ${POPULATION_ROOT}.staging" >&2
  exit 3
}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
cd "${CODE}"
exec "${PYTHON}" -m scripts.freeze_pretraining_populations \
  --config "${CONFIG}" \
  --tokenizer-root "${TOKENIZER}" \
  --pcqm-cache "${PCQM_CACHE}" \
  --pubchem-cache "${PUBCHEM_CACHE}" \
  --paired-text-cache "${PAIRED_TEXT_CACHE}" \
  --pubmed-cache "${PUBMED_CACHE}" \
  --output-dir "${POPULATION_ROOT}" \
  --workers "${WORKERS}" \
  --chunksize "${CHUNKSIZE}"
