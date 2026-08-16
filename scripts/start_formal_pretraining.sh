#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_FORMAL_PRETRAINING:-}" != "YES" ]]; then
  echo "Set CONFIRM_FORMAL_PRETRAINING=YES after all CPU and four-GPU gates pass." >&2
  exit 2
fi

PYTHON=${PYTHON:-/root/autodl-tmp/envs/most-t5-blackwell-v1/bin/python}
TORCHRUN=${TORCHRUN:-/root/autodl-tmp/envs/most-t5-blackwell-v1/bin/torchrun}
CODE=${CODE:-/root/autodl-tmp/MoSt-T5-pretraining-ready-20260816}
READY=${READY:-/root/autodl-tmp/pf10-six-view-ready-v1}
CONFIG=${CONFIG:-${CODE}/most_t5_next/configs/pretrain.yaml}
CHECKPOINT=${CHECKPOINT:-${READY}/fragsmiles-union-init-v1/shared_raw_t5}
TOKENIZER=${TOKENIZER:-${READY}/fragsmiles-tokenizer-runtime-candidate-v2}
PCQM_CACHE=${PCQM_CACHE:-${READY}/phase1-fragsmiles-training-cache-v1-r5}
PUBCHEM_CACHE=${PUBCHEM_CACHE:-${READY}/phase2-fragsmiles-training-cache-v5}
PAIRED_TEXT_CACHE=${PAIRED_TEXT_CACHE:-/root/autodl-tmp/phase2-paired-text-enriched-v4}
PUBMED_CACHE=${PUBMED_CACHE:-/root/autodl-tmp/medrag-pubmed-p2-txt-formal-v1}
POPULATION_ROOT=${POPULATION_ROOT:-/root/autodl-tmp/most-t5-formal-populations-v2}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/most-t5-formal-pretraining-v1}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

for path in "${PYTHON}" "${TORCHRUN}" "${CONFIG}" "${CHECKPOINT}" \
  "${TOKENIZER}" "${PCQM_CACHE}" "${PUBCHEM_CACHE}" \
  "${PAIRED_TEXT_CACHE}" "${PUBMED_CACHE}" "${POPULATION_ROOT}"; do
  [[ -e "${path}" ]] || { echo "Missing required asset: ${path}" >&2; exit 3; }
done

cd "${CODE}"
git diff --quiet && git diff --cached --quiet || {
  echo "Formal pretraining requires all tracked files to match the recorded commit." >&2
  exit 3
}

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "${GPU_COUNT}" -eq 4 ]] || {
  echo "Formal pretraining requires exactly four visible GPUs; found ${GPU_COUNT}." >&2
  exit 3
}

"${PYTHON}" - "${CONFIG}" "${POPULATION_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

from most_t5_next.configuration import load_pretraining_config

config = load_pretraining_config(sys.argv[1], require_launch_values=True)
manifest = json.loads((Path(sys.argv[2]) / "manifest.json").read_text())
if manifest.get("schema_version") != "most-t5/pretraining-populations/v2":
    raise SystemExit("formal population schema must be v2")
if manifest.get("status") != "pass" or manifest.get("training_admission") is not True:
    raise SystemExit("formal population is not admitted")
expected_updates = {
    "phase_one": config["curriculum"]["phase_one"]["total_updates"],
    "phase_two": config["curriculum"]["phase_two"]["total_updates"],
}
if manifest.get("phase_updates") != expected_updates:
    raise SystemExit("formal population update budget differs from the config")
batching = manifest.get("batching", {})
if batching.get("rank_local_effective_batch_size") != 96:
    raise SystemExit("rank-local population batch must be 96")
if batching.get("global_effective_batch_size") != 384:
    raise SystemExit("global population batch must be 384")
if batching.get("task_rank_counts") != {
    "CAP": 1, "M": 2, "MG": 2, "SYN": 1, "T2M": 1, "TXT": 1
}:
    raise SystemExit("population rank multiplicities differ from the curriculum")
for task in ("M", "MG", "SYN", "TXT", "CAP", "T2M"):
    descriptor = manifest.get("arrays", {}).get(task, {})
    path = Path(sys.argv[2]) / str(descriptor.get("file", ""))
    if not path.is_file() or int(descriptor.get("shape", [0])[0]) <= 0:
        raise SystemExit(f"population array is invalid: {task}")
PY

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  [[ -d "${OUTPUT_DIR}" ]] || { echo "Resume output directory is missing." >&2; exit 3; }
  [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Resume checkpoint is missing." >&2; exit 3; }
  RESUME_ARGS=(--resume-checkpoint "${RESUME_CHECKPOINT}")
else
  [[ ! -e "${OUTPUT_DIR}" ]] || { echo "Training output already exists." >&2; exit 3; }
  RESUME_ARGS=()
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
exec "${TORCHRUN}" --standalone --nproc-per-node=4 -m scripts.pretrain \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --tokenizer-root "${TOKENIZER}" \
  --pcqm-cache "${PCQM_CACHE}" \
  --pubchem-cache "${PUBCHEM_CACHE}" \
  --paired-text-cache "${PAIRED_TEXT_CACHE}" \
  --pubmed-cache "${PUBMED_CACHE}" \
  --population-root "${POPULATION_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}"
