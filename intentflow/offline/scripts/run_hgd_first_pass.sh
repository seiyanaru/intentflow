#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OFFLINE_DIR="${PROJECT_ROOT}/intentflow/offline"

CONDA_SH="${CONDA_SH:-/home/islab-shi/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-intentflow}"

EVAL_CONFIG="${1:-configs/tcformer_otta/tcformer_otta_bs1.yaml}"
SOURCE_DIR="${2:-results/paper_experiments/hgd/20260114_213132/base}"
GPU_ID="${3:-0}"
SEED="${4:-0}"
RUN_TAG="${5:-$(date +%Y%m%d_%H%M%S)}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${OFFLINE_DIR}"

if [[ "${GPU_ID}" != "-1" ]]; then
  python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("[hgd-first-pass] ERROR: gpu_id>=0 requested but CUDA is unavailable.", file=sys.stderr)
    sys.exit(1)
PY
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[hgd-first-pass] source dir not found: ${SOURCE_DIR}" >&2
  exit 1
fi

RUN_ROOT="results/phaseC_hgd_firstpass_${RUN_TAG}_seed${SEED}"
mkdir -p "${RUN_ROOT}"

echo "[hgd-first-pass] eval config: ${EVAL_CONFIG}"
echo "[hgd-first-pass] source dir:   ${SOURCE_DIR}"
echo "[hgd-first-pass] gpu_id:       ${GPU_ID}"
echo "[hgd-first-pass] seed:         ${SEED}"
echo "[hgd-first-pass] run root:     ${RUN_ROOT}"

run_one() {
  local tag="$1"
  local kwargs="$2"
  local out_dir="${RUN_ROOT}/${tag}"

  echo
  echo "============================================================"
  echo "[hgd-first-pass] Running ${tag}"
  echo "  results_dir=${out_dir}"
  echo "  model_kwargs=${kwargs}"
  echo "============================================================"

  python train_pipeline.py \
    --model tcformer_otta \
    --dataset hgd \
    --seed "${SEED}" \
    --gpu_id "${GPU_ID}" \
    --config "${EVAL_CONFIG}" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --results_dir "${out_dir}" \
    --model_kwargs "${kwargs}"
}

run_one source_only '{"use_eca": false, "adapt_mode": "source_only", "enable_otta": false}'
run_one vanilla_both '{"use_eca": false, "adapt_mode": "bn_stat_clean", "bn_momentum": 0.1, "bn_update_target": "both"}'
run_one hybrid_mom001 '{"use_eca": false, "adapt_mode": "bn_stat_clean", "bn_momentum": 0.01, "bn_update_target": "shallow_mean_deep_both"}'

echo
echo "[hgd-first-pass] all conditions completed"
echo "[hgd-first-pass] results: ${RUN_ROOT}"
