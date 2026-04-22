#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OFFLINE_DIR="${PROJECT_ROOT}/intentflow/offline"

CONDA_SH="${CONDA_SH:-/home/islab-shi/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-intentflow}"

TRAIN_CONFIG="${1:-configs/tcformer_aug_shinv/tcformer_aug_shinv_interaug.yaml}"
EVAL_CONFIG="${2:-configs/tcformer_otta/tcformer_otta_bs1.yaml}"
GPU_ID="${3:-0}"
SEED="${4:-0}"
SUITE_TAG="${5:-$(date +%Y%m%d_%H%M%S)}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${OFFLINE_DIR}"
chmod +x ./scripts/run_phaseB_condition.sh

SUITE_ROOT="results/phaseB_strong_suite_${SUITE_TAG}_seed${SEED}"
mkdir -p "${SUITE_ROOT}"

echo "[PhaseB-strong-suite] train config: ${TRAIN_CONFIG}"
echo "[PhaseB-strong-suite] eval config:  ${EVAL_CONFIG}"
echo "[PhaseB-strong-suite] gpu_id:       ${GPU_ID}"
echo "[PhaseB-strong-suite] seed:         ${SEED}"
echo "[PhaseB-strong-suite] suite root:   ${SUITE_ROOT}"

run_one() {
  local variant="$1"
  local gain_std="$2"
  local lambda_shinv="$3"
  local shinv_mode="$4"
  local lambda_aug="${5:-0.5}"
  local result_root="${SUITE_ROOT}/${variant}"

  echo
  echo "============================================================"
  echo "[PhaseB-strong-suite] Running ${variant}"
  echo "  gain_std=${gain_std}"
  echo "  lambda_shinv=${lambda_shinv}"
  echo "  shinv_mode=${shinv_mode}"
  echo "  lambda_aug=${lambda_aug}"
  echo "  result_root=${result_root}"
  echo "============================================================"

  ./scripts/run_phaseB_condition.sh \
    "${TRAIN_CONFIG}" \
    "${EVAL_CONFIG}" \
    "${GPU_ID}" \
    "${SEED}" \
    "${variant}" \
    "${gain_std}" \
    "${lambda_shinv}" \
    "${result_root}" \
    "${shinv_mode}" \
    "${lambda_aug}"
}

run_one strong_plain 0.0 0.0 mean_logvar 0.5
run_one strong_gain0025 0.025 0.0 mean_logvar 0.5
run_one strong_gain0025_varinv_l002 0.025 0.02 logvar_only 0.5
run_one strong_gain0025_varinv_l005 0.025 0.05 logvar_only 0.5

echo
echo "[PhaseB-strong-suite] all conditions completed"
echo "[PhaseB-strong-suite] results: ${SUITE_ROOT}"
