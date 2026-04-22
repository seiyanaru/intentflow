#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OFFLINE_DIR="${PROJECT_ROOT}/intentflow/offline"

CONDA_SH="${CONDA_SH:-/home/islab-shi/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-intentflow}"

TRAIN_CONFIG="${1:-configs/tcformer_aug_shinv/tcformer_aug_shinv.yaml}"
EVAL_CONFIG="${2:-configs/tcformer_otta/tcformer_otta_bs1.yaml}"
GPU_ID="${3:-0}"
SEED="${4:-0}"
SUITE_TAG="${5:-$(date +%Y%m%d_%H%M%S)}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${OFFLINE_DIR}"

chmod +x ./scripts/run_phaseB_condition.sh

SUITE_ROOT="results/phaseB_suite_${SUITE_TAG}_seed${SEED}"
mkdir -p "${SUITE_ROOT}"

echo "[PhaseB-suite] train config: ${TRAIN_CONFIG}"
echo "[PhaseB-suite] eval config:  ${EVAL_CONFIG}"
echo "[PhaseB-suite] gpu_id:       ${GPU_ID}"
echo "[PhaseB-suite] seed:         ${SEED}"
echo "[PhaseB-suite] suite root:   ${SUITE_ROOT}"

run_one() {
  local variant="$1"
  local gain_std="$2"
  local lambda_shinv="$3"
  local result_root="${SUITE_ROOT}/${variant}"

  echo
  echo "============================================================"
  echo "[PhaseB-suite] Running ${variant}"
  echo "  gain_std=${gain_std}"
  echo "  lambda_shinv=${lambda_shinv}"
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
    "${result_root}"
}

run_one plain 0.0 0.0
run_one aug_only_0025 0.025 0.0
run_one aug_only_005 0.05 0.0
run_one aug_shinv_0025 0.025 0.1
run_one aug_shinv_005 0.05 0.1

echo
echo "[PhaseB-suite] all conditions completed"
echo "[PhaseB-suite] results: ${SUITE_ROOT}"
