#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
LABEL_DIR="${2:-/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels}"
TS="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${OFFLINE_DIR}/configs/tcformer_otta/tcformer_otta_no_energy.yaml"

PYTHON_EXEC="${PYTHON_EXEC:-python3}"
RESULTS_DIR="results/energy_ablation_s0_${TS}"

cd "${OFFLINE_DIR}"

echo "========================================"
echo "BCIC2a pure Energy-gate ablation"
echo "GPU_ID      : ${GPU_ID}"
echo "LABEL_DIR   : ${LABEL_DIR}"
echo "CONFIG      : ${CONFIG_PATH}"
echo "RESULTS_DIR : ${RESULTS_DIR}"
echo "PYTHON_EXEC : ${PYTHON_EXEC}"
echo "========================================"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${LABEL_DIR}" ]]; then
  echo "ERROR: LABEL_DIR not found: ${LABEL_DIR}" >&2
  exit 1
fi

"${PYTHON_EXEC}" train_pipeline.py \
  --model tcformer_otta \
  --config "${CONFIG_PATH}" \
  --dataset bcic2a \
  --gpu_id "${GPU_ID}" \
  --seed 0 \
  --bcic2a_eval_label_path "${LABEL_DIR}" \
  --results_dir "${RESULTS_DIR}"

echo ""
echo "Done. Check:"
echo "  ${OFFLINE_DIR}/${RESULTS_DIR}/results.txt"
echo "  ${OFFLINE_DIR}/${RESULTS_DIR}/config.yaml"
