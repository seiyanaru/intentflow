#!/usr/bin/env bash
set -euo pipefail

# BCIC2a 3-way comparison under true T->E generalization
# 1) TCFormer baseline
# 2) TCFormer-OTTA (Pmax+SAL+Energy, Neuro OFF)
# 3) TCFormer-OTTA latest (Dual Gating: Neuro+Energy)

GPU_ID="${1:-0}"
LABEL_DIR="${2:-/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels}"
TS="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXEC="/home/islab-shi/anaconda3/envs/intentflow/bin/python"

cd "${OFFLINE_DIR}"

echo "========================================"
echo "BCIC2a 3-way run (T->E)"
echo "GPU_ID      : ${GPU_ID}"
echo "LABEL_DIR   : ${LABEL_DIR}"
echo "TIMESTAMP   : ${TS}"
echo "========================================"

if [[ ! -d "${LABEL_DIR}" ]]; then
  echo "ERROR: LABEL_DIR not found: ${LABEL_DIR}" >&2
  exit 1
fi

run_exp () {
  local name="$1"
  shift
  local out_dir="results/${name}_${TS}"
  local log_file="results/run_${name}_${TS}.log"
  echo ""
  echo ">>> Running: ${name}"
  echo ">>> out_dir: ${out_dir}"
  ${PYTHON_EXEC} train_pipeline.py "$@" --results_dir "${out_dir}" 2>&1 | tee "${log_file}"
}

# 1) Baseline
run_exp \
  "tcformer_bcic2a_base_evalE" \
  --model tcformer \
  --dataset bcic2a \
  --gpu_id "${GPU_ID}" \
  --bcic2a_eval_label_path "${LABEL_DIR}"

# 2) Pmax+SAL+Energy (Neuro OFF)
run_exp \
  "tcformer_otta_bcic2a_pmax_sal_energy_evalE" \
  --model tcformer_otta \
  --dataset bcic2a \
  --gpu_id "${GPU_ID}" \
  --bcic2a_eval_label_path "${LABEL_DIR}" \
  --model_kwargs '{"neuro_beta":0.0,"strict_tri_lock":true}'

# 3) Latest Dual Gating (Neuro+Energy)
run_exp \
  "tcformer_otta_bcic2a_dualgating_latest_evalE" \
  --model tcformer_otta \
  --dataset bcic2a \
  --gpu_id "${GPU_ID}" \
  --bcic2a_eval_label_path "${LABEL_DIR}" \
  --model_kwargs '{"neuro_beta":0.1,"strict_tri_lock":true}'

echo ""
echo "========================================"
echo "Done. Check:"
echo "  ${OFFLINE_DIR}/results/*_${TS}/results.txt"
echo "========================================"

