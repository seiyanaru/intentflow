#!/usr/bin/env bash
set -euo pipefail

# Gap-analysis batch runner for BCIC2a (T->E with true labels)
# Exp1: TCFormer baseline (seed=0, no early stopping)
# Exp2: TCFormer baseline 5-seed (no early stopping)
# Exp3: TCFormer-OTTA with relaxed thresholds

GPU_ID="${1:-0}"
LABEL_DIR="${2:-/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels}"
TS="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PY="${OFFLINE_DIR}/train_pipeline.py"
PYTHON_EXEC="/home/islab-shi/anaconda3/envs/intentflow/bin/python"

cd "${OFFLINE_DIR}"

echo "========================================"
echo "BCIC2a gap-analysis experiments"
echo "GPU_ID      : ${GPU_ID}"
echo "LABEL_DIR   : ${LABEL_DIR}"
echo "TIMESTAMP   : ${TS}"
echo "========================================"

if [[ ! -d "${LABEL_DIR}" ]]; then
  echo "ERROR: LABEL_DIR not found: ${LABEL_DIR}" >&2
  exit 1
fi

# Safety check: this script assumes EarlyStopping is already removed.
if rg -n "EarlyStopping|early_stopping" "${TRAIN_PY}" >/dev/null 2>&1; then
  echo "ERROR: train_pipeline.py still contains EarlyStopping-related code." >&2
  echo "Please remove EarlyStopping first, then rerun this script." >&2
  echo "File: ${TRAIN_PY}" >&2
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

# Exp1: single seed baseline
run_exp \
  "exp1_no_earlystop_s0" \
  --model tcformer \
  --dataset bcic2a \
  --gpu_id "${GPU_ID}" \
  --seed 0 \
  --bcic2a_eval_label_path "${LABEL_DIR}"

# Exp2: 5-seed baseline
for s in 0 1 2 3 4; do
  run_exp \
    "exp2_no_earlystop_5seed_s${s}" \
    --model tcformer \
    --dataset bcic2a \
    --gpu_id "${GPU_ID}" \
    --seed "${s}" \
    --bcic2a_eval_label_path "${LABEL_DIR}"
done

# Exp3: OTTA threshold tuning
run_exp \
  "exp3_otta_relaxed_thresh_s0" \
  --model tcformer_otta \
  --dataset bcic2a \
  --gpu_id "${GPU_ID}" \
  --seed 0 \
  --bcic2a_eval_label_path "${LABEL_DIR}" \
  --model_kwargs '{"pmax_threshold":0.6,"sal_threshold":0.3}'

echo ""
echo "========================================"
echo "Done. Check:"
echo "  ${OFFLINE_DIR}/results/*_${TS}/results.txt"
echo "Logs:"
echo "  ${OFFLINE_DIR}/results/run_*_${TS}.log"
echo "========================================"
