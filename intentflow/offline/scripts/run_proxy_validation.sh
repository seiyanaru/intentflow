#!/usr/bin/env bash
set -euo pipefail

# Phase-A proxy validation runner
# Usage:
#   ./scripts/run_proxy_validation.sh \
#       results/update_op_v2_20260401_125734/source_model \
#       0 \
#       1,2,3,4,5,6,7,8,9 \
#       0.05,0.1,0.2

SOURCE_MODEL_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
SUBJECT_IDS="${3:-1,2,3,4,5,6,7,8,9}"
GAIN_STDS="${4:-0.05,0.1,0.2}"
MODEL_NAME="${5:-tcformer}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OFFLINE_DIR}"

CONFIG_PATH="${OFFLINE_DIR}/configs/tcformer_otta/tcformer_otta_bs1.yaml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OFFLINE_DIR}/results/proxy_validation_phaseA_${TIMESTAMP}"
mkdir -p "${OUTDIR}"
LOGFILE="${OUTDIR}/run.log"

echo "=================================================" | tee -a "${LOGFILE}"
echo "Phase-A Proxy Validation Run" | tee -a "${LOGFILE}"
echo "Source checkpoints : ${SOURCE_MODEL_DIR}" | tee -a "${LOGFILE}"
echo "Config            : ${CONFIG_PATH}" | tee -a "${LOGFILE}"
echo "Subjects          : ${SUBJECT_IDS}" | tee -a "${LOGFILE}"
echo "Gain stds         : ${GAIN_STDS}" | tee -a "${LOGFILE}"
echo "Model             : ${MODEL_NAME}" | tee -a "${LOGFILE}"
echo "GPU               : ${GPU_ID}" | tee -a "${LOGFILE}"
echo "Output            : ${OUTDIR}" | tee -a "${LOGFILE}"
echo "Start             : $(date)" | tee -a "${LOGFILE}"
echo "=================================================" | tee -a "${LOGFILE}"

python3 "${OFFLINE_DIR}/utils/bn_stat_collector.py" \
  --config "${CONFIG_PATH}" \
  --checkpoint_dir "${SOURCE_MODEL_DIR}" \
  --dataset "bcic2a" \
  --model "${MODEL_NAME}" \
  --subject_ids "${SUBJECT_IDS}" \
  --gain_stds "${GAIN_STDS}" \
  --gpu_id "${GPU_ID}" \
  --batch_size 64 \
  --num_workers 0 \
  --seed 0 \
  --output_dir "${OUTDIR}" \
  2>&1 | tee -a "${LOGFILE}"

echo "End               : $(date)" | tee -a "${LOGFILE}"
echo "Done. Results at  : ${OUTDIR}" | tee -a "${LOGFILE}"
