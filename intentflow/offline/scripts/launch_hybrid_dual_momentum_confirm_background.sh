#!/bin/bash
# Launch the focused dual-momentum confirmation in the background with nohup.

set -euo pipefail

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
SEEDS_CSV="${3:-0,1,2,3,4}"
HOURS="${4:-5}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

OFFLINE_DIR="/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${OFFLINE_DIR}/results/hybrid_dual_momentum_confirm_${TIMESTAMP}"
LOG_FILE="${OFFLINE_DIR}/results/launch_hybrid_dual_momentum_confirm_${TIMESTAMP}.log"

mkdir -p "${OFFLINE_DIR}/results"

echo "Launching hybrid dual-momentum confirmation in background"
echo "Source: ${SOURCE_DIR}"
echo "GPU: ${GPU_ID}"
echo "Seeds: ${SEEDS_CSV}"
echo "Hours: ${HOURS}"
echo "Results: ${RESULTS_DIR}"
echo "Log: ${LOG_FILE}"

cd "${OFFLINE_DIR}"
nohup bash ./scripts/run_hybrid_dual_momentum_confirm.sh \
    "${SOURCE_DIR}" \
    "${GPU_ID}" \
    "${SEEDS_CSV}" \
    "${HOURS}" \
    "${RESULTS_DIR}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > /tmp/intentflow_hybrid_dual_momentum_confirm.pid

echo ""
echo "Started with PID: ${PID}"
echo "Monitor:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check:"
echo "  ps -p ${PID}"
echo ""
echo "Stop:"
echo "  kill ${PID}"
