#!/bin/bash
# Run clean baseline experiment in the background with proper logging

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Configuration
EXPERIMENT_NAME="clean_baseline_tcformer_s0_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/${EXPERIMENT_NAME}"
LOG_FILE="/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/run_${EXPERIMENT_NAME}.log"

echo "Starting experiment: ${EXPERIMENT_NAME}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Run in background with nohup
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline
nohup python3 train_pipeline.py \
  --model tcformer \
  --dataset bcic2a \
  --seed 0 \
  --gpu_id 0 \
  --results_dir "${RESULTS_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Experiment started with PID: ${PID}"
echo "${PID}" > /tmp/intentflow_experiment.pid
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check if still running:"
echo "  ps -p ${PID}"
echo ""
echo "To kill if needed:"
echo "  kill ${PID}"
