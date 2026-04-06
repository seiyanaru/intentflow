#!/bin/bash
# Single seed baseline experiment
# Usage: ./run_baseline_single_seed.sh <SEED> [GPU_ID]

set -e

# Arguments
SEED=${1:-0}
GPU_ID=${2:-0}

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Configuration
MODEL="tcformer"
DATASET="bcic2a"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE="${OFFLINE_DIR}/results"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="baseline_seed${SEED}_${TIMESTAMP}"
RESULTS_DIR="${RESULTS_BASE}/${EXPERIMENT_NAME}"
LOG_FILE="${RESULTS_BASE}/${EXPERIMENT_NAME}.log"

echo "=========================================="
echo "Baseline TCFormer - Seed ${SEED}"
echo "=========================================="
echo "Model:        ${MODEL}"
echo "Dataset:      ${DATASET}"
echo "Seed:         ${SEED}"
echo "GPU:          ${GPU_ID}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Log file:     ${LOG_FILE}"
echo "=========================================="
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Run training
cd "${OFFLINE_DIR}"
python3 train_pipeline.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  --results_dir "${RESULTS_DIR}" \
  2>&1 | tee "${LOG_FILE}"

# Display summary
if [ -f "${RESULTS_DIR}/results.txt" ]; then
    echo ""
    echo "=== Experiment Summary ==="
    tail -10 "${RESULTS_DIR}/results.txt"
fi

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
