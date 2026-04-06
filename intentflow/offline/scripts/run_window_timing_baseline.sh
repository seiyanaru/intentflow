#!/bin/bash
# Window Timing Baseline Experiment for TCFormer
# - Window timing: 0.5-4.5s (align with paper)
# - EarlyStopping removed (train for full 1000 epochs)
# - Validation properly split from training data
# - Expected improvement: 82.79% → 84.5~85%

set -e  # Exit on error

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Configuration
MODEL="tcformer"
DATASET="bcic2a"
GPU_ID=0
SEED=0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE="${OFFLINE_DIR}/results"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="window_timing_baseline_s${SEED}_${TIMESTAMP}"
RESULTS_DIR="${RESULTS_BASE}/${EXPERIMENT_NAME}"
LOG_FILE="${RESULTS_BASE}/run_${EXPERIMENT_NAME}.log"

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "Window Timing Baseline Experiment"
echo "=========================================="
echo "Model:        ${MODEL}"
echo "Dataset:      ${DATASET}"
echo "Window:       0.5-4.5s (paper-aligned)"
echo "Seed:         ${SEED}"
echo "GPU:          ${GPU_ID}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Log file:     ${LOG_FILE}"
echo "=========================================="
echo ""
echo "Starting training..."
echo ""

# Run training
cd "${OFFLINE_DIR}"
python3 train_pipeline.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  --results_dir "${RESULTS_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo "Results saved to: ${RESULTS_DIR}"
echo "Log saved to:     ${LOG_FILE}"
echo ""

# Display summary
if [ -f "${RESULTS_DIR}/results.txt" ]; then
    echo "=== Summary ==="
    tail -20 "${RESULTS_DIR}/results.txt"
fi
