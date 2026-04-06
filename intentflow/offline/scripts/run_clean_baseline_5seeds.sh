#!/bin/bash
# Clean Baseline Experiment for TCFormer with 5 seeds
# - EarlyStopping removed (train for full 1000 epochs)
# - Validation properly split from training data (no test leakage)
# - Eval labels properly loaded for session_E

set -e  # Exit on error

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Configuration
MODEL="tcformer"
DATASET="bcic2a"
GPU_ID=0
SEEDS=(0 1 2 3 4)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE="${OFFLINE_DIR}/results"

# Generate timestamp (shared across all seeds)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_EXPERIMENT_NAME="clean_baseline_5seeds"

echo "=========================================="
echo "Clean Baseline TCFormer - 5 Seeds"
echo "=========================================="
echo "Model:        ${MODEL}"
echo "Dataset:      ${DATASET}"
echo "Seeds:        ${SEEDS[@]}"
echo "GPU:          ${GPU_ID}"
echo "Base name:    ${BASE_EXPERIMENT_NAME}"
echo "Timestamp:    ${TIMESTAMP}"
echo "=========================================="
echo ""

# Run for each seed
for SEED in "${SEEDS[@]}"; do
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_s${SEED}_${TIMESTAMP}"
    RESULTS_DIR="${RESULTS_BASE}/${EXPERIMENT_NAME}"
    LOG_FILE="${RESULTS_BASE}/run_${EXPERIMENT_NAME}.log"

    echo ""
    echo "=========================================="
    echo "Running Seed ${SEED}..."
    echo "=========================================="
    echo "Results dir: ${RESULTS_DIR}"
    echo "Log file:    ${LOG_FILE}"
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

    # Display summary for this seed
    if [ -f "${RESULTS_DIR}/results.txt" ]; then
        echo ""
        echo "=== Seed ${SEED} Summary ==="
        tail -5 "${RESULTS_DIR}/results.txt"
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""

# Aggregate results
echo "=== Aggregated Results Across Seeds ==="
for SEED in "${SEEDS[@]}"; do
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_s${SEED}_${TIMESTAMP}"
    RESULTS_DIR="${RESULTS_BASE}/${EXPERIMENT_NAME}"
    RESULTS_FILE="${RESULTS_DIR}/results.txt"

    if [ -f "${RESULTS_FILE}" ]; then
        ACC=$(grep "Average Test Accuracy" "${RESULTS_FILE}" | awk '{print $4}')
        echo "Seed ${SEED}: ${ACC}%"
    fi
done
