#!/bin/bash
# Window Timing Full Experiment (Baseline + OTTA)
# - Window timing: 0.5-4.5s (align with paper)
# - Phase 1: Baseline training (~3 hours)
# - Phase 2: OTTA training (~3 hours)
# - Total: ~6 hours

set -e  # Exit on error

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Configuration
DATASET="bcic2a"
GPU_ID=0
SEED=0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE="${OFFLINE_DIR}/results"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_BASE="window_timing_full_s${SEED}_${TIMESTAMP}"

echo "=========================================="
echo "Window Timing Full Experiment"
echo "=========================================="
echo "Window:       0.5-4.5s (paper-aligned)"
echo "Seed:         ${SEED}"
echo "GPU:          ${GPU_ID}"
echo "Timestamp:    ${TIMESTAMP}"
echo "=========================================="
echo ""

#############################################
# Phase 1: Baseline Training
#############################################
echo "=========================================="
echo "Phase 1: Baseline Training"
echo "=========================================="

BASELINE_NAME="${EXPERIMENT_BASE}_baseline"
BASELINE_DIR="${RESULTS_BASE}/${BASELINE_NAME}"
BASELINE_LOG="${RESULTS_BASE}/run_${BASELINE_NAME}.log"

mkdir -p "${BASELINE_DIR}"

echo "Model:        tcformer"
echo "Results dir:  ${BASELINE_DIR}"
echo "Log file:     ${BASELINE_LOG}"
echo ""
echo "Starting baseline training..."
echo ""

cd "${OFFLINE_DIR}"
python3 train_pipeline.py \
  --model tcformer \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  --results_dir "${BASELINE_DIR}" \
  2>&1 | tee "${BASELINE_LOG}"

echo ""
echo "Phase 1 completed!"
echo ""

# Display baseline summary
if [ -f "${BASELINE_DIR}/results.txt" ]; then
    echo "=== Baseline Summary ==="
    tail -20 "${BASELINE_DIR}/results.txt"
    echo ""
fi

#############################################
# Phase 2: OTTA Training
#############################################
echo "=========================================="
echo "Phase 2: OTTA Training"
echo "=========================================="

OTTA_NAME="${EXPERIMENT_BASE}_otta"
OTTA_DIR="${RESULTS_BASE}/${OTTA_NAME}"
OTTA_LOG="${RESULTS_BASE}/run_${OTTA_NAME}.log"

mkdir -p "${OTTA_DIR}"

echo "Model:        tcformer_otta"
echo "Results dir:  ${OTTA_DIR}"
echo "Log file:     ${OTTA_LOG}"
echo ""
echo "Starting OTTA training..."
echo ""

cd "${OFFLINE_DIR}"
python3 train_pipeline.py \
  --model tcformer_otta \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --gpu_id "${GPU_ID}" \
  --results_dir "${OTTA_DIR}" \
  2>&1 | tee "${OTTA_LOG}"

echo ""
echo "Phase 2 completed!"
echo ""

# Display OTTA summary
if [ -f "${OTTA_DIR}/results.txt" ]; then
    echo "=== OTTA Summary ==="
    tail -20 "${OTTA_DIR}/results.txt"
    echo ""
fi

#############################################
# Final Summary
#############################################
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Baseline results: ${BASELINE_DIR}"
echo "OTTA results:     ${OTTA_DIR}"
echo ""

# Extract and compare results
if [ -f "${BASELINE_DIR}/results.txt" ] && [ -f "${OTTA_DIR}/results.txt" ]; then
    echo "=== Comparison ==="
    echo ""
    echo "--- Baseline (0.5-4.5s window) ---"
    grep -E "Average|Mean" "${BASELINE_DIR}/results.txt" | tail -5 || echo "No average found"
    echo ""
    echo "--- OTTA (0.5-4.5s window) ---"
    grep -E "Average|Mean" "${OTTA_DIR}/results.txt" | tail -5 || echo "No average found"
    echo ""
fi

echo "Logs:"
echo "  Baseline: ${BASELINE_LOG}"
echo "  OTTA:     ${OTTA_LOG}"
echo ""
echo "Experiment finished at: $(date)"
echo "=========================================="
