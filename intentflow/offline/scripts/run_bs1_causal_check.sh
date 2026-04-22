#!/bin/bash
# =============================================================
# BS=1 Causal Investigation
# =============================================================
# Goal: determine whether S2 degradation is caused by
#       batch-shared BN stats (artifact) or genuine gating failure.
#
# Design (minimal):
#   - source checkpoint reused from existing run (no retraining)
#   - 3 conditions: source_only, bn_stat_clean, tent_bn
#   - test_batch_size=1 (online simulation, no batch stat mixing)
#
# Interpretation:
#   S2 recovers significantly    -> prior degradation was evaluation artifact
#   S2 still degrades at bs=1   -> gating passes harmful updates
#
# Usage:
#   ./scripts/run_bs1_causal_check.sh [SOURCE_CKPT_DIR] [GPU_ID]
#
# Example:
#   ./scripts/run_bs1_causal_check.sh \
#       results/update_op_v2_20260401_125734/source_model 0
# =============================================================

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/bs1_causal_${TIMESTAMP}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1.yaml"

echo "================================================="
echo "BS=1 Causal Check"
echo "Source checkpoints: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "================================================="

# Verify source checkpoints exist
CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 subject checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi
echo "[OK] Found ${CKPT_COUNT} checkpoints."

# Conditions to run (source_only as reference, bn_stat_clean as core, tent_bn for gradient check)
CONDITIONS=(source_only bn_stat_clean tent_bn)

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo "================================================="
    echo "Condition: ${COND}  (bs=1)"
    echo "Start: $(date)"
    echo "================================================="

    RESULT_DIR="${BASE_DIR}/${COND}"

    if [ "${COND}" = "source_only" ]; then
        EXTRA_KWARGS='{"adapt_mode": "source_only", "enable_otta": false}'
    else
        EXTRA_KWARGS="{\"adapt_mode\": \"${COND}\"}"
    fi

    timeout 1800 python3 train_pipeline.py \
        --model tcformer_otta \
        --dataset bcic2a \
        --gpu_id "${GPU_ID}" \
        --config "${CONFIG}" \
        --results_dir "${RESULT_DIR}" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs "${EXTRA_KWARGS}" \
        2>&1 | tee "${BASE_DIR}_${COND}.log"

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "=== ${COND} COMPLETED ==="
    else
        echo "=== ${COND} FAILED (exit=${EXIT_CODE}) ==="
    fi
done

echo ""
echo "================================================="
echo "All conditions finished: $(date)"
echo "================================================="

echo ""
echo "=== SUMMARY ==="
for COND in "${CONDITIONS[@]}"; do
    echo "--- ${COND} (bs=1) ---"
    tail -6 "${BASE_DIR}/${COND}/results.txt" 2>/dev/null || echo "NO RESULTS"
    echo ""
done

echo "Key: compare S2 accuracy between bs=48 (prior) and bs=1 (this run)."
echo "Prior bs=48: source_only=67.71%, bn_stat_clean=59.38%"
