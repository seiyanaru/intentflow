#!/bin/bash
# =============================================================
# Update Operator Comparison v2 — train-once / test-many
# =============================================================
# Phase 1: Train source model (1000ep) and save checkpoints
# Phase 2: Test 5 OTTA conditions using shared checkpoints
# =============================================================

cd "$(dirname "$0")/.." || exit 1

GPU_ID=0
SEED=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/update_op_v2_${TIMESTAMP}"
SOURCE_DIR="${BASE_DIR}/source_model"
CONFIG="configs/tcformer_otta/tcformer_otta.yaml"

echo "================================================="
echo "Update Operator Comparison v2"
echo "Base dir: ${BASE_DIR}"
echo "GPU: ${GPU_ID}, Seed: ${SEED}"
echo "================================================="

# ----------------------------------------------------------
# Phase 1: Train source model (1000ep, save checkpoints)
# ----------------------------------------------------------
echo ""
echo "[Phase 1] Training source model (1000ep)..."
echo "Start: $(date)"

timeout 14400 python3 train_pipeline.py \
    --model tcformer_otta \
    --dataset bcic2a \
    --seed "${SEED}" \
    --gpu_id "${GPU_ID}" \
    --config "${CONFIG}" \
    --results_dir "${SOURCE_DIR}" \
    --model_kwargs '{"enable_otta": false}'

if [ $? -ne 0 ]; then
    echo "[FATAL] Phase 1 failed. Aborting."
    exit 1
fi
echo "[Phase 1] Source model training complete: $(date)"

# Verify checkpoints exist
CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
echo "[Phase 1] Checkpoints saved: ${CKPT_COUNT}"
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Expected 9 checkpoints, got ${CKPT_COUNT}. Aborting."
    exit 1
fi

# ----------------------------------------------------------
# Phase 2: Test 5 OTTA conditions (test-only, shared ckpts)
# ----------------------------------------------------------
CONDITIONS=(source_only bn_stat bn_stat_clean tent_bn tent_bn_ln)

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo "================================================="
    echo "[Phase 2] Condition: ${COND}"
    echo "Start: $(date)"
    echo "================================================="

    RESULT_DIR="${BASE_DIR}/${COND}"

    if [ "${COND}" = "source_only" ]; then
        EXTRA_KWARGS='{"adapt_mode": "source_only", "enable_otta": false}'
    else
        EXTRA_KWARGS="{\"adapt_mode\": \"${COND}\"}"
    fi

    timeout 3600 python3 train_pipeline.py \
        --model tcformer_otta \
        --dataset bcic2a \
        --seed "${SEED}" \
        --gpu_id "${GPU_ID}" \
        --config "${CONFIG}" \
        --results_dir "${RESULT_DIR}" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs "${EXTRA_KWARGS}" \
        2>&1 | tee "${BASE_DIR}/${COND}.log"

    if [ $? -eq 0 ]; then
        echo "=== ${COND} COMPLETED ==="
    else
        echo "=== ${COND} FAILED (exit=$?) ==="
    fi
done

echo ""
echo "================================================="
echo "All conditions finished: $(date)"
echo "Results: ${BASE_DIR}"
echo "================================================="

# Print summary
echo ""
echo "=== SUMMARY ==="
for COND in "${CONDITIONS[@]}"; do
    echo "--- ${COND} ---"
    tail -5 "${BASE_DIR}/${COND}/results.txt" 2>/dev/null || echo "NO RESULTS"
    echo ""
done
