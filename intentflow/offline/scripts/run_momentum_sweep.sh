#!/bin/bash
# =============================================================
# BN Momentum Sweep — adaptation strength causal investigation
# =============================================================
# Goal: determine whether S2 harm / S7 gain / S9 drift-collapse
#       are monotone in bn_momentum.
#
# Design:
#   - Subject: S2 (worst harm), S7 (best gain), S9 (feedback collapse)
#   - bn_momentum: {0.1, 0.01, 0.001}
#   - adapt_mode: bn_stat_clean (pure BN stats update, no affine/dropout noise)
#   - bs=1 (online simulation)
#   - Source checkpoint reused (no retraining)
#
# Interpretation:
#   If harm/gain is monotone in momentum -> adaptation *strength* is the key axis
#   If not monotone                      -> direction/layer selection is dominant
#
# Usage:
#   ./scripts/run_momentum_sweep.sh [SOURCE_CKPT_DIR] [GPU_ID]
# =============================================================

cd "$(dirname "$0")/.." || exit 1

SOURCE_DIR="${1:-results/update_op_v2_20260401_125734/source_model}"
GPU_ID="${2:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/momentum_sweep_${TIMESTAMP}"
CONFIG="configs/tcformer_otta/tcformer_otta_bs1_s279.yaml"

echo "================================================="
echo "BN Momentum Sweep (S2/S7/S9, bs=1)"
echo "Source: ${SOURCE_DIR}"
echo "Output: ${BASE_DIR}"
echo "GPU: ${GPU_ID}"
echo "================================================="

CKPT_COUNT=$(ls "${SOURCE_DIR}/checkpoints/"*.ckpt 2>/dev/null | wc -l)
if [ "${CKPT_COUNT}" -lt 9 ]; then
    echo "[FATAL] Need 9 checkpoints in ${SOURCE_DIR}/checkpoints/, found ${CKPT_COUNT}"
    exit 1
fi

# source_only reference (no momentum dependence)
echo ""
echo "--- source_only (reference) ---"
timeout 600 python3 train_pipeline.py \
    --model tcformer_otta --dataset bcic2a \
    --gpu_id "${GPU_ID}" --config "${CONFIG}" \
    --results_dir "${BASE_DIR}/source_only" \
    --checkpoint_dir "${SOURCE_DIR}" \
    --model_kwargs '{"adapt_mode": "source_only", "enable_otta": false}' \
    2>&1 | tee "${BASE_DIR}_source_only.log"

# bn_stat_clean across 3 momentum values
for MOM in 0.1 0.01 0.001; do
    TAG="bn_clean_mom${MOM}"
    echo ""
    echo "================================================="
    echo "bn_stat_clean, bn_momentum=${MOM}"
    echo "Start: $(date)"
    echo "================================================="

    timeout 600 python3 train_pipeline.py \
        --model tcformer_otta --dataset bcic2a \
        --gpu_id "${GPU_ID}" --config "${CONFIG}" \
        --results_dir "${BASE_DIR}/${TAG}" \
        --checkpoint_dir "${SOURCE_DIR}" \
        --model_kwargs "{\"adapt_mode\": \"bn_stat_clean\", \"bn_momentum\": ${MOM}}" \
        2>&1 | tee "${BASE_DIR}_${TAG}.log"

    if [ $? -eq 0 ]; then
        echo "=== ${TAG} COMPLETED ==="
    else
        echo "=== ${TAG} FAILED ==="
    fi
done

echo ""
echo "================================================="
echo "All done: $(date)"
echo "================================================="

echo ""
echo "=== SUMMARY (S2/S7/S9) ==="
echo "--- source_only ---"
grep "Subject [279] =>" "${BASE_DIR}/source_only/results.txt" 2>/dev/null
echo ""
for MOM in 0.1 0.01 0.001; do
    TAG="bn_clean_mom${MOM}"
    echo "--- bn_stat_clean, mom=${MOM} ---"
    grep "Subject [279] =>" "${BASE_DIR}/${TAG}/results.txt" 2>/dev/null
    echo ""
done

echo "Reference (bs=48, prior run):"
echo "  S2: source=67.71%, bn_clean=59.38%  (harm -8.33%)"
echo "  S7: source=92.36%, bn_clean=95.14%  (gain +2.78%)"
echo "  S9: source=88.19%, bn_clean=88.19%  (neutral at bs=48, -2.08% at bs=1)"
