#!/bin/bash
# Update Operator Comparison: 5 conditions on same checkpoints
# Purpose: Separate gate effect from update operator effect
#
# Conditions:
# 1. source_only       – No adaptation at all
# 2. bn_stat           – Current BN running stat update (Dropout/DropPath ON = noisy)
# 3. bn_stat_clean     – BN running stat update (Dropout/DropPath OFF = clean)
# 4. tent_bn           – Entropy minimization on BN affine params (gradient-based)
# 5. tent_bn_ln        – Entropy minimization on BN + LayerNorm affine params

source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow
rm -f ~/.mne/mne-python.json.lock

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
cd "${OFFLINE_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="results/update_op_comparison_${TIMESTAMP}"
mkdir -p "${RESULTS_BASE}"

# Use same config as SAL=0.50 baseline (default OTTA config)
CONFIG="${OFFLINE_DIR}/configs/tcformer_otta/tcformer_otta.yaml"
GPU_ID=0
SEED=0

echo "=========================================="
echo "Update Operator Comparison"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Config:    ${CONFIG}"
echo ""

# Verify GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'" || { echo "ERROR: No GPU"; exit 1; }

CONDITIONS=(source_only bn_stat bn_stat_clean tent_bn tent_bn_ln)
FAILED=()
PASSED=()

for COND in "${CONDITIONS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Condition: ${COND}"
    echo "Started:   $(date)"
    echo "=========================================="

    RESULT_DIR="${RESULTS_BASE}/${COND}"
    LOG_FILE="${RESULTS_BASE}/${COND}.log"
    mkdir -p "${RESULT_DIR}"

    # Pass adapt_mode via model_kwargs JSON override
    if [ "${COND}" == "source_only" ]; then
        EXTRA_KWARGS='{"adapt_mode": "source_only", "enable_otta": false}'
    else
        EXTRA_KWARGS="{\"adapt_mode\": \"${COND}\"}"
    fi

    timeout 14400 python3 train_pipeline.py \
        --model tcformer_otta \
        --dataset bcic2a \
        --seed ${SEED} \
        --gpu_id ${GPU_ID} \
        --config "${CONFIG}" \
        --results_dir "${RESULT_DIR}" \
        --model_kwargs "${EXTRA_KWARGS}" \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ] && [ -f "${RESULT_DIR}/results.txt" ]; then
        echo "=== ${COND} COMPLETED ==="
        grep "Average Test Accuracy" "${RESULT_DIR}/results.txt"
        PASSED+=("${COND}")
    else
        echo "=== ${COND} FAILED (exit=${EXIT_CODE}) ==="
        FAILED+=("${COND}")
    fi
    echo "Finished: $(date)"
done

echo ""
echo "=========================================="
echo "Update Operator Comparison — Results"
echo "=========================================="
for COND in "${CONDITIONS[@]}"; do
    RF="${RESULTS_BASE}/${COND}/results.txt"
    if [ -f "${RF}" ]; then
        ACC=$(grep "Average Test Accuracy" "${RF}" | awk '{print $4, $5, $6}')
        echo "${COND}: ${ACC}"
    else
        echo "${COND}: FAILED"
    fi
done
echo ""
echo "Passed: ${PASSED[@]:-none}"
echo "Failed: ${FAILED[@]:-none}"
