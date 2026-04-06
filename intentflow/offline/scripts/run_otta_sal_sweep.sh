#!/bin/bash
# SAL threshold sweep for OTTA optimization
# Tests SAL thresholds: 0.35, 0.40, 0.45, 0.50 (baseline)
# Fixed: removed set -e to allow continuation on individual failures
# Fixed: added timeout and error handling per SAL value

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Remove stale MNE lock file if present
rm -f ~/.mne/mne-python.json.lock

# Configuration
MODEL="tcformer_otta"
DATASET="bcic2a"
GPU_ID=0
SEED=0
SAL_THRESHOLDS=(0.35 0.40 0.45 0.50)

# Per-SAL timeout: 500 epochs * 9 subjects * ~25min/subject = ~225min, add margin
TIMEOUT_SEC=18000  # 5 hours per SAL value

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_BASE="${OFFLINE_DIR}/results"
CONFIG_DIR="${OFFLINE_DIR}/configs/tcformer_otta"

# Generate timestamp (shared across all experiments)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_EXPERIMENT_NAME="otta_sal_sweep"

echo "=========================================="
echo "OTTA SAL Threshold Sweep"
echo "=========================================="
echo "Model:        ${MODEL}"
echo "Dataset:      ${DATASET}"
echo "Seed:         ${SEED}"
echo "GPU:          ${GPU_ID}"
echo "SAL values:   ${SAL_THRESHOLDS[@]}"
echo "Timestamp:    ${TIMESTAMP}"
echo "Timeout/SAL:  ${TIMEOUT_SEC}s"
echo "=========================================="
echo ""

# Verify GPU availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')" || {
    echo "ERROR: CUDA not available. Aborting."
    exit 1
}

FAILED_SALS=()
PASSED_SALS=()

for SAL in "${SAL_THRESHOLDS[@]}"; do
    SAL_INT=$(echo "$SAL" | sed 's/\.//')  # 0.35 -> 035
    CONFIG_NAME="tcformer_otta_sal${SAL_INT}.yaml"
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}"

    echo ""
    echo "=========================================="
    echo "Creating config for SAL=${SAL}"
    echo "=========================================="

    # Copy base config and modify SAL threshold
    cp "${CONFIG_DIR}/tcformer_otta.yaml" "${CONFIG_PATH}"

    # Update SAL threshold using Python
    python3 << EOFPYTHON
import yaml

config_path = "${CONFIG_PATH}"
sal_value = ${SAL}

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['model_kwargs']['sal_threshold'] = sal_value

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Updated config: sal_threshold = {sal_value}")
EOFPYTHON

    # Setup experiment
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_sal${SAL_INT}_s${SEED}_${TIMESTAMP}"
    RESULTS_DIR="${RESULTS_BASE}/${EXPERIMENT_NAME}"
    LOG_FILE="${RESULTS_BASE}/${EXPERIMENT_NAME}.log"

    echo ""
    echo "Running SAL=${SAL}..."
    echo "Results dir: ${RESULTS_DIR}"
    echo "Log file:    ${LOG_FILE}"
    echo "Started at:  $(date)"
    echo ""

    mkdir -p "${RESULTS_DIR}"

    # Run experiment with timeout to prevent hangs
    cd "${OFFLINE_DIR}"
    timeout ${TIMEOUT_SEC} python3 train_pipeline.py \
      --model "${MODEL}" \
      --dataset "${DATASET}" \
      --seed "${SEED}" \
      --gpu_id "${GPU_ID}" \
      --config "${CONFIG_PATH}" \
      --results_dir "${RESULTS_DIR}" \
      2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ] && [ -f "${RESULTS_DIR}/results.txt" ]; then
        echo ""
        echo "=== SAL=${SAL} COMPLETED ==="
        tail -5 "${RESULTS_DIR}/results.txt"
        PASSED_SALS+=("${SAL}")
    elif [ ${EXIT_CODE} -eq 124 ]; then
        echo ""
        echo "=== SAL=${SAL} TIMEOUT (${TIMEOUT_SEC}s) ==="
        FAILED_SALS+=("${SAL}:TIMEOUT")
    else
        echo ""
        echo "=== SAL=${SAL} FAILED (exit code: ${EXIT_CODE}) ==="
        FAILED_SALS+=("${SAL}:ERROR_${EXIT_CODE}")
    fi

    echo "Finished at: $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "SAL Sweep Completed!"
echo "=========================================="
echo ""

# Aggregate results
echo "=== Results Summary ==="
for SAL in "${SAL_THRESHOLDS[@]}"; do
    SAL_INT=$(echo "$SAL" | sed 's/\.//')
    EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_sal${SAL_INT}_s${SEED}_${TIMESTAMP}"
    RESULTS_FILE="${RESULTS_BASE}/${EXPERIMENT_NAME}/results.txt"

    if [ -f "${RESULTS_FILE}" ]; then
        ACC=$(grep "Average Test Accuracy" "${RESULTS_FILE}" | awk '{print $4}')
        STD=$(grep "Average Test Accuracy" "${RESULTS_FILE}" | awk '{print $6}')
        echo "SAL=${SAL}: ${ACC} ± ${STD}%"
    else
        echo "SAL=${SAL}: FAILED"
    fi
done

echo ""
echo "Passed: ${PASSED_SALS[@]:-none}"
echo "Failed: ${FAILED_SALS[@]:-none}"
echo ""
echo "Recommendation: Choose SAL with highest accuracy for next experiments."
