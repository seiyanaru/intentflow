#!/bin/bash
# Launch 5-seed baseline experiments in background using nohup
# Each seed runs sequentially to avoid GPU conflicts

set -e

# Configuration
GPU_ID=0
SEEDS=(0 1 2 3 4)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log file for the launcher
LAUNCHER_LOG="${OFFLINE_DIR}/results/launcher_5seeds_${TIMESTAMP}.log"

echo "=========================================="
echo "Launching 5-Seed Baseline Experiments"
echo "=========================================="
echo "Seeds:        ${SEEDS[@]}"
echo "GPU:          ${GPU_ID}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo "=========================================="
echo ""
echo "NOTE: This will run in the background using nohup."
echo "      Each seed takes ~3.5 hours (1000 epochs)."
echo "      Total estimated time: ~17.5 hours"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LAUNCHER_LOG}"
echo ""
echo "To check running processes:"
echo "  ps aux | grep train_pipeline"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""

# Create launcher script
LAUNCHER_SCRIPT="${OFFLINE_DIR}/results/run_5seeds_sequential_${TIMESTAMP}.sh"

cat > "${LAUNCHER_SCRIPT}" << 'EOFSCRIPT'
#!/bin/bash
# Auto-generated launcher script

set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="OFFLINE_DIR_PLACEHOLDER"

cd "${OFFLINE_DIR}"

SEEDS=(0 1 2 3 4)
GPU_ID=0
TIMESTAMP=REPLACE_TIMESTAMP

echo "=========================================="
echo "5-Seed Baseline - Sequential Execution"
echo "Started at: $(date)"
echo "=========================================="
echo ""

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting Seed ${SEED} at $(date)"
    echo "=========================================="

    EXPERIMENT_NAME="baseline_5seed_s${SEED}_${TIMESTAMP}"
    RESULTS_DIR="${OFFLINE_DIR}/results/${EXPERIMENT_NAME}"
    LOG_FILE="${OFFLINE_DIR}/results/${EXPERIMENT_NAME}.log"

    mkdir -p "${RESULTS_DIR}"

    python3 train_pipeline.py \
      --model tcformer \
      --dataset bcic2a \
      --seed "${SEED}" \
      --gpu_id "${GPU_ID}" \
      --results_dir "${RESULTS_DIR}" \
      2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "=== Seed ${SEED} Completed at $(date) ==="
    if [ -f "${RESULTS_DIR}/results.txt" ]; then
        grep "Average Test Accuracy" "${RESULTS_DIR}/results.txt" || true
    fi
    echo ""
done

echo ""
echo "=========================================="
echo "All 5 seeds completed at $(date)!"
echo "=========================================="
echo ""
echo "=== Final Results Summary ==="
for SEED in "${SEEDS[@]}"; do
    EXPERIMENT_NAME="baseline_5seed_s${SEED}_${TIMESTAMP}"
    RESULTS_FILE="${OFFLINE_DIR}/results/${EXPERIMENT_NAME}/results.txt"
    if [ -f "${RESULTS_FILE}" ]; then
        ACC=$(grep "Average Test Accuracy" "${RESULTS_FILE}" | awk '{print $4}' || echo "N/A")
        echo "Seed ${SEED}: ${ACC}%"
    fi
done
EOFSCRIPT

# Replace placeholders
sed -i "s|OFFLINE_DIR_PLACEHOLDER|${OFFLINE_DIR}|g" "${LAUNCHER_SCRIPT}"
sed -i "s|REPLACE_TIMESTAMP|${TIMESTAMP}|g" "${LAUNCHER_SCRIPT}"

chmod +x "${LAUNCHER_SCRIPT}"

echo "Launcher script created: ${LAUNCHER_SCRIPT}"
echo ""
read -p "Press Enter to start the experiments in background, or Ctrl+C to cancel..."

# Launch in background with nohup
nohup bash "${LAUNCHER_SCRIPT}" > "${LAUNCHER_LOG}" 2>&1 &
PID=$!

echo ""
echo "=========================================="
echo "Experiments launched in background!"
echo "=========================================="
echo "Process ID:   ${PID}"
echo "Launcher log: ${LAUNCHER_LOG}"
echo ""
echo "To monitor:"
echo "  tail -f ${LAUNCHER_LOG}"
echo ""
echo "To check status:"
echo "  ps -p ${PID}"
echo ""
echo "To view GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Estimated completion: $(date -d '+18 hours' '+%Y-%m-%d %H:%M')"
echo "=========================================="
