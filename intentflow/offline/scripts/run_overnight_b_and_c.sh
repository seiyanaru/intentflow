#!/usr/bin/env bash
# Orchestrator: launch Plan C on GPU 2 immediately, wait for any in-flight
# train_pipeline on GPU 0 to finish, then launch Plan B on GPU 0.
#
# Usage: ./scripts/run_overnight_b_and_c.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

LOG_DIR="/tmp"
TS=$(date +%Y%m%d_%H%M%S)
C_LOG="${LOG_DIR}/plan_c_${TS}.log"
B_LOG="${LOG_DIR}/plan_b_${TS}.log"

echo "[orchestrator] Launching Plan C on GPU 2 ..."
nohup bash scripts/run_c_aug_true_9subj.sh 2 > "${C_LOG}" 2>&1 &
C_PID=$!
echo "[orchestrator] Plan C started, pid=${C_PID}, log=${C_LOG}"

echo "[orchestrator] Waiting for any train_pipeline on GPU 0 to finish ..."
while pgrep -f "train_pipeline.py.*--gpu_id 0" > /dev/null 2>&1; do
    sleep 30
done
echo "[orchestrator] GPU 0 idle. Launching Plan B ..."
nohup bash scripts/run_b_5seed_4subj.sh 0 > "${B_LOG}" 2>&1 &
B_PID=$!
echo "[orchestrator] Plan B started, pid=${B_PID}, log=${B_LOG}"

echo
echo "[orchestrator] Both plans dispatched."
echo "  Plan C log: ${C_LOG}"
echo "  Plan B log: ${B_LOG}"
echo "  Aggregated outputs:"
echo "    results/c_aug_true_9subj_<ts>/summary.md"
echo "    results/b_5seed_4subj_<ts>/summary.md"
