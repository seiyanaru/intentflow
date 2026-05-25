#!/usr/bin/env bash
# Launch the DC-Replay 72h sweep in the background and record log/PID paths.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

GPU_ID="${1:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/islabshi/anaconda3/envs/intentflow/bin/python}"
LOG_DIR="results/dc_replay_72h_logs"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/dc_replay_72h_${STAMP}.log"
PID_PATH="${LOG_DIR}/dc_replay_72h_${STAMP}.pid"

# setsid keeps the sweep detached from the launching shell/session. This is
# more robust than plain nohup in sandboxed assistant sessions.
setsid env PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_dc_replay_72h_sweep.sh "${GPU_ID}" \
    > "${LOG_PATH}" 2>&1 < /dev/null &
PID="$!"

echo "${PID}" > "${PID_PATH}"
ln -sfn "$(basename "${LOG_PATH}")" "${LOG_DIR}/latest.log"
ln -sfn "$(basename "${PID_PATH}")" "${LOG_DIR}/latest.pid"

echo "Started DC-Replay 72h sweep"
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
echo "PID file: ${PID_PATH}"
