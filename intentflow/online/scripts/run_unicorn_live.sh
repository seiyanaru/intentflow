#!/bin/bash

set -euo pipefail

ROOT_DIR="/workspace-cloud/seiya.narukawa/intentflow"
DEFAULT_RESULT_DIR="${ROOT_DIR}/intentflow/offline/results/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260306_2319"

SUBJECT_ID="${SUBJECT_ID:-1}"
RESULT_DIR="${RESULT_DIR:-${DEFAULT_RESULT_DIR}}"
CHECKPOINT="${CHECKPOINT:-${RESULT_DIR}/checkpoints/subject_${SUBJECT_ID}_model.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-${RESULT_DIR}/config.yaml}"

GPU_ID="${GPU_ID:-0}"
WS_HOST="${WS_HOST:-0.0.0.0}"
WS_PORT="${WS_PORT:-8765}"
UDP_HOST="${UDP_HOST:-0.0.0.0}"
UDP_PORT="${UDP_PORT:-1001}"
UDP_PACKET_FORMAT="${UDP_PACKET_FORMAT:-auto}"
UDP_DELIMITER="${UDP_DELIMITER:-,}"
STREAM_CHANNELS="${STREAM_CHANNELS:-8}"
STREAM_SFREQ="${STREAM_SFREQ:-250}"
WINDOW_SEC="${WINDOW_SEC:-4.0}"
HOP_SEC="${HOP_SEC:-0.25}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.55}"
NORMALIZER="${NORMALIZER:-window}"
STREAM_ALPHA="${STREAM_ALPHA:-0.01}"
STREAM_WARMUP_SAMPLES="${STREAM_WARMUP_SAMPLES:-250}"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

source /home/islab-shi/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

cd "${ROOT_DIR}"

echo "=========================================="
echo "Intentflow Unicorn Live"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Config:     ${CONFIG_PATH}"
echo "WS:         ws://${WS_HOST}:${WS_PORT}"
echo "UDP:        ${UDP_HOST}:${UDP_PORT}/${UDP_PACKET_FORMAT}"
echo "Stream:     ${STREAM_CHANNELS}ch @ ${STREAM_SFREQ}Hz"
echo "Window:     ${WINDOW_SEC}s hop ${HOP_SEC}s"
echo "Normalizer: ${NORMALIZER}"
echo "Conf th:    ${CONFIDENCE_THRESHOLD}"
echo "=========================================="
echo ""
echo "Note: current live path maps Unicorn 8ch input to a BCIC2a 22ch model via zero-pad/truncate."
echo "Treat this as a transport and smoke test unless you have subject-specific Unicorn calibration data."
echo ""

python -m intentflow.online.server.ttt_broadcaster \
  --source unicorn_udp \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG_PATH}" \
  --gpu_id "${GPU_ID}" \
  --host "${WS_HOST}" \
  --port "${WS_PORT}" \
  --udp_host "${UDP_HOST}" \
  --udp_port "${UDP_PORT}" \
  --udp_packet_format "${UDP_PACKET_FORMAT}" \
  --udp_delimiter "${UDP_DELIMITER}" \
  --stream_channels "${STREAM_CHANNELS}" \
  --stream_sfreq "${STREAM_SFREQ}" \
  --window_sec "${WINDOW_SEC}" \
  --hop_sec "${HOP_SEC}" \
  --confidence_threshold "${CONFIDENCE_THRESHOLD}" \
  --normalizer "${NORMALIZER}" \
  --stream_alpha "${STREAM_ALPHA}" \
  --stream_warmup_samples "${STREAM_WARMUP_SAMPLES}" \
  --two_class_only
