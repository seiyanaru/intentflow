#!/bin/bash
# Launch Unicorn UDP web-based waveform viewer
#
# Usage:
#   ./view_unicorn_web.sh
#   UDP_PORT=11003 WEB_PORT=8080 ./view_unicorn_web.sh

set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# Default settings
UDP_HOST="${UDP_HOST:-0.0.0.0}"
UDP_PORT="${UDP_PORT:-11002}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8080}"
CHANNELS="${CHANNELS:-8}"
WINDOW="${WINDOW:-5.0}"
SFREQ="${SFREQ:-250}"
PACKET_FORMAT="${PACKET_FORMAT:-auto}"

python -m intentflow.online.scripts.unicorn_web_viewer \
    --udp-host "${UDP_HOST}" \
    --udp-port "${UDP_PORT}" \
    --web-host "${WEB_HOST}" \
    --web-port "${WEB_PORT}" \
    --channels "${CHANNELS}" \
    --window "${WINDOW}" \
    --sfreq "${SFREQ}" \
    --packet-format "${PACKET_FORMAT}"
