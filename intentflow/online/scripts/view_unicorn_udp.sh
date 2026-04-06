#!/bin/bash
# Launch Unicorn UDP real-time waveform viewer
#
# Usage:
#   ./view_unicorn_udp.sh
#   UDP_PORT=11003 WINDOW=10 ./view_unicorn_udp.sh

set -e

# Default settings
UDP_HOST="${UDP_HOST:-0.0.0.0}"
UDP_PORT="${UDP_PORT:-11002}"
CHANNELS="${CHANNELS:-8}"
WINDOW="${WINDOW:-5.0}"
SFREQ="${SFREQ:-250}"
PACKET_FORMAT="${PACKET_FORMAT:-auto}"
SCALE="${SCALE:-50.0}"
UPDATE_MS="${UPDATE_MS:-50}"

echo "=========================================="
echo "Unicorn UDP Waveform Viewer"
echo "=========================================="
echo "UDP:        ${UDP_HOST}:${UDP_PORT}"
echo "Channels:   ${CHANNELS}"
echo "Window:     ${WINDOW}s @ ${SFREQ}Hz"
echo "Format:     ${PACKET_FORMAT}"
echo "Y-scale:    ${SCALE} µV"
echo "Update:     ${UPDATE_MS}ms"
echo "=========================================="
echo ""
echo "Press Ctrl+C or close the plot window to stop."
echo ""

python -m intentflow.online.scripts.unicorn_udp_viewer \
    --host "${UDP_HOST}" \
    --port "${UDP_PORT}" \
    --channels "${CHANNELS}" \
    --window "${WINDOW}" \
    --sfreq "${SFREQ}" \
    --packet-format "${PACKET_FORMAT}" \
    --scale "${SCALE}" \
    --update-interval "${UPDATE_MS}"
