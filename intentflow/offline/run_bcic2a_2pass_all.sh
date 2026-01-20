#!/bin/bash
# BCIC 2a 2-Pass Entropy Gating 全被験者実験
# 
# 目的: 2-pass dynamic gatingの効果を全被験者で検証
# - Pass 1: Attention-only (TTT OFF) → Entropy計算
# - Entropy Gating: H > threshold なら Pass 2 へ
# - Pass 2: TTT ON with gated α, lr_scale → 最終予測
#
# 出力:
# - debug_s{subject}_TCFormer_Hybrid.json: 2-pass flip analysis
# - twopass_s{subject}_TCFormer_Hybrid.json: サンプルごとの詳細データ

set -e

# ===== Configuration =====
GPU_ID=${1:-0}
ENTROPY_THRESHOLD=${2:-0.85}
ALPHA_MAX=${3:-0.5}
LR_SCALE_MAX=${4:-0.5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "BCIC 2a 2-Pass Experiment (All Subjects)"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Entropy Threshold: $ENTROPY_THRESHOLD"
echo "Alpha Max: $ALPHA_MAX"
echo "LR Scale Max: $LR_SCALE_MAX"
echo "=========================================="

# Run experiment
python3 run_bcic2a_2pass.py \
    --gpu_id "$GPU_ID" \
    --all_subjects \
    --entropy_threshold "$ENTROPY_THRESHOLD" \
    --alpha_max "$ALPHA_MAX" \
    --lr_scale_max "$LR_SCALE_MAX"

# Get the latest results directory
RESULTS_DIR=$(ls -td results/bcic2a_2pass/*/ 2>/dev/null | head -1)

if [ -n "$RESULTS_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "Generating visualizations..."
    echo "=========================================="
    python3 visualize_2pass.py "$RESULTS_DIR"
    
    echo ""
    echo "=========================================="
    echo "Experiment completed!"
    echo "Results: $RESULTS_DIR"
    echo "=========================================="
else
    echo "Warning: Could not find results directory"
fi


