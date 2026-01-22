#!/bin/bash
# ============================================================================
# HGD (High Gamma Dataset) 実験スクリプト
# 
# 3つのモデルを比較実験:
#   1. Base (TCFormer)
#   2. Hybrid Static (固定α)
#   3. Hybrid Dynamic (Entropy-driven)
# ============================================================================

set -e

# GPU ID (デフォルト: 0)
GPU_ID=${1:-0}

# タイムスタンプで結果ディレクトリを作成
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="intentflow/offline/results/paper_experiments/hgd/${TIMESTAMP}"

echo "============================================"
echo "HGD Dataset Experiments"
echo "============================================"
echo "GPU: ${GPU_ID}"
echo "Results: ${RESULTS_BASE}"
echo "Subjects: 14"
echo "Classes: 4 (feet, hand(L), rest, hand(R))"
echo "============================================"
echo ""

# 結果ディレクトリ作成
mkdir -p "${RESULTS_BASE}/base"
mkdir -p "${RESULTS_BASE}/hybrid_static"
mkdir -p "${RESULTS_BASE}/hybrid_entropy"

# Conda環境をアクティベート
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

cd /workspace-cloud/seiya.narukawa/intentflow

# ============================================================================
# 1. Base Model (TCFormer)
# ============================================================================
echo ""
echo "=== [1/3] Base Model (TCFormer) ==="
echo "開始: $(date)"
echo ""

python -u intentflow/offline/train_pipeline.py \
    --model tcformer \
    --dataset hgd \
    --gpu_id ${GPU_ID} \
    --seed 42 \
    --no_interaug \
    --results_dir "${RESULTS_BASE}/base"

echo ""
echo "Base Model 完了: $(date)"
echo ""

# ============================================================================
# 2. Hybrid Static (固定α)
# ============================================================================
echo ""
echo "=== [2/3] Hybrid Model (Static α) ==="
echo "開始: $(date)"
echo ""

python -u intentflow/offline/train_pipeline.py \
    --model tcformer_hybrid \
    --dataset hgd \
    --gpu_id ${GPU_ID} \
    --seed 42 \
    --no_interaug \
    --model_kwargs '{"use_dynamic_gating": false}' \
    --results_dir "${RESULTS_BASE}/hybrid_static"

echo ""
echo "Hybrid Static 完了: $(date)"
echo ""

# ============================================================================
# 3. Hybrid Dynamic (Entropy-driven gating)
# ============================================================================
echo ""
echo "=== [3/3] Hybrid Model (Entropy-driven Dynamic α) ==="
echo "開始: $(date)"
echo ""

python -u intentflow/offline/train_pipeline.py \
    --model tcformer_hybrid \
    --dataset hgd \
    --gpu_id ${GPU_ID} \
    --seed 42 \
    --no_interaug \
    --results_dir "${RESULTS_BASE}/hybrid_entropy"

echo ""
echo "Hybrid Dynamic 完了: $(date)"
echo ""

# ============================================================================
# 結果サマリー
# ============================================================================
echo ""
echo "============================================"
echo "全実験完了！"
echo "============================================"
echo "結果保存先: ${RESULTS_BASE}"
echo ""

# 結果ファイルがあれば表示
for model in base hybrid_static hybrid_entropy; do
    if [ -f "${RESULTS_BASE}/${model}/results.txt" ]; then
        echo "--- ${model} ---"
        cat "${RESULTS_BASE}/${model}/results.txt"
        echo ""
    fi
done

echo "============================================"
echo "完了時刻: $(date)"
echo "============================================"




