#!/bin/bash
# =============================================================================
# BCI Competition IV 2b: 3モデル比較実験
# =============================================================================
# 実行するモデル:
#   1. Base (TCFormer)         - ベースラインモデル
#   2. Hybrid (静的α)          - use_dynamic_gating=False
#   3. Hybrid (動的α/Entropy)  - use_dynamic_gating=True (デフォルト)
#
# 使用方法:
#   chmod +x scripts/run_bcic2b_experiments.sh
#   ./scripts/run_bcic2b_experiments.sh [GPU_ID]
#
# 例:
#   ./scripts/run_bcic2b_experiments.sh 0    # GPU 0 を使用
#   ./scripts/run_bcic2b_experiments.sh 1    # GPU 1 を使用
# =============================================================================

set -e  # エラー時に停止

# デフォルト設定
GPU_ID=${1:-0}
SEED=42
DATASET="bcic2b"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="intentflow/offline/results/paper_experiments/bcic2b/${TIMESTAMP}"

# プロジェクトルートに移動
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "=============================================="
echo " BCI Competition IV 2b 実験スクリプト"
echo "=============================================="
echo " 開始時刻: $(date)"
echo " GPU ID: ${GPU_ID}"
echo " Seed: ${SEED}"
echo " 結果保存先: ${RESULTS_BASE}"
echo "=============================================="

# Conda環境のアクティベート
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# 結果ディレクトリの作成
mkdir -p "${RESULTS_BASE}/base"
mkdir -p "${RESULTS_BASE}/hybrid_static"
mkdir -p "${RESULTS_BASE}/hybrid_entropy"

# =============================================================================
# 実験 1: Base (TCFormer)
# =============================================================================
echo ""
echo "=============================================="
echo " [1/3] Base (TCFormer) - 開始"
echo "=============================================="
echo " 開始時刻: $(date)"

python -u intentflow/offline/train_pipeline.py \
    --model tcformer \
    --dataset ${DATASET} \
    --gpu_id ${GPU_ID} \
    --seed ${SEED} \
    --no_interaug \
    --results_dir "${RESULTS_BASE}/base"

echo " [1/3] Base (TCFormer) - 完了"
echo " 終了時刻: $(date)"

# =============================================================================
# 実験 2: Hybrid (静的α)
# =============================================================================
echo ""
echo "=============================================="
echo " [2/3] Hybrid (静的α) - 開始"
echo "=============================================="
echo " 開始時刻: $(date)"

python -u intentflow/offline/train_pipeline.py \
    --model tcformer_hybrid \
    --dataset ${DATASET} \
    --gpu_id ${GPU_ID} \
    --seed ${SEED} \
    --no_interaug \
    --model_kwargs '{"use_dynamic_gating": false}' \
    --results_dir "${RESULTS_BASE}/hybrid_static"

echo " [2/3] Hybrid (静的α) - 完了"
echo " 終了時刻: $(date)"

# =============================================================================
# 実験 3: Hybrid (動的α / Entropy Gating)
# =============================================================================
echo ""
echo "=============================================="
echo " [3/3] Hybrid (動的α / Entropy Gating) - 開始"
echo "=============================================="
echo " 開始時刻: $(date)"

python -u intentflow/offline/train_pipeline.py \
    --model tcformer_hybrid \
    --dataset ${DATASET} \
    --gpu_id ${GPU_ID} \
    --seed ${SEED} \
    --no_interaug \
    --results_dir "${RESULTS_BASE}/hybrid_entropy"

echo " [3/3] Hybrid (動的α / Entropy Gating) - 完了"
echo " 終了時刻: $(date)"

# =============================================================================
# サマリー
# =============================================================================
echo ""
echo "=============================================="
echo " 全実験完了"
echo "=============================================="
echo " 終了時刻: $(date)"
echo ""
echo " 結果の確認:"
echo "   Base:           ${RESULTS_BASE}/base/results.txt"
echo "   Hybrid (静的):  ${RESULTS_BASE}/hybrid_static/results.txt"
echo "   Hybrid (動的):  ${RESULTS_BASE}/hybrid_entropy/results.txt"
echo ""

# 結果をまとめて表示
echo "=============================================="
echo " 結果サマリー"
echo "=============================================="

for dir in base hybrid_static hybrid_entropy; do
    result_file="${RESULTS_BASE}/${dir}/results.txt"
    if [ -f "${result_file}" ]; then
        echo ""
        echo "--- ${dir} ---"
        cat "${result_file}"
    fi
done

echo ""
echo "=============================================="

