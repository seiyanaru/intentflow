#!/bin/bash
# =============================================================================
# BCI Competition IV 2b: Hybridモデルのみ実行
# =============================================================================
# Baseは完了済み。Hybrid (静的α) と Hybrid (動的α/Entropy) のみ実行
# =============================================================================

set -e  # エラー時に停止

# デフォルト設定
GPU_ID=${1:-0}
SEED=42
DATASET="bcic2b"

# 既存のBaseの結果ディレクトリを使用
RESULTS_BASE="intentflow/offline/results/paper_experiments/bcic2b/20260113_110603"

# プロジェクトルートに移動
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "=============================================="
echo " BCI Competition IV 2b - Hybrid Only"
echo "=============================================="
echo " 開始時刻: $(date)"
echo " GPU ID: ${GPU_ID}"
echo " Seed: ${SEED}"
echo " 結果保存先: ${RESULTS_BASE}"
echo "=============================================="

# Conda環境のアクティベート
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# 結果ディレクトリのクリア（前回の途中結果を削除）
rm -rf "${RESULTS_BASE}/hybrid_static"
rm -rf "${RESULTS_BASE}/hybrid_entropy"
mkdir -p "${RESULTS_BASE}/hybrid_static"
mkdir -p "${RESULTS_BASE}/hybrid_entropy"

# =============================================================================
# 実験 1: Hybrid (静的α)
# =============================================================================
echo ""
echo "=============================================="
echo " [1/2] Hybrid (静的α) - 開始"
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

echo " [1/2] Hybrid (静的α) - 完了"
echo " 終了時刻: $(date)"

# =============================================================================
# 実験 2: Hybrid (動的α / Entropy Gating)
# =============================================================================
echo ""
echo "=============================================="
echo " [2/2] Hybrid (動的α / Entropy Gating) - 開始"
echo "=============================================="
echo " 開始時刻: $(date)"

python -u intentflow/offline/train_pipeline.py \
    --model tcformer_hybrid \
    --dataset ${DATASET} \
    --gpu_id ${GPU_ID} \
    --seed ${SEED} \
    --no_interaug \
    --results_dir "${RESULTS_BASE}/hybrid_entropy"

echo " [2/2] Hybrid (動的α / Entropy Gating) - 完了"
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







