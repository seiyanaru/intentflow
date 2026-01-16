#!/bin/bash
# Ablation Experiments for Hybrid Dynamic Model
# Tests 4 different configurations to identify the cause of HGD degradation

GPU_ID=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="intentflow/offline/results/ablation_experiments/${TIMESTAMP}"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

echo "=============================================="
echo "Ablation Experiments"
echo "GPU: ${GPU_ID}"
echo "Results: ${RESULTS_BASE}"
echo "=============================================="

# Common settings
COMMON_ARGS="--model tcformer_hybrid --gpu_id ${GPU_ID} --seed 42"

# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: Eval時もfeature_stats gatingを使用 (entropy gating OFF)
# Purpose: entropy経路が原因かを確認
# ─────────────────────────────────────────────────────────────────────────────
run_exp_a() {
    local dataset=$1
    local extra_args=$2
    echo ""
    echo "=== Exp A: feature_stats gating (entropy OFF) - ${dataset} ==="
    mkdir -p "${RESULTS_BASE}/exp_a_feature_stats/${dataset}"
    python -u intentflow/offline/train_pipeline.py \
        ${COMMON_ARGS} \
        --dataset ${dataset} \
        ${extra_args} \
        --model_kwargs '{"gating_mode": "feature_stats"}' \
        --results_dir "${RESULTS_BASE}/exp_a_feature_stats/${dataset}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: エントロピー正規化 + threshold=0.7
# Purpose: 閾値調整で2bも発火するようになるか確認
# ─────────────────────────────────────────────────────────────────────────────
run_exp_b() {
    local dataset=$1
    local extra_args=$2
    echo ""
    echo "=== Exp B: Normalized entropy + threshold=0.7 - ${dataset} ==="
    mkdir -p "${RESULTS_BASE}/exp_b_norm_entropy/${dataset}"
    python -u intentflow/offline/train_pipeline.py \
        ${COMMON_ARGS} \
        --dataset ${dataset} \
        ${extra_args} \
        --model_kwargs '{"gating_mode": "entropy", "entropy_threshold": 0.7, "entropy_gating_in_train": false}' \
        --results_dir "${RESULTS_BASE}/exp_b_norm_entropy/${dataset}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: Train時もentropy gating有効 (train/test不一致解消)
# Purpose: train/test不一致がHGD崩壊の原因か確認
# ─────────────────────────────────────────────────────────────────────────────
run_exp_c() {
    local dataset=$1
    local extra_args=$2
    echo ""
    echo "=== Exp C: entropy_gating_in_train=True + threshold=0.7 - ${dataset} ==="
    mkdir -p "${RESULTS_BASE}/exp_c_train_entropy/${dataset}"
    python -u intentflow/offline/train_pipeline.py \
        ${COMMON_ARGS} \
        --dataset ${dataset} \
        ${extra_args} \
        --model_kwargs '{"gating_mode": "entropy", "entropy_threshold": 0.7, "entropy_gating_in_train": true}' \
        --results_dir "${RESULTS_BASE}/exp_c_train_entropy/${dataset}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment D: TTT Drop + entropy_gating_in_train=True
# Purpose: "TTTが無いと死ぬモデル"を防げるか確認
# ─────────────────────────────────────────────────────────────────────────────
run_exp_d() {
    local dataset=$1
    local extra_args=$2
    echo ""
    echo "=== Exp D: TTT Drop (0.2) + entropy_gating_in_train=True - ${dataset} ==="
    mkdir -p "${RESULTS_BASE}/exp_d_ttt_drop/${dataset}"
    python -u intentflow/offline/train_pipeline.py \
        ${COMMON_ARGS} \
        --dataset ${dataset} \
        ${extra_args} \
        --model_kwargs '{"gating_mode": "entropy", "entropy_threshold": 0.7, "entropy_gating_in_train": true, "ttt_drop_prob": 0.2}' \
        --results_dir "${RESULTS_BASE}/exp_d_ttt_drop/${dataset}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run all experiments for all datasets
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo "Running experiments for BCIC 2a"
echo "=============================================="
run_exp_a "bcic2a" ""
run_exp_b "bcic2a" ""
run_exp_c "bcic2a" ""
run_exp_d "bcic2a" ""

echo ""
echo "=============================================="
echo "Running experiments for BCIC 2b"
echo "=============================================="
run_exp_a "bcic2b" "--no_interaug"
run_exp_b "bcic2b" "--no_interaug"
run_exp_c "bcic2b" "--no_interaug"
run_exp_d "bcic2b" "--no_interaug"

echo ""
echo "=============================================="
echo "Running experiments for HGD"
echo "=============================================="
run_exp_a "hgd" ""
run_exp_b "hgd" ""
run_exp_c "hgd" ""
run_exp_d "hgd" ""

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_BASE}"
echo "=============================================="

# Summary of experiments:
# Exp A: gating_mode=feature_stats (eval時もtrain時と同じゲーティング)
# Exp B: entropy正規化 + threshold=0.7 (train時はfeature_stats、eval時はentropy)
# Exp C: entropy_gating_in_train=True (train/eval両方でentropy gating)
# Exp D: ttt_drop_prob=0.2 (学習中に確率的にTTTを無効化)



