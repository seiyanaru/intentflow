#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=== Strategy 1: Learning Rate Optimization ==="
# LR = 0.1
echo "Running Training with LR=0.1..."
python intentflow/offline/train_pipeline.py \
    --model tcformer_ttt --dataset bcic2a --gpu_id 0 --seed 42 \
    --interaug \
    --model_kwargs '{"ttt_config": {"base_lr": 0.1}}'

# Identify Result Dir
RES_DIR_LR01=$(ls -td intentflow/offline/results/TCFormer_TTT_bcic2a_seed-42_* | head -1)
echo "Result LR=0.1: $RES_DIR_LR01"

# LR = 0.01
echo "Running Training with LR=0.01..."
python intentflow/offline/train_pipeline.py \
    --model tcformer_ttt --dataset bcic2a --gpu_id 0 --seed 42 \
    --interaug \
    --model_kwargs '{"ttt_config": {"base_lr": 0.01}}'

RES_DIR_LR001=$(ls -td intentflow/offline/results/TCFormer_TTT_bcic2a_seed-42_* | head -1)
echo "Result LR=0.01: $RES_DIR_LR001"


echo "=== Strategy 2: Regularized TTT ==="
# LR = 1.0 (Default) + Reg = 0.01
echo "Running Training with Reg=0.01..."
python intentflow/offline/train_pipeline.py \
    --model tcformer_ttt --dataset bcic2a --gpu_id 0 --seed 42 \
    --interaug \
    --model_kwargs '{"ttt_config": {"base_lr": 1.0, "ttt_reg_lambda": 0.01}}'

RES_DIR_REG=$(ls -td intentflow/offline/results/TCFormer_TTT_bcic2a_seed-42_* | head -1)
echo "Result Reg=0.01: $RES_DIR_REG"


echo "=== Analysis ==="
# Base Model Directory (Reuse from previous run if possible, or assume it exists)
BASE_DIR=$(ls -td intentflow/offline/results/TCFormer_bcic2a_* | head -1)
echo "Base Dir: $BASE_DIR"

ANALYSIS_ROOT="intentflow/offline/analysis/strategy_comparison"
mkdir -p $ANALYSIS_ROOT

# Analyze Subject 6 (Focus Subject)
SUBJ=6
echo "Analyzing Subject $SUBJ for all strategies..."

python intentflow/offline/analysis/analyze_behavior.py --subject_id $SUBJ --base_dir "$BASE_DIR" --ttt_dir "$RES_DIR_LR01" --out_dir "$ANALYSIS_ROOT/lr_0.1" --gpu_id 0
python intentflow/offline/analysis/analyze_behavior.py --subject_id $SUBJ --base_dir "$BASE_DIR" --ttt_dir "$RES_DIR_LR001" --out_dir "$ANALYSIS_ROOT/lr_0.01" --gpu_id 0
python intentflow/offline/analysis/analyze_behavior.py --subject_id $SUBJ --base_dir "$BASE_DIR" --ttt_dir "$RES_DIR_REG" --out_dir "$ANALYSIS_ROOT/reg_0.01" --gpu_id 0

echo "Comparison Complete. Check $ANALYSIS_ROOT"

