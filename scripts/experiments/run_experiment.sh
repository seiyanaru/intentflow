#!/bin/bash
set -e

# Setup environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting TCFormer (Base) training..."
# Run Base Training
python intentflow/offline/train_pipeline.py --model tcformer --dataset bcic2a --gpu_id 0

# Capture the latest Base result directory
BASE_DIR=$(ls -td intentflow/offline/results/TCFormer_bcic2a_* | head -1)
echo "Base model results saved in: $BASE_DIR"

echo "Starting TCFormer_TTT (TTT) training..."
# Run TTT Training
python intentflow/offline/train_pipeline.py --model tcformer_ttt --dataset bcic2a --gpu_id 0

# Capture the latest TTT result directory
TTT_DIR=$(ls -td intentflow/offline/results/TCFormer_TTT_bcic2a_* | head -1)
echo "TTT model results saved in: $TTT_DIR"

echo "Training complete. Starting analysis for all subjects (1-9)..."

# Create output directory for analysis
ANALYSIS_OUT="intentflow/offline/analysis/comparison_plots"
mkdir -p $ANALYSIS_OUT

# Run analysis for each subject
for subj in {1..9}; do
    echo "Analyzing Subject $subj..."
    python intentflow/offline/analysis/analyze_behavior.py \
        --subject_id $subj \
        --base_dir "$BASE_DIR" \
        --ttt_dir "$TTT_DIR" \
        --out_dir "$ANALYSIS_OUT" \
        --gpu_id 0
done

echo "All analyses complete. Plots saved to $ANALYSIS_OUT"


