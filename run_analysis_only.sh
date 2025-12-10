#!/bin/bash
set -e

# Setup environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Automatically detect the latest result directories
BASE_DIR=$(ls -td intentflow/offline/results/TCFormer_bcic2a_* | head -1)
TTT_DIR=$(ls -td intentflow/offline/results/TCFormer_TTT_bcic2a_* | head -1)

echo "Using Base model results from: $BASE_DIR"
echo "Using TTT model results from: $TTT_DIR"

if [ -z "$BASE_DIR" ] || [ -z "$TTT_DIR" ]; then
    echo "Error: Could not find result directories."
    exit 1
fi

# Create output directory for analysis
ANALYSIS_OUT="intentflow/offline/analysis/comparison_plots"
mkdir -p $ANALYSIS_OUT

echo "Starting analysis for all subjects (1-9)..."

# Run analysis for each subject
for subj in {1..9}; do
    echo "------------------------------------------------"
    echo "Analyzing Subject $subj..."
    python intentflow/offline/analysis/analyze_behavior.py \
        --subject_id $subj \
        --base_dir "$BASE_DIR" \
        --ttt_dir "$TTT_DIR" \
        --out_dir "$ANALYSIS_OUT" \
        --gpu_id 0
done

echo "------------------------------------------------"
echo "All analyses complete. Plots saved to $ANALYSIS_OUT"


