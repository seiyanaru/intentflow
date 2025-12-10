#!/bin/bash
set -e
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=== TCFormer Hybrid Experiment ==="

# 1. Train Hybrid Model (All Subjects)
echo "Starting Hybrid Model training for All Subjects..."

# Run Training
python intentflow/offline/train_pipeline.py \
    --model tcformer_hybrid \
    --dataset bcic2a \
    --gpu_id 0 \
    --seed 42 
    # Removed specific subject_ids to train on all subjects (defined in config/datamodule)

# Identify Result Dirs
# Latest Hybrid Result
HYBRID_DIR=$(ls -td intentflow/offline/results/TCFormer_Hybrid_bcic2a_seed-42_* | head -1)
echo "Hybrid Result Dir: $HYBRID_DIR"

# Latest Base Result (reuse existing)
BASE_DIR=$(ls -td intentflow/offline/results/TCFormer_bcic2a_* | head -1)
if [ -z "$BASE_DIR" ]; then
    echo "Warning: Base model result not found. Skipping comparison."
else
    echo "Base Result Dir: $BASE_DIR"
    
    # 2. Analyze Behavior (Base vs Hybrid)
    ANALYSIS_OUT="intentflow/offline/analysis/hybrid_comparison"
    mkdir -p $ANALYSIS_OUT
    
    echo "Analyzing behavior (Base vs Hybrid) for all subjects..."
    
    for subj in {1..9}; do
        echo "------------------------------------------------"
        echo "Analyzing Subject $subj..."
        python intentflow/offline/analysis/analyze_behavior.py \
            --subject_id $subj \
            --base_dir "$BASE_DIR" \
            --ttt_dir "$HYBRID_DIR" \
            --out_dir "$ANALYSIS_OUT" \
            --gpu_id 0
    done
        
    echo "Analysis complete. Results saved to $ANALYSIS_OUT"
fi

