#!/bin/bash

# Default dataset
DATASET="bcic2a"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set paths
PROJECT_ROOT="/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline"
cd "$PROJECT_ROOT" || exit 1

PYTHON_EXEC="/home/islab-shi/anaconda3/envs/intentflow/bin/python"

# Generate Timestamp and Results Directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="$PROJECT_ROOT/results/paper_experiments/${DATASET}/${TIMESTAMP}"
DATA_DIR="${EXP_DIR}/data"
FIG_DIR="${EXP_DIR}/figures"

mkdir -p "$DATA_DIR"
mkdir -p "$FIG_DIR"

echo ">>> Starting Paper Experiments for Dataset: $DATASET"
echo ">>> Results will be saved to: $EXP_DIR"

# Flags based on dataset
EXTRA_FLAGS=""
if [ "$DATASET" == "bcic2b" ]; then
    EXTRA_FLAGS="--no_interaug"
fi

# Run Models
# Note: models will save files directly to DATA_DIR because we pass it as results_dir
echo ">>> Running TCFormer_Hybrid..."
$PYTHON_EXEC train_pipeline.py --model tcformer_hybrid --dataset "$DATASET" --gpu_id 0 $EXTRA_FLAGS --results_dir "$DATA_DIR"

echo ">>> Running TCFormer_TTT..."
$PYTHON_EXEC train_pipeline.py --model tcformer_ttt --dataset "$DATASET" --gpu_id 0 $EXTRA_FLAGS --results_dir "$DATA_DIR"

echo ">>> Running TCFormer (Base)..."
$PYTHON_EXEC train_pipeline.py --model tcformer --dataset "$DATASET" --gpu_id 0 $EXTRA_FLAGS --results_dir "$DATA_DIR"

# Generate Figures
echo ">>> Generating Figures..."
$PYTHON_EXEC scripts/viz_paper_figs.py --data_dir "$DATA_DIR" --output_dir "$FIG_DIR"

echo ">>> All Done! Check $FIG_DIR for results."

