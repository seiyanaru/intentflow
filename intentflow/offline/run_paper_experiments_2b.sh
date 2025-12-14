#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Explicitly use the python executable from the intentflow environment
PYTHON_EXEC="/home/islab-shi/anaconda3/envs/intentflow/bin/python"

# Change directory to where the scripts are located
cd "$(dirname "$0")"

echo ">>> Starting Paper Experiments for BCIC IV-2b Dataset..."

# 1. Run TCFormer_Hybrid (Hybrid TTT)
echo ">>> Running TCFormer_Hybrid (Hybrid) on BCIC IV-2b..."
$PYTHON_EXEC train_pipeline.py --model tcformer_hybrid --dataset bcic2b --gpu_id 0

# 2. Run TCFormer_TTT (Pure TTT)
echo ">>> Running TCFormer_TTT (Pure TTT) on BCIC IV-2b..."
$PYTHON_EXEC train_pipeline.py --model tcformer_ttt --dataset bcic2b --gpu_id 0

# 3. Run TCFormer (Base)
echo ">>> Running TCFormer (Base) on BCIC IV-2b..."
$PYTHON_EXEC train_pipeline.py --model tcformer --dataset bcic2b --gpu_id 0

echo ">>> Experiments Complete."

# 4. Generate Figures
echo ">>> Generating Figures for BCIC IV-2b..."

# Consolidate results
mkdir -p results/paper_data_2b

# Use correct casing for copy
echo "Copying result files for BCIC IV-2b..."
# Copy Hybrid
cp -r results/*TCFormer_Hybrid_bcic2b*/*s*TCFormer_Hybrid* results/paper_data_2b/ 2>/dev/null || echo "No Hybrid files found"
cp results/*TCFormer_Hybrid_bcic2b*/final_acc* results/paper_data_2b/ 2>/dev/null || echo "No Hybrid acc found"

# Copy TTT
cp -r results/*TCFormer_TTT_bcic2b*/*s*TCFormer_TTT* results/paper_data_2b/ 2>/dev/null || echo "No TTT files found"
cp results/*TCFormer_TTT_bcic2b*/final_acc* results/paper_data_2b/ 2>/dev/null || echo "No TTT acc found"

# Copy Base
cp -r results/*TCFormer_bcic2b*/*s*TCFormer* results/paper_data_2b/ 2>/dev/null || echo "No Base files found"
cp results/*TCFormer_bcic2b*/final_acc* results/paper_data_2b/ 2>/dev/null || echo "No Base acc found"

$PYTHON_EXEC generate_paper_figs.py --results_dir results/paper_data_2b --save_dir fig_2b

echo ">>> Done! Figures are in 'fig_2b/' directory."

