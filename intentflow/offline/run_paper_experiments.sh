#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Explicitly use the python executable from the intentflow environment
PYTHON_EXEC="/home/islab-shi/anaconda3/envs/intentflow/bin/python"

# Change directory to where the scripts are located
cd "$(dirname "$0")"

echo ">>> Starting Paper Experiments..."

# 1. Run TCFormer_Hybrid (Hybrid TTT)
echo ">>> Running TCFormer_Hybrid (Hybrid)..."
$PYTHON_EXEC train_pipeline.py --model tcformer_hybrid --dataset bcic2a --gpu_id 0

# 2. Run TCFormer_TTT (Pure TTT)
echo ">>> Running TCFormer_TTT (Pure TTT)..."
$PYTHON_EXEC train_pipeline.py --model tcformer_ttt --dataset bcic2a --gpu_id 0

# 3. Run TCFormer (Base)
echo ">>> Running TCFormer (Base)..."
$PYTHON_EXEC train_pipeline.py --model tcformer --dataset bcic2a --gpu_id 0

echo ">>> Experiments Complete."

# 4. Generate Figures
echo ">>> Generating Figures..."

# Consolidate results
mkdir -p results/paper_data

# Use correct casing for copy
echo "Copying result files..."
# Copy Hybrid
cp -r results/*TCFormer_Hybrid*/*s*TCFormer_Hybrid* results/paper_data/ 2>/dev/null || echo "No Hybrid files found"
cp results/*TCFormer_Hybrid*/final_acc* results/paper_data/ 2>/dev/null || echo "No Hybrid acc found"

# Copy TTT
cp -r results/*TCFormer_TTT*/*s*TCFormer_TTT* results/paper_data/ 2>/dev/null || echo "No TTT files found"
cp results/*TCFormer_TTT*/final_acc* results/paper_data/ 2>/dev/null || echo "No TTT acc found"

# Copy Base (Avoid matching TTT/Hybrid by excluding _)
# Or just copy explicit TCFormer_bcic2a pattern
cp -r results/*TCFormer_bcic2a*/*s*TCFormer* results/paper_data/ 2>/dev/null || echo "No Base files found"
cp results/*TCFormer_bcic2a*/final_acc* results/paper_data/ 2>/dev/null || echo "No Base acc found"

$PYTHON_EXEC generate_paper_figs.py --results_dir results/paper_data --save_dir fig

echo ">>> Done! Figures are in 'fig/' directory."