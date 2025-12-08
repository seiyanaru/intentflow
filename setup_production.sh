#!/bin/bash

# setup_production.sh
# 
# Usage: source setup_production.sh
# 
# This script sets up the 'intentflow' conda environment for production use,
# installing dependencies from intentflow/intentflow/offline/requirements.txt
# with relaxed version constraints and numpy<2.

ENV_NAME="intentflow"
PYTHON_VERSION="3.11.13"
REQUIREMENTS_FILE="/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/requirements.txt"
CONDA_PATH="/home/islab-shi/anaconda3/etc/profile.d/conda.sh"

# 1. Environment Cleanup and Recreation
echo ">>> Initializing Conda..."
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "Error: conda.sh not found at $CONDA_PATH"
    exit 1
fi

echo ">>> Deactivating current environment..."
conda deactivate || true

echo ">>> Removing existing '$ENV_NAME' environment..."
conda env remove -n $ENV_NAME -y

echo ">>> Creating '$ENV_NAME' environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the new environment
conda activate $ENV_NAME

# 2. Dependency Installation
echo ">>> Installing PyTorch 2.1.2 with CUDA 11.8 support..."
# Explicitly install the version compatible with the system's CUDA driver
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

echo ">>> Installing remaining dependencies from $REQUIREMENTS_FILE..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    # Install other dependencies, allowing numpy<2. 
    # Existing torch installation should satisfy requirements or be skipped/checked.
    sed 's/[>=]=.*//' "$REQUIREMENTS_FILE" | pip install -r /dev/stdin "numpy<2"
else
    echo "Error: Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi

# 3. Verification
echo ">>> Verifying installation..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy Version: {numpy.__version__}')"

echo ">>> Setup complete. Activate the environment with: conda activate $ENV_NAME"

