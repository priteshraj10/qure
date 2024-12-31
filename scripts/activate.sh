#!/bin/bash

# Project directories
PROJECT_ROOT=$(pwd)
VENV_PATH="$PROJECT_ROOT/venv"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TORCH_HOME="$PROJECT_ROOT/outputs/cache"
export HF_HOME="$PROJECT_ROOT/outputs/cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/outputs/cache"
export WANDB_DIR="$PROJECT_ROOT/outputs/logs"

echo "Environment activated! You can now run training." 