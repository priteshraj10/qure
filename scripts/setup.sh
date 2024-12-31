#!/bin/bash

# Colors for status messages
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Project directories
PROJECT_ROOT=$(pwd)
VENV_PATH="$PROJECT_ROOT/venv"
MODELS_PATH="$PROJECT_ROOT/outputs/models"
LOGS_PATH="$PROJECT_ROOT/outputs/logs"
CACHE_PATH="$PROJECT_ROOT/outputs/cache"

# Create virtual environment
python -m venv $VENV_PATH

# Create project directories
mkdir -p "$MODELS_PATH" "$LOGS_PATH" "$CACHE_PATH"

# Activate virtual environment and install dependencies
source "$VENV_PATH/bin/activate"
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TORCH_HOME="$CACHE_PATH"
export HF_HOME="$CACHE_PATH"
export TRANSFORMERS_CACHE="$CACHE_PATH"
export WANDB_DIR="$LOGS_PATH"

echo -e "${GREEN}Setup completed successfully!${NC}" 