#!/bin/bash

# Default installation path
INSTALL_PATH="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local path=$2
    local available_gb=$(df -BG "$path" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        print_status "$RED" "Error: Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    return 0
}

# Function to setup environment
setup_env() {
    print_status "$GREEN" "Setting up environment in: $INSTALL_PATH"
    
    # Create necessary directories
    mkdir -p "$INSTALL_PATH"/{outputs/{models,logs,cache},venv}
    
    # Check minimum disk space (10GB)
    if ! check_disk_space 10 "$INSTALL_PATH"; then
        exit 1
    fi
    
    # Create virtual environment
    python -m venv "$INSTALL_PATH/venv"
    
    # Activate virtual environment
    source "$INSTALL_PATH/venv/bin/activate"
    
    if [ $? -ne 0 ]; then
        print_status "$RED" "Failed to activate virtual environment"
        exit 1
    fi
    
    # Install dependencies
    print_status "$GREEN" "Installing dependencies..."
    pip install -r requirements.txt
    
    # Set environment variables
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    export TORCH_HOME="$INSTALL_PATH/outputs/cache"
    export HF_HOME="$INSTALL_PATH/outputs/cache"
    export TRANSFORMERS_CACHE="$INSTALL_PATH/outputs/cache"
    export WANDB_DIR="$INSTALL_PATH/outputs/logs"
    
    # Verify CUDA
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        print_status "$GREEN" "CUDA is working correctly"
        python -c "
import torch
gpu = torch.cuda.get_device_properties(0)
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {gpu.total_memory / 1024**3:.1f}GB')
"
    else
        print_status "$RED" "Error: CUDA is not working"
        exit 1
    fi
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    print_status "$YELLOW" "Cleaning old checkpoints..."
    find "$INSTALL_PATH/outputs/models" -name "checkpoint-*" -type d | sort -r | tail -n +4 | xargs -r rm -rf
}

# Function to run training
run_training() {
    print_status "$GREEN" "Running system checks..."
    
    # Activate virtual environment
    source "$INSTALL_PATH/venv/bin/activate"
    
    if [ $? -ne 0 ]; then
        print_status "$RED" "Failed to activate virtual environment. Please run setup first."
        exit 1
    fi
    
    # Set environment variables
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    export TORCH_HOME="$INSTALL_PATH/outputs/cache"
    export HF_HOME="$INSTALL_PATH/outputs/cache"
    export TRANSFORMERS_CACHE="$INSTALL_PATH/outputs/cache"
    export WANDB_DIR="$INSTALL_PATH/outputs/logs"
    
    # Check disk space and clean if needed
    if ! check_disk_space 50 "$INSTALL_PATH"; then
        clean_old_checkpoints
    fi
    
    # Verify CUDA
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        print_status "$RED" "Error: CUDA is not working. Please check your GPU setup."
        exit 1
    fi
    
    print_status "$GREEN" "Starting training..."
    python train.py
}

# Parse command line arguments
case "${1:-}" in
    "setup")
        setup_env
        ;;
    "train")
        run_training
        ;;
    "clean")
        clean_old_checkpoints
        ;;
    *)
        echo "Usage: $0 {setup|train|clean}"
        exit 1
        ;;
esac 