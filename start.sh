#!/bin/bash

# Default installation path
INSTALL_PATH="$(pwd)"

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

# Function to check permissions
check_permissions() {
    local dir=$1
    if ! touch "$dir/.permissions_test" 2>/dev/null; then
        print_status "$RED" "Error: No write permissions in $dir"
        print_status "$YELLOW" "Please run: sudo chown -R $USER:$USER $dir"
        return 1
    fi
    rm -f "$dir/.permissions_test"
    return 0
}

# Function to setup environment
setup_env() {
    print_status "$GREEN" "Setting up environment in: $INSTALL_PATH"
    
    # Check directory permissions first
    if ! check_permissions "$INSTALL_PATH"; then
        exit 1
    fi
    
    # Create necessary directories with proper permissions
    for dir in "outputs/models" "outputs/logs" "outputs/cache" "outputs/checkpoints" "venv"; do
        mkdir -p "$INSTALL_PATH/$dir"
        chmod 755 "$INSTALL_PATH/$dir"
    done
    
    # Create virtual environment with system site packages
    python3 -m venv "$INSTALL_PATH/venv" --system-site-packages
    
    if [ ! -f "$INSTALL_PATH/venv/bin/activate" ]; then
        print_status "$RED" "Failed to create virtual environment"
        exit 1
    fi
    
    # Source the virtual environment
    source "$INSTALL_PATH/venv/bin/activate"
    
    if [ $? -ne 0 ]; then
        print_status "$RED" "Failed to activate virtual environment"
        exit 1
    fi
    
    # Install dependencies
    print_status "$GREEN" "Installing dependencies..."
    pip install --no-cache-dir -r requirements.txt
    
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