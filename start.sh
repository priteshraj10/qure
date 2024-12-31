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

# Function to detect GPU and install appropriate torch version
install_torch() {
    print_status "$GREEN" "Detecting GPU..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_status "$YELLOW" "No NVIDIA GPU detected, installing CPU version..."
        pip install torch torchvision torchaudio
        return
    fi
    
    # Get GPU model
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    
    print_status "$GREEN" "Detected GPU: $gpu_name"
    
    # Clear pip cache to save space
    pip cache purge
    
    if [[ $gpu_name =~ "RTX 30" ]] || [[ $gpu_name =~ "RTX 40" ]] || [[ $gpu_name =~ "A100" ]]; then
        print_status "$GREEN" "Installing PyTorch for modern GPUs..."
        pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
    else
        print_status "$YELLOW" "Installing PyTorch for older GPUs..."
        pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
    fi
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
    
    # Check for minimum disk space (10GB)
    if ! check_disk_space 10 "$INSTALL_PATH"; then
        exit 1
    fi
    
    # Create directories
    local dirs=(
        "$INSTALL_PATH/outputs/models"
        "$INSTALL_PATH/outputs/logs"
        "$INSTALL_PATH/outputs/cache"
        "$INSTALL_PATH/outputs/checkpoints"
        "$INSTALL_PATH/venv"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    # Create and activate virtual environment
    python3 -m venv "$INSTALL_PATH/venv"
    source "$INSTALL_PATH/venv/bin/activate"
    
    # Install torch and dependencies
    install_torch
    
    # Create environment file
    cat > "$INSTALL_PATH/.env" << EOL
export PYTHONPATH="$INSTALL_PATH:\$PYTHONPATH"
export TORCH_HOME="$INSTALL_PATH/outputs/cache"
export HF_HOME="$INSTALL_PATH/outputs/cache"
export TRANSFORMERS_CACHE="$INSTALL_PATH/outputs/cache"
export WANDB_DIR="$INSTALL_PATH/outputs/logs"
EOL
    
    print_status "$GREEN" "Setup completed successfully!"
    print_status "$YELLOW" "To activate the environment, run:"
    echo "source $INSTALL_PATH/venv/bin/activate && source $INSTALL_PATH/.env"
}

# Function to run training
run_training() {
    if [ ! -f "$INSTALL_PATH/venv/bin/activate" ]; then
        print_status "$RED" "Virtual environment not found. Please run setup first."
        exit 1
    fi
    
    # Load environment variables
    if [ -f "$INSTALL_PATH/.env" ]; then
        source "$INSTALL_PATH/.env"
    fi
    
    # Activate virtual environment and run training
    source "$INSTALL_PATH/venv/bin/activate"
    python train.py
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    print_status "$YELLOW" "Cleaning old checkpoints..."
    find "$INSTALL_PATH/outputs/models" -name "checkpoint-*" -type d | \
        sort -r | tail -n +4 | xargs -r rm -rf
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