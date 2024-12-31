#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get absolute path of script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/venv"
ENV_FILE="$SCRIPT_DIR/.env"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to setup virtual environment
setup_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print_status "$YELLOW" "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
        if [ $? -ne 0 ]; then
            print_status "$RED" "Failed to create virtual environment"
            return 1
        fi
    fi
    
    source "$VENV_PATH/bin/activate"
    if [ $? -ne 0 ]; then
        print_status "$RED" "Failed to activate virtual environment"
        return 1
    fi
    
    # Install base requirements
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch based on GPU
    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
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
    
    return 0
}

# Function to check CUDA setup
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_status "$RED" "NVIDIA drivers not found"
        return 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_status "$RED" "NVIDIA driver not working properly"
        return 1
    fi
    
    # Verify PyTorch CUDA support
    python3 - <<EOF
import torch
if not torch.cuda.is_available():
    exit(1)
gpu = torch.cuda.get_device_properties(0)
print(f"Using GPU: {gpu.name} ({gpu.total_memory/1024**3:.1f}GB)")
EOF
    
    if [ $? -ne 0 ]; then
        print_status "$RED" "PyTorch CUDA verification failed"
        return 1
    fi
    
    return 0
}

# Setup environment variables
setup_env() {
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOL
export PYTHONPATH="$SCRIPT_DIR:\$PYTHONPATH"
export TORCH_HOME="$SCRIPT_DIR/outputs/cache"
export HF_HOME="$SCRIPT_DIR/outputs/cache"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/outputs/cache"
export WANDB_DIR="$SCRIPT_DIR/outputs/logs"
EOL
    fi
    source "$ENV_FILE"
}

# Main execution
main() {
    # Create required directories
    mkdir -p "$SCRIPT_DIR/outputs/"{models,logs,cache,checkpoints}
    
    # Setup virtual environment
    if ! setup_venv; then
        print_status "$RED" "Failed to setup virtual environment"
        exit 1
    fi
    
    # Setup environment variables
    setup_env
    
    # Check CUDA setup
    print_status "$GREEN" "Checking CUDA setup..."
    if ! check_cuda; then
        print_status "$RED" "CUDA setup failed"
        exit 1
    fi
    
    # Start training
    print_status "$GREEN" "Starting training..."
    python "$SCRIPT_DIR/train.py"
    
    deactivate
}

main "$@"