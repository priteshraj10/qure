#!/bin/bash

# Default installation path and requirements
INSTALL_PATH="$(pwd)"
MIN_DISK_SPACE_GB=10
MIN_RAM_GB=8
MIN_PYTHON_VERSION="3.8"
MAX_PYTHON_VERSION="3.12"

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

# Function to compare version numbers
version_compare() {
    echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'
}

# Function to check Python version
check_python_version() {
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    local min_version=$MIN_PYTHON_VERSION
    local max_version=$MAX_PYTHON_VERSION
    
    if [ $(version_compare $python_version) -lt $(version_compare $min_version) ]; then
        print_status "$RED" "Error: Python version too old. Required: >= ${min_version} (found ${python_version})"
        return 1
    fi
    
    if [ $(version_compare $python_version) -gt $(version_compare $max_version) ]; then
        print_status "$YELLOW" "Warning: Python version ${python_version} is newer than recommended ${max_version}"
        print_status "$YELLOW" "Some packages might not be compatible. Continue? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    return 0
}

# Function to check system requirements
check_system_requirements() {
    # Check Python version
    if ! check_python_version; then
        return 1
    fi

    # Check RAM
    local total_ram_gb=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "$total_ram_gb" -lt "$MIN_RAM_GB" ]; then
        print_status "$RED" "Error: Insufficient RAM. Required: ${MIN_RAM_GB}GB, Available: ${total_ram_gb}GB"
        return 1
    fi

    # Check disk space
    local available_space_gb=$(df -BG "$INSTALL_PATH" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space_gb" -lt "$MIN_DISK_SPACE_GB" ]; then
        print_status "$RED" "Error: Insufficient disk space. Required: ${MIN_DISK_SPACE_GB}GB, Available: ${available_space_gb}GB"
        return 1
    fi

    return 0
}

# Function to detect and install CUDA dependencies
install_cuda_dependencies() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_status "$YELLOW" "No NVIDIA GPU detected, installing CPU version..."
        pip install torch torchvision torchaudio
        return
    fi

    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    
    print_status "$GREEN" "Detected GPU: $gpu_name (CUDA $cuda_version)"
    
    # Clear pip cache
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

    # Verify CUDA installation
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        print_status "$RED" "CUDA installation failed. Please check your GPU drivers."
        return 1
    fi

    return 0
}

# Function to setup environment
setup_env() {
    print_status "$GREEN" "Setting up environment in: $INSTALL_PATH"
    
    # Check system requirements
    if ! check_system_requirements; then
        exit 1
    fi
    
    # Create project structure
    local dirs=(
        "outputs/models"
        "outputs/logs"
        "outputs/cache"
        "outputs/checkpoints"
        "venv"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$INSTALL_PATH/$dir"
    done
    
    # Setup Python virtual environment
    python3 -m venv "${INSTALL_PATH}/venv"
    source "${INSTALL_PATH}/venv/bin/activate"
    
    # Upgrade pip and install dependencies
    pip install --upgrade pip setuptools wheel
    
    # Install CUDA and ML dependencies
    if ! install_cuda_dependencies; then
        print_status "$RED" "Failed to install CUDA dependencies"
        exit 1
    fi
    
    # Create environment file
    cat > "${INSTALL_PATH}/.env" << EOL
export PYTHONPATH="${INSTALL_PATH}:\$PYTHONPATH"
export TORCH_HOME="${INSTALL_PATH}/outputs/cache"
export HF_HOME="${INSTALL_PATH}/outputs/cache"
export TRANSFORMERS_CACHE="${INSTALL_PATH}/outputs/cache"
export WANDB_DIR="${INSTALL_PATH}/outputs/logs"
EOL
    
    print_status "$GREEN" "Setup completed successfully!"
    print_status "$YELLOW" "To activate the environment, run:"
    echo "source \"${INSTALL_PATH}/venv/bin/activate\" && source \"${INSTALL_PATH}/.env\""
}

# Function to run training
run_training() {
    if [ ! -f "${INSTALL_PATH}/venv/bin/activate" ]; then
        print_status "$RED" "Virtual environment not found. Please run setup first."
        exit 1
    fi
    
    # Load environment variables
    if [ -f "${INSTALL_PATH}/.env" ]; then
        source "${INSTALL_PATH}/.env"
    fi
    
    # Activate virtual environment and run training
    source "${INSTALL_PATH}/venv/bin/activate"
    python train.py
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    print_status "$YELLOW" "Cleaning old checkpoints..."
    find "${INSTALL_PATH}/outputs/models" -name "checkpoint-*" -type d | \
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