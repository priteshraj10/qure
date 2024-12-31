#!/bin/bash

# Default installation path (will be overridden by command line argument)
INSTALL_PATH="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
        print_status "$RED" "Error: Insufficient disk space on $(df -h "$path" | awk 'NR==2 {print $1}')"
        print_status "$RED" "Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    return 0
}

# Function to setup environment
setup_env() {
    print_status "$GREEN" "Setting up environment in: $INSTALL_PATH"
    
    # Create necessary directories
    mkdir -p "$INSTALL_PATH"/{models,logs,cache}
    
    # Check minimum disk space for setup (10GB)
    if ! check_disk_space 10 "$INSTALL_PATH"; then
        exit 1
    fi
    
    # Check CUDA
    if ! command_exists nvidia-smi; then
        print_status "$RED" "Error: NVIDIA GPU driver not found"
        exit 1
    fi
    
    # Get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    CUDA_MAJOR_VERSION=${CUDA_VERSION%.*}
    print_status "$GREEN" "Detected CUDA version: $CUDA_VERSION"
    
    # Create new conda environment
    ENV_NAME="qure_env"
    if conda env list | grep -q "^$ENV_NAME "; then
        print_status "$YELLOW" "Removing existing conda environment..."
        conda deactivate
        conda env remove -n $ENV_NAME -y
    fi
    
    print_status "$GREEN" "Creating new conda environment..."
    conda create -n $ENV_NAME python=3.11 -y
    
    # Activate the environment
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    if [ $? -ne 0 ]; then
        print_status "$RED" "Failed to activate conda environment"
        exit 1
    fi
    
    # Set environment variables
    export TORCH_HOME="$INSTALL_PATH/cache"
    export HF_HOME="$INSTALL_PATH/cache"
    export TRANSFORMERS_CACHE="$INSTALL_PATH/cache"
    export WANDB_DIR="$INSTALL_PATH/logs"
    
    print_status "$GREEN" "Installing core dependencies..."
    conda install -y pip setuptools wheel
    conda install -y numpy pandas scipy scikit-learn matplotlib
    
    print_status "$GREEN" "Installing PyTorch with CUDA support..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    print_status "$GREEN" "Installing transformers and other dependencies..."
    pip install --no-cache-dir transformers==4.37.2 datasets==2.16.1 wandb==0.16.3 tqdm==4.66.2
    pip install --no-cache-dir psutil==5.9.8 pyyaml==6.0.1
    
    # Install optional dependencies for better performance
    print_status "$GREEN" "Installing optional performance-enhancing dependencies..."
    pip install --no-cache-dir ninja
    pip install --no-cache-dir flash-attn --no-build-isolation
    pip install --no-cache-dir xformers
    
    # Verify installations
    print_status "$GREEN" "Verifying installations..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        print_status "$GREEN" "CUDA is working correctly"
        
        # Print GPU information
        python -c "
import torch
gpu = torch.cuda.get_device_properties(0)
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {gpu.total_memory / 1024**3:.1f}GB')
print(f'CUDA Capability: {gpu.major}.{gpu.minor}')
"
    else
        print_status "$RED" "Warning: CUDA is not working, falling back to CPU"
    fi
    
    # Set PYTHONPATH to include the project root
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    
    print_status "$GREEN" "Environment setup complete in $INSTALL_PATH"
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    print_status "$GREEN" "Cleaning old checkpoints in $INSTALL_PATH/models..."
    cd "$INSTALL_PATH/models" && ls -t checkpoint-*.pt | tail -n +4 | xargs -r rm && cd - > /dev/null
}

# Function to run training
run_training() {
    print_status "$GREEN" "Running system checks..."
    
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate qure_env
    
    if [ $? -ne 0 ]; then
        print_status "$RED" "Failed to activate conda environment. Please run setup first."
        exit 1
    fi
    
    # Set environment variables
    export TORCH_HOME="$INSTALL_PATH/cache"
    export HF_HOME="$INSTALL_PATH/cache"
    export TRANSFORMERS_CACHE="$INSTALL_PATH/cache"
    export WANDB_DIR="$INSTALL_PATH/logs"
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    
    # Clean old checkpoints if space is low
    if ! check_disk_space 50 "$INSTALL_PATH"; then
        print_status "$YELLOW" "Low disk space detected. Cleaning old checkpoints..."
        clean_old_checkpoints
    fi
    
    # Verify CUDA is working
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        print_status "$RED" "Error: CUDA is not working. Please check your GPU setup."
        exit 1
    fi
    
    # Run system checks
    print_status "$GREEN" "Running final system checks..."
    python backend/system_check.py
    if [ $? -ne 0 ]; then
        print_status "$RED" "System requirements not met. Please check the errors above."
        exit 1
    fi
    
    print_status "$GREEN" "Starting training..."
    # Run training with custom paths
    MODELS_PATH="$INSTALL_PATH/models" \
    LOGS_PATH="$INSTALL_PATH/logs" \
    python train.py
}

# Function to show help
show_help() {
    echo "Usage: $0 {setup|train|clean} [--path /path/to/external/drive]"
    echo "Commands:"
    echo "  setup - Set up the environment"
    echo "  train - Run the training"
    echo "  clean - Clean old checkpoints"
    echo "Options:"
    echo "  --path  Path to external drive for installation (default: current directory)"
    exit 1
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        setup|train|clean)
            COMMAND="$1"
            shift
            ;;
        --path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        *)
            show_help
            ;;
    esac
done

# Ensure command is provided
if [ -z "$COMMAND" ]; then
    show_help
fi

# Ensure install path exists and is writable
if [ ! -d "$INSTALL_PATH" ]; then
    echo "Error: Installation path does not exist: $INSTALL_PATH"
    exit 1
fi
if [ ! -w "$INSTALL_PATH" ]; then
    echo "Error: Installation path is not writable: $INSTALL_PATH"
    exit 1
fi

# Execute command
case "$COMMAND" in
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
        show_help
        ;;
esac 