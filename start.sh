#!/bin/bash

# Default installation path (will be overridden by command line argument)
INSTALL_PATH="."

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
        echo "Error: Insufficient disk space on $(df -h "$path" | awk 'NR==2 {print $1}')"
        echo "Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    return 0
}

# Function to setup environment
setup_env() {
    echo "Setting up environment in: $INSTALL_PATH"
    
    # Create necessary directories
    mkdir -p "$INSTALL_PATH"/{models,logs}
    
    # Check minimum disk space for setup (10GB)
    if ! check_disk_space 10 "$INSTALL_PATH"; then
        exit 1
    fi
    
    # Check CUDA
    if ! command_exists nvidia-smi; then
        echo "Error: NVIDIA GPU driver not found"
        exit 1
    fi
    
    # Get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    CUDA_MAJOR_VERSION=${CUDA_VERSION%.*}
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Create new conda environment
    ENV_NAME="qure_env"
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "Removing existing conda environment..."
        conda deactivate
        conda env remove -n $ENV_NAME -y
    fi
    
    echo "Creating new conda environment..."
    conda create -n $ENV_NAME python=3.11 -y
    
    # Activate the environment
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment"
        exit 1
    fi
    
    echo "Installing PyTorch with CUDA support..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    echo "Installing other dependencies..."
    pip install -r requirements.txt
    
    # Set PYTHONPATH to include the project root
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    
    echo "Environment setup complete in $INSTALL_PATH"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    nvidia-smi
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    echo "Cleaning old checkpoints in $INSTALL_PATH/models..."
    cd "$INSTALL_PATH/models" && ls -t checkpoint-*.pt | tail -n +4 | xargs -r rm && cd - > /dev/null
}

# Function to run training
run_training() {
    echo "Running system checks..."
    
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate qure_env
    
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment. Please run setup first."
        exit 1
    fi
    
    # Set PYTHONPATH
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    
    # Clean old checkpoints if space is low
    if ! check_disk_space 50 "$INSTALL_PATH"; then
        echo "Low disk space detected. Cleaning old checkpoints..."
        clean_old_checkpoints
    fi
    
    # Run system check with custom paths
    MODELS_PATH="$INSTALL_PATH/models" \
    LOGS_PATH="$INSTALL_PATH/logs" \
    python backend/system_check.py
    
    if [ $? -eq 0 ]; then
        echo "Starting training..."
        # Run training with custom paths
        MODELS_PATH="$INSTALL_PATH/models" \
        LOGS_PATH="$INSTALL_PATH/logs" \
        python train.py
    else
        echo "System check failed. Please ensure system requirements are met."
        exit 1
    fi
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