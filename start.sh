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
    mkdir -p "$INSTALL_PATH"/{models,logs,venv}
    
    # Check minimum disk space for setup (10GB)
    if ! check_disk_space 10 "$INSTALL_PATH"; then
        exit 1
    fi
    
    # Check CUDA
    if ! command_exists nvidia-smi; then
        echo "Error: NVIDIA GPU driver not found"
        exit 1
    fi
    
    # Create and activate virtual environment in the external drive
    if [ ! -d "$INSTALL_PATH/venv" ]; then
        python -m venv "$INSTALL_PATH/venv"
    fi
    source "$INSTALL_PATH/venv/bin/activate"
    
    # Set PYTHONPATH to include the project root
    export PYTHONPATH="$INSTALL_PATH:$PYTHONPATH"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other requirements
    pip install -r requirements.txt
    
    echo "Environment setup complete in $INSTALL_PATH"
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    echo "Cleaning old checkpoints in $INSTALL_PATH/models..."
    # Keep only the 3 most recent checkpoints
    cd "$INSTALL_PATH/models" && ls -t checkpoint-*.pt | tail -n +4 | xargs -r rm && cd - > /dev/null
}

# Function to run training
run_training() {
    echo "Running system checks..."
    
    # Activate virtual environment from external drive
    source "$INSTALL_PATH/venv/bin/activate"
    
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