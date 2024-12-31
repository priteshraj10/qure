#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check CUDA setup
check_cuda() {
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_status "$RED" "nvidia-smi not found. Please install NVIDIA drivers."
        return 1
    }

    # Check NVIDIA driver
    if ! nvidia-smi &> /dev/null; then
        print_status "$RED" "NVIDIA driver not working properly."
        return 1
    }

    # Get GPU info
    local gpu_info=$(nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader)
    if [ -z "$gpu_info" ]; then
        print_status "$RED" "No GPU detected."
        return 1
    }

    print_status "$GREEN" "Found GPU: $gpu_info"
    return 0
}

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/venv"
ENV_FILE="$SCRIPT_DIR/.env"

# Check if conda is available and active
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    print_status "$YELLOW" "Using Conda environment: $CONDA_DEFAULT_ENV"
else
    # Check if virtual environment exists
    if [ ! -d "$VENV_PATH" ]; then
        print_status "$YELLOW" "Virtual environment not found. Creating one..."
        python3 -m venv "$VENV_PATH"
        if [ $? -ne 0 ]; then
            print_status "$RED" "Failed to create virtual environment."
            exit 1
        fi
    fi

    # Activate virtual environment
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    else
        print_status "$RED" "Virtual environment activation script not found."
        exit 1
    fi
fi

# Check if required packages are installed
print_status "$GREEN" "Checking dependencies..."
python -c "import torch" 2>/dev/null || {
    print_status "$YELLOW" "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# Load environment variables if .env exists
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    print_status "$YELLOW" "No .env file found. Creating default environment variables..."
    cat > "$ENV_FILE" << EOL
export PYTHONPATH="$SCRIPT_DIR:\$PYTHONPATH"
export TORCH_HOME="$SCRIPT_DIR/outputs/cache"
export HF_HOME="$SCRIPT_DIR/outputs/cache"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/outputs/cache"
export WANDB_DIR="$SCRIPT_DIR/outputs/logs"
EOL
    source "$ENV_FILE"
fi

# Check CUDA setup
print_status "$GREEN" "Checking CUDA setup..."
if ! check_cuda; then
    print_status "$RED" "CUDA setup failed. Please check your GPU installation."
    exit 1
fi

# Verify CUDA with PyTorch
print_status "$GREEN" "Verifying PyTorch CUDA support..."
python - <<EOF
import torch
import sys

if not torch.cuda.is_available():
    print("PyTorch CUDA is not available")
    sys.exit(1)

gpu = torch.cuda.get_device_properties(0)
print(f"Using GPU: {gpu.name} ({gpu.total_memory/1024**3:.1f}GB)")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
EOF

if [ $? -ne 0 ]; then
    print_status "$RED" "PyTorch CUDA verification failed."
    exit 1
fi

# Create required directories
mkdir -p "$SCRIPT_DIR/outputs/"{models,logs,cache,checkpoints}

# Start training
print_status "$GREEN" "Starting training..."
python "$SCRIPT_DIR/train.py"

# Check training exit status
if [ $? -ne 0 ]; then
    print_status "$RED" "Training failed. Check the logs for details."
    exit 1
fi

print_status "$GREEN" "Training completed successfully!"

# Deactivate virtual environment if not using conda
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    deactivate
fi