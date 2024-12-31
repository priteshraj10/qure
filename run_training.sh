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

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/venv"
ENV_FILE="$SCRIPT_DIR/.env"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    print_status "$RED" "Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Load environment variables if .env exists
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Verify CUDA availability
print_status "$GREEN" "Checking CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    print_status "$RED" "CUDA is not available. Please check your GPU setup."
    exit 1
fi

# Print GPU info
python -c "import torch; gpu = torch.cuda.get_device_properties(0); print(f'Using GPU: {gpu.name} ({gpu.total_memory/1024**3:.1f}GB)')"

# Start training
print_status "$GREEN" "Starting training..."
python "$SCRIPT_DIR/train.py"

# Deactivate virtual environment
deactivate