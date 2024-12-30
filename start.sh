#!/bin/bash

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        echo "Error: Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    return 0
}

# Function to setup environment
setup_env() {
    echo "Setting up environment..."
    
    # Check minimum disk space for setup (10GB)
    if ! check_disk_space 10; then
        exit 1
    fi
    
    # Check CUDA
    if ! command_exists nvidia-smi; then
        echo "Error: NVIDIA GPU driver not found"
        exit 1
    fi
    
    # Create and activate virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python -m venv venv
    fi
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other requirements
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p models logs
    
    echo "Environment setup complete"
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    echo "Cleaning old checkpoints..."
    # Keep only the 3 most recent checkpoints
    cd models && ls -t checkpoint-*.pt | tail -n +4 | xargs -r rm && cd ..
}

# Function to run training
run_training() {
    echo "Running system checks..."
    
    # Clean old checkpoints if space is low
    if ! check_disk_space 50; then
        echo "Low disk space detected. Cleaning old checkpoints..."
        clean_old_checkpoints
    fi
    
    python backend/system_check.py
    
    if [ $? -eq 0 ]; then
        echo "Starting training..."
        python train.py
    else
        echo "System check failed. Please ensure system requirements are met."
        exit 1
    fi
}

# Main script
case "$1" in
    "setup")
        setup_env
        ;;
    "train")
        source venv/bin/activate
        run_training
        ;;
    "clean")
        clean_old_checkpoints
        ;;
    *)
        echo "Usage: $0 {setup|train|clean}"
        echo "  setup - Set up the environment"
        echo "  train - Run the training"
        echo "  clean - Clean old checkpoints"
        exit 1
        ;;
esac 