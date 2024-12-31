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

# Function to check if running with sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        print_status "$RED" "Please run with sudo"
        exit 1
    fi
}

# Function to check and fix permissions
fix_permissions() {
    local dir=$1
    local user=$SUDO_USER
    
    # Check if directory exists, create if not
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
    
    # Fix permissions
    chown -R $user:$user "$dir"
    chmod -R 755 "$dir"
}

# Function to setup environment
setup_env() {
    print_status "$GREEN" "Setting up environment in: $INSTALL_PATH"
    
    # Create directories
    local dirs=(
        "$INSTALL_PATH/outputs/models"
        "$INSTALL_PATH/outputs/logs"
        "$INSTALL_PATH/outputs/cache"
        "$INSTALL_PATH/outputs/checkpoints"
        "$INSTALL_PATH/venv"
    )
    
    for dir in "${dirs[@]}"; do
        fix_permissions "$dir"
    done
    
    # Switch to user context for venv creation
    su - $SUDO_USER -c "
        # Create virtual environment
        python3 -m venv '$INSTALL_PATH/venv' --system-site-packages
        
        # Activate virtual environment
        source '$INSTALL_PATH/venv/bin/activate'
        
        # Install dependencies
        pip install --no-cache-dir -r '$INSTALL_PATH/requirements.txt'
        
        # Test CUDA
        python3 -c '
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")
    print(f\"Memory: {gpu.total_memory / 1024**3:.1f}GB\")
else:
    print(\"CUDA not available\")
'
    "
    
    # Create environment file
    cat > "$INSTALL_PATH/.env" << EOL
export PYTHONPATH="$INSTALL_PATH:\$PYTHONPATH"
export TORCH_HOME="$INSTALL_PATH/outputs/cache"
export HF_HOME="$INSTALL_PATH/outputs/cache"
export TRANSFORMERS_CACHE="$INSTALL_PATH/outputs/cache"
export WANDB_DIR="$INSTALL_PATH/outputs/logs"
EOL
    
    # Fix permissions for .env file
    chown $SUDO_USER:$SUDO_USER "$INSTALL_PATH/.env"
    chmod 644 "$INSTALL_PATH/.env"
    
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
    su - $SUDO_USER -c "
        source '$INSTALL_PATH/venv/bin/activate'
        cd '$INSTALL_PATH'
        python train.py
    "
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    print_status "$YELLOW" "Cleaning old checkpoints..."
    find "$INSTALL_PATH/outputs/models" -name "checkpoint-*" -type d | \
        sort -r | tail -n +4 | xargs -r rm -rf
}

# Check if running with sudo
check_sudo

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
        echo "Usage: sudo $0 {setup|train|clean}"
        exit 1
        ;;
esac 