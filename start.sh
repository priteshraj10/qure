#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print progress
print_step() {
    echo "===> $1"
}

# Function to handle errors
handle_error() {
    echo "Error: $1"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    exit 1
}

# Function to check if a Python package is installed
package_installed() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Function to detect GPU and CUDA
detect_gpu() {
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
        if [[ $GPU_INFO == *"RTX 30"* ]] || [[ $GPU_INFO == *"RTX 40"* ]] || [[ $GPU_INFO == *"A100"* ]] || [[ $GPU_INFO == *"H100"* ]]; then
            echo "ampere"
        else
            echo "older"
        fi
    elif [[ $(uname -m) == 'arm64' ]]; then
        echo "apple_silicon"
    else
        echo "cpu"
    fi
}

# Check for required commands and install if missing
print_step "Checking system dependencies..."
if ! command_exists brew; then
    handle_error "Homebrew is required. Please install from https://brew.sh"
fi

if ! command_exists python3; then
    handle_error "python3 is not installed"
fi

if ! command_exists npm; then
    handle_error "npm is not installed"
fi

# Install Rust if not present
if ! command_exists rustc; then
    print_step "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Setup Python virtual environment
print_step "Setting up Python virtual environment..."
cd backend
if [ ! -d "venv" ]; then
    print_step "Creating new virtual environment..."
    python3 -m venv venv || handle_error "Failed to create virtual environment"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate || handle_error "Failed to activate virtual environment"

# Upgrade pip first
print_step "Checking pip version..."
python3 -m pip install --upgrade pip || handle_error "Failed to upgrade pip"

# Detect GPU type and install appropriate PyTorch version
GPU_TYPE=$(detect_gpu)
print_step "Detected GPU type: $GPU_TYPE"

if [ "$GPU_TYPE" = "ampere" ]; then
    print_step "Installing PyTorch for Ampere GPU..."
    python3 -m pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || handle_error "Failed to install PyTorch"
    print_step "Installing Unsloth with Ampere optimizations..."
    python3 -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
    python3 -m pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes || handle_error "Failed to install dependencies"
elif [ "$GPU_TYPE" = "older" ]; then
    print_step "Installing PyTorch for older NVIDIA GPU..."
    python3 -m pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || handle_error "Failed to install PyTorch"
    print_step "Installing Unsloth for older GPUs..."
    python3 -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
    python3 -m pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes || handle_error "Failed to install dependencies"
else
    print_step "Installing PyTorch for CPU/Apple Silicon..."
    python3 -m pip install torch torchvision torchaudio || handle_error "Failed to install PyTorch"
    print_step "Installing Unsloth without GPU optimizations..."
    python3 -m pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
fi

# Install web dependencies
print_step "Installing web dependencies..."
if ! package_installed "fastapi"; then
    python3 -m pip install "fastapi[all]" python-multipart || handle_error "Failed to install web dependencies"
fi

# Verify installations
print_step "Verifying installations..."
if [ "$GPU_TYPE" != "apple_silicon" ] && [ "$GPU_TYPE" != "cpu" ]; then
    if command_exists nvcc; then
        nvcc --version
    fi
    python3 -m xformers.info || print_step "xformers verification skipped"
    python3 -m bitsandbytes || print_step "bitsandbytes verification skipped"
fi

# Start backend server
print_step "Starting backend server..."
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Verify backend server started
print_step "Waiting for backend server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_step "Backend server is running!"
        break
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_step "Backend server failed to start. Server logs:"
        cat backend.log
        handle_error "Backend server failed to start"
    fi
    sleep 1
    if [ $i -eq 60 ]; then
        handle_error "Backend server took too long to start"
    fi
    echo -n "."
done

# Frontend setup
print_step "Setting up frontend..."
cd ../frontend

if [ ! -d "node_modules" ]; then
    print_step "Installing frontend dependencies..."
    npm install || handle_error "Failed to install frontend dependencies"
fi

if [ ! -d ".next" ]; then
    print_step "Building frontend..."
    npm run build || handle_error "Failed to build frontend"
fi

print_step "Starting frontend server..."
npm start

# Cleanup
print_step "Cleaning up..."
kill $BACKEND_PID 2>/dev/null
deactivate 