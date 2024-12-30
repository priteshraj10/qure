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
        deactivate 2>/dev/null
    fi
    exit 1
}

# Function to check if a Python package is installed
package_installed() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        MINGW*|MSYS*|CYGWIN*) echo "windows";;
        *)          echo "unknown";;
    esac
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
    elif [[ "$(uname -m)" == "arm64" ]] && [[ "$(detect_os)" == "macos" ]]; then
        echo "apple_silicon"
    else
        echo "cpu"
    fi
}

# Detect operating system
OS_TYPE=$(detect_os)
print_step "Detected OS: $OS_TYPE"

# Check for required commands
print_step "Checking system dependencies..."

# Check for Python
if ! command_exists python3 && ! command_exists python; then
    case $OS_TYPE in
        "linux")
            echo "Python is not installed. Please install Python 3.10 or higher:"
            echo "For Ubuntu/Debian: sudo apt-get install python3"
            echo "For Fedora: sudo dnf install python3"
            echo "For other distributions, visit: https://www.python.org/downloads/"
            ;;
        "macos")
            echo "Python is not installed. Please install Python 3.10 or higher:"
            echo "Visit: https://www.python.org/downloads/"
            ;;
        "windows")
            echo "Python is not installed. Please install Python 3.10 or higher:"
            echo "Visit: https://www.python.org/downloads/"
            ;;
    esac
    exit 1
fi

# Check for npm/Node.js
if ! command_exists npm; then
    case $OS_TYPE in
        "linux")
            echo "Node.js/npm is not installed. Please install Node.js 16 or higher:"
            echo "Visit: https://nodejs.org/en/download/"
            echo "Or use your distribution's package manager"
            ;;
        "macos")
            echo "Node.js/npm is not installed. Please install Node.js 16 or higher:"
            echo "Visit: https://nodejs.org/en/download/"
            ;;
        "windows")
            echo "Node.js/npm is not installed. Please install Node.js 16 or higher:"
            echo "Visit: https://nodejs.org/en/download/"
            ;;
    esac
    exit 1
fi

# Install Rust if not present (optional, only if needed)
if ! command_exists rustc && [ "$OS_TYPE" != "windows" ]; then
    print_step "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Setup Python virtual environment
print_step "Setting up Python virtual environment..."
cd backend || handle_error "Backend directory not found"

# Create virtual environment based on OS
if [ "$OS_TYPE" = "windows" ]; then
    PYTHON_CMD="python"
    VENV_ACTIVATE="venv/Scripts/activate"
else
    PYTHON_CMD="python3"
    VENV_ACTIVATE="venv/bin/activate"
fi

if [ ! -d "venv" ]; then
    print_step "Creating new virtual environment..."
    $PYTHON_CMD -m venv venv || handle_error "Failed to create virtual environment"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source "$VENV_ACTIVATE" || handle_error "Failed to activate virtual environment"

# Upgrade pip first
print_step "Upgrading pip..."
python -m pip install --upgrade pip || handle_error "Failed to upgrade pip"

# Detect GPU type and install appropriate PyTorch version
GPU_TYPE=$(detect_gpu)
print_step "Detected GPU type: $GPU_TYPE"

if [ "$GPU_TYPE" = "ampere" ]; then
    print_step "Installing PyTorch for Ampere GPU..."
    python -m pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || handle_error "Failed to install PyTorch"
    print_step "Installing Unsloth with Ampere optimizations..."
    python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
    python -m pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes || handle_error "Failed to install dependencies"
elif [ "$GPU_TYPE" = "older" ]; then
    print_step "Installing PyTorch for older NVIDIA GPU..."
    python -m pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || handle_error "Failed to install PyTorch"
    print_step "Installing Unsloth for older GPUs..."
    python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
    python -m pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes || handle_error "Failed to install dependencies"
else
    print_step "Installing PyTorch for CPU/Apple Silicon..."
    if [ "$GPU_TYPE" = "apple_silicon" ]; then
        python -m pip install torch torchvision torchaudio || handle_error "Failed to install PyTorch"
    else
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || handle_error "Failed to install PyTorch"
    fi
    print_step "Installing Unsloth without GPU optimizations..."
    python -m pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
fi

# Install web dependencies
print_step "Installing web dependencies..."
if ! package_installed "fastapi"; then
    python -m pip install "fastapi[all]" python-multipart || handle_error "Failed to install web dependencies"
fi

# Verify installations
print_step "Verifying installations..."
if [ "$GPU_TYPE" != "apple_silicon" ] && [ "$GPU_TYPE" != "cpu" ]; then
    if command_exists nvcc; then
        nvcc --version
    fi
    python -m xformers.info || print_step "xformers verification skipped"
    python -m bitsandbytes || print_step "bitsandbytes verification skipped"
fi

# Start backend server
print_step "Starting backend server..."
if [ "$OS_TYPE" = "windows" ]; then
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
else
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
fi
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
cd ../frontend || handle_error "Frontend directory not found"

if [ ! -d "node_modules" ]; then
    print_step "Installing frontend dependencies..."
    npm install || handle_error "Failed to install frontend dependencies"
fi

if [ ! -d ".next" ]; then
    print_step "Building frontend..."
    npm run build || handle_error "Failed to build frontend"
fi

print_step "Starting frontend server..."
if [ "$OS_TYPE" = "windows" ]; then
    cmd //c "npm start"
else
    npm start
fi

# Cleanup
print_step "Cleaning up..."
kill $BACKEND_PID 2>/dev/null
deactivate 2>/dev/null 