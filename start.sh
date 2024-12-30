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

# Function to detect CPU architecture
detect_arch() {
    case "$(uname -m)" in
        x86_64*)    echo "x86_64";;
        arm64*)     echo "arm64";;
        aarch64*)   echo "arm64";;
        *)          echo "unknown";;
    esac
}

# Function to install PyTorch based on OS and architecture
install_pytorch() {
    local os_type=$1
    local arch_type=$2
    local gpu_type=$3
    
    print_step "Installing PyTorch for $os_type ($arch_type) with $gpu_type support..."
    
    case "$os_type" in
        "macos")
            if [ "$arch_type" = "arm64" ]; then
                python -m pip install torch torchvision torchaudio || handle_error "Failed to install PyTorch for M1/M2 Mac"
            else
                python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || handle_error "Failed to install PyTorch for Intel Mac"
            fi
            ;;
        "linux")
            if [ "$gpu_type" = "ampere" ] || [ "$gpu_type" = "older" ]; then
                python -m pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || handle_error "Failed to install PyTorch with CUDA support"
            else
                python -m pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || handle_error "Failed to install PyTorch for CPU"
            fi
            ;;
        "windows")
            if [ "$gpu_type" = "ampere" ] || [ "$gpu_type" = "older" ]; then
                python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || handle_error "Failed to install PyTorch with CUDA support"
            else
                python -m pip install torch torchvision torchaudio || handle_error "Failed to install PyTorch for CPU"
            fi
            ;;
        *)
            handle_error "Unsupported operating system"
            ;;
    esac
}

# Function to install Unsloth based on GPU type
install_unsloth() {
    local gpu_type=$1
    
    print_step "Installing Unsloth with appropriate optimizations..."
    
    case "$gpu_type" in
        "ampere")
            python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
            python -m pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes || handle_error "Failed to install Unsloth dependencies"
            ;;
        "older")
            python -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
            python -m pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes || handle_error "Failed to install Unsloth dependencies"
            ;;
        *)
            python -m pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install Unsloth"
            ;;
    esac
}

# Detect system information
OS_TYPE=$(detect_os)
ARCH_TYPE=$(detect_arch)
print_step "Detected OS: $OS_TYPE ($ARCH_TYPE)"

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

# Detect GPU type
GPU_TYPE=$(detect_gpu)
print_step "Detected GPU type: $GPU_TYPE"

# Install PyTorch based on system configuration
install_pytorch "$OS_TYPE" "$ARCH_TYPE" "$GPU_TYPE"

# Install Unsloth based on GPU type
install_unsloth "$GPU_TYPE"

# Install web dependencies
print_step "Installing web dependencies..."
if ! package_installed "fastapi"; then
    python -m pip install "fastapi[all]" python-multipart || handle_error "Failed to install web dependencies"
fi

# Verify installations
print_step "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || handle_error "PyTorch installation verification failed"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || print_step "CUDA not available"

if [ "$GPU_TYPE" != "cpu" ]; then
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