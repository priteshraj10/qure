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
if python3 -m pip list | grep -q "pip.*24."; then
    print_step "Pip is up to date"
else
    print_step "Upgrading pip..."
    python3 -m pip install --upgrade pip || handle_error "Failed to upgrade pip"
fi

# Install PyTorch if not present
print_step "Checking PyTorch installation..."
if ! package_installed "torch"; then
    print_step "Installing PyTorch for your system..."
    if [[ $(uname -m) == 'arm64' ]]; then
        print_step "Detected Apple Silicon (M1/M2), installing PyTorch with MPS support..."
        python3 -m pip install torch torchvision torchaudio || handle_error "Failed to install PyTorch"
    else
        print_step "Installing PyTorch for Intel Mac..."
        python3 -m pip install torch torchvision torchaudio || handle_error "Failed to install PyTorch"
    fi
else
    print_step "PyTorch is already installed"
fi

# Check and install core ML dependencies
print_step "Checking core ML dependencies..."
MISSING_DEPS=()
for pkg in "transformers" "datasets" "accelerate" "bitsandbytes" "tqdm" "peft"; do
    if ! package_installed $pkg; then
        MISSING_DEPS+=($pkg)
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    print_step "Installing missing ML dependencies: ${MISSING_DEPS[*]}"
    python3 -m pip install ${MISSING_DEPS[@]} || handle_error "Failed to install core ML dependencies"
else
    print_step "All core ML dependencies are installed"
fi

# Check and install unsloth
print_step "Checking unsloth installation..."
if ! package_installed "unsloth"; then
    print_step "Installing unsloth..."
    python3 -m pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" || handle_error "Failed to install unsloth"
else
    print_step "Unsloth is already installed"
fi

# Check and install web dependencies
print_step "Checking web dependencies..."
if ! package_installed "fastapi"; then
    print_step "Installing web dependencies..."
    python3 -m pip install "fastapi[all]" python-multipart || handle_error "Failed to install web dependencies"
else
    print_step "Web dependencies are already installed"
fi

# Start backend server
print_step "Starting backend server..."
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Verify backend server started (give it more time to start)
print_step "Waiting for backend server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_step "Backend server is running!"
        break
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        # If server failed, print the logs
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

# Install and build frontend
print_step "Setting up frontend..."
cd ../frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_step "Installing frontend dependencies..."
    npm install || handle_error "Failed to install frontend dependencies"
else
    print_step "Frontend dependencies are already installed"
fi

# Check if .next directory exists
if [ ! -d ".next" ]; then
    print_step "Building frontend..."
    npm run build || handle_error "Failed to build frontend"
else
    print_step "Frontend is already built"
fi

print_step "Starting frontend server..."
npm start

# When frontend is terminated, also terminate the backend and cleanup
print_step "Cleaning up..."
kill $BACKEND_PID 2>/dev/null
deactivate 