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

# Function to install Python dependencies
install_python_dependencies() {
    print_step "Installing required Python packages..."
    
    # Core dependencies first
    print_step "Installing core dependencies..."
    python -m pip install --upgrade pip setuptools wheel || handle_error "Failed to upgrade pip and core tools"
    
    # Install transformers and related packages
    print_step "Installing Hugging Face Transformers..."
    python -m pip install \
        transformers==4.37.2 \
        tokenizers==0.15.1 \
        accelerate==0.27.1 \
        safetensors==0.4.2 || handle_error "Failed to install transformers and related packages"
    
    # Install web framework dependencies
    print_step "Installing web dependencies..."
    python -m pip install \
        "fastapi[all]>=0.68.0" \
        "uvicorn[standard]>=0.15.0" \
        python-multipart>=0.0.5 \
        pydantic>=1.8.2 || handle_error "Failed to install web dependencies"
    
    # Install utility packages
    print_step "Installing utility packages..."
    python -m pip install \
        numpy>=1.21.0 \
        pandas>=1.3.0 \
        tqdm>=4.65.0 \
        requests>=2.26.0 \
        pyyaml>=5.4.1 || handle_error "Failed to install utility packages"
}

# Check for Python
if ! command_exists python3 && ! command_exists python; then
    echo "Python is not installed. Please install Python 3.10 or higher:"
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check for npm/Node.js
if ! command_exists npm; then
    echo "Node.js/npm is not installed. Please install Node.js 16 or higher:"
    echo "Visit: https://nodejs.org/en/download/"
    exit 1
fi

# Setup Python virtual environment
print_step "Setting up Python virtual environment..."
cd backend || handle_error "Backend directory not found"

# Create virtual environment
if [ ! -d "venv" ]; then
    print_step "Creating new virtual environment..."
    python3 -m venv venv || handle_error "Failed to create virtual environment"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate || handle_error "Failed to activate virtual environment"

# Install base dependencies
install_python_dependencies

# Run system check script
print_step "Checking system configuration..."
SYSTEM_INFO=$(python system_check.py)
if [ $? -ne 0 ]; then
    handle_error "Failed to check system configuration"
fi

# Parse system info
PYTORCH_INSTALL_TYPE=$(echo $SYSTEM_INFO | python -c "import sys, json; print(json.load(sys.stdin)['pytorch_install_type'])")
PYTORCH_COMMAND=$(echo $SYSTEM_INFO | python -c "import sys, json; print(json.load(sys.stdin)['install_commands']['pytorch'])")
EXTRA_COMMANDS=$(echo $SYSTEM_INFO | python -c "import sys, json; print('\n'.join(json.load(sys.stdin)['install_commands']['extras']))")

# Install PyTorch based on system configuration
print_step "Installing PyTorch for $PYTORCH_INSTALL_TYPE..."
eval "$PYTORCH_COMMAND" || handle_error "Failed to install PyTorch"

# Install extra dependencies
if [ ! -z "$EXTRA_COMMANDS" ]; then
    print_step "Installing additional dependencies..."
    while IFS= read -r cmd; do
        eval "$cmd" || handle_error "Failed to install additional dependencies"
    done <<< "$EXTRA_COMMANDS"
fi

# Verify PyTorch installation
print_step "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || handle_error "PyTorch installation verification failed"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || print_step "CUDA not available"

# Verify all critical dependencies
print_step "Verifying critical dependencies..."
python -c "
import sys
required_packages = ['torch', 'transformers', 'fastapi', 'uvicorn', 'pydantic']
missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package} installed successfully')
    except ImportError as e:
        missing_packages.append(package)
        print(f'✗ {package} not found')
if missing_packages:
    sys.exit(1)
" || handle_error "Critical dependencies missing: ${missing_packages[*]}"

# Start backend server
print_step "Starting backend server..."
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Verify backend server started
print_step "Waiting for backend server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/docs >/dev/null 2>&1; then
        print_step "Backend server is running!"
        break
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
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
npm start

# Cleanup
print_step "Cleaning up..."
kill $BACKEND_PID 2>/dev/null
deactivate 2>/dev/null 