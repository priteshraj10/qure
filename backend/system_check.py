import platform
import subprocess
import json
import sys
from pathlib import Path
import torch
import logging
import psutil
import os
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get paths from environment variables or use defaults
MODELS_PATH = Path(os.getenv('MODELS_PATH', 'models'))
LOGS_PATH = Path(os.getenv('LOGS_PATH', 'logs'))

# Disk space thresholds
MIN_FREE_SPACE_GB = 50  # Minimum free space required in GB
MIN_FREE_SPACE_PERCENT = 15  # Minimum free space required in percentage
CHECKPOINT_SIZE_ESTIMATE_GB = 5  # Estimated size of each checkpoint in GB

def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_nvidia_gpu_info():
    """Get NVIDIA GPU information if available"""
    try:
        nvidia_smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=gpu_name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        if nvidia_smi.returncode == 0:
            gpus = []
            for line in nvidia_smi.stdout.strip().split('\n'):
                name, driver, memory = line.split(', ')
                gpus.append({
                    'name': name,
                    'driver_version': driver,
                    'memory': memory
                })
            return gpus
        return None
    except FileNotFoundError:
        return None

def get_cuda_version():
    """Get CUDA version if available"""
    try:
        # First try nvcc
        nvcc_output = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if nvcc_output.returncode == 0:
            for line in nvcc_output.stdout.split('\n'):
                if 'release' in line.lower():
                    return line.split('V')[-1].split('.')[0]
        
        # If nvcc not found, try nvidia-smi (common in Colab)
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            for line in nvidia_smi.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    return line.split('CUDA Version:')[1].strip().split('.')[0]
        return None
    except FileNotFoundError:
        return None

def get_system_info():
    """Get detailed system information"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = platform.python_version()
    is_in_colab = is_colab()
    
    info = {
        'os': system,
        'architecture': machine,
        'python_version': python_version,
        'is_colab': is_in_colab,
        'cuda': {
            'available': False,
            'version': None,
            'gpus': None
        },
        'pytorch_install_type': 'cpu'  # default to CPU
    }
    
    # Check for NVIDIA GPU and CUDA
    cuda_version = get_cuda_version()
    gpus = get_nvidia_gpu_info()
    
    if cuda_version and gpus:
        info['cuda']['available'] = True
        info['cuda']['version'] = cuda_version
        info['cuda']['gpus'] = gpus
        
        # Determine PyTorch install type based on GPU and CUDA version
        if int(cuda_version) >= 11:
            gpu_names = [gpu['name'].lower() for gpu in gpus]
            # In Colab, we'll use optimized settings for T4/P100/V100
            if is_in_colab:
                info['pytorch_install_type'] = 'cuda-colab'
            elif any('rtx' in name and ('30' in name or '40' in name) for name in gpu_names):
                info['pytorch_install_type'] = 'cuda-ampere'
            else:
                info['pytorch_install_type'] = 'cuda-normal'
    
    # Check for Apple Silicon
    elif system == 'darwin' and machine == 'arm64':
        info['pytorch_install_type'] = 'mps'
    
    # Generate installation commands
    info['install_commands'] = get_install_commands(info)
    
    return info

def get_install_commands(info):
    """Generate appropriate installation commands based on system info"""
    commands = {
        'pytorch': None,
        'extras': []
    }
    
    if info['is_colab']:
        # Colab-specific installations
        commands['pytorch'] = 'pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
        commands['extras'] = [
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
            'pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes',
            'pip install ipywidgets'  # For Colab progress bars
        ]
    elif info['pytorch_install_type'] == 'cuda-ampere':
        commands['pytorch'] = 'pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
        commands['extras'] = [
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
            'pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes'
        ]
    elif info['pytorch_install_type'] == 'cuda-normal':
        commands['pytorch'] = 'pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
        commands['extras'] = [
            'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
            'pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes'
        ]
    elif info['pytorch_install_type'] == 'mps':
        commands['pytorch'] = 'pip install torch torchvision torchaudio'
        commands['extras'] = [
            'pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"'
        ]
    else:  # CPU
        if info['os'] == 'linux':
            commands['pytorch'] = 'pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
        else:
            commands['pytorch'] = 'pip install torch torchvision torchaudio'
        commands['extras'] = [
            'pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"'
        ]
    
    # Add common Colab utilities if in Colab
    if info['is_colab']:
        commands['extras'].extend([
            'pip install jupyter-dash plotly',  # For interactive visualizations
            'pip install google-colab'  # Ensure Colab integration
        ])
    
    return commands

def check_system_requirements() -> Dict[str, Any]:
    """Verify all system requirements including GPU, CPU, memory, and disk space"""
    system_info = {
        "gpu": check_gpu_requirements(),
        "cpu": check_cpu_requirements(),
        "memory": check_memory_requirements(),
        "disk": check_disk_requirements()
    }
    
    # Validate disk space requirements
    disk_info = system_info["disk"]
    if not is_disk_space_sufficient(disk_info):
        raise RuntimeError(
            f"Insufficient disk space. Required: {MIN_FREE_SPACE_GB}GB free and "
            f"{MIN_FREE_SPACE_PERCENT}% free space. Current: {disk_info['free']:.1f}GB free "
            f"({100 - disk_info['percent_used']:.1f}% free)"
        )
    
    logger.info("System Requirements Check Complete")
    return system_info

def is_disk_space_sufficient(disk_info: Dict[str, float]) -> bool:
    """Check if available disk space meets requirements"""
    free_gb = disk_info["free"]
    free_percent = 100 - disk_info["percent_used"]
    
    return (free_gb >= MIN_FREE_SPACE_GB and 
            free_percent >= MIN_FREE_SPACE_PERCENT)

def estimate_required_space(num_epochs: int) -> float:
    """Estimate required disk space for training in GB"""
    # Estimate space for checkpoints
    checkpoint_space = num_epochs * CHECKPOINT_SIZE_ESTIMATE_GB
    
    # Add buffer for logs and temporary files
    buffer_space = 5  # 5GB buffer
    
    return checkpoint_space + buffer_space

def get_available_space(path: str = '.') -> Tuple[float, float]:
    """Get available disk space in GB and percentage"""
    disk = psutil.disk_usage(path)
    return disk.free / (1024**3), 100 - disk.percent

def check_disk_requirements() -> Dict[str, Any]:
    """Check disk space and return detailed information"""
    # Get disk space for different relevant directories
    model_path = MODELS_PATH
    log_path = LOGS_PATH
    
    # Create directories if they don't exist
    model_path.mkdir(exist_ok=True)
    log_path.mkdir(exist_ok=True)
    
    # Get disk info for each path
    model_disk = psutil.disk_usage(str(model_path))
    log_disk = psutil.disk_usage(str(log_path))
    
    # Use the path with less available space for the main check
    disk = model_disk if model_disk.free < log_disk.free else log_disk
    
    return {
        "total": disk.total / (1024**3),  # GB
        "free": disk.free / (1024**3),  # GB
        "percent_used": disk.percent,
        "model_dir_free": model_disk.free / (1024**3),  # GB
        "log_dir_free": log_disk.free / (1024**3),  # GB
        "min_required_gb": MIN_FREE_SPACE_GB,
        "min_required_percent": MIN_FREE_SPACE_PERCENT,
        "models_path": str(model_path),
        "logs_path": str(log_path)
    }

def main():
    """Main function to check system and output information"""
    try:
        info = get_system_info()
        
        # Create output directory if it doesn't exist
        Path('system_info').mkdir(exist_ok=True)
        
        # Save full system info
        with open('system_info/system_check.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        # Output installation type for shell script
        print(json.dumps({
            'pytorch_install_type': info['pytorch_install_type'],
            'install_commands': info['install_commands'],
            'is_colab': info['is_colab']
        }))
        
    except Exception as e:
        print(json.dumps({
            'error': str(e),
            'pytorch_install_type': 'cpu',
            'is_colab': False,
            'install_commands': get_install_commands({
                'pytorch_install_type': 'cpu',
                'os': platform.system().lower(),
                'is_colab': False
            })
        }))

if __name__ == "__main__":
    main() 