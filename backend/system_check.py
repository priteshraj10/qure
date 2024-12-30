import platform
import subprocess
import json
import sys
from pathlib import Path

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
        nvcc_output = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if nvcc_output.returncode == 0:
            for line in nvcc_output.stdout.split('\n'):
                if 'release' in line.lower():
                    return line.split('V')[-1].split('.')[0]
        return None
    except FileNotFoundError:
        return None

def get_system_info():
    """Get detailed system information"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = platform.python_version()
    
    info = {
        'os': system,
        'architecture': machine,
        'python_version': python_version,
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
            if any('rtx' in name and ('30' in name or '40' in name) for name in gpu_names):
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
    
    if info['pytorch_install_type'] == 'cuda-ampere':
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
    
    return commands

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
            'install_commands': info['install_commands']
        }))
        
    except Exception as e:
        print(json.dumps({
            'error': str(e),
            'pytorch_install_type': 'cpu',
            'install_commands': get_install_commands({'pytorch_install_type': 'cpu', 'os': platform.system().lower()})
        }))

if __name__ == "__main__":
    main() 