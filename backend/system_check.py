import platform
import subprocess
import json
import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get paths from environment variables or use defaults
MODELS_PATH = Path(os.getenv('MODELS_PATH', 'models'))
LOGS_PATH = Path(os.getenv('LOGS_PATH', 'logs'))

# System requirements
MIN_FREE_SPACE_GB = 50
MIN_FREE_SPACE_PERCENT = 15
CHECKPOINT_SIZE_ESTIMATE_GB = 5
MIN_GPU_MEMORY_GB = 6
MIN_CUDA_VERSION = 11.0
RECOMMENDED_CUDA_VERSION = 12.1

def import_optional_dependency(name: str) -> Optional[Any]:
    """Import an optional dependency, return None if not available"""
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None

# Import optional dependencies
torch = import_optional_dependency("torch")
psutil = import_optional_dependency("psutil")

def get_gpu_memory_info() -> Dict[str, float]:
    """Get detailed GPU memory information"""
    try:
        if not torch or not torch.cuda.is_available():
            return {"total": 0, "used": 0, "free": 0}
        
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory = gpu_properties.total_memory / (1024**3)  # Convert to GB
        
        # Get current memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        
        return {
            "total": total_memory,
            "used": memory_allocated,
            "reserved": memory_reserved,
            "free": total_memory - memory_reserved
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {str(e)}")
        return {"total": 0, "used": 0, "free": 0}

def get_nvidia_gpu_info() -> Optional[list]:
    """Get NVIDIA GPU information if available"""
    try:
        # Try nvidia-smi first
        nvidia_smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=gpu_name,driver_version,memory.total,memory.used,memory.free,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in nvidia_smi.stdout.strip().split('\n'):
            name, driver, total, used, free, temp = line.split(', ')
            gpus.append({
                'name': name,
                'driver_version': driver,
                'memory_total_gb': float(total) / 1024,  # Convert MB to GB
                'memory_used_gb': float(used) / 1024,
                'memory_free_gb': float(free) / 1024,
                'temperature_c': float(temp)
            })
        return gpus
    except subprocess.SubprocessError:
        # If nvidia-smi fails, try using torch
        try:
            if torch and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpus = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_info = get_gpu_memory_info()
                    gpus.append({
                        'name': props.name,
                        'memory_total_gb': props.total_memory / (1024**3),
                        'memory_used_gb': memory_info['used'],
                        'memory_free_gb': memory_info['free'],
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                return gpus
        except Exception as e:
            logger.error(f"Error getting GPU info from PyTorch: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
        return None

def check_cuda_env() -> Dict[str, Any]:
    """Check CUDA environment variables and settings"""
    cuda_env = {
        'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES', 'Not set'),
        'CUDA_HOME': os.getenv('CUDA_HOME', 'Not set'),
        'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', 'Not set'),
        'PATH': os.getenv('PATH', 'Not set')
    }
    
    # Check CUDA version
    try:
        if torch and torch.cuda.is_available():
            cuda_env['cuda_version'] = torch.version.cuda
            cuda_env['cudnn_version'] = torch.backends.cudnn.version()
            cuda_env['cuda_arch_list'] = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else None
    except Exception as e:
        logger.error(f"Error getting CUDA version info: {str(e)}")
    
    # Check if CUDA paths exist in LD_LIBRARY_PATH
    if cuda_env['LD_LIBRARY_PATH'] != 'Not set':
        cuda_libs = [p for p in cuda_env['LD_LIBRARY_PATH'].split(':') if 'cuda' in p.lower()]
        cuda_env['cuda_library_paths'] = cuda_libs
    
    return cuda_env

def get_system_info() -> Dict[str, Any]:
    """Get detailed system information"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    info = {
        'os': {
            'system': system,
            'release': platform.release(),
            'version': platform.version(),
            'machine': machine,
            'processor': platform.processor()
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler()
        },
        'cuda': {
            'available': False,
            'version': None,
            'gpus': None,
            'env': check_cuda_env(),
            'memory': get_gpu_memory_info()
        },
        'memory': {},
        'disk': {}
    }
    
    # Check memory
    if psutil:
        memory = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    
    # Check disk space
    disk_info = check_disk_requirements()
    if disk_info:
        info['disk'] = disk_info
    
    # Check for NVIDIA GPU and CUDA
    gpus = get_nvidia_gpu_info()
    if gpus:
        info['cuda']['available'] = True
        info['cuda']['gpus'] = gpus
        info['pytorch_install_type'] = 'cuda'
        
        # Check if GPU meets minimum requirements
        gpu_memory = gpus[0]['memory_total_gb']
        if gpu_memory < MIN_GPU_MEMORY_GB:
            logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is below recommended minimum ({MIN_GPU_MEMORY_GB}GB)")
    
    elif system == 'darwin' and machine == 'arm64':
        info['pytorch_install_type'] = 'mps'
    else:
        info['pytorch_install_type'] = 'cpu'
    
    return info

def check_disk_requirements() -> Optional[Dict[str, Any]]:
    """Check disk space and return detailed information"""
    if psutil is None:
        logger.warning("psutil not available, skipping disk space check")
        return None
    
    try:
        # Get disk space for different relevant directories
        model_path = MODELS_PATH
        log_path = LOGS_PATH
        
        # Create directories if they don't exist
        model_path.mkdir(exist_ok=True, parents=True)
        log_path.mkdir(exist_ok=True, parents=True)
        
        # Get disk info for each path
        model_disk = psutil.disk_usage(str(model_path))
        log_disk = psutil.disk_usage(str(log_path))
        
        # Use the path with less available space for the main check
        disk = model_disk if model_disk.free < log_disk.free else log_disk
        
        disk_info = {
            "total": disk.total / (1024**3),  # GB
            "free": disk.free / (1024**3),  # GB
            "used": disk.used / (1024**3),  # GB
            "percent_used": disk.percent,
            "model_dir": {
                "path": str(model_path),
                "free_gb": model_disk.free / (1024**3),
                "total_gb": model_disk.total / (1024**3)
            },
            "log_dir": {
                "path": str(log_path),
                "free_gb": log_disk.free / (1024**3),
                "total_gb": log_disk.total / (1024**3)
            }
        }
        
        # Check if space requirements are met
        if disk_info["free"] < MIN_FREE_SPACE_GB:
            logger.warning(f"Available disk space ({disk_info['free']:.1f}GB) is below minimum requirement ({MIN_FREE_SPACE_GB}GB)")
        
        free_percent = 100 - disk_info["percent_used"]
        if free_percent < MIN_FREE_SPACE_PERCENT:
            logger.warning(f"Available disk space percentage ({free_percent:.1f}%) is below minimum requirement ({MIN_FREE_SPACE_PERCENT}%)")
        
        return disk_info
        
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return None

def main() -> int:
    """Main function to check system and output information"""
    try:
        info = get_system_info()
        
        # Create output directory if it doesn't exist
        Path('system_info').mkdir(exist_ok=True)
        
        # Save full system info
        with open('system_info/system_check.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        # Check system requirements
        requirements_met = True
        
        # Check GPU/CUDA
        if not info['cuda']['available']:
            logger.error("No CUDA-capable GPU found")
            requirements_met = False
        else:
            gpu_memory = info['cuda']['memory']['total']
            if gpu_memory < MIN_GPU_MEMORY_GB:
                logger.error(f"Insufficient GPU memory: {gpu_memory:.1f}GB (minimum {MIN_GPU_MEMORY_GB}GB required)")
                requirements_met = False
        
        # Check disk space
        if info['disk']:
            if info['disk']['free'] < MIN_FREE_SPACE_GB:
                logger.error(f"Insufficient disk space: {info['disk']['free']:.1f}GB (minimum {MIN_FREE_SPACE_GB}GB required)")
                requirements_met = False
        
        # Output result
        result = {
            'requirements_met': requirements_met,
            'pytorch_install_type': info['pytorch_install_type'],
            'cuda_available': info['cuda']['available'],
            'gpus': info['cuda']['gpus']
        }
        
        print(json.dumps(result))
        return 0 if requirements_met else 1
        
    except Exception as e:
        logger.error(f"System check failed: {str(e)}")
        print(json.dumps({
            'error': str(e),
            'requirements_met': False,
            'pytorch_install_type': 'cpu',
            'cuda_available': False,
            'gpus': None
        }))
        return 1

if __name__ == "__main__":
    sys.exit(main()) 