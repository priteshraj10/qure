import torch
import psutil
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ..config.settings import SystemRequirements

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    name: str
    memory_total: float  # GB
    memory_used: float   # GB
    cuda_capability: Tuple[int, int]
    cuda_version: str

@dataclass
class DiskInfo:
    total: float  # GB
    used: float   # GB
    free: float   # GB
    required: float  # GB

class SystemMonitor:
    def __init__(self, requirements: SystemRequirements):
        self.requirements = requirements
        self._gpu_info: Optional[GPUInfo] = None
        self._disk_info: Optional[DiskInfo] = None

    @property
    def gpu_info(self) -> GPUInfo:
        if self._gpu_info is None:
            self._gpu_info = self._get_gpu_info()
        return self._gpu_info

    @property
    def disk_info(self) -> DiskInfo:
        if self._disk_info is None:
            self._disk_info = self._get_disk_info()
        return self._disk_info

    def _get_gpu_info(self) -> GPUInfo:
        """Get detailed GPU information with error handling"""
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            return GPUInfo(
                name=props.name,
                memory_total=props.total_memory / 1024**3,
                memory_used=torch.cuda.memory_allocated(device) / 1024**3,
                cuda_capability=(props.major, props.minor),
                cuda_version=torch.version.cuda
            )
        except Exception as e:
            logger.error(f"Failed to get GPU info: {str(e)}")
            raise RuntimeError(f"GPU initialization failed: {str(e)}")

    def _get_disk_info(self) -> DiskInfo:
        """Get disk space information"""
        try:
            total, used, free = psutil.disk_usage(str(Path.cwd()))
            return DiskInfo(
                total=total / 1024**3,
                used=used / 1024**3,
                free=free / 1024**3,
                required=self.requirements.min_disk_space
            )
        except Exception as e:
            logger.error(f"Failed to get disk info: {str(e)}")
            raise RuntimeError(f"Disk space check failed: {str(e)}")

    def verify_requirements(self) -> None:
        """Comprehensive system requirements verification"""
        errors = []

        # GPU checks
        try:
            gpu = self.gpu_info
            if gpu.memory_total < self.requirements.min_gpu_memory:
                errors.append(
                    f"Insufficient GPU memory: {gpu.memory_total:.1f}GB < {self.requirements.min_gpu_memory}GB required"
                )
            
            cuda_version = tuple(map(int, gpu.cuda_version.split('.')))
            if cuda_version < self.requirements.min_cuda_version:
                errors.append(
                    f"CUDA version too old: {gpu.cuda_version} < {'.'.join(map(str, self.requirements.min_cuda_version))} required"
                )
        except Exception as e:
            errors.append(f"GPU check failed: {str(e)}")

        # Disk space checks
        try:
            disk = self.disk_info
            if disk.free < disk.required:
                errors.append(
                    f"Insufficient disk space: {disk.free:.1f}GB < {disk.required}GB required"
                )
        except Exception as e:
            errors.append(f"Disk check failed: {str(e)}")

        if errors:
            raise SystemError("\n".join(errors))

    def monitor_resources(self) -> Dict[str, Any]:
        """Real-time resource monitoring"""
        return {
            "gpu": {
                "name": self.gpu_info.name,
                "memory_used_gb": self.gpu_info.memory_used,
                "memory_total_gb": self.gpu_info.memory_total,
                "utilization": torch.cuda.utilization(0)
            },
            "disk": {
                "free_gb": self.disk_info.free,
                "total_gb": self.disk_info.total,
                "used_percent": (self.disk_info.used / self.disk_info.total) * 100
            }
        } 