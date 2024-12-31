import torch
import psutil
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def verify_system_requirements():
    """Verify GPU and system requirements"""
    # Reference to llama_3_2_1b_+_3b_+_unsloth_2x_faster_finetuning.py for GPU checks
    startLine: 165
    endLine: 170
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    gpu_info = torch.cuda.get_device_properties(0)
    memory_gb = gpu_info.total_memory / 1024**3
    
    logger.info(f"GPU: {gpu_info.name}")
    logger.info(f"Total GPU Memory: {memory_gb:.2f} GB")
    
    return True 