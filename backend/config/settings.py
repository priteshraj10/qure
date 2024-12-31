import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.getenv('INSTALL_PATH', '.'))
MODELS_PATH = BASE_DIR / 'models'
LOGS_PATH = BASE_DIR / 'logs'
CACHE_PATH = BASE_DIR / 'cache'

# Training settings
BATCH_SIZE = 2
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_LENGTH = 2048
WARMUP_STEPS = 100
GRADIENT_CLIP_VAL = 1.0
SCHEDULER_TYPE = "cosine"

# System requirements
MIN_GPU_MEMORY = 8  # GB
MIN_CUDA_VERSION = "11.8"
MIN_DISK_SPACE = 50  # GB

# Environment variables
ENV_VARS = {
    "TORCH_HOME": str(CACHE_PATH),
    "HF_HOME": str(CACHE_PATH),
    "TRANSFORMERS_CACHE": str(CACHE_PATH),
    "WANDB_DIR": str(LOGS_PATH),
} 