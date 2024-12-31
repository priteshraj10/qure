from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Llama-3.2-3B"
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_steps: int = 5
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    weight_decay: float = 0.01
    seed: int = 3407

@dataclass
class SystemConfig:
    min_memory_gb: float = 16.0
    min_disk_space_gb: float = 50.0
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs") 