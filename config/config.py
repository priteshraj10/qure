from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct"
    output_dir: str = "outputs"
    load_in_4bit: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    seed: int = 3407

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    bias: str = "none"
    random_state: int = 3407 