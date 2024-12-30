from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    device: str
    max_seq_length: int = 2048
    warmup_steps: int = 100
    steps_per_epoch: int = 1000
    mixed_precision: bool = True 