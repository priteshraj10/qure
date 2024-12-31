from pathlib import Path

training_config = {
    # Core model settings
    "model_name": "unsloth/Llama-3.2-3B",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # Training hyperparameters
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "num_epochs": 5,
    "warmup_steps": 5,
    "max_steps": 60,
    "early_stopping_patience": 3,
    
    # LoRA settings
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    
    # Paths
    "output_dir": Path("outputs")
} 