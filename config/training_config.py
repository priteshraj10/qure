from pathlib import Path

training_config = {
    # Model settings
    "model_name": "unsloth/Llama-3.2-3B",  # or choose smaller "unsloth/Llama-3.2-1B"
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # Training hyperparameters
    "batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 5,
    "max_steps": 60,  # or use num_epochs for full training
    "weight_decay": 0.01,
    
    # LoRA settings
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    
    # Output settings
    "output_dir": Path("outputs"),
    "save_steps": 20,
    "logging_steps": 1
} 