#!/usr/bin/env python3
import logging
import sys
import torch
import psutil
import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from datetime import datetime
from contextlib import contextmanager

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "unsloth/Llama-3.2-3B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    
    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_steps: int = 5
    weight_decay: float = 0.01
    seed: int = 3407
    max_steps: int = 60
    early_stopping_patience: int = 3
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Output settings
    output_dir: Path = Path("outputs")
    experiment_name: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def save(self, path: Path) -> None:
        """Save config to JSON"""
        config_dict = asdict(self)
        config_dict["output_dir"] = str(config_dict["output_dir"])
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

@contextmanager
def train_session():
    """Context manager for training session"""
    start_time = datetime.now()
    start_memory = torch.cuda.max_memory_reserved() / 1024**3
    
    try:
        yield
    finally:
        end_time = datetime.now()
        end_memory = torch.cuda.max_memory_reserved() / 1024**3
        duration = (end_time - start_time).total_seconds() / 60
        memory_used = end_memory - start_memory
        
        logger = logging.getLogger(__name__)
        logger.info(f"Training duration: {duration:.2f} minutes")
        logger.info(f"Peak memory usage: {end_memory:.2f}GB")
        logger.info(f"Memory used for training: {memory_used:.2f}GB")

def setup_logging(config: TrainingConfig) -> logging.Logger:
    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{config.experiment_name}.log"
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file))
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def verify_system_requirements(logger: logging.Logger) -> Dict[str, Any]:
    """Verify and return system specifications"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    
    gpu_info = torch.cuda.get_device_properties(0)
    memory_gb = gpu_info.total_memory / 1024**3
    ram_gb = psutil.virtual_memory().total / 1024**3
    
    specs = {
        "gpu_name": gpu_info.name,
        "gpu_memory": memory_gb,
        "system_ram": ram_gb,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__
    }
    
    logger.info("System Specifications:")
    for key, value in specs.items():
        logger.info(f"{key}: {value}")
    
    if memory_gb < 8:
        raise RuntimeError(f"Insufficient GPU memory. Required: 8GB, Available: {memory_gb:.2f}GB")
    if ram_gb < 16:
        raise RuntimeError(f"Insufficient RAM. Required: 16GB, Available: {ram_gb:.2f}GB")
    
    return specs

def format_prompts(examples: Dict[str, List], tokenizer) -> Dict[str, List[str]]:
    """Format prompts for training"""
    EOS_TOKEN = tokenizer.eos_token
    alpaca_prompt = """### Instruction:
{0}

### Input:
{1}

### Response:
{2}"""
    
    texts = [
        alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    return {"text": texts}

def main():
    try:
        config = TrainingConfig()
        logger = setup_logging(config)
        
        with train_session():
            # Verify system and save specs
            specs = verify_system_requirements(logger)
            
            # Create output directories
            for subdir in ["models", "checkpoints", "cache", "configs"]:
                (config.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config.save(config.output_dir / "configs" / f"{config.experiment_name}.json")
            
            # Load model and tokenizer
            logger.info("Loading model and tokenizer...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit
            )
            
            # Add LoRA adapters
            logger.info("Adding LoRA adapters...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_r,
                target_modules=config.target_modules,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                use_gradient_checkpointing="unsloth"
            )
            
            # Load and preprocess dataset
            logger.info("Loading and preprocessing dataset...")
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            dataset = dataset.map(
                lambda x: format_prompts(x, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Initialize trainer
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=config.max_seq_length,
                dataset_num_proc=os.cpu_count(),
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=config.batch_size,
                    gradient_accumulation_steps=config.gradient_accumulation,
                    warmup_steps=config.warmup_steps,
                    max_steps=config.max_steps,
                    learning_rate=config.learning_rate,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=config.weight_decay,
                    lr_scheduler_type="linear",
                    seed=config.seed,
                    output_dir=str(config.output_dir / "checkpoints" / config.experiment_name)
                )
            )
            
            # Train model
            logger.info("Starting training...")
            stats = trainer.train()
            
            # Save the model
            logger.info("Saving model...")
            save_dir = config.output_dir / "models" / config.experiment_name
            trainer.save_model(str(save_dir))
            logger.info(f"Model saved to {save_dir}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 