#!/usr/bin/env python3
import logging
import sys
import torch
import psutil
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from datetime import datetime

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

def setup_logging(config: TrainingConfig) -> logging.Logger:
    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{config.experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )
    return logging.getLogger(__name__)

def verify_system_requirements(logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    
    gpu_info = torch.cuda.get_device_properties(0)
    memory_gb = gpu_info.total_memory / 1024**3
    ram_gb = psutil.virtual_memory().total / 1024**3
    
    logger.info(f"GPU: {gpu_info.name}")
    logger.info(f"GPU Memory: {memory_gb:.2f} GB")
    logger.info(f"System RAM: {ram_gb:.2f} GB")
    
    if memory_gb < 8:
        raise RuntimeError(f"Insufficient GPU memory. Required: 8GB, Available: {memory_gb:.2f}GB")
    if ram_gb < 16:
        raise RuntimeError(f"Insufficient RAM. Required: 16GB, Available: {ram_gb:.2f}GB")

def format_prompts(examples, tokenizer):
    EOS_TOKEN = tokenizer.eos_token
    alpaca_prompt = """### Instruction:
{0}

### Input:
{1}

### Response:
{2}"""
    
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def main():
    try:
        # Initialize configuration and logging
        config = TrainingConfig()
        logger = setup_logging(config)
        
        # Verify system requirements
        verify_system_requirements(logger)
        start_memory = torch.cuda.max_memory_reserved() / 1024**3
        
        # Create output directories
        for subdir in ["models", "checkpoints", "cache"]:
            (config.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
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
        
        # Log final statistics
        used_memory = torch.cuda.max_memory_reserved() / 1024**3
        used_memory_for_lora = used_memory - start_memory
        logger.info(f"Training completed in {stats.metrics['train_runtime']/60:.2f} minutes")
        logger.info(f"Peak memory usage: {used_memory:.2f}GB")
        logger.info(f"Memory used for LoRA: {used_memory_for_lora:.2f}GB")
        
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