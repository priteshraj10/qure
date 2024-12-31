#!/usr/bin/env python3
import click
import sys
import torch
import psutil
import os
import json
import wandb
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
import logging
from transformers import TextStreamer

console = Console()

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
    save_steps: int = 20
    logging_steps: int = 1
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            config_dict = asdict(self)
            config_dict["output_dir"] = str(config_dict["output_dir"])
            json.dump(config_dict, f, indent=4)

class UnslothTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_environment()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.logger = logging.getLogger("rich")
        
    def setup_environment(self):
        # Create directories
        for path in ["models", "logs", "cache", "checkpoints"]:
            (self.config.output_dir / path).mkdir(parents=True, exist_ok=True)
            
        # Set environment variables
        os.environ.update({
            "TORCH_HOME": str(self.config.output_dir / "cache"),
            "HF_HOME": str(self.config.output_dir / "cache"),
            "TRANSFORMERS_CACHE": str(self.config.output_dir / "cache"),
            "WANDB_DIR": str(self.config.output_dir / "logs")
        })

    def train(self):
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import load_dataset

        try:
            # Initialize model
            self.logger.info("Initializing model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit
            )

            # Add LoRA adapters
            self.logger.info("Adding LoRA adapters...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                use_gradient_checkpointing="unsloth"
            )

            # Prepare dataset with proper formatting
            self.logger.info("Loading dataset...")
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            
            # Format prompts using the template from the example
            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

            def format_prompts(examples):
                texts = []
                for instruction, input_text, output in zip(
                    examples["instruction"], 
                    examples["input"], 
                    examples["output"]
                ):
                    text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

            # Initialize trainer
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir / "checkpoints" / self.config.experiment_name),
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                optim="adamw_8bit",
                weight_decay=self.config.weight_decay,
                lr_scheduler_type="linear",
                seed=self.config.seed,
                report_to="wandb"
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                dataset_num_proc=os.cpu_count(),
                packing=False,
                args=training_args
            )

            # Train and save
            self.logger.info("Starting training...")
            stats = trainer.train()
            
            # Save model
            save_dir = self.config.output_dir / "models" / self.config.experiment_name
            trainer.save_model(str(save_dir))
            
            # Log final stats
            self.log_training_stats(stats)
            
            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def log_training_stats(self, stats):
        gpu_stats = torch.cuda.get_device_properties(0)
        used_memory = torch.cuda.max_memory_reserved() / 1024**3
        
        self.logger.info(f"Training completed in {stats.metrics['train_runtime']/60:.2f} minutes")
        self.logger.info(f"Peak memory usage: {used_memory:.2f}GB")
        self.logger.info(f"GPU: {gpu_stats.name}")

@click.group()
def cli():
    """Unsloth Training CLI"""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
def train(config):
    """Train the model"""
    if config:
        with open(config) as f:
            config_dict = json.load(f)
            config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    trainer = UnslothTrainer(config)
    trainer.train()

if __name__ == "__main__":
    cli() 