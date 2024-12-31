#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import torch
import psutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datasets import load_dataset
from tqdm import tqdm

@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Llama-3.2-3B"
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_steps: int = 5
    output_dir: Path = Path("outputs")
    seed: int = 3407
    load_in_4bit: bool = True
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    mixed_precision: bool = True

class UnslothTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        self.setup_logging()
        self.best_loss = float('inf')
        self.early_stopping_patience = 3
        self.no_improvement_count = 0
        
    def setup_environment(self) -> None:
        # Reference to start.sh for environment setup
        # startLine: 148
        # endLine: 153
        
        for path in ["models", "logs", "cache", "checkpoints"]:
            (self.config.output_dir / path).mkdir(parents=True, exist_ok=True)
            
        os.environ.update({
            "TORCH_HOME": str(self.config.output_dir / "cache"),
            "HF_HOME": str(self.config.output_dir / "cache"),
            "TRANSFORMERS_CACHE": str(self.config.output_dir / "cache"),
            "WANDB_DIR": str(self.config.output_dir / "logs")
        })

    def setup_logging(self) -> None:
        log_file = self.config.output_dir / "logs" / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(log_file))
            ]
        )
        self.logger = logging.getLogger(__name__)

    def verify_system(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu_info = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_info.total_memory / 1024**3, 3)
        
        self.logger.info(f"GPU: {gpu_info.name}")
        self.logger.info(f"Total GPU Memory: {max_memory}GB")
        self.logger.info(f"Reserved Memory: {start_gpu_memory}GB")

    def train(self):
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from transformers import TrainingArguments
        from trl import SFTTrainer
        import wandb

        # Initialize wandb
        wandb.init(
            project="medical-llm",
            name=f"training-{wandb.util.generate_id()}",
            config=self.config.__dict__
        )

        try:
            # Initialize model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit
            )

            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                use_gradient_checkpointing="unsloth"
            )

            # Prepare dataset
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")

            # Setup trainer with optimized arguments
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir / "checkpoints"),
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                warmup_steps=self.config.warmup_steps,
                num_train_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                seed=self.config.seed,
                optim="adamw_8bit",
                weight_decay=self.config.weight_decay,
                gradient_checkpointing=self.config.gradient_checkpointing
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                args=training_args
            )

            # Train and monitor
            self.logger.info("Starting training...")
            result = trainer.train()
            
            # Log final metrics
            used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            self.logger.info(f"Training completed in {result.metrics['train_runtime']:.2f} seconds")
            self.logger.info(f"Peak GPU memory usage: {used_memory}GB")
            
            # Save model
            trainer.save_model(str(self.config.output_dir / "final_model"))
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            wandb.finish()

def main():
    config = TrainingConfig()
    trainer = UnslothTrainer(config)
    
    try:
        trainer.verify_system()
        trainer.train()
    except Exception as e:
        logging.error(f"Training process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 