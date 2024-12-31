from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
import wandb
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    # Core model settings
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
    max_steps: int = 60  # Added from llama example
    early_stopping_patience: int = 3
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    
    # Paths and directories
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")
    log_dir: Path = Path("logs")

class UnslothTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_environment()
        self._setup_wandb()
        self._init_memory_tracking()
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def _setup_environment(self):
        """Setup directories and environment variables"""
        # Reference to train_model.py for environment setup
        ```python:train_model.py
        startLine: 40
        endLine: 53
        ```

    def _setup_wandb(self):
        wandb.init(
            project="llm-training",
            name=f"training-{wandb.util.generate_id()}",
            config=self.config.__dict__
        )

    def _init_memory_tracking(self):
        self.gpu_stats = torch.cuda.get_device_properties(0)
        self.start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        self.max_memory = round(self.gpu_stats.total_memory / 1024**3, 3)
        self.logger.info(f"GPU = {self.gpu_stats.name}. Max memory = {self.max_memory} GB")
        self.logger.info(f"Starting memory usage: {self.start_gpu_memory} GB")

    def train(self):
        try:
            # Initialize model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit
            )

            # Add LoRA adapters with improved settings
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                use_gradient_checkpointing="unsloth"
            )

            # Prepare dataset with improved processing
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            dataset = dataset.map(
                self.format_prompt,
                batched=True,
                num_proc=2,
                remove_columns=dataset.column_names
            )

            # Setup trainer with optimized arguments
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir / "checkpoints"),
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=self.config.weight_decay,
                lr_scheduler_type="linear",
                seed=self.config.seed,
                report_to="wandb",
                save_strategy="steps",
                save_steps=20,
                evaluation_strategy="steps",
                eval_steps=20,
                load_best_model_at_end=True,
                metric_for_best_model="loss"
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                dataset_num_proc=2,
                packing=False,
                args=training_args
            )

            # Custom training loop with early stopping
            def early_stopping_callback(args, state, control, **kwargs):
                if state.best_metric is not None:
                    if state.best_metric < self.best_loss:
                        self.best_loss = state.best_metric
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                        
                    if self.no_improvement_count >= self.config.early_stopping_patience:
                        control.should_training_stop = True
                return control

            trainer.add_callback(early_stopping_callback)

            # Train and monitor
            self.logger.info("Starting training...")
            trainer_stats = trainer.train()
            self._log_training_stats(trainer_stats)
            
            # Save model in different formats
            self._save_model(trainer, model, tokenizer)
            
            return trainer_stats

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            wandb.finish()

    def _log_training_stats(self, trainer_stats):
        """Log training statistics"""
        # Reference to llama example for memory stats
        ```python:llama_3_2_1b_+_3b_+_unsloth_2x_faster_finetuning.py
        startLine: 177
        endLine: 184
        ```

    def _save_model(self, trainer, model, tokenizer):
        """Save model in different formats"""
        # Save standard model
        trainer.save_model(str(self.config.output_dir / "final_model"))
        
        # Save merged model in different formats
        # Reference to llama example for saving formats
        ```python:llama_3_2_1b_+_3b_+_unsloth_2x_faster_finetuning.py
        startLine: 280
        endLine: 289
        ``` 