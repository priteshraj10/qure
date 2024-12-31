from dataclasses import dataclass
from typing import Optional, Dict, Any
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
import logging
from pathlib import Path
from ..system.checks import SystemMonitor

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    model_name: str
    max_seq_length: int = 2048
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    output_dir: str = "outputs"

class UnslothTrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.system_monitor = SystemMonitor()
        self._setup_model()
        self._setup_wandb()

    def _setup_model(self):
        """Initialize model with Unsloth optimizations"""
        # Reference to llama_3_2_1b_+_3b_+_unsloth_2x_faster_finetuning.py
        # Lines 63-86 for model setup
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True
        )
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            use_gradient_checkpointing="unsloth"
        )

    def train(self, dataset, callbacks=None):
        """Start training with monitoring"""
        # Reference to llama_3_2_1b_+_3b_+_unsloth_2x_faster_finetuning.py
        # Lines 138-163 for training setup
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=self.config.output_dir
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=callbacks
        )

        return trainer.train() 