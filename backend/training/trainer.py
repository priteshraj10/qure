import torch
import logging
import wandb
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from ..config.settings import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MAX_LENGTH,
    WARMUP_STEPS, GRADIENT_CLIP_VAL, SCHEDULER_TYPE
)

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Training implementation based on train.py"""
    # Reference to train.py lines 360-372 for trainer configuration
    def __init__(self, model_name: str = "meditron-7b"):
        self.model_name = model_name
        self.device = torch.device("cuda")
        self._setup_model()
        
    def _setup_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=MAX_LENGTH,
            load_in_4bit=True
        ) 