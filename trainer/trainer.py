import logging
from pathlib import Path
from typing import Dict, List
from transformers import TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def train(self, dataset: List[Dict]) -> Dict:
        try:
            FastVisionModel.for_training(self.model)
            
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
                train_dataset=dataset,
                args=SFTConfig(
                    output_dir=str(self.output_dir),
                    per_device_train_batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    warmup_steps=self.config.warmup_steps,
                    max_steps=self.config.max_steps,
                    learning_rate=self.config.learning_rate,
                    fp16=not is_bf16_supported(),
                    bf16=is_bf16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=self.config.weight_decay,
                    lr_scheduler_type="linear",
                    seed=self.config.seed,
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    dataset_num_proc=4,
                    max_seq_length=self.config.max_seq_length,
                )
            )
            
            return trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise 