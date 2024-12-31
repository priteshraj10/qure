import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct"
    output_dir: str = "outputs"
    load_in_4bit: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    seed: int = 3407

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    bias: str = "none"
    random_state: int = 3407

class RadioVisionTrainer:
    def __init__(self, training_config: TrainingConfig, lora_config: LoRAConfig):
        self.training_config = training_config
        self.lora_config = lora_config
        self.output_dir = Path(training_config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model()
        
    def _initialize_model(self) -> Tuple[Any, Any]:
        try:
            model, tokenizer = FastVisionModel.from_pretrained(
                self.training_config.model_name,
                load_in_4bit=self.training_config.load_in_4bit,
                use_gradient_checkpointing="unsloth"
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def setup_lora(self) -> None:
        try:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                random_state=self.lora_config.random_state
            )
        except Exception as e:
            logger.error(f"LoRA setup failed: {str(e)}")
            raise

    @staticmethod
    def prepare_dataset(dataset_name: str = "unsloth/Radiology_mini") -> List[Dict]:
        try:
            dataset = load_dataset(dataset_name, split="train")
            instruction = "You are an expert radiographer. Describe accurately what you see in this image."
            
            return [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image", "image": sample["image"]}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": sample["caption"]}
                            ]
                        }
                    ]
                }
                for sample in dataset
            ]
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise

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
                    per_device_train_batch_size=self.training_config.batch_size,
                    gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                    warmup_steps=self.training_config.warmup_steps,
                    max_steps=self.training_config.max_steps,
                    learning_rate=self.training_config.learning_rate,
                    fp16=not is_bf16_supported(),
                    bf16=is_bf16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=self.training_config.weight_decay,
                    lr_scheduler_type="linear",
                    seed=self.training_config.seed,
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    dataset_num_proc=4,
                    max_seq_length=self.training_config.max_seq_length,
                )
            )
            
            return trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def save_model(self, push_to_hub: bool = False, hub_token: Optional[str] = None) -> None:
        try:
            # Save locally
            self.model.save_pretrained(self.output_dir / "lora_model")
            self.tokenizer.save_pretrained(self.output_dir / "lora_model")
            
            # Push to HuggingFace Hub if requested
            if push_to_hub and hub_token:
                self.model.push_to_hub(
                    f"{os.environ.get('HF_USERNAME', 'default')}/lora_model",
                    token=hub_token
                )
                self.tokenizer.push_to_hub(
                    f"{os.environ.get('HF_USERNAME', 'default')}/lora_model",
                    token=hub_token
                )
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

def main():
    try:
        # Initialize configurations
        training_config = TrainingConfig()
        lora_config = LoRAConfig()

        # Initialize trainer
        trainer = RadioVisionTrainer(training_config, lora_config)
        
        # Setup LoRA
        trainer.setup_lora()
        
        # Prepare dataset
        dataset = trainer.prepare_dataset()
        
        # Train model
        training_stats = trainer.train(dataset)
        logger.info(f"Training completed in {training_stats.metrics['train_runtime']} seconds")
        
        # Save model
        trainer.save_model()
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 