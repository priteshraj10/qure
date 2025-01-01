import os
import logging
from pathlib import Path
import torch
import gc
from datasets import load_dataset, concatenate_datasets
from transformers import TextStreamer, set_seed
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct"
    output_dir: str = "outputs"
    load_in_4bit: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    seed: int = 3407
    num_train_epochs: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    mixed_precision: str = field(default="bf16" if is_bf16_supported() else "fp16")
    cache_dir: Optional[str] = field(default="cache")
    max_grad_norm: float = 0.3

def setup_environment(config: TrainingConfig):
    """Setup training environment"""
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def prepare_datasets():
    """Load and combine datasets with error handling"""
    try:
        logger.info("Loading datasets...")
        radio_dataset = load_dataset("unsloth/Radiology_mini", split="train")
        pubmed_dataset = load_dataset("FreedomIntelligence/PubMedVision", split="train")
        
        def convert_pubmed(example):
            try:
                return {
                    "image": example["image"],
                    "caption": example["text"]
                }
            except Exception as e:
                logger.warning(f"Failed to convert example: {str(e)}")
                return None
        
        pubmed_dataset = pubmed_dataset.map(
            convert_pubmed,
            remove_columns=pubmed_dataset.column_names,
            num_proc=4
        )
        pubmed_dataset = pubmed_dataset.filter(lambda x: x is not None)
        
        combined_dataset = concatenate_datasets([radio_dataset, pubmed_dataset])
        logger.info(f"Combined dataset size: {len(combined_dataset)} samples")
        
        return combined_dataset
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise

def convert_to_conversation(dataset):
    """Convert dataset to conversation format with progress bar"""
    instruction = "You are an expert medical professional. Describe accurately what you see in this image."
    
    converted_dataset = []
    for sample in tqdm(dataset, desc="Converting dataset"):
        try:
            conversation = {
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
            converted_dataset.append(conversation)
        except Exception as e:
            logger.warning(f"Failed to convert sample: {str(e)}")
            continue
    
    return converted_dataset

def setup_model(config: TrainingConfig):
    """Initialize model with memory optimization"""
    logger.info(f"Loading base model: {config.model_name}")
    
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            config.model_name,
            load_in_4bit=config.load_in_4bit,
            use_gradient_checkpointing="unsloth",
            cache_dir=config.cache_dir,
        )
        
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=config.seed,
        )
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

def train_model(model, tokenizer, dataset, config: TrainingConfig):
    """Train model with improved configuration"""
    logger.info("Starting training...")
    
    try:
        FastVisionModel.for_training(model)
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=config.output_dir,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                fp16=config.mixed_precision == "fp16",
                bf16=config.mixed_precision == "bf16",
                logging_steps=config.logging_steps,
                optim="adamw_8bit",
                weight_decay=config.weight_decay,
                lr_scheduler_type="linear",
                seed=config.seed,
                save_steps=config.save_steps,
                eval_steps=config.eval_steps,
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=config.max_seq_length,
                max_grad_norm=config.max_grad_norm,
                gradient_checkpointing=True,
            ),
        )
        
        return trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def inference(model, tokenizer, image, instruction):
    FastVisionModel.for_inference(model)
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    return model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=128,
        use_cache=True, 
        temperature=1.5, 
        min_p=0.1
    )

def main():
    try:
        config = TrainingConfig()
        setup_environment(config)
        
        dataset = prepare_datasets()
        converted_dataset = convert_to_conversation(dataset)
        
        model, tokenizer = setup_model(config)
        training_stats = train_model(model, tokenizer, converted_dataset, config)
        
        save_path = f"{config.output_dir}/final_model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        test_image = dataset[0]["image"]
        test_instruction = "You are an expert radiographer. Describe accurately what you see in this image."
        inference(model, tokenizer, test_image, test_instruction)
        
        runtime_seconds = training_stats.metrics['train_runtime']
        logger.info(f"Training completed in {runtime_seconds} seconds")
        logger.info(f"Minutes used for training: {round(runtime_seconds/60, 2)}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
