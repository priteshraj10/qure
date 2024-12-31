import os
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model():
    """Initialize model and tokenizer"""
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    # Add LoRA adapters
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
        random_state=3407,
    )
    
    return model, tokenizer

def prepare_dataset():
    """Load and prepare the dataset"""
    dataset = load_dataset("unsloth/Radiology_mini", split="train")
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."
    
    converted_dataset = []
    for sample in dataset:
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
    return converted_dataset

def train_model(model, tokenizer, dataset):
    """Configure and run training"""
    FastVisionModel.for_training(model)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="outputs",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )
    
    return trainer.train()

def main():
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Setup model and tokenizer
    logger.info("Initializing model and tokenizer...")
    model, tokenizer = setup_model()
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset()
    
    # Train model
    logger.info("Starting training...")
    training_stats = train_model(model, tokenizer, dataset)
    
    # Save model
    logger.info("Saving model...")
    model.save_pretrained("outputs/lora_model")
    tokenizer.save_pretrained("outputs/lora_model")
    
    logger.info(f"Training completed in {training_stats.metrics['train_runtime']} seconds")

if __name__ == "__main__":
    main()