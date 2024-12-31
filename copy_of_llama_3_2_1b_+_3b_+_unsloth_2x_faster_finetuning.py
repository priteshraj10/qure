# -*- coding: utf-8 -*-
import torch
import logging
import platform
import subprocess
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TrainingArguments
from unsloth import FastLanguageModel, FastLanguageModelForCausalLM, is_bfloat16_supported
from trl import SFTTrainer
from datasets import load_dataset
from PIL import Image
import os
import wandb
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_requirements():
    """Verify GPU availability and requirements"""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for training.")
        
        gpu_info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
        
        logger.info("GPU Requirements Check:")
        logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
        logger.info(f"GPU Count: {gpu_info['gpu_count']}")
        logger.info(f"GPU Name: {gpu_info['gpu_name']}")
        logger.info(f"GPU Memory: {gpu_info['gpu_memory']:.2f} GB")
        
        return gpu_info
    
    except Exception as e:
        logger.error(f"GPU check failed: {str(e)}")
        raise

def load_model(model_name, max_seq_length=2048, load_in_4bit=True):
    """Load model based on name with appropriate configurations"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer

def setup_peft_model(model):
    """Setup PEFT configuration"""
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model

def process_medical_dataset():
    """Process medical dataset for training"""
    dataset = load_dataset("FreedomIntelligence/PubMedVision")
    return dataset["train"]

def train_model(
    model_name="meditron-7b",
    batch_size=2,
    num_epochs=1,
    learning_rate=2e-4,
    output_dir="outputs"
):
    # Check GPU requirements
    gpu_info = check_gpu_requirements()
    
    # Initialize wandb
    wandb.init(
        project="medical-llm",
        name=f"gpu-training-{wandb.util.generate_id()}",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "gpu_name": gpu_info["gpu_name"]
        }
    )
    
    # Load model and setup training
    model, tokenizer = load_model(model_name)
    model = setup_peft_model(model)
    
    # Load dataset
    dataset = process_medical_dataset()
    
    # Setup training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_dir,
        report_to="wandb",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )
    
    # Log initial memory usage
    start_gpu_memory = torch.cuda.max_memory_reserved() / (1024**3)
    logger.info(f"Initial GPU memory usage: {start_gpu_memory:.2f} GB")
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Log final stats
    used_memory = torch.cuda.max_memory_reserved() / (1024**3)
    training_time = trainer_stats.metrics['train_runtime']
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Peak GPU memory usage: {used_memory:.2f} GB")
    
    # Save the model
    Path(output_dir).mkdir(exist_ok=True)
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    return model, tokenizer

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_model(
        model_name="meditron-7b",
        batch_size=2,
        num_epochs=1,
        learning_rate=2e-4,
        output_dir="medical_model_outputs"
    )