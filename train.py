import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Dict, Any
import sys
import psutil
import os
from backend.system_check import (
    check_system_requirements,
    check_disk_requirements,
    is_disk_space_sufficient,
    estimate_required_space,
    get_system_info,
    MIN_FREE_SPACE_GB,
    MIN_FREE_SPACE_PERCENT
)

# Get paths from environment variables or use defaults
MODELS_PATH = Path(os.getenv('MODELS_PATH', 'models'))
LOGS_PATH = Path(os.getenv('LOGS_PATH', 'logs'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOGS_PATH / 'training.log'))
    ]
)
logger = logging.getLogger(__name__)

class GPUTrainer:
    def __init__(self, model_name: str = "meditron-7b"):
        # Check CUDA availability with detailed error reporting
        logger.info("Checking GPU availability...")
        logger.info(f"CUDA is available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
        try:
            gpu_count = torch.cuda.device_count()
            logger.info(f"Number of GPUs available: {gpu_count}")
            
            if gpu_count > 0:
                for i in range(gpu_count):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            else:
                raise RuntimeError("No GPU devices found")
                
        except Exception as e:
            logger.error(f"Error checking GPU: {str(e)}")
            raise RuntimeError(f"GPU initialization failed: {str(e)}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for training")
        
        # Try to set the device and catch any errors
        try:
            self.device = torch.device("cuda")
            # Test CUDA device
            test_tensor = torch.tensor([1.0], device=self.device)
            logger.info(f"Successfully created test tensor on GPU: {test_tensor.device}")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA device: {str(e)}")
            raise RuntimeError(f"GPU initialization failed: {str(e)}")
        
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize model and move to GPU
        try:
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Moving model to GPU...")
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Model loaded and moved to GPU successfully")
        except Exception as e:
            logger.error(f"Failed to load or move model to GPU: {str(e)}")
            raise
        
        # Create output directories using environment paths
        MODELS_PATH.mkdir(exist_ok=True)
        LOGS_PATH.mkdir(exist_ok=True)
        
        # Initialize wandb with custom directory for offline runs
        os.environ["WANDB_DIR"] = str(LOGS_PATH)
        wandb.init(project="medical-llm", name=f"gpu-training-{wandb.util.generate_id()}")
        
    def prepare_data(self, batch_size: int = 4):
        """Prepare data loaders ensuring data is on GPU"""
        dataset = load_dataset("medical_dialogs")
        
        def prepare_batch(batch):
            inputs = self.tokenizer(
                batch["text"], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
        
        train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=prepare_batch
        )
        
        return train_loader
    
    def check_disk_space(self, num_epochs: int) -> None:
        """Check if there's enough disk space for training"""
        disk_info = check_disk_requirements()
        required_space = estimate_required_space(num_epochs)
        
        if disk_info["free"] < required_space:
            raise RuntimeError(
                f"Insufficient disk space for {num_epochs} epochs. "
                f"Required: {required_space:.1f}GB, Available: {disk_info['free']:.1f}GB"
            )
        
        logger.info(f"Disk space check passed. Required: {required_space:.1f}GB, Available: {disk_info['free']:.1f}GB")
    
    def monitor_disk_space(self) -> bool:
        """Monitor disk space during training
        
        Returns:
            bool: True if disk space is sufficient, False otherwise
        """
        disk_info = check_disk_requirements()
        if not is_disk_space_sufficient(disk_info):
            logger.error(
                f"Low disk space detected! Free space: {disk_info['free']:.1f}GB "
                f"({100 - disk_info['percent_used']:.1f}% free)"
            )
            return False
        return True

    def train(self, num_epochs: int = 5, learning_rate: float = 2e-5):
        """Train the model using GPU acceleration"""
        try:
            # Check if we have enough disk space for training
            self.check_disk_space(num_epochs)
            
            train_loader = self.prepare_data()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            scaler = torch.cuda.amp.GradScaler()
            
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Monitor disk space every 100 batches
                    if batch_idx % 100 == 0 and not self.monitor_disk_space():
                        raise RuntimeError("Training stopped due to low disk space")
                    
                    self.optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    
                    # Log metrics
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3,
                        "disk_space_free_gb": psutil.disk_usage('/').free / (1024**3)
                    })
                    
                    progress_bar.set_postfix({"loss": loss.item()})
                
                # Check disk space before saving checkpoint
                if not self.monitor_disk_space():
                    raise RuntimeError("Cannot save checkpoint due to low disk space")
                
                self.save_checkpoint(epoch)
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
        finally:
            # Cleanup
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint
        
        Args:
            epoch (int): Current training epoch
        """
        checkpoint_path = MODELS_PATH / f"checkpoint-{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    # Verify system requirements before starting
    try:
        system_info = get_system_info()
        if not system_info['cuda']['available']:
            raise RuntimeError("CUDA is not available. GPU is required for training.")
        logger.info(f"Using GPU: {system_info['cuda']['gpus'][0]['name']}")
    except Exception as e:
        logger.error(f"System check failed: {str(e)}")
        raise
    
    # Start training
    trainer = GPUTrainer()
    trainer.train() 