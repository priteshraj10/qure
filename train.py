import torch
import logging
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Dict, Any, Optional
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
    def __init__(
        self, 
        model_name: str = "meditron-7b",
        gradient_checkpointing: bool = True,
        mixed_precision: bool = True
    ):
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
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if mixed_precision else torch.float32,
                use_cache=not gradient_checkpointing
            )
            
            # Enable gradient checkpointing if requested
            if gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            logger.info("Moving model to GPU...")
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Model loaded and moved to GPU successfully")
            
            # Log model size
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"Model size: {model_size:.2f}B parameters")
            
        except Exception as e:
            logger.error(f"Failed to load or move model to GPU: {str(e)}")
            raise
        
        # Create output directories using environment paths
        MODELS_PATH.mkdir(exist_ok=True)
        LOGS_PATH.mkdir(exist_ok=True)
        
        # Initialize wandb with custom directory for offline runs
        os.environ["WANDB_DIR"] = str(LOGS_PATH)
        wandb.init(
            project="medical-llm",
            name=f"gpu-training-{wandb.util.generate_id()}",
            config={
                "model_name": model_name,
                "gradient_checkpointing": gradient_checkpointing,
                "mixed_precision": mixed_precision,
                "model_size_b": model_size
            }
        )
        
        # Initialize training state
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        self.early_stopping_patience = 3
        
    def prepare_data(self, batch_size: int = 4, max_length: int = 512):
        """Prepare data loaders ensuring data is on GPU"""
        dataset = load_dataset("medical_dialogs")
        
        def prepare_batch(batch):
            inputs = self.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            return {k: v.to(self.device) for k, v in inputs.items()}
        
        train_loader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=prepare_batch,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader
    
    def train(
        self,
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        max_length: int = 512,
        warmup_steps: int = 100,
        gradient_clip_val: float = 1.0,
        scheduler_type: str = "cosine"
    ):
        """Train the model using GPU acceleration"""
        try:
            # Check if we have enough disk space for training
            self.check_disk_space(num_epochs)
            
            train_loader = self.prepare_data(batch_size, max_length)
            num_training_steps = len(train_loader) * num_epochs
            
            # Initialize optimizer with weight decay
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
            
            # Initialize learning rate scheduler
            if scheduler_type == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, 
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:
                scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps
                )
            
            # Initialize gradient scaler for mixed precision
            scaler = torch.cuda.amp.GradScaler()
            
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Monitor disk space every 100 batches
                    if batch_idx % 100 == 0 and not self.monitor_disk_space():
                        raise RuntimeError("Training stopped due to low disk space")
                    
                    self.optimizer.zero_grad()
                    
                    try:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**batch)
                            loss = outputs.loss
                        
                        scaler.scale(loss).backward()
                        
                        # Clip gradients
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                        
                        scaler.step(self.optimizer)
                        scaler.update()
                        scheduler.step()
                        
                        # Update metrics
                        total_loss += loss.item()
                        current_lr = scheduler.get_last_lr()[0]
                        
                        # Log metrics
                        wandb.log({
                            "batch_loss": loss.item(),
                            "epoch": epoch,
                            "learning_rate": current_lr,
                            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3,
                            "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3,
                            "disk_space_free_gb": psutil.disk_usage('/').free / (1024**3)
                        })
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{current_lr:.2e}"
                        })
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.memory_allocated() > 0:
                                torch.cuda.empty_cache()
                            logger.error(f"GPU OOM in batch {batch_idx}. Skipping batch.")
                            continue
                        else:
                            raise e
                
                # Calculate average loss for the epoch
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs} Average Loss: {avg_loss:.4f}")
                
                # Early stopping check
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.no_improvement_count = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.no_improvement_count += 1
                    if self.no_improvement_count >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
                
                # Regular checkpoint save
                if not self.monitor_disk_space():
                    raise RuntimeError("Cannot save checkpoint due to low disk space")
                self.save_checkpoint(epoch)
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            wandb.finish()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint
        
        Args:
            epoch (int): Current training epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'no_improvement_count': self.no_improvement_count
        }
        
        # Save regular checkpoint
        checkpoint_path = MODELS_PATH / f"checkpoint-{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model if this is the best checkpoint
        if is_best:
            best_path = MODELS_PATH / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
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
    
    # Start training with improved settings
    trainer = GPUTrainer(
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        mixed_precision=True  # Enable mixed precision training
    )
    trainer.train(
        num_epochs=5,
        learning_rate=2e-5,
        batch_size=4,
        max_length=512,
        warmup_steps=100,
        gradient_clip_val=1.0,
        scheduler_type="cosine"  # Use cosine learning rate scheduler
    ) 