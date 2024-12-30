import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Any
import time

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_epochs * config.steps_per_epoch
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize wandb
        wandb.init(
            project="qure-medical",
            config=vars(config),
            name=f"run_{wandb.util.generate_id()}"
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(batch)
                loss = outputs["loss"]
                
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
                
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            total_loss += loss.item()
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
            # Log metrics
            if step % self.config.log_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": step,
                })
                
            progress_bar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                loss = outputs["loss"]
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader)
        wandb.log({"val/loss": avg_loss})
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": vars(self.config)
        }
        
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        wandb.save(str(path)) 
    
    async def train_with_events(self, train_loader, val_loader, event_emitter):
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = await self._train_epoch(train_loader, event_emitter)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            
            # Combine metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "learning_rate": self.scheduler.get_last_lr()[0],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"]
            }
            
            # Save checkpoint if best model
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, val_metrics["loss"])
            
            # Emit metrics
            await event_emitter.emit_metrics(metrics)
            
            # Early stopping check
            if self._should_stop(val_metrics["loss"]):
                await event_emitter.emit_status("completed", {"reason": "early_stopping"})
                break
                
        await event_emitter.emit_status("completed", {"reason": "finished"}) 