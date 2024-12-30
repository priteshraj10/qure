import torch
from torch.cuda.amp import autocast, GradScaler
import logging
from tqdm import tqdm
from .model import MedicalLLM
from .config import TrainingConfig
from .data_loader import create_dataloaders
from .events import TrainingEventEmitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = MedicalLLM(config)
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Get dataloaders
        self.train_loader, self.val_loader = create_dataloaders(config)

    async def train(self, event_emitter: TrainingEventEmitter):
        try:
            logger.info("Starting training pipeline...")
            best_val_loss = float('inf')
            
            for epoch in range(self.config.num_epochs):
                # Training phase
                train_metrics = await self._train_epoch(epoch, event_emitter)
                
                # Validation phase
                val_metrics = await self._validate_epoch(epoch)
                
                # Combine metrics
                metrics = {
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader),
                    "totalSteps": self.config.num_epochs * len(self.train_loader),
                    "loss": train_metrics["loss"],
                    "accuracy": train_metrics["accuracy"],
                    "validationLoss": val_metrics["loss"],
                    "validationAccuracy": val_metrics["accuracy"],
                    "learningRate": self.optimizer.param_groups[0]["lr"]
                }
                
                # Emit metrics
                await event_emitter.emit_metrics(metrics)
                
                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, val_metrics["loss"])
                
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} completed")
            
            await event_emitter.emit_status("completed", {"message": "Training completed successfully"})
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            await event_emitter.emit_status("error", {"error": str(e)})
            raise

    async def _train_epoch(self, epoch: int, event_emitter: TrainingEventEmitter):
        self.model.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                with autocast(enabled=self.config.mixed_precision):
                    loss, accuracy = self._process_batch(batch)
                    
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                batch_metrics = {
                    "epoch": epoch,
                    "step": batch_idx,
                    "loss": loss.item(),
                    "accuracy": accuracy
                }
                
                if batch_idx % 10 == 0:  # Emit every 10 batches
                    await event_emitter.emit_metrics(batch_metrics)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0
        }

    async def _validate_epoch(self, epoch: int):
        self.model.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}"):
                loss, accuracy = self._process_batch(batch)
                total_loss += loss.item()
                
        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0
        }

    def _process_batch(self, batch):
        inputs = batch["text"]
        encoded = self.model.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.model(**encoded, labels=encoded["input_ids"])
        loss = outputs.loss
        
        # Calculate accuracy (simplified for demonstration)
        predictions = outputs.logits.argmax(-1)
        correct = (predictions == encoded["input_ids"]).float().mean()
        
        return loss, correct.item()

    def _save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt') 