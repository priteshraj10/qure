import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

class MedicalTrainer:
    def __init__(self, model_name="meditron-7b", batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        
        # Initialize wandb for tracking
        wandb.init(project="qure-medical", name="training-run-1")
        
    def load_dataset(self):
        dataset = load_dataset("FreedomIntelligence/PubMedVision")
        return dataset
        
    def train(self, num_epochs=5):
        dataset = self.load_dataset()
        train_loader = DataLoader(dataset["train"], batch_size=self.batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader)
            
            for batch in progress_bar:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Log metrics
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                })
                
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, avg_loss)
            
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt') 