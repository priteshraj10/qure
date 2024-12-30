from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from typing import Dict, List
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class MedicalDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        
        # Load dataset
        self.dataset = load_dataset(config.dataset_name, split=split)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Process text
        text = f"<MEDICAL> {item['text']} <DIAGNOSIS> {item['diagnosis']}"
        encoded = self.config.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Process image if available
        image = None
        if "image" in item:
            image = self.image_transform(Image.open(item["image"]))
            
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "image": image,
            "labels": encoded["input_ids"].squeeze()
        }

def create_dataloaders(config):
    # Load medical dataset (replace with your actual dataset)
    dataset = load_dataset("medical_dialogs")  # Example dataset
    
    def format_function(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            texts.append(text)
        return {"text": texts}

    # Split and process dataset
    train_dataset = dataset["train"].map(format_function)
    val_dataset = dataset["validation"].map(format_function)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    return train_loader, val_loader 