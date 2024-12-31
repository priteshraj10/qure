from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Tuple, Any

def load_medical_dataset() -> Any:
    """Load and preprocess the medical dataset"""
    dataset = load_dataset("FreedomIntelligence/PubMedVision")
    return dataset

def create_dataloaders(dataset: Any, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size
    )
    return train_loader, val_loader 