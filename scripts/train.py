#!/usr/bin/env python3
import sys
from pathlib import Path
import logging
import torch
from backend.config.settings import ModelConfig, SystemConfig
from backend.training.pipeline import UnslothTrainer
from backend.system.checks import SystemMonitor

def main():
    # Setup configurations
    model_config = ModelConfig()
    system_config = SystemConfig()
    
    # Initialize system monitor
    monitor = SystemMonitor(system_config)
    monitor.verify_requirements()
    
    # Initialize trainer
    trainer = UnslothTrainer(model_config)
    
    try:
        # Prepare dataset
        dataset = trainer.prepare_dataset()
        
        # Train model
        trainer.train(dataset)
        
        # Save model
        trainer.save_model()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 