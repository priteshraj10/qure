#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from backend.training.pipeline import UnslothTrainer, TrainingConfig
from config.training_config import training_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Create config from dictionary
        config = TrainingConfig(**training_config)
        
        # Initialize trainer
        trainer = UnslothTrainer(config)
        
        # Start training
        logger.info("Starting training process...")
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 