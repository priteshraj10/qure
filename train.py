#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
from backend.training.pipeline import UnslothTrainer, TrainingConfig
from backend.utils.system_checks import verify_system_requirements

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "training.log")
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    
    try:
        # Reference to system checks
        ```shell:start.sh
        startLine: 161
        endLine: 173
        ```
        
        # Initialize training
        config = TrainingConfig()
        trainer = UnslothTrainer(config)
        
        logger.info("Starting training process...")
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 