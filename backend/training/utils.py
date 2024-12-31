import logging
from pathlib import Path
from typing import List
import os

logger = logging.getLogger(__name__)

def clean_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    """Clean old checkpoints based on start.sh"""
    # Reference to start.sh lines 130-133 for checkpoint cleaning logic
    try:
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob("checkpoint-*.pt")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[keep_last_n:]:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint}")
    except Exception as e:
        logger.error(f"Failed to clean checkpoints: {str(e)}") 