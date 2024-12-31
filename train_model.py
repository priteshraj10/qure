import logging
from pathlib import Path
import torch
from config.config import TrainingConfig, LoRAConfig
from models.vision_model import VisionModelHandler
from utils.data_utils import DatasetPreparator
from trainer.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_gpu_info():
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
        return start_gpu_memory, max_memory
    return 0, 0

def main():
    try:
        # Print GPU info
        start_gpu_memory, max_memory = print_gpu_info()

        # Initialize configurations
        training_config = TrainingConfig()
        lora_config = LoRAConfig()

        # Create output directory
        Path(training_config.output_dir).mkdir(exist_ok=True)

        # Initialize model
        logger.info("Initializing model...")
        model_handler = VisionModelHandler(training_config.model_name)
        model, tokenizer = model_handler.initialize_model()
        
        # Setup LoRA
        logger.info("Setting up LoRA...")
        model_handler.setup_lora(lora_config)
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        dataset = DatasetPreparator.prepare_dataset()
        
        # Initialize trainer and train
        logger.info("Starting training...")
        trainer = ModelTrainer(model, tokenizer, training_config)
        training_stats = trainer.train(dataset)
        
        # Print final stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        logger.info(f"Training completed in {training_stats.metrics['train_runtime']} seconds")
        logger.info(f"Minutes used for training: {round(training_stats.metrics['train_runtime']/60, 2)}")
        logger.info(f"Peak reserved memory: {used_memory} GB")
        logger.info(f"Peak reserved memory for training: {used_memory_for_lora} GB")
        logger.info(f"Peak reserved memory % of max memory: {used_percentage}%")
        logger.info(f"Peak reserved memory for training % of max memory: {lora_percentage}%")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()