import logging
from config.config import TrainingConfig, LoRAConfig
from models.vision_model import VisionModelHandler
from utils.data_utils import DatasetPreparator
from trainer.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize configurations
        training_config = TrainingConfig()
        lora_config = LoRAConfig()

        # Initialize model
        model_handler = VisionModelHandler(training_config.model_name)
        model, tokenizer = model_handler.initialize_model()
        
        # Setup LoRA
        model_handler.setup_lora(lora_config)
        
        # Prepare dataset
        dataset = DatasetPreparator.prepare_dataset()
        
        # Initialize trainer and train
        trainer = ModelTrainer(model, tokenizer, training_config)
        training_stats = trainer.train(dataset)
        
        logger.info(f"Training completed in {training_stats.metrics['train_runtime']} seconds")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 