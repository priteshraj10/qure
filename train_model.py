import argparse
import logging
from pathlib import Path
from backend.training.pipeline import UnslothTrainingPipeline, TrainingConfig
from backend.rag.pipeline import RAGPipeline
from backend.system.checks import SystemMonitor
from datasets import load_dataset
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description='Train LLM with RAG')
    parser.add_argument('--model', default="unsloth/Llama-3.2-3B", help='Model name')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    return parser.parse_args()

def main():
    # Parse arguments
    args = setup_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # System checks
    logger.info("Performing system checks...")
    system_monitor = SystemMonitor()
    system_monitor.verify_requirements()
    
    # Initialize wandb
    wandb.init(
        project="medical-llm-training",
        config={
            "model": args.model,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr
        }
    )
    
    try:
        # Initialize training config
        config = TrainingConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=str(output_dir)
        )
        
        # Initialize pipeline
        pipeline = UnslothTrainingPipeline(config)
        
        # Load and prepare dataset
        logger.info("Loading dataset...")
        dataset = load_dataset("your_dataset_name", split="train")
        
        # Train model
        logger.info("Starting training...")
        pipeline.train(dataset)
        
        # Save model
        logger.info("Saving model...")
        pipeline.model.save_pretrained(output_dir / "final_model")
        pipeline.tokenizer.save_pretrained(output_dir / "final_model")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 