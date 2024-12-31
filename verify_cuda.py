import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_cuda_setup():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA memory
        try:
            x = torch.rand(5, 5).cuda()
            logger.info("Successfully created CUDA tensor")
        except Exception as e:
            logger.error(f"Failed to create CUDA tensor: {e}")
    else:
        logger.error("CUDA is not available")

if __name__ == "__main__":
    verify_cuda_setup() 