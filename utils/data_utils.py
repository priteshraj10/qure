import logging
from typing import List, Dict
from datasets import load_dataset

logger = logging.getLogger(__name__)

class DatasetPreparator:
    @staticmethod
    def prepare_dataset(dataset_name: str = "unsloth/Radiology_mini") -> List[Dict]:
        try:
            dataset = load_dataset(dataset_name, split="train")
            instruction = "You are an expert radiographer. Describe accurately what you see in this image."
            
            return [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image", "image": sample["image"]}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": sample["caption"]}
                            ]
                        }
                    ]
                }
                for sample in dataset
            ]
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise 