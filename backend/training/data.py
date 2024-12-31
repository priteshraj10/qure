from datasets import load_dataset
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def format_prompt(self, instruction: str, input_text: str, output: str = "") -> str:
        # Reference to prompt formatting
        startLine: 101
        endLine: 122
        
    def prepare_dataset(self, dataset_name: str = "yahma/alpaca-cleaned"):
        dataset = load_dataset(dataset_name, split="train")
        return dataset.map(
            self.format_prompt,
            batched=True,
            num_proc=2
        ) 