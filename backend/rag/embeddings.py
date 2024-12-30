from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union

class MedicalEmbeddings:
    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device=self.device
        ) 